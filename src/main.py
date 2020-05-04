from __future__ import print_function

import csv
import sys
from multiprocessing import Pool
from typing import List, AnyStr

import string
import fasttext
import fasttext.util
from numpy import dot, ndarray, float32
from numpy.linalg import norm


def eprint(*args, **kwargs):
    """Define print that prints to stderr"""
    print(*args, file=sys.stderr, **kwargs)


class Translation:
    """A wrapper for the different languages and the translations of the various words"""
    def __init__(self, language: AnyStr, man: AnyStr, woman: AnyStr):
        self.language = language
        self.man = man
        self.woman = woman

    def __str__(self):
        return f"{self.language}\t{self.man}\t{self.woman}"


def load_translations(language_codes: List[AnyStr]) -> List[Translation]:
    """Load the translations of the words 'man' and 'woman' for various languages."""
    translation_list: List[Translation] = list()
    with open("data/translations.txt") as translations:
        translations = csv.DictReader(translations)
        for translation in translations:
            if translation['language'] in language_codes:
                translation_list.append(Translation(
                    translation['language'],
                    translation['man'],
                    translation['woman']))
    return translation_list


def get_languages(language_codes: List[AnyStr]):
    """Download all the required language fastText word embeddings. It takes a list of language codes as defined on
    https://fasttext.cc/docs/en/crawl-vectors.html and downloads them in order.
    """
    for language in language_codes:
        fasttext.util.download_model(language, if_exists='ignore')


# Use global variables here (oops) to allow the worker function to access these variables.
global current_model, current_man_vec, current_woman_vec


def worker(word: AnyStr) -> (AnyStr, float32):
    """Worker function for the pool cannot be inner function because it cannot be pickled that way"""
    # Check if there is any punctuation in the word. If there is, skip it. This used `string.punctuation` minus `-`,
    # since this symbol can occur in actual words
    if any((c in "!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~.") for c in word):
        return None, None
    word_vector = current_model.get_word_vector(word)
    diff_man = cosine_similarity(word_vector, current_man_vec)
    diff_woman = cosine_similarity(word_vector, current_woman_vec)
    diff = diff_man - diff_woman
    return word, diff


def perform_calculation(language_codes: List[AnyStr]):
    translations = load_translations(language_codes)
    for translation in translations:
        global current_model, current_man_vec, current_woman_vec
        # Load the model from the file
        current_model = fasttext.load_model(f'cc.{translation.language}.300.bin')

        # Load the vectors of the translations
        current_man_vec = current_model.get_word_vector(translation.man)
        current_woman_vec = current_model.get_word_vector(translation.woman)

        amount_of_words = len(current_model.get_words())
        amount_of_words_done = 0

        with Pool() as pool:
            it = pool.imap(func=worker, iterable=current_model.get_words(), chunksize=100)
            with open(f"output/{translation.language}.txt", "a") as f:
                while True:
                    try:
                        word, diff = next(it)
                        if word is None:
                            continue
                        print(f"{word},{int(diff * 1e10)}", file=f)

                        # Update and print percentage
                        amount_of_words_done += 1
                        print_status(translation.language, amount_of_words_done, amount_of_words)
                    except StopIteration:
                        break
        print(f"\nFinished {translation.language}")


def print_status(language: AnyStr, done: int, total: int):
    """Print the status of this language. Only do it every so ofter, to avoid spending too much resources on the
    printing."""
    if done % 10000 == 0:
        eprint(" " * 40, end='\r')
        percentage = round((done / total) * 100, 2)
        f_string = f"{percentage}%\t of {language}\t" \
                   f"Q={done}\t" \
                   f"T={total}"
        eprint(f_string, end='\r')


def cosine_similarity(a: ndarray, b: ndarray) -> float32:
    """Calculate the cosine similarity between two vectors. The definition is based on
     https://en.wikipedia.org/wiki/Cosine_similarity and the snippet of code is based on
     https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists/43043160#43043160"""
    return dot(a, b) / (norm(a) * norm(b))


def main():
    languages: List[AnyStr] = ['en', 'nl', 'sv', 'es']
    get_languages(languages)
    perform_calculation(languages)


if __name__ == '__main__':
    main()
