from __future__ import print_function

import csv
import sys
import subprocess
import os
from operator import itemgetter
from multiprocessing import Pool
from pathlib import Path
from typing import List, AnyStr, Tuple

import fasttext
import fasttext.util
from numpy import dot, ndarray, float32
from numpy.linalg import norm


class Translation:
    """A wrapper for the different languages and the translations of the various words"""

    def __init__(self, language: AnyStr, man: AnyStr, woman: AnyStr):
        self.language = language
        self.man = man
        self.woman = woman

    def __str__(self):
        return f"{self.language}\t{self.man}\t{self.woman}"


def override_and_print(string: AnyStr):
    width = os.get_terminal_size().columns
    print(" " * width, end='\r')
    print(f"{string}", end='\r')


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
    root = os.getcwd()
    os.chdir(root + "/data")
    for language in language_codes:
        fasttext.util.download_model(language, if_exists='ignore')
        subprocess.run(["rm", "-f", f"cc.{language}.300.bin.gz"])
    os.chdir(root)


# Use global variables here (oops) to allow the worker function to access these variables.
global current_model, current_man_vec, current_woman_vec


def worker(word: AnyStr) -> (AnyStr, float32):
    """Worker function for the pool cannot be inner function because it cannot be pickled that way"""
    # Check if there is any punctuation in the word. If there is, skip it. This used `string.punctuation` minus `-`,
    # since this symbol can occur in actual words
    if any((c in "0123456789.") for c in word):
        return None, None
    word_vector = current_model.get_word_vector(word)
    diff_man = cosine_similarity(word_vector, current_man_vec)
    diff_woman = cosine_similarity(word_vector, current_woman_vec)
    diff = diff_man - diff_woman
    return word, diff


def perform_calculation(language_codes: List[AnyStr]):
    translations = load_translations(language_codes)
    current_translation = 0
    for translation in translations:
        current_translation = current_translation + 1
        global current_model, current_man_vec, current_woman_vec

        print(f"Starting to process language {translation.language} - {current_translation}/{len(translations)}")
        override_and_print(f"Loading language {translation.language} into memory...")

        # Enter data directory
        root = os.getcwd()
        os.chdir(root + "/data")
        # Load the model from the file
        current_model = fasttext.load_model(f'cc.{translation.language}.300.bin')
        os.chdir(root)

        override_and_print(f"Loaded language {translation.language}")
        override_and_print(f"Processing language {translation.language}")

        # Load the vectors of the translations
        current_man_vec = current_model.get_word_vector(translation.man)
        current_woman_vec = current_model.get_word_vector(translation.woman)

        amount_of_words = len(current_model.get_words())
        amount_of_words_done = 0

        words = list()

        with Pool() as pool:
            it = pool.imap(func=worker, iterable=current_model.get_words(), chunksize=100)
            while True:
                try:
                    word, diff = next(it)

                    # Update and print percentage
                    amount_of_words_done += 1
                    print_status(translation.language, amount_of_words_done, amount_of_words)

                    if word is None:
                        continue
                    words.append((word, diff))
                except StopIteration:
                    break

        # Mark the model for deletion
        del current_model, current_man_vec, current_woman_vec

        override_and_print(f"Sorting result of language {translation.language}")
        words = sort_output(words)

        override_and_print(f"Writing result of language {translation.language} to disk")
        # Write the result
        directory = "output"
        write_result(directory, translation.language, words)

        print(f"Finished language {translation.language}! Result in {directory}/{translation.language}.txt")


def write_result(directory: AnyStr, language: AnyStr, result: List[Tuple[AnyStr, float]]):
    """Write the result to a file in the given directory"""
    path = f"{directory}/{language}.txt"
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for word, diff in result:
            # convert the value to int for easier usage with external tools
            print(f"{word}\t{int(diff * 10e10)}", file=f)


def sort_output(words: (AnyStr, float)) -> List[Tuple[AnyStr, float]]:
    """Sort the result list on the calculated cosine distance"""
    words.sort(key=itemgetter(1))
    return words


def print_status(language: AnyStr, done: int, total: int):
    """Print the status of this language. Only do it every so ofter, to avoid spending too much resources on the
    printing."""
    if done % 10000 == 0:
        percentage = round((done / total) * 100, 2)
        f_string = f"{percentage}%\t of {language}\t" \
                   f"Q={done}\t" \
                   f"T={total}"
        override_and_print(f_string)
        sys.stdout.flush()


def cosine_similarity(a: ndarray, b: ndarray) -> float32:
    """Calculate the cosine similarity between two vectors. The definition is based on
     https://en.wikipedia.org/wiki/Cosine_similarity and the snippet of code is based on
     https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists/43043160#43043160"""
    return dot(a, b) / (norm(a) * norm(b))


def main():
    languages: List[AnyStr] = ["en", "de", "el", "es", "fi", "fr", "nl", "pl", "pt", "ru", "sv"]
    get_languages(languages)
    perform_calculation(languages)


if __name__ == '__main__':
    main()
