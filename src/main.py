from __future__ import print_function
from numpy import dot, ndarray, float32, abs
from numpy.linalg import norm
from typing import List, AnyStr
import fasttext
import fasttext.util
import csv
import threading
import queue
import sys
import os
from time import sleep


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


def load_translations(language_codes: List[AnyStr]) -> Translation:
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


def perform_calculation(language_codes: List[AnyStr]):
    translations = load_translations(language_codes)
    for translation in translations:
        # Load the model from the file
        model = fasttext.load_model(f'cc.{translation.language}.300.bin')

        # Load the vectors of the translations
        man = model.get_word_vector(translation.man)
        woman = model.get_word_vector(translation.woman)

        # Create a queue and populate it with all the words in the model
        job_queue = queue.Queue()
        [job_queue.put(i) for i in model.get_words()]

        # Create a queue for all the results
        result_queue = queue.Queue()

        percentage = 0.0
        amount_of_words = len(model.words)

        def queue_worker():
            while True:
                word = job_queue.get()
                word_vector = model.get_word_vector(word)
                diff_man = cosine_similarity(word_vector, man)
                diff_woman = cosine_similarity(word_vector, woman)
                diff = abs(diff_man - diff_woman)
                result_queue.put((word, diff))
                job_queue.task_done()

        for i in range(0, os.cpu_count()-1):
            threading.Thread(target=queue_worker, daemon=True).start()

        while percentage < 100:
            queue_size = job_queue.qsize()
            percentage = round((1 - queue_size / amount_of_words) * 100, 2)
            eprint(f"{percentage}% of {translation.language}\tQ={queue_size}\tT={amount_of_words}", end='\r')
            sleep(0.3)

        job_queue.join()
        print(result_queue)


def cosine_similarity(a: ndarray, b: ndarray) -> float32:
    """Calculate the cosine similarity between two vectors. The definition is based on
     https://en.wikipedia.org/wiki/Cosine_similarity and the snippet of code is based on
     https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists/43043160#43043160"""
    return dot(a, b) / (norm(a) * norm(b))


def main():
    languages: List[AnyStr] = ['en']
    get_languages(languages)
    perform_calculation(languages)


if __name__ == '__main__':
    main()
