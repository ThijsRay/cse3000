import fasttext
import fasttext.util
from numpy import dot
from numpy.linalg import norm
import csv
from typing import List, AnyStr


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
    translation_list = list[Translation]()
    with open("data/translations.txt") as translations:
        translations = csv.DictReader(translations)
        for translation in translations:
            if translation['language'] in language_codes:
                translation_list.push(Translation(
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
    for language in language_codes:
        model = fasttext.load_model(f'cc.{language}.300.bin')


def cosine_similarity(a, b):
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
