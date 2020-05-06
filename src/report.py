from typing import AnyStr, List
from translation import load_translations


class WordValue:
    """A wrapper for words and their calculated value"""

    def __init__(self, word: AnyStr, value: int):
        self.word = word
        self.value = value


def load_results() -> List[WordValue]:
    pass


def generate_reports(data_directory: AnyStr, output_directory: AnyStr, languages: List[AnyStr]):
    translations = load_translations(languages)
