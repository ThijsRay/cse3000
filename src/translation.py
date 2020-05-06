from typing import AnyStr, List
from csv import DictReader


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
        translations = DictReader(translations)
        for translation in translations:
            if translation['language'] in language_codes:
                translation_list.append(Translation(
                    translation['language'],
                    translation['man'],
                    translation['woman']))
    return translation_list


