from typing import AnyStr, List
from csv import DictReader


class Translation:
    """A wrapper for the different languages and the translations of the various words"""

    def __init__(self, language: AnyStr, language_code: AnyStr, family: AnyStr, branch: AnyStr,
                 man: AnyStr, woman: AnyStr):
        self.language = language
        self.language_code = language_code
        self.language_family = family
        self.language_branch = branch
        self.man = man
        self.woman = woman

    def __str__(self):
        return f"{self.language_code}\t{self.man}\t{self.woman}"


def load_translations(language_codes: List[AnyStr]) -> List[Translation]:
    """Load the translations of the words 'man' and 'woman' for various languages."""
    translation_list: List[Translation] = list()
    with open("data/translations.txt") as translations:
        translations = DictReader(translations)
        for translation in translations:
            if translation['language_code'] in language_codes:
                translation_list.append(Translation(
                    translation['language'],
                    translation['language_code'],
                    translation['family'],
                    translation['branch'],
                    translation['man'],
                    translation['woman']))
    return translation_list


