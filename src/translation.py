from typing import AnyStr


class Translation:
    """A wrapper for the different languages and the translations of the various words"""

    def __init__(self, language: AnyStr, man: AnyStr, woman: AnyStr):
        self.language = language
        self.man = man
        self.woman = woman

    def __str__(self):
        return f"{self.language}\t{self.man}\t{self.woman}"
