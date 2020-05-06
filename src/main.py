from model import download_languages
from calculate import perform_calculation
from report import generate_reports
from typing import List, AnyStr


OUTPUT_DIRECTORY = "output"
DATA_DIRECTORY = "data"
LANGUAGES: List[AnyStr] = ["en", "de", "el", "es", "fi", "fr", "nl", "pl", "pt", "ru", "sv"]


def calculate():
    download_languages(DATA_DIRECTORY, LANGUAGES)
    perform_calculation(DATA_DIRECTORY, OUTPUT_DIRECTORY, LANGUAGES)


def report():
    generate_reports(DATA_DIRECTORY, OUTPUT_DIRECTORY, LANGUAGES)


def main():
    calculate()


if __name__ == '__main__':
    main()
