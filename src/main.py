from model import download_languages
from calculate import perform_calculation
from report import generate_reports
from typing import List, AnyStr


OUTPUT_DIRECTORY = "output"
DATA_DIRECTORY = "data"
LANGUAGES: List[AnyStr] = ["zh", "es", "en", "hi", "pt", "ru", "ja", "tr", "ko", "fr", "de", "it", "pl", "nl", "el",
                           "fi", "ar", "sv", "yo", "hu", "te", "my", "th", "km", "jv", "eu"]


def main():
    #download_languages(DATA_DIRECTORY, LANGUAGES)
    perform_calculation(DATA_DIRECTORY, OUTPUT_DIRECTORY, LANGUAGES)
    generate_reports(DATA_DIRECTORY, OUTPUT_DIRECTORY, LANGUAGES)


if __name__ == '__main__':
    main()
