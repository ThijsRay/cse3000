import fasttext
import fasttext.util
from typing import List, AnyStr


def get_languages(language_codes: List[AnyStr]):
    """Download all the required language fastText word embeddings. It takes a list of language codes as defined on
    https://fasttext.cc/docs/en/crawl-vectors.html and downloads them in order.
    """
    for language in language_codes:
        fasttext.util.download_model(language, if_exists='ignore')


def main():
    get_languages(['en', 'nl'])
    pass


if __name__ == '__main__':
    main()
