from os import getcwd, chdir
from typing import List, AnyStr
from subprocess import run
from fasttext.util import download_model


def download_languages(language_codes: List[AnyStr]):
    """Download all the required language fastText word embeddings. It takes a list of language codes as defined on
    https://fasttext.cc/docs/en/crawl-vectors.html and downloads them in order.
    """
    root = getcwd()
    chdir(root + "/data")
    for language in language_codes:
        download_model(language, if_exists='ignore')
        run(["rm", "-f", f"cc.{language}.300.bin.gz"])
    chdir(root)
