from os import getcwd, chdir
from typing import List, AnyStr
from subprocess import run
from fasttext import load_model as ft_load_model, FastText
from fasttext.util import download_model


def download_languages(directory: AnyStr, language_codes: List[AnyStr]):
    """Download all the required language fastText word embeddings. It takes a list of language codes as defined on
    https://fasttext.cc/docs/en/crawl-vectors.html and downloads them in order.
    """
    root = getcwd()
    chdir(root + f"/{directory}")
    for language in language_codes:
        download_model(language, if_exists='ignore')
        run(["rm", "-f", f"cc.{language}.300.bin.gz"])
    chdir(root)


def load_model(directory: AnyStr, language: AnyStr) -> FastText:
    """Load the given model into memory"""
    root = getcwd()
    chdir(root + f"/{directory}")
    current_model = ft_load_model(f'cc.{language}.300.bin')
    chdir(root)
    return current_model
