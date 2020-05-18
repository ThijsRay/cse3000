from os import getcwd, chdir, path
from typing import List, AnyStr, Dict, Iterator
from subprocess import run
from fasttext import load_model as ft_load_model, FastText
from fasttext.util import download_model
import fasttext.util as fu
from io import open
from urllib import request
from gzip import GzipFile, open as gzipo
from shutil import copyfileobj


def download_languages_bin(directory: AnyStr, language_codes: List[AnyStr]):
    """Download all the required language fastText word embeddings. It takes a list of language codes as defined on
    https://fasttext.cc/docs/en/crawl-vectors.html and downloads them in order.
    """
    root = getcwd()
    chdir(root + f"/{directory}")
    for language in language_codes:
        download_model(language, if_exists='ignore')
        run(["rm", "-f", f"cc.{language}.300.bin.gz"])
    chdir(root)


def _download_file(output: AnyStr, url: AnyStr):
    # Download archive
    try:
        # Read the file inside the .gz archive located at url
        with request.urlopen(url) as response:
            print(f"Downloaded!", end='\r')
            with GzipFile(fileobj=response) as uncompressed:
                print(f"Uncompressing...", end='\r')
                file_content = uncompressed.read()

        # write to file in binary mode 'wb'
        with open(output, 'wb') as f:
            f.write(file_content)
            print("Uncompressed!", end='\r')
            return 0
    except Exception as e:
        print(e)
        return 1


def _download_language_txt(lang_id: AnyStr):
    file_name = "cc.%s.300.vec" % lang_id
    gz_file_name = "%s.gz" % file_name

    if path.isfile(file_name):
        return file_name

    print(f"Downloading {lang_id}", end='\r')
    _download_file(file_name, f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{gz_file_name}")
    print(f"Downloaded {lang_id}!", end='\n')

    return file_name


def download_languages(directory: AnyStr, language_codes: List[AnyStr]):
    root = getcwd()
    chdir(root + f"/{directory}")
    for language in language_codes:
        _download_language_txt(language)
    chdir(root)


def load_vectors(directory: AnyStr, language: AnyStr, length: int = 10e12) -> Dict[AnyStr, Iterator[float]]:
    root = getcwd()
    chdir(root + f"/{directory}")
    fin = open(f'cc.{language}.300.vec', 'r', encoding='utf-8', newline='\n', errors='ignore', )
    chdir(root)
    n, d = map(int, fin.readline().split())
    data = {}
    current_amount = 0
    for line in fin:
        if current_amount < length:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
            current_amount += 1
        else:
            break
    assert(len(data) == length)
    return data


def load_model(directory: AnyStr, language: AnyStr) -> FastText:
    """Load the given model into memory"""
    root = getcwd()
    chdir(root + f"/{directory}")
    current_model = ft_load_model(f'cc.{language}.300.bin')
    chdir(root)
    return current_model
