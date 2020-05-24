import sys
import os
from csv import DictReader,  QUOTE_NONE
from subprocess import run, PIPE

import model
from util import cosine_similarity
from translation import load_translations, Translation
from operator import itemgetter
from multiprocessing import Pool
from pathlib import Path
from typing import List, AnyStr, Tuple, Iterable, Dict
from numpy import float32

# Use global variables here (oops) to allow the worker function to access these variables.
global current_model, current_man_vec, current_woman_vec


def override_and_print(string: AnyStr):
    width = os.get_terminal_size().columns
    print(" " * width, end='\r')
    print(f"{string}", end='\r')


def worker(word_tuple: (AnyStr, Iterable[float])) -> (AnyStr, float32):
    """Worker function for the pool cannot be inner function because it cannot be pickled that way"""
    # Check if there is any punctuation in the word. If there is, skip it. This used `string.punctuation` minus `-`,
    # since this symbol can occur in actual words
    word, vec = word_tuple
    word_vector = list(vec)
    if len(word_vector) != 300:
        return None, None
    diff_man = cosine_similarity(word_vector, current_man_vec)
    diff_woman = cosine_similarity(word_vector, current_woman_vec)
    diff = diff_man - diff_woman
    return word, diff


def write_merged_file(output_directory: AnyStr, translations: List[Translation]):
    files = ""
    for t in translations:
        files += f"{output_directory}/{t.language_code}.txt "
    run(f"cat {files} > {output_directory}/all.txt", shell=True, check=True)


def get_man_and_woman_vectors(man: AnyStr, woman: AnyStr, data_directory: AnyStr,
                              language_code: AnyStr) -> (Iterable[float], Iterable[float]):
    man_command = ["grep", "-m 1", f"^{man} ", f"{data_directory}/cc.{language_code}.300.vec"]
    woman_command = ["grep", "-m 1", f"^{woman} ", f"{data_directory}/cc.{language_code}.300.vec"]
    man = run(man_command, stdout=PIPE).stdout.decode('utf-8')
    woman = run(woman_command, stdout=PIPE).stdout.decode('utf-8')

    man = man.rstrip().split(' ')[1:]
    woman = woman.rstrip().split(' ')[1:]
    assert len(man) == 300
    assert len(woman) == 300

    man = map(float, man)
    woman = map(float, woman)

    return man, woman


def perform_calculation(data_directory: AnyStr, output_directory: AnyStr, language_codes: List[AnyStr],
                        length: int = 10e9):
    translations = load_translations(language_codes)
    translation_count = 0
    for translation in translations:
        translation_count = translation_count + 1

        output_file = Path(f"{output_directory}/{translation.language_code}.txt")
        if output_file.exists():
            continue

        global current_model, current_man_vec, current_woman_vec

        print(f"Starting to process {translation.language} - {translation_count}/{len(translations)}")

        override_and_print(f"Loading {translation.language} into memory...")
        man, woman = get_man_and_woman_vectors(translation.man, translation.woman, data_directory, translation.language_code)
        current_model = model.load_vectors(data_directory, translation.language_code, length)
        override_and_print(f"Loaded {translation.language}")

        override_and_print(f"Processing {translation.language}")

        # Load the vectors of the translations
        current_man_vec = list(man)
        current_woman_vec = list(woman)

        amount_of_words = len(current_model)
        rank = 0

        words = list()

        harmonic = harmonic_series(amount_of_words)

        with Pool() as pool:
            it = pool.imap(func=worker, iterable=current_model.items(), chunksize=1000)
            while True:
                try:
                    word, diff = next(it)

                    # Update and print percentage
                    rank += 1
                    print_status(translation.language, rank, amount_of_words)

                    if word is None:
                        continue

                    freq = 1/(rank * harmonic)
                    words.append((word, diff, freq))
                except StopIteration:
                    break

        # Mark the model for deletion
        del current_model, current_man_vec, current_woman_vec

        # override_and_print(f"Sorting result of {translation}")
        # words = sort_output(words)

        override_and_print(f"Writing result of {translation.language} to disk")
        write_result(output_directory, translation.language_code, words)

        print(f"Finished {translation.language}! Result in {output_directory}/{translation.language_code}.txt")

    output_file = Path(f"{output_directory}/all.txt")
    if not output_file.exists():
        write_merged_file(output_directory, translations)


def write_result(directory: AnyStr, language: AnyStr, result: List[Tuple[AnyStr, float, float]]):
    """Write the result to a file in the given directory"""
    path = f"{directory}/{language}.txt"
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for word, diff, freq in result:
            print(f"{word}\t{diff:.15f}\t{freq:.15}\t{language}", file=f)


def sort_output(words: (AnyStr, float)) -> List[Tuple[AnyStr, float]]:
    """Sort the result list on the calculated cosine distance"""
    words.sort(key=itemgetter(1))
    return words


def print_status(language: AnyStr, done: int, total: int):
    """Print the status of this language. Only do it every so ofter, to avoid spending too much resources on the
    printing."""
    if done % 10000 == 0:
        percentage = round((done / total) * 100, 2)
        f_string = f"{percentage}%\t of {language}\t" \
                   f"Q={done}\t" \
                   f"T={total}"
        override_and_print(f_string)
        sys.stdout.flush()


def harmonic_series(n: int) -> float:
    i = 1
    s = 0.0
    for i in range(1, n + 1):
        s = s + 1 / i
    return s
