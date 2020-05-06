import sys
import os
import model
from util import cosine_similarity
from translation import load_translations
from operator import itemgetter
from multiprocessing import Pool
from pathlib import Path
from typing import List, AnyStr, Tuple
from numpy import float32

# Use global variables here (oops) to allow the worker function to access these variables.
global current_model, current_man_vec, current_woman_vec


def override_and_print(string: AnyStr):
    width = os.get_terminal_size().columns
    print(" " * width, end='\r')
    print(f"{string}", end='\r')


def worker(word: AnyStr) -> (AnyStr, float32):
    """Worker function for the pool cannot be inner function because it cannot be pickled that way"""
    # Check if there is any punctuation in the word. If there is, skip it. This used `string.punctuation` minus `-`,
    # since this symbol can occur in actual words
    if any((c in "0123456789.") for c in word):
        return None, None
    word_vector = current_model.get_word_vector(word)
    diff_man = cosine_similarity(word_vector, current_man_vec)
    diff_woman = cosine_similarity(word_vector, current_woman_vec)
    diff = diff_man - diff_woman
    return word, diff


def perform_calculation(data_directory: AnyStr, output_directory: AnyStr, language_codes: List[AnyStr]):
    translations = load_translations(language_codes)
    translation_count = 0
    for translation in translations:
        translation_count = translation_count + 1
        global current_model, current_man_vec, current_woman_vec

        print(f"Starting to process {translation.language} - {translation_count}/{len(translations)}")

        override_and_print(f"Loading {translation.language} into memory...")
        current_model = model.load_model(data_directory, translation.language_code)
        override_and_print(f"Loaded {translation.language}")

        override_and_print(f"Processing {translation.language}")

        # Load the vectors of the translations
        current_man_vec = current_model.get_word_vector(translation.man)
        current_woman_vec = current_model.get_word_vector(translation.woman)

        amount_of_words = len(current_model.get_words())
        amount_of_words_done = 0

        words = list()

        with Pool() as pool:
            it = pool.imap(func=worker, iterable=current_model.get_words(), chunksize=100)
            while True:
                try:
                    word, diff = next(it)

                    # Update and print percentage
                    amount_of_words_done += 1
                    print_status(translation.language, amount_of_words_done, amount_of_words)

                    if word is None:
                        continue
                    words.append((word, diff))
                except StopIteration:
                    break

        # Mark the model for deletion
        del current_model, current_man_vec, current_woman_vec

        override_and_print(f"Sorting result of {translation}")
        words = sort_output(words)

        override_and_print(f"Writing result of {translation.language} to disk")
        write_result(output_directory, translation.language_code, words)

        print(f"Finished {translation.language}! Result in {output_directory}/{translation.language_code}.txt")


def write_result(directory: AnyStr, language: AnyStr, result: List[Tuple[AnyStr, float]]):
    """Write the result to a file in the given directory"""
    path = f"{directory}/{language}.txt"
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for word, diff in result:
            print(f"{word}\t{diff:.15f}", file=f)


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
