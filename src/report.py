from multiprocessing import Pool, cpu_count
from typing import AnyStr, List
from translation import load_translations, Translation
from subprocess import run
from shlex import quote

OUTPUT_DIRECTORY: AnyStr = None


def worker(language: Translation):
    print(f"Plotting {language.language}")
    input_file = f"{quote(OUTPUT_DIRECTORY)}/{quote(language.language_code)}.txt"
    command = f"data <- read.delim(\"{input_file}\", quote = \"\");"

    output_path = f"{quote(OUTPUT_DIRECTORY)}/{quote(language.language_code)}"
    command += create_histogram(output_path, language)
    command += create_qq_plot(output_path, language)
    command += outlier_percentage(output_path, language)
    command += write_summary(output_path, language)
    run_r(command)


def run_r(command: AnyStr):
    run(f"R -q -e '{command}'",
        # Check for errors and run it in a shell
        shell=True, check=True)


def outlier_percentage(output_path: AnyStr, language: Translation):
    return f"library(\"StatMeasures\");" \
           f"cat(outliers(data[[2]])$numOutliers / length(data[[2]])," \
           f"file=\"{output_path}_outlier_percentage.txt\")" \
           f"detach(\"StatMeasures\")"


def write_summary(output_path: AnyStr, language: Translation):
    # Write the summary info to a file
    return f"cat(\"{quote(language.language)}\", summary(data[[2]]), " \
           f"file=\"{output_path}_summary.txt\", sep=\"\\n\")"


def create_qq_plot(output_path: AnyStr, language: Translation):
    return f"png(file=\"{output_path}_qq.png\", width=1000, height=1000);" \
           f"qqnorm(data[[2]]);" \
           f"qqline(data[[2]]);" \
           f"dev.off();"


def create_histogram(output_path: AnyStr, language: Translation):
    return f"png(file=\"{output_path}_hist.png\", width=1000, height=500);" \
           f"hist(data[[2]], main=\"{quote(language.language)}\", " \
           f"xlab=\"Difference in cosine distance between " \
           f"words '{quote(language.man)}' and '{quote(language.woman)}'\", " \
           f"breaks=200," \
           f"prob=TRUE); " \
           f"dev.off();"


def generate_reports(data_directory: AnyStr, output_directory: AnyStr, languages: List[AnyStr]):
    global OUTPUT_DIRECTORY
    OUTPUT_DIRECTORY = output_directory

    # Limit the amount of processes in the pool to avoid memory starvation
    with Pool() as pool:
        it = pool.imap(func=worker, iterable=load_translations(languages), chunksize=1)
        while True:
            try:
                next(it)
            except StopIteration:
                break
