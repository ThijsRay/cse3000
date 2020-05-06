from typing import AnyStr, List
from translation import load_translations, Translation
from subprocess import run
from shlex import quote


def create_histogram(output_directory: AnyStr, languages: List[Translation]):
    for language in languages:
        print("Plotting {}")
        run(f"R -q -e 'data <- read.delim(\"{quote(output_directory)}/{quote(language.language)}.txt\", quote = \"\"); "
            f"png(file=\"{quote(output_directory)}/{quote(language.language)}.png\", width=1200, height=700);"
            f"hist(data[[2]], main=\"{quote(language.language)}\", "
            f"xlab=\"Difference in cosine distance between "
            f"words '{quote(language.man)}' and '{quote(language.woman)}'\", "
            f"breaks=10000); dev.off()'", shell=True, check=True)


def generate_reports(data_directory: AnyStr, output_directory: AnyStr, languages: List[AnyStr]):
    create_histogram(output_directory, load_translations(languages))
