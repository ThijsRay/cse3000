from multiprocessing import Pool
from typing import AnyStr, List
from translation import load_translations, Translation
from subprocess import run
from shlex import quote

OUTPUT_DIRECTORY: AnyStr = None


def create_individual_histogram(language: Translation):
    print(f"Plotting {language.language}")
    input_file = f"{quote(OUTPUT_DIRECTORY)}/{quote(language.language_code)}.txt"
    output_file = f"{quote(OUTPUT_DIRECTORY)}/{quote(language.language_code)}"
    run(f"R -q -e '"
        # Read the data into the data variable
        f"data <- read.delim(\"{input_file}\", quote = \"\"); "
        # Define the output to be a PNG
        f"png(file=\"{output_file}_hist.png\", width=1000, height=500);"
        # Generate the histogram
        f"hist(data[[2]], main=\"{quote(language.language)}\", "
        # Define the label of the x-axis
        f"xlab=\"Difference in cosine distance between "
        f"words '{quote(language.man)}' and '{quote(language.woman)}'\", "
        # Define the amount of breaks (bins-1) in the histogram
        f"breaks=200,"
        # Normalize the histogram for fair comparison
        f"prob=TRUE); "
        # Close the PNG
        f"dev.off();"
        # Define the output of the qq to be a png.
        f"png(file=\"{output_file}_qq.png\", width=1000, height=1000);"
        # Draw the Normal Q-Q plot
        f"qqnorm(data[[2]]);"
        f"qqline(data[[2]]);"
        # Close the PNG
        f"dev.off();"
        # Write the summary info to a file
        f"cat(\"{quote(language.language)}\", summary(data[[2]]), "
        f"file=\"{quote(OUTPUT_DIRECTORY)}/summary.txt\", sep=\"\\n\", append=TRUE)"
        # End quote
        f"'",
        # Check for errors and run it in a shell
        shell=True, check=True)


def generate_reports(data_directory: AnyStr, output_directory: AnyStr, languages: List[AnyStr]):
    global OUTPUT_DIRECTORY
    OUTPUT_DIRECTORY = output_directory
    # Limit the amount of processes in the pool to avoid memory starvation
    with Pool(processes=5) as pool:
        it = pool.imap(func=create_individual_histogram, iterable=load_translations(languages), chunksize=1)
        while True:
            try:
                next(it)
            except StopIteration:
                break
