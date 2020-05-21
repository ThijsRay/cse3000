from multiprocessing import Pool, cpu_count
from typing import AnyStr, List
from translation import load_translations, Translation
from subprocess import run
from shlex import quote

OUTPUT_DIRECTORY: AnyStr = None


def worker(language: Translation):
    print(f"Plotting {language.language}")
    input_file = f"{quote(OUTPUT_DIRECTORY)}/{quote(language.language_code)}.txt"
    command = f"library(\"data.table\"); data <- fread(\"{input_file}\", quote = \"\");"

    output_path = f"{quote(OUTPUT_DIRECTORY)}/{quote(language.language_code)}"
    command += calculate_skewness(output_path, language)
    command += create_histogram(output_path, language)
    command += create_qq_plot(output_path, language)
    command += outlier_percentage(output_path, language)
    command += write_summary(output_path, language)
    run_r(command)


def run_r(command: AnyStr):
    run(f"R -q -e '{command}'",
        # Check for errors and run it in a shell
        shell=True, check=True)


def calculate_skewness(output_path: AnyStr, language: Translation) -> AnyStr:
    return f"" \
           f"Mode <- function(x) {{ ux <- unique(x); ux[which.max(tabulate(match(x, ux)))] }};" \
           f"cat(\"{quote(language.language_code)}\", " \
           f"(mean(data[[2]]) - Mode(data[[2]])) / sd(data[[2]])," \
           f"(mean(data[[2]]) - median(data[[2]])) / sd(data[[2]])," \
           f"file=\"{output_path}_skew.txt\", sep=\"\\t\");"


def outlier_percentage(output_path: AnyStr, language: Translation):
    return f"library(\"StatMeasures\");" \
           f"cat(outliers(data[[2]])$numOutliers / length(data[[2]])," \
           f"file=\"{output_path}_outlier_percentage.txt\");"


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
           f"breaks=20," \
           f"prob=TRUE); " \
           f"dev.off();"


def grouped_calculations(output_directory: AnyStr):
    command = "library(\"data.table\"); " \
              f"data <- fread(\"{quote(output_directory)}/all.txt\", quote = \"\");"
    # Set command output file
    command += f"sink(\"{quote(output_directory)}/calculations.txt\");"
    # Convert language codes to factors
    command += "data$V3 <- as.factor(data$V3);"
    # Boxplot
    command += f"png(file=\"{quote(output_directory)}/boxplot.png\", width=1000, height=500); " \
               f"boxplot(data$V2~data$V3); " \
               f"dev.off();"
    # Do Kruskal-Wallis Rank sum test
    command += "kw <- kruskal.test(data$V2, data$V3);"
    # Check if value is significant, if so, run
    command += "if (kw$p.value < 0.01) { " \
               "    library(dunn.test);" \
               "    dunn.test(data$V2, data$V3, table=FALSE, list=TRUE, method=\"holm\")" \
               "} else {" \
               "    print(\"Kruskal-Wallis not significant, cannot reject H0\")" \
               "};"
    # Add frequency to the data
    command += "data$frequency <- 0.1 / (((1:nrow(data) - 1) %% 1000) + 1);"
    # Define aggregate and print function
    command += "aggr_and_print <- function(x, data, f) {" \
               "x <- aggregate(x, data=data, FUN=f);" \
               "x[order(x[[2]]),]};"
    # Calculate mean
    command += "cat(\"Mean\\n\"); aggr_and_print(V2~V3, data, mean);"
    # Calculate median
    command += "cat(\"Median\\n\"); aggr_and_print(V2~V3, data, median);"
    # Calculate skewness
    command += "cat(\"Skewness\\n\");"
    command += "library(\"e1071\"); aggr_and_print(V2~V3, data, skewness);"
    # Calculate ranksum
    command += "cat(\"Rank sum\\n\");"
    command += "aggr_and_print(rank(V2)~V3, data, sum);"
    # Calculate weighted mean
    command += "options(scipen = 999);"
    command += "cat(\"Weighted mean\\n\");"
    command += "x <- aggr_and_print(V2 * frequency ~V3, data, function(x, y) {mean(x)}); x;"
    # Calculate normalized mean
    command += "cat(\"Weighted normalized mean\\n\");"
    command += "x[2] <- (1/sum(x[2]))*x[2]; x;"
    command += "sink()"
    run_r(command)


def generate_reports(data_directory: AnyStr, output_directory: AnyStr, languages: List[AnyStr]):
    global OUTPUT_DIRECTORY
    OUTPUT_DIRECTORY = output_directory

    grouped_calculations(output_directory)

    ## Limit the amount of processes in the pool to avoid memory starvation
    #with Pool() as pool:
    #    it = pool.imap(func=worker, iterable=load_translations(languages), chunksize=1)
    #    while True:
    #        try:
    #            next(it)
    #        except StopIteration:
    #            break

