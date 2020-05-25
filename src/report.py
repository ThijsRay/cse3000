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
    command = "library(\"hyperSMURF\");" \
              "library(\"dplyr\");" \
              "library(\"data.table\");" \
              f"data <- fread(\"{quote(output_directory)}/all.txt\", quote = \"\");"
    # Set command output file
    command += f"sink(\"{quote(output_directory)}/calculations.txt\");"
    # Set the headers of the dataframe
    command += "colnames(data)[1] = \"word\";"
    command += "colnames(data)[2] = \"bias\";"
    command += "colnames(data)[3] = \"freq\";"
    command += "colnames(data)[4] = \"lang\";"
    # Define harmonic series
    command += "harmonic <- function(n) { i<-1; s<-0.0; for(i in 1:n) { s<-s+(1/i);  }; s};"
    # Convert language codes to factors
    command += "data$lang <- as.factor(data$lang);"
    # Add weighted bias column
    command += "data$wbias = data$bias * data$freq;"
    # Weighted bias per language
    command += "aggregate(wbias~lang, data = data, FUN = sum); "
    # Create pairs
    command += "langs <- as.character(unique(data$lang)); langs <- expand.grid(langs, langs);" \
               "x <- unique(as.data.frame(t(apply(langs, 1, sort))));" \
               "langs <- x;" \
               "langs <- x %>% filter(V1!=V2);"
    command += "x<-data.frame(L1=character(), L2=character(), diff=double(), p=double(), effect_size=double());"
    command += "for (i in 1:nrow(langs)) {" \
               "    x[nrow(x) + 1, ] = list(as.character(langs[i, 1]), as.character(langs[i, 2]), 0, 0, 0)" \
               "};" \
               "langs <- x;" \
    # Filtered sets
    command += "filtered <- list();" \
               "for(x in 1:nrow(langs)) {" \
               "    y <- langs[x,];" \
               "    filtered[[x]] <- data %>% filter(lang==y$L1 | lang==y$L2);" \
               "    filtered[[x]] <- data.table(filtered[[x]]);" \
               "};"
    # Fill diff column of langs
    command += "diff <- c(); " \
               "for(x in 1:nrow(langs)) {" \
               "    sums <- aggregate(wbias~lang, data=filtered[[x]], FUN=sum);" \
               "    diff <- c(diff, sums[1,2]-sums[2,2]); " \
               "}; " \
               "langs$diff <- diff;"
    # Calculate the p-score for each pair
    command += "n_of_tests <- 10000; " \
               "p <- c(); " \
               "for(x in 1:nrow(langs)) {" \
               "    y <- langs[x,];" \
               "    ptest_success <- 0;" \
               "    for(z in 1:n_of_tests) {" \
               "        partition <- do.random.partition(nrow(filtered[[x]]), 2);" \
               "        f1<-filtered[[x]][partition[[1]]];" \
               "        f2<-filtered[[x]][partition[[2]]];" \
               "        ptest <- sum(f1$wbias) - sum(f2$wbias);" \
               "        if(ptest > y$diff) { " \
               "            ptest_success <- ptest_success+1; " \
               "        } " \
               "    }; " \
               "    p <- c(p, ptest_success/n_of_tests);" \
               "}; " \
               "langs$p <- p;"
    # Calculate effect size
    command += "effect_size <- c(); " \
               "for(x in 1:nrow(langs)) {" \
               "    y <- langs[x,];" \
               "    effect_size <- c(effect_size, y$diff / sd(filtered[[x]]$wbias));" \
               "}; " \
               "langs$effect_size <- effect_size;"
    # Print langs table
    command += "langs;"
    # Boxplot
    #command += f"png(file=\"{quote(output_directory)}/boxplot.png\", width=1000, height=500); " \
    #           f"boxplot(data$bias~data$lang); " \
    #           f"dev.off();"
    # Do Kruskal-Wallis Rank sum test
    #command += "kw <- kruskal.test(data$bias, data$lang);"
    # Check if value is significant, if so, run
    #command += "if (kw$p.value < 0.01) { " \
    #           "    library(dunn.test);" \
    #           "    dunn.test(data$bias, data$lang, table=FALSE, list=TRUE, method=\"holm\")" \
    #           "} else {" \
    #           "    print(\"Kruskal-Wallis not significant, cannot reject H0\")" \
    #           "};"
    # Test statistic by Caliskan
    #command += "sum(apply(data[x[[1]]], MARGIN=1, FUN=function(x) { as.numeric(x[2]) * as.numeric(x[3]) }));"
    # Define aggregate and print function
    #command += "aggr_and_print <- function(x, data, f) {" \
    #           "x <- aggregate(x, data=data, FUN=f);" \
    #           "x[order(x[[2]]),]};"
    ## Calculate mean
    #command += "cat(\"Mean\\n\"); aggr_and_print(V2~V4, data, mean);"
    ## Calculate median
    #command += "cat(\"Median\\n\"); aggr_and_print(V2~V4, data, median);"
    ## Calculate skewness
    #command += "cat(\"Skewness\\n\");"
    #command += "library(\"e1071\"); aggr_and_print(V2~V4, data, skewness);"
    ## Calculate ranksum
    #command += "cat(\"Rank sum\\n\");"
    #command += "aggr_and_print(rank(V2)~V3, data, sum);"
    ## Calculate weighted mean
    #command += "options(scipen = 999);"
    #command += "cat(\"Weighted mean\\n\");"
    #command += "x <- aggr_and_print(V2*V3 ~ V4, data, function(x, y) {mean(x)}); x;"
    ## Calculate normalized mean
    #command += "cat(\"Weighted normalized mean\\n\");"
    #command += "x[2] <- (1/sum(x[2]))*x[2]; x;"
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

