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
              "library(\"doParallel\");" \
              "library(\"radiant.data\");" \
              "registerDoParallel(10);" \
              f"data <- fread(\"{quote(output_directory)}/all.txt\", quote = \"\");"
    # Set command output file
    command += f"sink(\"{quote(output_directory)}/calculations.txt\");"
    # Set scientific precision
    command += "options(scipen=20);"
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
    # Bias per language
    command += "aggregate(bias~lang, data = data, FUN = mean); "
    # Weighted bias per language
    command += "aggregate(wbias~lang, data = data, FUN = sum); "
    # Create pairs with "zero" language
    command += "langs <- as.character(unique(data$lang)); langs <- expand.grid(langs, \"00\");"
    command += "x<-data.frame(" \
               "    L1=character()," \
               "    L2=character()," \
               "    diff=double()," \
               "    wdiff=double()," \
               "    p=double()," \
               "    wp=double()," \
               "    effect_size=double()," \
               "    weffect_size=double()" \
               ");"
    command += "for (i in 1:nrow(langs)) {" \
               "    x[nrow(x) + 1, ] = list(as.character(langs[i, 1]), as.character(langs[i, 2]), 0, 0, 0, 0, 0, 0)" \
               "};" \
               "langs <- x;" \
    # Filtered sets
    command += "size <- count(data,lang);" \
               "zero_combined <- foreach(x=1:nrow(langs)) %dopar% {" \
               "    length <- size[x,2];" \
               "    y <- langs[x,];" \
               "    rbind(" \
               "        data.table(" \
               "            data %>% filter(lang==y$L1)" \
               "        ), " \
               "        data.table(" \
               "            word=rep(\"\", length)," \
               "            bias=rep(0, length)," \
               "            freq=rep(1/length, length)," \
               "            lang=rep(\"00\", length)," \
               "            wbias=rep(0, length)" \
               "        )" \
               "    ) " \
               "};" \
    # Fill diff column of langs
    command += "diff <- foreach(x=1:nrow(langs)) %dopar% {" \
               "    list(diff=sum(zero_combined[[x]]$bias), wdiff=sum(zero_combined[[x]]$wbias)) " \
               "};" \
               "diff <- do.call(rbind.data.frame, diff);" \
               "langs$diff <- diff$diff;" \
               "langs$wdiff <- diff$wdiff;" \
               "langs;"
    # Calculate the p-score for each pair
    command += "n_of_tests <- 1000; " \
               "p <- foreach(x=1:nrow(langs)) %dopar% {" \
               "    y <- langs[x,];" \
               "    p_test_success <- 0;" \
               "    wp_test_success <- 0;" \
               "    for(z in 1:n_of_tests) {" \
               "        partition <- do.random.partition(nrow(zero_combined[[x]]), 2);" \
               "        f1<-zero_combined[[x]][partition[[1]]];" \
               "        f2<-zero_combined[[x]][partition[[2]]];" \
               "        p_test <- sum(f1$bias) - sum(f2$bias);" \
               "        wp_test <- sum(f1$wbias) - sum(f2$wbias);" \
               "        if(p_test > y$diff) { " \
               "            p_test_success <- p_test_success+1; " \
               "        };" \
               "        if(wp_test > y$wdiff) { " \
               "            wp_test_success <- wp_test_success+1; " \
               "        };" \
               "    };" \
               "    if(y$diff < 0) {" \
               "        p_test_success <- n_of_tests-p_test_success;" \
               "    };" \
               "    if(y$wdiff < 0) {" \
               "        wp_test_success <- n_of_tests-wp_test_success;" \
               "    };" \
               "    list(p=p_test_success/n_of_tests, wp=wp_test_success/n_of_tests)" \
               "};" \
               "p <- do.call(rbind.data.frame, p);" \
               "langs$p <- p$p;" \
               "langs$wp <- p$wp;" \
               "langs;"
    # Calculate effect size
    command += "effect_size <- foreach(x=1:nrow(langs)) %dopar% {" \
               "    orig_lang <- zero_combined[[x]] %>% filter(lang!=00);" \
               "    e<-mean(orig_lang$bias)/sd(zero_combined[[x]]$bias);" \
               "    we<-sum(orig_lang$wbias)/weighted.sd(" \
               "        unlist(zero_combined[[x]]$bias), unlist(zero_combined[[x]]$freq)" \
               "    );" \
               "    list(effect_size=e, weffect_size=we)" \
               "}; " \
               "effect_size <- do.call(rbind.data.frame,effect_size);effect_size" \
               "langs$effect_size <- effect_size$effect_size;" \
               "langs$weffect_size <- effect_size$weffect_size;"
    # Print langs table
    command += "langs;"
    # Plot weighted biases
    command += "library('ggplot2');" \
               "data$lang <- factor(data$lang, levels=unique(data$lang[order(data$wbias)]));"
    # Boxplot
    command += f"png(file=\"{quote(output_directory)}/boxplot.png\", width=1000, height=500); " \
               f"boxplot(data$bias~data$lang); " \
               f"dev.off();"
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

