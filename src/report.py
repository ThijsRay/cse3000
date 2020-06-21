from multiprocessing import Pool, cpu_count
from typing import AnyStr, List
from translation import load_translations, Translation
from subprocess import run
from shlex import quote


def run_r(command: AnyStr):
    run(f"R -q -e '{command}'",
        # Check for errors and run it in a shell
        shell=True, check=True)


def prepare_files(output_path: AnyStr):
    output_path = quote(output_path)
    run(f"head -n 27 {output_path}/calculations.txt > {output_path}/bias.txt", shell=True, check=True)
    run(f"tail -n +28 {output_path}/calculations.txt | head -n 27 > {output_path}/wbias.txt", shell=True, check=True)
    run(f"tail -n 29 {output_path}/calculations.txt | head -n 27 > {output_path}/total.txt", shell=True, check=True)


def create_histogram(output_path: AnyStr, language: Translation):
    return f"png(file=\"{output_path}_hist.png\", width=1000, height=500);" \
           f"hist(data[[2]], main=\"{quote(language.language)}\", " \
           f"xlab=\"Difference in cosine distance between " \
           f"words '{quote(language.man)}' and '{quote(language.woman)}'\", " \
           f"breaks=20," \
           f"prob=TRUE); " \
           f"dev.off();"


def _plot_hist(file_name: AnyStr, column_name: AnyStr, x_label: AnyStr, language_names: AnyStr = "language"):
    return f"pdf(\"{file_name}\", height=6, width=5);" \
            "print(" \
            "    langs " \
            f"    %>% arrange({column_name}) " \
            f"    %>% mutate({language_names}=factor({language_names}, levels={language_names})) " \
            f"    %>% ggplot(aes(x={column_name}, y={language_names}))" \
            "    + geom_bar(stat=\"identity\")" \
            f"    + xlab(\"{x_label}\")" \
            "    + ylab(\"Languages\")" \
            "    + theme(text=element_text(size=15))" \
            ");" \
            "dev.off();"


def create_graphs(data_dir: AnyStr, output_dir: AnyStr):
    data_dir = quote(data_dir)
    output_dir = quote(output_dir)
    command = "library(\"ggplot2\");" \
              "library(\"data.table\");" \
              "library(\"dplyr\");" \
              "library(\"doParallel\");" \
              f"langs <- fread(\"{data_dir}/translations.txt\");" \
              f"bias <- fread(\"{output_dir}/bias.txt\");" \
              f"total <- fread(\"{output_dir}/total.txt\");" \
              "langs <- merge(x=langs, y=bias, by.x=\"language_code\", by.y=\"lang\");" \
              "langs <- merge(x=langs, y=total, by.x=\"language_code\", by.y=\"L1\");" \
              "langs$language_wp <- foreach(x=1:nrow(langs)) %do% {" \
              "     if(langs[x]$wp > 0.001) {" \
              "         paste(langs[x]$language, \" (ns) \",sep=\"\")" \
              "     } else {" \
              "         langs[x]$language" \
              "     } " \
              "};" \
              + _plot_hist(f"{output_dir}/hist_bias.pdf", "bias",
                           "Mean of cosine distances") \
              + _plot_hist(f"{output_dir}/hist_wdiff.pdf", "wdiff",
                           "Weighted mean of cosine distances", language_names="language_wp") \
              + _plot_hist(f"{output_dir}/hist_effect_size.pdf", "effect_size",
                           "Effect size d1") \
              + _plot_hist(f"{output_dir}/hist_weffect_size.pdf", "weffect_size",
                           "Effect size d2", language_names="language_wp")
    run_r(command)


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
               "p <- foreach(x=1:nrow(langs)) %do% {" \
               "    y <- langs[x,];" \
               "    p_test_success <- 0;" \
               "    wp_test_success <- 0;" \
               "    cat(\"Start processing language\", y$L1, \"\\n\");" \
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
               "        cat(y$L1, z, \"/\", n_of_tests, \"=\", p_test_success, wp_test_success,\"\\n\");" \
               "    };" \
               "    if(y$diff < 0) {" \
               "        p_test_success <- n_of_tests-p_test_success;" \
               "    };" \
               "    if(y$wdiff < 0) {" \
               "        wp_test_success <- n_of_tests-wp_test_success;" \
               "    };gc();" \
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
               "effect_size <- do.call(rbind.data.frame,effect_size);effect_size;" \
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
    command += "sink()"
    run_r(command)


def generate_reports(data_directory: AnyStr, output_directory: AnyStr, languages: List[AnyStr]):
    #grouped_calculations(output_directory)
    #prepare_files(output_directory)
    create_graphs(data_directory, output_directory)
