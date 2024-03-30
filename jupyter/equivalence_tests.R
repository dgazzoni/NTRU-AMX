library(TOSTER)

cpu <- "M3_dit"

res <- data.frame()

for (pset in c("hps2048509", "hps2048677", "hps4096821", "hrss701")) {
    for (alloc in c("stack", "mmap")) {
        if (pset == "hps2048677" || pset == "hrss701") {
            impls <- c("NG21", "CCHY23")
        } else {
            impls <- c("NG21")
        }
        for (impl in impls) {
            if (impl == "NG21") {
                variants <- c("amx", "neon")
            } else {
                if (pset == "hps2048677") {
                    variants <- c("amx", "tc", "tmvp")
                } else {
                    variants <- c("amx", "tmvp")
                }
            }
            for (variant in variants) {
                file <- sprintf("../ct_results_%s/ntru:%s:%s:%s:%s.txt", cpu, pset, alloc, impl, variant)

                data <- read.table(file, sep = " ")
                data_zeros <- data[, 1]
                data_random <- data[, 2]

                test <- t_TOST(x = data_zeros, y = data_random, hypothesis = "EQU", paired = TRUE, eqb = 1, alpha = 1e-6)

                res <- rbind(res, data.frame(pset, alloc, impl, variant, test$smd$dlow, test$smd$dhigh, max(test$TOST["TOST Lower", "p.value"], test$TOST["TOST Upper", "p.value"])))
            }
        }
    }
}

res <- setNames(res, c("parameter set", "memory allocation", "implementation", "variant", "CI low", "CI high", "p-value"))
output_file <- sprintf("../ct_results_%s/results.csv", cpu)
write.table(res, file = output_file, row.names = FALSE, quote = FALSE, sep = "\t")
