#!/usr/bin/env Rscript
# Normalization metric for omnibenchmark.
#
# Measures the influence of size factors on a PCA embedding using canonical
# correlation. Smaller values indicate less sequencing-depth confounding.

suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
})

# arg parsing
source("src/common/cli.R")
p <- arg_parser("NORM-M module")
p <- add_base_args(p)                    # --output_dir, --name
p <- add_stage_args(p, "NORM-M")         # --pcas_tsv, --size_factor_tsv
args <- parse_args(p)

# logging
cat(sprintf("Full command: %s\n", paste(commandArgs(trailingOnly = FALSE), collapse = " ")))
cat(sprintf("LOG: command line args\n----------------------------------\n"))
for (i in 1:length(args)) {
  cat(sprintf("  %s: %s\n", names(args)[i], args[[i]]))
}
cat(sprintf("----------------------------------\n"))

dir.create(args$output_dir, showWarnings = FALSE, recursive = TRUE)

pca <- fread(args$pcas_tsv, header = TRUE)
size_factors <- fread(args$size_factor_tsv, header = TRUE)

# Align size factors to the PCA row order by cell_id.
idx <- match(pca$cell_id, size_factors$cell_id)
size_factor <- size_factors$size_factor[idx]

pc_cols <- setdiff(names(pca), "cell_id")
n_pcs_used <- min(10L, length(pc_cols))
selected_pc_columns <- pc_cols[seq_len(n_pcs_used)]
pca_matrix <- as.matrix(pca[, ..selected_pc_columns])

rho <- stats::cancor(
  pca_matrix,
  matrix(size_factor, ncol = 1)
)$cor[1]

result <- list(
  canonical_correlation = as.numeric(rho),
  n_pcs_used = n_pcs_used,
  n_cells = nrow(pca_matrix)
)

out <- file.path(args$output_dir, sprintf("%s_normalization_metrics.json", args$name))
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE), out)
