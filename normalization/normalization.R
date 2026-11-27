#!/usr/bin/env Rscript
# Normalization metric for omnibenchmark.
#
# Measures the influence of size factors on an embedding using canonical
# correlation. Smaller values indicate less sequencing-depth confounding.
# Embedding columns are taken positionally, so any producer's dimension
# naming (PC*, dim_*) works.

suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
})

# arg parsing
# ob runs entrypoints from the output tree, not the module dir: resolve paths from this file
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "../src/common/cli.R"))
p <- arg_parser("NORM_M: size-factor influence on one embedding")
p <- add_base_args(p)                    # --output_dir, --name
p <- add_argument(p, "--embedding_tsv", help = "embedding (cell_id, then dimensions)")
p <- add_argument(p, "--size_factor_tsv", help = "size factors (cell_id, size_factor)")
args <- parse_args_outputs(p)

# logging
cat(sprintf("Full command: %s\n", paste(commandArgs(trailingOnly = FALSE), collapse = " ")))
cat(sprintf("LOG: command line args\n----------------------------------\n"))
for (i in 1:length(args)) {
  cat(sprintf("  %s: %s\n", names(args)[i], args[[i]]))
}
cat(sprintf("----------------------------------\n"))


pca <- fread(args$embedding_tsv, header = TRUE)
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

out <- output_path(args, "normalization_metrics", "_normalization_metrics.json")
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE), out)
