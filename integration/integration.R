#!/usr/bin/env Rscript
# Integration (batch-correction) metrics for omnibenchmark.
#
# Implementation notes
# --------------------
# - Infrastructure-only slice: aligns the corrected embedding against ground truth
#   labels and reports alignment bookkeeping. Metric computation is deferred.

suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
})

# arg parsing
source("src/common/cli.R")
p <- arg_parser("INTG8-M module")
p <- add_base_args(p)                    # --output_dir, --name
p <- add_stage_args(p, "INTG8-M")        # the stage I/O contract
# your own method params — argparser directly (its add_argument requires `help`):
args <- parse_args(p)                    # argparser's own parser

# logging
cat(sprintf("Full command: %s\n", paste(commandArgs(trailingOnly = FALSE), collapse = " ")))
cat(sprintf("LOG: command line args\n----------------------------------\n"))
for (i in 1:length(args)) {
  cat(sprintf("  %s: %s\n", names(args)[i], args[[i]]))
}
cat(sprintf("----------------------------------\n"))


dir.create(args$output_dir, showWarnings = FALSE, recursive = TRUE)

corrected <- fread(args$corrected_tsv, header = TRUE)
truth <- fread(args$rawdata_clusters_truth, header = TRUE)

# Align the corrected embedding rows with truth labels by cell_id.
idx <- match(corrected$cell_id, truth$cell_id)
mask <- !is.na(idx)

n_cells <- sum(mask)
n_labels <- length(unique(truth$truths[idx[mask]]))
n_dropped <- sum(!mask)

result <- list(n_cells = n_cells, n_labels = n_labels, n_dropped = n_dropped)
out <- file.path(args$output_dir, sprintf("%s_integration_metrics.json", args$name))
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE, na = "null"), out)
