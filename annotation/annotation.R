#!/usr/bin/env Rscript
# Cell-type annotation metrics for omnibenchmark.
#
# Implementation notes
# --------------------
# - Infrastructure-only slice: aligns predicted annotations against ground truth
#   labels and reports alignment bookkeeping. Metric computation is deferred.

suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
})

# arg parsing
source("src/common/cli.R")
p <- arg_parser("ANNO-M module")
p <- add_base_args(p)                    # --output_dir, --name
p <- add_stage_args(p, "ANNO-M")     # the stage I/O contract
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

pred <- fread(args$annotations_tsv, header = TRUE)
truth <- fread(args$rawdata_clusters_truth, header = TRUE)

merged <- merge(pred, truth, by = "cell_id", all = FALSE)

n_cells <- nrow(merged)
n_dropped <- nrow(pred) - n_cells
n_labels <- length(unique(merged$truths))

result <- list(n_cells = n_cells, n_labels = n_labels, n_dropped = n_dropped)
out <- file.path(args$output_dir, sprintf("%s_annotation_metrics.json", args$name))
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE, na = "null"), out)
