#!/usr/bin/env Rscript
# Partition metrics (R/poem) for omnibenchmark.
#
# Implementation notes
# --------------------
# - Compares predicted clusters against ground truth labels.
# - All metrics require >= 2 labels; returns NA otherwise.

suppressPackageStartupMessages({
  library(poem)
  library(jsonlite)
  library(data.table)
})


# Add or remove metrics here; must be valid for level = "dataset".
METRICS <- c(
  "ARI",
  "AMI",
  "FM",
  "VM",
  "EH",
  "EC"
)

# arg parsing
source("src/common/cli.R")
p <- arg_parser("CLUST-M module")
p <- add_base_args(p)                    # --output_dir, --name
p <- add_stage_args(p, "CLUST-M")     # the stage I/O contract
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

pred <- fread(args$clusters_tsv, header = TRUE)
truth <- fread(args$rawdata_clusters_truth, header = TRUE)

merged <- merge(pred, truth, by = "cell_id", all = FALSE)

n_cells <- nrow(merged)
n_dropped <- nrow(pred) - n_cells
n_labels <- length(unique(merged$truths))

if (n_labels >= 2) {
  result_df <- getPartitionMetrics(
    true = merged$truths,
    pred = merged$cluster,
    metrics = METRICS,
    level = "dataset"
  )
  scores <- as.list(result_df[1, ])
  scores <- lapply(scores, as.numeric)
} else {
  scores <- setNames(as.list(rep(NA_real_, length(METRICS))), METRICS)
}

result <- c(
  list(n_cells = n_cells, n_labels = n_labels, n_dropped = n_dropped),
  scores
)
out <- file.path(args$output_dir, sprintf("%s_cluster_metrics.json", args$name))
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE, na = "null"), out)
