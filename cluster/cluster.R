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

cargs <- commandArgs(trailingOnly = FALSE)
m <- grep("--file=", cargs)
.run_dir <- dirname(gsub("--file=", "", cargs[[m]]))

source(file.path(.run_dir, "..", "cli", "cli.R"))

# Add or remove metrics here; must be valid for level = "dataset".
METRICS <- c(
  "ARI",
  "AMI",
  "FM",
  "VM",
  "EH",
  "EC"
)

args <- parse_args("cluster.json")
dir.create(args$output_dir, showWarnings = FALSE, recursive = TRUE)

pred <- fread(args$clusters, header = TRUE)
truth <- fread(args$clusters_truth, header = TRUE)

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
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE), out)
