#!/usr/bin/env Rscript
# Embedding quality metrics (R/poem) for omnibenchmark.
#
# Implementation notes
# --------------------
# - All metrics require >= 2 labels; returns NA otherwise (handled by poem).

suppressPackageStartupMessages({
  library(poem)
  library(jsonlite)
})

cargs <- commandArgs(trailingOnly = FALSE)
m <- grep("--file=", cargs)
.run_dir <- dirname(gsub("--file=", "", cargs[[m]]))

source(file.path(.run_dir, "..", "cli", "cli.R"))

# Add or remove metrics here; must be valid for level = "dataset".
METRICS <- c("meanSW", "cdbw", "dbcv")

args <- parse_args()
dir.create(args$output_dir, showWarnings = FALSE, recursive = TRUE)

pca <- read.table(args$pcas)

truth <- read.table(
  args$clusters_truth,
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE
)

# Align embedding rows with truth labels by cell_id.
idx <- match(rownames(pca), truth$cell_id)
mask <- !is.na(idx)
aligned_embedding <- pca[mask, , drop = FALSE]
aligned_labels <- as.factor(truth$truths[idx[mask]])

n_cells <- sum(mask)
n_labels <- length(unique(aligned_labels))
n_dropped <- sum(!mask)

if (n_labels >= 2) {
  result_df <- getEmbeddingMetrics(
    aligned_embedding,
    labels = aligned_labels,
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
out <- file.path(
  args$output_dir,
  sprintf("%s_embedding_metrics.json", args$name)
)
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE), out)
