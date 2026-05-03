#!/usr/bin/env Rscript
# Embedding quality metrics (R/POem) for omnibenchmark.
#
# Implementation notes
# --------------------
# - rhdf5 reads Python-written HDF5 matrices transposed relative to R convention;
#   shape is validated against cell_ids and corrected if needed.
# - All metrics require >= 2 labels; returns NA otherwise (handled by POem).

suppressPackageStartupMessages({
  library(rhdf5)
  library(poem)
  library(jsonlite)
})

cargs <- commandArgs(trailingOnly = FALSE)
m <- grep("--file=", cargs)
.run_dir <- dirname(gsub("--file=", "", cargs[[m]]))

source(file.path(.run_dir, "..", "cli", "cli.R"))

# Add or remove metrics here; must be valid for level = "dataset".
METRICS <- c("meanSW", "cdbw", "dbcv")


load_pca <- function(path) {
  raw <- h5read(path, "embedding")
  cell_ids <- as.character(h5read(path, "cell_ids"))

  # R/Python HDF5 convention mismatch: ensure (n_cells, n_components).
  if (nrow(raw) != length(cell_ids)) {
    raw <- t(raw)
  }
  list(embedding = raw, cell_ids = cell_ids)
}


main <- function() {
  args <- parse_args()
  dir.create(args$output_dir, showWarnings = FALSE, recursive = TRUE)

  pca <- load_pca(args$pca)

  truth <- read.table(
    args$clusters_truth,
    header = TRUE,
    sep = "\t",
    stringsAsFactors = FALSE
  )

  # Align embedding rows with truth labels by cell_id.
  idx <- match(pca$cell_ids, truth$cell_id)
  mask <- !is.na(idx)
  aligned_embedding <- pca$embedding[mask, , drop = FALSE]
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
}

main()
