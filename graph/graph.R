#!/usr/bin/env Rscript
# Graph quality metrics (R/poem) for omnibenchmark.
#
# Implementation notes
# --------------------
# - Reads KNN distances as CSR from HDF5 (cell_ids, data, indices, indptr).
# - Converts to igraph for POem's getGraphMetrics.
# - All metrics require >= 2 labels; returns NA otherwise.

suppressPackageStartupMessages({
  library(poem)
  library(rhdf5)
  library(Matrix)
  library(igraph)
  library(jsonlite)
  library(data.table)
})

cargs <- commandArgs(trailingOnly = FALSE)
m <- grep("--file=", cargs)
.run_dir <- dirname(gsub("--file=", "", cargs[[m]]))

source(file.path(.run_dir, "..", "cli", "cli.R"))

# Add or remove metrics here; must be valid for level = "dataset".
METRICS <- c(
  "SI",
  "ISI",
  "NP",
  "AMSP",
  "PWC",
  "NCE",
  "adhesion",
  "cohesion"
)

read_csr_h5 <- function(path) {
  cell_ids <- h5read(path, "cell_ids")
  data <- h5read(path, "data")
  indices <- h5read(path, "indices") + 1L # 0-indexed → 1-indexed
  indptr <- h5read(path, "indptr")
  n <- length(indptr) - 1L
  rows <- rep(seq_len(n), diff(indptr))
  mat <- sparseMatrix(i = rows, j = indices, x = data, dims = c(n, n))
  rownames(mat) <- colnames(mat) <- cell_ids
  mat
}

args <- parse_args("graph.json")
dir.create(args$output_dir, showWarnings = FALSE, recursive = TRUE)

dist_mat <- read_csr_h5(args$distances)
cell_ids <- rownames(dist_mat)

truth <- fread(args$clusters_truth, header = TRUE)

idx <- match(cell_ids, truth$cell_id)
mask <- !is.na(idx)
aligned_labels <- truth$truths[idx[mask]]

n_cells <- sum(mask)
n_labels <- length(unique(aligned_labels))
n_dropped <- sum(!mask)

if (mask[1] != TRUE || !all(mask)) {
  dist_mat <- dist_mat[mask, mask]
}

g <- graph_from_adjacency_matrix(dist_mat, mode = "directed", weighted = TRUE)

if (n_labels >= 2) {
  result_df <- getGraphMetrics(
    g,
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
out <- file.path(args$output_dir, sprintf("%s_graph_metrics.json", args$name))
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE), out)
