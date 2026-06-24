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

# arg parsing
source("src/common/cli.R")
p <- arg_parser("GRAPH-M module")
p <- add_base_args(p)                    # --output_dir, --name
p <- add_stage_args(p, "GRAPH-M")     # the stage I/O contract
# your own method params — argparser directly (its add_argument requires `help`):
args <- parse_args(p)                    # argparser's own parser

# logging
cat(sprintf("Full command: %s\n", paste(commandArgs(trailingOnly = FALSE), collapse = " ")))
cat(sprintf("LOG: command line args\n----------------------------------\n"))
for (i in 1:length(args)) {
  cat(sprintf("  %s: %s\n", names(args)[i], args[[i]]))
}
cat(sprintf("----------------------------------\n"))


read_csr_h5 <- function(path) {
  cell_ids <- as.character(h5read(path, "cell_ids"))
  data <- as.numeric(h5read(path, "data"))
  indices <- as.integer(h5read(path, "indices")) + 1L # 0-indexed → 1-indexed
  indptr <- as.integer(h5read(path, "indptr"))
  n <- length(indptr) - 1L
  rows <- rep(seq_len(n), diff(indptr))
  mat <- sparseMatrix(i = rows, j = indices, x = data, dims = c(n, n))
  rownames(mat) <- colnames(mat) <- cell_ids
  mat
}

dir.create(args$output_dir, showWarnings = FALSE, recursive = TRUE)


cat(sprintf("before read_csr_h5.\n"))
dist_mat <- read_csr_h5(args$distances)
cat(sprintf("after read_csr_h5.\n"))
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
