#!/usr/bin/env Rscript
# GRAPH_M: graph quality metrics (poem) for one kNN graph.
#
# Writes one flat JSON (n_cells, n_labels, n_dropped, then one key per metric);
# metrics are NA when the graph sees < 2 labels. A per-dataset table is a
# collector's job (obkit.collector over a gather), not this module's.
#
# - Reads KNN distances as CSR from HDF5 (cell_ids, data, indices, indptr).
# - Converts to a directed igraph for poem's getGraphMetrics.

suppressPackageStartupMessages({
  library(poem)
  library(rhdf5)
  library(Matrix)
  library(igraph)
  library(jsonlite)
  library(data.table)
})

# Add or remove metrics here; must be valid for level = "dataset".
METRICS <- c("SI", "ISI", "NP", "AMSP", "PWC", "NCE", "adhesion", "cohesion")

# ob runs entrypoints from the output tree, not the module dir: resolve paths from this file
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "../src/common/cli.R"))
SCHEMA_DIR <- file.path(here, "../src/common/schema")
p <- arg_parser("GRAPH_M: graph metrics for one kNN graph")
p <- add_base_args(p)  # --output_dir, --name
p <- add_argument(p, "--neighbors_h5", help = "kNN graph (CSR distances, HDF5)")
p <- add_argument(p, "--rawdata_clusters_truth", help = "truth labels (cell_id, truths)")
args <- parse_args_outputs(p)

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

dist_mat <- read_csr_h5(args$neighbors_h5)
truth <- fread(args$rawdata_clusters_truth, header = TRUE)
idx <- match(rownames(dist_mat), truth$cell_id)
mask <- !is.na(idx)
labels <- truth$truths[idx[mask]]
if (!all(mask)) dist_mat <- dist_mat[mask, mask]
n_labels <- length(unique(labels))

scores <- setNames(as.list(rep(NA_real_, length(METRICS))), METRICS)
if (n_labels >= 2) {
  g <- graph_from_adjacency_matrix(dist_mat, mode = "directed", weighted = TRUE)
  res <- getGraphMetrics(g, labels = labels, metrics = METRICS, level = "dataset")
  scores <- lapply(setNames(METRICS, METRICS), function(m) as.numeric(res[1, m]))
}

result <- c(list(n_cells = sum(mask), n_labels = n_labels, n_dropped = sum(!mask)), scores)
out <- output_path(args, "graph_metrics_json", "_graph_metrics.json")
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE, na = "null", digits = NA), out)
cat(sprintf("wrote %s\n", out))
