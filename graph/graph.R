#!/usr/bin/env Rscript
# GRAPH_M (gather): graph quality metrics (poem) for every kNN graph of one dataset.
#
# ob gathers every producer of neighbors_h5 for one dataset (group_by DATA) plus
# that dataset's truth labels. Each graph is labelled by its ob output path:
# norm / feat module, the stage + module right before NNG (dimred), and the NNG
# module + params hash.
#
# Output: one long TSV, one row per (graph, metric). Rows n_cells, n_labels and
# n_dropped come first; metrics are NA when a graph sees < 2 labels.
#
# Implementation notes
# - Reads KNN distances as CSR from HDF5 (cell_ids, data, indices, indptr).
# - Converts to a directed igraph for poem's getGraphMetrics.

suppressPackageStartupMessages({
  library(poem)
  library(rhdf5)
  library(Matrix)
  library(igraph)
  library(data.table)
})

# Add or remove metrics here; must be valid for level = "dataset".
METRICS <- c("SI", "ISI", "NP", "AMSP", "PWC", "NCE", "adhesion", "cohesion")

source("src/common/cli.R")
p <- arg_parser("GRAPH_M gather: graph metrics over every kNN graph of a dataset")
p <- add_base_args(p)  # --output_dir, --name
p <- add_argument(p, "--neighbors_h5", nargs = Inf, help = "every gathered kNN graph")
p <- add_argument(p, "--rawdata_clusters_truth", nargs = Inf, help = "the dataset's truth labels")
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

# STAGE.module.params path segments -> named list stage -> c(module, params)
lineage <- function(path) {
  parts <- strsplit(path, "/", fixed = TRUE)[[1]]
  seg <- regmatches(parts, regexec("^([A-Z][A-Z0-9_]*)\\.([A-Za-z0-9_]+)\\.([A-Za-z0-9_]+)$", parts))
  seg <- Filter(length, seg)
  setNames(lapply(seg, `[`, 3:4), vapply(seg, `[`, "", 2))
}

graph_metrics <- function(path, truth) {
  dist_mat <- read_csr_h5(path)
  idx <- match(rownames(dist_mat), truth$cell_id)
  mask <- !is.na(idx)
  labels <- truth$truths[idx[mask]]
  if (!all(mask)) dist_mat <- dist_mat[mask, mask]
  n_labels <- length(unique(labels))
  scores <- setNames(rep(NA_real_, length(METRICS)), METRICS)
  if (n_labels >= 2) {
    g <- graph_from_adjacency_matrix(dist_mat, mode = "directed", weighted = TRUE)
    res <- getGraphMetrics(g, labels = labels, metrics = METRICS, level = "dataset")
    scores[] <- vapply(METRICS, function(m) as.numeric(res[1, m]), 0)
  }
  c(n_cells = sum(mask), n_labels = n_labels, n_dropped = sum(!mask), scores)
}

stopifnot(length(args$rawdata_clusters_truth) == 1L)
truth <- fread(args$rawdata_clusters_truth, header = TRUE)

# ~90 s and ~1.3 GB per 28k-cell graph, single-threaded: one forked worker per
# graph, as many as the rule's threads (snakemake exports OMP_NUM_THREADS from
# resources.cores)
workers <- max(1L, suppressWarnings(as.integer(Sys.getenv("OMP_NUM_THREADS", "1"))), na.rm = TRUE)
rows <- parallel::mclapply(args$neighbors_h5, mc.cores = workers, mc.preschedule = FALSE, function(path) {
  lin <- lineage(path)
  st <- names(lin)
  nng <- which(st == "NNG")
  if (!length(nng)) stop(sprintf("no NNG segment in %s", path), call. = FALSE)
  dimred <- if (nng > 1) sprintf("%s.%s", st[nng - 1], lin[[nng - 1]][1]) else NA_character_
  cat(sprintf("graph: %s\n", path))
  v <- graph_metrics(path, truth)
  data.table(
    dataset = lin$DATA[1] %||% NA_character_,
    norm = lin$NORM[1] %||% NA_character_,
    feat = lin$FEAT[1] %||% NA_character_,
    dimred = dimred,
    nng = lin$NNG[1], nng_params = lin$NNG[2],
    metric = names(v), value = unname(v)
  )
})

out <- output_path(args, "graph_metrics_tsv", "_graph_metrics.tsv")
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
failed <- vapply(rows, inherits, NA, "try-error")
if (any(failed)) stop(paste(unlist(rows[failed]), collapse = "\n"), call. = FALSE)
dt <- rbindlist(rows)
fwrite(dt, out, sep = "\t", na = "NA", quote = FALSE)
cat(sprintf("wrote %d rows for %d graphs: %s\n", nrow(dt), length(rows), out))
