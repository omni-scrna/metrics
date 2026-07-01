#!/usr/bin/env Rscript
# Cluster-significance metric (scSHC) for omnibenchmark.
#
# Unlike cluster.R (which scores predicted vs. truth labels), this is an
# unsupervised sanity check on the clustering itself: scSHC::testClusters builds
# a pseudobulk hierarchy over the predicted clusters and tests each split for
# significance, merging the ones that aren't. The score is how many clusters
# survive -> frac_retained = surviving_K / input_K (1.0 = none merged;
# < 1.0 = over-clustered, some clusters weren't statistically distinct).
#
# NOTE: scSHC needs the raw counts, which the CLUST-M contract does not yet
# provide. Until it's added to the stage schema, --rawdata_h5ad is declared here
# as an author param and must be wired in by the benchmark plan.

suppressPackageStartupMessages({
  library(scSHC)
  library(hdf5r)
  library(Matrix)
  library(jsonlite)
  library(data.table)
})

# arg parsing
source("src/common/cli.R")
p <- arg_parser("CLUST-M module (scSHC significance)")
p <- add_base_args(p)                 # --output_dir, --name
p <- add_stage_args(p, "CLUST-M")     # --clusters_tsv, --rawdata_clusters_truth (truth unused here)
p <- add_argument(p, "--rawdata_h5ad", help = "rawdata H5AD (raw counts in layers/counts)")
p <- add_argument(p, "--cores", type = "integer", default = 4, help = "workers for the null simulation")
args <- parse_args(p)

cat(sprintf("Full command: %s\n", paste(commandArgs(trailingOnly = FALSE), collapse = " ")))
cat("LOG: command line args\n----------------------------------\n")
for (i in seq_along(args)) cat(sprintf("  %s: %s\n", names(args)[i], args[[i]]))
cat("----------------------------------\n")

dir.create(args$output_dir, showWarnings = FALSE, recursive = TRUE)

# raw counts as genes x cells (anndata CSR-by-cell has the same p/i/x as CSC genes x cells)
f <- H5File$new(args$rawdata_h5ad, "r")
g <- f[["layers/counts"]]
sh <- as.integer(h5attributes(g)$shape)          # (n_cells, n_genes)
counts <- new("dgCMatrix",
  p = g[["indptr"]][], i = g[["indices"]][],
  x = as.numeric(g[["data"]][]), Dim = c(sh[2], sh[1]))
rownames(counts) <- f[["var/_index"]][]
colnames(counts) <- f[["obs/_index"]][]
f$close_all()

pred <- fread(args$clusters_tsv, header = TRUE)
pred <- pred[pred$cell_id %in% colnames(counts)]
n_dropped <- nrow(fread(args$clusters_tsv, header = TRUE, select = "cell_id")) - nrow(pred)
ids <- as.character(pred$cluster)
input_K <- length(unique(ids))

if (input_K >= 2) {
  surviving <- testClusters(counts[, pred$cell_id, drop = FALSE], ids, cores = args$cores)[[1]]
  surviving_K <- length(unique(surviving))
} else {
  surviving_K <- input_K
}

result <- list(
  n_cells = nrow(pred),
  n_dropped = n_dropped,
  input_K = input_K,
  surviving_K = surviving_K,
  frac_retained = if (input_K >= 2) surviving_K / input_K else NA_real_
)
out <- file.path(args$output_dir, sprintf("%s_cluster_shc.json", args$name))
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE), out)
