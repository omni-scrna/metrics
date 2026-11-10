#!/usr/bin/env Rscript
# Embedding quality metrics (R/poem) for omnibenchmark.
#
# Implementation notes
# --------------------
# - All metrics require >= 2 labels; returns NA otherwise (handled by poem).

suppressPackageStartupMessages({
  library(poem)
  library(jsonlite)
  library(data.table)
})


# Add or remove metrics here; must be valid for level = "dataset".
METRICS <- c(
  "meanSW",
  "meanClassSW",
  "pnSW",
  "minClassSW",
  "cdbw",
  "cohesion",
  "compactness",
  "sep",
  "dbcv"
)

# arg parsing
# ob runs entrypoints from the output tree, not the module dir: resolve paths from this file
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "../src/common/cli.R"))
p <- arg_parser("EMBED_M: embedding metrics for one embedding")
p <- add_base_args(p)                    # --output_dir, --name
p <- add_argument(p, "--embedding_tsv", help = "embedding (cell_id, then dimensions)")
p <- add_argument(p, "--rawdata_clusters_truth", help = "truth labels (cell_id, truths)")
args <- parse_args_outputs(p)

# logging
cat(sprintf("Full command: %s\n", paste(commandArgs(trailingOnly = FALSE), collapse = " ")))
cat(sprintf("LOG: command line args\n----------------------------------\n"))
for (i in 1:length(args)) {
  cat(sprintf("  %s: %s\n", names(args)[i], args[[i]]))
}
cat(sprintf("----------------------------------\n"))



pca <- fread(args$embedding_tsv, header = TRUE)
truth <- fread(args$rawdata_clusters_truth, header = TRUE)

# Align embedding rows with truth labels by cell_id.
idx <- match(pca$cell_id, truth$cell_id)
mask <- !is.na(idx)
aligned_embedding <- pca[mask, !"cell_id"]
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
out <- output_path(args, "embedding_metrics", "_embedding_metrics.json")
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE), out)
