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
source("src/common/cli.R")
p <- arg_parser("EMBED-M module")
p <- add_base_args(p)                    # --output_dir, --name
p <- add_stage_args(p, "EMBED-M")     # the stage I/O contract
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

pca <- fread(args$pcas_tsv, header = TRUE)
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
out <- file.path(
  args$output_dir,
  sprintf("%s_embedding_metrics.json", args$name)
)
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE), out)
