#!/usr/bin/env Rscript
# Cell-type annotation metrics for omnibenchmark.
#
# Implementation notes
# --------------------
# - Predicted vs. true cell-type labels is a label-matching (classification-
#   agreement) problem, not a clustering/partition-comparison one, so this
#   uses accuracy-family metrics rather than cluster.R's poem::getPartitionMetrics.
# - All metrics require >= 2 labels; returns NA otherwise (same shape as cluster.R).

suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
  library(MLmetrics)
})

METRICS <- c("ACC", "BACC", "F1", "KAPPA")

# Accuracy/recall/F1 use MLmetrics' vetted implementations (recall and F1 are
# binary-only, so macro-averaged here via a one-vs-rest loop over classes).
# Cohen's kappa is hand-computed from the confusion matrix marginals: no lean
# CRAN/conda-forge package provides multiclass kappa alone without pulling in
# a much heavier stack (e.g. caret, yardstick).
classification_metrics <- function(pred, truth) {
  classes <- union(unique(truth), unique(pred))

  acc <- Accuracy(y_pred = pred, y_true = truth)

  per_class_recall <- sapply(classes, function(k) Recall(y_true = truth, y_pred = pred, positive = k))
  bacc <- mean(per_class_recall, na.rm = TRUE)

  per_class_f1 <- sapply(classes, function(k) F1_Score(y_true = truth, y_pred = pred, positive = k))
  macro_f1 <- mean(per_class_f1, na.rm = TRUE)

  cm <- table(factor(truth, levels = classes), factor(pred, levels = classes))
  n <- length(truth)
  p_e <- sum((rowSums(cm) / n) * (colSums(cm) / n))
  kappa <- if (p_e == 1) NA_real_ else (acc - p_e) / (1 - p_e)

  list(ACC = acc, BACC = bacc, F1 = macro_f1, KAPPA = kappa)
}

# arg parsing
source("src/common/cli.R")
p <- arg_parser("ANNO-M module")
p <- add_base_args(p)                    # --output_dir, --name
p <- add_stage_args(p, "ANNO-M")     # the stage I/O contract
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

pred <- fread(args$annotations_tsv, header = TRUE)
truth <- fread(args$rawdata_clusters_truth, header = TRUE)

merged <- merge(pred, truth, by = "cell_id", all = FALSE)

n_cells <- nrow(merged)
n_dropped <- nrow(pred) - n_cells
n_labels <- length(unique(merged$truths))

# NOTE: predicted (merged$predicted_labels) and truth (merged$truths) labels are
# compared as-is below, with no label harmonization / resolution mapping —
# i.e. no reconciliation of differing cell-type ontologies or granularities
# (e.g. "CD4 T cell" predicted vs. "T cell" truth) between the two label
# vocabularies.
if (n_labels >= 2) {
  scores <- classification_metrics(merged$predicted_labels, merged$truths)
} else {
  scores <- setNames(as.list(rep(NA_real_, length(METRICS))), METRICS)
}

result <- c(
  list(n_cells = n_cells, n_labels = n_labels, n_dropped = n_dropped),
  scores
)
out <- file.path(args$output_dir, sprintf("%s_annotation_metrics.json", args$name))
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE, na = "null"), out)
