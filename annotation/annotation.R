#!/usr/bin/env Rscript
# Cell-type annotation metrics for omnibenchmark.
#
# Implementation notes
# --------------------
# - Predicted vs. true cell-type labels is a label-matching (classification-
#   agreement) problem, not a clustering/partition-comparison one, so this
#   uses accuracy-family metrics rather than cluster.R's poem::getPartitionMetrics.
# - All metrics require >= 2 labels; returns NA otherwise (same shape as cluster.R).
# - Resolution-aware Cell Ontology accuracy (EXACT/PARENT/CHILD/SIBLING/NO_MATCH,
#   from popV: https://www.nature.com/articles/s41588-024-01993-3) scores CL_pred vs.
#   CL_label (CL term IDs) against the DAG in --cell_ontology_obo, independently of the
#   n_labels >= 2 guard above (it's informative even with few/one truths value — the
#   scenario the paper calls out as most useful). Match semantics are one-hop only,
#   verified against popV's actual reference implementation
#   (popv/reproducibility/_accuracy.py's match_type()), not just the paper's prose:
#   PARENT/CHILD are an *immediate* is_a relation, SIBLING is sharing an *immediate*
#   parent (checked first), anything farther apart is NO_MATCH.

suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
  library(MLmetrics)
  library(ontologyIndex)
})

METRICS <- c("ACC", "BACC", "F1", "KAPPA")
RESOLUTION_METRICS <- c("EXACT", "PARENT", "CHILD", "SIBLING", "NO_MATCH")

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

# Resolution-aware Cell Ontology accuracy (popV). pred_ids/truth_ids are CL term
# IDs (CL_pred/CL_label); ontology is an ontologyIndex object from
# get_ontology(cell_ontology_obo) (is_a relations only). Every pred/truth ID must
# be a node in the ontology — fail loudly rather than silently bucketing a typo'd
# or out-of-scope term as NO_MATCH.
resolution_metrics <- function(pred_ids, truth_ids, ontology) {
  unknown <- setdiff(union(pred_ids, truth_ids), ontology$id)
  if (length(unknown) > 0) {
    stop(sprintf("CL_pred/CL_label term(s) not found in --cell_ontology_obo: %s",
                  paste(unknown, collapse = ", ")), call. = FALSE)
  }

  match_type <- function(p, t) {
    if (identical(p, t)) return("EXACT")
    if (length(intersect(ontology$parents[[p]], ontology$parents[[t]])) > 0) return("SIBLING")
    if (p %in% ontology$parents[[t]]) return("PARENT")  # pred is truth's immediate parent
    if (t %in% ontology$parents[[p]]) return("CHILD")   # pred is truth's immediate child
    "NO_MATCH"
  }
  types <- mapply(match_type, pred_ids, truth_ids)
  n <- length(types)
  setNames(as.list(sapply(RESOLUTION_METRICS, function(m) sum(types == m) / n)), RESOLUTION_METRICS)
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

# NOTE: predicted (merged$predicted_labels) and truth (merged$truths) free-text
# labels are compared as-is below, with no label harmonization / resolution
# mapping — i.e. no reconciliation of differing cell-type ontologies or
# granularities (e.g. "CD4 T cell" predicted vs. "T cell" truth) between the two
# label vocabularies. This is a separate, deferred non-goal for this
# ACC/BACC/F1/KAPPA block specifically; it's not resolved by the CL_pred/CL_label
# resolution-aware metrics below, which score a different pair of (ID-based)
# columns against the Cell Ontology DAG.
if (n_labels >= 2) {
  scores <- classification_metrics(merged$predicted_labels, merged$truths)
} else {
  scores <- setNames(as.list(rep(NA_real_, length(METRICS))), METRICS)
}

if (!all(c("CL_pred", "CL_label") %in% names(merged))) {
  stop("--annotations_tsv must have a CL_pred column and --rawdata_clusters_truth a CL_label column", call. = FALSE)
}
if (n_cells > 0) {
  ontology <- get_ontology(args$cell_ontology_obo, propagate_relationships = "is_a", extract_tags = "minimal")
  resolution_scores <- resolution_metrics(merged$CL_pred, merged$CL_label, ontology)
} else {
  resolution_scores <- setNames(as.list(rep(NA_real_, length(RESOLUTION_METRICS))), RESOLUTION_METRICS)
}

result <- c(
  list(n_cells = n_cells, n_labels = n_labels, n_dropped = n_dropped),
  scores,
  resolution_scores
)
out <- file.path(args$output_dir, sprintf("%s_annotation_metrics.json", args$name))
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE, na = "null"), out)
