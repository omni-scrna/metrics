#!/usr/bin/env Rscript
# EMBED_M (gather): embedding quality metrics (R/poem) for every embedding of one dataset.
#
# ob gathers every embedding_tsv under one DATA node (PCA, ISOMAP, CNTFCT, RDIMR
# arms alike) and hands them over in one file (`--from-files`, lineage.json v2,
# ob api 0.8), with that DATA node's truth labels as `--rawdata_clusters_truth`.
# Each embedding is scored in turn, so the packages load once per dataset.
#
# Output: one long TSV, one row per (embedding, metric), shaped like PCA_M's:
# dataset, upstream stages (norm, feat, ...), embedding_stage, embedding,
# embedding_params, <stage>_<parameter> (member_lineage_cols in src/fanin.R;
# from the manifest, never from paths), then n_cells, n_labels, n_dropped,
# metric, value, error.
#
# Implementation notes
# --------------------
# - All metrics require >= 2 labels; NA otherwise.
# - An embedding that fails to score keeps its rows, with NA values and the
#   error message, so one bad member does not lose the rest of the table.

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
# resolve paths from this file, not the working directory
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "../src/common/cli.R"))
source(file.path(here, "../src/fanin.R"))
SCHEMA_DIR <- file.path(here, "../src/common/schema") # cli.R's default is relative to the cwd
p <- arg_parser("EMBED_M: embedding metrics for every embedding of one dataset")
p <- add_base_args(p)                    # --output_dir, --name
p <- add_stage_args(p, "EMBED_M")        # --from-files, --rawdata_clusters_truth
args <- parse_args_outputs(p)

# logging
cat(sprintf("Full command: %s\n", paste(commandArgs(trailingOnly = FALSE), collapse = " ")))


# Score one embedding against the truth labels, aligned by cell_id.
score <- function(path, truth) {
  emb <- fread(path, header = TRUE)
  idx <- match(emb$cell_id, truth$cell_id)
  mask <- !is.na(idx)
  labels <- as.factor(truth$truths[idx[mask]])
  n_labels <- length(unique(labels))
  values <- setNames(rep(NA_real_, length(METRICS)), METRICS)
  if (n_labels >= 2) {
    result_df <- getEmbeddingMetrics(
      emb[mask, !"cell_id"],
      labels = labels,
      metrics = METRICS,
      level = "dataset"
    )
    got <- intersect(METRICS, names(result_df))
    values[got] <- vapply(got, function(m) as.numeric(result_df[[m]][1]), numeric(1))
  }
  list(n_cells = sum(mask), n_labels = n_labels, n_dropped = sum(!mask), values = values)
}


members <- manifest_members(args$from_files, "embedding_tsv")
if (length(members) == 0L) stop(sprintf("no embedding_tsv member in %s", args$from_files))
truth <- fread(args$rawdata_clusters_truth, header = TRUE)
cat(sprintf("%d embeddings, %d truth labels\n", length(members), nrow(truth)))

rows <- vector("list", length(members))
for (i in seq_along(members)) {
  m <- members[[i]]
  lin <- member_lineage_cols(m, as = "embedding")
  t0 <- Sys.time()
  res <- tryCatch(score(m$files$embedding_tsv, truth), error = function(e) e)
  err <- if (inherits(res, "error")) conditionMessage(res) else NA_character_
  cat(sprintf("[%d/%d] %s.%s %s: %s (%.0fs)\n", i, length(members), lin$embedding_stage,
              lin$embedding, lin$embedding_params %||% "",
              if (is.na(err)) "ok" else paste("FAILED:", err),
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  if (!is.na(err)) {
    res <- list(n_cells = NA_integer_, n_labels = NA_integer_, n_dropped = NA_integer_,
                values = setNames(rep(NA_real_, length(METRICS)), METRICS))
  }
  rows[[i]] <- as.data.table(c(lin, list(
    n_cells = res$n_cells, n_labels = res$n_labels, n_dropped = res$n_dropped,
    metric = names(res$values), value = unname(res$values), error = err
  )))
}

# Parameter columns differ between arms: fill leaves them empty where absent.
table <- rbindlist(rows, fill = TRUE)
tail_cols <- c("n_cells", "n_labels", "n_dropped", "metric", "value", "error")
setcolorder(table, c(setdiff(names(table), tail_cols), tail_cols))

out <- output_path(args, "embedding_metrics_tsv", "_embedding_metrics.tsv")
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
fwrite(table, out, sep = "\t", na = "")
cat(sprintf("wrote %d rows for %d embeddings: %s\n", nrow(table), length(members), out))
