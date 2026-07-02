# Fixture writers and entrypoint runner for annotation-r tests. This slice is
# infrastructure-only (annotation.R computes no metrics yet), so fixtures only
# need to exercise cell_id-based alignment and n_cells/n_dropped/n_labels
# bookkeeping — not any metric value. Reuses write_tsv() and .repo_root from
# helper-fixtures.R (testthat loads all helper-*.R files into one shared env).

# Basic aligned fixture: 6 cells, 2 truth labels, predicted "annotation" rows
# interleaved relative to truth's blocked order (not just block-swapped) to
# prove the join is by cell_id, not row position.
annotation_fixture_basic <- function(dir) {
  cell_ids <- sprintf("c%d", 1:6)
  truth_df <- data.frame(cell_id = cell_ids, truths = c(rep("A", 3), rep("B", 3)))
  pred_df  <- data.frame(cell_id = cell_ids, annotation = c(rep("A", 3), rep("B", 3)))
  pred_df  <- pred_df[c(1, 4, 2, 5, 3, 6), ]  # interleave A/B rows

  annotations_tsv <- file.path(dir, "basic_annotations.tsv")
  truth_tsv       <- file.path(dir, "basic_truth.tsv")
  write_tsv(pred_df, annotations_tsv)
  write_tsv(truth_df, truth_tsv)

  list(annotations_tsv = annotations_tsv, truth_tsv = truth_tsv,
       n_cells = 6L, n_dropped = 0L, n_labels = 2L)
}

# One predicted-only cell (c7), one truth-only cell (c8): proves n_cells/
# n_dropped reflect an inner join by cell_id, not just row counts. Same shape
# as cluster_fixture_dropped(): n_dropped is the pred-side join loss only.
annotation_fixture_dropped <- function(dir) {
  pred_df <- data.frame(
    cell_id    = sprintf("c%d", 1:7),
    annotation = c(rep("A", 3), rep("B", 4))
  )
  truth_df <- data.frame(
    cell_id = c(sprintf("c%d", 1:6), "c8"),
    truths  = c(rep("A", 3), rep("B", 3), "B")
  )

  annotations_tsv <- file.path(dir, "dropped_annotations.tsv")
  truth_tsv       <- file.path(dir, "dropped_truth.tsv")
  write_tsv(pred_df, annotations_tsv)
  write_tsv(truth_df, truth_tsv)

  list(annotations_tsv = annotations_tsv, truth_tsv = truth_tsv,
       n_cells = 6L, n_dropped = 1L, n_labels = 2L)
}

# Runs annotation/annotation.R as a real subprocess (the CLI is the actual
# contract under test) and returns the parsed output JSON. Mirrors
# run_cluster_entrypoint(); reuses the shared .repo_root.
run_annotation_entrypoint <- function(annotations_tsv, truth_tsv, output_dir, name = "toy") {
  old_wd <- setwd(.repo_root)
  on.exit(setwd(old_wd), add = TRUE)

  args <- c(
    "annotation/annotation.R",
    "--output_dir", output_dir,
    "--name", name,
    "--annotations_tsv", annotations_tsv,
    "--rawdata_clusters_truth", truth_tsv
  )
  result <- system2("Rscript", args, stdout = TRUE, stderr = TRUE)
  status <- attr(result, "status")
  if (!is.null(status) && status != 0) {
    stop("annotation.R exited with status ", status, ":\n", paste(result, collapse = "\n"))
  }
  out_path <- file.path(output_dir, sprintf("%s_annotation_metrics.json", name))
  if (!file.exists(out_path)) {
    stop("expected output not found: ", out_path, "\n", paste(result, collapse = "\n"))
  }
  jsonlite::fromJSON(out_path)
}
