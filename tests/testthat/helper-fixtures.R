# Fixture writers and entrypoint runner for cluster-r tests. Each fixture
# returns paths + the expected values hand-computed from its construction, so
# tests assert against those, not against poem's implementation.

write_tsv <- function(df, path) {
  write.table(df, path, sep = "\t", row.names = FALSE, quote = FALSE)
}

# Perfect partition match: 8 cells, 2 true groups. Predicted clusters use a
# different label alphabet ("0"/"1" vs "A"/"B") to prove ARI/AMI/FM are
# invariant to label naming. pred rows are interleaved relative to truth's
# blocked order — not just block-swapped, which a position-based (not
# cell_id-based) join could still score as a perfect match by accident — to
# actually prove the join is by cell_id, not row position.
cluster_fixture_perfect <- function(dir) {
  cell_ids <- sprintf("c%d", 1:8)
  truth_df <- data.frame(cell_id = cell_ids, truths = c(rep("A", 4), rep("B", 4)))
  pred_df  <- data.frame(cell_id = cell_ids, cluster = c(rep("0", 4), rep("1", 4)))
  pred_df  <- pred_df[c(1, 5, 2, 6, 3, 7, 4, 8), ]  # interleave A/B rows

  clusters_tsv <- file.path(dir, "perfect_clusters.tsv")
  truth_tsv    <- file.path(dir, "perfect_truth.tsv")
  write_tsv(pred_df, clusters_tsv)
  write_tsv(truth_df, truth_tsv)

  list(clusters_tsv = clusters_tsv, truth_tsv = truth_tsv,
       n_cells = 8L, n_dropped = 0L, n_labels = 2L)
}

# One predicted cell (c9) absent from truth, one truth cell (c10) absent from
# predicted: proves n_cells/n_dropped reflect an inner join by cell_id, not
# just row counts. cluster.R computes n_dropped as nrow(pred) - n_cells, so
# it only reflects the pred-side loss (9 -> 8 matched = 1 dropped); c10's
# exclusion from the truth side isn't separately counted — asserted as-is.
cluster_fixture_dropped <- function(dir) {
  pred_df <- data.frame(
    cell_id = sprintf("c%d", 1:9),
    cluster = c(rep("0", 4), rep("1", 5))
  )
  truth_df <- data.frame(
    cell_id = c(sprintf("c%d", 1:8), "c10"),
    truths  = c(rep("A", 4), rep("B", 4), "B")
  )

  clusters_tsv <- file.path(dir, "dropped_clusters.tsv")
  truth_tsv    <- file.path(dir, "dropped_truth.tsv")
  write_tsv(pred_df, clusters_tsv)
  write_tsv(truth_df, truth_tsv)

  list(clusters_tsv = clusters_tsv, truth_tsv = truth_tsv,
       n_cells = 8L, n_dropped = 1L, n_labels = 2L)
}

# All truth labels identical: hits cluster.R's n_labels < 2 branch, which must
# return NA for every metric instead of crashing.
cluster_fixture_single_label <- function(dir) {
  pred_df  <- data.frame(cell_id = sprintf("c%d", 1:4), cluster = c("0", "0", "1", "1"))
  truth_df <- data.frame(cell_id = sprintf("c%d", 1:4), truths  = rep("A", 4))

  clusters_tsv <- file.path(dir, "single_label_clusters.tsv")
  truth_tsv    <- file.path(dir, "single_label_truth.tsv")
  write_tsv(pred_df, clusters_tsv)
  write_tsv(truth_df, truth_tsv)

  list(clusters_tsv = clusters_tsv, truth_tsv = truth_tsv,
       n_cells = 4L, n_dropped = 0L, n_labels = 1L)
}

# testthat::test_dir() runs each test file with the working directory set to
# tests/testthat, but cluster.R itself does source("src/common/cli.R") with a
# path relative to the repo root — so the *subprocess* needs cwd = repo root,
# not just a correctly-resolved script path. testthat::test_path() locates
# the repo root robustly regardless of testthat's cwd-of-the-moment.
.repo_root <- normalizePath(testthat::test_path("..", ".."))

# Runs cluster/cluster.R as a real subprocess (the CLI is the actual contract
# under test — no calling internals) and returns the parsed output JSON.
# clusters_tsv/truth_tsv/output_dir must be absolute paths (tempfile()-based
# fixtures already are), since the subprocess's cwd is forced to the repo root.
run_cluster_entrypoint <- function(clusters_tsv, truth_tsv, output_dir, name = "toy") {
  old_wd <- setwd(.repo_root)
  on.exit(setwd(old_wd), add = TRUE)

  args <- c(
    "cluster/cluster.R",
    "--output_dir", output_dir,
    "--name", name,
    "--clusters_tsv", clusters_tsv,
    "--rawdata_clusters_truth", truth_tsv
  )
  result <- system2("Rscript", args, stdout = TRUE, stderr = TRUE)
  status <- attr(result, "status")
  if (!is.null(status) && status != 0) {
    stop("cluster.R exited with status ", status, ":\n", paste(result, collapse = "\n"))
  }
  out_path <- file.path(output_dir, sprintf("%s_cluster_metrics.json", name))
  if (!file.exists(out_path)) {
    stop("expected output not found: ", out_path, "\n", paste(result, collapse = "\n"))
  }
  jsonlite::fromJSON(out_path)
}
