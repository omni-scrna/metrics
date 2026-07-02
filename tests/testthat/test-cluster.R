CLUST_METRICS <- c("ARI", "AMI", "FM", "VM", "EH", "EC")

with_tmp_out <- function(fn) {
  dir <- tempfile("cluster-fixture-")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  fn(dir, file.path(dir, "out"))
}

test_that("perfect partition match: ARI, AMI, FM are all 1", {
  with_tmp_out(function(dir, out_dir) {
    fx  <- cluster_fixture_perfect(dir)
    out <- run_cluster_entrypoint(fx$clusters_tsv, fx$truth_tsv, out_dir)

    expect_equal(out$n_cells, fx$n_cells)
    expect_equal(out$n_dropped, fx$n_dropped)
    expect_equal(out$n_labels, fx$n_labels)
    expect_equal(as.numeric(out$ARI), 1, tolerance = 1e-8)
    expect_equal(as.numeric(out$AMI), 1, tolerance = 1e-8)
    expect_equal(as.numeric(out$FM), 1, tolerance = 1e-8)
  })
})

test_that("cells are aligned by cell_id and n_dropped reflects the pred-side join loss", {
  with_tmp_out(function(dir, out_dir) {
    fx  <- cluster_fixture_dropped(dir)
    out <- run_cluster_entrypoint(fx$clusters_tsv, fx$truth_tsv, out_dir)

    expect_equal(out$n_cells, fx$n_cells)
    expect_equal(out$n_dropped, fx$n_dropped)
    expect_equal(out$n_labels, fx$n_labels)
  })
})

test_that("single-label truth returns NULL for every metric instead of crashing", {
  with_tmp_out(function(dir, out_dir) {
    fx  <- cluster_fixture_single_label(dir)
    out <- run_cluster_entrypoint(fx$clusters_tsv, fx$truth_tsv, out_dir)

    expect_equal(out$n_cells, fx$n_cells)
    expect_equal(out$n_labels, fx$n_labels)

    for (metric in CLUST_METRICS) {
      expect_identical(out[[metric]], NULL)
    }
  })
})
