with_tmp_out <- function(fn) {
  dir <- tempfile("annotation-fixture-")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  fn(dir, file.path(dir, "out"))
}

test_that("basic aligned fixture: n_cells/n_labels/n_dropped are correct", {
  with_tmp_out(function(dir, out_dir) {
    fx  <- annotation_fixture_basic(dir)
    out <- run_annotation_entrypoint(fx$annotations_tsv, fx$truth_tsv, out_dir)

    expect_equal(out$n_cells, fx$n_cells)
    expect_equal(out$n_dropped, fx$n_dropped)
    expect_equal(out$n_labels, fx$n_labels)
  })
})

test_that("cells are aligned by cell_id and n_dropped reflects the pred-side join loss", {
  with_tmp_out(function(dir, out_dir) {
    fx  <- annotation_fixture_dropped(dir)
    out <- run_annotation_entrypoint(fx$annotations_tsv, fx$truth_tsv, out_dir)

    expect_equal(out$n_cells, fx$n_cells)
    expect_equal(out$n_dropped, fx$n_dropped)
    expect_equal(out$n_labels, fx$n_labels)
  })
})
