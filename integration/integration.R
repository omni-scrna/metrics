#!/usr/bin/env Rscript
# Integration (batch-correction) metrics for omnibenchmark.
#
# Implementation notes
# --------------------
# - Aligns the corrected embedding against ground-truth cell-type labels (by cell_id) for
#   n_cells/n_labels/n_dropped bookkeeping — unchanged from the infrastructure-only slice.
# - Separately aligns the corrected embedding against per-cell *batch* labels (a different
#   piece of metadata than the cell-type truth above) to compute CellMixS batch-mixing
#   metrics: cms, entropy, isi. Batch labels are sourced from the pre-integration
#   rawdata_h5ad's obs/<batch_var> column, with batch_var named via properties_info — the
#   same source harmony.R (the INTG8 method stage) itself reads for batch correction.
# - CellMixS's other metrics (ldfDiff, localStructure, mixingMetric) are out of scope: the
#   first two need a second, pre-integration embedding this stage doesn't have, and the third
#   wraps Seurat::MixingMetric (a much heavier dependency). See docs/human/GOAL.md.

suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
  library(rhdf5)
  library(yaml)
  library(SingleCellExperiment)
  library(CellMixS)
})

CMS_METRICS <- c("cms", "entropy", "isi")

# arg parsing
source("src/common/cli.R")
p <- arg_parser("INTG8-M module")
p <- add_base_args(p)                    # --output_dir, --name
p <- add_stage_args(p, "INTG8-M")        # the stage I/O contract
# your own method params — argparser directly (its add_argument requires `help`):
p <- add_argument(p, "--k", type = "integer", default = 20L,
  help = "Neighborhood size (knn) for CellMixS cms/entropy/isi")
p <- add_argument(p, "--n_dim", type = "integer", default = 10L,
  help = "Requested number of corrected-embedding dims to use as the CellMixS subspace (clamped to the embedding's actual width)")
p <- add_argument(p, "--cell_min", type = "integer", default = 10L,
  help = "Minimum cells per batch for cms's Anderson-Darling test (CellMixS docs: must be > 4)")
args <- parse_args(p)                    # argparser's own parser

# logging
cat(sprintf("Full command: %s\n", paste(commandArgs(trailingOnly = FALSE), collapse = " ")))
cat(sprintf("LOG: command line args\n----------------------------------\n"))
for (i in 1:length(args)) {
  cat(sprintf("  %s: %s\n", names(args)[i], args[[i]]))
}
cat(sprintf("----------------------------------\n"))


dir.create(args$output_dir, showWarnings = FALSE, recursive = TRUE)

corrected <- fread(args$corrected_tsv, header = TRUE)
truth <- fread(args$rawdata_clusters_truth, header = TRUE)

# Align the corrected embedding rows with truth labels by cell_id.
idx <- match(corrected$cell_id, truth$cell_id)
mask <- !is.na(idx)

n_cells <- sum(mask)
n_labels <- length(unique(truth$truths[idx[mask]]))
n_dropped <- sum(!mask)

# Batch labels: a different source than the cell-type truth above. Some datasets declare no
# batch_var at all (e.g. "sc-mix"), in which case batch-mixing metrics aren't computable.
props <- yaml::read_yaml(args$properties_info)
batch_var <- props$batch_var
have_batch <- !is.null(batch_var) && nzchar(batch_var)

if (have_batch) {
  h5_cell_ids <- as.character(rhdf5::h5read(args$rawdata_h5ad, "obs/_index"))
  h5_batch_vals <- as.character(rhdf5::h5read(args$rawdata_h5ad, paste0("obs/", batch_var)))
  batch_dt <- data.table(cell_id = h5_cell_ids, batch = h5_batch_vals)

  idx_batch <- match(corrected$cell_id, batch_dt$cell_id)
  mask_metric <- mask & !is.na(idx_batch)
  n_batches <- length(unique(batch_dt$batch[idx_batch[mask_metric]]))
} else {
  mask_metric <- rep(FALSE, nrow(corrected))
  n_batches <- 0L
}

if (have_batch && n_batches >= 2 && sum(mask_metric) > 0) {
  dim_cols <- setdiff(colnames(corrected), "cell_id")
  emb <- as.matrix(corrected[mask_metric, ..dim_cols])
  rownames(emb) <- corrected$cell_id[mask_metric]

  batch_factor <- factor(batch_dt$batch[idx_batch[mask_metric]])

  sce <- SingleCellExperiment(
    reducedDims = list(PCA = emb),
    colData = DataFrame(batch = batch_factor, row.names = rownames(emb))
  )

  n_dim_eff <- min(args$n_dim, ncol(emb))

  sce <- evalIntegration(
    metrics = CMS_METRICS, sce = sce, group = "batch",
    dim_red = "PCA", n_dim = n_dim_eff, k = args$k, cell_min = args$cell_min
  )

  cd <- colData(sce)
  scores <- list(
    cms_mean     = mean(cd$cms, na.rm = TRUE),
    entropy_mean = mean(cd$entropy, na.rm = TRUE),
    isi_mean     = mean(cd$isi, na.rm = TRUE)
  )
} else {
  scores <- setNames(as.list(rep(NA_real_, length(CMS_METRICS))), paste0(CMS_METRICS, "_mean"))
}

result <- c(
  list(n_cells = n_cells, n_labels = n_labels, n_dropped = n_dropped, n_batches = n_batches),
  scores
)
out <- file.path(args$output_dir, sprintf("%s_integration_metrics.json", args$name))
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE, na = "null"), out)
