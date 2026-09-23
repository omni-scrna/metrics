# metrics

Metrics module for the [omni-scrna](https://github.com/omni-scrna) OmniBenchmark pipeline.

## Entrypoints

| Entrypoint | Language | Input | Metrics |
|---|---|---|---|
| `cluster-r` | R (poem) | Cluster assignment TSV | ARI, AMI, FM, VM, EH, EC |
| `annotation-r` | R (MLmetrics) | Predicted cell-type annotation TSV | ACC, BACC, F1, KAPPA |
| `integration-r` | R (CellMixS) | Batch-corrected embedding TSV | cms, entropy, isi, ldfDiff |
| `integration-py` | Python (scib-metrics + sklearn) | Batch-corrected embedding TSV | label ASW, cLISI, PCR comparison, ARI, NMI |
| `embedding-py` | Python (sklearn) | PCA embedding TSV | silhouette, Davies-Bouldin, Calinski-Harabasz |
| `embedding-r` | R (poem) | PCA embedding TSV | meanSW, meanClassSW, pnSW, minClassSW, CDbW, cohesion, compactness, sep, DBCV |
| `graph-r` | R (poem) | KNN neighbor graph (HDF5) | SI, ISI, NP, AMSP, PWC, NCE, adhesion, cohesion |
| `normalization-r` | R (stats) | PCA embedding TSV and size factors TSV | canonical correlation |

Each entrypoint takes a benchmark stage's method output (a cluster assignment, predicted
cell-type annotation, batch-corrected embedding, PCA embedding, or KNN neighbor graph) plus a
ground-truth cell-type labels TSV, aligns rows by `cell_id`, computes its fixed metric suite,
and writes one JSON scores file (`<name>_<stage>_metrics.json`) to `--output_dir`.

`normalization-r` measures sequencing-depth confounding in an embedding. It aligns
the PCA embedding and size factors by `cell_id`, uses the first 10 dimensions or all
available dimensions when fewer than 10 are present, and reports the first canonical
correlation from `stats::cancor()`. Lower values indicate less association between
the embedding and library-size-derived size factors. The output also records the
number of dimensions and cells used.

`cluster-r` and `annotation-r` both compare predicted vs. true cell-type labels, but as
different kinds of problem: `cluster-r` treats it as a clustering/partition comparison (poem's
partition metrics), while `annotation-r` treats it as a classification-agreement problem
(accuracy-family metrics) — predicted and truth label vocabularies are compared as-is, with no
label harmonization / ontology resolution mapping.

`integration-r` is a third kind of problem again: batch-mixing rather than label agreement.
Its metrics need a per-cell *batch* label rather than the cell-type truth every other
entrypoint uses, so it takes two additional inputs (`--rawdata_h5ad`, `--properties_info`) to
source it, mirroring the upstream `INTG8` method stage's own contract. It also takes a third
additional input, `--pcas_tsv` (the `PCA` stage's joint pre-integration embedding), to compute
`ldfDiff` — a local-structure-distortion metric that compares each cell's local density factor
before vs. after integration, split per batch. CellMixS's `locStructure` metric is intentionally
not included: its `dim_red` argument is dead code in the installed version (it always
recomputes a fresh per-batch PCA from a raw expression assay internally, so it can't reuse a
precomputed embedding the way `ldfDiff` can), and `mixMetric` wraps `Seurat::MixingMetric` (a
much heavier dependency).

`integration-py` targets the same `INTG8-M` stage as `integration-r` but evaluates bio-
conservation and local-structure distortion instead of batch-mixing: label ASW and cLISI (both
from [scib-metrics](https://github.com/YosefLab/scib-metrics)) compare the corrected embedding
against cell-type truth, while PCR comparison (`scib_metrics.pcr_comparison`) compares batch
covariate variance explained in the pre- vs. post-integration embedding — so it now uses
`--rawdata_h5ad`/`--properties_info` (for per-cell batch labels) and `--pcas_tsv` (for the
pre-integration embedding) in addition to `--corrected_tsv`/`--rawdata_clusters_truth`. ARI and
NMI (`sklearn.metrics`) are a fourth, biological-preservation metric pair: how well cluster
labels obtained by clustering the *integrated* embedding (`--clusters_corrected_tsv`, the
`CLUST-C` stage's output) recover the cell-type truth. `scib-metrics` itself has no function
that scores externally-supplied cluster labels — its own `nmi_ari_cluster_labels_kmeans/leiden`
cluster internally and are thin wrappers around exactly these two `sklearn` calls — so ARI/NMI
are computed directly against the real `CLUST-C` labels instead of re-clustering.
