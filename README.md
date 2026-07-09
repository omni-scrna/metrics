# metrics

Metrics module for the [omni-scrna](https://github.com/omni-scrna) OmniBenchmark pipeline.

## Entrypoints

| Entrypoint | Language | Input | Metrics |
|---|---|---|---|
| `cluster-r` | R (poem) | Cluster assignment TSV | ARI, AMI, FM, VM, EH, EC |
| `annotation-r` | R (MLmetrics, ontologyIndex) | Predicted cell-type annotation TSV | ACC, BACC, F1, KAPPA, EXACT, PARENT, CHILD, SIBLING, NO_MATCH |
| `integration-r` | R (CellMixS) | Batch-corrected embedding TSV | cms, entropy, isi, ldfDiff |
| `integration-py` | Python (scib-metrics) | Batch-corrected embedding TSV | label ASW, cLISI, PCR comparison |
| `embedding-py` | Python (sklearn) | PCA embedding TSV | silhouette, Davies-Bouldin, Calinski-Harabasz |
| `embedding-r` | R (poem) | PCA embedding TSV | meanSW, meanClassSW, pnSW, minClassSW, CDbW, cohesion, compactness, sep, DBCV |
| `graph-r` | R (poem) | KNN neighbor graph (HDF5) | SI, ISI, NP, AMSP, PWC, NCE, adhesion, cohesion |

Each entrypoint takes a benchmark stage's method output (a cluster assignment, predicted
cell-type annotation, batch-corrected embedding, PCA embedding, or KNN neighbor graph) plus a
ground-truth cell-type labels TSV, aligns rows by `cell_id`, computes its fixed metric suite,
and writes one JSON scores file (`<name>_<stage>_metrics.json`) to `--output_dir`.

`cluster-r` and `annotation-r` both compare predicted vs. true cell-type labels, but as
different kinds of problem: `cluster-r` treats it as a clustering/partition comparison (poem's
partition metrics), while `annotation-r` treats it as a classification-agreement problem
(accuracy-family metrics) — predicted and truth label vocabularies (`predicted_labels`/
`truths`) are compared as-is, with no label harmonization / ontology resolution mapping.
`annotation-r` additionally scores a second, ontology-ID-based pair of columns (`CL_pred`/
`CL_label`) for resolution-aware accuracy (popV,
https://www.nature.com/articles/s41588-024-01993-3): `EXACT`/`PARENT`/`CHILD`/`SIBLING` match
rates + a `NO_MATCH` rate, classifying each cell's prediction against truth via one-hop `is_a`
relationships in a dataset-specific Cell Ontology DAG supplied as `--cell_ontology_obo`.

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
pre-integration embedding) in addition to `--corrected_tsv`/`--rawdata_clusters_truth`.
