# metrics

Metrics module for the [omni-scrna](https://github.com/omni-scrna) OmniBenchmark pipeline.

## Entrypoints

| Entrypoint | Language | Input | Metrics |
|---|---|---|---|
| `cluster-r` | R (poem) | Cluster assignment TSV | ARI, AMI, FM, VM, EH, EC |
| `annotation-r` | R (MLmetrics) | Predicted cell-type annotation TSV | ACC, BACC, F1, KAPPA |
| `integration-r` | R (CellMixS) | Batch-corrected embedding TSV | cms, entropy, isi |
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
(accuracy-family metrics) — predicted and truth label vocabularies are compared as-is, with no
label harmonization / ontology resolution mapping.

`integration-r` is a third kind of problem again: batch-mixing rather than label agreement.
Its metrics need a per-cell *batch* label rather than the cell-type truth every other
entrypoint uses, so it takes two additional inputs (`--rawdata_h5ad`, `--properties_info`) to
source it, mirroring the upstream `INTG8` method stage's own contract.
