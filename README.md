# metrics

Embedding quality metrics module for the [omni-scrna](https://github.com/omni-scrna) benchmark.

## Entrypoints

| Entrypoint | Language | Metrics |
|---|---|---|
| `embedding-py` | Python (sklearn) | silhouette, Davies-Bouldin, Calinski-Harabasz |
| `embedding-r` | R (POem) | mean silhouette width, CDbW, DBCV |

Both entrypoints take a PCA embedding (HDF5) and ground truth cluster labels (TSV), align by cell ID, and write a JSON of scores.
