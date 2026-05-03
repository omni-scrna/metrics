#!/usr/bin/env python3
# Embedding quality metrics for omnibenchmark.
#
# Implementation notes
# --------------------
# - rhdf5 writes R matrices in column-major order; the stored HDF5 dimensions
#   are [n_cols, n_rows] from h5py's perspective. Shape is validated against
#   cell_ids at load time and transposed if needed.
# - All metrics require >= 2 labels and >= 2 samples; return NaN otherwise.

import json
import sys
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import polars as pl
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "cli"))
from cli import parse_args  # noqa: E402

# Add or remove metrics here; all must have signature fn(X, labels) -> float.
METRICS: dict[str, Callable] = {
    "silhouette": silhouette_score,
    "davies_bouldin": davies_bouldin_score,
    "calinski_harabasz": calinski_harabasz_score,
}

_SCHEMA = Path(__file__).parent.parent / "cli" / "embedding.json"


def load_pca(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        raw = f["embedding"][:]
        cell_ids = f["cell_ids"][:].astype(str)

    # R/Python HDF5 convention mismatch: transpose to (n_cells, n_components).
    if raw.shape[0] != len(cell_ids):
        raw = raw.T
    return raw, cell_ids


def main() -> None:
    args = parse_args(_SCHEMA)

    embedding, cell_ids = load_pca(args.pca)

    # Align embedding rows with truth labels by cell_id.
    truth_df = pl.read_csv(args.clusters_truth, separator="\t")
    pca_df = pl.DataFrame({"cell_id": cell_ids, "idx": range(len(cell_ids))})
    merged = pca_df.join(
        truth_df.select(["cell_id", "truths"]), on="cell_id", how="inner"
    )

    aligned_embedding = embedding[merged["idx"].to_numpy()]
    aligned_labels = merged["truths"].cast(str).to_numpy()
    n_cells = len(merged)
    n_labels = merged["truths"].n_unique()
    n_dropped = len(cell_ids) - n_cells

    if n_cells >= 2 and n_labels >= 2:
        scores = {
            name: float(fn(aligned_embedding, aligned_labels))
            for name, fn in METRICS.items()
        }
    else:
        scores = {name: float("nan") for name in METRICS}

    with open(args.output_dir / f"{args.name}_embedding_metrics.json", "w") as fh:
        json.dump(
            {
                "n_cells": n_cells,
                "n_labels": n_labels,
                "n_dropped": n_dropped,
                **scores,
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
