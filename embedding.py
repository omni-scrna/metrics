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

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / "src"))
from cli import parse_embedding_args  # noqa: E402

METRICS: dict[str, Callable] = {
    "silhouette": silhouette_score,
    "davies_bouldin": davies_bouldin_score,
    "calinski_harabasz": calinski_harabasz_score,
}


def load_pca(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        raw = f["embedding"][:]
        cell_ids = f["cell_ids"][:].astype(str)

    # R/Python HDF5 convention mismatch: transpose to (n_cells, n_components).
    if raw.ndim == 2 and raw.shape[0] != len(cell_ids):
        raw = raw.T
    assert raw.shape[0] == len(cell_ids), (
        f"embedding rows ({raw.shape[0]}) != cell_ids ({len(cell_ids)})"
    )
    return raw, cell_ids


def main() -> None:
    args = parse_embedding_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "pca", "clusters_truth"):
        print(f"  {k}: {getattr(args, k)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    embedding, cell_ids = load_pca(args.pca)
    print(f"  embedding: {embedding.shape[0]} cells x {embedding.shape[1]} components")

    truth_df = pl.read_csv(args.clusters_truth, separator="\t").drop_nulls(
        subset=["truths"]
    )
    print(f"  ground truth cells: {len(truth_df)}")

    pca_df = pl.DataFrame({"cell_id": cell_ids, "idx": range(len(cell_ids))})
    merged = pca_df.join(
        truth_df.select(["cell_id", "truths"]), on="cell_id", how="inner"
    )

    aligned_embedding = embedding[merged["idx"].to_numpy()]
    aligned_labels = merged["truths"].cast(str).to_numpy()
    n_cells = len(merged)
    n_labels = merged["truths"].n_unique()
    n_dropped = len(cell_ids) - n_cells
    print(f"  aligned cells: {n_cells} ({n_dropped} dropped — no truth label)")

    valid = n_cells >= 2 and n_labels >= 2
    scores = {
        name: float(fn(aligned_embedding, aligned_labels)) if valid else float("nan")
        for name, fn in METRICS.items()
    }
    for name, val in scores.items():
        print(f"  {name}: {val:.4f}")

    out = args.output_dir / f"{args.name}_embedding_metrics.json"
    with out.open("w") as fh:
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
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
