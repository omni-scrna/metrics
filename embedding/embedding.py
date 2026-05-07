#!/usr/bin/env python3
# Embedding quality metrics for omnibenchmark.
#
# Implementation notes
# --------------------
# - All metrics require >= 2 labels; return NaN otherwise.

import json
import sys
from pathlib import Path
from typing import Callable

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


def main() -> None:
    args = parse_args()

    pca_df = pl.read_csv(args.pcas, separator="\t")
    truth_df = pl.read_csv(args.clusters_truth, separator="\t")

    # Align embedding rows with truth labels by cell_id.
    merged = pca_df.join(
        truth_df.select(["cell_id", "truths"]), on="cell_id", how="inner"
    )

    embedding = merged.drop(["cell_id", "truths"]).to_numpy()
    aligned_labels = merged["truths"].cast(str).to_numpy()
    n_cells = len(merged)
    n_labels = merged["truths"].n_unique()
    n_dropped = len(pca_df) - n_cells

    if n_labels >= 2:
        scores = {
            name: float(fn(embedding, aligned_labels)) for name, fn in METRICS.items()
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
