#!/usr/bin/env python3
# Embedding quality metrics for omnibenchmark.
#
# Implementation notes
# --------------------
# - All metrics require >= 2 labels; return NaN otherwise.

import argparse
import sys
from pathlib import Path
from typing import Callable
import polars as pl
import json

sys.path.insert(0, str(Path(__file__).parent / "../src"))  # vendored `common` (src/common) + module-local writers
from common import cli  # noqa: E402

def parse_args():
    # We own the parser; src/common/cli injects the shared contract (base args + the
    # `EMBED-M` stage I/O from common/schema). This module's method params are
    # hand-rolled below, so the whole CLI stays visible here.
    p = argparse.ArgumentParser(description="EMBED-M module (scanpy-backed)")
    cli.add_base_args(p)              # --output_dir, --name
    cli.add_stage_args(p, "EMBED-M")  # --pcas_tsv, --rawdata_clusters_truth
    #p.add_argument("--number_selected", type=int, required=True, help="Number of features to select")
    return p.parse_args()


from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# Add or remove metrics here; all must have signature fn(X, labels) -> float.
METRICS: dict[str, Callable] = {
    "silhouette": silhouette_score,
    "davies_bouldin": davies_bouldin_score,
    "calinski_harabasz": calinski_harabasz_score,
}


def main() -> None:
    args = parse_args()

    # logging
    print(f"Output directory: {args.output_dir}") 
    print(f"Module name: {args.name}")
    print(f"pcas_tsv: {args.pcas_tsv}")
    print(f"rawdata_clusters_truth: {args.rawdata_clusters_truth}")

    pca_df = pl.read_csv(args.pcas_tsv, separator="\t")
    truth_df = pl.read_csv(args.rawdata_clusters_truth, separator="\t")

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) 

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
