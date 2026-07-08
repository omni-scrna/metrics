#!/usr/bin/env python3
# Integration (batch-correction) metrics for omnibenchmark — scib-metrics backed.
#
# Implementation notes
# --------------------
# - Aligns the corrected embedding against ground-truth cell-type labels (by cell_id) for
#   n_cells/n_labels/n_dropped bookkeeping and the metrics themselves — unlike integration-r
#   (CellMixS), these metrics compare the embedding against cell-type labels, not per-cell
#   *batch* labels, so --rawdata_h5ad/--properties_info are accepted (required by the shared
#   INTG8-M schema) but unused here.
# - label_asw: scib_metrics.silhouette_label — Average Silhouette Width over cell-type labels.
# - clisi: scib_metrics.clisi_knn — cell-type LISI, needs a precomputed KNN graph
#   (scib_metrics.nearest_neighbors.pynndescent) rather than the raw embedding directly.
#   n_neighbors=90 matches scib-metrics' own Benchmarker default for LISI-family metrics.

import argparse
import sys
from pathlib import Path
import polars as pl
import json

sys.path.insert(0, str(Path(__file__).parent / "../src"))  # vendored `common` (src/common) + module-local writers
from common import cli  # noqa: E402

import scib_metrics
from scib_metrics.nearest_neighbors import pynndescent


def parse_args():
    # We own the parser; src/common/cli injects the shared contract (base args + the
    # `INTG8-M` stage I/O from common/schema). This module's method params are
    # hand-rolled below, so the whole CLI stays visible here.
    p = argparse.ArgumentParser(description="INTG8-M module (scib-metrics-backed)")
    cli.add_base_args(p)              # --output_dir, --name
    cli.add_stage_args(p, "INTG8-M")  # --corrected_tsv, --rawdata_clusters_truth, --rawdata_h5ad, --properties_info
    p.add_argument("--n_neighbors", type=int, default=90,
                   help="Neighborhood size (knn) for the graph clisi_knn is computed over")
    p.add_argument("--random_state", type=int, default=0,
                   help="Random state for the pynndescent approximate KNN search")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # logging
    print(f"Output directory: {args.output_dir}")
    print(f"Module name: {args.name}")
    print(f"corrected_tsv: {args.corrected_tsv}")
    print(f"rawdata_clusters_truth: {args.rawdata_clusters_truth}")

    corrected_df = pl.read_csv(args.corrected_tsv, separator="\t")
    truth_df = pl.read_csv(args.rawdata_clusters_truth, separator="\t")

    # Align the corrected embedding rows with truth labels by cell_id.
    merged = corrected_df.join(
        truth_df.select(["cell_id", "truths"]), on="cell_id", how="inner"
    )

    embedding = merged.drop(["cell_id", "truths"]).to_numpy()
    aligned_labels = merged["truths"].cast(str).to_numpy()
    n_cells = len(merged)
    n_labels = merged["truths"].n_unique()
    n_dropped = len(corrected_df) - n_cells

    if n_labels >= 2:
        label_asw = float(scib_metrics.silhouette_label(embedding, aligned_labels))

        n_neighbors = min(args.n_neighbors, n_cells - 1)
        neighbors = pynndescent(
            embedding, n_neighbors=n_neighbors, random_state=args.random_state
        )
        clisi = float(scib_metrics.clisi_knn(neighbors, aligned_labels))
    else:
        label_asw = float("nan")
        clisi = float("nan")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_dir / f"{args.name}_integration_metrics.json", "w") as fh:
        json.dump(
            {
                "n_cells": n_cells,
                "n_labels": n_labels,
                "n_dropped": n_dropped,
                "label_asw": label_asw,
                "clisi": clisi,
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
