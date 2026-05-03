#!/usr/bin/env python3
# Argument parsers for omnibenchmark metrics module.
#
# Reusable across entrypoints (embedding.py, and future clustering.py, graph.py, ...).
#
# Conventions:
# - All arguments are required. No defaults — callers (omnibenchmark configs)
#   must pass everything explicitly so runs are fully reproducible from the
#   invocation line.
# - Argument names mirror the upstream stage output IDs from benchmark_conda.yaml.

import argparse
from pathlib import Path


def parse_embedding_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OmniBenchmark embedding metrics")
    parser.add_argument(
        "--output_dir", required=True, type=Path, help="Output directory for results"
    )
    parser.add_argument("--name", required=True, help="Module name/identifier")
    parser.add_argument(
        "--pca",
        required=True,
        type=Path,
        help="PCA HDF5 file (embedding matrix + cell_ids)",
    )
    parser.add_argument(
        "--rawdata.clusters_truth",
        dest="clusters_truth",
        required=True,
        type=Path,
        help="TSV of ground truth cluster labels (columns: cell_id, truths)",
    )
    return parser.parse_args()
