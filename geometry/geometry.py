#!/usr/bin/env python3

"""Geometry metrics for OmniBenchmark.

This module compares Euclidean distances in a PCA representation with
shortest-path distances on the corresponding k-nearest-neighbor graph.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent / "../src"))
from common import cli  # noqa: E402


RANDOM_SEED = 42
MAX_SOURCE_CELLS = 200
TARGETS_PER_SOURCE = 50


def log(message: str) -> None:
    """Write a diagnostic message to stderr."""
    print(f"[geometry.py] {message}", file=sys.stderr, flush=True)


def parse_args():
    """Parse OmniBenchmark base arguments and GEOM-M stage inputs."""
    parser = argparse.ArgumentParser(description="GEOM-M module")
    cli.add_base_args(parser)
    cli.add_stage_args(parser, "GEOM-M")
    return parser.parse_args()


def read_neighbor_graph(path: Path) -> tuple[list[str], csr_matrix]:
    """Read the distance-based CSR neighbor graph from HDF5."""
    with h5py.File(path, "r") as handle:
        cell_ids = handle["cell_ids"].asstr()[:].tolist()
        data = np.asarray(handle["data"][:], dtype=float)
        indices = np.asarray(handle["indices"][:], dtype=int)
        indptr = np.asarray(handle["indptr"][:], dtype=int)

    n_cells = len(cell_ids)

    graph = csr_matrix(
        (data, indices, indptr),
        shape=(n_cells, n_cells),
    )

    return cell_ids, graph


def symmetrize_distance_graph(graph: csr_matrix) -> csr_matrix:
    """Convert a directed distance graph into an undirected distance graph.

    A one-directional edge is retained. If both directions are present,
    the smaller distance is used.
    """
    graph = graph.tocoo()

    edge_weights: dict[tuple[int, int], float] = {}

    for row, col, weight in zip(graph.row, graph.col, graph.data):
        if row == col:
            continue

        i = int(min(row, col))
        j = int(max(row, col))
        key = (i, j)

        weight = float(weight)

        if key not in edge_weights or weight < edge_weights[key]:
            edge_weights[key] = weight

    rows = []
    cols = []
    data = []

    for (i, j), weight in edge_weights.items():
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([weight, weight])

    symmetric_graph = coo_matrix(
        (data, (rows, cols)),
        shape=graph.shape,
        dtype=float,
    ).tocsr()

    symmetric_graph.eliminate_zeros()

    return symmetric_graph


def exact_disconnected_pair_fraction(graph: csr_matrix) -> tuple[int, float]:
    """Calculate connected components and exact disconnected-pair fraction."""
    n_components, labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )

    n_cells = graph.shape[0]

    if n_cells < 2:
        return n_components, 0.0

    component_sizes = np.bincount(labels)

    total_pairs = n_cells * (n_cells - 1) // 2

    connected_pairs = int(
        sum(
            size * (size - 1) // 2
            for size in component_sizes
        )
    )

    disconnected_pairs = total_pairs - connected_pairs

    disconnected_fraction = disconnected_pairs / total_pairs

    return n_components, float(disconnected_fraction)


def sample_cell_pairs(
    n_cells: int,
    seed: int = RANDOM_SEED,
    max_sources: int = MAX_SOURCE_CELLS,
    targets_per_source: int = TARGETS_PER_SOURCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample reproducible source-target cell pairs.

    The number of source cells is capped to avoid computing an all-pairs
    shortest-path matrix on large datasets.
    """
    if n_cells < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    rng = np.random.default_rng(seed)

    n_sources = min(max_sources, n_cells)

    source_cells = rng.choice(
        n_cells,
        size=n_sources,
        replace=False,
    )

    source_indices = []
    target_indices = []

    for source in source_cells:
        possible_targets = np.delete(np.arange(n_cells), source)

        n_targets = min(
            targets_per_source,
            len(possible_targets),
        )

        targets = rng.choice(
            possible_targets,
            size=n_targets,
            replace=False,
        )

        source_indices.extend([source] * n_targets)
        target_indices.extend(targets.tolist())

    return (
        np.asarray(source_indices, dtype=int),
        np.asarray(target_indices, dtype=int),
    )


def calculate_sampled_distances(
    coordinates: np.ndarray,
    graph: csr_matrix,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Euclidean and graph-geodesic distances for sampled pairs."""
    if len(source_indices) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    euclidean_distances = np.linalg.norm(
        coordinates[source_indices] - coordinates[target_indices],
        axis=1,
    )

    unique_sources, source_inverse = np.unique(
        source_indices,
        return_inverse=True,
    )

    shortest_paths = dijkstra(
        graph,
        directed=False,
        indices=unique_sources,
        return_predecessors=False,
    )

    geodesic_distances = shortest_paths[
        source_inverse,
        target_indices,
    ]

    return euclidean_distances, geodesic_distances


def calculate_geometry_metrics(
    euclidean_distances: np.ndarray,
    geodesic_distances: np.ndarray,
) -> dict:
    """Calculate Euclidean-geodesic agreement metrics."""
    finite_mask = (
        np.isfinite(euclidean_distances)
        & np.isfinite(geodesic_distances)
    )

    finite_euclidean = euclidean_distances[finite_mask]
    finite_geodesic = geodesic_distances[finite_mask]

    positive_mask = finite_euclidean > 0

    ratio_euclidean = finite_euclidean[positive_mask]
    ratio_geodesic = finite_geodesic[positive_mask]

    if len(finite_euclidean) >= 2:
        pearson_result = pearsonr(
            finite_euclidean,
            finite_geodesic,
        )
        spearman_result = spearmanr(
            finite_euclidean,
            finite_geodesic,
        )

        pearson_correlation = float(pearson_result.statistic)
        spearman_correlation = float(spearman_result.statistic)
    else:
        pearson_correlation = None
        spearman_correlation = None

    if len(ratio_euclidean) > 0:
        distance_ratios = ratio_geodesic / ratio_euclidean

        median_ratio = float(np.median(distance_ratios))
    else:
        median_ratio = None

    return {
        "n_sampled_pairs": int(len(euclidean_distances)),
        "n_finite_sampled_pairs": int(np.sum(finite_mask)),
        "sampled_disconnected_pair_fraction": float(
            1.0 - np.mean(finite_mask)
        ) if len(finite_mask) > 0 else None,
        "pearson_euclidean_geodesic": pearson_correlation,
        "spearman_euclidean_geodesic": spearman_correlation,
        "median_geodesic_to_euclidean_ratio": median_ratio,
    }


def main() -> None:
    args = parse_args()

    log(f"pcas_tsv={args.pcas_tsv}")
    log(f"neighbors_h5={args.neighbors_h5}")

    pca_df = pl.read_csv(
        args.pcas_tsv,
        separator="\t",
    )

    graph_cell_ids, directed_graph = read_neighbor_graph(
        args.neighbors_h5
    )

    graph_ids_df = pl.DataFrame(
        {
            "cell_id": graph_cell_ids,
            "graph_index": np.arange(
                len(graph_cell_ids),
                dtype=int,
            ),
        }
    )

    aligned = (
        graph_ids_df
        .join(
            pca_df,
            on="cell_id",
            how="inner",
        )
        .sort("graph_index")
    )

    n_pca_cells = len(pca_df)
    n_graph_cells = len(graph_cell_ids)
    n_aligned_cells = len(aligned)

    log(
        "cell counts: "
        f"pca={n_pca_cells}, "
        f"graph={n_graph_cells}, "
        f"aligned={n_aligned_cells}"
    )

    if n_aligned_cells != n_graph_cells:
        raise ValueError(
            "PCA and graph inputs do not contain the same filtered cell set."
        )

    pca_columns = [
        column
        for column in pca_df.columns
        if column != "cell_id"
    ]

    if not pca_columns:
        raise ValueError(
            "No PCA coordinate columns were found."
        )

    coordinates = aligned.select(
        pca_columns
    ).to_numpy()

    if not np.all(np.isfinite(coordinates)):
        raise ValueError(
            "PCA coordinates contain NaN or infinite values."
        )

    symmetric_graph = symmetrize_distance_graph(
        directed_graph
    )

    log(
        "graph edges: "
        f"directed_nnz={directed_graph.nnz}, "
        f"symmetrized_nnz={symmetric_graph.nnz}"
    )

    n_components, disconnected_fraction = (
        exact_disconnected_pair_fraction(
            symmetric_graph
        )
    )

    log(
        "graph connectivity: "
        f"components={n_components}, "
        f"disconnected_pair_fraction={disconnected_fraction:.6f}"
    )

    source_indices, target_indices = sample_cell_pairs(
        n_cells=n_aligned_cells,
    )

    euclidean_distances, geodesic_distances = (
        calculate_sampled_distances(
            coordinates=coordinates,
            graph=symmetric_graph,
            source_indices=source_indices,
            target_indices=target_indices,
        )
    )

    geometry_metrics = calculate_geometry_metrics(
        euclidean_distances,
        geodesic_distances,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{args.name}_geometry_metrics.json"
    )

    result = {
        "n_pca_cells": n_pca_cells,
        "n_graph_cells": n_graph_cells,
        "n_aligned_cells": n_aligned_cells,
        "n_pca_dimensions": len(pca_columns),
        "directed_graph_nnz": int(directed_graph.nnz),
        "symmetrized_graph_nnz": int(
            symmetric_graph.nnz
        ),
        "n_connected_components": int(
            n_components
        ),
        "disconnected_pair_fraction": (
            disconnected_fraction
        ),
        "sampling": {
            "random_seed": RANDOM_SEED,
            "max_source_cells": MAX_SOURCE_CELLS,
            "targets_per_source": TARGETS_PER_SOURCE,
        },
        "metrics": geometry_metrics,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
        )

    log(f"wrote {output_path}")


if __name__ == "__main__":
    main()
