#!/usr/bin/env python3

"""Pairwise representation geometry metrics for OmniBenchmark.

This module compares pairwise distances across the gene-space
representation, embedding space, and graph-geodesic space.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import networkx as nx
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent / "../src"))
from common import cli  # noqa: E402
from readers import read_gene_representation, read_neighbor_graph  # noqa: E402


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


def align_representations(
    gene_cell_ids: list[str],
    gene_matrix: csr_matrix,
    embedding_df: pl.DataFrame,
    graph_cell_ids: list[str],
) -> tuple[csr_matrix, np.ndarray, list[str]]:
    """Align gene and embedding representations to the graph cell order."""
    if gene_matrix.shape[0] != len(gene_cell_ids):
        raise ValueError(
            "Gene matrix row count does not match the number of gene-space cell IDs."
        )

    if "cell_id" not in embedding_df.columns:
        raise ValueError(
            "Embedding input does not contain a 'cell_id' column."
        )

    embedding_cell_ids = embedding_df["cell_id"].to_list()

    id_groups = {
        "gene": gene_cell_ids,
        "embedding": embedding_cell_ids,
        "graph": graph_cell_ids,
    }

    for name, cell_ids in id_groups.items():
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError(
                f"{name} input contains duplicate cell IDs."
            )

    graph_id_set = set(graph_cell_ids)

    for name, cell_ids in (
        ("gene", gene_cell_ids),
        ("embedding", embedding_cell_ids),
    ):
        cell_id_set = set(cell_ids)

        missing = graph_id_set - cell_id_set
        extra = cell_id_set - graph_id_set

        if missing or extra:
            raise ValueError(
                f"{name} and graph inputs do not contain the same cell set: "
                f"missing={len(missing)}, extra={len(extra)}."
            )

    gene_index = {
        cell_id: index
        for index, cell_id in enumerate(gene_cell_ids)
    }

    gene_order = np.asarray(
        [
            gene_index[cell_id]
            for cell_id in graph_cell_ids
        ],
        dtype=int,
    )

    aligned_gene_matrix = gene_matrix[gene_order]

    embedding_columns = [
        column
        for column in embedding_df.columns
        if column != "cell_id"
    ]

    if not embedding_columns:
        raise ValueError(
            "No embedding coordinate columns were found."
        )

    embedding_coordinates = embedding_df.select(
        embedding_columns
    ).to_numpy()

    embedding_index = {
        cell_id: index
        for index, cell_id in enumerate(embedding_cell_ids)
    }

    embedding_order = np.asarray(
        [
            embedding_index[cell_id]
            for cell_id in graph_cell_ids
        ],
        dtype=int,
    )

    aligned_embedding_coordinates = embedding_coordinates[embedding_order]

    if not np.all(np.isfinite(aligned_embedding_coordinates)):
        raise ValueError(
            "Embedding coordinates contain NaN or infinite values."
        )

    return (
        aligned_gene_matrix,
        aligned_embedding_coordinates,
        embedding_columns,
    )

def count_undirected_graph_nnz(graph: csr_matrix) -> int:
    """Count stored entries in the effective undirected graph."""
    graph = graph.tocoo()

    undirected_edges = {
        (min(int(row), int(col)), max(int(row), int(col)))
        for row, col in zip(graph.row, graph.col)
        if row != col
    }

    return 2 * len(undirected_edges)

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

def calculate_euclidean_distances(
    coordinates,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    """Calculate Euclidean distances for sampled cell pairs."""
    if len(source_indices) != len(target_indices):
        raise ValueError(
            "Source and target index arrays must have the same length."
        )

    if len(source_indices) == 0:
        return np.array([], dtype=float)

    if issparse(coordinates):
        differences = (
            coordinates[source_indices]
            - coordinates[target_indices]
        )

        squared_distances = np.asarray(
            differences.multiply(differences).sum(axis=1)
        ).ravel()

        return np.sqrt(squared_distances)

    differences = (
        coordinates[source_indices]
        - coordinates[target_indices]
    )

    return np.linalg.norm(
        differences,
        axis=1,
    )

def calculate_geodesic_distances(
    graph: csr_matrix,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    """Calculate graph-geodesic distances for sampled cell pairs."""
    if len(source_indices) != len(target_indices):
        raise ValueError(
            "Source and target index arrays must have the same length."
        )

    if len(source_indices) == 0:
        return np.array([], dtype=float)

    nx_graph = nx.from_scipy_sparse_array(
        graph,
        create_using=nx.Graph,
        edge_attribute="weight",
    )

    distances = np.full(
        len(source_indices),
        np.inf,
        dtype=float,
    )

    unique_sources = np.unique(source_indices)

    for source in unique_sources:
        source = int(source)

        shortest_paths = nx.single_source_dijkstra_path_length(
            nx_graph,
            source,
            weight="weight",
        )

        pair_positions = np.flatnonzero(
            source_indices == source
        )

        for position in pair_positions:
            target = int(target_indices[position])

            distances[position] = shortest_paths.get(
                target,
                np.inf,
            )

    return distances

def calculate_sampled_distances(
    coordinates: np.ndarray,
    graph: csr_matrix,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Euclidean and graph-geodesic distances for sampled pairs."""
    euclidean_distances = calculate_euclidean_distances(
        coordinates=coordinates,
        source_indices=source_indices,
        target_indices=target_indices,
    )

    geodesic_distances = calculate_geodesic_distances(
        graph=graph,
        source_indices=source_indices,
        target_indices=target_indices,
    )

    return euclidean_distances, geodesic_distances
def build_sampled_pairs_table(
    cell_ids: list[str],
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    gene_distances: np.ndarray,
    embedding_distances: np.ndarray,
    geodesic_distances: np.ndarray,
) -> pl.DataFrame:
    """Build a pair-level table for the sampled cell pairs."""
    n_pairs = len(source_indices)

    arrays = {
        "target_indices": target_indices,
        "gene_distances": gene_distances,
        "embedding_distances": embedding_distances,
        "geodesic_distances": geodesic_distances,
    }

    for name, values in arrays.items():
        if len(values) != n_pairs:
            raise ValueError(
                f"{name} must have the same length as source_indices."
            )

    source_cell_ids = [
        cell_ids[int(index)]
        for index in source_indices
    ]

    target_cell_ids = [
        cell_ids[int(index)]
        for index in target_indices
    ]

    return pl.DataFrame(
        {
            "cell_id_1": source_cell_ids,
            "cell_id_2": target_cell_ids,
            "gene_euclidean_distance": gene_distances,
            "embedding_euclidean_distance": embedding_distances,
            "geodesic_distance": geodesic_distances,
            "geodesic_is_finite": np.isfinite(
                geodesic_distances
            ),
        }
    )
def compare_distances(
    reference_distances: np.ndarray,
    comparison_distances: np.ndarray,
) -> dict:
    """Compare pairwise distances between two representations."""
    if len(reference_distances) != len(comparison_distances):
        raise ValueError(
            "Distance arrays must have the same length."
        )

    finite_mask = (
        np.isfinite(reference_distances)
        & np.isfinite(comparison_distances)
    )

    finite_reference = reference_distances[finite_mask]
    finite_comparison = comparison_distances[finite_mask]

    positive_reference_mask = finite_reference > 0

    ratio_reference = finite_reference[
        positive_reference_mask
    ]
    ratio_comparison = finite_comparison[
        positive_reference_mask
    ]

    if len(finite_reference) >= 2:
        pearson_result = pearsonr(
            finite_reference,
            finite_comparison,
        )

        spearman_result = spearmanr(
            finite_reference,
            finite_comparison,
        )

        pearson_correlation = float(
            pearson_result.statistic
        )

        spearman_correlation = float(
            spearman_result.statistic
        )
    else:
        pearson_correlation = None
        spearman_correlation = None

    if len(ratio_reference) > 0:
        distance_ratios = (
            ratio_comparison / ratio_reference
        )

        median_ratio = float(
            np.median(distance_ratios)
        )
    else:
        median_ratio = None

    return {
        "n_sampled_pairs": int(
            len(reference_distances)
        ),
        "n_finite_sampled_pairs": int(
            np.sum(finite_mask)
        ),
        "nonfinite_pair_fraction": (
            float(1.0 - np.mean(finite_mask))
            if len(finite_mask) > 0
            else None
        ),
        "pearson": pearson_correlation,
        "spearman": spearman_correlation,
        "median_comparison_to_reference_ratio": (
            median_ratio
        ),
    }

def calculate_geometry_metrics(
    euclidean_distances: np.ndarray,
    geodesic_distances: np.ndarray,
) -> dict:
    """Calculate Euclidean-geodesic agreement metrics."""
    comparison = compare_distances(
        reference_distances=euclidean_distances,
        comparison_distances=geodesic_distances,
    )

    return {
        "n_sampled_pairs": comparison[
            "n_sampled_pairs"
        ],
        "n_finite_sampled_pairs": comparison[
            "n_finite_sampled_pairs"
        ],
        "sampled_disconnected_pair_fraction": comparison[
            "nonfinite_pair_fraction"
        ],
        "pearson_euclidean_geodesic": comparison[
            "pearson"
        ],
        "spearman_euclidean_geodesic": comparison[
            "spearman"
        ],
        "median_geodesic_to_euclidean_ratio": comparison[
            "median_comparison_to_reference_ratio"
        ],
    }


def main() -> None:
    args = parse_args()

    log(f"normalized_selected_h5={args.normalized_selected_h5}")
    log(f"embedding_tsv={args.embedding_tsv}")
    log(f"neighbors_h5={args.neighbors_h5}")

    gene_cell_ids, gene_matrix = read_gene_representation(
        args.normalized_selected_h5
    )

    embedding_df = pl.read_csv(
        args.embedding_tsv,
        separator="\t",
    )

    graph_cell_ids, directed_graph = read_neighbor_graph(
        args.neighbors_h5
    )

    n_gene_cells = len(gene_cell_ids)
    n_embedding_cells = len(embedding_df)
    n_graph_cells = len(graph_cell_ids)

    log(
        "cell counts before alignment: "
        f"gene={n_gene_cells}, "
        f"embedding={n_embedding_cells}, "
        f"graph={n_graph_cells}"
    )

    if not np.all(np.isfinite(gene_matrix.data)):
        raise ValueError(
            "Gene representation contains NaN or infinite values."
        )

    (
        aligned_gene_matrix,
        aligned_embedding_coordinates,
        embedding_columns,
    ) = align_representations(
        gene_cell_ids=gene_cell_ids,
        gene_matrix=gene_matrix,
        embedding_df=embedding_df,
        graph_cell_ids=graph_cell_ids,
    )

    n_aligned_cells = len(graph_cell_ids)

    log(
        "aligned representations: "
        f"cells={n_aligned_cells}, "
        f"genes={aligned_gene_matrix.shape[1]}, "
        f"embedding_dims={len(embedding_columns)}"
    )

    symmetrized_graph_nnz = count_undirected_graph_nnz(
        directed_graph
    )

    log(
        "graph edges: "
        f"directed_nnz={directed_graph.nnz}, "
        f"symmetrized_nnz={symmetrized_graph_nnz}"
    )

    n_components, disconnected_fraction = (
        exact_disconnected_pair_fraction(
            directed_graph
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

    gene_distances = calculate_euclidean_distances(
        coordinates=aligned_gene_matrix,
        source_indices=source_indices,
        target_indices=target_indices,
    )

    embedding_distances = calculate_euclidean_distances(
        coordinates=aligned_embedding_coordinates,
        source_indices=source_indices,
        target_indices=target_indices,
    )

    geodesic_distances = calculate_geodesic_distances(
        graph=directed_graph,
        source_indices=source_indices,
        target_indices=target_indices,
    )

    gene_embedding_metrics = compare_distances(
        reference_distances=gene_distances,
        comparison_distances=embedding_distances,
    )

    gene_geodesic_metrics = compare_distances(
        reference_distances=gene_distances,
        comparison_distances=geodesic_distances,
    )

    embedding_geodesic_metrics = compare_distances(
        reference_distances=embedding_distances,
        comparison_distances=geodesic_distances,
    )
    sampled_pairs_table = build_sampled_pairs_table(
        cell_ids=graph_cell_ids,
        source_indices=source_indices,
        target_indices=target_indices,
        gene_distances=gene_distances,
        embedding_distances=embedding_distances,
        geodesic_distances=geodesic_distances,
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
    pairs_output_path = (
        output_dir
        / f"{args.name}_geometry_pair_summary.tsv"
    )
    sampled_pairs_table.write_csv(
        pairs_output_path,
        separator="\t",
    )

    log(f"wrote sampled pairs to {pairs_output_path}")
    result = {
        "n_gene_cells": n_gene_cells,
        "n_embedding_cells": n_embedding_cells,
        "n_graph_cells": n_graph_cells,
        "n_aligned_cells": n_aligned_cells,
        "n_selected_genes": int(
            aligned_gene_matrix.shape[1]
        ),
        "n_embedding_dimensions": len(embedding_columns),
        "directed_graph_nnz": int(
            directed_graph.nnz
        ),
        "symmetrized_graph_nnz": int(
            symmetrized_graph_nnz
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
        "metrics": {
            "gene_embedding": {
                "reference_space": "gene",
                "comparison_space": "embedding",
                **gene_embedding_metrics,
            },
            "gene_geodesic": {
                "reference_space": "gene",
                "comparison_space": "geodesic",
                **gene_geodesic_metrics,
            },
            "embedding_geodesic": {
                "reference_space": "embedding",
                "comparison_space": "geodesic",
                **embedding_geodesic_metrics,
            },
        },
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

    log(f"wrote metrics to {output_path}")

if __name__ == "__main__":
    main()
