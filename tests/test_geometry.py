import h5py
import numpy as np
import pytest
import polars as pl
from scipy.sparse import csc_matrix, csr_matrix
from geometry.geometry import (
    calculate_sampled_distances,
    calculate_euclidean_distances,
    calculate_geodesic_distances,
    compare_distances,
    exact_disconnected_pair_fraction,
    read_gene_representation,
    sample_cell_pairs,
    align_representations,
    symmetrize_distance_graph,
    symmetrize_distance_graph,
)


def test_read_gene_representation_returns_cells_by_genes(tmp_path):
    cell_by_gene = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 4.0],
        ]
    )

    stored = csc_matrix(cell_by_gene.T)

    path = tmp_path / "normalized_selected.h5"

    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")

        matrix.create_dataset(
            "barcodes",
            data=np.asarray(
                ["cell_a", "cell_b", "cell_c"],
                dtype="S",
            ),
        )
        matrix.create_dataset(
            "genes",
            data=np.asarray(
                ["gene_1", "gene_2"],
                dtype="S",
            ),
        )
        matrix.create_dataset(
            "data",
            data=stored.data,
        )
        matrix.create_dataset(
            "indices",
            data=stored.indices,
        )
        matrix.create_dataset(
            "indptr",
            data=stored.indptr,
        )
        matrix.create_dataset(
            "shape",
            data=np.asarray(stored.shape),
        )

    cell_ids, gene_matrix = read_gene_representation(path)

    assert cell_ids == [
        "cell_a",
        "cell_b",
        "cell_c",
    ]

    assert isinstance(gene_matrix, csr_matrix)
    assert gene_matrix.shape == (3, 2)

    np.testing.assert_allclose(
        gene_matrix.toarray(),
        cell_by_gene,
    )

def test_align_representations_uses_graph_cell_order():
    gene_cell_ids = [
        "cell_b",
        "cell_c",
        "cell_a",
    ]

    gene_matrix = csr_matrix(
        np.array(
            [
                [2.0, 20.0],
                [3.0, 30.0],
                [1.0, 10.0],
            ]
        )
    )

    pca_df = pl.DataFrame(
        {
            "cell_id": [
                "cell_c",
                "cell_a",
                "cell_b",
            ],
            "PC1": [300.0, 100.0, 200.0],
            "PC2": [30.0, 10.0, 20.0],
        }
    )

    graph_cell_ids = [
        "cell_a",
        "cell_b",
        "cell_c",
    ]

    aligned_gene, aligned_pca, pca_columns = (
        align_representations(
            gene_cell_ids=gene_cell_ids,
            gene_matrix=gene_matrix,
            pca_df=pca_df,
            graph_cell_ids=graph_cell_ids,
        )
    )

    expected_gene = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    expected_pca = np.array(
        [
            [100.0, 10.0],
            [200.0, 20.0],
            [300.0, 30.0],
        ]
    )

    np.testing.assert_allclose(
        aligned_gene.toarray(),
        expected_gene,
    )

    np.testing.assert_allclose(
        aligned_pca,
        expected_pca,
    )

    assert pca_columns == ["PC1", "PC2"]

def test_align_representations_rejects_mismatched_cell_sets():
    gene_cell_ids = [
        "cell_a",
        "cell_b",
        "cell_x",
    ]

    gene_matrix = csr_matrix(
        np.eye(3)
    )

    pca_df = pl.DataFrame(
        {
            "cell_id": [
                "cell_a",
                "cell_b",
                "cell_c",
            ],
            "PC1": [1.0, 2.0, 3.0],
        }
    )

    graph_cell_ids = [
        "cell_a",
        "cell_b",
        "cell_c",
    ]

    with pytest.raises(
        ValueError,
        match="gene and graph inputs do not contain the same cell set",
    ):
        align_representations(
            gene_cell_ids=gene_cell_ids,
            gene_matrix=gene_matrix,
            pca_df=pca_df,
            graph_cell_ids=graph_cell_ids,
        )

def test_align_representations_rejects_duplicate_cell_ids():
    gene_cell_ids = [
        "cell_a",
        "cell_a",
        "cell_c",
    ]

    gene_matrix = csr_matrix(
        np.eye(3)
    )

    pca_df = pl.DataFrame(
        {
            "cell_id": [
                "cell_a",
                "cell_b",
                "cell_c",
            ],
            "PC1": [1.0, 2.0, 3.0],
        }
    )

    graph_cell_ids = [
        "cell_a",
        "cell_b",
        "cell_c",
    ]

    with pytest.raises(
        ValueError,
        match="gene input contains duplicate cell IDs",
    ):
        align_representations(
            gene_cell_ids=gene_cell_ids,
            gene_matrix=gene_matrix,
            pca_df=pca_df,
            graph_cell_ids=graph_cell_ids,
        )

def test_symmetrize_distance_graph_keeps_single_direction_edges():
    graph = csr_matrix(
        np.array(
            [
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0],
            ]
        )
    )

    symmetric = symmetrize_distance_graph(graph).toarray()

    expected = np.array(
        [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 3.0],
            [0.0, 3.0, 0.0],
        ]
    )

    np.testing.assert_allclose(symmetric, expected)


def test_symmetrize_distance_graph_uses_smaller_bidirectional_distance():
    graph = csr_matrix(
        np.array(
            [
                [0.0, 5.0],
                [2.0, 0.0],
            ]
        )
    )

    symmetric = symmetrize_distance_graph(graph).toarray()

    expected = np.array(
        [
            [0.0, 2.0],
            [2.0, 0.0],
        ]
    )

    np.testing.assert_allclose(symmetric, expected)


def test_disconnected_pair_fraction_connected_graph():
    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )

    n_components, fraction = exact_disconnected_pair_fraction(graph)

    assert n_components == 1
    assert fraction == 0.0


def test_disconnected_pair_fraction_two_components():
    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
    )

    n_components, fraction = exact_disconnected_pair_fraction(graph)

    assert n_components == 2
    assert fraction == 4 / 6
def test_sample_cell_pairs_is_deterministic():
    sources_1, targets_1 = sample_cell_pairs(
        n_cells=20,
        seed=42,
        max_sources=5,
        targets_per_source=3,
    )

    sources_2, targets_2 = sample_cell_pairs(
        n_cells=20,
        seed=42,
        max_sources=5,
        targets_per_source=3,
    )

    np.testing.assert_array_equal(sources_1, sources_2)
    np.testing.assert_array_equal(targets_1, targets_2)


def test_calculate_sampled_distances_known_line_graph():
    coordinates = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )

    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )

    source_indices = np.array([0, 0, 1])
    target_indices = np.array([1, 2, 2])

    euclidean, geodesic = calculate_sampled_distances(
        coordinates=coordinates,
        graph=graph,
        source_indices=source_indices,
        target_indices=target_indices,
    )

    expected = np.array([1.0, 2.0, 1.0])

    np.testing.assert_allclose(euclidean, expected)
    np.testing.assert_allclose(geodesic, expected)

def test_calculate_euclidean_distances_dense():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 8.0],
        ]
    )

    source_indices = np.array([0, 0, 1])
    target_indices = np.array([1, 2, 2])

    distances = calculate_euclidean_distances(
        coordinates=coordinates,
        source_indices=source_indices,
        target_indices=target_indices,
    )

    expected = np.array(
        [
            5.0,
            10.0,
            5.0,
        ]
    )

    np.testing.assert_allclose(
        distances,
        expected,
    )

def test_calculate_euclidean_distances_sparse():
    coordinates = csr_matrix(
        np.array(
            [
                [0.0, 0.0],
                [3.0, 4.0],
                [6.0, 8.0],
            ]
        )
    )

    source_indices = np.array([0, 0, 1])
    target_indices = np.array([1, 2, 2])

    distances = calculate_euclidean_distances(
        coordinates=coordinates,
        source_indices=source_indices,
        target_indices=target_indices,
    )

    expected = np.array(
        [
            5.0,
            10.0,
            5.0,
        ]
    )

    np.testing.assert_allclose(
        distances,
        expected,
    )

def test_calculate_geodesic_distances_known_line_graph():
    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )

    source_indices = np.array([0, 0, 1])
    target_indices = np.array([1, 2, 2])

    distances = calculate_geodesic_distances(
        graph=graph,
        source_indices=source_indices,
        target_indices=target_indices,
    )

    expected = np.array(
        [
            1.0,
            2.0,
            1.0,
        ]
    )

    np.testing.assert_allclose(
        distances,
        expected,
    )

def test_compare_distances_identical_geometry():
    reference = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    comparison = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    metrics = compare_distances(
        reference_distances=reference,
        comparison_distances=comparison,
    )

    assert metrics["n_sampled_pairs"] == 4
    assert metrics["n_finite_sampled_pairs"] == 4
    assert metrics["nonfinite_pair_fraction"] == 0.0

    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["spearman"] == pytest.approx(1.0)

    assert metrics[
        "median_comparison_to_reference_ratio"
    ] == pytest.approx(1.0)


def test_compare_distances_detects_distance_scaling():
    reference = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    comparison = np.array(
        [2.0, 4.0, 6.0, 8.0]
    )

    metrics = compare_distances(
        reference_distances=reference,
        comparison_distances=comparison,
    )

    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["spearman"] == pytest.approx(1.0)

    assert metrics[
        "median_comparison_to_reference_ratio"
    ] == pytest.approx(2.0)


def test_compare_distances_excludes_nonfinite_pairs():
    reference = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    comparison = np.array(
        [1.0, np.inf, 3.0, np.inf]
    )

    metrics = compare_distances(
        reference_distances=reference,
        comparison_distances=comparison,
    )

    assert metrics["n_sampled_pairs"] == 4
    assert metrics["n_finite_sampled_pairs"] == 2

    assert metrics[
        "nonfinite_pair_fraction"
    ] == pytest.approx(0.5)


def test_compare_distances_rejects_mismatched_lengths():
    reference = np.array(
        [1.0, 2.0, 3.0]
    )

    comparison = np.array(
        [1.0, 2.0]
    )

    with pytest.raises(
        ValueError,
        match="Distance arrays must have the same length",
    ):
        compare_distances(
            reference_distances=reference,
            comparison_distances=comparison,
        )
