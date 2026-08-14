import numpy as np
from scipy.sparse import csr_matrix

from geometry.geometry import (
    calculate_sampled_distances,
    exact_disconnected_pair_fraction,
    sample_cell_pairs,
    symmetrize_distance_graph,
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
