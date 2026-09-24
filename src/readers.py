"""Input readers for OmniBenchmark metric modules."""

from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


def read_gene_representation(
    path: Path,
) -> tuple[list[str], csr_matrix]:
    """Read the normalized gene-selected matrix as cells by genes.

    The OmniBenchmark FEAT artifact stores the sparse matrix as
    genes by cells in CSC format. This function transposes it to
    cells by genes in CSR format for pairwise cell-distance calculations.
    """
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]

        cell_ids = matrix["barcodes"].asstr()[:].tolist()
        n_genes = len(matrix["genes"])

        data = np.asarray(
            matrix["data"][:],
            dtype=float,
        )
        indices = np.asarray(
            matrix["indices"][:],
            dtype=int,
        )
        indptr = np.asarray(
            matrix["indptr"][:],
            dtype=int,
        )
        stored_shape = tuple(
            int(value)
            for value in matrix["shape"][:]
        )

    n_cells = len(cell_ids)
    expected_shape = (n_genes, n_cells)

    if stored_shape != expected_shape:
        raise ValueError(
            "Normalized selected matrix has an unexpected shape: "
            f"stored={stored_shape}, expected={expected_shape}."
        )

    stored_matrix = csc_matrix(
        (data, indices, indptr),
        shape=stored_shape,
    )

    gene_matrix = stored_matrix.T.tocsr()

    return cell_ids, gene_matrix


def read_neighbor_graph(
    path: Path,
) -> tuple[list[str], csr_matrix]:
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