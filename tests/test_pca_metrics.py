"""PCA_M: lineage pairing and the accuracy metrics, on synthetic PCA outputs."""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "pca"))
from pca_metrics import run  # noqa: E402


def _write(d, name, ids, x, idcol):
    d.mkdir(parents=True, exist_ok=True)
    cols = {idcol: ids} | {f"PC{i + 1}": x[:, i] for i in range(x.shape[1])}
    pl.DataFrame(cols).write_csv(d / name, separator="\t")


def _pca(x, k):
    xc = x - x.mean(axis=0)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    return u[:, :k] * s[:k], vt[:k].T


def _arm(root, norm, feat, pca, scores, load, cells, genes):
    d = root / "DATA.d1.aaaaaaaa" / f"NORM.{norm}.default" / f"FEAT.{feat}.bbbbbbbb" / f"PCA.{pca}.cccccccc"
    _write(d, f"{pca}_embedding.tsv", cells, scores, "cell_id")
    _write(d, f"{pca}_loadings.tsv", genes, load, "gene_id")
    return str(d / f"{pca}_embedding.tsv"), str(d / f"{pca}_loadings.tsv")


def test_pairs_by_lineage_and_scores_exact_and_rotated(tmp_path):
    rng = np.random.default_rng(0)
    cells = [f"c{i}" for i in range(60)]
    genes = [f"g{i}" for i in range(12)]
    emb, load = [], []
    for norm, scale in (("nr_a", 1.0), ("nr_b", 3.0)):  # two different input matrices
        x = rng.normal(size=(60, 12)) * np.linspace(3, 0.5, 12) * scale
        s, l = _pca(x, 6)
        for pca, (ss, ll) in {
            "pc_exact": (s, l),
            # sign flips + shuffled rows: identical solution
            "pc_flip": (-s[::-1], -l[::-1]),
            # rotate PCs 1-2 into each other: per-PC error, same 2-d subspace
            "pc_rot": (s @ _rot(6), l @ _rot(6)),
        }.items():
            c, g = (cells[::-1], genes[::-1]) if pca == "pc_flip" else (cells, genes)
            e, lo = _arm(tmp_path, norm, "fe_x", pca, ss, ll, c, g)
            emb.append(e)
            load.append(lo)
    # a non-PCA producer of embedding_tsv must be ignored
    iso = tmp_path / "DATA.d1.aaaaaaaa" / "ISOMAP.dr_isomap.dddddddd"
    _write(iso, "dr_isomap_embedding.tsv", cells, rng.normal(size=(60, 6)), "cell_id")
    emb.append(str(iso / "dr_isomap_embedding.tsv"))

    # integer-looking leading rows must not fix a column's type (NMF zeros)
    lo0 = load[1]  # a non-reference member: extra genes are aligned away
    lines = open(lo0).read().splitlines()
    head, first = lines[0], lines[1].split("\t")
    lines.insert(1, "\t".join(["g_zero"] + ["0"] * (len(first) - 1)))
    open(lo0, "w").write("\n".join(lines) + "\n")

    df = run(emb, load, "pc_exact", ks=[2, 6])
    assert set(df["pca"]) == {"pc_exact", "pc_flip", "pc_rot"}
    assert set(df["norm"]) == {"nr_a.default", "nr_b.default"}

    def val(pca, metric, idx):
        sel = df.filter((pl.col("pca") == pca) & (pl.col("metric") == metric) & (pl.col("index") == idx))
        return sel["value"].to_numpy()

    for pca in ("pc_exact", "pc_flip"):  # paired with its OWN norm's reference
        for metric in ("eig_rel_err", "pc_sin", "subspace_sin", "scores_sin"):
            assert np.all(df.filter(pl.col("pca") == pca).filter(pl.col("metric") == metric)["value"].to_numpy() < 1e-8)
    assert np.all(val("pc_rot", "pc_sin", 1) > 0.1)          # PC1 moved
    assert np.all(val("pc_rot", "subspace_sin", 2) < 1e-8)   # span(PC1, PC2) did not
    assert np.all(val("pc_rot", "pc_sin", 3) < 1e-8)         # untouched PC


def test_ipr_bounds():
    from pca_metrics import ipr
    p = 100
    one_gene = np.zeros((p, 1)); one_gene[7] = -3.0          # all weight on one gene
    uniform = np.full((p, 1), 0.2)                            # spread evenly
    assert np.isclose(ipr(one_gene)[0], 1.0)
    assert np.isclose(ipr(uniform)[0], 1.0 / p)
    rnd = np.random.default_rng(1).normal(size=(p, 2000))     # random directions
    assert abs(ipr(rnd).mean() - 3.0 / (p + 2)) < 2e-3


def test_total_variance_matches_dense(tmp_path):
    import h5py
    import scipy.sparse as sp
    from pca_metrics import total_variance
    rng = np.random.default_rng(2)
    x = sp.random(30, 200, density=0.2, random_state=3, format="csc") * 5  # genes x cells
    with h5py.File(tmp_path / "m.h5", "w") as f:
        g = f.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = x.data, x.indices, x.indptr
        g["shape"] = np.array(x.shape)
    dense = x.toarray()
    assert np.isclose(total_variance(str(tmp_path / "m.h5")), dense.var(axis=1, ddof=1).sum())


def test_group_without_pca_members_writes_empty_table(tmp_path):
    rng = np.random.default_rng(5)
    d = tmp_path / "DATA.d_pca.aaaaaaaa" / "RDIMR.rd_x.bbbbbbbb"
    _write(d, "rd_x_embedding.tsv", ["c1", "c2", "c3"], rng.normal(size=(3, 2)), "cell_id")
    _write(d, "rd_x_loadings.tsv", ["g1", "g2"], rng.normal(size=(2, 2)), "gene_id")
    df = run([str(d / "rd_x_embedding.tsv")], [str(d / "rd_x_loadings.tsv")], "pc_exact", ks=[2])
    assert df.height == 0 and "value" in df.columns


def _rot(k, theta=0.5):
    r = np.eye(k)
    c, s = np.cos(theta), np.sin(theta)
    r[:2, :2] = [[c, -s], [s, c]]
    return r


def test_from_files_reads_member_paths_by_output_id(tmp_path):
    import json

    from pca_metrics import cli

    record = {"version": 2, "members": [
        {"files": {"embedding_tsv": "/a/e.tsv", "loadings_tsv": "/a/l.tsv"}},
        {"files": {"embedding_tsv": "/b/e.tsv"}},  # e.g. an RDIMR embedding, no loadings
    ]}
    path = tmp_path / "lineage.json"
    path.write_text(json.dumps(record))
    assert cli.from_files(path, "embedding_tsv") == ["/a/e.tsv", "/b/e.tsv"]
    assert cli.from_files(path, "loadings_tsv") == ["/a/l.tsv"]


def test_member_parameters_become_columns(tmp_path):
    import json

    from pca_metrics import cli, frame

    record = {"version": 2, "members": [
        {"files": {"embedding_tsv": "/a/e.tsv"},
         "lineage": [{"stage": "DATA", "parameters": {"dataset_name": "be1"}},
                     {"stage": "PCA", "parameters": {"solver": "arpack", "n": 50, "ks": [1, 2]}}]},
        {"files": {"embedding_tsv": "/b/e.tsv"},
         "lineage": [{"stage": "DATA", "parameters": {"dataset_name": "be1"}},
                     {"stage": "PCA", "parameters": {"n": 30.5}}]},
    ]}
    path = tmp_path / "lineage.json"
    path.write_text(json.dumps(record))
    labels = cli.member_parameters(path)
    assert labels["/a/e.tsv"] == {"data_dataset_name": "be1", "pca_solver": "arpack",
                                  "pca_n": 50, "pca_ks": "[1, 2]"}
    base = {"is_reference": False, "metric": "m", "index": 1, "value": 0.5}
    df = frame([{"pca": "x", **labels["/a/e.tsv"], **base},
                {"pca": "y", **labels["/b/e.tsv"], **base}])
    assert df.columns[-4:] == ["is_reference", "metric", "index", "value"]
    assert df["pca_n"].to_list() == [50.0, 30.5]       # int + float -> float
    assert df["pca_solver"].to_list() == ["arpack", None]  # absent -> null


def test_results_row_bind_keeps_every_column(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent / "../pca"))
    from pca_results import concat
    (tmp_path / "a.tsv").write_text("pca\tpca_params\tvalue\nx\t12345678\t0.5\n")
    (tmp_path / "b.tsv").write_text("pca\tpca_params\tpca_solver\tvalue\ny\tabcdef12\tfull\t1e-13\n")
    df = concat([tmp_path / "a.tsv", tmp_path / "b.tsv"])
    assert df.columns == ["pca", "pca_params", "value", "pca_solver"]
    assert df["pca_params"].to_list() == ["12345678", "abcdef12"]
    assert df["pca_solver"].to_list() == [None, "full"]
    assert df["value"].to_list() == ["0.5", "1e-13"]  # text passes through untouched
