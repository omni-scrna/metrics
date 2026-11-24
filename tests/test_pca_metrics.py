"""PCA_M: pairing by the gather record's lineage, and the accuracy metrics, on synthetic PCA outputs."""

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


def _node(stage, module, params=None, **parameters):
    return {"node_id": f"{stage}-{module}-{params}", "stage": stage, "module": module,
            "params": params, "parameters": parameters}


def _member(files, *lineage):
    return {"node_id": lineage[-1]["node_id"], "files": files, "lineage": list(lineage)}


def _arm(root, norm, feat, pca, scores, load, cells, genes):
    """One PCA member of a gather record; its directory name means nothing to PCA_M."""
    d = root / f"{norm}-{pca}"
    _write(d, f"{pca}_embedding.tsv", cells, scores, "cell_id")
    _write(d, f"{pca}_loadings.tsv", genes, load, "gene_id")
    files = {"embedding_tsv": str(d / f"{pca}_embedding.tsv"), "loadings_tsv": str(d / f"{pca}_loadings.tsv")}
    feat_node = _node("FEAT", feat, "bbbbbbbb", number_selected=12)
    feat_node["node_id"] += f"-{norm}"  # one FEAT node per norm: one reference each
    return _member(files, _node("DATA", "d1", "aaaaaaaa", dataset_name="toy"), _node("NORM", norm),
                   feat_node, _node("PCA", pca, "cccccccc", n_components=6))


def test_pairs_by_lineage_and_scores_exact_and_rotated(tmp_path):
    rng = np.random.default_rng(0)
    cells = [f"c{i}" for i in range(60)]
    genes = [f"g{i}" for i in range(12)]
    members = []
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
            members.append(_arm(tmp_path, norm, "fe_x", pca, ss, ll, c, g))
    # a member whose own stage is not PCA must be ignored, whatever files it has
    iso = _arm(tmp_path, "nr_a", "fe_x", "dr_isomap", rng.normal(size=(60, 6)), rng.normal(size=(12, 6)), cells, genes)
    iso["lineage"][-1] = _node("ISOMAP", "dr_isomap", "dddddddd")
    members.append(iso)

    # integer-looking leading rows must not fix a column's type (NMF zeros)
    lo0 = members[1]["files"]["loadings_tsv"]  # a non-reference member: extra genes are aligned away
    lines = open(lo0).read().splitlines()
    head, first = lines[0], lines[1].split("\t")
    lines.insert(1, "\t".join(["g_zero"] + ["0"] * (len(first) - 1)))
    open(lo0, "w").write("\n".join(lines) + "\n")

    df = run(members, "pc_exact", ks=[2, 6])
    assert set(df["pca"]) == {"pc_exact", "pc_flip", "pc_rot"}
    # lineage columns from the record (fanin.member_lineage_cols): a node without
    # parameters is its bare module id, never an invented ".default"
    assert set(df["norm"]) == {"nr_a", "nr_b"}
    assert set(df["feat"]) == {"fe_x.bbbbbbbb"} and set(df["pca_params"]) == {"cccccccc"}
    assert set(df["dataset"]) == {"d1"} and set(df["data_dataset_name"]) == {"toy"}
    assert "pca_stage" not in df.columns
    assert df.columns[:5] == ["dataset", "norm", "feat", "pca", "pca_params"]
    assert df.filter(pl.col("is_reference"))["pca"].unique().to_list() == ["pc_exact"]

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
    rd = _member({"embedding_tsv": "/x/e.tsv", "loadings_tsv": "/x/l.tsv"},
                 _node("DATA", "d_pca", "aaaaaaaa"), _node("RDIMR", "rd_x", "bbbbbbbb"))
    df = run([rd], "pc_exact", ks=[2])
    assert df.height == 0 and "value" in df.columns


def _rot(k, theta=0.5):
    r = np.eye(k)
    c, s = np.cos(theta), np.sin(theta)
    r[:2, :2] = [[c, -s], [s, c]]
    return r


def test_frame_types_parameter_columns():
    from pca_metrics import frame
    base = {"is_reference": False, "metric": "m", "index": 1, "value": 0.5}
    df = frame([{"pca": "x", "pca_solver": "arpack", "pca_n": 50, **base},
                {"pca": "y", "pca_n": 30.5, **base}])
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


def test_procrustes_rel_err():
    from pca_metrics import procrustes_rel_err
    rng = np.random.default_rng(1)
    z = rng.normal(size=(200, 5))
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    assert procrustes_rel_err(z, z @ q.T) < 1e-12  # any rotation/reflection is free
    assert abs(procrustes_rel_err(z, 2 * z) - 1.0) < 1e-12  # scale is not: ||Z - 2Z|| / ||Z||
    # brute force: no random orthogonal Q beats the closed form
    zh = z + 0.3 * rng.normal(size=z.shape)
    best = procrustes_rel_err(z, zh)
    for _ in range(200):
        r, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        assert np.linalg.norm(z - zh @ r) / np.linalg.norm(z) >= best - 1e-12


def test_pc_sin_vw_weights_by_reference_variance():
    from pca_metrics import compare
    rng = np.random.default_rng(2)
    load = np.linalg.qr(rng.normal(size=(30, 3)))[0]
    emb = rng.normal(size=(500, 3)) * [10.0, 3.0, 1.0]
    rows = {(m, i): v for m, i, v in compare(emb, load, emb, load, [3])}
    assert rows[("pc_sin_vw", 3)] < 1e-12  # identical arm
    tilted = load.copy()
    tilted[:, 2] = np.linalg.qr(np.c_[load, rng.normal(size=30)])[0][:, 3]  # PC3 orthogonal
    rows = {(m, i): v for m, i, v in compare(emb, load, emb, tilted, [3])}
    var = emb.var(axis=0, ddof=1)
    assert abs(rows[("pc_sin_vw", 3)] - var[2] / var.sum()) < 1e-12  # only PC3's share counts
