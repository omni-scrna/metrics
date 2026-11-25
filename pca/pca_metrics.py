#!/usr/bin/env python3
"""PCA_M (gather): accuracy of every PCA arm against an exact reference.

ob gathers every producer of embedding_tsv / loadings_tsv under one FEAT node
and hands them over in one file (`--from-files`, lineage.json v2, ob api 0.8.0),
with that node's own matrix as `--normalized_selected_h5` (total variance). Each PCA member is
compared with the reference member (--reference_module, default pc_exact) under
the same FEAT node. Everything about a member (its files, its stage, module and
params, the FEAT node it hangs off) comes from the record, never from its path;
members whose own stage is not PCA are skipped.

Output: one long TSV, one row per (member, metric, index). Lineage columns as
src/fanin.py writes them (dataset, filt, norm, feat, pca, pca_params, then one
`<stage>_<parameter>` per lineage parameter, e.g. `feat_number_selected`), the
same as EMBED_M's, so the two tables join.
  eig_rel_err    per PC   |var_i - var_ref_i| / var_ref_i, var = score variance
  pc_sin         per PC   sin(angle) between loading vectors i (sign-free)
  subspace_sin   per k    largest principal-angle sine, loadings[:, :k]
  scores_sin     per k    same on the cell scores (cells aligned by id)
  ipr            per PC   inverse participation ratio of the member's OWN loading
                          vector, sum l^4 / (sum l^2)^2; 1/IPR = effective number
                          of genes. lambda_i * l_g^2 is gene g's share of
                          eigenvalue i, so IPR says how concentrated it is.
  ipr_floor      index 0  3 / n_genes, E[IPR] of a random unit vector
  var_explained  per PC   var_i / total variance of the FEAT matrix (sum of every
                          gene's variance, not of the kept PCs)

References
  IPR: Plerou V et al. (2002) Random matrix approach to cross correlations in
    financial data. Phys Rev E 65:066126. On scRNA-seq, sparsity alone can
    localise eigenvectors: Aparicio L et al. (2020) Patterns 1(3):100035.
IPR is unstable inside near-degenerate eigenspaces (where pc_sin rises), and
means different things under different preprocessing (rare population vs a
one-gene noise axis), so compare it within one input matrix only.
Sines from projection residuals: 1 - cos (or sqrt(1 - cos^2)) floors at ~1e-8.
NMF factors are neither orthogonal nor variance-ordered, so only the
subspace rows mean anything for them.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent / "../src"))  # vendored `common`, fanin
import fanin  # noqa: E402
from common import cli  # noqa: E402

COLUMNS = {"dataset": pl.Utf8, "filt": pl.Utf8, "norm": pl.Utf8, "feat": pl.Utf8, "pca": pl.Utf8,
           "pca_params": pl.Utf8, "is_reference": pl.Boolean, "metric": pl.Utf8,
           "index": pl.Int64, "value": pl.Float64}


def read_matrix(path):
    """First column = row ids (cells or genes), the rest = factors."""
    # no type inference: a factor column whose first rows are integers (NMF
    # zeros) would be typed Int64 and fail on the first decimal
    df = pl.read_csv(path, separator="\t", infer_schema=False)
    return df[:, 0].to_list(), df[:, 1:].cast(pl.Float64).to_numpy()


def align(ids, x, ref_ids):
    pos = {r: i for i, r in enumerate(ids)}
    missing = [r for r in ref_ids if r not in pos]
    if missing:
        raise ValueError(f"{len(missing)} ids missing from a member, e.g. {missing[:3]}")
    return x[[pos[r] for r in ref_ids]]


def max_sin(a, b):
    """Sine of the largest principal angle between span(a) and span(b), from
    the projection residual ||(I - Qa Qa^T) Qb||_2. sqrt(1 - cos^2) would floor
    at ~1.5e-8 (cancellation near cos = 1) and hide real solver differences."""
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    return float(np.linalg.norm(qb - qa @ (qa.T @ qb), ord=2))


def ipr(load):
    """Per-column inverse participation ratio; scale-free (no unit norm needed)."""
    return (load ** 4).sum(axis=0) / (load ** 2).sum(axis=0) ** 2


def compare(ref_emb, ref_load, emb, load, ks):
    """Metric rows for one member against its reference (already aligned)."""
    k = min(ref_emb.shape[1], emb.shape[1])
    rows = [("ipr_floor", 0, 3.0 / load.shape[0])]
    rows += [("ipr", i + 1, float(v)) for i, v in enumerate(ipr(load))]
    var_ref = ref_emb[:, :k].var(axis=0, ddof=1)
    var = emb[:, :k].var(axis=0, ddof=1)
    for i in range(k):
        rows.append(("eig_rel_err", i + 1, abs(var[i] - var_ref[i]) / var_ref[i]))
        rows.append(("pc_sin", i + 1, max_sin(ref_load[:, [i]], load[:, [i]])))
    for kk in ks:
        if kk <= k:
            rows.append(("subspace_sin", kk, max_sin(ref_load[:, :kk], load[:, :kk])))
            rows.append(("scores_sin", kk, max_sin(ref_emb[:, :kk], emb[:, :kk])))
    return rows


def read_tenx(h5):
    """TENx genes x cells CSC matrix as scipy CSR (one row per gene)."""
    with h5py.File(h5, "r") as f:
        g = f["matrix"]
        shape = tuple(int(v) for v in g["shape"][:])
        x = sp.csc_matrix((g["data"][:].astype(np.float64), g["indices"][:], g["indptr"][:]), shape=shape)
    return x.tocsr()


def total_variance(x):
    """Sum of per-gene variances (ddof=1): the trace of the covariance PCA
    decomposes, without densifying. `x` is genes x cells CSR, or a TENx path."""
    if isinstance(x, str):
        x = read_tenx(x)
    n = x.shape[1]
    s1 = np.asarray(x.sum(axis=1)).ravel()
    s2 = np.asarray(x.multiply(x).sum(axis=1)).ravel()
    return float(((s2 - s1 * s1 / n) / (n - 1)).sum())


def anchor_key(member, anchor):
    """The node the member hangs off at `anchor` (one input matrix), from its lineage."""
    e = fanin.lineage_entry(member, anchor)
    return e["node_id"] if e else None


def run(members, reference_module, ks, stage="PCA", anchor="FEAT", matrix=None):
    """`members`: gather-record members carrying embedding_tsv and loadings_tsv
    (fanin.manifest_members). Each `stage` member is compared with the
    `reference_module` member under the same `anchor` node."""
    members = sorted((m for m in members if m["lineage"][-1]["stage"] == stage),
                     key=lambda m: m["files"]["embedding_tsv"])
    refs = {}
    for m in members:
        if m["lineage"][-1]["module"] == reference_module:
            key = anchor_key(m, anchor)
            if key in refs:
                raise ValueError(f"two {reference_module} members under {key}")
            refs[key] = m
    if not members:  # group without PCA (e.g. only RDIMR embeddings): nothing to compare
        print(f"no {stage} members in this group; writing an empty table")
        return pl.DataFrame(schema=COLUMNS)
    if not refs:
        raise ValueError(f"no {reference_module} member among {len(members)} {stage} members")

    # One gather group is one FEAT node, so one matrix serves every member.
    total = total_variance(matrix) if matrix else None

    rows = []
    for m in members:
        e, lo = m["files"]["embedding_tsv"], m["files"]["loadings_tsv"]
        key = anchor_key(m, anchor)
        if key not in refs:
            print(f"WARN no {reference_module} under {key}; skipping {e}")
            continue
        r = refs[key]
        cells, ref_emb = read_matrix(r["files"]["embedding_tsv"])
        genes, ref_load = read_matrix(r["files"]["loadings_tsv"])
        emb = align(*read_matrix(e), cells)
        load = align(*read_matrix(lo), genes)
        lin = fanin.member_lineage_cols(m, "pca")
        lin.pop("pca_stage")  # always PCA here
        extra = []
        if total is not None:
            extra = [("var_explained", i + 1, float(v / total))
                     for i, v in enumerate(emb.var(axis=0, ddof=1))]
        for metric, idx, value in compare(ref_emb, ref_load, emb, load, ks) + extra:
            rows.append({
                **lin,
                "is_reference": m is r,
                "metric": metric,
                "index": idx,
                "value": value,
            })
    return frame(rows)


def frame(rows):
    """Rows -> a typed table (fanin.frame), ending in is_reference, metric, index, value."""
    return fanin.frame(rows, ["is_reference", "metric", "index", "value"], COLUMNS)


def parse_args():
    p = argparse.ArgumentParser(description="PCA_M gather: PCA arms vs an exact reference")
    cli.add_base_args(p)  # --output_dir, --name, --output id=PATH
    p.add_argument("--from-files", dest="from_files", required=True,
                   help="ob gather record (lineage.json v2): every member's embedding_tsv/loadings_tsv")
    p.add_argument("--normalized_selected_h5", default=None,
                   help="the FEAT node's matrix (variance explained; optional)")
    p.add_argument("--reference_module", default="pc_exact",
                   help="PCA module id taken as the exact reference")
    p.add_argument("--ks", default="5,10,20,30,50",
                   help="leading-subspace sizes for the principal-angle metrics")
    return p.parse_args()


def main():
    args = parse_args()
    ks = [int(k) for k in args.ks.split(",")]
    members = fanin.manifest_members(args.from_files, "embedding_tsv", "loadings_tsv")
    df = run(members, args.reference_module, ks, matrix=args.normalized_selected_h5)
    out = cli.output_path(args, "pca_metrics_tsv", "_pca_metrics.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out, separator="\t")
    print(f"wrote {df.height} rows for {df['pca'].n_unique() if df.height else 0} PCA modules: {out}")


if __name__ == "__main__":
    main()
