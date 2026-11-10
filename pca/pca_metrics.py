#!/usr/bin/env python3
"""PCA_M (gather): accuracy of every PCA arm against an exact reference.

ob gathers every producer of embedding_tsv / loadings_tsv under one FEAT node
and hands them over in one file (`--from-files`, lineage.json v2, ob api 0.8.0),
with that node's own matrix as `--normalized_selected_h5` (total variance). Each PCA member is
compared with the reference member (--reference_module, default pc_exact).
Members without a PCA stage in their path (ISOMAP, CNTFCT, RDIMR share the
embedding_tsv id) are skipped.

Output: one long TSV, one row per (member, metric, index). After the lineage
columns, one column per `<stage>_<parameter>` of the member's lineage (e.g.
`feat_number_selected`, `pca_solver`), taken from --from-files.
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
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent / "../src"))  # vendored `common`
from common import cli  # noqa: E402

COLUMNS = {"dataset": pl.Utf8, "norm": pl.Utf8, "feat": pl.Utf8, "pca": pl.Utf8,
           "pca_params": pl.Utf8, "is_reference": pl.Boolean, "metric": pl.Utf8,
           "index": pl.Int64, "value": pl.Float64}
# a parameterless node may carry no param field (`STAGE.module`): read it as "default"
SEGMENT = re.compile(r"^([A-Z][A-Z0-9_]*)\.([A-Za-z0-9_]+)(?:\.([A-Za-z0-9_]+))?$")


def lineage(path):
    """Stage -> (module, params) from an ob output path, in path order."""
    out = {}
    for part in Path(path).parts:
        m = SEGMENT.match(part)
        if m:
            out[m.group(1)] = (m.group(2), m.group(3) or "default")
    return out


def prefix(path, stage):
    """The path up to and including the `stage` segment: one input matrix."""
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part.startswith(stage + "."):
            return str(Path(*parts[: i + 1]))
    return None


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


def pair_members(embeddings, loadings, stage):
    """{embedding path: loadings path} for members that ran `stage`."""
    by_dir = {str(Path(p).parent): p for p in loadings}
    out = {}
    for e in embeddings:
        if stage not in lineage(e):
            continue
        lo = by_dir.get(str(Path(e).parent))
        if lo is None:
            raise ValueError(f"no loadings_tsv next to {e}")
        out[e] = lo
    return out


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


def run(embeddings, loadings, reference_module, ks, stage="PCA", anchor="FEAT", matrix=None,
        labels=None):
    """`labels`: member embedding path -> extra columns (its lineage's parameter
    values, from cli.member_parameters), placed after pca_params."""
    members = pair_members(embeddings, loadings, stage)
    refs = {}
    for e in members:
        if lineage(e)[stage][0] == reference_module:
            key = prefix(e, anchor)
            if key in refs:
                raise ValueError(f"two {reference_module} members under {key}")
            refs[key] = e
    if not members:  # group without PCA (e.g. only RDIMR embeddings): nothing to compare
        print(f"no {stage} members in this group; writing an empty table")
        return pl.DataFrame(schema=COLUMNS)
    if not refs:
        raise ValueError(f"no {reference_module} member among {len(members)} {stage} members")

    # One gather group is one FEAT node, so one matrix serves every member.
    total = total_variance(matrix) if matrix else None

    rows = []
    for e, lo in sorted(members.items()):
        key = prefix(e, anchor)
        if key not in refs:
            print(f"WARN no {reference_module} under {key}; skipping {e}")
            continue
        r = refs[key]
        cells, ref_emb = read_matrix(r)
        genes, ref_load = read_matrix(members[r])
        emb = align(*read_matrix(e), cells)
        load = align(*read_matrix(lo), genes)
        lin = lineage(e)
        meta = {s: f"{m}.{h}" for s, (m, h) in lin.items()}
        extra = []
        if total is not None:
            extra = [("var_explained", i + 1, float(v / total))
                     for i, v in enumerate(emb.var(axis=0, ddof=1))]
        for metric, idx, value in compare(ref_emb, ref_load, emb, load, ks) + extra:
            rows.append({
                "dataset": lin.get("DATA", ("", ""))[0],
                "norm": meta.get("NORM", ""),
                "feat": meta.get("FEAT", ""),
                "pca": lin[stage][0],
                "pca_params": lin[stage][1],
                **(labels or {}).get(e, {}),
                "is_reference": e == r,
                "metric": metric,
                "index": idx,
                "value": value,
            })
    return frame(rows)


def frame(rows):
    """Rows -> a typed table. Parameter columns differ between members (a module
    without the parameter leaves it null); a column whose values mix types
    becomes text, ints mixed with floats become floats."""
    keys = list(dict.fromkeys(k for r in rows for k in r))
    tail = ["is_reference", "metric", "index", "value"]
    keys = [k for k in keys if k not in tail] + tail
    cols = {}
    for k in keys:
        vals = [r.get(k) for r in rows]
        if k in COLUMNS:
            cols[k] = pl.Series(k, vals, dtype=COLUMNS[k])
            continue
        kinds = {type(v) for v in vals if v is not None}
        if kinds == {int, float}:
            vals = [None if v is None else float(v) for v in vals]
        elif len(kinds) > 1:
            vals = [None if v is None else str(v) for v in vals]
        cols[k] = pl.Series(k, vals)
    return pl.DataFrame(cols)


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
    embeddings = cli.from_files(args.from_files, "embedding_tsv")
    loadings = cli.from_files(args.from_files, "loadings_tsv")
    labels = cli.member_parameters(args.from_files)
    df = run(embeddings, loadings, args.reference_module, ks,
             matrix=args.normalized_selected_h5, labels=labels)
    out = cli.output_path(args, "pca_metrics_tsv", "_pca_metrics.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out, separator="\t")
    print(f"wrote {df.height} rows for {df['pca'].n_unique() if df.height else 0} PCA modules: {out}")


if __name__ == "__main__":
    main()
