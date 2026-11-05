#!/usr/bin/env python3
"""PCA_M (gather): accuracy of every PCA arm against an exact reference.

ob gathers every producer of embedding_tsv / loadings_tsv for one dataset
(group_by DATA). A gather group mixes NORM and FEAT branches, so the reference
is chosen here, not by the runtime: each PCA member is compared with the
reference member (--reference_module, default pc_exact) whose lineage matches
up to FEAT, i.e. the PCA computed on the same normalized_selected matrix.
Members without a PCA stage in their path (ISOMAP, CNTFCT, RDIMR share the
embedding_tsv id) are skipped.

Output: one long TSV, one row per (member, metric, index):
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
                          gene's variance, not of the kept PCs), when
                          --normalized_selected_h5 is gathered too
IPR is unstable inside near-degenerate eigenspaces (where pc_sin rises), and
means different things under different preprocessing (rare population vs a
one-gene noise axis), so compare it within one input matrix only.
Sines from projection residuals: 1 - cos (or sqrt(1 - cos^2)) floors at ~1e-8.
NMF factors are neither orthogonal nor variance-ordered, so only the
subspace rows mean anything for them.
"""

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent / "../src"))  # vendored `common`
from common import cli  # noqa: E402

SEGMENT = re.compile(r"^([A-Z][A-Z0-9_]*)\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)$")


def lineage(path):
    """Stage -> (module, params) from an ob output path, in path order."""
    out = {}
    for part in Path(path).parts:
        m = SEGMENT.match(part)
        if m:
            out[m.group(1)] = (m.group(2), m.group(3))
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


def total_variance(h5):
    """Sum of per-gene variances (ddof=1) of a TENx genes x cells CSC matrix:
    the trace of the covariance PCA decomposes, without densifying."""
    with h5py.File(h5, "r") as f:
        g = f["matrix"]
        n_genes, n_cells = (int(v) for v in g["shape"][:])
        idx, x = g["indices"][:], g["data"][:].astype(np.float64)
    s1 = np.bincount(idx, weights=x, minlength=n_genes)
    s2 = np.bincount(idx, weights=x * x, minlength=n_genes)
    return float(((s2 - s1 * s1 / n_cells) / (n_cells - 1)).sum())


def run(embeddings, loadings, reference_module, ks, stage="PCA", anchor="FEAT", matrices=()):
    members = pair_members(embeddings, loadings, stage)
    # FEAT node dir -> its matrix, for the variance-explained denominator
    totals = {str(Path(h).parent): h for h in matrices}
    refs = {}
    for e in members:
        if lineage(e)[stage][0] == reference_module:
            key = prefix(e, anchor)
            if key in refs:
                raise ValueError(f"two {reference_module} members under {key}")
            refs[key] = e
    if not refs:
        raise ValueError(f"no {reference_module} member among {len(members)} {stage} members")

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
        if key in totals:
            if isinstance(totals[key], str):
                totals[key] = total_variance(totals[key])
            var = emb.var(axis=0, ddof=1)
            extra = [("var_explained", i + 1, float(v / totals[key])) for i, v in enumerate(var)]
        for metric, idx, value in compare(ref_emb, ref_load, emb, load, ks) + extra:
            rows.append({
                "dataset": lin.get("DATA", ("", ""))[0],
                "norm": meta.get("NORM", ""),
                "feat": meta.get("FEAT", ""),
                "pca": lin[stage][0],
                "pca_params": lin[stage][1],
                "is_reference": e == r,
                "metric": metric,
                "index": idx,
                "value": value,
            })
    return pl.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="PCA_M gather: PCA arms vs an exact reference")
    cli.add_base_args(p)  # --output_dir, --name, --output id=PATH
    p.add_argument("--embedding_tsv", nargs="+", required=True, help="every gathered embedding")
    p.add_argument("--loadings_tsv", nargs="+", required=True, help="every gathered loadings")
    p.add_argument("--normalized_selected_h5", nargs="*", default=[],
                   help="FEAT matrices, for variance explained (optional)")
    p.add_argument("--reference_module", default="pc_exact",
                   help="PCA module id taken as the exact reference")
    p.add_argument("--ks", default="5,10,20,30,50",
                   help="leading-subspace sizes for the principal-angle metrics")
    return p.parse_args()


def main():
    args = parse_args()
    ks = [int(k) for k in args.ks.split(",")]
    df = run(args.embedding_tsv, args.loadings_tsv, args.reference_module, ks,
             matrices=args.normalized_selected_h5)
    out = cli.output_path(args, "pca_metrics_tsv", "_pca_metrics.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out, separator="\t")
    print(f"wrote {df.height} rows for {df['pca'].n_unique()} PCA modules: {out}")


if __name__ == "__main__":
    main()
