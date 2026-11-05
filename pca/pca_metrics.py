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
  pa_edge        index 0  parallel-analysis noise edge as a share of total
                          variance: the --pa_quantile of the TOP eigenvalue
                          after shuffling every gene across cells. PCs above it
                          carry structure; below it is the noise bulk.
                          Edge form, not Horn's i-th vs i-th comparison (that is
                          PCAtools::parallelPCA); for the first component the
                          two coincide and equal the Tracy-Widom test.

References
  Horn JL (1965) A rationale and test for the number of factors in factor
    analysis. Psychometrika 30:179-185. doi:10.1007/BF02289447
  Buja A, Eyuboglu N (1992) Remarks on parallel analysis. Multivariate Behav
    Res 27(4):509-540. doi:10.1207/s15327906mbr2704_2  (permutation form)
  Glorfeld LW (1995) An improvement on Horn's parallel analysis methodology for
    selecting the correct number of factors to retain. Educ Psychol Meas
    55(3):377-393.  (95th percentile instead of the mean/median)
  Saccenti E, Timmerman ME (2017) Considering Horn's parallel analysis from a
    random matrix theory point of view. Psychometrika 82:186-209.
    doi:10.1007/s11336-016-9515-z
  Single-cell use: PCAtools::parallelPCA (Blighe & Lun, Bioconductor); Lun A,
    "Choosing the number of PCs for denoising" (CRUK technical report).
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
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh

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


def top_eigenvalue(x):
    """Largest eigenvalue of the gene covariance of genes x cells CSR `x`,
    centring implicitly: C v = (X (X^T v) - n mu (mu^T v)) / (n - 1)."""
    n = x.shape[1]
    mu = np.asarray(x.mean(axis=1)).ravel()
    xt = x.T.tocsr()
    op = LinearOperator((x.shape[0],) * 2, dtype=np.float64,
                        matvec=lambda v: (x @ (xt @ v) - n * mu * (mu @ v)) / (n - 1))
    return float(eigsh(op, k=1, which="LA", return_eigenvectors=False)[0])


def shuffle_genes(x, rng):
    """Permute every gene independently across cells. For a sparse row that is
    moving its nonzeros to random distinct cells; per-gene values are kept."""
    n = x.shape[1]
    cols = np.empty_like(x.indices)
    for g in range(x.shape[0]):
        a, b = x.indptr[g], x.indptr[g + 1]
        cols[a:b] = rng.choice(n, b - a, replace=False)
    y = sp.csr_matrix((x.data.copy(), cols, x.indptr.copy()), shape=x.shape)
    y.sort_indices()
    return y


def noise_edge(x, n_perm, quantile, seed):
    """Parallel-analysis edge: `quantile` of the top eigenvalue over `n_perm`
    gene-wise shuffles of `x`."""
    rng = np.random.default_rng(seed)
    tops = [top_eigenvalue(shuffle_genes(x, rng)) for _ in range(n_perm)]
    return float(np.quantile(tops, quantile))


def run(embeddings, loadings, reference_module, ks, stage="PCA", anchor="FEAT", matrices=(),
        pa_n_perm=20, pa_quantile=0.95, pa_seed=42):
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
            if isinstance(totals[key], str):  # once per FEAT node, then cached
                x = read_tenx(totals[key])
                tot = total_variance(x)
                edge = noise_edge(x, pa_n_perm, pa_quantile, pa_seed) if pa_n_perm > 0 else np.nan
                totals[key] = (tot, edge)
                print(f"{key}: total variance {tot:.4g}, PA edge {edge / tot:.4%} of it")
            tot, edge = totals[key]
            var = emb.var(axis=0, ddof=1)
            extra = [("var_explained", i + 1, float(v / tot)) for i, v in enumerate(var)]
            extra.append(("pa_edge", 0, float(edge / tot)))
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
    p.add_argument("--pa_n_perm", type=int, default=20,
                   help="parallel-analysis shuffles per FEAT matrix (0 = skip)")
    p.add_argument("--pa_quantile", type=float, default=0.95,
                   help="quantile of the shuffled top eigenvalue taken as the edge")
    p.add_argument("--random_seed", type=int, default=42, help="seed for the shuffles")
    p.add_argument("--reference_module", default="pc_exact",
                   help="PCA module id taken as the exact reference")
    p.add_argument("--ks", default="5,10,20,30,50",
                   help="leading-subspace sizes for the principal-angle metrics")
    return p.parse_args()


def main():
    args = parse_args()
    ks = [int(k) for k in args.ks.split(",")]
    df = run(args.embedding_tsv, args.loadings_tsv, args.reference_module, ks,
             matrices=args.normalized_selected_h5, pa_n_perm=args.pa_n_perm,
             pa_quantile=args.pa_quantile, pa_seed=args.random_seed)
    out = cli.output_path(args, "pca_metrics_tsv", "_pca_metrics.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out, separator="\t")
    print(f"wrote {df.height} rows for {df['pca'].n_unique()} PCA modules: {out}")


if __name__ == "__main__":
    main()
