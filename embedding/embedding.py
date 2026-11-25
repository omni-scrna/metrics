#!/usr/bin/env python3
"""EMBED_M (gather): embedding quality metrics (scikit-learn) for every embedding of one dataset.

ob gathers every embedding_tsv under one DATA node (PCA, ISOMAP, ... arms alike)
and hands them over in one file (`--from-files`, lineage.json v2, ob api 0.8),
with that DATA node's truth labels as `--rawdata_clusters_truth`. Members are
scored in parallel, one per worker process; the number of workers is the
stage's `resources.cores`, which ob passes on as OMP_NUM_THREADS (default 1).

Outputs: two long TSVs, lineage columns first (src/fanin.py: dataset, filt,
norm, feat, ..., embedding_stage, embedding, embedding_params, then
`<stage>_<parameter>`; from the manifest, never from paths), so they stack with
the R module's table and join with each other and with PCA_M / GRAPH_M.

  embedding_metrics_tsv        one row per (embedding, metric), then
                               n_cells, n_labels, n_dropped, metric, value, error
    meanSW             mean silhouette width over cells (= silhouette_score)
    meanClassSW        mean over labels of the label's mean silhouette width
    minClassSW         the lowest label mean silhouette width
    pnSW               share of cells with a negative silhouette width
    davies_bouldin     Davies-Bouldin index (lower is better)
    calinski_harabasz  Calinski-Harabasz index (higher is better)
  embedding_class_metrics_tsv  one row per (embedding, truth label, metric), then
                               label, label_n_cells, metric, value, error
    SW                 the label's mean silhouette width
    pnSW               share of the label's cells with a negative silhouette width

The silhouette names follow poem's (embedding.R), so meanSW is the same number
in both modules. Silhouettes are Euclidean, on every cell (no subsampling).

Implementation notes
--------------------
- Every metric needs >= 2 labels; NaN otherwise.
- Cells are aligned to the truth by cell_id; cells without a truth label (absent
  or null) are dropped and counted in n_dropped.
- An embedding that fails to score keeps its rows, with NaN values and the
  error message, so one bad member does not lose the rest of the tables.
"""

import argparse
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent / "../src"))  # vendored `common`, fanin
import fanin  # noqa: E402
from common import cli  # noqa: E402

DATASET_METRICS = ["meanSW", "meanClassSW", "minClassSW", "pnSW",
                   "davies_bouldin", "calinski_harabasz"]
CLASS_METRICS = ["SW", "pnSW"]

DATASET_TAIL = {"n_cells": pl.Int64, "n_labels": pl.Int64, "n_dropped": pl.Int64,
                "metric": pl.Utf8, "value": pl.Float64, "error": pl.Utf8}
CLASS_TAIL = {"label": pl.Utf8, "label_n_cells": pl.Int64,
              "metric": pl.Utf8, "value": pl.Float64, "error": pl.Utf8}


def parse_args():
    p = argparse.ArgumentParser(description="EMBED_M gather: embedding metrics (scikit-learn)")
    cli.add_base_args(p)              # --output_dir, --name, --output id=PATH
    cli.add_stage_args(p, "EMBED_M")  # --from-files, --rawdata_clusters_truth
    return p.parse_args()


def workers():
    """Worker processes: the stage's cores, as ob exports them (OMP_NUM_THREADS)."""
    try:
        return max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
    except ValueError:
        return 1


def read_truth(path):
    """cell_id -> truth label (as text); cells with a null label are left out."""
    df = pl.read_csv(path, separator="\t", infer_schema=False).drop_nulls("truths")
    return dict(zip(df["cell_id"].to_list(), df["truths"].to_list()))


def read_embedding(path):
    # no type inference: a column whose first rows are integers (NMF zeros)
    # would be typed Int64 and fail on the first decimal
    df = pl.read_csv(path, separator="\t", infer_schema=False)
    return df[:, 0].to_list(), df[:, 1:].cast(pl.Float64).to_numpy()


def score(x, labels):
    """Dataset- and class-level metrics of embedding `x` (cells x dims) against
    `labels` (one per cell). Returns ({metric: value}, {label: (n, {metric: value})})."""
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        return ({m: float("nan") for m in DATASET_METRICS},
                {c: (int(n), {m: float("nan") for m in CLASS_METRICS})
                 for c, n in zip(classes, counts)})
    sw = silhouette_samples(x, labels)
    per_class = {c: (int(n), {"SW": float(sw[labels == c].mean()),
                              "pnSW": float((sw[labels == c] < 0).mean())})
                 for c, n in zip(classes, counts)}
    class_sw = [v["SW"] for _, v in per_class.values()]
    dataset = {
        "meanSW": float(sw.mean()),
        "meanClassSW": float(np.mean(class_sw)),
        "minClassSW": float(np.min(class_sw)),
        "pnSW": float((sw < 0).mean()),
        "davies_bouldin": float(davies_bouldin_score(x, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
    }
    return dataset, per_class


def score_member(path, truth):
    """Score one embedding file against `truth` (cell_id -> label). Runs in a
    worker: BLAS is held to one thread so N workers use N cores, not N^2."""
    t0 = time.monotonic()
    try:
        with threadpool_limits(limits=1):
            cells, x = read_embedding(path)
            keep = [i for i, c in enumerate(cells) if c in truth]
            labels = np.array([truth[cells[i]] for i in keep])
            dataset, per_class = score(x[keep], labels)
        return {"n_cells": len(keep), "n_labels": len(per_class),
                "n_dropped": len(cells) - len(keep), "dataset": dataset,
                "per_class": per_class, "error": None, "secs": time.monotonic() - t0}
    except Exception as e:  # one bad member must not lose the rest of the table
        return {"error": f"{type(e).__name__}: {e}", "secs": time.monotonic() - t0}


def rows_for(member, res, truth_labels):
    """The member's rows for both tables. A failed member gets NaN rows, one
    per metric (and per truth label in the class table), with its error."""
    lin = fanin.member_lineage_cols(member, "embedding")
    err = res["error"]
    if err:
        dataset = [{**lin, "n_cells": None, "n_labels": None, "n_dropped": None,
                    "metric": m, "value": float("nan"), "error": err} for m in DATASET_METRICS]
        per_class = [{**lin, "label": c, "label_n_cells": None, "metric": m,
                      "value": float("nan"), "error": err}
                     for c in truth_labels for m in CLASS_METRICS]
        return dataset, per_class
    dataset = [{**lin, "n_cells": res["n_cells"], "n_labels": res["n_labels"],
                "n_dropped": res["n_dropped"], "metric": m, "value": v, "error": None}
               for m, v in res["dataset"].items()]
    per_class = [{**lin, "label": c, "label_n_cells": n, "metric": m, "value": v, "error": None}
                 for c, (n, vals) in res["per_class"].items() for m, v in vals.items()]
    return dataset, per_class


def run(members, truth, n_workers=1):
    """Score every member; returns (dataset table, class table), in manifest order."""
    paths = [m["files"]["embedding_tsv"] for m in members]
    if n_workers > 1 and len(members) > 1:
        # spawn, not fork: polars' thread pool is already running in this process
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(n_workers, len(members)), mp_context=ctx) as pool:
            results = list(pool.map(score_member, paths, [truth] * len(paths)))
    else:
        results = [score_member(p, truth) for p in paths]

    truth_labels = sorted(set(truth.values()))
    dataset_rows, class_rows = [], []
    for i, (m, res) in enumerate(zip(members, results), 1):
        own = m["lineage"][-1]
        status = "ok" if res["error"] is None else f"FAILED: {res['error']}"
        print(f"[{i}/{len(members)}] {own['stage']}.{own['module']}.{own.get('params') or ''}"
              f": {status} ({res['secs']:.0f}s)")
        d, c = rows_for(m, res, truth_labels)
        dataset_rows += d
        class_rows += c
    return fanin.frame(dataset_rows, list(DATASET_TAIL), DATASET_TAIL), \
        fanin.frame(class_rows, list(CLASS_TAIL), CLASS_TAIL)


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    members = fanin.manifest_members(args.from_files, "embedding_tsv")
    if not members:
        raise SystemExit(f"no embedding_tsv member in {args.from_files}")
    truth = read_truth(args.rawdata_clusters_truth)
    n = workers()
    print(f"{len(members)} embeddings, {len(truth)} labelled cells, {n} worker(s)")

    dataset, per_class = run(members, truth, n)
    for oid, suffix, df in [("embedding_metrics_tsv", "_embedding_metrics.tsv", dataset),
                            ("embedding_class_metrics_tsv", "_embedding_class_metrics.tsv", per_class)]:
        out = cli.output_path(args, oid, suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(out, separator="\t")
        print(f"wrote {df.height} rows: {out}")


if __name__ == "__main__":
    main()
