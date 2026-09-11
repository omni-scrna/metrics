#!/usr/bin/env python3
# Integration (batch-correction) metrics for omnibenchmark — scib-metrics backed.
#
# - label_asw/clisi compare the corrected embedding against cell-type truth labels (by
#   cell_id), not batch labels. clisi's n_neighbors=90 matches scib-metrics' own
#   Benchmarker default for LISI-family metrics.
# - pcr (pcr_comparison) compares batch-covariate variance explained pre- vs.
#   post-integration: needs a *second*, pre-integration embedding (--embedding_tsv) plus
#   per-cell batch labels from --rawdata_h5ad's obs/<batch_var> (named via
#   --properties_info), row-aligned with the corrected embedding rather than grouped per
#   batch. categorical=True since batch labels are strings.
# - ari/nmi score the CLUST-C stage's real cluster labels (--clusters_corrected_tsv)
#   against cell-type truth; scib_metrics only exposes internally-clustering
#   nmi_ari_cluster_labels_kmeans/leiden, so we call sklearn's
#   adjusted_rand_score/normalized_mutual_info_score directly instead.

import argparse
import sys
from pathlib import Path
import polars as pl
import json

import h5py
import yaml

sys.path.insert(0, str(Path(__file__).parent / "../src"))  # vendored `common` (src/common) + module-local writers
from common import cli  # noqa: E402

import scib_metrics
from scib_metrics.nearest_neighbors import pynndescent
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def log(msg: str) -> None:
    # Unbuffered stderr trail of what step main() is on -- if the process dies mid-run
    # (e.g. on a machine with different data/library versions), this is what makes the
    # otherwise-sparse log show exactly where and why, instead of just stopping silently.
    print(f"[integration.py] {msg}", file=sys.stderr, flush=True)


def parse_args():
    # We own the parser; src/common/cli injects the shared contract (base args + the
    # `INTG8-M` stage I/O from common/schema). This module's method params are
    # hand-rolled below, so the whole CLI stays visible here.
    p = argparse.ArgumentParser(description="INTG8-M module (scib-metrics-backed)")
    cli.add_base_args(p)              # --output_dir, --name
    cli.add_stage_args(p, "INTG8-M")  # --corrected_tsv, --rawdata_clusters_truth, --rawdata_h5ad, --properties_info, --embedding_tsv, --clusters_corrected_tsv
    p.add_argument("--n_neighbors", type=int, default=90,
                   help="Neighborhood size (knn) for the graph clisi_knn is computed over")
    p.add_argument("--random_state", type=int, default=0,
                   help="Random state for the pynndescent approximate KNN search")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # logging
    print(f"Output directory: {args.output_dir}")
    print(f"Module name: {args.name}")
    print(f"corrected_tsv: {args.corrected_tsv}")
    print(f"rawdata_clusters_truth: {args.rawdata_clusters_truth}")

    # diagnostics (stderr) -- one line per stage, so a crash's last line says where and
    # why instead of just stopping silently. Full invocation first, for reproducibility.
    log(f"args: corrected_tsv={args.corrected_tsv} rawdata_clusters_truth={args.rawdata_clusters_truth} "
        f"rawdata_h5ad={args.rawdata_h5ad} properties_info={args.properties_info} "
        f"embedding_tsv={args.embedding_tsv} clusters_corrected_tsv={args.clusters_corrected_tsv} "
        f"n_neighbors={args.n_neighbors} random_state={args.random_state}")

    try:
        corrected_df = pl.read_csv(args.corrected_tsv, separator="\t")
        truth_df = pl.read_csv(args.rawdata_clusters_truth, separator="\t")

        # Align the corrected embedding rows with truth labels by cell_id.
        merged = corrected_df.join(
            truth_df.select(["cell_id", "truths"]), on="cell_id", how="inner"
        )

        embedding = merged.drop(["cell_id", "truths"]).to_numpy()
        aligned_labels = merged["truths"].cast(str).to_numpy()
        n_cells = len(merged)
        n_labels = merged["truths"].n_unique()
        n_dropped = len(corrected_df) - n_cells
        log(f"n_cells={n_cells} n_labels={n_labels} n_dropped={n_dropped}")

        if n_labels >= 2:
            label_asw = float(scib_metrics.silhouette_label(embedding, aligned_labels))

            n_neighbors = min(args.n_neighbors, n_cells - 1)
            neighbors = pynndescent(
                embedding, n_neighbors=n_neighbors, random_state=args.random_state
            )
            clisi = float(scib_metrics.clisi_knn(neighbors, aligned_labels))
        else:
            label_asw = float("nan")
            clisi = float("nan")
        log(f"label_asw={label_asw} clisi={clisi}")

        # ari/nmi: cell-type truth vs. the CLUST-C cluster labels, row-aligned by cell_id against
        # the same corrected_tsv/truth-restricted cell set used for label_asw/clisi above.
        clusters_df = pl.read_csv(args.clusters_corrected_tsv, separator="\t")
        merged_clusters = merged.join(
            clusters_df.select(["cell_id", "cluster"]), on="cell_id", how="inner"
        )

        if n_labels >= 2:
            aligned_truths = merged_clusters["truths"].cast(str).to_numpy()
            aligned_clusters = merged_clusters["cluster"].cast(str).to_numpy()
            ari = float(adjusted_rand_score(aligned_truths, aligned_clusters))
            nmi = float(normalized_mutual_info_score(aligned_truths, aligned_clusters))
        else:
            ari = float("nan")
            nmi = float("nan")
        log(f"ari={ari} nmi={nmi}")

        # Batch labels: a different source than the cell-type truth above. Some datasets declare no
        # batch_var at all, in which case batch-based metrics aren't computable.
        props = yaml.safe_load(open(args.properties_info))
        batch_var = props.get("batch_var")
        have_batch = bool(batch_var)
        log(f"batch_var={batch_var!r} have_batch={have_batch}")

        if have_batch:
            with h5py.File(args.rawdata_h5ad, "r") as f:
                log(f"h5ad obs keys: {list(f['obs'].keys())}")
                h5_cell_ids = f["obs/_index"].asstr()[:]
                h5_batch_vals = f[f"obs/{batch_var}"].asstr()[:]
            batch_df = pl.DataFrame({"cell_id": h5_cell_ids, "batch": h5_batch_vals})

        pcas_df = pl.read_csv(args.embedding_tsv, separator="\t")
        pre_cols = [c for c in pcas_df.columns if c != "cell_id"]
        pcas_renamed = pcas_df.rename({c: f"pre__{c}" for c in pre_cols})

        if have_batch:
            merged_pcr = corrected_df.join(pcas_renamed, on="cell_id", how="inner").join(
                batch_df, on="cell_id", how="inner"
            )
            n_batches = merged_pcr["batch"].n_unique()
        else:
            merged_pcr = corrected_df.clear()
            n_batches = 0

        if have_batch and n_batches >= 2 and len(merged_pcr) >= 2:
            post_cols = [c for c in corrected_df.columns if c != "cell_id"]
            X_post = merged_pcr.select(post_cols).to_numpy()
            X_pre = merged_pcr.select([f"pre__{c}" for c in pre_cols]).to_numpy()
            covariate = merged_pcr["batch"].cast(str).to_numpy()
            pcr = float(scib_metrics.pcr_comparison(X_pre, X_post, covariate, categorical=True))
        else:
            pcr = float("nan")
        log(f"pcr={pcr} (n_batches={n_batches})")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = args.output_dir / f"{args.name}_integration_metrics.json"
        with open(out_path, "w") as fh:
            json.dump(
                {
                    "n_cells": n_cells,
                    "n_labels": n_labels,
                    "n_dropped": n_dropped,
                    "n_batches": n_batches,
                    "label_asw": label_asw,
                    "clisi": clisi,
                    "pcr": pcr,
                    "ari": ari,
                    "nmi": nmi,
                },
                fh,
                indent=2,
            )
        log(f"wrote {out_path}")
    except Exception as e:
        log(f"FAILED: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
