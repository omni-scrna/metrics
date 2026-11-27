"""EMBED_M (gather, embedding.R): one long table over every member of a manifest.

Runs the R entrypoint on a synthetic lineage.json v2 record: a PCA member, an
ISOMAP member (different stage, same embedding_tsv id) and one whose file is
missing, which must leave NA rows with its error instead of failing the table.
Needs an Rscript with poem, argparser, jsonlite and data.table (the metrics
env): set RSCRIPT to point at one, else it is looked up on PATH.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
RSCRIPT = os.environ.get("RSCRIPT") or shutil.which("Rscript")
METRICS = ["meanSW", "meanClassSW", "pnSW", "minClassSW", "cdbw",
           "cohesion", "compactness", "sep", "dbcv"]


def _lineage(stage, module, params, parameters):
    return [
        {"node_id": "DATA-d1", "stage": "DATA", "module": "d1", "params": "aaaaaaaa",
         "parameters": {"dataset_name": "toy", "batch_var": "Sample"}},
        {"node_id": "NORM-n1", "stage": "NORM", "module": "n1", "params": None, "parameters": {}},
        {"node_id": "FEAT-f1", "stage": "FEAT", "module": "f1", "params": "bbbbbbbb",
         "parameters": {"number_selected": 2000}},
        {"node_id": f"{stage}-{module}", "stage": stage, "module": module,
         "params": params, "parameters": parameters},
    ]


@pytest.mark.skipif(RSCRIPT is None, reason="no Rscript (set RSCRIPT to the metrics env's)")
def test_scores_every_member_into_one_long_table(tmp_path):
    rng = np.random.default_rng(0)
    n, k = 150, 3
    labels = np.repeat(["a", "b", "c"], n // k)
    centers = {"a": 0.0, "b": 6.0, "c": 12.0}
    cells = [f"c{i}" for i in range(n)]
    pl.DataFrame({"cell_id": cells, "truths": labels}).write_csv(tmp_path / "truth.tsv", separator="\t")

    def embedding(name, dims, prefix):
        x = rng.normal(size=(n, dims)) + np.array([centers[l] for l in labels])[:, None]
        path = tmp_path / name
        pl.DataFrame({"cell_id": cells} | {f"{prefix}{j + 1}": x[:, j] for j in range(dims)}) \
            .write_csv(path, separator="\t")
        return str(path)

    members = [
        {"node_id": "pca", "commit": "x", "dir": str(tmp_path),
         "files": {"embedding_tsv": embedding("pca.tsv", 5, "PC")},
         "lineage": _lineage("PCA", "pc_scanpy", "f02267b9", {"n_components": 5, "solver": "arpack"})},
        {"node_id": "iso", "commit": "x", "dir": str(tmp_path),
         "files": {"embedding_tsv": embedding("iso.tsv", 4, "dim_")},
         "lineage": _lineage("ISOMAP", "dr_isomap", "0d1c2b3a", {"n_neighbors": 30})},
        {"node_id": "gone", "commit": "x", "dir": str(tmp_path),
         "files": {"embedding_tsv": str(tmp_path / "missing.tsv")},
         "lineage": _lineage("PCA", "pc_exact", "c0ffee00", {"n_components": 5})},
    ]
    manifest = tmp_path / "lineage.json"
    manifest.write_text(json.dumps({"version": 2, "node_id": "g", "stage": "EMBED_M",
                                    "module": "em_metrics_r", "kind": "gather", "members": members}))
    out = tmp_path / "out" / "em_embedding_metrics.tsv"

    run = subprocess.run(
        [RSCRIPT, str(ROOT / "embedding" / "embedding.R"),
         "--output_dir", str(tmp_path / "od"), "--name", "em_metrics_r",
         "--from-files", str(manifest), "--rawdata_clusters_truth", str(tmp_path / "truth.tsv"),
         "--output", f"embedding_metrics_tsv={out}"],
        cwd=tmp_path, capture_output=True, text=True)  # not the module root: paths resolve from the script
    assert run.returncode == 0, run.stdout + run.stderr

    df = pl.read_csv(out, separator="\t", infer_schema=False)
    assert df.height == 3 * len(METRICS)
    assert df.columns[-6:] == ["n_cells", "n_labels", "n_dropped", "metric", "value", "error"]
    # PCA_M's shape: dataset is the DATA module id, upstream stages are module.hash,
    # the member's own node is embedding_stage / embedding / embedding_params.
    assert df.columns[:6] == ["dataset", "norm", "feat", "embedding_stage", "embedding", "embedding_params"]
    assert set(df["embedding_stage"]) == {"PCA", "ISOMAP"}
    assert set(df["dataset"]) == {"d1"} and set(df["data_dataset_name"]) == {"toy"}
    assert set(df["feat"]) == {"f1.bbbbbbbb"} and set(df["norm"]) == {"n1"}
    assert "pca" not in df.columns and "isomap" not in df.columns  # own stage only as embedding_*
    assert set(df.filter(pl.col("embedding") == "pc_scanpy")["embedding_params"]) == {"f02267b9"}

    ok = df.filter(pl.col("embedding") != "pc_exact")
    assert ok["error"].is_null().all()
    assert set(ok["n_cells"]) == {str(n)} and set(ok["n_labels"]) == {"3"}
    sw = ok.filter(pl.col("metric") == "meanSW")["value"].cast(pl.Float64)
    assert ((sw > 0.5) & (sw <= 1)).all()  # well-separated clusters
    assert ok.filter(pl.col("embedding_stage") == "ISOMAP")["isomap_n_neighbors"].to_list() == ["30"] * len(METRICS)

    bad = df.filter(pl.col("embedding") == "pc_exact")
    assert bad["value"].is_null().all() and bad["error"].is_not_null().all()
