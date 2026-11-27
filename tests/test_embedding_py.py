"""EMBED_M (gather, embedding.py): two long tables over every member of a manifest.

Synthetic lineage.json v2 record: a PCA member, an ISOMAP member (different
stage, same embedding_tsv id) and one whose file is missing, which must leave
NaN rows with its error instead of failing the tables.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import silhouette_samples, silhouette_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "embedding"))
from embedding import CLASS_METRICS, DATASET_METRICS, run, score  # noqa: E402

LABELS = ["a", "b", "c"]
N = 150


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


def _labels():
    return np.repeat(LABELS, N // len(LABELS))


@pytest.fixture
def record(tmp_path):
    """(manifest path, truth path) for three members: PCA, ISOMAP, missing file."""
    rng = np.random.default_rng(0)
    labels = _labels()
    centers = {"a": 0.0, "b": 6.0, "c": 12.0}
    cells = [f"c{i}" for i in range(N)]
    # one truth-less cell and one null label: both dropped, counted in n_dropped
    truth = pl.DataFrame({"cell_id": cells[:-1] + ["extra"],
                          "truths": list(labels[:-2]) + [None, "a"]})
    truth.write_csv(tmp_path / "truth.tsv", separator="\t")

    def embedding(name, dims, prefix):
        x = rng.normal(size=(N, dims)) + np.array([centers[l] for l in labels])[:, None]
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
                                    "module": "em_metrics_py", "kind": "gather", "members": members}))
    return manifest, tmp_path / "truth.tsv"


def test_score_matches_sklearn_and_poem_definitions():
    rng = np.random.default_rng(1)
    labels = _labels()
    x = rng.normal(size=(N, 3)) + (labels == "b")[:, None] * 3.0
    dataset, per_class = score(x, labels)
    sw = silhouette_samples(x, labels)
    assert dataset["meanSW"] == pytest.approx(silhouette_score(x, labels))
    class_sw = {c: sw[labels == c].mean() for c in LABELS}
    assert {c: v["SW"] for c, (_, v) in per_class.items()} == pytest.approx(class_sw)
    assert dataset["meanClassSW"] == pytest.approx(np.mean(list(class_sw.values())))
    assert dataset["minClassSW"] == pytest.approx(min(class_sw.values()))
    assert dataset["pnSW"] == pytest.approx((sw < 0).mean())
    assert per_class["a"][1]["pnSW"] == pytest.approx((sw[labels == "a"] < 0).mean())
    assert {c: n for c, (n, _) in per_class.items()} == {"a": 50, "b": 50, "c": 50}
    assert list(dataset) == DATASET_METRICS


def test_one_label_is_nan():
    dataset, per_class = score(np.zeros((4, 2)), np.array(["a"] * 4))
    assert all(np.isnan(v) for v in dataset.values())
    assert np.isnan(per_class["a"][1]["SW"])


def test_parallel_equals_serial(record):
    manifest, truth_path = record
    from embedding import read_truth
    import fanin
    members = fanin.manifest_members(manifest, "embedding_tsv")
    truth = read_truth(truth_path)
    serial = run(members, truth, 1)
    parallel = run(members, truth, 3)
    for s, p in zip(serial, parallel):
        assert s.equals(p, null_equal=True)


def test_cli_writes_both_tables(record, tmp_path):
    manifest, truth = record
    out = tmp_path / "out"
    run_ = subprocess.run(
        [sys.executable, str(ROOT / "embedding" / "embedding.py"),
         "--output_dir", str(tmp_path / "od"), "--name", "em_metrics_py",
         "--from-files", str(manifest), "--rawdata_clusters_truth", str(truth),
         "--output", f"embedding_metrics_tsv={out / 'em.tsv'}",
         "--output", f"embedding_class_metrics_tsv={out / 'em_class.tsv'}"],
        cwd=tmp_path, capture_output=True, text=True,
        env=os.environ | {"OMP_NUM_THREADS": "2"})
    assert run_.returncode == 0, run_.stdout + run_.stderr
    assert "2 worker(s)" in run_.stdout

    df = pl.read_csv(out / "em.tsv", separator="\t")
    assert df.height == 3 * len(DATASET_METRICS)
    assert df.columns[:6] == ["dataset", "norm", "feat", "embedding_stage", "embedding", "embedding_params"]
    assert df.columns[-6:] == ["n_cells", "n_labels", "n_dropped", "metric", "value", "error"]
    assert "pca" not in df.columns and "isomap" not in df.columns
    assert df["dataset"].unique().to_list() == ["d1"]
    assert sorted(df["embedding_stage"].unique().to_list()) == ["ISOMAP", "PCA"]
    ok = df.filter(pl.col("error").is_null())
    assert ok.height == 2 * len(DATASET_METRICS)
    assert ok["n_cells"].unique().to_list() == [N - 2]   # the last cell has no truth, one is null
    assert ok["n_dropped"].unique().to_list() == [2]
    assert ok["n_labels"].unique().to_list() == [3]
    assert ok.filter(pl.col("metric") == "meanSW")["value"].min() > 0.5  # well separated
    gone = df.filter(pl.col("embedding") == "pc_exact")
    assert gone["error"].str.contains("FileNotFound").all() and gone["value"].is_nan().all()

    cl = pl.read_csv(out / "em_class.tsv", separator="\t")
    assert cl.columns[-5:] == ["label", "label_n_cells", "metric", "value", "error"]
    assert cl.height == 3 * len(LABELS) * len(CLASS_METRICS)   # the failed member keeps its rows
    assert sorted(cl["label"].unique().to_list()) == LABELS
    sizes = cl.filter(pl.col("error").is_null()).group_by("label").agg(pl.col("label_n_cells").first())
    assert dict(sizes.iter_rows()) == {"a": 50, "b": 50, "c": 48}
