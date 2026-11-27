"""src/fanin.py (and its R twin src/fanin.R): reading ob's gather record."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import fanin  # noqa: E402

RSCRIPT = os.environ.get("RSCRIPT") or shutil.which("Rscript")

MEMBER = {
    "node_id": "pca", "files": {"embedding_tsv": "/a/e.tsv", "loadings_tsv": "/a/l.tsv"},
    "lineage": [
        {"node_id": "D", "stage": "DATA", "module": "d1", "params": "aaaaaaaa",
         "parameters": {"dataset_name": "be1"}},
        {"node_id": "F", "stage": "FILT", "module": "fi_x", "params": None, "parameters": {}},
        {"node_id": "N", "stage": "NORM", "module": "nr_x", "params": None, "parameters": {}},
        {"node_id": "E", "stage": "FEAT", "module": "fe_x", "params": "bbbbbbbb",
         "parameters": {"number_selected": 2000}},
        {"node_id": "P", "stage": "PCA", "module": "pc_x", "params": "cccccccc",
         "parameters": {"solver": "arpack", "n_components": 50, "ks": [1, 2], "dense": True}},
    ],
}
COLS = {
    "dataset": "d1", "filt": "fi_x", "norm": "nr_x", "feat": "fe_x.bbbbbbbb",
    "embedding_stage": "PCA", "embedding": "pc_x", "embedding_params": "cccccccc",
    "data_dataset_name": "be1", "feat_number_selected": 2000,
    "pca_solver": "arpack", "pca_n_components": 50, "pca_ks": "[1,2]", "pca_dense": True,
}


def _record(tmp_path):
    rd = {"node_id": "rd", "files": {"embedding_tsv": "/b/e.tsv"}, "lineage": MEMBER["lineage"][:1]}
    path = tmp_path / "lineage.json"
    path.write_text(json.dumps({"version": 2, "members": [MEMBER, rd]}))
    return path


def test_members_carry_every_requested_id(tmp_path):
    path = _record(tmp_path)
    assert [m["node_id"] for m in fanin.manifest_members(path, "embedding_tsv")] == ["pca", "rd"]
    assert [m["node_id"] for m in fanin.manifest_members(path, "embedding_tsv", "loadings_tsv")] == ["pca"]
    assert fanin.member_files(path, "embedding_tsv") == ["/a/e.tsv", "/b/e.tsv"]


def test_lineage_cols():
    cols = fanin.member_lineage_cols(MEMBER, "embedding")
    assert cols == COLS
    assert list(cols)[:7] == list(COLS)[:7]  # dataset, upstream stages, then the member's own node


@pytest.mark.skipif(RSCRIPT is None, reason="no Rscript (set RSCRIPT to the metrics env's)")
def test_r_twin_writes_the_same_columns(tmp_path):
    path = tmp_path / "member.json"
    path.write_text(json.dumps(MEMBER))
    code = (f'source("{ROOT / "src" / "fanin.R"}"); '
            f'm <- jsonlite::fromJSON("{path}", simplifyVector = FALSE); '
            'cat(jsonlite::toJSON(member_lineage_cols(m, "embedding"), auto_unbox = TRUE, na = "null"))')
    run = subprocess.run([RSCRIPT, "-e", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    r = json.loads(run.stdout)
    assert list(r) == list(COLS) and r == COLS
