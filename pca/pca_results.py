#!/usr/bin/env python3
"""PCA_RESULTS (global gather): every PCA_M table concatenated into one.

ob hands over the members in `--from-files` (lineage.json). Each table
already carries its lineage columns, so this is a plain row bind; a column
missing from a table (parameter columns differ between arms) is left empty.
"""

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent / "../src"))  # shipped `common`
from common import cli  # noqa: E402


def concat(files):
    # as text: a hash column that is all digits in one table and not in the
    # next would otherwise get two types
    tables = [pl.read_csv(f, separator="\t", infer_schema=False) for f in files]
    return pl.concat(tables, how="diagonal")


def main():
    p = argparse.ArgumentParser(description="PCA_RESULTS gather: row-bind the PCA_M tables")
    cli.add_base_args(p)  # --output_dir, --name, --output id=PATH
    p.add_argument("--from-files", dest="from_files", required=True,
                   help="ob gather record (lineage.json v2): every member's pca_metrics_tsv")
    args = p.parse_args()
    files = cli.from_files(args.from_files, "pca_metrics_tsv")
    if not files:
        sys.exit(f"no pca_metrics_tsv in {args.from_files}")
    df = concat(files)
    out = Path(cli.output_path(args, "pca_results_tsv", "_pca_results.tsv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out, separator="\t")
    print(f"wrote {df.height} rows from {len(files)} tables: {out}")


if __name__ == "__main__":
    main()
