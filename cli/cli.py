#!/usr/bin/env python3
# Schema-driven argument parser for omnibenchmark metrics.
# Reads a JSON schema and builds an argparse parser from it.
#
# Convention: all arguments are required; no defaults.

import argparse
import json
from pathlib import Path

_TYPE_MAP = {"path": Path, "integer": int}
_SCHEMA = Path(__file__).parent / "embedding.json"


def parse_args() -> argparse.Namespace:
    with open(_SCHEMA) as f:
        schema = json.load(f)

    parser = argparse.ArgumentParser(description="OmniBenchmark embedding metrics")
    for arg in schema:
        kwargs: dict = {
            "required": True,
            "type": _TYPE_MAP.get(arg["type"], str),
            "help": arg.get("help", ""),
        }
        if "dest" in arg:
            kwargs["dest"] = arg["dest"]
        parser.add_argument(arg["flag"], **kwargs)

    return parser.parse_args()
