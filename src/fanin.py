"""Reading an ob gather record (`--from-files`, lineage.json v2, ob api 0.8) in Python.

Module-owned, not under src/common/ (which `sync` overwrites); the Python twin
of src/fanin.R, for any Python gather in this module. Every member lists its
files by output id and its full lineage, so nothing here parses ob's output
paths. Both write the same columns, so tables from R and Python gathers join.
"""

import json


def manifest_members(path, *ids):
    """Members that carry every output id in `ids` (e.g. "embedding_tsv",
    "loadings_tsv"), in manifest order."""
    with open(path) as fh:
        members = json.load(fh)["members"]
    return [m for m in members if all(i in m["files"] for i in ids)]


def member_files(path, output_id):
    """Every member's file for `output_id`, in manifest order."""
    return [m["files"][output_id] for m in manifest_members(path, output_id)]


def cell(v):
    """A parameter value as one table cell: scalars as they are, lists and
    mappings as compact JSON text (as jsonlite writes them in fanin.R)."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return json.dumps(v, separators=(",", ":"))


def node_label(entry):
    """`<module>.<params hash>`, or `<module>` for a node without parameters."""
    return entry["module"] if entry.get("params") is None else f"{entry['module']}.{entry['params']}"


def lineage_entry(member, stage):
    """The member's lineage entry for `stage`, or None."""
    return next((e for e in member["lineage"] if e["stage"] == stage), None)


def member_lineage_cols(member, as_):
    """One member's lineage as named table columns (same as fanin.R):
      dataset                  the DATA module id
      <stage>                  node_label() per upstream stage (filt, norm, feat, ...)
      <as_>_stage, <as_>, <as_>_params
                               the member's own node (its last lineage entry): stage,
                               module, params hash. `as_` names what is gathered,
                               e.g. "embedding", or "pca" for PCA_M.
      <stage>_<parameter>      every parameter value of the lineage
    """
    lin = member["lineage"]
    own = lin[-1]
    data = lineage_entry(member, "DATA")
    cols = {"dataset": data["module"] if data else None}
    for e in lin[:-1]:
        if e["stage"] != "DATA":
            cols[e["stage"].lower()] = node_label(e)
    cols[f"{as_}_stage"] = own["stage"]
    cols[as_] = own["module"]
    cols[f"{as_}_params"] = own.get("params")
    for e in lin:
        for k, v in (e.get("parameters") or {}).items():
            cols[f"{e['stage'].lower()}_{k}"] = cell(v)
    return cols
