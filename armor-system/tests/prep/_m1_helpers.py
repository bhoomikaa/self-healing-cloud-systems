"""Shared helpers for the M1-03 tests: a synthetic D1-shaped schema and a small synthetic timeline."""
import csv
import json
import os
import unittest

import _paths  # noqa: F401
import numpy as np

from prep import m1_split as ms, paths

# M1_REQUIRE_REAL_DATA=1 (set by data/m1_integration.py) turns every "skip if the data is missing" into a real failure.
REQUIRE_REAL = os.environ.get("M1_REQUIRE_REAL_DATA") == "1"
CACHE_PRESENT = os.path.exists(paths.CACHE_FILE)
ARTIFACTS_PRESENT = os.path.exists(os.path.join(paths.ART, "splits", f"{ms.MANIFEST_NAME}.json"))


def needs(condition, reason):
    """Skip decorator for real-data tests; a no-op (so the test runs and fails loudly) when real data is required."""
    return (lambda obj: obj) if REQUIRE_REAL else unittest.skipUnless(condition, reason)


def synthetic_schema():
    """D1-shaped schema: 6 VM nodes + 40 pods; channels 0-52 kpi_container, 53-61 trace, 62-72 log, 73-129 system."""
    nodes = [f"node-{i}" for i in range(1, 7)] + [f"pod{i}-0" for i in range(40)]
    chans = []
    for i in range(130):
        if i < 53:
            chans.append({"index": i, "name": f"kpi_container_m{i}", "modality": "metric"})
        elif i < 62:
            chans.append({"index": i, "name": f"trace_t{i}", "modality": "trace"})
        elif i < 73:
            chans.append({"index": i, "name": f"log_l{i}", "modality": "log"})
        else:
            chans.append({"index": i, "name": f"system.s{i}", "modality": "metric"})
    ranges = {m: [c["index"] for c in chans if c["modality"] == m] for m in ("metric", "trace", "log")}
    return {"tensor_shape_per_sample": [46, 130], "dtype": "float32", "node_order": nodes, "channel_order": chans,
            "modality_index_ranges": ranges, "channel_order_sha256": "c" * 64, "node_order_sha256": "n" * 64,
            "category_ids": {"0": "Container Hardware", "1": "Container Network", "3": "Node CPU", "4": "Node Disk", "5": "Node Memory"}}


def synthetic_absent_entry(schema):
    """Mimic D1 layer 1: VM trace/log blocks and 10 pod log blocks absent."""
    a = np.zeros((46, 130), bool)
    a[:6, 53:73] = True
    a[6:16, 62:73] = True
    return a


T_BASE = 1_000_000


def synthetic_timeline(n=4000):
    """ts one minute apart, offset so case times are not minute-aligned. Cases and boundaries chosen so Tc is admissible."""
    ts = T_BASE + 60 * np.arange(n, dtype=np.int64)
    idx = [300, 700, 1100, 1500, 2100, 2300, 3000, 3300]
    ct = ts[idx] + 17
    cases = {"timestamp": ct.astype(np.int64), "class_id": np.array([0, 1, 2, 3, 4, 0, 1, 2], np.int64),
             "group_id": np.arange(8, dtype=np.int64), "case_index": np.arange(8, dtype=np.int64),
             "outer_split": np.array(["train"] * 4 + ["validation"] * 2 + ["test"] * 2)}
    boundaries = (int(ts[1900]), int(ts[2700]))
    tc = int((ct[2] + ct[3]) // 2)
    return ts, cases, boundaries, ms.SplitConfig(tc=tc)


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def read_text(path):
    with open(path) as fh:
        return fh.read()


def read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))
