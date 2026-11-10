"""Materialise the prepared D1 data as a partitioned Parquet lakehouse.

The pipeline's own artifacts are summaries: statistics, manifests, configs. They
are what a reviewer reads. They are not a place anything can query. This builds
the queryable layer, which is what the data-pipeline demo asks for: ingestion
from source through to a warehouse or lakehouse, with a file layout, tables and
schemas you can show.

Five tables, a star schema around one fact table:

    dim_node          46 rows    one per node in the D1 layout
    dim_channel      130 rows    one per channel, tagged with its modality
    dim_split          3 rows    the approved Model 1 split and its boundaries
    fact_case        210 rows    the labelled failure cases
    fact_telemetry   43,056,000 rows, partitioned by UTC date

`fact_telemetry` is long rather than wide: one row per timestamp, node and
channel. Wide would be 46 x 7200 rows of 130 float columns, which is smaller on
disk but makes every question about a single channel a column-name problem, and
D1's channel names contain dots that most warehouses will not accept as
identifiers. Long keeps the names as data, where they can be joined.

Values are the pre-scaler features from the feature cache, so this is the
transformed layer, not the model-ready layer. Scaling is fitted on the training
period only and belongs to the model, not the warehouse; putting scaled values
here would bake one split's statistics into shared storage.

    python -I build_warehouse.py [--out DIR] [--limit-days N]

Writes `_manifest.json` beside the tables with a row count and SHA-256 for every
file, so a load into BigQuery or S3 can be verified rather than assumed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(HERE, "..", "prep")
if PREP not in sys.path:
    sys.path.insert(0, PREP)

import paths  # noqa: E402

SPLIT_MANIFEST = "model1_split_101_25_84_v1"
# Compression: zstd beats snappy by about a third on this data and every engine
# that matters reads it. Level 3 is the point where more effort stops paying.
COMPRESSION, COMPRESSION_LEVEL = "zstd", 3


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_date(ts):
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%d")


def load_inputs():
    schema = json.load(open(paths.art("inspection", "schema.json"), encoding="utf-8"))
    split = json.load(open(paths.art("splits", f"{SPLIT_MANIFEST}.json"), encoding="utf-8"))
    cache = np.load(paths.CACHE_FILE, allow_pickle=True)
    cases = pd.read_csv(os.path.join(paths.D1_DIR, "cases", "cases.csv"))
    return schema, split, cache, cases


def build_dim_node(schema):
    nodes = schema["node_order"]
    # D1 mixes two entity kinds in one axis: service replicas such as
    # "adservice-0" and the VMs they run on, "node-1" to "node-6". A query that
    # does not separate them silently averages a container against a hypervisor.
    kind = ["vm" if n.startswith("node-") else "container" for n in nodes]
    service = [None if k == "vm" else n.rsplit("-", 1)[0] for n, k in zip(nodes, kind)]
    replica = [None if k == "vm" else int(n.rsplit("-", 1)[1]) for n, k in zip(nodes, kind)]
    return pd.DataFrame({
        "node_index": np.arange(len(nodes), dtype="int16"),
        "node_name": nodes,
        "node_kind": kind,
        "service": service,
        "replica": pd.array(replica, dtype="Int16"),
    })


def build_dim_channel(schema):
    # channel_order entries are objects: {"index", "name", "modality"}.
    channels = schema["channel_order"]
    names = [c["name"] for c in channels]
    modality = [c["modality"] for c in channels]

    # modality_index_ranges says the same thing a second way. Cross-check rather
    # than pick one, because a disagreement would silently mislabel a whole
    # modality and the model's missing-aware fusion keys off exactly this.
    ranges = schema["modality_index_ranges"]
    from_ranges = [None] * len(channels)
    for mod, idxs in ranges.items():
        for i in idxs:
            from_ranges[i] = mod
    disagree = [(i, a, b) for i, (a, b) in enumerate(zip(modality, from_ranges)) if a != b]
    if disagree:
        raise SystemExit(f"channel_order and modality_index_ranges disagree: {disagree[:5]}")

    return pd.DataFrame({
        "channel_index": np.arange(len(channels), dtype="int16"),
        "channel_name": names,
        "modality": modality,
    })


def build_dim_split(split):
    bounds = {tuple(b["between"]): b for b in split["sample_time_boundaries"]}
    tv = bounds[("train", "validation")]["boundary_ts"]
    vt = bounds[("validation", "test")]["boundary_ts"]
    counts = split["sample_partition"]
    ranges = split["case_ranges"]
    rows = []
    for name, lo, hi in (("train", None, tv), ("validation", tv, vt), ("test", vt, None)):
        r = ranges[name]
        rows.append({
            "split": name,
            "start_ts_inclusive": lo,
            "end_ts_exclusive": hi,
            "n_samples": int(counts[name]),
            "n_cases": int(r["n_cases"]),
            "first_case_ts": int(r["first_case_ts"]),
            "last_case_ts": int(r["last_case_ts"]),
            "class_counts_json": json.dumps(r["class_counts"], sort_keys=True),
        })
    df = pd.DataFrame(rows)
    # The first split has no lower bound and the last has no upper bound.
    # Int64 keeps those as a null rather than demoting the whole column to
    # float, which would print a timestamp as 1651713858.0.
    for c in ("start_ts_inclusive", "end_ts_exclusive"):
        df[c] = df[c].astype("Int64")
    return df, tv, vt


def split_of(ts, tv, vt):
    return np.where(ts < tv, "train", np.where(ts < vt, "validation", "test"))


def build_fact_case(cases, tv, vt, schema):
    df = cases.copy()
    df.insert(0, "case_index", np.arange(len(df), dtype="int32"))
    df["utc"] = [dt.datetime.fromtimestamp(int(t), dt.timezone.utc).isoformat().replace("+00:00", "Z")
                 for t in df["timestamp"]]
    df["utc_date"] = [utc_date(t) for t in df["timestamp"]]
    df["split"] = split_of(df["timestamp"].to_numpy(), tv, vt)
    # cases.csv labels are fine-grained ("Node CPU Failure"); the model predicts
    # the five coarse classes. Carry both so a query can group either way.
    # category_ids is keyed by the id as a *string*, so the lookup is via str().
    fine_to_id = schema["fine_type_to_category_id"]
    id_to_class = {str(k): v for k, v in schema["category_ids"].items()}
    df["category_id"] = pd.array([fine_to_id.get(f) for f in df["failure_type"]], dtype="Int16")
    df["failure_class"] = [id_to_class.get(str(fine_to_id.get(f))) for f in df["failure_type"]]
    unmapped = sorted(set(df.loc[df["failure_class"].isna(), "failure_type"]))
    if unmapped:
        raise SystemExit(f"failure types with no coarse class in the schema: {unmapped}")
    df = df.rename(columns={"cmdb_id": "node_name", "level": "entity_level"})
    return df[["case_index", "timestamp", "utc", "utc_date", "entity_level",
               "node_name", "failure_type", "category_id", "failure_class", "split"]]


def telemetry_schema():
    # utc_date is carried as a real DATE column, not only as the directory name.
    # Hive partitioning tells a file reader which files to open, but a warehouse
    # table loaded from those files has no such column unless it is in the data,
    # and without it every time filter scans the whole table.
    return pa.schema([
        pa.field("utc_date", pa.date32(), nullable=False),
        pa.field("timestamp", pa.int64(), nullable=False),
        pa.field("node_index", pa.int16(), nullable=False),
        pa.field("channel_index", pa.int16(), nullable=False),
        pa.field("value", pa.float32(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
    ])


def write_fact_telemetry(out_dir, cache, tv, vt, limit_days=None, log=print):
    ts = cache["timestamps"].astype("int64")
    X = cache["X"]                                    # (T, 46, 130) float32
    n_t, n_nodes, n_chan = X.shape
    dates = np.array([utc_date(t) for t in ts])
    uniq = sorted(set(dates.tolist()))
    if limit_days:
        uniq = uniq[:limit_days]

    node_idx = np.tile(np.repeat(np.arange(n_nodes, dtype="int16"), n_chan), 1)
    chan_idx = np.tile(np.arange(n_chan, dtype="int16"), n_nodes)

    base = os.path.join(out_dir, "fact_telemetry")
    os.makedirs(base, exist_ok=True)
    files, total = [], 0
    for d in uniq:
        sel = np.flatnonzero(dates == d)
        block = X[sel]                                # (k, 46, 130)
        k = len(sel)
        rows = k * n_nodes * n_chan
        day = dt.date.fromisoformat(d)
        table = pa.table({
            "utc_date": pa.array([day] * rows, type=pa.date32()),
            "timestamp": pa.array(np.repeat(ts[sel], n_nodes * n_chan), type=pa.int64()),
            "node_index": pa.array(np.tile(node_idx, k), type=pa.int16()),
            "channel_index": pa.array(np.tile(chan_idx, k), type=pa.int16()),
            "value": pa.array(block.reshape(-1), type=pa.float32()),
            "split": pa.array(np.repeat(split_of(ts[sel], tv, vt), n_nodes * n_chan), type=pa.string()),
        }, schema=telemetry_schema())
        part = os.path.join(base, f"utc_date={d}")
        os.makedirs(part, exist_ok=True)
        path = os.path.join(part, "part-0.parquet")
        pq.write_table(table, path, compression=COMPRESSION,
                       compression_level=COMPRESSION_LEVEL, use_dictionary=["split"])
        size = os.path.getsize(path)
        total += rows
        files.append({"path": os.path.relpath(path, out_dir).replace(os.sep, "/"),
                      "utc_date": d, "rows": rows, "bytes": size, "sha256": sha256_file(path)})
        log(f"  utc_date={d}  {rows:>10,} rows  {size/1e6:7.1f} MB")
    return files, total


def write_table(out_dir, name, df, log=print):
    path = os.path.join(out_dir, f"{name}.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path,
                   compression=COMPRESSION, compression_level=COMPRESSION_LEVEL)
    size = os.path.getsize(path)
    log(f"  {name:<16} {len(df):>10,} rows  {size/1e3:7.1f} kB")
    return {"path": f"{name}.parquet", "rows": int(len(df)), "bytes": size,
            "sha256": sha256_file(path)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(paths.DATA, "warehouse"))
    ap.add_argument("--limit-days", type=int, default=None,
                    help="only the first N UTC dates; for a fast smoke test")
    a = ap.parse_args(argv)

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    print(f"warehouse root: {out}")

    schema, split, cache, cases = load_inputs()
    dim_split, tv, vt = build_dim_split(split)

    print("dimension and case tables:")
    manifest_tables = [
        write_table(out, "dim_node", build_dim_node(schema)),
        write_table(out, "dim_channel", build_dim_channel(schema)),
        write_table(out, "dim_split", dim_split),
        write_table(out, "fact_case", build_fact_case(cases, tv, vt, schema)),
    ]

    print("fact_telemetry, partitioned by UTC date:")
    parts, total_rows = write_fact_telemetry(out, cache, tv, vt, a.limit_days)

    manifest = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "dataset": "ARMOR D1 (Zenodo 21744694, CC-BY-4.0)",
            "feature_cache": os.path.relpath(paths.CACHE_FILE, paths.ROOT).replace(os.sep, "/"),
            "split_manifest": SPLIT_MANIFEST,
        },
        "layer": "transformed, pre-scaler",
        "note": ("Model 1 scaling is fitted on the training period only, so scaled values "
                 "are not stored here; keeping them out stops one split's statistics "
                 "leaking into shared storage."),
        "format": {"file_format": "parquet", "compression": COMPRESSION,
                   "compression_level": COMPRESSION_LEVEL,
                   "partitioning": "fact_telemetry by utc_date (Hive style)"},
        "split_boundaries": {"train_validation_ts": int(tv), "validation_test_ts": int(vt)},
        "tables": manifest_tables,
        "fact_telemetry": {"rows": int(total_rows), "partitions": parts},
        "total_bytes": int(sum(t["bytes"] for t in manifest_tables) + sum(p["bytes"] for p in parts)),
    }
    mpath = os.path.join(out, "_manifest.json")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")

    print(f"\nfact_telemetry: {total_rows:,} rows across {len(parts)} partitions")
    print(f"total on disk:  {manifest['total_bytes']/1e6:.1f} MB")
    print(f"manifest:       {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
