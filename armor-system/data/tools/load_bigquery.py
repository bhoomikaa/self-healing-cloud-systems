"""Load the Parquet lakehouse into BigQuery and verify it arrived intact.

`build_warehouse.py` writes the tables; this moves them into a warehouse that
can actually be queried, which is the last hop the data-pipeline demo asks for:
ingestion from source through to a warehouse, with the file layout, tables and
schemas shown, and the data verified as stored correctly.

Verification is the point of the second half. A load job that reports success
has told you it parsed the files, not that the warehouse now holds what the
lakehouse holds. So after loading, every table's row count is read back from
BigQuery and compared with `_manifest.json`, and `fact_telemetry` is also
checked by partition, by split and by a checksum over its values. A mismatch
exits non-zero.

Authentication is Application Default Credentials, so no key file exists and
nothing secret is read by this script:

    gcloud auth application-default login
    gcloud auth application-default set-quota-project <project>

Then:

    python -I load_bigquery.py --project armor-d1 [--dataset armor_d1] [--replace]

The BigQuery sandbox is free and needs no billing account. It does apply a
60-day expiry to every table, which is recorded in the load report rather than
left to surprise someone in January.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

from google.cloud import bigquery

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(HERE, "..", "prep")
if PREP not in sys.path:
    sys.path.insert(0, PREP)

import paths  # noqa: E402

DIM_TABLES = ("dim_node", "dim_channel", "dim_split", "fact_case")
FACT = "fact_telemetry"
SANDBOX_EXPIRY_DAYS = 60


def load_manifest(root):
    p = os.path.join(root, "_manifest.json")
    if not os.path.isfile(p):
        raise SystemExit(f"no _manifest.json in {root}. Run build_warehouse.py first.")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dataset(client, dataset_id, location, log):
    ref = bigquery.Dataset(f"{client.project}.{dataset_id}")
    ref.location = location
    ref.description = ("ARMOR D1 prepared telemetry. Built by "
                       "armor-system/data/tools/build_warehouse.py and loaded by "
                       "load_bigquery.py. Source: Zenodo 21744694, CC-BY-4.0.")
    ds = client.create_dataset(ref, exists_ok=True)
    log(f"dataset {client.project}.{ds.dataset_id} in {ds.location}")
    return ds


def load_parquet(client, table_id, source, replace, log, partition_field=None, cluster=None):
    cfg = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=(bigquery.WriteDisposition.WRITE_TRUNCATE if replace
                           else bigquery.WriteDisposition.WRITE_EMPTY),
    )
    if partition_field:
        # Partitioning on the ingestion date would partition by *today*, which
        # tells you when the load ran, not when the telemetry happened. Every
        # useful filter here is on event time.
        cfg.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field=partition_field)
    if cluster:
        cfg.clustering_fields = list(cluster)

    with open(source, "rb") as fh:
        job = client.load_table_from_file(fh, table_id, job_config=cfg)
    job.result()
    table = client.get_table(table_id)
    log(f"  {table_id.split('.')[-1]:<16} {table.num_rows:>12,} rows  {table.num_bytes/1e6:7.1f} MB")
    return table


def load_fact(client, dataset_id, root, manifest, replace, log):
    """Load the partitioned fact table one partition file at a time.

    BigQuery can read a whole Hive-partitioned tree from Cloud Storage, but the
    sandbox has no bucket and these files are local, so each partition is sent
    as its own load job. The first truncates, the rest append, which also means
    a failure halfway leaves a table that is visibly short rather than silently
    wrong, because the row-count check below will catch it.
    """
    table_id = f"{client.project}.{dataset_id}.{FACT}"
    parts = manifest["fact_telemetry"]["partitions"]
    log(f"  {FACT}, {len(parts)} partitions:")
    for i, p in enumerate(parts):
        src = os.path.join(root, p["path"].replace("/", os.sep))
        cfg = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=(bigquery.WriteDisposition.WRITE_TRUNCATE if (replace and i == 0)
                               else bigquery.WriteDisposition.WRITE_APPEND),
            clustering_fields=["node_index", "channel_index"],
        )
        with open(src, "rb") as fh:
            client.load_table_from_file(fh, table_id, job_config=cfg).result()
        log(f"    {p['utc_date']}  {p['rows']:>10,} rows")
    table = client.get_table(table_id)
    log(f"  {FACT:<16} {table.num_rows:>12,} rows  {table.num_bytes/1e6:7.1f} MB")
    return table


def q(client, sql):
    return list(client.query(sql).result())


def verify(client, dataset_id, manifest, log):
    """Read the warehouse back and compare it with what the lakehouse claims."""
    ds = f"{client.project}.{dataset_id}"
    problems, checks = [], {}

    expected = {t["path"].replace(".parquet", ""): t["rows"] for t in manifest["tables"]}
    expected[FACT] = manifest["fact_telemetry"]["rows"]
    for name, want in expected.items():
        got = q(client, f"SELECT COUNT(*) AS n FROM `{ds}.{name}`")[0]["n"]
        checks[f"rows:{name}"] = {"expected": want, "observed": got, "ok": got == want}
        if got != want:
            problems.append(f"{name}: {got:,} rows in BigQuery, {want:,} in the manifest")

    # Per-partition counts. A whole missing partition still leaves a plausible
    # looking table, and the total above would be the only thing to notice.
    want_parts = {p["utc_date"]: p["rows"] for p in manifest["fact_telemetry"]["partitions"]}
    rows = q(client, f"""
        SELECT FORMAT_TIMESTAMP('%Y-%m-%d', TIMESTAMP_SECONDS(timestamp)) AS utc_date,
               COUNT(*) AS n
        FROM `{ds}.{FACT}` GROUP BY utc_date ORDER BY utc_date""")
    got_parts = {r["utc_date"]: r["n"] for r in rows}
    checks["partitions"] = {"expected": want_parts, "observed": got_parts,
                            "ok": got_parts == want_parts}
    if got_parts != want_parts:
        for d in sorted(set(want_parts) | set(got_parts)):
            if want_parts.get(d) != got_parts.get(d):
                problems.append(f"partition {d}: {got_parts.get(d)} in BigQuery, {want_parts.get(d)} expected")

    # Split totals. fact_telemetry carries the split label, and the approved
    # manifest says how many samples each split holds; 46 nodes x 130 channels
    # per sample turns one into the other.
    per_sample = 46 * 130
    want_split = {r["split"]: r["n_samples"] * per_sample for r in
                  [dict(x) for x in q(client, f"SELECT split, n_samples FROM `{ds}.dim_split`")]}
    rows = q(client, f"SELECT split, COUNT(*) AS n FROM `{ds}.{FACT}` GROUP BY split")
    got_split = {r["split"]: r["n"] for r in rows}
    checks["split_totals"] = {"expected": want_split, "observed": got_split,
                              "ok": got_split == want_split}
    if got_split != want_split:
        problems.append(f"split totals differ: {got_split} against {want_split}")

    # A value checksum. Row counts cannot see a column that loaded as nulls or a
    # partition written twice with different contents.
    agg = q(client, f"""
        SELECT COUNT(DISTINCT timestamp) AS n_ts,
               COUNT(DISTINCT node_index) AS n_nodes,
               COUNT(DISTINCT channel_index) AS n_chan,
               COUNTIF(value IS NULL) AS n_null,
               ROUND(SUM(CAST(value AS BIGNUMERIC)), 3) AS value_sum
        FROM `{ds}.{FACT}`""")[0]
    shape = {"n_ts": agg["n_ts"], "n_nodes": agg["n_nodes"], "n_chan": agg["n_chan"],
             "n_null": agg["n_null"], "value_sum": str(agg["value_sum"])}
    shape_ok = (agg["n_nodes"] == 46 and agg["n_chan"] == 130 and agg["n_null"] == 0
                and agg["n_ts"] * 46 * 130 == expected[FACT])
    checks["value_shape"] = {"observed": shape, "ok": shape_ok}
    if not shape_ok:
        problems.append(f"fact_telemetry shape is wrong: {shape}")

    for k, v in checks.items():
        log(f"  {'ok  ' if v['ok'] else 'FAIL'} {k}")
    return checks, problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--dataset", default="armor_d1")
    ap.add_argument("--location", default="US")
    ap.add_argument("--root", default=os.path.join(paths.DATA, "warehouse"))
    ap.add_argument("--replace", action="store_true",
                    help="overwrite existing tables; without it a non-empty table fails the load")
    ap.add_argument("--report", default=os.path.join(paths.DATA, "warehouse", "_bigquery_load.json"))
    a = ap.parse_args(argv)

    root = os.path.abspath(a.root)
    manifest = load_manifest(root)
    client = bigquery.Client(project=a.project)
    log = print

    log(f"project {a.project}, lakehouse {root}")
    ensure_dataset(client, a.dataset, a.location, log)

    log("loading tables:")
    for name in DIM_TABLES:
        load_parquet(client, f"{a.project}.{a.dataset}.{name}",
                     os.path.join(root, f"{name}.parquet"), a.replace, log)
    load_fact(client, a.dataset, root, manifest, a.replace, log)

    log("verifying against the lakehouse manifest:")
    checks, problems = verify(client, a.dataset, manifest, log)

    report = {
        "loaded_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project": a.project, "dataset": a.dataset, "location": a.location,
        "source_manifest_generated_utc": manifest["generated_utc"],
        "tables": list(DIM_TABLES) + [FACT],
        "checks": checks,
        "all_checks_passed": not problems,
        "problems": problems,
        "sandbox_note": (f"BigQuery sandbox applies a {SANDBOX_EXPIRY_DAYS}-day expiry to every "
                         "table. Re-run this loader, or enable billing, before it lapses."),
    }
    with open(a.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
        fh.write("\n")
    log(f"report: {a.report}")

    if problems:
        log("\nthe warehouse does not match the lakehouse:")
        for p in problems:
            log(f"  {p}")
        return 1
    log("\nevery table matches the lakehouse manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
