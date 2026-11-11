# Linear tickets for Bhoomika — 298A Checkpoint 3

Every task shown in the demo must exist in Linear as a closed issue with a named
assignee and a linked artifact. Work that is not in Linear does not count as
individual contribution.

Assignee: Bhoomika. Cycle 6 (11/5 – 11/11) unless noted.
Repository: https://github.com/sjsu-298a-team1/self-healing-cloud-systems

---

## 1. Local Airflow orchestration for the ARMOR D1 pipeline
Wrap the committed data pipeline as scheduled Airflow tasks: LocalExecutor and
Postgres in Docker, one task per `pipeline.py --stage`, retries, per-stage logs,
and a deliberate-corruption negative control.
**Artifacts:** `015f958`, `66f93ed` · `armor-system/airflow/`

## 2. Stop the DAG writing to the committed checkout
A scheduled run overwrote 33 reviewed artifacts because the repository was
mounted read-write. Mount it read-only, regenerate into a disposable workspace,
and diff the result against the original. Also replaced a test gate that could
never fail, and stopped the negative control overwriting the real validation
report.
**Artifacts:** `5ebfe50`, `49450dd`, `e3515dd` · 24 structure tests

## 3. Containerise the Model 1 v2 inference service
Dockerfile and compose for the serving API. Checkpoints mounted at run time and
hash-checked against the frozen test evidence, never baked into a layer. Image
slimmed from 3.1 GB to 321 MB by using the CPU torch wheel; verified
byte-identical predictions across a full 875-minute replay.
**Artifacts:** `427ea6a`, `6469f10`, `6f41446` · `results/container_verification.json`

## 4. Materialise the prepared data as a Parquet lakehouse
Star schema over the prepared D1 data: `dim_node`, `dim_channel`, `dim_split`,
`fact_case`, and `fact_telemetry` at 43,056,000 rows, Hive-partitioned by UTC
date, zstd, 30.6 MB.
**Artifacts:** `1d45e62` · `armor-system/data/tools/build_warehouse.py`

## 5. Load the lakehouse into BigQuery with verification
Load five tables into the BigQuery sandbox and verify against the lakehouse
manifest: row counts per table and per partition, split totals derived
independently, and a value checksum. Found that sandbox partition expiry
silently dropped every 2022-dated partition; clustered on timestamp instead.
**Artifacts:** `caa1a96`, `57bff72` · `armor-system/data/warehouse/_bigquery_load.json`

## 6. Deploy the Model 1 prototype
Public Streamlit app reading the BigQuery warehouse live and scoring with the
five-seed ensemble in process. Audience picks the input; nothing precomputed.
**Artifacts:** `57bff72`, `64da31d`, `3651cf2` · https://armor-model1.streamlit.app

## 7. Failure alerting for the DAG
Alerting was configured but sent nothing: SMTP sat at localhost:25. Four sinks
with log and file always on, webhook and email when configured, and an
unconfigured sink now says so. Verified on a real induced failure.
**Artifacts:** `874b5d2` · `armor-system/airflow/plugins/armor_alerts.py`

## 8. Architecture diagrams and training record
Data-flow and model diagrams in Mermaid, environment and trust-boundary tables,
and an honest training record: per-epoch history was never logged, so learning
curves cannot be produced without breaking the freeze.
**Artifacts:** `c8f6c31` · `armor-system/docs/architecture.md`
