# Airflow orchestration (local)

Wraps the existing ARMOR D1 data-preparation pipeline as scheduled Airflow tasks.

**State: local development only.** This runs on one machine in Docker. Nothing is
deployed to a cloud, because the provider, object storage, credentials policy and
budget are still undecided (`docs/team_handoff_readiness_audit.md`, section 7).
Everything here is intended to lift to a managed Airflow later without rewriting
the DAG.

## Why Docker

Airflow is not supported natively on Windows. Docker is the supported route here,
not a preference.

## What the DAG does

`dags/armor_d1_pipeline.py`, DAG id `armor_d1_data_pipeline`.

```
start → fetch_source → verify_source → build_feature_cache → inspect → splits
      → preprocess → eda → figures → augmentation → quality_gate
      → ├─ run_tests        ─┐
        └─ corruption_gate  ─┴→ publish_evidence → end
```

| Task | What it does |
|---|---|
| `fetch_source` | Idempotent fetch of the D1 release from Zenodo. Re-downloads only missing bytes. Writes its manifests to a scratch directory so a scheduled run cannot rewrite the committed ones. |
| `verify_source` | Re-hashes every extracted file against `file_inventory.csv`. Non-zero exit on any mismatch, so a corrupted release stops the run here. |
| `build_feature_cache` … `quality_gate` | One task per `pipeline.py --stage`. Same commands a developer runs by hand. |
| `run_tests` | The committed suites (data acquisition, preparation, dashboard). |
| `corruption_gate` | Negative control. Corrupts a **scratch copy** and asserts the checksum gate rejects it. Succeeds only when validation fails. |
| `publish_evidence` | Reproducibility check and evidence JSON. |

The DAG contains no preparation logic. Every task shells out to a committed
script, so the DAG and a manual run cannot drift apart.

## Known cost: tasks recompute their upstream stages

`pipeline.py --stage X` runs every stage **up to and including** X. Stages pass
state to each other in memory (the split result and the fitted scalers are Python
objects, not files that can be reloaded), so a task cannot resume from the
previous task's output.

Measured on one laptop, D1 present and the feature cache already built:

| Command | Wall time |
|---|---|
| `--stage inspect` | 11 s |
| `--stage splits` | 10 s |
| `--stage preprocess` | 31 s |
| `--stage eda` | 86 s |
| `--stage quality` | 98 s |
| First run including cache build | 123 s |

So the granular DAG costs roughly **7–8 minutes** per run against about **1.5
minutes** for a single `--stage quality` task. That is the price of a separate
log and a separate retry point per stage, and it is paid against a static 62 MB
dataset, so it was judged worth it.

To collapse it, replace the eight preparation tasks with one task running
`--stage quality`. To remove the cost properly, `prep/` needs loaders that
rebuild the split result and the scalers from the committed artifacts; then each
task could run exactly one stage. That is follow-up work, not done here.

## Setup

Requires Docker Desktop running.

```bash
cd armor-system/airflow
cp .env.example .env          # then set AIRFLOW_UID on Linux/macOS: id -u
mkdir -p logs plugins
docker compose build
docker compose up -d
```

Open http://localhost:8080. Default credentials come from `.env`
(`airflow` / `airflow` unless changed). The DAG is **paused on creation**:
unpause it in the UI, or trigger a single run without scheduling it.

```bash
docker compose logs -f airflow-scheduler    # follow
docker compose down                         # stop
docker compose down -v                      # stop and drop the metadata DB
```

### The data has to be there

`fetch_source` downloads D1 into `armor-system/data/raw/`, which is Git-ignored
and mounted into the container. The first run takes a few minutes on a cold
cache. Later runs skip bytes already present.

## Tests

Structure tests only. They parse the DAG and assert its shape; they do not run
the wrapped commands and do not need Docker, a database or the dataset.

```bash
python -m unittest discover -s armor-system/airflow/tests -p 'test_*.py'
```

They skip cleanly when `apache-airflow` is not installed, so the other suites are
unaffected on a machine without it.

## Design notes

**Pipeline dependencies are isolated.** The project pins pandas 3.x and numpy
2.5.x; Airflow constrains both. Installing them together resolves to versions
neither side tested. The image therefore builds a separate virtualenv at
`/opt/pipeline-venv` holding only the pipeline's dependencies, and the DAG calls
that interpreter. Keep `requirements-pipeline.txt` in step with
`dashboard/requirements.txt` so a DAG run and a dashboard run see the same
numbers.

**Python version.** The project was tested on Python 3.14. This image is Python
3.12, the newest Airflow 2.10.5 supports. That gap is a real difference between a
DAG run and a developer's laptop and should be closed by pinning the team to one
version.

**Why the pins matter, measured.** Running the pipeline on this machine against
unpinned libraries (numpy 2.3.1, pandas 2.3.1, scipy 1.16.2, matplotlib 3.10.3
instead of the pinned 2.5.3 / 3.0.6 / 1.18.1 / 3.11.2) reproduced every stage and
every check, but 19 committed artifacts came back differing in the last one or two
decimal places, for example a correlation of `0.1517462410902621` against
`0.15174624109026205`. The figures differed too.

That is floating-point noise from different library builds, not pipeline
nondeterminism, and it does not contradict the project's byte-identical
reproducibility result, which was measured inside the pinned environment. But it
does mean **artifact hashes only reproduce on the pinned versions**, so a DAG
that installed "numpy" unpinned would produce a different evidence hash every
time the image was rebuilt. Hence the exact pins in
`requirements-pipeline.txt`.

**Schedule.** Weekly, Mondays 06:00 UTC, paused by default. D1 is a fixed
published release, so there is nothing new to ingest. The schedule exists to
prove the pipeline still reproduces its committed artifacts, not to pick up new
data. `catchup` is off and `max_active_runs` is 1, because concurrent runs would
write the same artifact paths.

**Alerts.** Failures surface in the UI and the task logs. Email delivery needs
the SMTP variables in `.env`; without them nothing is sent and nothing breaks.

## Not done

- No cloud deployment, no managed Airflow, no object storage or warehouse. Those
  need the decisions in the readiness audit, section 7.
- No ingestion of anything other than D1. D2 and raw telemetry stay deferred.
- No model training, inference or serving. Model 1 does not exist yet, so there
  is no inference task to schedule.
- Not run on Linux, and not run against a GPU.
- The DAG has been parsed and structurally tested, and every wrapped command has
  been run by hand on this dataset, but a full scheduled run inside Docker has
  not been executed end to end yet.
