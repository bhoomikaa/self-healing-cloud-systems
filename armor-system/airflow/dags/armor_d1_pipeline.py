"""ARMOR D1 data-preparation DAG.

Wraps the existing, already-tested CPU pipeline (`armor-system/data/pipeline.py`)
as scheduled Airflow tasks. No preparation logic lives here: every task shells out
to the same commands a developer runs by hand, so the DAG and a manual run cannot
drift apart.

Task boundaries follow `pipeline.py --stage`, which runs every stage *up to and
including* the one named. Stages hand state to each other in memory (the split
result and the fitted scalers are Python objects, not files that can be reloaded),
so a task cannot resume from the previous task's output. Each task therefore
recomputes its upstream stages. That is a deliberate, documented cost: D1 is a
static 62 MB release and a full run is minutes, and the benefit is a real retry
point and a separate log per stage. Removing the recompute needs artifact-reload
helpers in `prep/` (see README, "Known cost").

Schedule is paused by default and set to weekly: D1 is a fixed published release,
so there is nothing new to ingest. The schedule exists to prove the pipeline still
reproduces its committed artifacts, not to pick up new data.
"""
from __future__ import annotations

import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

REPO = "/opt/airflow/repo"
# The pipeline runs in its own virtualenv inside the image, not in Airflow's.
# The project pins pandas 3.x and numpy 2.5.x, which Airflow does not agree with;
# keeping them apart means neither constrains the other. See Dockerfile.
PY = "/opt/pipeline-venv/bin/python"
PIPELINE = f"{REPO}/armor-system/data/pipeline.py"
TOOLS = f"{REPO}/armor-system/data/tools"

# Stages of pipeline.py that this DAG exposes as tasks, in order.
# "tests" and "evidence" are handled separately below.
PREP_STAGES = [
    ("build_feature_cache", "cache", "Decode the verified D1 release into the float32 feature cache."),
    ("inspect", "inspect", "Schema, counts, class distribution and leakage surface."),
    ("splits", "splits", "Regenerate the split manifests and re-run their hard checks."),
    ("preprocess", "preprocess", "Fit the train-only scaler and write the preprocessing config."),
    ("eda", "eda", "Exploratory statistics over the fitting period only."),
    ("figures", "figures", "Render the committed EDA figures."),
    ("augmentation", "augmentation", "Re-run the augmentation assessment (assessed, not adopted)."),
    ("quality_gate", "quality", "Nine-stage validation, including the deliberate-corruption cases."),
]

default_args = {
    "owner": "bhoomika",
    "retries": 2,
    "retry_delay": pendulum.duration(minutes=2),
    "depends_on_past": False,
}

with DAG(
    dag_id="armor_d1_data_pipeline",
    description="Verify, prepare and validate the ARMOR D1 release; reproduce the committed artifacts.",
    default_args=default_args,
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    schedule="0 6 * * 1",          # Mondays 06:00 UTC
    catchup=False,
    max_active_runs=1,
    tags=["armor", "d1", "data-prep", "298a"],
    doc_md=__doc__,
) as dag:

    start = EmptyOperator(task_id="start")

    # ---------------------------------------------------------------- ingest
    # D1 is a static published release. "Ingestion" is a verified, idempotent
    # fetch: fetch_d1.py re-downloads only what is missing and checks CRC32 and
    # size against the official central directory.
    #
    # --manifest-dir points at a scratch directory so a scheduled run can never
    # rewrite the committed evidence manifests (START_HERE records that
    # fetch_d1.py rewrites them in place when run with defaults).
    fetch_source = BashOperator(
        task_id="fetch_source",
        bash_command=(
            f"mkdir -p /tmp/d1_fetch_manifests && "
            f"{PY} -I {TOOLS}/fetch_d1.py --manifest-dir /tmp/d1_fetch_manifests"
        ),
        doc_md="Idempotent fetch of the D1 release from Zenodo. Skips bytes already present.",
    )

    # Re-hash every extracted file against the committed inventory. Exits
    # non-zero on any MISMATCH, so a corrupted or tampered release stops the
    # DAG here rather than silently feeding the pipeline.
    verify_source = BashOperator(
        task_id="verify_source",
        bash_command=(
            f"{PY} -I {TOOLS}/validate_d1.py "
            f"--raw {REPO}/armor-system/data/raw/armor_zenodo_21744694/extracted/D1"
        ),
        doc_md="Checksum gate. Re-hashes the extracted release against `file_inventory.csv`.",
    )

    # ------------------------------------------------------------ preparation
    previous = verify_source
    for task_id, stage, doc in PREP_STAGES:
        task = BashOperator(
            task_id=task_id,
            bash_command=f"{PY} {PIPELINE} --stage {stage}",
            doc_md=f"{doc}\n\nRuns `pipeline.py --stage {stage}`.",
        )
        previous >> task
        previous = task
    last_prep = previous

    # ----------------------------------------------------------------- tests
    run_tests = BashOperator(
        task_id="run_tests",
        bash_command=f"{PY} {PIPELINE} --stage tests",
        doc_md="All committed suites (data acquisition, preparation, dashboard).",
    )

    # ------------------------------------------------------- negative control
    # Course requirement 3B: "Show one sanity check failing when you deliberately
    # corrupt an input."
    #
    # This copies the release to a scratch directory, truncates one file there,
    # and asserts the checksum gate REJECTS it. The task succeeds only when the
    # validator fails, so a validator that silently passed everything would turn
    # this task red. The real release under data/raw/ is never written to.
    corruption_gate = BashOperator(
        task_id="corruption_gate",
        bash_command=(
            f"set -e; "
            f"SRC={REPO}/armor-system/data/raw/armor_zenodo_21744694/extracted/D1; "
            f"WORK=$(mktemp -d); "
            f"cp -r $SRC $WORK/D1; "
            f"VICTIM=$(find $WORK/D1 -type f | sort | head -1); "
            f'echo "corrupting: $VICTIM"; '
            f"printf 'CORRUPTED' >> $VICTIM; "
            f"if {PY} -I {TOOLS}/validate_d1.py --raw $WORK/D1; then "
            f'  echo "FAIL: the validator accepted a corrupted release"; rm -rf $WORK; exit 1; '
            f"else "
            f'  echo "PASS: corrupted release rejected, as required"; rm -rf $WORK; exit 0; '
            f"fi"
        ),
        retries=0,  # a negative control must not be retried into passing
        doc_md=(
            "Deliberate-corruption negative control. Succeeds only if the checksum "
            "gate **rejects** a corrupted copy. Operates on a scratch copy; the real "
            "release is never modified."
        ),
    )

    # -------------------------------------------------------------- evidence
    publish_evidence = BashOperator(
        task_id="publish_evidence",
        bash_command=f"{PY} {PIPELINE} --stage evidence",
        doc_md="Reproducibility check and evidence JSON for the run.",
    )

    end = EmptyOperator(task_id="end")

    start >> fetch_source >> verify_source
    last_prep >> [run_tests, corruption_gate] >> publish_evidence >> end
