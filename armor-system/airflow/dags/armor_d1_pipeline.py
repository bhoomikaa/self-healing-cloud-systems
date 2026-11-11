"""ARMOR D1 data-preparation DAG.

Wraps the existing, already-tested CPU pipeline (`armor-system/data/pipeline.py`)
as scheduled Airflow tasks. No preparation logic lives here: every task shells out
to the same commands a developer runs by hand, so the DAG and a manual run cannot
drift apart.

**The committed repository is never written to.** It is mounted read-only at
`/opt/airflow/repo` and treated as the source of truth. Each run rsyncs the code
and the committed artifacts into a disposable workspace at
`/opt/airflow/workspace`, regenerates everything there, and then diffs the result
against the committed copies. That diff is the point of the schedule: it proves
the pipeline still reproduces its artifacts. An earlier version of this DAG ran
the pipeline directly in the repository, which overwrote the team's reviewed
artifacts with each run and so proved nothing.

`data/raw` and `data/processed` live in the workspace and persist between runs,
because the 466 MB release should not be re-downloaded weekly and `fetch_d1.py`
is idempotent. Everything else in the workspace is replaced from the repository
on every run.

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

One thing the workspace costs: `pipeline.py` shells out to `git` to record the
branch and commit in its evidence, and the workspace has no `.git`, so those
fields come out empty. The DAG records the repository commit itself in the
`sync_workspace` log instead.
"""
from __future__ import annotations

import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

# plugins/ is on sys.path inside the image. Alerting lives there rather than in
# this file so it can be exercised on its own with --selftest.
import armor_alerts

# The committed repository, mounted read-only. Never written to.
REPO = "/opt/airflow/repo"
REPO_ARMOR = f"{REPO}/armor-system"

# The disposable run workspace, writable.
WS = "/opt/airflow/workspace"
WS_ARMOR = f"{WS}/armor-system"

# The pipeline runs in its own virtualenv inside the image, not in Airflow's.
# The project pins pandas 3.x and numpy 2.5.x, which Airflow does not agree with;
# keeping them apart means neither constrains the other. See Dockerfile.
PY = "/opt/pipeline-venv/bin/python"

PIPELINE = f"{WS_ARMOR}/data/pipeline.py"
TOOLS = f"{WS_ARMOR}/data/tools"
RAW = f"{WS_ARMOR}/data/raw/armor_zenodo_21744694/extracted/D1"

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

# The suites this DAG is responsible for: the data-acquisition tools, the four
# data-preparation test files, and the dashboard smoke tests. 72 tests.
#
# `armor-system/tests/prep` also holds 29 `test_m1_*.py` files, 245 tests, which
# belong to Model 1 rather than to data preparation. They need torch and
# xgboost, which the pipeline virtualenv deliberately does not install, so they
# error here rather than run. `assert_model_tests_out_of_scope` keeps that
# boundary honest instead of leaving it as a comment that rots.
OWNED_SUITES = [
    ("armor-system/tests/data", "test_*.py"),
    ("armor-system/tests/prep", "test_artifacts.py"),
    ("armor-system/tests/prep", "test_preprocess.py"),
    ("armor-system/tests/prep", "test_quality_detectors.py"),
    ("armor-system/tests/prep", "test_splits.py"),
    ("armor-system/tests/dashboard", "test_*.py"),
]

default_args = {
    "owner": "bhoomika",
    "retries": 2,
    "retry_delay": pendulum.duration(minutes=2),
    "depends_on_past": False,
    # Airflow's own email_on_failure is deliberately not used: it fails silently
    # when SMTP is unconfigured, which is how this DAG ran for days looking like
    # it had alerting while sending nothing. These callbacks always write a log
    # block and a JSON record, and additionally post to a webhook or send mail
    # when those are configured. See plugins/armor_alerts.py.
    "on_failure_callback": armor_alerts.on_failure,
    "on_retry_callback": armor_alerts.on_retry,
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="armor_d1_data_pipeline",
    description="Verify, prepare and validate the ARMOR D1 release; prove it reproduces the committed artifacts.",
    default_args=default_args,
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    schedule="0 6 * * 1",          # Mondays 06:00 UTC
    catchup=False,
    max_active_runs=1,
    tags=["armor", "d1", "data-prep", "298a"],
    sla_miss_callback=armor_alerts.on_sla_miss,
    doc_md=__doc__,
) as dag:

    start = EmptyOperator(task_id="start")

    # ------------------------------------------------------------- workspace
    # Replace the workspace's code and committed artifacts from the read-only
    # repository, while keeping the fetched release and the feature cache.
    #
    # --delete so a file deleted in the repository disappears here too, which
    # keeps the workspace an honest copy rather than an accumulating one.
    sync_workspace = BashOperator(
        task_id="sync_workspace",
        bash_command=(
            f"set -eo pipefail; "
            f"mkdir -p {WS_ARMOR}; "
            f'echo "repository commit: $(git -C {REPO} rev-parse HEAD 2>/dev/null || echo unavailable)"; '
            f'echo "repository branch: $(git -C {REPO} rev-parse --abbrev-ref HEAD 2>/dev/null || echo unavailable)"; '
            f"rsync -a --delete "
            f"--exclude 'data/raw/' --exclude 'data/processed/' "
            f"--exclude '__pycache__/' --exclude '*.pyc' "
            f"{REPO_ARMOR}/ {WS_ARMOR}/; "
            f"mkdir -p {WS_ARMOR}/data/raw {WS_ARMOR}/data/processed; "
            # Seed the release from the checkout the first time, if it has one.
            # data/raw is Git-ignored local data, so a fresh clone will not, and
            # fetch_source then downloads it. Either way fetch_source re-checks
            # CRC32 and size, so seeding is a shortcut, not a trust decision.
            f"if [ -z \"$(ls -A {WS_ARMOR}/data/raw 2>/dev/null)\" ] && "
            f"   [ -n \"$(ls -A {REPO_ARMOR}/data/raw 2>/dev/null)\" ]; then "
            f'  echo "seeding data/raw from the checkout (first run)"; '
            f"  cp -r {REPO_ARMOR}/data/raw/. {WS_ARMOR}/data/raw/; "
            f"fi; "
            f'echo "workspace synced: $(du -sh {WS_ARMOR} | cut -f1)"'
        ),
        doc_md=(
            "Refresh the disposable workspace from the read-only repository. "
            "`data/raw` and `data/processed` are preserved; everything else is "
            "replaced. The committed repository is never written to."
        ),
    )

    # ---------------------------------------------------------------- ingest
    # D1 is a static published release. "Ingestion" is a verified, idempotent
    # fetch: fetch_d1.py re-downloads only what is missing and checks CRC32 and
    # size against the official central directory.
    #
    # --manifest-dir points at a scratch directory so a run can never rewrite
    # the committed evidence manifests (START_HERE records that fetch_d1.py
    # rewrites them in place when run with defaults).
    fetch_source = BashOperator(
        task_id="fetch_source",
        bash_command=(
            f"set -eo pipefail; "
            f"mkdir -p /tmp/d1_fetch_manifests && "
            f"cd {WS} && "
            f"{PY} -I {TOOLS}/fetch_d1.py --manifest-dir /tmp/d1_fetch_manifests"
        ),
        doc_md="Idempotent fetch of the D1 release from Zenodo into the workspace. Skips bytes already present.",
    )

    # Re-hash every extracted file against the committed inventory. Exits
    # non-zero on any MISMATCH, so a corrupted or tampered release stops the
    # DAG here rather than silently feeding the pipeline.
    verify_source = BashOperator(
        task_id="verify_source",
        bash_command=f"set -eo pipefail; {PY} -I {TOOLS}/validate_d1.py --raw {RAW}",
        doc_md="Checksum gate. Re-hashes the extracted release against `file_inventory.csv`.",
    )

    # ------------------------------------------------------------ preparation
    previous = verify_source
    for task_id, stage, doc in PREP_STAGES:
        task = BashOperator(
            task_id=task_id,
            bash_command=f"set -eo pipefail; cd {WS} && {PY} {PIPELINE} --stage {stage}",
            doc_md=f"{doc}\n\nRuns `pipeline.py --stage {stage}` in the workspace.",
        )
        previous >> task
        previous = task
    last_prep = previous

    # ----------------------------------------------------------------- tests
    # A real gate. `pipeline.py --stage tests` records results and returns
    # normally even when suites fail, so using it as a gate gave a task that
    # could never go red; this runs unittest directly and lets its exit code
    # decide.
    _suite_cmds = " ".join(
        f'echo "--- {d} ({p}) ---"; {PY} -I -m unittest discover -s {d} -p "{p}" || FAILED=1; '
        for d, p in OWNED_SUITES
    )
    run_tests = BashOperator(
        task_id="run_tests",
        bash_command=(
            f"cd {WS}; FAILED=0; {_suite_cmds} "
            f'if [ "$FAILED" -ne 0 ]; then echo "FAIL: a suite this DAG owns did not pass"; exit 1; fi; '
            f'echo "PASS: every suite this DAG owns passed"'
        ),
        doc_md=(
            "The 72 tests this DAG owns: D1 acquisition tools, the four "
            "data-preparation test files, and the dashboard smoke tests. Fails "
            "the task if any suite fails. Model tests are excluded on purpose; "
            "see `assert_model_tests_out_of_scope`."
        ),
    )

    # Make the test boundary active rather than a comment. The 29 `test_m1_*.py`
    # files need torch and xgboost. The pipeline virtualenv installs neither, so
    # they cannot run here and are covered by the Model 1 serving container
    # instead. If either package ever appears in this environment, this task
    # goes red to say the gate above should be widened to include them.
    assert_model_tests_out_of_scope = BashOperator(
        task_id="assert_model_tests_out_of_scope",
        bash_command=(
            f"set -eo pipefail; "
            f"{PY} -I -c \""
            f"import importlib.util as u, sys; "
            f"present=[m for m in ('torch','xgboost') if u.find_spec(m)]; "
            f"print('model-test dependencies present:', present or 'none'); "
            f"sys.exit(0) if not present else (print('These are now installed, so the 245 model "
            f"tests in armor-system/tests/prep/test_m1_*.py can run here. Widen OWNED_SUITES in "
            f"the DAG to include them.') or sys.exit(1))\""
        ),
        retries=0,
        doc_md=(
            "Asserts that torch and xgboost are absent, which is why the 245 "
            "Model 1 tests are out of scope for this DAG. Goes red if that "
            "changes, so the exclusion cannot quietly become a blind spot."
        ),
    )

    # ------------------------------------------------------- negative control
    # Course requirement 3B: "Show one sanity check failing when you deliberately
    # corrupt an input."
    #
    # This copies the release to a scratch directory, truncates one file there,
    # and asserts the checksum gate REJECTS it. The task succeeds only when the
    # validator fails, so a validator that silently passed everything would turn
    # this task red. The real release is never written to.
    corruption_gate = BashOperator(
        task_id="corruption_gate",
        bash_command=(
            f"set -e; "
            f"WORK=$(mktemp -d); "
            f"cp -r {RAW} $WORK/D1; "
            # validate_d1.py WRITES d1_validation.json into its --manifest-dir,
            # and that happens whether the release passed or failed. Without
            # this scratch copy it writes the corrupted verdict over the real
            # report, which then looks like genuine corruption of
            # cases/ad_cases.pkl: the alphabetically first file, so the one the
            # victim search below always picks.
            f"mkdir -p $WORK/manifests; "
            f"cp {WS_ARMOR}/data/manifests/armor_d1/d1_inspection.json "
            f"   {WS_ARMOR}/data/manifests/armor_d1/file_inventory.csv $WORK/manifests/; "
            f"VICTIM=$(find $WORK/D1 -type f | sort | head -1); "
            f'echo "corrupting: $VICTIM"; '
            f"printf 'CORRUPTED' >> $VICTIM; "
            f"if {PY} -I {TOOLS}/validate_d1.py --manifest-dir $WORK/manifests --raw $WORK/D1; then "
            f'  echo "FAIL: the validator accepted a corrupted release"; rm -rf $WORK; exit 1; '
            f"else "
            f'  echo "PASS: corrupted release rejected, as required"; rm -rf $WORK; exit 0; '
            f"fi"
        ),
        retries=0,  # a negative control must not be retried into passing
        doc_md=(
            "Deliberate-corruption negative control. Succeeds only if the checksum "
            "gate **rejects** a corrupted copy. Both the release and the manifest "
            "directory are scratch copies, because `validate_d1.py` writes its "
            "report into `--manifest-dir` on every run, pass or fail."
        ),
    )

    # -------------------------------------------------------------- evidence
    publish_evidence = BashOperator(
        task_id="publish_evidence",
        bash_command=f"set -eo pipefail; cd {WS} && {PY} {PIPELINE} --stage evidence",
        doc_md=(
            "Writes the run's evidence JSON in the workspace. Its `git` fields are "
            "empty because the workspace is not a checkout; `sync_workspace` logs "
            "the repository commit. Its test counts include the out-of-scope model "
            "tests, so `run_tests` is the gate, not this."
        ),
    )

    # ------------------------------------------------------- reproducibility
    # The reason the schedule exists. Byte-compare every artifact the run
    # regenerated against the committed copy and fail on any difference.
    compare_artifacts = BashOperator(
        task_id="compare_artifacts",
        bash_command=(
            f"set -eo pipefail; "
            f"{PY} -I {REPO_ARMOR}/airflow/scripts/compare_artifacts.py "
            f"--committed {REPO_ARMOR} --generated {WS_ARMOR} "
            f"--report {WS}/reports/artifact_drift_{{{{ ts_nodash }}}}.json"
        ),
        retries=0,  # drift is a finding, not a flake
        doc_md=(
            "Byte-compares the regenerated artifacts against the committed ones "
            "and fails on any drift. The run's evidence directory is excluded "
            "because it records when and where the run happened. A drift report "
            "is written under `/opt/airflow/workspace/reports/`."
        ),
    )

    end = EmptyOperator(task_id="end")

    start >> sync_workspace >> fetch_source >> verify_source
    last_prep >> [run_tests, assert_model_tests_out_of_scope, corruption_gate]
    [run_tests, assert_model_tests_out_of_scope, corruption_gate] >> publish_evidence
    publish_evidence >> compare_artifacts >> end
