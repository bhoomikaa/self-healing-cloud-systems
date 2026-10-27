"""Structure tests for the ARMOR D1 DAG.

These run without Docker, without Airflow's metadata database and without the D1
release: they only parse the DAG file and assert its shape. A broken DAG is
otherwise discovered by the scheduler at run time, which is a slow way to find a
typo.

What is NOT covered here: that the wrapped commands succeed. That is the
pipeline's own suites (`armor-system/tests/{data,prep}`) plus a real DAG run.

    python -m unittest discover -s armor-system/airflow/tests -p 'test_*.py'
"""
import os
import unittest

DAGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dags")
DAG_ID = "armor_d1_data_pipeline"

try:
    from airflow.models.dagbag import DagBag
    AIRFLOW = True
except ImportError:                                     # pragma: no cover
    AIRFLOW = False


@unittest.skipUnless(AIRFLOW, "apache-airflow is not installed in this environment")
class TestArmorD1Dag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # include_examples=False keeps Airflow's bundled example DAGs out of the bag.
        cls.bag = DagBag(dag_folder=DAGS_DIR, include_examples=False)

    def test_dag_file_imports_without_error(self):
        self.assertEqual(self.bag.import_errors, {}, f"DAG import errors: {self.bag.import_errors}")

    def test_dag_is_registered(self):
        self.assertIn(DAG_ID, self.bag.dags, f"found instead: {sorted(self.bag.dags)}")

    def test_expected_tasks_exist(self):
        dag = self.bag.dags[DAG_ID]
        expected = {
            "start", "fetch_source", "verify_source",
            "build_feature_cache", "inspect", "splits", "preprocess",
            "eda", "figures", "augmentation", "quality_gate",
            "run_tests", "corruption_gate", "publish_evidence", "end",
        }
        self.assertEqual(set(dag.task_ids), expected)

    def test_preparation_runs_in_pipeline_order(self):
        """Stage order is a correctness property: preprocess must not precede splits."""
        dag = self.bag.dags[DAG_ID]
        chain = ["verify_source", "build_feature_cache", "inspect", "splits",
                 "preprocess", "eda", "figures", "augmentation", "quality_gate"]
        for upstream, downstream in zip(chain, chain[1:]):
            self.assertIn(downstream, dag.get_task(upstream).downstream_task_ids,
                          f"{downstream} must run after {upstream}")

    def test_source_is_verified_before_anything_is_built(self):
        """The checksum gate must sit between the fetch and the first build step."""
        dag = self.bag.dags[DAG_ID]
        self.assertIn("verify_source", dag.get_task("fetch_source").downstream_task_ids)
        self.assertIn("build_feature_cache", dag.get_task("verify_source").downstream_task_ids)

    def test_corruption_gate_is_never_retried(self):
        """A negative control that retries can pass by accident. It must not retry."""
        dag = self.bag.dags[DAG_ID]
        self.assertEqual(dag.get_task("corruption_gate").retries, 0)

    def test_corruption_gate_does_not_touch_the_real_release(self):
        """The command must work on a scratch copy, never write under data/raw/."""
        cmd = self.bag.dags[DAG_ID].get_task("corruption_gate").bash_command
        self.assertIn("mktemp -d", cmd)
        self.assertNotIn("printf 'CORRUPTED' >> $SRC", cmd)

    def test_fetch_does_not_rewrite_committed_manifests(self):
        """fetch_d1.py rewrites the committed manifests when run with defaults."""
        cmd = self.bag.dags[DAG_ID].get_task("fetch_source").bash_command
        self.assertIn("--manifest-dir", cmd)

    def test_tasks_have_retries_configured(self):
        dag = self.bag.dags[DAG_ID]
        for tid in ("build_feature_cache", "preprocess", "quality_gate"):
            self.assertGreater(dag.get_task(tid).retries, 0, f"{tid} should be retryable")

    def test_dag_does_not_backfill(self):
        dag = self.bag.dags[DAG_ID]
        self.assertFalse(dag.catchup, "catchup would replay runs for a static dataset")
        self.assertEqual(dag.max_active_runs, 1, "concurrent runs would write the same artifacts")


if __name__ == "__main__":
    unittest.main()
