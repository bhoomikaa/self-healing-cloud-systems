"""Tests of the attempt-2 runner fix (TEST_INCIDENT_attempt1.md): worker output paths in test mode, attempt-2 markers, and the
byte-for-byte comparison with the aborted attempt 1. Synthetic paths and temporary markers only: no test window, test label or
official label is read, and no test run can start (every test-mode call stops at a marker or token refusal before the freeze check).
"""
import os
import shutil
import tempfile
import unittest
from unittest import mock

import _m1_model_helpers as MH  # noqa: F401

from model1_transformer import run_test as rt, test_freeze as tf


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


class OutputPathTests(unittest.TestCase):
    def test_test_mode_accepts_the_test_directory_and_its_subfolders_only(self):
        self.assertTrue(rt.out_ok("test", tf.TEST_OUT))
        self.assertTrue(rt.out_ok("test", os.path.join(tf.TEST_OUT, "xgb_features")))
        self.assertTrue(rt.out_ok("test", os.path.join(tf.TEST_OUT, "a", "b")))
        self.assertFalse(rt.out_ok("test", rt.ATTEMPT1_DIR))                              # a sibling sharing the prefix is not inside
        self.assertFalse(rt.out_ok("test", tempfile.gettempdir()))

    def test_validation_mode_never_writes_into_the_test_directory(self):
        self.assertFalse(rt.out_ok("validation", tf.TEST_OUT))
        self.assertFalse(rt.out_ok("validation", os.path.join(tf.TEST_OUT, "xgb_features")))
        self.assertTrue(rt.out_ok("validation", tempfile.gettempdir()))
        self.assertTrue(rt.out_ok("validation", rt.ATTEMPT1_DIR))


class TestModeWorkerTests(unittest.TestCase):
    """Every worker the orchestrator launches in test mode passes the path check and reaches the token gate (the attempt-1 defect:
    the XGBoost-features worker was refused by the path check with SystemExit instead)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        for d, _, fs in os.walk(self.tmp):
            for f in fs:
                os.chmod(os.path.join(d, f), 0o644)
        shutil.rmtree(self.tmp)

    def specs(self):
        return rt.worker_specs("test", tf.TEST_OUT, 10, rt.SEEDS)

    def test_the_orchestrator_launches_seven_workers_with_paths_inside_the_test_directory(self):
        s = self.specs()
        self.assertEqual(sorted(s), sorted([f"missing_{i}" for i in rt.SEEDS] + ["xgb_features", "xgb_train"]))
        for name, args in s.items():
            for flag in ("--out", "--features"):
                if flag in args:
                    self.assertTrue(rt.out_ok("test", args[args.index(flag) + 1]), (name, flag))

    def test_every_test_mode_worker_reaches_the_token_gate_and_is_refused_without_its_markers(self):
        markers = (os.path.join(self.tmp, "m1.json"), os.path.join(self.tmp, "m2.json"))      # do not exist
        with mock.patch.object(rt, "MARKERS", markers):
            for name, args in self.specs().items():
                with self.assertRaisesRegex(tf.FreezeError, "marker missing", msg=name):
                    rt.main(args + ["--run-id", "synthetic"])
        self.assertFalse(os.path.exists(os.path.join(tf.TEST_OUT, "xgb_features")))

    def test_a_worker_with_another_run_id_is_refused(self):
        markers = (os.path.join(self.tmp, "m1.json"), os.path.join(self.tmp, "m2.json"))
        for m in markers:
            write(m, b'{"run_id": "the-real-run"}')
        with mock.patch.object(rt, "MARKERS", markers):
            with self.assertRaisesRegex(tf.FreezeError, "another run"):
                rt.main(self.specs()["xgb_features"] + ["--run-id", "synthetic"])

    def test_attempt2_markers_are_new_names_and_an_existing_one_refuses_a_start(self):
        self.assertTrue(set(rt.MARKERS).isdisjoint({tf.REPO_MARKER, tf.EXTERNAL_MARKER}))
        self.assertEqual(os.path.dirname(rt.MARKERS[0]), tf.TEST_OUT)
        self.assertEqual(os.path.dirname(rt.MARKERS[1]), tf.CKPT_ROOT)
        markers = (os.path.join(self.tmp, "m1.json"), os.path.join(self.tmp, "m2.json"))
        for m in markers:
            write(m, b"{}")
        with self.assertRaises(tf.FreezeError):                                          # marker present (or test_v1 already in use): refused
            rt.start_test_run(markers)

    def test_a_non_empty_output_directory_refuses_a_start(self):
        out = os.path.join(self.tmp, "test_v1")
        write(os.path.join(out, "leftover.json"), b"{}")
        markers = (os.path.join(self.tmp, "m1.json"), os.path.join(self.tmp, "m2.json"))
        for m in markers:
            write(m, b"{}")                                                              # belt and braces: the marker check also refuses
        with mock.patch.object(tf, "TEST_OUT", out):
            with self.assertRaisesRegex(tf.FreezeError, "not empty"):
                rt.start_test_run(markers)


class Attempt1ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.a, self.b = os.path.join(self.tmp, "run"), os.path.join(self.tmp, "archive")
        for s in range(4):
            for f in rt.ATTEMPT1_FILES:
                for d in (self.a, self.b):
                    write(os.path.join(d, f"seed_{s}", f), f"{s}{f}".encode())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_identical_files_are_recorded_as_identical(self):
        r = rt.compare_attempt1(self.a, self.b)
        self.assertTrue(r["all_identical"])
        self.assertEqual(len(r["files"]), 12)

    def test_one_changed_byte_or_a_missing_file_is_recorded(self):
        write(os.path.join(self.a, "seed_2", "result.json"), b"2result.jsoN")
        os.remove(os.path.join(self.a, "seed_3", "predictions_v0.csv"))
        r = rt.compare_attempt1(self.a, self.b)
        self.assertFalse(r["all_identical"])
        bad = sorted(k for k, v in r["files"].items() if not v["identical"])
        self.assertEqual(bad, ["seed_2/result.json", "seed_3/predictions_v0.csv"])
        self.assertIsNone(r["files"]["seed_3/predictions_v0.csv"]["sha256_this_run"])

    def test_the_committed_attempt1_archive_holds_the_twelve_seed_files_and_both_markers(self):
        r = rt.compare_attempt1(rt.ATTEMPT1_DIR, rt.ATTEMPT1_DIR)
        self.assertTrue(r["all_identical"])
        self.assertEqual(tf.sha(os.path.join(rt.ATTEMPT1_DIR, "TEST_RUN_MARKER.json")), tf.sha(os.path.join(rt.ATTEMPT1_DIR, "EXTERNAL_MARKER_copy_model1_test_run_marker.json")))


if __name__ == "__main__":
    unittest.main()
