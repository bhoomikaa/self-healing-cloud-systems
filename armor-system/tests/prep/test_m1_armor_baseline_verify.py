"""Baseline-only verification of ARMOR's published detection metrics (scoring harness, Linear 298-51)."""
import ast
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import _m1_helpers as H

DET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "detection"))
if DET not in sys.path:
    sys.path.insert(0, DET)

import verify_armor_published_ad as v  # noqa: E402

HAVE = all(os.path.isfile(p) for p in (v.INTERVAL_FILE, v.LABEL_FILE, v.RES_JSON, v.SPLIT_FILE, v.ACQ_MANIFEST))
NEEDS_FILES = H.needs(HAVE, "needs ARMOR's released interval file and the D1 labels under armor-system/data/raw (git-ignored)")
ALLOWED_KEYS = {"verification", "scope", "status", "reasons", "precision", "recall", "f1", "published", "provenance", "matches_published", "original_eval_AD_agrees"}


@NEEDS_FILES
class PublishedReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = v.verify()

    def test_reproduces_the_published_precision_recall_f1(self):
        self.assertEqual(self.r["status"], "PASS")
        self.assertEqual((self.r["precision"], self.r["recall"], self.r["f1"]), (0.9245, 1.0, 0.9608))
        self.assertEqual(self.r["matches_published"], {"precision": True, "recall": True, "f1": True})
        self.assertEqual(self.r["reasons"], [])
        self.assertTrue(self.r["original_eval_AD_agrees"])

    def test_checksums_are_verified_and_recorded(self):
        p = self.r["provenance"]
        self.assertEqual(p["interval_file_sha256"], p["interval_file_sha256_pinned"])
        self.assertEqual(p["interval_file_sha256"], "c8da7bfa44cf18c44403739b98de8e45e98d1f628ae500c3e0ccb57f46c0e9d7")
        self.assertEqual((p["ad_cases_pkl"]["crc32"], p["ad_cases_pkl"]["bytes"]), (p["ad_cases_pkl"]["pinned_crc32"], p["ad_cases_pkl"]["pinned_bytes"]))
        self.assertEqual(p["armor_ad_split_timestamp"], 1651852800)
        for k in ("res_json_sha256", "split_file_content_sha256", "armor_scoring_py_sha256", "script_sha256", "armor_task_metrics_py_sha256"):
            self.assertRegex(p[k], r"^[0-9a-f]{64}$")

    def test_output_holds_only_aggregates_and_hashes_no_labels_no_counts_no_per_case_data(self):
        self.assertEqual(set(self.r), ALLOWED_KEYS)
        text = json.dumps(self.r)
        tens = set(re.findall(r"(?<![0-9a-f])\d{10}(?![0-9a-f])", text))
        self.assertEqual(tens, {"1651852800"})                                        # the only epoch-like number is the committed split time
        for key in ("labels", "intervals", "n_labels", "TP", "FP", "FN", "cases"):
            self.assertNotIn(f'"{key}"', text)

    def test_the_published_constants_equal_the_released_res_json(self):
        with open(v.RES_JSON) as fh:
            ad = json.load(fh)["AD"]
        self.assertEqual({k: ad[k] for k in v.PUBLISHED}, v.PUBLISHED)

    def test_command_line_exit_code_and_file_output(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "r.json")
            p = subprocess.run([sys.executable, "-I", os.path.join(DET, "verify_armor_published_ad.py"), "--out", out], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(out) as fh:
                self.assertEqual(json.load(fh)["status"], "PASS")


@NEEDS_FILES
class FailureModeTests(unittest.TestCase):
    """Missing or altered inputs never fall back to generated data."""

    def test_missing_interval_file_is_blocked(self):
        r = v.verify(interval_file="/nonexistent/pre_interval.json")
        self.assertEqual(r["status"], "BLOCKED")
        self.assertEqual((r["precision"], r["recall"], r["f1"]), (None, None, None))
        self.assertTrue(any("nothing is substituted" in x for x in r["reasons"]))

    def test_checksum_mismatch_is_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "pre_interval.json")
            with open(v.INTERVAL_FILE) as fh:
                data = json.load(fh)
            data[0][1] += 60
            with open(p, "w") as fh:
                json.dump(data, fh)
            r = v.verify(interval_file=p)
        self.assertEqual(r["status"], "BLOCKED")
        self.assertIn("checksum", r["reasons"][0])
        self.assertIsNone(r["f1"])

    def test_altered_label_file_is_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "ad_cases.pkl")
            shutil.copy(v.LABEL_FILE, p)
            with open(p, "r+b") as fh:
                fh.seek(100)
                b = fh.read(1)
                fh.seek(100)
                fh.write(bytes([b[0] ^ 1]))
            r = v.verify(label_file=p)
        self.assertEqual(r["status"], "BLOCKED")
        self.assertIn("ad_cases.pkl", r["reasons"][0])

    def test_changed_intervals_with_a_matching_pin_give_fail_not_pass(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "pre_interval.json")
            with open(v.INTERVAL_FILE) as fh:
                data = json.load(fh)
            data = data[:-3]                                                            # drop three intervals
            with open(p, "w") as fh:
                json.dump(data, fh)
            with mock.patch.object(v, "PINNED_INTERVAL_SHA256", v.sha256_file(p)):
                r = v.verify(interval_file=p)
        self.assertEqual(r["status"], "FAIL")
        self.assertFalse(all(r["matches_published"].values()))

    def test_the_label_reader_forbids_pickle_globals(self):
        with self.assertRaises(pickle.UnpicklingError):
            v._NoGlobals(io.BytesIO(pickle.dumps(os.getcwd))).load()
        self.assertEqual(v._NoGlobals(io.BytesIO(pickle.dumps([1, 2, 3]))).load(), [1, 2, 3])


class ImportIsolationTests(unittest.TestCase):
    def test_the_script_imports_only_stdlib_numpy_and_the_scoring_port(self):
        with open(v.__file__) as fh:
            tree = ast.parse(fh.read())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods.add(n.module)
        self.assertEqual(mods, {"argparse", "hashlib", "io", "json", "os", "pickle", "sys", "zlib", "importlib.util", "armor_scoring"})

    def test_importing_it_loads_no_model_dataset_checkpoint_or_training_code(self):
        code = ("import sys; sys.path.insert(0, %r); import verify_armor_published_ad; "
                "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'prep', 'model1_transformer', 'xgboost', 'pandas')]; "
                "print(bad); assert not bad" % DET)
        p = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_no_model_one_names_appear_in_the_source(self):
        with open(v.__file__) as fh:
            src = fh.read()
        self.assertIsNone(re.search(r"Model1|m1_dataset|checkpoint|calibrat|training_|torch|\.pt\b", src.split('"""', 2)[2]))


if __name__ == "__main__":
    unittest.main()
