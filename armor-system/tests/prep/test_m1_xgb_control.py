"""XGBoost control: saved-result checks, leakage and reproducibility.

xgboost and torch cannot share one macOS process (two OpenMP runtimes), so nothing here imports xgboost: training
checks run the script or a snippet in a subprocess.
"""
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import _m1_helpers as H
from prep import paths

MODEL_DIR = os.path.join(paths.ROOT, "models", "model1_xgboost_control")
sys.path.insert(0, MODEL_DIR)
import metrics as mt  # noqa: E402
import xgb_control as xc  # noqa: E402  (no torch / xgboost at import time)

REPO = os.path.abspath(os.path.join(paths.ROOT, ".."))
RESULTS = xc.RESULTS_DIR
RESULTS_PRESENT = os.path.exists(os.path.join(RESULTS, "validation_summary.json"))
HAVE_XGB = importlib.util.find_spec("xgboost") is not None
SCRIPT = os.path.join(MODEL_DIR, "xgb_control.py")


def jload(*p):
    with open(os.path.join(RESULTS, *p)) as fh:
        return json.load(fh)


def csv_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class ConfigTests(unittest.TestCase):
    def test_fixed_configuration(self):
        cfg, sha = xc.load_config()
        p = cfg["xgboost_params"]
        self.assertEqual({k: p[k] for k in ("objective", "num_class", "tree_method", "max_depth", "eta", "subsample", "colsample_bytree", "min_child_weight", "lambda", "nthread")},
                         {"objective": "multi:softprob", "num_class": 5, "tree_method": "hist", "max_depth": 3, "eta": 0.1, "subsample": 0.8,
                          "colsample_bytree": 0.5, "min_child_weight": 1, "lambda": 1, "nthread": 1})
        self.assertEqual((cfg["num_boost_round"], cfg["seeds"]), (100, [0, 1, 2, 3, 4]))
        self.assertEqual(cfg["partitions"], {"fit_cases": 88, "validation_cases": 25, "calibration_cases_used": 0, "held_out_cases_used": 0})
        self.assertEqual(sha, xc.load_config()[1])
        self.assertEqual(tuple(cfg["class_order"]), mt.CLASS_NAMES)

    def test_forbidden_settings_are_rejected(self):
        cfg, _ = xc.load_config()
        d = tempfile.mkdtemp()
        try:
            for mutate in (lambda c: c.update(early_stopping=10), lambda c: c.update(hyperparameter_search=True),
                           lambda c: c["partitions"].update(held_out_cases_used=1), lambda c: c["partitions"].update(calibration_cases_used=13),
                           lambda c: c.update(class_order=list(reversed(c["class_order"])))):
                bad = json.loads(json.dumps(cfg))
                mutate(bad)
                path = os.path.join(d, "bad.json")
                with open(path, "w") as fh:
                    json.dump(bad, fh)
                with self.assertRaises(ValueError):
                    xc.load_config(path)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_held_out_evaluation_is_blocked(self):
        with self.assertRaises(xc.FinalTestLocked):
            xc.final_test()
        with open(os.path.join(MODEL_DIR, "README.md")) as fh:
            self.assertIn("Test is evaluated once, together with Model 1, after both configurations are frozen.", fh.read())

    def test_no_command_line_switch_requests_the_held_out_set(self):
        src = open(SCRIPT).read()
        flags = [seg.split('"')[1] for seg in src.split("ap.add_argument(")[1:]]
        self.assertEqual(sorted(flags), ["--dump-features", "--out-dir", "--write-evidence"])


class HelperTests(unittest.TestCase):
    def test_forbidden_key_scan(self):
        self.assertEqual(xc.forbidden_keys({"a": {"test_metrics": 1}, "b": [{"x_test": 2}]}), ["/a/test_metrics", "/b[0]/x_test"])
        self.assertEqual(xc.forbidden_keys({"suite_runs": 1, "attestation": 2, "latest": 3, "validation_macro_f1": 4}), [])

    def test_compare_dirs(self):
        a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
        try:
            for d in (a, b):
                os.makedirs(os.path.join(d, "per_seed"))
                with open(os.path.join(d, "per_seed", "x.csv"), "w") as fh:
                    fh.write("1\n")
            self.assertTrue(xc.compare_dirs(a, b)[0])
            with open(os.path.join(b, "per_seed", "x.csv"), "w") as fh:
                fh.write("2\n")
            self.assertFalse(xc.compare_dirs(a, b)[0])
            os.remove(os.path.join(b, "per_seed", "x.csv"))
            self.assertFalse(xc.compare_dirs(a, b)[0])
        finally:
            shutil.rmtree(a, ignore_errors=True)
            shutil.rmtree(b, ignore_errors=True)

    def test_evidence_file_is_excluded_from_the_result_hash_list(self):
        self.assertIn(xc.EVIDENCE_NAME, xc.RESULT_FILES_EXCLUDED)


@H.needs(RESULTS_PRESENT, "XGBoost results not generated (xgb_control.py)")
class SavedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.prov, cls.maj = jload("validation_summary.json"), jload("case_features_provenance.json"), jload("majority_control.json")
        cls.m1 = H.read_json(os.path.join(paths.ART, "splits", "model1_calibration_split_v1.json"))
        cls.by_role = cls.m1["cases"]["by_role"]
        cls.seeds = xc.load_config()[0]["seeds"]

    def test_case_accounting_and_shapes(self):
        p = self.prov
        self.assertEqual(p["cases"], {"fit": 88, "validation": 25, "calibration_used": 0, "held_out_used": 0})
        self.assertEqual(p["feature_matrix_shapes"], {"fit": [88, 520], "validation": [25, 520]})
        self.assertEqual(p["fit_case_index"], self.by_role["fit"])
        self.assertEqual(p["validation_case_index"], self.by_role["validation"])
        self.assertEqual(p["class_distribution"], {"fit": [33, 29, 9, 13, 4], "validation": [10, 6, 3, 5, 1]})
        self.assertEqual(p["incident_groups"]["shared"], 0)
        self.assertEqual(p["class_order"], list(mt.CLASS_NAMES))

    def test_class_weights_come_from_the_88_fitting_cases_only(self):
        np.testing.assert_allclose(self.prov["class_weights_from_fit_cases"], [88 / (5 * n) for n in (33, 29, 9, 13, 4)])

    def test_inputs_are_the_approved_m1_03_hashes(self):
        i = self.prov["inputs"]
        self.assertEqual(i["calibration_manifest_content_sha256"], self.m1["content_sha256"])
        self.assertEqual(i["fit_only_scaler_file_sha256"], self.m1["fit_only_scaler"]["file_sha256"])
        self.assertEqual(i["feature_cache_content_sha256"], self.m1["inputs"]["feature_cache_content_sha256"])
        self.assertEqual(i["outer_split_content_sha256"], "32ff03fcffca09f3d3953777a020514b4af9c882bc85e3dc1693e282607638ac")
        self.assertEqual(self.summary["config_sha256"], xc.load_config()[1])
        self.assertEqual(jload("config_used.json")["config_sha256"], xc.load_config()[1])

    def test_predictions_cover_exactly_the_validation_cases(self):
        banned = set(self.by_role["calibration"]) | set(self.by_role["test"]) | set(self.by_role["fit"])
        for s in self.seeds:
            rows = csv_rows(os.path.join(RESULTS, "per_seed", f"seed_{s}_predictions.csv"))
            ids = [int(r["case_index"]) for r in rows]
            self.assertEqual(ids, self.by_role["validation"], s)
            self.assertFalse(set(ids) & banned)
            for r in rows:
                self.assertEqual(r["true_name"], mt.CLASS_NAMES[int(r["true_class"])])
                self.assertEqual(r["predicted_name"], mt.CLASS_NAMES[int(r["predicted_class"])])
                p = [float(r[f"p_{i}"]) for i in range(5)]
                self.assertAlmostEqual(sum(p), 1.0, places=5)
                self.assertEqual(int(np.argmax(p)), int(r["predicted_class"]))

    def test_stored_metrics_equal_a_fresh_recomputation_from_the_predictions(self):
        for s in self.seeds:
            rows = csv_rows(os.path.join(RESULTS, "per_seed", f"seed_{s}_predictions.csv"))
            yt, yp = [int(r["true_class"]) for r in rows], [int(r["predicted_class"]) for r in rows]
            fresh, stored = mt.summary(yt, yp), jload("per_seed", f"seed_{s}_metrics.json")
            for k in ("accuracy", "macro_f1", "weighted_f1", "confusion_matrix", "per_class"):
                self.assertEqual(json.loads(json.dumps(fresh[k])), stored[k], (s, k))
            self.assertAlmostEqual(stored["macro_f1_excluding_node_memory"], mt.macro_f1_excluding(yt, yp, 4))
            self.assertEqual(stored["per_class"]["support"], [10, 6, 3, 5, 1])
            self.assertEqual(stored["seed"], s)

    def test_mean_and_sample_sd_across_seeds(self):
        vals = [jload("per_seed", f"seed_{s}_metrics.json")["macro_f1"] for s in self.seeds]
        a = self.summary["mean_sd_across_seeds"]["macro_f1"]
        self.assertAlmostEqual(a["mean"], float(np.mean(vals)))
        self.assertAlmostEqual(a["sd"], float(np.std(vals, ddof=1)))
        self.assertEqual(a["n"], 5)
        self.assertEqual([x["seed"] for x in self.summary["per_seed"]], self.seeds)
        self.assertNotIn("best", json.dumps(self.summary).lower())             # no best-seed selection anywhere

    def test_majority_control_is_the_most_frequent_fitting_class(self):
        self.assertEqual(self.maj["predicted_class"], 0)
        self.assertEqual(self.maj["predicted_name"], "Container Hardware")
        self.assertEqual(self.maj["fit_class_counts"], [33, 29, 9, 13, 4])
        self.assertAlmostEqual(self.maj["accuracy"], 0.4)
        self.assertAlmostEqual(self.maj["macro_f1"], (2 * 0.4 / 1.4) / 5)
        self.assertEqual(self.maj["per_class"]["recall"], [1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(self.maj["confusion_matrix"][0], [10, 0, 0, 0, 0])

    def test_family_accounting_is_complete_for_every_seed(self):
        for s in self.seeds:
            d = jload("per_seed", f"seed_{s}_metrics.json")["family_diagnostic"]
            tot = 0
            for f, fam in d["families"].items():
                self.assertEqual(fam["exactly_correct"] + fam["wrong_inside_family"] + fam["wrong_outside_family"], fam["n_cases"])
                self.assertEqual(sum(sum(r) for r in fam["confusion_true_rows_vs_all_predicted_columns"]), fam["n_cases"])
                self.assertEqual(fam["family_level_accuracy"]["denominator"], fam["n_cases"])
                tot += fam["n_cases"]
            self.assertEqual(tot, 25)
            self.assertEqual((d["families"]["container"]["n_cases"], d["families"]["node"]["n_cases"]), (16, 9))
            self.assertEqual(int(np.sum(d["family_confusion_true_rows_pred_cols"]["matrix"])), 25)

    def test_limitations_are_stated(self):
        text = " ".join(self.summary["limitations"])
        for needle in ("ONE validation case", "Seed SD measures training randomness", "Node CPU has 3", "not used"):
            self.assertIn(needle, text)
        self.assertEqual(self.summary["validation_support"]["Node Memory"], 1)

    def test_no_held_out_results_in_any_saved_artifact(self):
        for d, _, files in os.walk(RESULTS):
            for f in files:
                path = os.path.join(d, f)
                if f.endswith(".json"):
                    with open(path) as fh:
                        self.assertEqual(xc.forbidden_keys(json.load(fh)), [], f)
                elif f.endswith(".csv"):
                    ids = {int(r["case_index"]) for r in csv_rows(path)}
                    self.assertFalse(ids & (set(self.by_role["test"]) | set(self.by_role["calibration"])), f)

    def test_observation_timing_is_recorded_and_consistent(self):
        t = jload("observation_timing.json")
        rows = t["per_case"]
        self.assertEqual([sum(1 for r in rows if r["role"] == k) for k in ("fit", "validation")], [88, 25])
        self.assertEqual([r["case_index"] for r in rows if r["role"] == "fit"], self.by_role["fit"])
        self.assertEqual([r["case_index"] for r in rows if r["role"] == "validation"], self.by_role["validation"])
        for r in rows:
            self.assertGreater(r["n_before_injection"], 0)                    # pre-injection telemetry is included ...
            self.assertGreater(r["n_at_or_after_injection"], 0)               # ... and so is post-injection telemetry
            self.assertEqual(r["n_samples"], r["n_before_injection"] + r["n_at_or_after_injection"])
            self.assertTrue(-360 <= r["first_offset_s"] < 0 < r["last_offset_s"] <= 360)
            self.assertEqual(r["earliest_prediction_ts"], r["last_sample_ts"])
            self.assertEqual(r["earliest_prediction_offset_s"], r["last_offset_s"])
            self.assertGreaterEqual(r["earliest_prediction_offset_s"], 280)     # never available before ~4.7 minutes after injection
            self.assertEqual(r["first_sample_ts"] - r["case_ts"], r["first_offset_s"])
        self.assertTrue(all(12 >= r["n_samples"] >= 11 for r in rows))

    def test_observation_isolation_checks_are_clean(self):
        c = jload("observation_timing.json")["isolation_checks"]
        self.assertTrue(c["fit_samples_before_fit_cutoff"] and c["validation_samples_inside_validation_period"])
        self.assertEqual((c["samples_shared_by_fit_and_validation"], c["samples_in_calibration_region"], c["samples_in_held_out_period"]), (0, 0, 0))
        text = " ".join(jload("observation_timing.json")["reading"])
        self.assertIn("not early incident detection", text)
        self.assertIn("same 88 fitting cases", text)
        self.assertEqual(self.m1["protocol"]["Tc"], 1651637521)
        for r in jload("observation_timing.json")["per_case"]:
            self.assertLess(r["last_sample_ts"], self.m1["protocol"]["outer_period_boundaries"]["validation_test"])   # nothing from the held-out period

    def test_versions_and_pin(self):
        with open(os.path.join(MODEL_DIR, "requirements.txt")) as fh:
            pin = fh.read().strip()
        self.assertEqual(pin, "xgboost==" + self.summary["versions"]["xgboost"])


class EvidenceVerificationTests(unittest.TestCase):
    def test_saved_evidence_verifies_and_detects_tampering(self):
        # The evidence is produced by `xgb_control.py --write-evidence`, which runs this suite first and verifies the
        # file itself right after writing it; before that first run there is nothing to check here.
        if not os.path.exists(os.path.join(RESULTS, xc.EVIDENCE_NAME)):
            return
        ev = jload(xc.EVIDENCE_NAME)
        self.assertEqual(xc.verify_evidence(ev, RESULTS), [])
        self.assertIsNone(ev["commit_reference"]["final_merge_commit"])
        self.assertEqual(ev["cases"]["held_out_used"], 0)
        for mutate, text in ((lambda e: e["suite_runs"].update(skipped=1), "skips"),
                             (lambda e: e["reproducibility"].update(identical_to_each_other=False), "reproducibility"),
                             (lambda e: e["result_files_sha256"].update({"validation_summary.json": "0" * 64}), "result files"),
                             (lambda e: e["commit_reference"].update(source_set_sha256="0" * 64), "source files")):
            bad = json.loads(json.dumps(ev))
            mutate(bad)
            self.assertTrue(any(text in p for p in xc.verify_evidence(bad, RESULTS)), text)


XGB_NEEDS = H.needs(HAVE_XGB and H.CACHE_PRESENT and H.ARTIFACTS_PRESENT, "needs xgboost, the D1 cache and the M1-03 artifacts")

LEAK_SNIPPET = r"""
import sys, json, numpy as np
sys.path.insert(0, %(model)r)
import xgb_control as xc
cfg, _ = xc.load_config()
rng = np.random.default_rng(0)
fit = {"X": rng.normal(size=(60, 20)).astype(np.float32), "y": np.arange(60) %% 5}
val = {"X": rng.normal(size=(12, 20)).astype(np.float32), "y": np.arange(12) %% 5}
w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
p1, d1 = xc.train_predict(cfg, fit, val, w, 7)
val2 = {"X": val["X"], "y": (val["y"] + 1) %% 5}                      # different validation labels
p2, d2 = xc.train_predict(cfg, fit, val2, w, 7)
val3 = {"X": val["X"] + 10.0, "y": val["y"]}                            # different validation features
p3, d3 = xc.train_predict(cfg, fit, val3, w, 7)
fit2 = {"X": fit["X"], "y": (fit["y"] + 1) %% 5}
p4, d4 = xc.train_predict(cfg, fit2, val, w, 7)                        # different FIT labels must matter
p5, d5 = xc.train_predict(cfg, fit, val, w, 8)                         # different seed must matter
print(json.dumps({"labels_unused": bool((p1 == p2).all() and d1 == d2), "val_features_do_not_change_the_model": d1 == d3,
                  "val_features_change_predictions": bool((p1 != p3).any()), "fit_labels_matter": d1 != d4, "seed_matters": d1 != d5,
                  "shape": list(p1.shape), "sums_to_one": bool(np.allclose(p1.sum(1), 1, atol=1e-5))}))
"""


@XGB_NEEDS
class TrainingLeakageAndReproducibilityTests(unittest.TestCase):
    def test_training_uses_only_fit_data(self):
        out = subprocess.run([sys.executable, "-I", "-c", LEAK_SNIPPET % {"model": MODEL_DIR}], capture_output=True, text=True, cwd=tempfile.gettempdir())
        self.assertEqual(out.returncode, 0, out.stderr[-1500:])
        r = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertTrue(r["labels_unused"], "validation labels influenced the fit or the predictions")
        self.assertTrue(r["val_features_do_not_change_the_model"], "validation features influenced the fitted model")
        self.assertTrue(r["val_features_change_predictions"])
        self.assertTrue(r["fit_labels_matter"] and r["seed_matters"])
        self.assertEqual(r["shape"], [12, 5])
        self.assertTrue(r["sums_to_one"])

    def test_two_clean_reruns_are_identical_per_seed_and_equal_the_saved_results(self):
        dirs = []
        try:
            for i in (1, 2):
                d = tempfile.mkdtemp(prefix=f"xgbtest{i}_")
                dirs.append(d)
                p = subprocess.run([sys.executable, "-I", SCRIPT, "--out-dir", d], capture_output=True, text=True, cwd=tempfile.gettempdir())
                self.assertEqual(p.returncode, 0, (p.stdout + p.stderr)[-1500:])
            same, files, _ = xc.compare_dirs(dirs[0], dirs[1])
            self.assertTrue(same)
            for s in range(5):
                self.assertIn(f"per_seed/seed_{s}_predictions.csv", files)
                self.assertIn(f"per_seed/seed_{s}_metrics.json", files)
            if RESULTS_PRESENT:
                self.assertTrue(xc.compare_dirs(dirs[0], RESULTS)[0], "saved results differ from a fresh run")
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
