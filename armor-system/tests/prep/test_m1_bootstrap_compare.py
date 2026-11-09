"""Focused tests of the paired case-level bootstrap (Model 1 Transformer vs frozen XGBoost, saved validation predictions only)."""
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import unittest

import numpy as np

MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
if MODELS not in sys.path:
    sys.path.insert(0, MODELS)

from model1_transformer import bootstrap_compare as bc  # noqa: E402

MT = bc._metrics_module()


def copy_inputs(td):
    tr, xg = os.path.join(td, "tr"), os.path.join(td, "xgb")
    for s in bc.SEEDS:
        os.makedirs(os.path.join(tr, f"seed_{s}"), exist_ok=True)
        os.makedirs(os.path.join(xg, "per_seed"), exist_ok=True)
        shutil.copy(os.path.join(bc.TR_RES, f"seed_{s}", "predictions.csv"), os.path.join(tr, f"seed_{s}", "predictions.csv"))
        shutil.copy(os.path.join(bc.XGB_RES, "per_seed", f"seed_{s}_predictions.csv"), os.path.join(xg, "per_seed", f"seed_{s}_predictions.csv"))
    shutil.copy(os.path.join(bc.XGB_RES, "case_features_provenance.json"), os.path.join(xg, "case_features_provenance.json"))
    return tr, xg


def edit_csv(path, fn):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fn(rows)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class InputVerificationTests(unittest.TestCase):
    def test_the_real_saved_predictions_agree_on_cases_labels_and_class_order(self):
        cases, y, ptr, pxg, files = bc.load_inputs()
        self.assertEqual((len(cases), ptr.shape, pxg.shape), (25, (5, 25), (5, 25)))
        self.assertEqual(np.bincount(y, minlength=5).tolist(), [10, 6, 3, 5, 1])
        self.assertEqual(len(files), 10)
        self.assertTrue(all("test" not in os.path.basename(f) and "calibration" not in f for f in files.values()))

    def check_fails(self, mutate, which="tr"):
        with tempfile.TemporaryDirectory() as td:
            tr, xg = copy_inputs(td)
            path = os.path.join(tr, "seed_2", "predictions.csv") if which == "tr" else os.path.join(xg, "per_seed", "seed_3_predictions.csv")
            edit_csv(path, mutate)
            with self.assertRaises(bc.ComparisonError):
                bc.load_inputs(tr, xg)

    def test_swapped_case_order_is_rejected(self):
        self.check_fails(lambda rows: rows.__setitem__(slice(0, 2), [rows[1], rows[0]]))
        self.check_fails(lambda rows: rows.__setitem__(slice(0, 2), [rows[1], rows[0]]), "xgb")

    def test_a_changed_label_is_rejected(self):
        def flip(rows):
            rows[0]["true_class"], rows[0]["true_name"] = "1", "Container Network"
        self.check_fails(flip)
        self.check_fails(flip, "xgb")

    def test_wrong_class_names_or_out_of_range_predictions_are_rejected(self):
        def rename(rows):
            rows[3]["predicted_name"] = "Node Disk" if rows[3]["predicted_name"] != "Node Disk" else "Node CPU"
        self.check_fails(rename)

        def bad(rows):
            rows[3]["predicted_class"] = "7"
        self.check_fails(bad, "xgb")

    def test_a_missing_or_extra_case_is_rejected(self):
        self.check_fails(lambda rows: rows.pop())
        self.check_fails(lambda rows: rows.append(dict(rows[0], case_index="999")), "xgb")


class MetricDefinitionTests(unittest.TestCase):
    def test_absent_class_keeps_its_place_in_the_five_class_macro_average(self):
        y = np.array([0, 0, 1, 2, 3, 3])                                          # class 4 (Node Memory) absent
        r = bc.five_seed_means(MT, y, np.stack([y, y]))
        self.assertAlmostEqual(r["macro_f1"], 4 / 5)                              # four perfect classes, one empty class counts 0
        self.assertAlmostEqual(r["weighted_f1"], 1.0)                             # weighted by support: the absent class has weight 0

    def test_a_false_prediction_of_an_absent_class_is_still_an_error(self):
        y = np.array([0, 0, 1, 2, 3, 3])
        p = np.array([0, 0, 1, 2, 3, 4])
        r = MT.summary(y, p)
        self.assertLess(r["macro_f1"], 0.8)

    def test_interval_uses_percentiles(self):
        r = bc.interval(np.arange(101.0))
        self.assertEqual((r["mean"], r["ci95_low"], r["ci95_high"]), (50.0, 2.5, 97.5))


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0] * 10 + [1] * 6 + [2] * 3 + [3] * 5 + [4])
        rng = np.random.default_rng(1)
        self.good = np.tile(self.y, (5, 1))
        self.noisy = self.good.copy()
        flip = rng.random(self.noisy.shape) < 0.2
        self.noisy[flip] = rng.integers(0, 5, flip.sum())

    def test_identical_models_have_exactly_zero_difference_in_every_resample(self):
        r = bc.bootstrap(self.y, self.noisy, self.noisy, n_boot=200, seed=3, mt=MT)
        for m in bc.METRICS:
            d = r["metrics"][m]["difference_transformer_minus_xgboost"]
            self.assertEqual((d["mean"], d["ci95_low"], d["ci95_high"]), (0.0, 0.0, 0.0))     # paired resampling: same cases for both models
            self.assertEqual(r["metrics"][m]["fraction_difference_zero"], 1.0)
            self.assertEqual(r["metrics"][m]["fraction_difference_positive"], 0.0)

    def test_a_clearly_better_model_gives_a_positive_difference(self):
        r = bc.bootstrap(self.y, self.good, self.noisy, n_boot=300, seed=3, mt=MT)
        m = r["metrics"]["weighted_f1"]
        self.assertGreater(m["fraction_difference_positive"], 0.95)
        self.assertGreater(m["difference_transformer_minus_xgboost"]["ci95_low"], 0)
        self.assertAlmostEqual(m["fraction_difference_positive"] + m["fraction_difference_zero"] + m["fraction_difference_negative"], 1.0)

    def test_fixed_seed_is_reproducible_and_a_different_seed_differs(self):
        a = bc.bootstrap(self.y, self.good, self.noisy, n_boot=100, seed=5, mt=MT)
        b = bc.bootstrap(self.y, self.good, self.noisy, n_boot=100, seed=5, mt=MT)
        c = bc.bootstrap(self.y, self.good, self.noisy, n_boot=100, seed=6, mt=MT)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertNotEqual(json.dumps(a, sort_keys=True), json.dumps(c, sort_keys=True))

    def test_resamples_without_a_class_are_counted(self):
        r = bc.bootstrap(self.y, self.good, self.noisy, n_boot=400, seed=2, mt=MT)
        self.assertEqual(r["resamples_without_class"]["Container Hardware"], 0.0)
        self.assertAlmostEqual(r["resamples_without_class"]["Node Memory"], (24 / 25) ** 25, delta=0.06)


class SavedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(bc.OUT) as fh:
            cls.r = json.load(fh)

    def test_observed_means_equal_the_frozen_summaries_of_both_models(self):
        with open(os.path.join(bc.XGB_RES, "validation_summary.json")) as fh:
            x = json.load(fh)["mean_sd_across_seeds"]
        with open(os.path.join(bc.TR_RES, "validation_summary.json")) as fh:
            t = json.load(fh)["mean_sd_across_seeds"]
        for m in bc.METRICS:
            self.assertAlmostEqual(self.r["observed"]["xgboost"][m], x[m]["mean"], places=12)
            self.assertAlmostEqual(self.r["observed"]["transformer"][m], t[m]["mean"], places=12)

    def test_design_of_the_saved_result(self):
        self.assertEqual((self.r["resamples"], self.r["bootstrap_seed"], self.r["evaluated_on"][:2]), (1000, bc.SEED, "25"))
        self.assertEqual(self.r["class_order"], list(bc.CLASS_NAMES))
        for m in bc.METRICS:
            s = self.r["metrics"][m]
            for k in ("transformer", "xgboost", "difference_transformer_minus_xgboost"):
                self.assertLessEqual(s[k]["ci95_low"], s[k]["ci95_high"])
        self.assertIn("NOT a probability", self.r["reading"]["fraction_positive"])
        self.assertEqual(len(self.r["provenance"]["input_files_sha256"]), 10)

    def test_regeneration_is_byte_identical(self):
        cases, y, ptr, pxg, files = bc.load_inputs()
        again = bc.bootstrap(y, ptr, pxg, mt=MT)
        for k in ("observed", "metrics", "resamples_without_class"):
            self.assertEqual(json.dumps(self.r[k], sort_keys=True), json.dumps(again[k], sort_keys=True))
        for name, f in files.items():
            self.assertEqual(self.r["provenance"]["input_files_sha256"][name], bc.sha256_file(f))


class NoHeldOutAccessTests(unittest.TestCase):
    def test_the_module_reaches_no_dataset_model_calibration_or_test_data(self):
        with open(bc.__file__) as fh:
            src = fh.read()
        self.assertIsNone(re.search(r"^\s*(import|from)\s+(torch|prep|model1_transformer|xgboost)", src, flags=re.M))
        self.assertIsNone(re.search(r"""["'](test|calibration|embargo|fit)["']|Model1Data|evaluation_dataset|training_dataset|\.pt\b|\.npz\b|d1_features""", src))
        self.assertNotIn("torch", sys.modules.get("model1_transformer.bootstrap_compare").__dict__)

    def test_inputs_are_only_the_ten_prediction_files(self):
        _, _, _, _, files = bc.load_inputs()
        self.assertTrue(all(f.endswith("predictions.csv") or f.endswith("_predictions.csv") for f in files.values()))


if __name__ == "__main__":
    unittest.main()
