import os
import sys
import unittest

import numpy as np

import _m1_helpers as H
from prep import paths

sys.path.insert(0, os.path.join(paths.ROOT, "models", "model1_xgboost_control"))
import metrics as mt  # noqa: E402

YT = [0, 0, 1, 1, 2, 2, 3, 4]
YP = [0, 1, 1, 1, 2, 0, 3, 3]


class KnownAnswerMetricTests(unittest.TestCase):
    def test_confusion_matrix_true_rows_pred_cols(self):
        cm = mt.confusion_matrix(YT, YP)
        self.assertEqual(cm.tolist(), [[1, 1, 0, 0, 0], [0, 2, 0, 0, 0], [1, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]])

    def test_per_class_and_scalar_metrics(self):
        pc = mt.per_class(YT, YP)
        np.testing.assert_allclose(pc["precision"], [0.5, 2 / 3, 1.0, 0.5, 0.0])      # class 4 never predicted: zero division -> 0
        np.testing.assert_allclose(pc["recall"], [0.5, 1.0, 0.5, 1.0, 0.0])
        np.testing.assert_allclose(pc["f1"], [0.5, 0.8, 2 / 3, 2 / 3, 0.0])
        self.assertEqual(pc["support"].tolist(), [2, 2, 2, 1, 1])
        s = mt.summary(YT, YP)
        self.assertAlmostEqual(s["accuracy"], 5 / 8)
        self.assertAlmostEqual(s["macro_f1"], (0.5 + 0.8 + 2 / 3 + 2 / 3 + 0.0) / 5)
        self.assertAlmostEqual(s["weighted_f1"], 4.6 / 8)
        self.assertAlmostEqual(mt.macro_f1_excluding(YT, YP, exclude=4), (0.5 + 0.8 + 2 / 3 + 2 / 3) / 4)

    def test_perfect_and_all_wrong(self):
        self.assertEqual(mt.summary(YT, YT)["macro_f1"], 1.0)
        self.assertEqual(mt.summary([0, 1], [1, 0])["macro_f1"], 0.0)

    def test_class_absent_from_truth_and_predictions_counts_as_zero_f1(self):
        s = mt.summary([0, 0, 1], [0, 0, 1])
        self.assertAlmostEqual(s["macro_f1"], 2 / 5)                                   # three empty classes contribute 0

    def test_invalid_input_raises(self):
        for a, b in (([], []), ([0, 1], [0]), ([0, 5], [0, 1]), ([0, -1], [0, 1])):
            with self.assertRaises(ValueError):
                mt.confusion_matrix(a, b)


class MajorityControlTests(unittest.TestCase):
    def test_majority_class_and_tie_break(self):
        self.assertEqual(mt.majority_class([0, 0, 1, 2, 2, 2]), 2)
        self.assertEqual(mt.majority_class([1, 1, 3, 3]), 1)                           # tie -> lowest id
        self.assertEqual(mt.majority_class([4]), 4)

    def test_majority_control_on_the_real_validation_distribution(self):
        yv = np.repeat(np.arange(5), [10, 6, 3, 5, 1])
        s = mt.summary(yv, np.zeros(25, np.int64))
        self.assertAlmostEqual(s["accuracy"], 0.4)
        self.assertAlmostEqual(s["macro_f1"], (2 * 0.4 * 1.0 / 1.4) / 5)               # only class 0 has non-zero F1
        self.assertAlmostEqual(s["weighted_f1"], 0.4 * (2 * 0.4 / 1.4))
        self.assertEqual(s["per_class"]["recall"], [1.0, 0.0, 0.0, 0.0, 0.0])


class FamilyDiagnosticTests(unittest.TestCase):
    def test_family_mapping_from_names(self):
        self.assertEqual([mt.family_of(n) for n in mt.CLASS_NAMES], ["container", "container", "node", "node", "node"])
        with self.assertRaises(ValueError):
            mt.family_of("Disk Fault")

    def test_accounting_keeps_cross_family_mistakes_visible(self):
        d = mt.family_diagnostic(YT, YP)
        c, n = d["families"]["container"], d["families"]["node"]
        self.assertEqual((c["n_cases"], n["n_cases"]), (4, 4))
        self.assertEqual((c["exactly_correct"], c["wrong_inside_family"], c["wrong_outside_family"]), (3, 1, 0))
        self.assertEqual((n["exactly_correct"], n["wrong_inside_family"], n["wrong_outside_family"]), (2, 1, 1))
        for f in (c, n):
            self.assertEqual(f["exactly_correct"] + f["wrong_inside_family"] + f["wrong_outside_family"], f["n_cases"])
        self.assertEqual((c["family_level_accuracy"]["numerator"], c["family_level_accuracy"]["denominator"]), (4, 4))
        self.assertEqual((n["family_level_accuracy"]["numerator"], n["family_level_accuracy"]["denominator"]), (3, 4))
        self.assertEqual((c["conditional_within_family_accuracy"]["numerator"], c["conditional_within_family_accuracy"]["denominator"]), (3, 4))
        self.assertEqual((n["conditional_within_family_accuracy"]["numerator"], n["conditional_within_family_accuracy"]["denominator"]), (2, 3))
        self.assertAlmostEqual(n["conditional_within_family_accuracy"]["value"], 2 / 3)
        self.assertEqual(n["confusion_true_rows_vs_all_predicted_columns"], [[1, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]])
        self.assertEqual(n["confusion_true_rows_vs_all_predicted_columns"][0][0], 1)    # the Node CPU case predicted as Container Hardware is retained
        self.assertEqual(len(n["confusion_true_rows_vs_all_predicted_columns"][0]), 5)
        self.assertEqual(d["family_confusion_true_rows_pred_cols"]["matrix"], [[4, 0], [1, 3]])
        self.assertIn("not prove", d["note"])

    def test_totals_equal_the_number_of_cases(self):
        rng = np.random.default_rng(0)
        yt, yp = rng.integers(0, 5, 60), rng.integers(0, 5, 60)
        d = mt.family_diagnostic(yt, yp)
        self.assertEqual(sum(f["n_cases"] for f in d["families"].values()), 60)
        self.assertEqual(int(np.sum(d["family_confusion_true_rows_pred_cols"]["matrix"])), 60)

    def test_empty_denominator_is_reported_as_none_not_zero(self):
        d = mt.family_diagnostic([2, 3], [0, 1])                    # every node case predicted in the container family
        n = d["families"]["node"]
        self.assertIsNone(n["conditional_within_family_accuracy"]["value"])
        self.assertEqual(n["conditional_within_family_accuracy"]["denominator"], 0)
        self.assertEqual(n["family_level_accuracy"]["value"], 0.0)


if __name__ == "__main__":
    unittest.main()
