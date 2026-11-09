"""ARMOR-equivalent scoring port: semantics of the port itself and equivalence with ARMOR's original functions (synthetic data only)."""
import ast
import os
import sys
import unittest

import _m1_helpers as H
import numpy as np

DET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "detection"))
if DET not in sys.path:
    sys.path.insert(0, DET)

import armor_scoring as sc  # noqa: E402

try:
    import equivalence as eq  # noqa: E402
    HAVE_ARMOR = eq.available()
except ImportError:                                           # pragma: no cover
    eq, HAVE_ARMOR = None, False

NEEDS_ARMOR = H.needs(HAVE_ARMOR, "needs ARMOR's released source files under armor-system/data/raw (git-ignored)")
B = 1_000_000


class IntervalSemanticsTests(unittest.TestCase):
    def test_gap_of_600_seconds_splits_and_599_merges(self):
        self.assertEqual(sc.get_pre_interval([B, B + 60, B + 60 + 599, B + 60 + 599 + 60]), [(B, B + 60 + 599 + 60)])
        self.assertEqual(sc.get_pre_interval([B, B + 60, B + 60 + 600, B + 60 + 600 + 60]), [(B, B + 60), (B + 660, B + 720)])
        self.assertEqual(sc.get_pre_interval([B, B + 60, B + 60 + 601, B + 60 + 601 + 60]), [(B, B + 60), (B + 661, B + 721)])

    def test_singletons_are_dropped_but_two_alerts_are_kept(self):
        self.assertEqual(sc.get_pre_interval([B]), [])
        self.assertEqual(sc.get_pre_interval([B, B + 1200, B + 2400]), [])
        self.assertEqual(sc.get_pre_interval([B, B + 60]), [(B, B + 60)])
        self.assertEqual(sc.get_pre_interval([B, B + 60, B + 1200, B + 2400, B + 2460]), [(B, B + 60), (B + 2400, B + 2460)])
        self.assertEqual(sc.get_pre_interval([]), [])

    def test_repeated_timestamps_merge_and_identical_only_is_dropped(self):
        self.assertEqual(sc.get_pre_interval([B, B, B, B + 60]), [(B, B + 60)])
        self.assertEqual(sc.get_pre_interval([B, B, B]), [])

    def test_the_input_order_is_used_as_given(self):
        self.assertEqual(sc.get_pre_interval([B + 120, B, B + 60]), [(B + 120, B + 60)])        # ARMOR never sorts

    def test_custom_delay(self):
        self.assertEqual(sc.get_pre_interval([B, B + 60, B + 180, B + 240], delay=120), [(B, B + 60), (B + 180, B + 240)])
        self.assertEqual(sc.get_pre_interval([B, B + 60, B + 180, B + 240], delay=121), [(B, B + 240)])

    def test_returned_values_are_plain_int_pairs(self):
        iv = sc.get_pre_interval(np.array([B, B + 60], dtype=np.int64))
        self.assertEqual((type(iv[0]), type(iv[0][0]), type(iv[0][1])), (tuple, int, int))


class ScoringSemanticsTests(unittest.TestCase):
    def detail(self, iv, labels, impact=300):
        return sc.eval_AD_detail(iv, labels, impact)

    def test_one_interval_matching_many_labels(self):
        d = self.detail([(B, B + 600)], [B + 100, B + 200, B + 300, B + 890])
        self.assertEqual((d["TP"], d["FP"], d["FN"], float(d["density"])), (4, 0, 0, 4.0))          # TP counts labels, FP counts intervals; density = labels per matched interval

    def test_one_label_matched_by_two_intervals(self):
        d = self.detail([(B, B + 60), (B + 120, B + 180)], [B + 100])
        self.assertEqual((d["TP"], d["FP"], d["FN"], float(d["density"])), (1, 0, 0, 1.0))          # the label is hit once however many intervals reach it; each interval holds one label

    def test_unmatched_interval_is_a_false_positive_and_unmatched_label_a_false_negative(self):
        d = self.detail([(B, B + 60), (B + 100000, B + 100060)], [B + 100, B + 50000])
        self.assertEqual((d["TP"], d["FP"], d["FN"]), (1, 1, 1))
        self.assertEqual((float(d["precision"]), float(d["recall"])), (0.5, 0.5))

    def test_duplicates_count_once(self):
        d = self.detail([(B, B + 60), (B, B + 60), (B + 9000, B + 9060)], [B + 100, B + 100, B + 500000])
        self.assertEqual((d["n_intervals"], d["n_labels"], d["TP"], d["FP"], d["FN"]), (2, 2, 1, 1, 1))

    def test_closed_boundaries(self):
        iv = [(B + 300, B + 360)]
        self.assertEqual(self.detail(iv, [B + 360 + 300])["TP"], 1)             # label - 300 == interval end: touching counts
        self.assertEqual(self.detail(iv, [B + 360 + 301])["TP"], 0)
        self.assertEqual(self.detail(iv, [B + 300 - 300])["TP"], 1)             # label + 300 == interval start: touching counts
        self.assertEqual(self.detail(iv, [B + 300 - 301])["TP"], 0)

    def test_f1_is_computed_from_the_rounded_precision_and_recall(self):
        """Find counts where ARMOR's F1 (from rounded P and R) differs from the exact F1; the port must return ARMOR's value."""
        found = None
        for tp in range(1, 40):
            for fp in range(0, 40):
                for fn in range(0, 40):
                    p, r = np.round(tp / (tp + fp), 4), np.round(tp / (tp + fn), 4)
                    armor = np.round(2 * p * r / (p + r), 4)
                    exact = np.round(2 * tp / (2 * tp + fp + fn), 4)
                    if armor != exact:
                        found = (tp, fp, fn, armor, exact)
                        break
                if found:
                    break
            if found:
                break
        self.assertIsNotNone(found)
        tp, fp, fn, armor, exact = found
        iv = [(B + 10000 * i, B + 10000 * i + 60) for i in range(tp + fp)]
        labels = [B + 10000 * i for i in range(tp)] + [B + 10_000_000 + 10000 * i for i in range(fn)]
        d = sc.eval_AD_detail(iv, labels)
        self.assertEqual((d["TP"], d["FP"], d["FN"]), (tp, fp, fn))
        self.assertEqual(d["f1"], armor)
        self.assertNotEqual(d["f1"], exact)

    def test_zero_division_and_nan_density_behaviour(self):
        with self.assertRaises(ZeroDivisionError):
            sc.eval_AD([], [B])                                                    # TP + FP = 0
        with self.assertRaises(ZeroDivisionError):
            sc.eval_AD([(B, B + 60)], [])                                          # TP + FN = 0
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = sc.eval_AD_detail([(B, B + 60)], [B + 99999])
        self.assertTrue(np.isnan(d["density"]))

    def test_list_of_lists_intervals_raise_like_armor(self):
        with self.assertRaises(TypeError):
            sc.eval_AD([[B, B + 60]], [B])

    def test_eval_AD_returns_the_triple_of_the_detail(self):
        iv, lab = [(B, B + 600), (B + 9000, B + 9600)], [B + 100, B + 9100, B + 70000]
        d = sc.eval_AD_detail(iv, lab)
        self.assertEqual(sc.eval_AD(iv, lab), (d["precision"], d["recall"], d["f1"]))

    def test_the_port_imports_only_numpy(self):
        with open(sc.__file__) as fh:
            tree = ast.parse(fh.read())
        mods = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"numpy"})


@NEEDS_ARMOR
class EquivalenceWithOriginalARMORTests(unittest.TestCase):
    """ARMOR's ORIGINAL functions (read from its released source) against the port, on synthetic alert sequences and labels only."""

    @classmethod
    def setUpClass(cls):
        cls.r = eq.run_battery()

    def test_all_cases_identical(self):
        self.assertTrue(self.r["all_identical"])
        self.assertEqual(len(self.r["edge_cases"]), 25)
        for c in self.r["edge_cases"]:
            self.assertTrue(c["intervals_identical"] and c["scores_identical"], c["case"])

    def test_the_battery_covers_scored_and_raising_cases_and_the_required_boundaries(self):
        names = {c["case"] for c in self.r["edge_cases"]}
        for need in ("gap_exactly_599", "gap_exactly_600_splits", "gap_601_splits", "singleton_dropped", "one_interval_matches_many_labels", "one_label_matched_by_two_intervals",
                     "repeated_timestamps_merge", "duplicate_labels_count_once", "label_touches_interval_end_inclusive", "label_one_second_after_reach", "no_alerts_zero_division"):
            self.assertIn(need, names)
        outcomes = {c["outcome"] for c in self.r["edge_cases"]}
        self.assertEqual(outcomes, {"scored", "raised"})
        self.assertEqual((self.r["random_cases"], self.r["unsorted_cases"]), (400, 60))
        self.assertGreater(self.r["random_scored"], 100)
        self.assertGreater(self.r["random_raised_same_exception"], 20)                  # identical ZeroDivisionError behaviour is covered too
        self.assertTrue(self.r["unsorted_identical"] and self.r["list_of_lists_intervals_identical_error"])

    def test_original_source_files_are_the_pinned_ones(self):
        p = self.r["provenance"]
        self.assertEqual(p["task_metrics.py_sha256"], "1463c8de9d32e4aaeb1279193b8c8bf50ca0fd794b1e093945406cb27219cbc6")
        self.assertEqual(p["anomaly_detector.py_sha256"], "5166023fd39aad6deb72ff528d2009631fae282373972adbec11adcda32414e3")
        self.assertEqual(p["get_pre_interval_source_sha256"], "0cc9c7cfc7345d61002f7cf9afca88d3b7bfa5196222254bd8a6e93e7f544b01")

    def test_a_deliberately_wrong_port_is_detected(self):
        """The comparison can fail: a port with a >= / > mistake in the delay rule differs from the original."""
        orig_eval, orig_get, _ = eq.load_original()
        ts = [B, B + 60, B + 60 + 600, B + 60 + 600 + 60]
        wrong = [(B, B + 720)]                                                              # what a '> delay' port would return (merged)
        self.assertNotEqual(orig_get(eq.original_frame(ts, [1, 1, 1, 1]), 600), wrong)
        self.assertEqual(orig_get(eq.original_frame(ts, [1, 1, 1, 1]), 600), sc.get_pre_interval(ts))


if __name__ == "__main__":
    unittest.main()
