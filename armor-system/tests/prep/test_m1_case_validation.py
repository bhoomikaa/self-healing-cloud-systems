"""Case-level validation interface and timestamp alignment with the frozen XGBoost provenance (Linear 298-51).

No model is run on real validation windows here: the synthetic tests use synthetic predictions, and the real-data tests read
only the window TABLE (case ids, timestamps, labels), never validation telemetry.
"""
import copy
import re
import unittest

import _m1_model_helpers as MH
import numpy as np
import torch

import _m1_helpers as H
from model1_transformer import case_eval as ce
from prep import m1_case_features as cf, m1_dataset as md

NEEDS_REAL = H.needs(H.CACHE_PRESENT and H.ARTIFACTS_PRESENT, "needs the D1 feature cache and the M1-03 artifacts")
CASES = [101, 103, 104, 110]
LABELS = [0, 1, 3, 4]


def fixture(windows=6):
    """Synthetic validation windows and a matching frozen provenance: 4 cases, `windows` consecutive windows each (12 distinct minutes)."""
    ci, ts, fc, timing = [], [], [], {}
    for i, (c, y) in enumerate(zip(CASES, LABELS)):
        t0 = 1_000_000 + 100_000 * i
        for j in range(windows):
            ci.append(c)
            ts.append(t0 + 60 * (j + np.arange(7)))
            fc.append(y)
        last = t0 + 60 * (windows + 5)
        timing[c] = {"case_index": c, "role": "validation", "case_ts": t0 + 17, "first_sample_ts": t0, "last_sample_ts": last, "n_samples": windows + 6}
    ci, ts, fc = np.array(ci, np.int64), np.array(ts, np.int64), np.array(fc, np.int64)
    prov = {"validation_case_index": CASES, "class_order": list(ce.CLASS_NAMES), "windows_per_case": {"validation": [windows, windows]},
            "class_distribution": {"validation": np.bincount(LABELS, minlength=5).tolist()},
            "array_sha256": {"validation_case_index": ce.array_sha256(np.array(CASES, np.int64)), "y_validation": ce.array_sha256(np.array(LABELS, np.int64))}}
    return ci, ts, fc, {"timing": timing, "provenance": prov}


def onehot_probs(classes, p=0.9):
    out = np.full((len(classes), 5), (1 - p) / 4)
    out[np.arange(len(classes)), classes] = p
    return out


class AggregationTests(unittest.TestCase):
    def test_mean_probability_once_per_case_then_argmax(self):
        ci = np.array([7, 7, 7, 9, 9])
        pr = np.array([[0.4, 0.6, 0, 0, 0], [0.4, 0.6, 0, 0, 0], [0.95, 0.05, 0, 0, 0], [0, 0, 0, 0.2, 0.8], [0, 0, 0, 0.2, 0.8]])
        a = ce.aggregate_cases(ci, pr)
        np.testing.assert_allclose(a["probs"][0], [(0.4 + 0.4 + 0.95) / 3, (0.6 + 0.6 + 0.05) / 3, 0, 0, 0])
        self.assertEqual(a["pred_class"].tolist(), [0, 4])                       # mean probability picks 0; a majority vote of window argmaxes would pick 1
        self.assertEqual(a["case_index"].tolist(), [7, 9])
        self.assertEqual(a["n_windows"].tolist(), [3, 2])
        np.testing.assert_allclose(a["probs"].sum(1), 1.0)

    def test_every_case_counts_once_whatever_its_number_of_windows(self):
        ci = np.array([1] * 6 + [2] * 5)
        pr = np.vstack([onehot_probs([0] * 6), onehot_probs([1] * 5)])
        a = ce.aggregate_cases(ci, pr)
        self.assertEqual(len(a["pred_class"]), 2)
        np.testing.assert_allclose(a["probs"][0], pr[0])

    def test_input_order_does_not_matter_and_output_is_sorted_by_case(self):
        ci, ts, fc, _ = fixture()
        pr = onehot_probs(np.repeat([0, 1, 3, 4], 6)) * 0 + np.random.default_rng(0).dirichlet(np.ones(5), len(ci))
        a = ce.aggregate_cases(ci, pr, fc)
        perm = np.random.default_rng(1).permutation(len(ci))
        b = ce.aggregate_cases(ci[perm], pr[perm], fc[perm])
        for k in a:
            np.testing.assert_allclose(a[k], b[k], atol=1e-12)
        self.assertEqual(a["case_index"].tolist(), sorted(CASES))
        self.assertEqual(a["true_class"].tolist(), LABELS)

    def test_class_order_is_preserved(self):
        for k in range(5):
            a = ce.aggregate_cases(np.array([5, 5]), onehot_probs([k, k]))
            self.assertEqual(a["pred_class"].tolist(), [k])
            self.assertEqual(int(a["probs"][0].argmax()), k)
        self.assertEqual(ce.CLASS_NAMES, ("Container Hardware", "Container Network", "Node CPU", "Node Disk", "Node Memory"))

    def test_a_tie_goes_to_the_lowest_class_id(self):
        a = ce.aggregate_cases(np.array([1]), np.array([[0.0, 0.5, 0.5, 0.0, 0.0]]))
        self.assertEqual(a["pred_class"].tolist(), [1])

    def test_invalid_probabilities_are_rejected(self):
        ci = np.array([1, 1])
        good = onehot_probs([0, 0])
        for bad in (good * 1.1, np.where(np.eye(2, 5) > 0, np.nan, good), good[:, :4], -good, np.vstack([good[0], good[0] * 0])):
            with self.assertRaises(ce.AlignmentError):
                ce.aggregate_cases(ci, bad)
        with self.assertRaises(ce.AlignmentError):
            ce.aggregate_cases(np.array([1, 1]), good, fault_class=np.array([0, 1]))        # one case, two labels
        with self.assertRaises(ce.AlignmentError):
            ce.aggregate_cases(np.array([], np.int64), np.zeros((0, 5)))

    def test_rows_use_the_xgboost_prediction_layout(self):
        ci, ts, fc, _ = fixture()
        a = ce.aggregate_cases(ci, onehot_probs(np.repeat([0, 1, 3, 4], 6)), fc)
        rows = ce.xgb_style_rows(a, group_id=[1, 2, 3, 4])
        self.assertEqual(list(rows[0]), ["case_index", "group_id", "true_class", "true_name", "predicted_class", "predicted_name", "p_0", "p_1", "p_2", "p_3", "p_4"])
        self.assertEqual((rows[2]["true_name"], rows[2]["predicted_name"]), ("Node Disk", "Node Disk"))


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.ci, self.ts, self.fc, self.frozen = fixture()

    def check(self, ci=None, ts=None, fc=None, frozen=None, case_ts=None):
        return ce.check_alignment(self.ci if ci is None else ci, self.ts if ts is None else ts, self.fc if fc is None else fc,
                                  frozen or self.frozen, case_ts)

    def test_matching_cases_minutes_labels_and_case_times_pass(self):
        r = self.check(case_ts={c: self.frozen["timing"][c]["case_ts"] for c in CASES})
        self.assertEqual((r["cases"], r["windows"]), (4, 24))
        self.assertEqual(r, self.check())
        self.assertEqual(len(r["minute_union_sha256"]), 64)

    def test_minute_union_is_the_distinct_samples_of_the_windows(self):
        u = ce.minute_unions(self.ci, self.ts)
        self.assertEqual([x["n_samples"] for x in u], [12] * 4)                 # 6 windows x 7 minutes with 6-minute overlap -> 12 distinct
        self.assertTrue(all(x["contiguous"] for x in u))

    def fails(self, **kw):
        with self.assertRaises(ce.AlignmentError):
            self.check(**kw)

    def test_shifted_timestamps_fail(self):
        ts = self.ts.copy()
        ts[ts == self.frozen["timing"][103]["first_sample_ts"]] += 60
        self.fails(ts=ts)
        self.fails(ts=self.ts + 1)                                               # not on the frozen minutes

    def test_a_dropped_or_extra_window_fails(self):
        keep = np.ones(len(self.ci), bool)
        keep[5] = False                                                          # last window of case 101: fewer minutes and 5 windows
        self.fails(ci=self.ci[keep], ts=self.ts[keep], fc=self.fc[keep])
        extra_ci = np.append(self.ci, 101)
        extra_ts = np.vstack([self.ts, self.ts[5] + 60])
        self.fails(ci=extra_ci, ts=extra_ts, fc=np.append(self.fc, 0))

    def test_missing_extra_or_renamed_cases_fail(self):
        keep = self.ci != 110
        self.fails(ci=self.ci[keep], ts=self.ts[keep], fc=self.fc[keep])
        ci = np.where(self.ci == 110, 111, self.ci)
        self.fails(ci=ci)
        self.fails(ci=np.append(self.ci, 999), ts=np.vstack([self.ts, self.ts[0]]), fc=np.append(self.fc, 0))

    def test_changed_labels_fail(self):
        fc = np.where(self.ci == 104, 2, self.fc)
        self.fails(fc=fc)

    def test_non_consecutive_minutes_fail(self):
        ts = self.ts.copy()
        ts[0, 3] += 5
        self.fails(ts=ts)

    def test_case_time_class_order_and_window_range_are_checked(self):
        self.fails(case_ts={c: self.frozen["timing"][c]["case_ts"] + (1 if c == 104 else 0) for c in CASES})
        bad = copy.deepcopy(self.frozen)
        bad["provenance"]["class_order"][2], bad["provenance"]["class_order"][3] = bad["provenance"]["class_order"][3], bad["provenance"]["class_order"][2]
        self.fails(frozen=bad)
        bad = copy.deepcopy(self.frozen)
        bad["provenance"]["windows_per_case"]["validation"] = [5, 5]
        self.fails(frozen=bad)

    def test_the_error_names_the_mismatch(self):
        ts = self.ts + 60
        with self.assertRaisesRegex(ce.AlignmentError, "first_sample_ts"):
            self.check(ts=ts)


def synthetic_loader(role="validation", category="event_associated", ambiguous=False, n_batches=2):
    spec, app, _ = MH.synthetic_spec_and_applicable()
    out = []
    for s in range(n_batches):
        b = MH.make_batch(app, B=3, seed=s, holes=0.05)
        b.update({"role": [role] * 3, "category": [category] * 3, "fault_label_ambiguous": torch.full((3,), ambiguous), "fault_class_present": torch.ones(3, dtype=torch.bool),
                  "case_index": torch.tensor([101 + s, 101 + s, 103 + s]), "start_index": torch.arange(3) + 10 * s, "timestamps": torch.arange(21).reshape(3, 7) + 1000 * s,
                  "fault_class": torch.tensor([0, 0, 1])})
        out.append(b)
    return spec, out


class PredictionInterfaceTests(unittest.TestCase):
    def test_probabilities_for_every_window_in_eval_mode_without_gradients(self):
        spec, batches = synthetic_loader()
        model = MH.make_model(spec).train()
        r = ce.predict_window_probabilities(model, batches)
        self.assertEqual(r["probs"].shape, (6, 5))
        np.testing.assert_allclose(r["probs"].sum(1), 1.0, atol=1e-12)
        self.assertEqual(r["probs"].dtype, np.float64)
        self.assertEqual(r["case_index"].tolist(), [101, 101, 103, 102, 102, 104])
        self.assertTrue(model.training)                                           # mode restored
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        model.eval()
        again = ce.predict_window_probabilities(model, batches)
        np.testing.assert_array_equal(r["probs"], again["probs"])                 # eval mode is deterministic (no dropout)
        a = ce.aggregate_cases(r["case_index"], r["probs"], r["fault_class"])
        self.assertEqual(len(a["pred_class"]), 4)

    def test_only_validation_event_windows_are_accepted(self):
        for role in ("calibration", "test", "fit", "embargo"):
            spec, batches = synthetic_loader(role=role)
            with self.assertRaises(ce.RoleLocked):
                ce.predict_window_probabilities(MH.make_model(spec), batches)
        spec, batches = synthetic_loader(category="normal_reference")
        with self.assertRaises(ce.RoleLocked):
            ce.predict_window_probabilities(MH.make_model(spec), batches)
        spec, batches = synthetic_loader(ambiguous=True)
        with self.assertRaises(ce.AlignmentError):
            ce.predict_window_probabilities(MH.make_model(spec), batches)

    def test_other_roles_cannot_be_requested(self):
        spec, batches = synthetic_loader()
        for role in ("test", "calibration", "fit"):
            with self.assertRaises(ce.RoleLocked):
                ce.predict_window_probabilities(MH.make_model(spec), batches, role=role)

    def test_empty_loader_is_an_error(self):
        spec, _ = synthetic_loader()
        with self.assertRaises(ce.AlignmentError):
            ce.predict_window_probabilities(MH.make_model(spec), [])

    def test_the_module_never_names_the_locked_partitions(self):
        with open(ce.__file__) as fh:
            src = fh.read()
        self.assertIsNone(re.search(r"""["'](test|calibration|embargo)["']""", src))
        self.assertIsNone(re.search(r"optim|\.backward\(|\.step\(", src))


@NEEDS_REAL
class RealAlignmentTests(unittest.TestCase):
    """Window metadata of the approved validation selection against the frozen XGBoost provenance (no telemetry is read)."""

    @classmethod
    def setUpClass(cls):
        cls.D = md.Model1Data.load()
        cls.meta = ce.window_metadata_from_table(cls.D)
        cls.frozen = ce.load_frozen()

    def test_exact_approved_windows(self):
        t = self.D.table
        sel = self.D.selection(["validation"], ["event_associated"], drop_ambiguous=True)
        self.assertEqual(len(sel), 150)
        self.assertEqual(sorted(self.meta["start_index"].tolist()), self.D.manifest["windows"]["validation"]["start_index"]["event_associated"])
        self.assertTrue((t.role[sel] == "validation").all() and (t.category[sel] == "event_associated").all())
        self.assertEqual(len(set(self.meta["case_index"].tolist())), 25)
        self.assertEqual(np.bincount(np.unique(self.meta["case_index"], return_counts=True)[1]).tolist(), [0, 0, 0, 0, 0, 0, 25])     # 6 windows per case

    def test_minute_union_equals_the_frozen_xgboost_provenance(self):
        r = ce.check_alignment(self.meta["case_index"], self.meta["timestamps"], self.meta["fault_class"], self.frozen,
                               {int(c): int(self.D.cases["timestamp"][c]) for c in np.unique(self.meta["case_index"])})
        self.assertEqual((r["cases"], r["windows"]), (25, 150))
        self.assertEqual(r["validation_case_index_sha256"], self.frozen["provenance"]["array_sha256"]["validation_case_index"])

    def test_same_sample_selection_as_the_xgboost_feature_builder(self):
        t = self.D.table
        rows = self.D.selection(["validation"], ["event_associated"], drop_ambiguous=True)
        for c in np.unique(t.case_index[rows]):
            idx = cf.case_minutes(t, [r for r in rows if t.case_index[r] == c])
            f = self.frozen["timing"][int(c)]
            self.assertEqual((int(self.D.ts[idx[0]]), int(self.D.ts[idx[-1]]), len(idx)), (f["first_sample_ts"], f["last_sample_ts"], f["n_samples"]))

    def test_no_validation_minute_is_shared_with_the_fitting_cases_and_all_lie_in_the_validation_period(self):
        ts = self.meta["timestamps"]
        self.assertGreaterEqual(int(ts.min()), int(self.D.boundaries[0]))
        self.assertLess(int(ts.max()), int(self.D.boundaries[1]))
        fit = self.D.selection(["fit"], ["event_associated"], drop_ambiguous=True)
        self.assertFalse(set(ts.ravel().tolist()) & set(self.D.table.T[fit].ravel().tolist()))

    def test_labels_and_class_order_match_the_frozen_control(self):
        a = ce.aggregate_cases(self.meta["case_index"], onehot_probs(self.meta["fault_class"]), self.meta["fault_class"])
        self.assertEqual(np.bincount(a["true_class"], minlength=5).tolist(), [10, 6, 3, 5, 1])
        self.assertEqual(self.frozen["provenance"]["class_order"], list(ce.CLASS_NAMES))
        self.assertEqual(self.D.manifest["label_map"]["classes"], list(ce.CLASS_NAMES))
        self.assertEqual(a["pred_class"].tolist(), a["true_class"].tolist())               # a perfect synthetic predictor reproduces the labels in class order


if __name__ == "__main__":
    unittest.main()
