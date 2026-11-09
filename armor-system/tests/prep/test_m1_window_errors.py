"""Per-window masked prediction errors and alert timestamps (scoring harness, Linear 298-51). Synthetic windows only; no real data."""
import os
import re
import sys
import unittest

import _m1_model_helpers as MH
import numpy as np
import torch

from model1_transformer import window_errors as we
from prep import detection_calibration as dc

DET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "detection"))
if DET not in sys.path:
    sys.path.insert(0, DET)

import armor_scoring as sc  # noqa: E402


def synthetic_loader(role="validation", n_batches=2, B=3, category="normal_reference", holes=0.1):
    spec, app, _ = MH.synthetic_spec_and_applicable()
    out = []
    for s in range(n_batches):
        b = MH.make_batch(app, B=B, seed=s, holes=holes)
        b.update({"role": [role] * B, "category": [category] * B, "case_index": torch.full((B,), -1), "start_index": torch.arange(B) + 100 * s,
                  "timestamps": 1_000_000 + 60 * (torch.arange(B)[:, None] + torch.arange(7)[None, :]) + 10_000 * s})
        out.append(b)
    return spec, out


class ErrorExtractionTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.batches = synthetic_loader()
        self.model = MH.make_model(self.spec, seed=1)

    def manual(self, b):
        with torch.no_grad():
            pred = self.model(b["inputs"], b["input_mask"], applicable_entry=b["applicable_entry"])["prediction"].double().numpy()
        tgt, m = b["target"].double().numpy(), b["target_mask"].numpy()
        W, N, _ = pred.shape
        err, valid = np.zeros((W, N)), np.zeros((W, N), bool)
        for w in range(W):
            for n in range(N):
                e = m[w, n]
                if e.any():
                    err[w, n] = ((pred[w, n][e] - tgt[w, n][e]) ** 2).mean()
                    valid[w, n] = True
        return err, valid

    def test_errors_are_the_per_node_mean_over_effective_entries_of_the_target_minute(self):
        r = we.collect_window_errors(self.model, self.batches)
        ref = [self.manual(b) for b in self.batches]
        np.testing.assert_allclose(r["errors"], np.concatenate([x[0] for x in ref]), rtol=1e-12, atol=1e-12)
        self.assertTrue((r["node_valid"] == np.concatenate([x[1] for x in ref])).all())
        self.assertEqual(r["errors"].shape, (6, 46))
        self.assertTrue(r["node_valid"].all())                                      # every D1 node has effective entries

    def test_unavailable_target_values_do_not_change_any_error(self):
        r1 = we.collect_window_errors(self.model, self.batches)
        polluted = [dict(b, target=torch.where(b["target_mask"], b["target"], torch.full_like(b["target"], 1e30))) for b in self.batches]
        r2 = we.collect_window_errors(self.model, polluted)
        self.assertTrue(np.array_equal(r1["errors"], r2["errors"]))

    def test_a_node_without_effective_entries_is_invalid_with_zero_error(self):
        bs = [dict(b) for b in self.batches]
        bs[0]["target_mask"] = bs[0]["target_mask"].clone()
        bs[0]["target_mask"][1, 5] = False
        r = we.collect_window_errors(self.model, bs)
        self.assertFalse(r["node_valid"][1, 5])
        self.assertEqual(r["errors"][1, 5], 0.0)
        self.assertTrue(r["node_valid"][0, 5])

    def test_metadata_alert_time_is_the_seventh_sample_and_order_is_kept(self):
        r = we.collect_window_errors(self.model, self.batches)
        want = np.concatenate([b["timestamps"][:, 6].numpy() for b in self.batches])
        self.assertTrue(np.array_equal(r["target_ts"], want))
        self.assertEqual(r["start_index"].tolist(), [0, 1, 2, 100, 101, 102])
        self.assertEqual(set(r["role"]), {"validation"})
        self.assertEqual(set(r["category"]), {"normal_reference"})

    def test_eval_mode_no_gradient_mode_restored_and_deterministic(self):
        self.model.train()
        r1 = we.collect_window_errors(self.model, self.batches)
        self.assertTrue(self.model.training)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        r2 = we.collect_window_errors(self.model, self.batches)
        self.assertTrue(np.array_equal(r1["errors"], r2["errors"]))                  # dropout is off in eval mode

    def test_calibration_windows_are_scorable(self):
        spec, bs = synthetic_loader(role="calibration")
        r = we.collect_window_errors(MH.make_model(spec), bs)
        self.assertEqual(set(r["role"]), {"calibration"})

    def test_fitting_embargo_and_test_windows_cannot_be_scored(self):
        for role in ("fit", "embargo", "test"):
            spec, bs = synthetic_loader(role=role)
            with self.assertRaises(we.ScoringRoleError):
                we.collect_window_errors(MH.make_model(spec), bs)
        spec, bs = synthetic_loader()
        bs[1]["role"][0] = "test"
        with self.assertRaises(we.ScoringRoleError):
            we.collect_window_errors(MH.make_model(spec), bs)
        with self.assertRaises(we.ScoringRoleError):
            we.collect_window_errors(MH.make_model(spec), bs, allowed_roles=("validation", "test"))
        with self.assertRaises(ValueError):
            we.collect_window_errors(MH.make_model(spec), [])


class AlertTimestampTests(unittest.TestCase):
    def test_ascending_plain_ints_of_the_flagged_minutes(self):
        ts = np.array([1_000_240, 1_000_000, 1_000_120, 1_000_060, 1_000_180])
        out = we.alert_timestamps(ts, [True, False, True, True, False])
        self.assertEqual(out, [1_000_060, 1_000_120, 1_000_240])
        self.assertTrue(all(type(x) is int for x in out))
        self.assertEqual(we.alert_timestamps(ts, [False] * 5), [])

    def test_duplicates_and_shape_errors_are_rejected(self):
        with self.assertRaises(ValueError):
            we.alert_timestamps([5, 5, 6], [True, True, False])
        with self.assertRaises(ValueError):
            we.alert_timestamps([1, 2, 3], [True, False])
        with self.assertRaises(ValueError):
            we.alert_timestamps([[1, 2]], [[True, False]])


class SyntheticCompositionTests(unittest.TestCase):
    def test_calibration_scores_thresholds_alert_intervals_and_event_scoring_compose(self):
        """Synthetic errors only: calibrate on quiet windows, score a period with two injected anomalies, build ARMOR intervals, score hits."""
        rng = np.random.default_rng(0)
        cal = rng.gamma(2.0, 0.05, size=(329, 46))                                    # synthetic calibration-reference errors
        c = dc.calibrate(cal)
        tau = c["thresholds"]["m2_max_normalised"]
        n = 300
        ts = 2_000_000 + 60 * np.arange(n)
        err = rng.gamma(2.0, 0.05, size=(n, 46))
        err[100:106, 7] += 3.0                                                        # anomaly A around minute 100 (six minutes)
        err[200:204, 20] += 3.0                                                       # anomaly B around minute 200
        z = dc.window_scores(err, c["stats"])["m2_max_normalised"]
        alerts = we.alert_timestamps(ts, z > tau)
        iv = sc.get_pre_interval(alerts)
        labels = [int(ts[102]) + 5, int(ts[201]) + 5]
        d = sc.eval_AD_detail(iv, labels)
        self.assertEqual((d["TP"], d["FN"]), (2, 0))
        self.assertGreaterEqual(len(iv), 2)


class NoLockedAccessTests(unittest.TestCase):
    def test_window_errors_reaches_no_training_checkpoint_dataset_or_test_code(self):
        with open(we.__file__) as fh:
            src = fh.read()
        self.assertIsNone(re.search(r"^\s*(import|from)\s+\S*(training|checkpoint|weighting|m1_dataset|m1_split|case_eval)", src, flags=re.M))
        self.assertIsNone(re.search(r"""["'](test|embargo|fit)["']|\.backward\(|optim""", src))
        self.assertEqual(we.SCORABLE_ROLES, ("calibration", "validation"))


if __name__ == "__main__":
    unittest.main()
