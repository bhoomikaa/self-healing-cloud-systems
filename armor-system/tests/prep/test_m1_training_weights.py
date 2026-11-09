"""Case-balanced class weighting and the weighted per-window fault loss (Linear 298-51)."""
import unittest

import _m1_model_helpers as MH  # noqa: F401
import numpy as np
import torch

import _m1_helpers as H
from model1_transformer import losses as L, weighting as W
from prep import m1_dataset as md

COUNTS = [33, 29, 9, 13, 4]
NEEDS_REAL = H.needs(H.CACHE_PRESENT and H.ARTIFACTS_PRESENT, "needs the D1 feature cache and the M1-03 artifacts")


def synthetic():
    """Cases 0..4, one per class, with 6, 5, 6, 3, 6 eligible windows (case 3 lost windows, like a case with an excluded window)."""
    n = [6, 5, 6, 3, 6]
    case = np.repeat(np.arange(5), n)
    return W.CaseBalancedWeights(case, np.repeat(np.arange(5), n)), n


class ClassWeightTests(unittest.TestCase):
    def test_exact_values_from_the_approved_case_counts(self):
        w = W.class_weights_from_case_counts(COUNTS)
        np.testing.assert_allclose(w, [88 / (5 * n) for n in COUNTS], rtol=1e-15)
        self.assertEqual([round(float(x), 4) for x in w], [0.5333, 0.6069, 1.9556, 1.3538, 4.4])
        self.assertEqual(w.dtype, np.float64)
        self.assertAlmostEqual(float((w * COUNTS).sum()), 88.0, places=12)               # sum_k n_k w_k = N

    def test_rejects_empty_or_invalid_counts(self):
        for bad in ([33, 29, 0, 13, 4], [33, 29, -1, 13, 4], [33, 29.5, 9, 13, 4], [[1, 2]], [5]):
            with self.assertRaises(W.WeightError):
                W.class_weights_from_case_counts(bad)

    def test_agrees_with_the_core_loss_helper(self):
        np.testing.assert_allclose(W.class_weights_from_case_counts(COUNTS), L.inverse_frequency_weights(COUNTS).double().numpy(), rtol=1e-6)


class CaseBalancedTests(unittest.TestCase):
    def test_windows_of_a_case_sum_to_its_class_weight(self):
        cb, n = synthetic()
        cw = W.class_weights_from_case_counts([1, 1, 1, 1, 1])
        for j, c in enumerate(cb.case_ids):
            self.assertAlmostEqual(cb.window_weight_by_case[j] * n[j], cw[cb.case_class[j]], places=12)
        self.assertAlmostEqual(cb.total_weight, 5.0, places=12)                           # = number of cases
        self.assertEqual(cb.n_windows, sum(n))

    def test_fewer_eligible_windows_means_larger_weight_each_but_same_case_total(self):
        cb, n = synthetic()
        w = cb.per_window(torch.tensor([0, 3]), normalised=False, dtype=torch.float64)
        self.assertGreater(w[1].item(), w[0].item())                                      # case 3 has 3 windows, case 0 has 6
        self.assertAlmostEqual(w[0].item() * 6, w[1].item() * 3, places=12)

    def test_normalisation_gives_mean_one_over_the_windows(self):
        cb, n = synthetic()
        all_w = cb.per_window(torch.as_tensor(np.repeat(np.arange(5), n)), dtype=torch.float64)
        self.assertAlmostEqual(all_w.mean().item(), 1.0, places=12)
        raw = cb.per_window(torch.as_tensor(np.repeat(np.arange(5), n)), normalised=False, dtype=torch.float64)
        self.assertAlmostEqual(raw.sum().item(), cb.total_weight, places=12)

    def test_unknown_or_unlabelled_cases_are_rejected(self):
        cb, _ = synthetic()
        for bad in ([7], [-1], [0, 99]):
            with self.assertRaises(W.WeightError):
                cb.per_window(torch.tensor(bad))

    def test_invalid_window_tables_are_rejected(self):
        with self.assertRaises(W.WeightError):
            W.CaseBalancedWeights([0, 0, 1], [0, 1, 1])                                    # case 0 holds two classes
        with self.assertRaises(W.WeightError):
            W.CaseBalancedWeights([0, -1], [0, 1])                                         # window without a case
        with self.assertRaises(W.WeightError):
            W.CaseBalancedWeights([0, 1], [0, -1])                                         # unlabelled window
        with self.assertRaises(W.WeightError):
            W.CaseBalancedWeights([0, 1], [0, 5])
        with self.assertRaises(W.WeightError):
            W.CaseBalancedWeights([], [])


class WeightedLossTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.cb, self.n = synthetic()
        self.case = torch.as_tensor(np.repeat(np.arange(5), self.n))
        self.y = torch.as_tensor(np.repeat(np.arange(5), self.n))
        self.logits = torch.randn(len(self.y), 5, generator=g, dtype=torch.float64)
        self.w = self.cb.per_window(self.case, dtype=torch.float64)

    def test_class_weight_is_applied_once(self):
        ce = torch.nn.functional.cross_entropy(self.logits, self.y, reduction="none")
        self.assertAlmostEqual(W.weighted_window_cross_entropy(self.logits, self.y, self.w).item(), (self.w * ce).mean().item(), places=12)
        ones = torch.ones_like(self.w)                                                     # no weights at all -> plain mean CE (nothing hidden)
        self.assertAlmostEqual(W.weighted_window_cross_entropy(self.logits, self.y, ones).item(), torch.nn.functional.cross_entropy(self.logits, self.y).item(), places=12)

    def test_gradient_matches_the_closed_form(self):
        lg = self.logits.clone().requires_grad_(True)
        W.weighted_window_cross_entropy(lg, self.y, self.w).backward()
        onehot = torch.nn.functional.one_hot(self.y, 5).double()
        expect = self.w[:, None] * (lg.detach().softmax(-1) - onehot) / len(self.y)       # fixed denominator = batch size
        self.assertTrue(torch.allclose(lg.grad, expect, atol=1e-12))

    def test_full_pass_over_any_batching_equals_the_epoch_objective(self):
        full = W.weighted_window_cross_entropy(self.logits, self.y, self.w).item()
        perm = torch.randperm(len(self.y), generator=torch.Generator().manual_seed(3))
        total = 0.0
        for chunk in (perm[:4], perm[4:11], perm[11:12], perm[12:]):                      # uneven batches, including a final short one
            total += len(chunk) * W.weighted_window_cross_entropy(self.logits[chunk], self.y[chunk], self.w[chunk]).item()
        self.assertAlmostEqual(total / len(self.y), full, places=12)

    def test_batch_scale_does_not_depend_on_the_other_classes_in_the_batch(self):
        """Fixed denominator: the contribution of a window is w~_i * ce_i / B whatever else is in the batch."""
        one = lambda idx: W.weighted_window_cross_entropy(self.logits[idx], self.y[idx], self.w[idx]).item() * len(idx)   # noqa: E731
        a, b = torch.tensor([0, 1, 2]), torch.tensor([20, 21, 22])
        self.assertAlmostEqual(one(torch.cat([a, b])), one(a) + one(b), places=12)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(L.LossInputError):
            W.weighted_window_cross_entropy(self.logits, torch.full_like(self.y, -1), self.w)
        with self.assertRaises(L.LossInputError):
            W.weighted_window_cross_entropy(self.logits, torch.full_like(self.y, 5), self.w)
        with self.assertRaises(L.LossInputError):
            W.weighted_window_cross_entropy(self.logits, self.y.int(), self.w)
        with self.assertRaises(L.LossInputError):
            W.weighted_window_cross_entropy(self.logits, self.y, self.w[:-1])
        with self.assertRaises(L.LossInputError):
            W.weighted_window_cross_entropy(self.logits, self.y, torch.zeros_like(self.w))
        with self.assertRaises(L.LossInputError):
            W.weighted_window_cross_entropy(self.logits[:0], self.y[:0], self.w[:0])


@NEEDS_REAL
class RealWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = md.Model1Data.load()
        cls.cb = W.CaseBalancedWeights.from_model1_data(cls.D)

    def test_approved_counts_and_exact_weights(self):
        s = self.cb.summary()
        self.assertEqual((s["cases"], s["eligible_windows"], s["case_counts_by_class"]), (88, 523, COUNTS))
        self.assertEqual([round(x, 4) for x in s["class_weights"]], [0.5333, 0.6069, 1.9556, 1.3538, 4.4])
        self.assertAlmostEqual(s["total_weight"], 88.0, places=9)
        self.assertEqual(s["windows_per_case_min_max"], [5, 6])

    def test_every_fitting_case_sums_to_its_class_weight(self):
        ds = md.training_dataset(self.D, "s2_fault_finetune")
        t = self.D.table
        case, cls = t.case_index[ds.rows], t.fault_class[ds.rows]
        raw = self.cb.per_window(torch.as_tensor(case), normalised=False, dtype=torch.float64).numpy()
        for c in np.unique(case):
            self.assertAlmostEqual(raw[case == c].sum(), self.cb.class_weights[cls[case == c][0]], places=12)
        self.assertAlmostEqual(self.cb.per_window(torch.as_tensor(case), dtype=torch.float64).mean().item(), 1.0, places=12)

    def test_the_ambiguous_conflicting_window_is_excluded(self):
        t = self.D.table
        eligible = set(md.training_dataset(self.D, "s2_fault_finetune").rows.tolist())
        all_fit_events = set(self.D.selection(["fit"], ["event_associated"]).tolist())
        dropped = sorted(all_fit_events - eligible)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(int(t.start[dropped[0]]), 2550)
        self.assertTrue(bool(t.ambiguous[dropped[0]]))
        self.assertEqual(int(t.fault_class[dropped[0]]), -1)
        self.assertEqual(self.cb.n_windows, len(eligible))

    def test_validation_calibration_and_unknown_cases_have_no_weight(self):
        t = self.D.table
        for role in ("validation", "calibration"):
            r = self.D.selection([role], ["event_associated"], drop_ambiguous=True)
            with self.assertRaises(W.WeightError):
                self.cb.per_window(torch.as_tensor(t.case_index[r[:3]]))
        with self.assertRaises(W.WeightError):
            self.cb.per_window(torch.tensor([-1]))


if __name__ == "__main__":
    unittest.main()
