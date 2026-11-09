"""MI-1 .. MI-7 of design note 3.4 for the Model 1 v0 core (Linear 298-50). CPU only.

MI-1..MI-5 and MI-7 run on D1-shaped synthetic batches (46 nodes, 130 channels, the real mask counts). MI-6 is the data
check on the real feature cache (and a synthetic violation proving the check can fail).
"""
import unittest

import _m1_model_helpers as MH
import numpy as np
import torch

import _m1_helpers as H
from model1_transformer import losses as L
from model1_transformer.model import Model1
from model1_transformer.spec import AdapterSpec
from prep import detection_calibration as dc, m1_masks, paths

W = L.inverse_frequency_weights([33, 29, 9, 13, 4])


def scores(pred, target, mask, node_valid=None):
    """Per-node errors, per-node normalised scores and window scores exactly as the evaluation uses them (numpy, float64)."""
    err, nv = dc.node_errors(pred.detach().numpy(), target.numpy(), mask.numpy())
    stats = dc.fit_statistics(err, node_valid if node_valid is not None else nv.any(0))
    return err, dc.normalise(err, stats), dc.window_scores(err, stats)


def total_loss(out, b):
    return L.weighted_cross_entropy(out["logits"], b["labels"], W) + L.masked_mse(out["prediction"], b["target"], b["target_mask"])


class InvarianceBase(unittest.TestCase):
    def setUp(self):
        self.spec, self.app, _ = MH.synthetic_spec_and_applicable()
        self.model = MH.make_model(self.spec, seed=1)
        self.b = MH.make_batch(self.app, B=4, seed=5, holes=0.15)           # applicable AND dynamically observed -> effective

    def run_model(self, b, model=None, **kw):
        return (model or self.model)(b["inputs"], b["input_mask"], applicable_entry=b["applicable_entry"], **kw)


class MI1ValueInvariance(InvarianceBase):
    def test_unavailable_input_and_target_values_do_not_change_anything(self):
        b = self.b
        g = torch.Generator().manual_seed(9)
        huge = lambda shape: (torch.randn(shape, generator=g) * 1e6) + 1e30 * (torch.rand(shape, generator=g) > 0.5)   # noqa: E731
        b2 = dict(b)
        b2["inputs"] = torch.where(b["input_mask"], b["inputs"], huge(b["inputs"].shape))
        b2["target"] = torch.where(b["target_mask"], b["target"], huge(b["target"].shape))
        self.assertGreater((b2["inputs"] != b["inputs"]).float().mean().item(), 0.2)       # a real perturbation, not a no-op
        o1, o2 = self.run_model(b), self.run_model(b2)
        self.assertTrue(torch.equal(o1["logits"], o2["logits"]))
        self.assertTrue(torch.equal(o1["prediction"], o2["prediction"]))
        self.assertEqual(total_loss(o1, b).item(), total_loss(o2, b2).item())
        e1, z1, s1 = scores(o1["prediction"], b["target"], b["target_mask"])
        e2, z2, s2 = scores(o2["prediction"], b2["target"], b2["target_mask"])
        np.testing.assert_allclose(e1, e2, atol=1e-6, rtol=0)
        np.testing.assert_allclose(z1, z2, atol=1e-6, rtol=0)
        for k in s1:
            np.testing.assert_allclose(s1[k], s2[k], atol=1e-6, rtol=0)

    def test_nan_and_inf_in_unavailable_slots_are_harmless(self):
        b = self.b
        bad = torch.where(b["input_mask"], b["inputs"], torch.tensor(float("nan")))
        bad[0, 0, 0, :5] = torch.where(b["input_mask"][0, 0, 0, :5], bad[0, 0, 0, :5], torch.tensor(float("inf")))
        o1, o2 = self.run_model(b), self.run_model(dict(b, inputs=bad))
        self.assertTrue(torch.equal(o1["logits"], o2["logits"]) and torch.equal(o1["prediction"], o2["prediction"]))


class MI2Gradient(InvarianceBase):
    def check(self, train_mode):
        self.model.train(train_mode)
        losses = {"fault": lambda o, b: L.weighted_cross_entropy(o["logits"], b["labels"], W),
                  "next_step": lambda o, b: L.masked_mse(o["prediction"], b["target"], b["target_mask"]),
                  "both": total_loss}
        for name, fn in losses.items():
            torch.manual_seed(0)
            x = torch.where(self.b["input_mask"], self.b["inputs"], torch.full_like(self.b["inputs"], 1e8)).requires_grad_(True)
            o = self.model(x, self.b["input_mask"], applicable_entry=self.b["applicable_entry"])
            fn(o, self.b).backward()
            self.assertTrue((x.grad[~self.b["input_mask"]] == 0).all(), name)
            self.assertGreater(x.grad[self.b["input_mask"]].abs().max().item(), 0, name)         # the effective entries do get gradient
            self.assertTrue(torch.isfinite(x.grad).all(), name)

    def test_gradient_at_unavailable_input_entries_is_exactly_zero_eval(self):
        self.check(False)

    def test_gradient_at_unavailable_input_entries_is_exactly_zero_with_dropout(self):
        self.check(True)


class MI3NotVacuous(InvarianceBase):
    def test_changing_a_legitimate_zero_changes_the_output(self):
        pod, vm = int(np.nonzero(self.spec.node_type == 0)[0][0]), int(np.nonzero(self.spec.node_type == 1)[0][0])
        for node, ch in ((pod, 7), (pod, 55), (vm, 90)):                                      # pod metric, pod trace, VM metric
            for minute in (0, 5):
                b = {k: v.clone() if torch.is_tensor(v) else v for k, v in self.b.items()}
                b["input_mask"][:, minute, node, ch] = True
                self.assertTrue(self.app[node, ch])
                b["inputs"][:, minute, node, ch] = 0.0                                          # a legitimate zero at an effective entry
                b2 = dict(b, inputs=b["inputs"].clone())
                b2["inputs"][:, minute, node, ch] = 0.5
                o1, o2 = self.run_model(b), self.run_model(b2)
                self.assertGreater((o1["logits"] - o2["logits"]).abs().max().item(), 0, (node, ch, minute))
                self.assertGreater((o1["prediction"] - o2["prediction"]).abs().max().item(), 0, (node, ch, minute))


class MI4NodeTypeEquivariance(InvarianceBase):
    def check(self, dtype, atol):
        model = MH.make_model(self.spec, seed=1, dtype=dtype)
        b = {k: (v.to(dtype) if torch.is_floating_point(v) else v) for k, v in self.b.items()}
        base = self.run_model(b, model)
        rng = np.random.default_rng(3)
        for k in (0, 1):
            idx = np.nonzero(self.spec.node_type == k)[0]
            pi = np.arange(46)
            pi[idx] = idx[rng.permutation(len(idx))]
            self.assertFalse((pi == np.arange(46)).all())
            pi = torch.as_tensor(pi)
            bp = dict(b, inputs=b["inputs"][:, :, pi], input_mask=b["input_mask"][:, :, pi], applicable_entry=b["applicable_entry"][pi],
                      target=b["target"][:, pi], target_mask=b["target_mask"][:, pi])
            out = self.run_model(bp, model)
            self.assertLessEqual((out["logits"] - base["logits"]).abs().max().item(), atol)
            self.assertLessEqual((out["prediction"] - base["prediction"][:, pi]).abs().max().item(), atol)
            self.assertAlmostEqual(L.masked_mse(out["prediction"], bp["target"], bp["target_mask"]).item(),
                                   L.masked_mse(base["prediction"], b["target"], b["target_mask"]).item(), delta=atol)

    def test_float64(self):
        self.check(torch.float64, 1e-9)

    def test_float32(self):
        self.check(torch.float32, 1e-5)


class MI5Padding(InvarianceBase):
    def check(self, dtype, atol, interleaved):
        model = MH.make_model(self.spec, seed=1, dtype=dtype)
        b = {k: (v.to(dtype) if torch.is_floating_point(v) else v) for k, v in self.b.items()}
        base = self.run_model(b, model, return_aux=True)
        B, T, N, C = b["inputs"].shape
        k = 6
        pos = torch.arange(N) + (torch.arange(N) // 8) * 1 if interleaved else torch.arange(N)        # real nodes keep their order
        n_tot = int(pos.max()) + 1 + (0 if interleaved else k)
        if interleaved:
            n_tot = N + k
            pos = torch.as_tensor(sorted(np.random.default_rng(0).choice(n_tot, N, replace=False)))
        pad = torch.ones(n_tot, dtype=torch.bool)
        pad[pos] = False
        g = torch.Generator().manual_seed(2)
        x = (torch.randn(B, T, n_tot, C, generator=g, dtype=dtype) * 1e6)
        x[:, :, torch.nonzero(pad).flatten(), :3] = float("nan")                              # garbage in the padded slots, even NaN
        m = torch.rand(B, T, n_tot, C, generator=g) > 0.3                                    # padded slots even claim to be observed
        x[:, :, pos], m[:, :, pos] = b["inputs"], b["input_mask"]
        nt = torch.ones(n_tot, dtype=torch.int64)
        nt[pos] = torch.as_tensor(self.spec.node_type)
        nv = ~pad[None].expand(B, n_tot)
        app_pad = torch.zeros(n_tot, C, dtype=torch.bool)
        app_pad[pos] = b["applicable_entry"]
        out = model(x, m, node_valid=nv, node_type=nt, applicable_entry=app_pad, return_aux=True)
        self.assertLessEqual((out["logits"] - base["logits"]).abs().max().item(), atol)
        self.assertLessEqual((out["prediction"][:, pos] - base["prediction"]).abs().max().item(), atol)
        self.assertTrue((out["prediction"][:, pad] == 0).all() and (out["hidden"][:, :, pad] == 0).all())
        self.assertTrue((out["fusion_weights"][:, :, pad] == 0).all())
        for key in ("logits", "prediction"):
            self.assertTrue(torch.isfinite(out[key]).all())
        # scores: per-node errors on the valid nodes and the window scores are unchanged
        tgt = torch.zeros(B, n_tot, C, dtype=dtype)
        tm = torch.zeros(B, n_tot, C, dtype=torch.bool)
        tgt[:, pos], tm[:, pos] = b["target"], b["target_mask"]
        e0, z0, s0 = scores(base["prediction"].double(), b["target"].double(), b["target_mask"])
        e1, z1, s1 = scores(out["prediction"].double(), tgt.double(), tm, node_valid=tm.any(-1).any(0))
        np.testing.assert_allclose(e1[:, pos], e0, atol=atol, rtol=0)
        for key in s0:
            np.testing.assert_allclose(s1[key], s0[key], atol=max(atol, 1e-6) * 10, rtol=0)

    def test_appended_padding_float64(self):
        self.check(torch.float64, 1e-9, False)

    def test_interleaved_padding_float64(self):
        self.check(torch.float64, 1e-9, True)

    def test_padding_float32(self):
        self.check(torch.float32, 1e-5, True)


class MI7AvailabilityHandling(InvarianceBase):
    def test_a_whole_modality_unavailable_gets_exactly_zero_fusion_weight(self):
        pod = int(np.nonzero(self.spec.node_type == 0)[0][14])                               # a pod whose log block exists
        self.assertTrue(self.app[pod, 62:73].all())
        b = {k: v.clone() if torch.is_tensor(v) else v for k, v in self.b.items()}
        b["input_mask"][:, :, pod, 53:62] = False                                            # all trace channels of one pod, all minutes
        b["input_mask"][:, 2, pod, 62:73] = False                                            # its log channels, one minute only
        o = self.run_model(b, return_aux=True)
        w = o["fusion_weights"]
        self.assertTrue((w[:, :, pod, 1] == 0).all())
        self.assertTrue((w[:, 2, pod, 2] == 0).all())
        self.assertTrue((w[:, :, pod, 0] > 0).all() and (w[:, 0, pod, 2] > 0).all())
        for key in ("logits", "prediction", "hidden"):
            self.assertTrue(torch.isfinite(o[key]).all())

    def test_vm_trace_and_log_never_have_weight_and_all_unavailable_gives_zero_weights(self):
        o = self.run_model(self.b, return_aux=True)
        vm = torch.as_tensor(self.spec.node_type == 1)
        self.assertTrue((o["fusion_weights"][:, :, vm][..., 1:] == 0).all())
        self.assertFalse(o["token_avail"][:, :, vm][..., 1:].any())
        b = dict(self.b, input_mask=torch.zeros_like(self.b["input_mask"]))
        o = self.run_model(b, return_aux=True)
        self.assertTrue((o["fusion_weights"] == 0).all())
        self.assertTrue(torch.isfinite(o["logits"]).all() and torch.isfinite(o["prediction"]).all())

    def test_an_unavailable_modality_does_not_change_the_available_ones(self):
        """Dropping a modality equals fusing the others only: the removed modality's values are irrelevant."""
        pod = int(np.nonzero(self.spec.node_type == 0)[0][0])
        b = {k: v.clone() if torch.is_tensor(v) else v for k, v in self.b.items()}
        b["input_mask"][:, :, pod, 53:62] = False
        b2 = dict(b, inputs=b["inputs"].clone())
        b2["inputs"][:, :, pod, 53:62] = 123.0
        self.assertTrue(torch.equal(self.run_model(b)["logits"], self.run_model(b2)["logits"]))


@H.needs(H.CACHE_PRESENT and H.ARTIFACTS_PRESENT, "needs the D1 feature cache and the M1-03 artifacts")
class MI6MaskStability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from prep import loader, m1_dataset as md
        cls.D = md.Model1Data.load()
        cls.ts, cls.X, _ = loader.load_arrays()

    def periods(self):
        D, ts = self.D, self.ts
        c, bd = D.cfg, D.boundaries
        return {"fit": ts < c.tc - c.embargo_s, "calibration_region": (ts >= c.tc) & (ts < bd[0]),
                "validation": (ts >= bd[0]) & (ts < bd[1]), "test": ts >= bd[1]}

    def test_no_nonzero_value_outside_the_applicable_mask_in_any_period(self):
        out = m1_masks.check_mask_stability(self.X, self.D.applicable, self.periods(), self.D.schema)
        self.assertEqual(out, {"fit": 0, "calibration_region": 0, "validation": 0, "test": 0})

    def test_the_model_channel_lists_cover_every_non_zero_entry(self):
        spec = AdapterSpec.from_d1(self.D.schema, self.D.applicable)
        covered = np.zeros(self.D.applicable.shape, bool)
        for k in range(2):
            covered[np.ix_(np.nonzero(spec.node_type == k)[0], spec.output_channels(k))] = True
        self.assertEqual(int(((self.X != 0).any(0) & ~covered).sum()), 0)
        self.assertEqual(int(self.D.applicable.sum()), 3152)

    def test_a_violation_fails_the_check(self):
        X = self.X[:50].copy()
        i, j = np.argwhere(~self.D.applicable)[0]
        X[3, i, j] = 1e-3
        with self.assertRaises(m1_masks.MaskError):
            m1_masks.check_mask_stability(X, self.D.applicable, None, self.D.schema)


if __name__ == "__main__":
    unittest.main()
