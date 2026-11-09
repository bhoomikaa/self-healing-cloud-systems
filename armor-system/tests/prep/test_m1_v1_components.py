"""v1 components: detection head model, simulated-missing masks, paired-batch guards, causal rolling maximum, v1 losses and training guards."""
import os
import re
import unittest
from unittest import mock

import _m1_model_helpers as MH
import numpy as np
import torch

from model1_transformer import paired, rolling, sim_missing as sm, training as T, training_v1 as tv, weighting as W
from model1_transformer.model import Model1
from model1_transformer.model_v1 import Model1V1
from model1_transformer.spec import ModelConfig


def batch_with_starts(app, B=4, seed=0, offset=0, holes=0.0):
    b = MH.make_batch(app, B=B, seed=seed, holes=holes)
    b["start_index"] = torch.arange(B) + offset
    return b


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.app, _ = MH.synthetic_spec_and_applicable()

    def test_parameter_count_and_v0_equivalence_of_the_shared_modules(self):
        m, base = Model1V1.build(self.spec, seed=3), Model1.build(self.spec, seed=3)
        self.assertEqual((base.n_parameters(), m.n_parameters(), m.n_parameters() - base.n_parameters()), (168200, 176521, 8321))
        self.assertEqual(sum(p.numel() for p in m.detection_head.parameters()), 8321)
        for k, v in base.state_dict().items():
            self.assertTrue(torch.equal(m.state_dict()[k], v), k)                       # the new head is created after the v0 modules

    def test_v0_outputs_are_preserved(self):
        m, base = Model1V1.build(self.spec, seed=1).eval(), Model1.build(self.spec, seed=1).eval()
        b = MH.make_batch(self.app, B=3, holes=0.1)
        o1, o0 = m(b["inputs"], b["input_mask"], applicable_entry=b["applicable_entry"]), base(b["inputs"], b["input_mask"], applicable_entry=b["applicable_entry"])
        self.assertTrue(torch.equal(o1["logits"], o0["logits"]) and torch.equal(o1["prediction"], o0["prediction"]))
        self.assertEqual(tuple(o1["detection_logit"].shape), (3,))
        self.assertTrue(torch.isfinite(o1["detection_logit"]).all())
        self.assertNotIn("hidden", o1)
        self.assertIn("hidden", m(b["inputs"], b["input_mask"], return_aux=True))

    def test_detection_logit_ignores_unavailable_values_and_padded_nodes(self):
        m = Model1V1.build(self.spec, seed=2).eval()
        b = MH.make_batch(self.app, B=2, seed=4, holes=0.2)
        base = m(b["inputs"], b["input_mask"])["detection_logit"]
        polluted = torch.where(b["input_mask"], b["inputs"], torch.full_like(b["inputs"], 1e30))
        self.assertTrue(torch.equal(base, m(polluted, b["input_mask"])["detection_logit"]))            # MI-1 for the new head
        nv = torch.ones(2, 46, dtype=torch.bool)
        nv[:, 3] = False
        x = b["inputs"].clone()
        x[:, :, 3] = float("nan")
        mk = b["input_mask"].clone()
        mk[:, :, 3] = True
        a, c = m(b["inputs"], b["input_mask"], node_valid=nv)["detection_logit"], m(x, mk, node_valid=nv)["detection_logit"]
        self.assertTrue(torch.equal(a, c))                                                             # garbage in a padded slot changes nothing

    def test_gradients_reach_the_detection_head_and_the_shared_encoder(self):
        m = Model1V1.build(self.spec, seed=2)
        b = MH.make_batch(self.app, B=2)
        m(b["inputs"], b["input_mask"])["detection_logit"].sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.detection_head.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.core.parameters()))
        self.assertTrue(all(p.grad is None for p in m.fault_head.parameters()))


class SimulatedMissingTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.app, _ = MH.synthetic_spec_and_applicable()
        self.blocks = sm.MissingBlocks(self.spec)
        self.b = batch_with_starts(self.app, B=6, holes=0.05)

    def run_sm(self, b=None, seed=0, stage="s1", epoch=1, rate=sm.RATE):
        return sm.simulate_missing(b or self.b, self.blocks, seed, stage, epoch, rate)

    def test_deterministic_and_independent_of_batch_order_and_size(self):
        a = self.run_sm()
        again = self.run_sm()
        for k in a:
            self.assertTrue(torch.equal(a[k], again[k]))
        perm = torch.tensor([3, 0, 5, 1, 4, 2])
        shuffled = {k: (v[perm] if torch.is_tensor(v) and v.shape[:1] == (6,) else v) for k, v in self.b.items()}
        s = self.run_sm(shuffled)
        self.assertTrue(torch.equal(s["artificial_missing"], a["artificial_missing"][perm]))             # a window's draws depend on its own key only
        half = {k: (v[:3] if torch.is_tensor(v) and v.shape[:1] == (6,) else v) for k, v in self.b.items()}
        self.assertTrue(torch.equal(self.run_sm(half)["artificial_missing"], a["artificial_missing"][:3]))

    def test_keys_change_the_masks(self):
        a = self.run_sm()["artificial_missing"]
        for kw in ({"seed": 1}, {"stage": "s2"}, {"epoch": 2}):
            self.assertFalse(torch.equal(a, self.run_sm(**kw)["artificial_missing"]), kw)
        other = batch_with_starts(self.app, B=6, offset=100, holes=0.05)
        self.assertFalse(torch.equal(a, self.run_sm(other)["artificial_missing"]))

    def test_empirical_rate_is_twenty_percent_of_applicable_blocks(self):
        b = batch_with_starts(self.app, B=64)
        art = self.run_sm(b)["artificial_missing"]
        block_app = self.blocks.block_applicable(self.app)                                               # (N, M)
        hidden_blocks = np.zeros((64, 6, 46, 3), bool)
        for m in range(3):
            for n in range(46):
                cols = self.blocks.chan_block[n, m]
                if cols.any():
                    hidden_blocks[:, :, n, m] = art[:, :, n][..., torch.as_tensor(cols)].any(-1).numpy()
        frac = hidden_blocks[:, :, block_app].mean()
        self.assertAlmostEqual(float(frac), 0.2, delta=0.02)

    def test_natural_absence_is_never_hidden_and_true_zeros_are_not_special(self):
        r = self.run_sm(rate=1.0)
        absent = ~self.b["applicable_entry"][None, None].expand_as(self.b["input_mask"])
        self.assertFalse(r["artificial_missing"][absent].any())                                          # not applicable: never droppable, never counted
        self.assertTrue((r["input_mask"] == (self.b["input_mask"] & ~r["artificial_missing"])).all())
        # at rate 1 every applicable block is hidden, so every previously effective entry is artificial
        self.assertTrue(torch.equal(r["artificial_missing"], self.b["input_mask"]))
        self.assertFalse(r["input_mask"].any())
        none = self.run_sm(rate=0.0)
        self.assertTrue(torch.equal(none["input_mask"], self.b["input_mask"]) and not none["artificial_missing"].any())
        zero = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.b.items()}
        idx = torch.nonzero(zero["input_mask"][0, 0, 10], as_tuple=False)[0, 0]
        zero["inputs"][0, 0, 10, idx] = 0.0                                                              # a legitimate zero at an effective entry
        self.assertTrue(self.run_sm(zero, rate=0.0)["input_mask"][0, 0, 10, idx])                        # stays visible and effective unless a draw hides its block

    def test_a_hidden_block_hides_exactly_that_nodes_modality_channels_at_that_minute(self):
        r = self.run_sm()
        art = r["artificial_missing"].numpy()
        N = 46
        for b in range(2):
            for t in range(6):
                for n in range(N):
                    hit = art[b, t, n]
                    if not hit.any():
                        continue
                    blocks_hit = [m for m in range(3) if self.blocks.chan_block[n, m][hit].any()]
                    for m in blocks_hit:
                        eff = self.b["input_mask"][b, t, n].numpy() & self.blocks.chan_block[n, m]
                        self.assertTrue(art[b, t, n][eff].all())                                         # the whole block's effective entries
                    union = np.any([self.blocks.chan_block[n, m] for m in blocks_hit], axis=0)
                    self.assertFalse(art[b, t, n][~union].any())                                         # and nothing outside the hidden blocks

    def test_targets_are_untouched_and_the_input_batch_is_not_modified(self):
        before = {k: v.clone() for k, v in self.b.items() if torch.is_tensor(v)}
        r = self.run_sm()
        self.assertEqual(set(r), {"inputs", "input_mask", "artificial_missing"})                        # no target field is produced
        for k, v in before.items():
            self.assertTrue(torch.equal(v, self.b[k]), k)
        self.assertTrue((r["inputs"][r["artificial_missing"]] == 0).all())

    def test_the_torch_rng_is_not_consumed(self):
        torch.manual_seed(5)
        state = torch.get_rng_state()
        self.run_sm()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))


class PairedGuardTests(unittest.TestCase):
    def good(self, n=3):
        cats = ["event_associated"] * n + ["normal_reference"] * n
        return {"role": ["fit"] * (2 * n), "category": cats, "det_label": torch.tensor([1] * n + [0] * n), "fault_label_ambiguous": torch.zeros(2 * n, dtype=torch.bool),
                "fault_class_present": torch.tensor([True] * n + [False] * n), "fault_class": torch.tensor([0, 2, 4] * 1 + [-1] * n)[: 2 * n] if n == 3 else torch.zeros(2 * n, dtype=torch.int64)}

    def test_a_clean_paired_batch_passes(self):
        paired.check_paired_batch(self.good())

    def test_every_forbidden_window_is_refused(self):
        for mutate in (lambda b: b.update(role=["validation"] + b["role"][1:]), lambda b: b.update(role=["calibration"] * 6), lambda b: b.update(role=["test"] * 6),
                       lambda b: b["category"].__setitem__(4, "ignore"), lambda b: b["category"].__setitem__(0, "normal_reference"),
                       lambda b: b.update(det_label=torch.tensor([1, 1, 1, 1, 0, 0])), lambda b: b.update(fault_label_ambiguous=torch.tensor([True, False, False, False, False, False])),
                       lambda b: b.update(fault_class=torch.tensor([0, -1, 4, -1, -1, -1]), fault_class_present=torch.tensor([True, False, True, False, False, False]))):
            b = self.good()
            mutate(b)
            with self.assertRaises(T.LeakageError):
                paired.check_paired_batch(b)

    def test_unequal_halves_are_refused(self):
        b = self.good()
        b["role"], b["category"] = b["role"][:5], b["category"][:5]
        b["det_label"] = b["det_label"][:5]
        with self.assertRaises(T.LeakageError):
            paired.check_paired_batch(b)

    def test_stage_dispatch(self):
        with self.assertRaises(ValueError):
            tv.check_batch_v1(self.good(), "s3")
        with self.assertRaises(T.LeakageError):
            tv.check_batch_v1({"role": ["validation"], "category": ["normal_reference"]}, "s1")


class RollingTests(unittest.TestCase):
    def test_matches_the_definition_and_uses_only_the_previous_minute(self):
        ts = 1000 + 60 * np.arange(6)
        s = np.array([1.0, 5.0, 2.0, 0.5, 9.0, 3.0])
        out = rolling.rolling_max_causal(s, ts, np.array(["v"] * 6))
        self.assertEqual(out.tolist(), [1.0, 5.0, 5.0, 2.0, 9.0, 9.0])

    def test_causality_future_scores_never_change_earlier_outputs(self):
        rng = np.random.default_rng(0)
        ts, per = 1000 + 60 * np.arange(40), np.array(["v"] * 40)
        s = rng.normal(size=40)
        base = rolling.rolling_max_causal(s, ts, per)
        for k in range(1, 40):
            s2 = s.copy()
            s2[k:] += rng.normal(size=40 - k) * 100
            out = rolling.rolling_max_causal(s2, ts, per)
            self.assertTrue(np.array_equal(out[:k], base[:k]), k)

    def test_gaps_and_periods_do_not_mix_and_the_first_minute_is_partial(self):
        ts = np.array([0, 60, 180, 240, 300])                                   # a 120 s gap between index 1 and 2
        s = np.array([10.0, 1.0, 2.0, 3.0, 0.0])
        out = rolling.rolling_max_causal(s, ts, np.array(["a"] * 5))
        self.assertEqual(out.tolist(), [10.0, 10.0, 2.0, 3.0, 3.0])
        out = rolling.rolling_max_causal(np.array([10.0, 1.0, 2.0]), np.array([0, 60, 120]), np.array(["calibration", "validation", "validation"]))
        self.assertEqual(out.tolist(), [10.0, 1.0, 2.0])                         # different periods never mix

    def test_input_order_does_not_matter_and_bad_input_is_rejected(self):
        ts, s, per = 1000 + 60 * np.arange(5), np.array([1.0, 4.0, 2.0, 3.0, 0.0]), np.array(["v"] * 5)
        perm = np.array([3, 0, 4, 1, 2])
        self.assertTrue(np.array_equal(rolling.rolling_max_causal(s[perm], ts[perm], per[perm]), rolling.rolling_max_causal(s, ts, per)[perm]))
        with self.assertRaises(ValueError):
            rolling.rolling_max_causal(s, np.array([0, 60, 60, 120, 180]), per)
        with self.assertRaises(ValueError):
            rolling.rolling_max_causal(np.array([1.0, np.nan]), np.array([0, 60]), np.array(["a", "a"]))


class LossAndGuardTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.app, _ = MH.synthetic_spec_and_applicable()
        n = [3, 2, 3, 2, 3]
        self.weights = W.CaseBalancedWeights(np.repeat(np.arange(5), n), np.repeat(np.arange(5), n))

    def paired_batch(self, n=4):
        b = batch_with_starts(self.app, B=2 * n, seed=7)
        b["det_label"] = torch.tensor([1] * n + [0] * n)
        b["case_index"] = torch.tensor([0, 3, 6, 9] * (n // 4) + [-1] * n)[: 2 * n] if n == 4 else torch.zeros(2 * n, dtype=torch.int64)
        b["fault_class"] = torch.tensor([0, 1, 2, 3] + [-1] * 4)
        return b

    def test_stage2_loss_is_l_fault_plus_l_det(self):
        w = W.CaseBalancedWeights(np.array([0, 0, 1, 1, 2, 2, 3, 3]), np.array([0, 0, 1, 1, 2, 2, 3, 3]), n_classes=5) if False else None
        cb = W.CaseBalancedWeights(np.array([0, 1, 2, 3, 4, 4]), np.array([0, 1, 2, 3, 4, 4]))
        model = Model1V1.build(self.spec, seed=0).eval()
        b = self.paired_batch()
        b["case_index"] = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1])
        with mock.patch.object(tv, "simulate_missing", lambda batch, *a, **k: {"inputs": batch["inputs"], "input_mask": batch["input_mask"], "artificial_missing": torch.zeros_like(batch["input_mask"])}):
            loss, parts = tv.make_s2_loss(sm.MissingBlocks(self.spec), 0, cb)(model, b, 1)
        out = model(b["inputs"], b["input_mask"], applicable_entry=b["applicable_entry"])
        ev = b["det_label"] == 1
        wt = cb.per_window(b["case_index"][ev], normalised=True, dtype=torch.float32)
        l_f = W.weighted_window_cross_entropy(out["logits"][ev], b["fault_class"][ev], wt)
        l_d = torch.nn.functional.binary_cross_entropy_with_logits(out["detection_logit"], b["det_label"].float())
        self.assertAlmostEqual(loss.item(), (l_f + l_d).item(), places=6)
        self.assertAlmostEqual(parts["l_fault"] + parts["l_det"], loss.item(), places=5)

    def test_stage_gradients_reach_the_right_heads_only(self):
        model = Model1V1.build(self.spec, seed=0)
        b = self.paired_batch()
        b["case_index"] = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1])
        cb = W.CaseBalancedWeights(np.array([0, 1, 2, 3, 4, 4]), np.array([0, 1, 2, 3, 4, 4]))
        tv.make_s2_loss(sm.MissingBlocks(self.spec), 0, cb)(model, b, 1)[0].backward()
        self.assertTrue(all(p.grad is not None for p in model.fault_head.parameters()) and all(p.grad is not None for p in model.detection_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.next_step_head.parameters()))                  # the next-step head is not trained in Stage 2
        model.zero_grad(set_to_none=True)
        s1 = batch_with_starts(self.app, B=4)
        tv.make_s1_loss(sm.MissingBlocks(self.spec), 0)(model, s1, 1).backward()
        self.assertTrue(all(p.grad is None for p in model.fault_head.parameters()) and all(p.grad is None for p in model.detection_head.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.next_step_head.parameters()))

    def test_stage1_loss_uses_untouched_targets_and_masked_inputs_only(self):
        model = Model1V1.build(self.spec, seed=0).eval()
        b = batch_with_starts(self.app, B=4, holes=0.1)
        loss = tv.make_s1_loss(sm.MissingBlocks(self.spec), 0)(model, b, 1)
        r = sm.simulate_missing(b, sm.MissingBlocks(self.spec), 0, "s1", 1)
        out = model(r["inputs"], r["input_mask"], applicable_entry=b["applicable_entry"])
        from model1_transformer import losses as L
        self.assertAlmostEqual(loss.item(), L.masked_mse(out["prediction"], b["target"], b["target_mask"]).item(), places=6)

    def test_training_loop_refuses_non_fitting_batches_in_both_stages(self):
        spec_model = Model1V1.build(self.spec, seed=0)
        opt = torch.optim.AdamW(spec_model.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in spec_model.parameters()]
        bad_s1 = {"role": ["validation"], "category": ["normal_reference"]}
        with self.assertRaises(T.LeakageError):
            tv.train_epochs_v1(spec_model, [bad_s1], lambda m, b, e: torch.tensor(0.0), opt, epochs=1, clip_norm=1.0, device=torch.device("cpu"), stage="s1")
        bad_s2 = {"role": ["fit", "fit"], "category": ["event_associated", "ignore"], "det_label": torch.tensor([1, 0]), "fault_label_ambiguous": torch.zeros(2, dtype=torch.bool),
                  "fault_class_present": torch.tensor([True, False]), "fault_class": torch.tensor([0, -1])}
        with self.assertRaises(T.LeakageError):
            tv.train_epochs_v1(spec_model, [bad_s2], lambda m, b, e: torch.tensor(0.0), opt, epochs=1, clip_norm=1.0, device=torch.device("cpu"), stage="s2")
        self.assertTrue(all(torch.equal(a, p) for a, p in zip(before, spec_model.parameters())))        # no step was taken

    def test_the_v1_training_modules_never_build_evaluation_datasets(self):
        for mod in (tv, paired, sm):
            with open(mod.__file__) as fh:
                src = fh.read()
            self.assertIsNone(re.search(r"""evaluation_(dataset|loader)|["'](validation|calibration|test|embargo)["']|ad_cases|\.pkl""", src), mod.__name__)


if __name__ == "__main__":
    unittest.main()
