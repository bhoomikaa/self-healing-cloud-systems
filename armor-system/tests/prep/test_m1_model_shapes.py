"""Shapes, schema-derived adapters, determinism, masking edge cases and losses of the Model 1 core (Linear 298-50). CPU only."""
import copy
import os
import re
import unittest

import _m1_model_helpers as MH
import numpy as np
import torch

import _m1_helpers as H
from model1_transformer import core as core_mod, losses as L
from model1_transformer.core import FactorisedBlock, MultiHeadAttention, SpatioTemporalCore
from model1_transformer.heads import FaultHead
from model1_transformer.model import Model1, state_digest
from model1_transformer.spec import AdapterSpec, ModelConfig, SpecError
from prep import m1_masks


class SpecTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.app, self.schema = MH.synthetic_spec_and_applicable()

    def test_channel_lists_come_from_the_declared_schema_rule(self):
        n = {(self.spec.type_names[k], m): len(c) for (k, m), c in self.spec.type_channels.items()}
        self.assertEqual(n, {("pod", "metric"): 53, ("pod", "trace"): 9, ("pod", "log"): 11, ("vm", "metric"): 57})
        names = [c["name"] for c in self.schema["channel_order"]]
        vm = self.spec.type_names.index("vm")
        self.assertTrue(all(names[j].startswith("system.") for j in self.spec.channels(vm, "metric")))
        self.assertEqual(self.spec.channels(vm, "trace").size + self.spec.channels(vm, "log").size, 0)
        self.assertEqual(self.spec.output_channels(self.spec.type_names.index("pod")).size, 73)
        self.assertEqual(self.spec.output_channels(vm).size, 57)

    def test_applicable_entries_lie_inside_the_lists(self):
        covered = np.zeros_like(self.app)
        for k in range(2):
            rows = np.nonzero(self.spec.node_type == k)[0]
            covered[np.ix_(rows, self.spec.output_channels(k))] = True
        self.assertFalse((self.app & ~covered).any())

    def test_no_hard_coded_node_positions(self):
        """Shuffle the node order in the schema: the node-type vector and every channel list follow the names."""
        rng = np.random.default_rng(0)
        perm = rng.permutation(46)
        schema = copy.deepcopy(self.schema)
        schema["node_order"] = [self.schema["node_order"][i] for i in perm]
        spec2 = AdapterSpec.from_d1(schema)
        self.assertEqual(spec2.node_type.tolist(), self.spec.node_type[perm].tolist())
        self.assertNotEqual(spec2.node_type.tolist(), self.spec.node_type.tolist())
        for key, cols in self.spec.type_channels.items():
            self.assertEqual(spec2.type_channels[key].tolist(), cols.tolist())

    def test_wrong_channel_counts_fail_loudly(self):
        with self.assertRaises(SpecError):
            AdapterSpec.from_d1(self.schema, expected={("pod", "metric"): 54, ("pod", "trace"): 9, ("pod", "log"): 11, ("vm", "metric"): 57})
        bad = np.ones((46, 130), bool)
        with self.assertRaises(SpecError):
            AdapterSpec.from_d1(self.schema, bad)

    def test_heads_must_divide_width(self):
        with self.assertRaises(SpecError):
            ModelConfig(d_model=64, n_heads=5)


class ShapeTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.app, _ = MH.synthetic_spec_and_applicable()
        self.model = MH.make_model(self.spec)

    def test_adapter_widths_are_twice_the_channels(self):
        got = {k: (v.in_features, v.out_features) for k, v in self.model.input_adapter.linears.items()}
        self.assertEqual(got, {"pod_metric": (106, 64), "pod_trace": (18, 64), "pod_log": (22, 64), "vm_metric": (114, 64)})
        out = {k: (v.in_features, v.out_features) for k, v in self.model.next_step_head.adapter.linears.items()}
        self.assertEqual(out, {"pod": (64, 73), "vm": (64, 57)})

    def test_approved_sizes_and_parameter_count(self):
        c = self.model.cfg
        self.assertEqual((c.d_model, c.n_heads, c.n_layers, c.d_ffn, c.dropout, c.n_classes, c.n_input_steps), (64, 4, 2, 256, 0.1, 5, 6))
        self.assertEqual(len(self.model.core.blocks), 2)
        self.assertEqual(self.model.n_parameters(), 168200)          # measured; changes only with an architecture change

    def test_forward_shapes_and_values_are_finite(self):
        b = MH.make_batch(self.app, B=3, holes=0.1)
        o = self.model(b["inputs"], b["input_mask"], return_aux=True)
        self.assertEqual(tuple(o["logits"].shape), (3, 5))
        self.assertEqual(tuple(o["prediction"].shape), (3, 46, 130))
        self.assertEqual(tuple(o["hidden"].shape), (3, 6, 46, 64))
        self.assertEqual(tuple(o["fusion_weights"].shape), (3, 6, 46, 3))
        for v in o.values():
            self.assertTrue(torch.isfinite(v.float()).all())

    def test_prediction_is_exactly_zero_outside_the_schema_channels_and_the_structural_mask(self):
        b = MH.make_batch(self.app)
        o = self.model(b["inputs"], b["input_mask"], applicable_entry=b["applicable_entry"])
        vm = self.spec.node_type == self.spec.type_names.index("vm")
        self.assertTrue((o["prediction"][:, torch.as_tensor(vm)][:, :, 53:73] == 0).all())       # VM has no trace/log channels
        self.assertTrue((o["prediction"][:, ~b["applicable_entry"]] == 0).all())
        o2 = self.model(b["inputs"], b["input_mask"])                                            # without the mask, only the scatter zeros remain
        self.assertTrue((o2["prediction"][:, torch.as_tensor(vm)][:, :, 53:73] == 0).all())
        self.assertTrue((o2["prediction"][:, ~b["applicable_entry"]] != 0).any())

    def test_heads_read_the_last_minute_only(self):
        b = MH.make_batch(self.app)
        o = self.model(b["inputs"], b["input_mask"], return_aux=True)
        nv = torch.ones(2, 46, dtype=torch.bool)
        self.assertTrue(torch.equal(o["logits"], self.model.fault_head(o["hidden"][:, -1], nv)))
        self.assertTrue(torch.equal(o["prediction"], self.model.next_step_head(o["hidden"][:, -1], torch.as_tensor(self.spec.node_type), nv)))

    def test_temporal_attention_is_bidirectional(self):
        b = MH.make_batch(self.app)
        o1 = self.model(b["inputs"], b["input_mask"], return_aux=True)["hidden"]
        x2 = b["inputs"].clone()
        x2[:, -1] += 0.7 * b["input_mask"][:, -1]                      # change only the LAST historical minute
        o2 = self.model(x2, b["input_mask"], return_aux=True)["hidden"]
        self.assertGreater((o1[:, 0] - o2[:, 0]).abs().max().item(), 1e-6)       # ... and minute 0 notices

    def test_wrong_input_shapes_are_rejected(self):
        b = MH.make_batch(self.app)
        with self.assertRaises(ValueError):
            self.model(b["inputs"], b["input_mask"].float())
        with self.assertRaises(ValueError):
            self.model(b["inputs"][:, :, :40], b["input_mask"][:, :, :40])      # node count differs and no node_type given
        with self.assertRaises(ValueError):
            self.model(b["inputs"][..., :129], b["input_mask"][..., :129])


class DeterminismTests(unittest.TestCase):
    def test_seeded_initialisation_is_bit_identical(self):
        spec, _, _ = MH.synthetic_spec_and_applicable()
        a, b, c = (Model1.build(spec, seed=s) for s in (0, 0, 1))
        self.assertEqual(state_digest(a), state_digest(b))
        self.assertNotEqual(state_digest(a), state_digest(c))

    def test_two_forward_runs_are_bit_identical(self):
        spec, app, _ = MH.synthetic_spec_and_applicable()
        b = MH.make_batch(app, holes=0.05)
        outs = []
        for _ in range(2):
            m = MH.make_model(spec, seed=3)
            outs.append(m(b["inputs"], b["input_mask"]))
        self.assertTrue(torch.equal(outs[0]["logits"], outs[1]["logits"]) and torch.equal(outs[0]["prediction"], outs[1]["prediction"]))


class CoreIndependenceTests(unittest.TestCase):
    def test_core_source_contains_no_dataset_constants(self):
        with open(core_mod.__file__) as fh:
            src = fh.read()
        code = "\n".join(line.split("#")[0] for line in src.splitlines())
        code = re.sub(r'""".*?"""', "", code, flags=re.S)
        self.assertEqual(re.findall(r"\b(46|130)\b", code), [])
        self.assertNotIn("node-", code)

    def test_core_runs_for_any_node_count_and_types(self):
        cfg = ModelConfig(n_node_types=3)
        core = SpatioTemporalCore(cfg).eval()
        for N in (1, 2, 5, 12, 77):
            g = torch.Generator().manual_seed(N)
            tok = torch.randn(2, 6, N, 3, 64, generator=g)
            av = torch.rand(2, 6, N, 3, generator=g) > 0.3
            nt = torch.arange(N) % 3
            h = core(tok, av, nt, torch.ones(2, N, dtype=torch.bool))
            self.assertEqual(tuple(h.shape), (2, 6, N, 64))
            self.assertTrue(torch.isfinite(h).all())

    def test_model_runs_with_a_different_node_count_when_node_type_is_given(self):
        spec, app, _ = MH.synthetic_spec_and_applicable()
        m = MH.make_model(spec)
        keep = torch.cat([torch.arange(0, 10), torch.arange(40, 44)])
        b = MH.make_batch(app)
        o = m(b["inputs"][:, :, keep], b["input_mask"][:, :, keep], node_type=torch.as_tensor(spec.node_type)[keep])
        self.assertEqual(tuple(o["prediction"].shape), (2, 14, 130))
        self.assertTrue(torch.isfinite(o["logits"]).all())


class MaskingEdgeCaseTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.app, _ = MH.synthetic_spec_and_applicable()
        self.model = MH.make_model(self.spec)

    def test_everything_masked_is_finite_and_uses_the_missing_vector(self):
        b = MH.make_batch(self.app)
        m = torch.zeros_like(b["input_mask"])
        o = self.model(b["inputs"] * 0 + 1e9, m, return_aux=True)
        self.assertTrue(all(torch.isfinite(v.float()).all() for v in o.values()))
        self.assertTrue((o["fusion_weights"] == 0).all())
        self.assertTrue(torch.equal(o["logits"][0], o["logits"][1]))                    # nothing left to tell the two samples apart

    def test_an_entirely_masked_node_is_finite_and_others_unchanged_in_shape(self):
        b = MH.make_batch(self.app)
        m = b["input_mask"].clone()
        m[:, :, 3] = False
        o = self.model(b["inputs"], m, return_aux=True)
        self.assertTrue(torch.isfinite(o["prediction"]).all() and torch.isfinite(o["hidden"]).all())
        self.assertTrue((o["fusion_weights"][:, :, 3] == 0).all())

    def test_sample_with_no_valid_node_gives_finite_zero_outputs(self):
        b = MH.make_batch(self.app, B=2)
        nv = torch.ones(2, 46, dtype=torch.bool)
        nv[1] = False
        o = self.model(b["inputs"], b["input_mask"], node_valid=nv)
        self.assertTrue(torch.isfinite(o["logits"]).all() and torch.isfinite(o["prediction"]).all())
        self.assertTrue((o["prediction"][1] == 0).all())

    def test_attention_rows_are_finite_and_padded_keys_get_exactly_zero_weight(self):
        torch.manual_seed(0)
        att = MultiHeadAttention(64, 4).eval()
        x = torch.randn(3, 7, 64)
        kv = torch.tensor([[1, 1, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
        x[0, 3] = float("nan")
        x[2] = float("inf")
        y, w = att(x, kv, return_weights=True)
        valid_q = kv[0]                                                                 # rows of padded queries are never read
        self.assertTrue(torch.isfinite(y[1]).all() and torch.isfinite(y[0][valid_q]).all() and torch.isfinite(w[0][:, valid_q]).all())
        self.assertTrue((w[0][:, valid_q][..., ~kv[0]] == 0).all())
        self.assertTrue(torch.allclose(w[0][:, valid_q][..., kv[0]].sum(-1), torch.ones(4, int(valid_q.sum()))))
        self.assertTrue((w[2] == 0).all())                                              # no valid key: zeros, not NaN
        self.assertTrue((y[2] == att.out.bias).all())
        _, w1 = att(torch.randn(2, 6, 64), None, return_weights=True)
        self.assertTrue(torch.allclose(w1.sum(-1), torch.ones(2, 4, 6)))

    def test_gradients_are_finite_with_extreme_values_in_masked_slots(self):
        b = MH.make_batch(self.app, holes=0.2)
        x = torch.where(b["input_mask"], b["inputs"], torch.full_like(b["inputs"], float("nan"))).requires_grad_(True)
        o = self.model(x, b["input_mask"], applicable_entry=b["applicable_entry"])
        loss = L.masked_mse(o["prediction"], torch.full_like(b["target"], float("inf")).where(~b["target_mask"], b["target"]), b["target_mask"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(x.grad).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))


class PaddedNodeGradientRegressionTests(unittest.TestCase):
    """Regression (found in the 298-50 review): NaN/inf in a padded node's slot with a True mask gave NaN adapter-weight gradients (0 * NaN)."""

    def test_garbage_in_padded_slots_never_reaches_any_gradient(self):
        spec, app, _ = MH.synthetic_spec_and_applicable()
        for garbage in (float("nan"), float("inf"), 1e30):
            model = MH.make_model(spec)
            model.train()
            b = MH.make_batch(app, B=2, holes=0.1)
            nv = torch.ones(2, 46, dtype=torch.bool)
            nv[:, 3] = False
            nv[1, 44] = False
            x, m = b["inputs"].clone(), b["input_mask"].clone()
            x[~nv[:, None, :, None].expand_as(x)] = garbage
            m[~nv[:, None, :, None].expand_as(m)] = True                       # padded slots even claim to be observed
            x.requires_grad_(True)
            torch.manual_seed(0)
            o = model(x, m, node_valid=nv, applicable_entry=b["applicable_entry"])
            loss = L.masked_mse(o["prediction"], b["target"], b["target_mask"] & nv[..., None]) + L.weighted_cross_entropy(o["logits"], b["labels"], L.inverse_frequency_weights([33, 29, 9, 13, 4]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss), garbage)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), garbage)
            self.assertTrue((x.grad[~nv[:, None, :, None].expand_as(x)] == 0).all(), garbage)
            # and the valid nodes see exactly what they would see with clean padded slots
            clean = x.detach().clone()
            clean[~nv[:, None, :, None].expand_as(clean)] = 0.0
            torch.manual_seed(0)
            o2 = model(clean, b["input_mask"], node_valid=nv, applicable_entry=b["applicable_entry"])
            self.assertTrue(torch.equal(o["logits"], o2["logits"]))

    def test_node_type_outside_the_configured_types_is_rejected(self):
        spec, app, _ = MH.synthetic_spec_and_applicable()
        model = MH.make_model(spec)
        b = MH.make_batch(app)
        with self.assertRaises(ValueError):
            model(b["inputs"], b["input_mask"], node_type=torch.full((46,), 2, dtype=torch.int64))


class LossGradientTests(unittest.TestCase):
    """Gradients and denominators checked against closed forms."""

    def test_masked_mse_gradient_and_denominator(self):
        g = torch.Generator().manual_seed(1)
        p = torch.randn(3, 4, generator=g, dtype=torch.float64, requires_grad=True)
        t = torch.randn(3, 4, generator=g, dtype=torch.float64)
        m = torch.rand(3, 4, generator=g) > 0.4
        loss = L.masked_mse(p, t, m)
        self.assertAlmostEqual(loss.item(), (((p - t) ** 2)[m].sum() / m.sum()).item(), places=12)       # denominator = effective entries of the whole batch
        loss.backward()
        expect = torch.where(m, 2 * (p - t) / m.sum(), torch.zeros_like(p))
        self.assertTrue(torch.allclose(p.grad, expect.detach(), atol=1e-12))

    def test_weighted_cross_entropy_gradient_and_weighted_mean_denominator(self):
        g = torch.Generator().manual_seed(2)
        logits = torch.randn(5, 5, generator=g, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0, 4, 4, 2, 1])
        w = L.inverse_frequency_weights([33, 29, 9, 13, 4]).double()
        loss = L.weighted_cross_entropy(logits, y, w)
        loss.backward()
        onehot = torch.nn.functional.one_hot(y, 5).double()
        expect = w[y][:, None] * (logits.detach().softmax(-1) - onehot) / w[y].sum()                 # denominator = sum of the batch's class weights
        self.assertTrue(torch.allclose(logits.grad, expect, atol=1e-12))
        # consequence of the weighted mean: a batch of only the rarest class has the same loss scale as a batch of only the commonest
        one = lambda c: L.weighted_cross_entropy(torch.zeros(3, 5, dtype=torch.float64), torch.full((3,), c), w).item()   # noqa: E731
        self.assertAlmostEqual(one(0), one(3), places=12)


class LossTests(unittest.TestCase):
    def test_masked_mse_value_and_empty_mask(self):
        p = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        t = torch.tensor([[0.0, 2.0, 9.0], [4.0, 1.0, 6.0]])
        m = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
        loss = L.masked_mse(p, t, m)
        self.assertAlmostEqual(loss.item(), (1 + 0 + 0 + 16) / 4)
        loss.backward()
        self.assertTrue((p.grad[~m] == 0).all())
        p2 = torch.randn(2, 3, requires_grad=True)
        z = L.masked_mse(p2, torch.full((2, 3), float("nan")), torch.zeros(2, 3, dtype=torch.bool))
        z.backward()
        self.assertEqual(z.item(), 0.0)
        self.assertTrue((p2.grad == 0).all())

    def test_masked_mse_rejects_bad_inputs(self):
        with self.assertRaises(L.LossInputError):
            L.masked_mse(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2, 3))              # float mask
        with self.assertRaises(L.LossInputError):
            L.masked_mse(torch.zeros(2, 3), torch.zeros(2, 4), torch.ones(2, 3, dtype=torch.bool))

    def test_class_weights_are_inverse_frequency_over_88_fit_cases(self):
        w = L.inverse_frequency_weights([33, 29, 9, 13, 4])
        np.testing.assert_allclose(w.numpy(), [88 / (5 * n) for n in (33, 29, 9, 13, 4)], rtol=1e-6)
        manifest = {"counts": {"roles": {"fit": {"cases": 88, "cases_by_class": [33, 29, 9, 13, 4]}}}}
        self.assertTrue(torch.equal(L.fit_class_weights(manifest), w))
        for bad in ({"cases": 101, "cases_by_class": [40, 29, 9, 13, 10]}, {"cases": 88, "cases_by_class": [33, 29, 9, 17]},
                    {"cases": 88, "cases_by_class": [33, 29, 9, 17, 0]}):
            with self.assertRaises(L.LossInputError):
                L.fit_class_weights({"counts": {"roles": {"fit": bad}}})
        with self.assertRaises(L.LossInputError):
            L.inverse_frequency_weights([3, 0, 1])

    def test_weighted_cross_entropy_matches_the_definition_and_rejects_bad_labels(self):
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(6, 5, generator=g, requires_grad=True)
        y = torch.tensor([0, 1, 2, 3, 4, 1])
        w = L.inverse_frequency_weights([33, 29, 9, 13, 4])
        nll = -torch.log_softmax(logits, -1)[torch.arange(6), y]
        self.assertAlmostEqual(L.weighted_cross_entropy(logits, y, w).item(), (w[y] * nll).sum().item() / w[y].sum().item(), places=5)
        for bad in (torch.tensor([0, 1, -1, 3, 4, 1]), torch.tensor([0, 1, 5, 3, 4, 1])):
            with self.assertRaises(L.LossInputError):
                L.weighted_cross_entropy(logits, bad, w)
        with self.assertRaises(L.LossInputError):
            L.weighted_cross_entropy(logits, y.int(), w)
        with self.assertRaises(L.LossInputError):
            L.weighted_cross_entropy(logits[:0], y[:0], w)


if __name__ == "__main__":
    unittest.main()
