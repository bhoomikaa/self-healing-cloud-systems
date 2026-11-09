"""Two-stage training infrastructure: optimizer, clipping, failure checks, checkpoints, config, leakage guards (Linear 298-51).

Synthetic tests use a tiny model; the real-data tests run a few optimizer steps on FITTING windows only (never validation,
calibration or test windows) to prove the wiring and determinism, not to train.
"""
import json
import math
import os
import re
import subprocess
import tempfile
import unittest
from unittest import mock

import _m1_model_helpers as MH
import torch
from torch import nn

import _m1_helpers as H
from model1_transformer import checkpoint as ck, training as T, weighting as W
from model1_transformer.model import Model1, state_digest
from model1_transformer.spec import ModelConfig
from prep import m1_dataset as md

NEEDS_REAL = H.needs(H.CACHE_PRESENT and H.ARTIFACTS_PRESENT, "needs the D1 feature cache and the M1-03 artifacts")
CPU = torch.device("cpu")


class Tiny(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = nn.Linear(3, 1, bias=False, dtype=torch.float64)

    def forward(self, x):
        return self.lin(x)


def batch(x, y, role="fit", category="normal_reference"):
    return {"x": x, "y": y, "role": [role] * len(x), "category": [category] * len(x)}


def make_loader(n_batches=3, seed=0, **kw):
    g = torch.Generator().manual_seed(seed)
    return [batch(torch.randn(4, 3, generator=g, dtype=torch.float64), torch.randn(4, 1, generator=g, dtype=torch.float64), **kw) for _ in range(n_batches)]


def mse(m, b):
    return ((m(b["x"]) - b["y"]) ** 2).mean()


class OptimizerTests(unittest.TestCase):
    def run_once(self, seed=0, clip=1e9, steps=None, epochs=2):
        m = Tiny(seed)
        opt = torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=0.01)
        h = T.train_epochs(m, make_loader(), mse, opt, epochs=epochs, clip_norm=clip, device=CPU, max_steps=steps)
        return m, h

    def test_deterministic_on_a_tiny_example(self):
        (a, ha), (b, hb) = self.run_once(), self.run_once()
        self.assertEqual(state_digest(a), state_digest(b))
        self.assertEqual(ha, hb)
        c, _ = self.run_once(seed=1)
        self.assertNotEqual(state_digest(a), state_digest(c))

    def test_first_step_matches_the_adamw_formula(self):
        m = Tiny(0)
        w0 = m.lin.weight.detach().clone()
        b = make_loader(1)[0]
        g = torch.autograd.grad(mse(m, b), m.lin.weight)[0]
        opt = torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=0.01)
        T.train_epochs(m, [b], mse, opt, epochs=1, clip_norm=1e9, device=CPU)
        expect = w0 * (1 - 0.01 * 0.01) - 0.01 * g / (g.abs() + 1e-8)                  # bias-corrected m = g, v = g^2 at step 1
        self.assertTrue(torch.allclose(m.lin.weight, expect, atol=1e-12))

    def test_gradient_clipping_is_applied_and_the_pre_clip_norm_is_logged(self):
        m = Tiny(0)
        b = make_loader(1)[0]
        g = torch.autograd.grad(mse(m, b), m.lin.weight)[0]
        opt = torch.optim.AdamW(m.parameters(), lr=0.01)
        h = T.train_epochs(m, [b], mse, opt, epochs=1, clip_norm=0.05, device=CPU)
        self.assertAlmostEqual(h[0]["grad_norm_max"], g.norm().item(), places=10)
        self.assertEqual(h[0]["clipped_batches"], 1)
        self.assertLessEqual(m.lin.weight.grad.norm().item(), 0.05 * (1 + 1e-6))      # grads left by the last step are clipped
        _, h2 = self.run_once(clip=1e9)
        self.assertEqual(sum(e["clipped_batches"] for e in h2), 0)

    def test_history_is_complete_and_has_no_wall_clock_values(self):
        _, h = self.run_once(epochs=3)
        self.assertEqual([e["epoch"] for e in h], [1, 2, 3])
        self.assertEqual(set(h[0]), {"stage", "epoch", "lr", "loss", "batches", "windows", "grad_norm_mean", "grad_norm_max", "clipped_batches"})
        self.assertEqual((h[0]["batches"], h[0]["windows"]), (3, 12))

    def test_max_steps_truncates(self):
        _, h = self.run_once(steps=4, epochs=5)
        self.assertEqual(sum(e["batches"] for e in h), 4)
        self.assertTrue(h[-1]["truncated"])

    def test_cosine_and_constant_schedules(self):
        sc = T.StageConfig(lr=1e-3, weight_decay=0.01, batch_size=4, clip_norm=1.0, schedule="cosine", epochs=4, patience=3)
        m = Tiny(0)
        opt = T.make_optimizer(m, sc)
        h = T.train_epochs(m, make_loader(1), mse, opt, epochs=4, clip_norm=1.0, device=CPU, scheduler=T.make_scheduler(opt, sc))
        for k, e in enumerate(h):
            self.assertAlmostEqual(e["lr"], 0.5 * 1e-3 * (1 + math.cos(math.pi * k / 4)), places=15)
        sc2 = T.StageConfig(**{**sc.__dict__, "schedule": "constant"})
        opt2 = T.make_optimizer(Tiny(0), sc2)
        self.assertIsNone(T.make_scheduler(opt2, sc2))
        self.assertEqual((opt2.param_groups[0]["lr"], opt2.param_groups[0]["weight_decay"]), (1e-3, 0.01))
        self.assertIsInstance(opt2, torch.optim.AdamW)


class FailureCheckTests(unittest.TestCase):
    def setUp(self):
        self.m = Tiny(0)
        self.before = state_digest(self.m)
        self.opt = torch.optim.AdamW(self.m.parameters(), lr=0.01)

    def run_loop(self, loader, fn=mse, stage="s1"):
        return T.train_epochs(self.m, loader, fn, self.opt, epochs=2, clip_norm=1.0, device=CPU, stage=stage)

    def test_non_finite_loss_stops_before_any_update(self):
        with self.assertRaises(T.TrainingFailure):
            self.run_loop(make_loader(), lambda m, b: mse(m, b) * float("nan"))
        self.assertEqual(state_digest(self.m), self.before)

    def test_non_finite_gradient_with_finite_loss_stops_before_the_update(self):
        sq = lambda m, b: torch.sqrt((m(b["x"]) * 0).pow(2).sum())   # noqa: E731  value 0, gradient NaN
        with self.assertRaises(T.TrainingFailure):
            self.run_loop(make_loader(), sq)
        self.assertEqual(state_digest(self.m), self.before)

    def test_empty_training_selection_is_an_error(self):
        with self.assertRaises(T.TrainingFailure):
            self.run_loop([])

    def test_non_fitting_windows_cannot_produce_a_gradient(self):
        for role in ("validation", "calibration", "test", "embargo"):
            with self.assertRaises(T.LeakageError):
                self.run_loop(make_loader(role=role))
            self.assertEqual(state_digest(self.m), self.before)
        mixed = make_loader(1)
        mixed[0]["role"][0] = "validation"
        with self.assertRaises(T.LeakageError):
            self.run_loop(mixed)

    def test_wrong_category_for_the_stage_is_refused(self):
        with self.assertRaises(T.LeakageError):
            self.run_loop(make_loader(category="event_associated"), stage="s1")
        with self.assertRaises(T.LeakageError):
            self.run_loop(make_loader(category="normal_reference"), stage="s2")
        with self.assertRaises(T.LeakageError):
            self.run_loop(make_loader(category="ignore"), stage="s2")

    def test_early_stopper(self):
        es = T.EarlyStopper(patience=3, mode="min")
        stops = [es.update(e, v) for e, v in enumerate([5.0, 4.0, 4.5, 4.2, 4.1, 3.9], 1)]
        self.assertEqual(stops, [False, False, False, False, True, False])            # three non-improving epochs after epoch 2 stop at epoch 5
        es2 = T.EarlyStopper(patience=2, mode="max")
        self.assertEqual([es2.update(e, v) for e, v in enumerate([0.1, 0.3, 0.2, 0.25], 1)], [False, False, False, True])
        self.assertEqual((es2.best, es2.best_epoch), (0.3, 2))
        with self.assertRaises(T.TrainingFailure):
            es2.update(5, float("nan"))
        with self.assertRaises(ValueError):
            T.EarlyStopper(0)

    def test_early_stop_hook_ends_training(self):
        es = T.EarlyStopper(patience=1, mode="min")
        vals = iter([1.0, 2.0, 0.5])
        h = T.train_epochs(self.m, make_loader(1), mse, self.opt, epochs=10, clip_norm=1.0, device=CPU, on_epoch_end=lambda e, ent, m: es.update(e, next(vals)))
        self.assertEqual(len(h), 2)
        self.assertTrue(h[-1]["stopped_early"])


class ConfigTests(unittest.TestCase):
    def test_design_defaults_and_seeds(self):
        c = T.load_train_config()
        self.assertEqual(c.seeds, (0, 1, 2, 3, 4))
        for s in (c.s1, c.s2):
            self.assertEqual((s.weight_decay, s.batch_size, s.clip_norm, s.patience), (0.01, 64, 1.0, 10))
        self.assertEqual((c.s1.lr, c.s1.schedule, c.s2.lr), (1e-3, "cosine", 1e-4))
        self.assertEqual(c.sha256, T.load_train_config().sha256)

    def test_seed_validation_and_bad_configurations(self):
        for s in (0, 1, 2, 3, 4):
            self.assertEqual(T.validate_seed(s), s)
        for s in (5, -1, 42):
            with self.assertRaises(ValueError):
                T.validate_seed(s)
        with self.assertRaises(ValueError):
            T.StageConfig(lr=0, weight_decay=0.01, batch_size=4, clip_norm=1, schedule="cosine", epochs=1, patience=1)
        with self.assertRaises(ValueError):
            T.StageConfig(lr=1e-3, weight_decay=0.01, batch_size=4, clip_norm=1, schedule="linear", epochs=1, patience=1)
        with open(T.CONFIG_PATH) as fh:
            raw = json.load(fh)
        raw["seeds"] = [0, 1, 2]
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "c.json")
            with open(p, "w") as fh:
                json.dump(raw, fh)
            with self.assertRaises(ValueError):
                T.load_train_config(p)

    def test_device_selection(self):
        self.assertEqual(T.select_device("cpu").type, "cpu")
        self.assertEqual(T.select_device("auto").type, "cuda" if torch.cuda.is_available() else "cpu")
        with self.assertRaises(ValueError):
            T.select_device("mps")
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                T.select_device("cuda")

    def test_no_evaluation_data_access_in_the_training_code(self):
        for mod in (T, W):
            with open(mod.__file__) as fh:
                src = fh.read()
            self.assertIsNone(re.search(r"""["'](validation|calibration|test|embargo)["']|evaluation_(dataset|loader)""", src), mod.__name__)


class EvaluateMseTests(unittest.TestCase):
    def test_exact_masked_mse_no_gradient_and_mode_restored(self):
        spec, app, _ = MH.synthetic_spec_and_applicable()
        model = MH.make_model(spec).train()
        bs = [MH.make_batch(app, B=2, seed=s, holes=0.1) for s in (1, 2)]
        got = T.evaluate_masked_mse(model, bs, CPU)
        self.assertTrue(model.training)
        sq, n = 0.0, 0
        model.eval()
        with torch.no_grad():
            for b in bs:
                p = model(b["inputs"], b["input_mask"], applicable_entry=b["applicable_entry"])["prediction"].double()
                m = b["target_mask"]
                sq += float((((p - b["target"].double()) ** 2) * m).sum())
                n += int(m.sum())
        self.assertAlmostEqual(got, sq / n, places=10)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with self.assertRaises(T.TrainingFailure):
            T.evaluate_masked_mse(model, [dict(bs[0], target_mask=torch.zeros_like(bs[0]["target_mask"]))], CPU)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.spec, _, _ = MH.synthetic_spec_and_applicable()
        self.model = MH.make_model(self.spec, seed=2)
        self.prov = {"manifest_content_sha256": "m" * 64, "fit_only_scaler_file_sha256": "s" * 64, "spec_fingerprint": self.spec.fingerprint(), "train_config_sha256": "t" * 64}
        self.td = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "ck", "s1_seed2")
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.meta = ck.save_checkpoint(self.path, self.model, stage="s1", seed=2, epoch=7, provenance=self.prov, history=[{"epoch": 1}], optimizer=opt)

    def tearDown(self):
        self.td.cleanup()

    def fresh(self, seed=9):
        return MH.make_model(self.spec, seed=seed)

    def test_round_trip_restores_the_exact_weights(self):
        m2 = self.fresh()
        self.assertNotEqual(state_digest(m2), state_digest(self.model))
        meta = ck.load_checkpoint(self.path, m2, expect=self.prov, stage="s1", seed=2)
        self.assertEqual(state_digest(m2), state_digest(self.model))
        self.assertEqual((meta["epoch"], meta["parameters"], meta["state_sha256"]), (7, 168200, state_digest(self.model)))
        self.assertEqual(meta["provenance"], self.prov)

    def test_checkpoint_holds_weights_and_metadata_only(self):
        blob = torch.load(self.path + ".pt", weights_only=True)
        self.assertEqual(set(blob), {"model_state", "meta_json", "optimizer_state"})
        self.assertEqual(set(blob["model_state"]), set(self.model.state_dict()))

    def test_flipped_byte_is_detected(self):
        with open(self.path + ".pt", "r+b") as fh:
            fh.seek(os.path.getsize(self.path + ".pt") // 2)
            b = fh.read(1)
            fh.seek(-1, 1)
            fh.write(bytes([b[0] ^ 1]))
        with self.assertRaises(ck.CheckpointError):
            ck.load_checkpoint(self.path, self.fresh())

    def test_edited_provenance_file_is_detected(self):
        with open(self.path + ".json") as fh:
            meta = json.load(fh)
        meta["seed"] = 3
        with open(self.path + ".json", "w") as fh:
            json.dump(meta, fh)
        with self.assertRaises(ck.CheckpointError):
            ck.load_checkpoint(self.path, self.fresh())

    def test_tampered_weights_with_a_matching_file_hash_are_detected_by_the_state_digest(self):
        blob = torch.load(self.path + ".pt", weights_only=True)
        k = next(iter(blob["model_state"]))
        blob["model_state"][k] = blob["model_state"][k] + 1.0
        torch.save(blob, self.path + ".pt")
        with open(self.path + ".json") as fh:
            meta = json.load(fh)
        meta["pt_file_sha256"] = ck._sha256_file(self.path + ".pt")
        with open(self.path + ".json", "w") as fh:
            json.dump(meta, fh)
        with self.assertRaises(ck.CheckpointError):
            ck.load_checkpoint(self.path, self.fresh())

    def test_expectations_are_enforced(self):
        with self.assertRaises(ck.CheckpointError):
            ck.load_checkpoint(self.path, self.fresh(), stage="s2")
        with self.assertRaises(ck.CheckpointError):
            ck.load_checkpoint(self.path, self.fresh(), seed=0)
        with self.assertRaises(ck.CheckpointError):
            ck.load_checkpoint(self.path, self.fresh(), expect={"manifest_content_sha256": "x" * 64})
        with self.assertRaises(ck.CheckpointError):
            ck.load_checkpoint(self.path + "_missing", self.fresh())

    def test_architecture_mismatch_is_detected(self):
        other = Model1.build(self.spec, ModelConfig(d_ffn=128), seed=0)
        with self.assertRaises(ck.CheckpointError):
            ck.load_checkpoint(self.path, other)

    def test_checkpoints_are_git_ignored(self):
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        r = subprocess.run(["git", "check-ignore", "-q", "armor-system/models/model1_transformer/results/x.pt"], cwd=repo)
        self.assertEqual(r.returncode, 0)


@NEEDS_REAL
class RealTrainingWiringTests(unittest.TestCase):
    """A few optimizer steps on real FITTING windows (batch 16): wiring and determinism only, no training claim."""

    @classmethod
    def setUpClass(cls):
        cls.D = md.Model1Data.load()
        c = T.load_train_config()

        def small(s):
            return T.StageConfig(**{**s.__dict__, "batch_size": 16})
        cls.cfg = T.TrainConfig(small(c.s1), small(c.s2), c.seeds, c.sha256)

    def pipeline(self, seed=0, ckdir=None):
        seen = []
        real = T.check_batch

        def spy(b, stage):
            seen.append((stage, tuple(sorted(set(b["role"]))), tuple(sorted(set(b["category"])))))
            return real(b, stage)

        with mock.patch.object(T, "check_batch", spy):
            m1, h1 = T.train_stage1(self.D, self.cfg, seed, max_steps=3, epochs=2)
            d1 = state_digest(m1)
            if ckdir:
                prov = ck.data_provenance(self.D, m1.spec, self.cfg)
                ck.save_checkpoint(os.path.join(ckdir, "s1"), m1, stage="s1", seed=seed, epoch=len(h1), provenance=prov, history=h1)
                m1b = MH.make_model(m1.spec, seed=99).train()
                ck.load_checkpoint(os.path.join(ckdir, "s1"), m1b, stage="s1", seed=seed,
                                   expect={k: prov[k] for k in ("manifest_content_sha256", "fit_only_scaler_file_sha256", "spec_fingerprint", "train_config_sha256")})
                self.assertEqual(state_digest(m1b), d1)
                m1 = m1b
            m2, h2, _ = T.train_stage2(m1, self.D, self.cfg, seed, max_steps=3, epochs=2)
        return d1, state_digest(m2), h1, h2, seen

    def test_two_stage_pipeline_uses_only_fitting_windows_of_the_right_category(self):
        with tempfile.TemporaryDirectory() as td:
            d1, d2, h1, h2, seen = self.pipeline(0, td)
        self.assertEqual({s for s in seen if s[0] == "s1"}, {("s1", ("fit",), ("normal_reference",))})
        self.assertEqual({s for s in seen if s[0] == "s2"}, {("s2", ("fit",), ("event_associated",))})
        self.assertEqual(sum(e["batches"] for e in h1), 3)
        self.assertEqual(sum(e["batches"] for e in h2), 3)
        for e in h1 + h2:
            self.assertTrue(math.isfinite(e["loss"]) and e["loss"] > 0 and math.isfinite(e["grad_norm_max"]))
        self.assertNotEqual(d1, d2)                                                          # stage 2 really changed the weights

    def test_runs_are_bit_reproducible_and_seeds_differ(self):
        a, b = self.pipeline(0), self.pipeline(0)
        self.assertEqual(a[:4], b[:4])
        self.assertNotEqual(a[0], self.pipeline(1)[0])

    def test_training_refuses_validation_calibration_and_test_datasets(self):
        for obj in ("validation_fault_eval", "validation_all", "s3_calibration_reference", "calibration_diagnostics", "s1_early_stopping"):
            with self.assertRaises(md.LeakageError):
                md.training_dataset(self.D, obj)
        with self.assertRaises(md.LeakageError):
            md.training_loader(md.evaluation_dataset(self.D, "validation_fault_eval"))

    def test_seed_outside_zero_to_four_is_rejected(self):
        with self.assertRaises(ValueError):
            T.train_stage1(self.D, self.cfg, 7, max_steps=1, epochs=1)


if __name__ == "__main__":
    unittest.main()
