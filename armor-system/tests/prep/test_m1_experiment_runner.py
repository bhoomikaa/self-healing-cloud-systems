"""Focused tests of the five-seed experiment runner (Linear 298-51): metric helpers, safety rules and a tiny real two-stage run."""
import csv
import json
import os
import re
import tempfile
import unittest
from unittest import mock

import _m1_model_helpers as MH  # noqa: F401
import numpy as np

import _m1_helpers as H
from model1_transformer import case_eval as ce, checkpoint as ck, run_experiment as rx, training as T
from model1_transformer.model import state_digest
from model1_transformer.spec import AdapterSpec
from prep import m1_dataset as md

NEEDS_REAL = H.needs(H.CACHE_PRESENT and H.ARTIFACTS_PRESENT, "needs the D1 feature cache and the M1-03 artifacts")


class HelperTests(unittest.TestCase):
    def test_case_level_nll(self):
        agg = {"case_index": np.array([1, 2]), "true_class": np.array([0, 3]), "probs": np.array([[0.5, 0.5, 0, 0, 0], [0.1, 0.1, 0.1, 0.25, 0.45]])}
        self.assertAlmostEqual(rx.case_nll(agg), -(np.log(0.5) + np.log(0.25)) / 2, places=12)
        agg["probs"][0, 0] = 0.0
        self.assertTrue(np.isfinite(rx.case_nll(agg)))                           # a zero probability is floored, not infinite

    def test_mean_and_sample_sd(self):
        r = rx.mean_sd([1.0, 2.0, 3.0])
        self.assertEqual((r["mean"], r["sd"], r["n"]), (2.0, 1.0, 3))              # ddof = 1
        self.assertEqual(rx.mean_sd([0.7])["sd"], 0.0)

    def test_the_frozen_xgboost_reference_numbers(self):
        with open(os.path.join(rx.XGB_DIR, "results", "xgb_control_v1", "validation_summary.json")) as fh:
            m = json.load(fh)["mean_sd_across_seeds"]
        self.assertEqual([(round(m[k]["mean"], 3), round(m[k]["sd"], 3)) for k in ("macro_f1", "weighted_f1", "accuracy")], [(0.814, 0.028), (0.828, 0.022), (0.832, 0.018)])

    def test_safety_rules_in_the_runner_source(self):
        with open(rx.__file__) as fh:
            src = fh.read()
        self.assertEqual(rx.OBJECTIVES_USED, ("s1_pretrain", "s2_fault_finetune", "s1_early_stopping", "validation_fault_eval"))
        from prep import m1_split as ms
        named = {k for k in ms.OBJECTIVES if re.search(r"[\"']%s[\"']" % k, src)}
        self.assertEqual(named, set(rx.OBJECTIVES_USED))                              # no other objective (calibration, test, ...) is even named
        self.assertIsNone(re.search(r"""["'](test|calibration|embargo)["']""", src))
        self.assertIsNone(re.search(r"torch\.optim|optimizer|\.step\(|\.backward\(", src))                 # the runner never steps an optimizer itself
        self.assertFalse(os.path.abspath(rx.DEFAULT_CKPT).startswith(rx.REPO))        # checkpoints live outside the repository
        with self.assertRaises(SystemExit):
            rx.main(["--seed", "0", "--ckpt-dir", os.path.join(rx.REPO, "armor-system", "ck")])
        with self.assertRaises(ValueError):
            T.validate_seed(5)

    def test_frozen_decisions_are_in_the_config(self):
        with open(T.CONFIG_PATH) as fh:
            c = json.load(fh)
        self.assertEqual(c["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual((c["s1"]["lr"], c["s1"]["schedule"], c["s1"]["epochs"], c["s1"]["batch_size"], c["s1"]["clip_norm"], c["s1"]["weight_decay"]), (1e-3, "cosine", 100, 64, 1.0, 0.01))
        self.assertEqual((c["s2"]["lr"], c["s2"]["schedule"], c["s2"]["epochs"], c["s2"]["batch_size"], c["s2"]["clip_norm"], c["s2"]["weight_decay"]), (1e-4, "constant", 100, 64, 1.0, 0.01))
        self.assertEqual((c["s1"]["early_stopping"]["patience"], c["s2"]["early_stopping"]["patience"]), (10, 10))
        self.assertIn("masked MSE", c["s1"]["early_stopping"]["metric"])
        self.assertIn("case-level negative log-likelihood", c["s2"]["early_stopping"]["metric"])
        self.assertIn("never a gradient update", c["validation_use"])
        self.assertIn("macro-F1", c["primary_reported_metric"])


@NEEDS_REAL
class TinyRealRunTests(unittest.TestCase):
    """2 epochs x 2 batches per stage on real data: wiring, selection, files and the leakage rules. Not a training claim."""

    @classmethod
    def setUpClass(cls):
        cls.D = md.Model1Data.load()
        cls.cfg = T.load_train_config()

    def test_tiny_two_stage_run(self):
        seen = []
        real = T.check_batch

        def spy(b, stage):
            seen.append((stage, tuple(sorted(set(b["role"]))), tuple(sorted(set(b["category"])))))
            return real(b, stage)

        with tempfile.TemporaryDirectory() as ckd, tempfile.TemporaryDirectory() as out, mock.patch.object(T, "check_batch", spy):
            r = rx.run_seed(self.D, self.cfg, 0, ckd, out, epochs=2, max_batches=2, log=lambda *_: None)
            run, s = r["run"], r["metrics"]
            # gradient steps only on fitting windows of the right category; validation never reached check_batch
            self.assertEqual({x for x in seen}, {("s1", ("fit",), ("normal_reference",)), ("s2", ("fit",), ("event_associated",))})
            self.assertIn(run["selected"]["s1_epoch"], (1, 2))
            self.assertIn(run["selected"]["s2_epoch"], (1, 2))
            for name in ("histories.json", "metrics.json", "run_meta.json", "predictions.csv"):
                self.assertTrue(os.path.isfile(os.path.join(out, name)))
            self.assertTrue(all("val_masked_mse" in e for e in r["histories"]["s1"]))
            self.assertTrue(all("val_case_nll" in e and "val_macro_f1" in e for e in r["histories"]["s2"]))
            with open(os.path.join(out, "predictions.csv")) as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 25)
            self.assertEqual([int(x["case_index"]) for x in rows], sorted(int(x["case_index"]) for x in rows))
            self.assertEqual(int(np.sum(s["confusion_matrix"])), 25)
            self.assertEqual([sum(x) for x in s["confusion_matrix"]], [10, 6, 3, 5, 1])
            self.assertEqual(run["alignment"]["cases"], 25)
            self.assertEqual(run["alignment"]["minute_union_sha256"], "004d53ea6f834f51ae571edb75c97ff6fd78a05b2cc98cdf7e7d87a7c9c6944d")
            # the saved selected checkpoint reloads to exactly the evaluated weights
            self.assertEqual(run["checkpoints"]["s2_state_sha256"], run["checkpoints"]["s2_state_sha256_reloaded"])
            self.assertFalse(os.path.abspath(ckd).startswith(rx.REPO))
            self.assertEqual(run["objectives_used"], list(rx.OBJECTIVES_USED))
            m = T.build_model(AdapterSpec.from_d1(self.D.schema, self.D.applicable), 0, T.torch.device("cpu"))
            meta = ck.load_checkpoint(os.path.join(ckd, "seed_0", "s2_best"), m, stage="s2", seed=0)
            self.assertEqual(meta["extra"]["s1_selected_epoch"], run["selected"]["s1_epoch"])
            self.assertEqual(state_digest(m), run["checkpoints"]["s2_state_sha256"])


if __name__ == "__main__":
    unittest.main()
