"""Leakage and layout tests for the v2 day-blocked CV folds (V2_ENSEMBLE_PROTOCOL.md section 5). Real D1 data; skipped without the cache.

No test-period window, test label or official label is read: the fold builder drops test rows, and these tests check that it does.
"""
import glob
import os
import sys
import tempfile
import unittest

ARMOR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in ("data", "models", os.path.join("evaluation", "detection")):
    if os.path.join(ARMOR, p) not in sys.path:
        sys.path.insert(0, os.path.join(ARMOR, p))

import numpy as np  # noqa: E402

from prep import m1_dataset as md, m1_split as ms, paths  # noqa: E402

HAVE_DATA = os.path.exists(paths.CACHE_FILE)


@unittest.skipUnless(HAVE_DATA, "D1 feature cache not present")
class FoldLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from model1_v2 import folds
        cls.folds = folds
        cls.D = md.Model1Data.load()
        cls.fd = {f: folds.build(cls.D, f) for f in folds.FOLDS}

    def starts(self, data, objective):
        return set(data.table.start[md.objective_rows(data, objective)].tolist())

    def test_case_sets_are_the_protocol_lists_and_disjoint(self):
        c = self.folds.fold_cases(self.D)
        self.assertEqual(c, self.folds.EXPECTED_CASES)
        allc = c["F1"] + c["F2"] + c["F3"]
        self.assertEqual(len(allc), len(set(allc)))

    def test_no_test_period_or_embargo_window_enters_any_fold(self):
        b2 = self.D.boundaries[1]
        for f, fd in self.fd.items():
            self.assertFalse((fd.table.role == "test").any(), f)
            self.assertFalse((fd.table.T[:, -1] >= b2).any(), f)
            self.assertFalse((fd.table.role == "embargo").any(), f)

    def test_held_out_windows_never_reach_a_training_objective(self):
        for f, fd in self.fd.items():
            held = self.starts(fd, "validation_all")
            train = self.starts(fd, "s1_pretrain") | self.starts(fd, "s2_fault_finetune")
            self.assertFalse(held & train, f)
            blk = fd.fold["block_of_row"]
            self.assertTrue((blk[md.objective_rows(fd, "validation_all")] == f).all())
            for obj in ("s1_pretrain", "s2_fault_finetune"):
                b = set(blk[md.objective_rows(fd, obj)].tolist())
                self.assertEqual(b, set(fd.fold["training"]), (f, obj))

    def test_calibration_windows_are_the_approved_329_and_never_trained(self):
        approved = self.starts(self.D, "s3_calibration_reference")
        self.assertEqual(len(approved), 329)
        for f, fd in self.fd.items():
            self.assertEqual(self.starts(fd, "s3_calibration_reference"), approved, f)
            self.assertFalse(approved & (self.starts(fd, "s1_pretrain") | self.starts(fd, "s2_fault_finetune")), f)

    def test_validation_fold_trains_on_exactly_v1s_training_windows(self):
        fd = self.fd["F3"]
        for obj in ("s1_pretrain", "s2_fault_finetune"):
            self.assertEqual(self.starts(fd, obj), self.starts(self.D, obj), obj)
        self.assertEqual(self.starts(fd, "validation_all"), self.starts(self.D, "validation_all"))

    def test_region_override_is_scoped_and_needed(self):
        original = md.m1_split_region_ok
        with self.assertRaises(ms.ManifestInvariantError):              # validation windows relabelled 'fit' fail the frozen check
            md.training_dataset(self.fd["F1"], "s1_pretrain")
        with self.folds.fold_regions():
            ds = md.training_dataset(self.fd["F1"], "s1_pretrain")
            self.assertGreater(len(ds), 0)
            self.assertTrue(md.m1_split_region_ok(self.D, md.objective_rows(self.D, "s1_pretrain")))   # approved data: original check
        self.assertIs(md.m1_split_region_ok, original)

    def test_fold_region_check_rejects_a_window_outside_its_block(self):
        fd = self.fd["F2"]
        rows = md.objective_rows(fd, "validation_all")
        self.assertTrue(self.folds.region_ok(fd, rows))
        bad = fd.table.role.copy()
        fd2 = self.folds.build(self.D, "F2")
        fd2.table.role[rows[0]] = "fit"                                   # a held-out (day 2) window cannot be a training window here
        self.assertFalse(self.folds.region_ok(fd2, rows[:1]))
        self.assertTrue((fd.table.role == bad).all())

    def test_paired_training_batches_hold_only_training_fold_windows(self):
        from model1_transformer.paired import PairedLoader
        fd = self.fd["F1"]
        train = self.starts(fd, "s1_pretrain") | self.starts(fd, "s2_fault_finetune")
        with self.folds.fold_regions():
            batch = next(iter(PairedLoader(fd, batch_size=16, seed=0)))
        self.assertEqual(set(batch["role"]), {"fit"})
        self.assertTrue(set(batch["start_index"].tolist()) <= train)

    def test_weights_use_only_training_fold_cases(self):
        for f, fd in self.fd.items():
            w = self.folds.s2_weights(fd)
            self.assertFalse(set(w.case_ids.tolist()) & set(fd.fold["cases"][f]), f)


class SourceTests(unittest.TestCase):
    def test_v2_code_never_unlocks_the_test_role(self):
        for path in glob.glob(os.path.join(ARMOR, "models", "model1_v2", "*.py")):
            with open(path) as fh:
                src = fh.read()
            for word in ("unlocked(", "claim_test_run", "resume_token", "test_structural"):
                self.assertNotIn(word, src, f"{os.path.basename(path)} uses {word}")


@unittest.skipUnless(HAVE_DATA, "D1 feature cache not present")
class SmokeTests(unittest.TestCase):
    def test_one_batch_cv_run_end_to_end(self):
        from model1_v2 import train_cv
        with tempfile.TemporaryDirectory() as tmp:
            res = train_cv.run("F1", 0, os.path.join(tmp, "ck"), os.path.join(tmp, "out"), s1_epochs=1, s2_epochs=1, max_batches=1, log=lambda *a: None)
            self.assertEqual(res["windows"]["held_out"], 1434)
            self.assertEqual(res["windows"]["calibration"], 329)
            self.assertEqual(res["n_labels"], 45)
            z = np.load(os.path.join(tmp, "ck", "F1", "seed_0", "outputs.npz"), allow_pickle=True)
            self.assertTrue(set(z["held_role"].tolist()) == {"validation"})


if __name__ == "__main__":
    unittest.main()
