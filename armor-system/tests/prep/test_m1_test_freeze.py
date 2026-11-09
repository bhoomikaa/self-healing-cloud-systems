"""Tests of the one-shot test freeze: freeze record and verification, one-shot markers, the gated unlock, the scoring-set builder,
the official-label wrapper, the ensemble and the runner's validation dry run. No test-period window, test label or official label is read:
freeze and marker tests use a temporary git repository and temporary marker paths, and the official wrapper is tested on synthetic files.
"""
import io
import json
import os
import pickle
import shutil
import subprocess
import tempfile
import unittest
import zlib

import _m1_model_helpers as MH  # noqa: F401
import numpy as np

import _m1_helpers as H
from model1_transformer import detection_v1 as dv, ensemble as en, official_scoring as osc, rolling, run_detection_v0 as rd, run_test as rt, scoring_sets as ss, test_freeze as tf, window_errors as we
from prep import m1_case_features as cf

NEEDS_REAL = H.needs(H.CACHE_PRESENT and H.ARTIFACTS_PRESENT, "needs the D1 cache and the M1-03 artifacts")
ENS_VAL = os.path.join(rt.HERE, "results", "ensemble_validation_v1")


def git(repo, *a):
    subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


class FreezeRecordTests(unittest.TestCase):
    """A temporary repository with the frozen layout; checkpoints in a temporary root."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo, self.ck = os.path.join(self.tmp, "repo"), os.path.join(self.tmp, "ck")
        os.makedirs(self.repo)
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.email", "t@example.com")
        git(self.repo, "config", "user.name", "t")
        for rel in (tf.PLAN_REL, tf.PROTOCOL_REL, "armor-system/data/prep/a.py", "armor-system/models/model1_transformer/b.py", "elsewhere/c.txt"):
            write(os.path.join(self.repo, rel), rel + "\n")
        for v, s, st in tf.CHECKPOINTS:
            for ext in (".pt", ".json"):
                write(os.path.join(self.ck, v, f"seed_{s}", st + ext), f"{v}{s}{st}{ext}")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "base")
        self.ff = os.path.join(self.repo, tf.FREEZE_REL)
        self._write_freeze()
        self.markers = (os.path.join(self.tmp, "m_repo.json"), os.path.join(self.tmp, "m_ext.json"))

    def tearDown(self):
        for m in getattr(self, "markers", ()):
            if os.path.exists(m):
                os.chmod(m, 0o644)
        shutil.rmtree(self.tmp)

    def _write_freeze(self):
        doc = tf.build(self.repo, self.ck)
        write(self.ff, json.dumps(doc, indent=1, sort_keys=True) + "\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "freeze")
        return doc

    def verify(self):
        return tf.verify(self.repo, self.ck, self.ff)

    def test_fresh_freeze_verifies_and_holds_no_self_hash(self):
        self.assertEqual(self.verify(), [])
        with open(self.ff) as fh:
            doc = json.load(fh)
        paths = [r["path"] for r in doc["tracked_files"]]
        self.assertNotIn(tf.FREEZE_REL, paths)
        self.assertNotIn("elsewhere/c.txt", paths)
        self.assertIn("armor-system/data/prep/a.py", paths)
        self.assertEqual(len(doc["checkpoints"]["files"]), 40)
        self.assertFalse(doc["test_data_accessed"])
        self.assertFalse(doc["external_inputs"]["armor_official_labels"]["read_by_freeze"])

    def test_every_kind_of_change_is_detected(self):
        write(os.path.join(self.repo, "armor-system/data/prep/a.py"), "changed\n")
        git(self.repo, "commit", "-qam", "edit")
        self.assertTrue(any("changed since freeze: armor-system/data/prep/a.py" in p for p in self.verify()))
        git(self.repo, "revert", "--no-edit", "HEAD")
        self.assertEqual(self.verify(), [])
        write(os.path.join(self.repo, "armor-system/data/prep/new.py"), "x\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "add")
        self.assertTrue(any("tracked file set changed" in p for p in self.verify()))

    def test_checkpoint_plan_dirty_tree_and_uncommitted_record_are_detected(self):
        p = os.path.join(self.ck, "model1_v1", "seed_3", "s2_best.pt")
        write(p, "tampered")
        self.assertTrue(any("checkpoint differs" in x for x in self.verify()))
        write(p, "model1_v13s2_best.pt")
        self.assertEqual(self.verify(), [])
        write(os.path.join(self.repo, "untracked.txt"), "x")
        self.assertTrue(any("working tree not clean" in x for x in self.verify()))
        os.remove(os.path.join(self.repo, "untracked.txt"))
        write(os.path.join(self.repo, tf.TEST_OUT_REL, "anything.json"), "{}")             # test outputs do not dirty the frozen state
        self.assertEqual(self.verify(), [])
        with open(self.ff) as fh:
            doc = json.load(fh)
        doc["scope"] = "edited"
        write(self.ff, json.dumps(doc))
        self.assertTrue(any("differs from the committed one" in x for x in self.verify()))

    def test_the_one_shot_marker_refuses_a_second_run(self):
        tok = tf.claim_test_run(self.repo, self.ck, self.ff, self.markers)
        self.assertTrue(all(os.path.isfile(m) for m in self.markers))
        with self.assertRaisesRegex(tf.FreezeError, "already started"):
            tf.claim_test_run(self.repo, self.ck, self.ff, self.markers)
        self.assertEqual(tf.resume_token(tok.run_id, self.repo, self.ck, self.ff, self.markers).run_id, tok.run_id)
        with self.assertRaisesRegex(tf.FreezeError, "another run"):
            tf.resume_token("someone-else", self.repo, self.ck, self.ff, self.markers)
        os.chmod(self.markers[0], 0o644)
        os.remove(self.markers[0])                                                       # one marker left is still a refusal
        with self.assertRaisesRegex(tf.FreezeError, "already started"):
            tf.claim_test_run(self.repo, self.ck, self.ff, self.markers)

    def test_a_failing_freeze_writes_no_marker(self):
        write(os.path.join(self.repo, tf.PLAN_REL), "edited plan\n")
        git(self.repo, "commit", "-qam", "edit plan")
        with self.assertRaisesRegex(tf.FreezeError, "freeze verification failed"):
            tf.claim_test_run(self.repo, self.ck, self.ff, self.markers)
        self.assertFalse(any(os.path.exists(m) for m in self.markers))

    def test_no_freeze_record_is_a_refusal(self):
        self.assertEqual(tf.verify(self.repo, self.ck, os.path.join(self.tmp, "nope.json")), ["no freeze record: " + os.path.relpath(os.path.join(self.tmp, "nope.json"), self.repo)])


class UnlockTests(unittest.TestCase):
    def test_unlock_needs_a_token_and_restores_the_locks(self):
        before = (we.SCORABLE_ROLES, cf.ALLOWED_ROLES)
        self.assertNotIn("test", before[0])
        self.assertNotIn("test", before[1])
        with self.assertRaises(tf.FreezeError):
            with tf.unlocked(None):
                pass
        tok = tf.TestRunToken("unit-test", "x")                                          # constructed directly: no test data is read in this test
        with self.assertRaises(RuntimeError):
            with tf.unlocked(tok):
                self.assertIn("test", we.SCORABLE_ROLES)
                self.assertIn("test", cf.ALLOWED_ROLES)
                raise RuntimeError("boom")
        self.assertEqual((we.SCORABLE_ROLES, cf.ALLOWED_ROLES), before)

    def test_outside_the_block_test_windows_cannot_be_scored(self):
        with self.assertRaises(we.ScoringRoleError):
            dv.collect_outputs(None, [], ("test",))

    def test_the_runner_refuses_other_output_directories_for_test_and_the_test_directory_for_validation(self):
        with self.assertRaises(SystemExit):
            rt.main(["--mode", "test", "--out", tempfile.gettempdir()])
        with self.assertRaises(SystemExit):
            rt.main(["--mode", "validation", "--out", tf.TEST_OUT])


class OfficialWrapperTests(unittest.TestCase):
    """Synthetic label, manifest and split files only (the real label file is test-period data)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lab = os.path.join(self.tmp, "ad_cases.pkl")
        with open(self.lab, "wb") as fh:
            pickle.dump([100, 200, 300, 400, 500], fh, protocol=4)
        b = open(self.lab, "rb").read()
        self.man = os.path.join(self.tmp, "checksums.json")
        write(self.man, json.dumps({"nested_d1_extraction": {"files": [{"path": tf.ARMOR_LABELS_MANIFEST_PATH, "size": len(b), "crc32": "%08x" % zlib.crc32(b)}]}}))
        self.split = os.path.join(self.tmp, "split.json")
        write(self.split, json.dumps({"ad_split": {"timestamp": 250}, "content_sha256": "c"}))
        self.tok = tf.TestRunToken("unit-test", "x")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_labels_need_a_token_and_keep_only_post_split_labels(self):
        with self.assertRaises(tf.FreezeError):
            osc.load_labels(None, self.lab, self.split, self.man, expected=3)
        r = osc.load_labels(self.tok, self.lab, self.split, self.man, expected=3)
        self.assertEqual(r["labels"], [300, 400, 500])
        with self.assertRaisesRegex(osc.OfficialInputError, "published"):
            osc.load_labels(self.tok, self.lab, self.split, self.man, expected=98)

    def test_a_changed_label_file_or_interval_file_is_refused(self):
        with open(self.lab, "ab") as fh:
            fh.write(b"x")
        with self.assertRaisesRegex(osc.OfficialInputError, "CRC32"):
            osc.load_labels(self.tok, self.lab, self.split, self.man, expected=None)
        iv = os.path.join(self.tmp, "iv.json")
        write(iv, "[[1, 2]]")
        with self.assertRaises(tf.FreezeError):
            osc.released_intervals(None, iv)
        with self.assertRaisesRegex(osc.OfficialInputError, "pinned"):
            osc.released_intervals(self.tok, iv)
        self.assertEqual(osc.released_intervals(self.tok, iv, tf.sha(iv)), [(1, 2)])

    def test_a_pickle_with_a_global_is_refused(self):
        evil = os.path.join(self.tmp, "evil.pkl")
        with open(evil, "wb") as fh:
            pickle.dump([io.BytesIO], fh)
        b = open(evil, "rb").read()
        write(self.man, json.dumps({"nested_d1_extraction": {"files": [{"path": tf.ARMOR_LABELS_MANIFEST_PATH, "size": len(b), "crc32": "%08x" % zlib.crc32(b)}]}}))
        with self.assertRaises(pickle.UnpicklingError):
            osc.load_labels(self.tok, evil, self.split, self.man, expected=None)

    def test_scoring_is_the_validation_construction_with_another_label_population(self):
        ts = np.arange(0, 60 * 60, 60, dtype=np.int64)
        flags = np.zeros(len(ts), bool)
        flags[10:14] = True
        cat = np.array(["normal_reference"] * len(ts))
        labels = [int(ts[11])]
        self.assertEqual(json.dumps(rd._py(osc.score(flags, ts, cat, labels)), sort_keys=True), json.dumps(rd._py(rd.evaluate_alerts(flags, ts, cat, labels)), sort_keys=True))


class EnsembleTests(unittest.TestCase):
    def _agg(self, probs):
        return {"case_index": np.array([3, 7]), "true_class": np.array([0, 2]), "probs": np.array(probs, float), "pred_class": np.array(probs).argmax(1), "n_windows": np.array([5, 6])}

    def test_fault_ensemble_averages_probabilities_and_ties_go_to_the_lowest_class(self):
        a = self._agg([[0.6, 0.4, 0, 0, 0], [0, 0, 1, 0, 0]])
        b = self._agg([[0.4, 0.6, 0, 0, 0], [0, 1, 0, 0, 0]])
        e = en.fault([a, b])
        np.testing.assert_allclose(e["probs"], [[0.5, 0.5, 0, 0, 0], [0, 0.5, 0.5, 0, 0]])
        self.assertEqual(e["pred_class"].tolist(), [0, 1])
        c = self._agg([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
        c["true_class"] = np.array([0, 3])
        with self.assertRaises(en.EnsembleError):
            en.fault([a, c])

    def _out(self, logits, ts):
        n = len(ts)
        return {"det_logit": np.array(logits, float), "target_ts": np.array(ts), "role": np.array(["validation"] * n), "category": np.array(["normal_reference"] * n), "start_index": np.arange(n)}

    def test_detection_ensemble_is_the_mean_sigmoid_then_rolling_max_then_the_90th_percentile(self):
        rng = np.random.default_rng(0)
        ts = np.arange(0, 60 * 50, 60)
        outs = [self._out(rng.normal(size=50), ts) for _ in range(5)]
        e = en.detection_scores(outs)
        np.testing.assert_allclose(e["det_prob"], np.mean([1 / (1 + np.exp(-o["det_logit"])) for o in outs], axis=0), rtol=1e-12)
        tau, info = en.threshold(e)
        rolled = rolling.rolling_max_causal(e["det_prob"], e["target_ts"], e["role"])
        self.assertEqual(tau, float(np.percentile(rolled, 90.0, method="linear")))
        self.assertEqual(info["rolled_fraction_above"], float((rolled > tau).mean()))
        bad = self._out(rng.normal(size=50), ts + 60)
        with self.assertRaises(en.EnsembleError):
            en.detection_scores(outs[:1] + [bad])
        with self.assertRaises(en.EnsembleError):
            en.detection_scores([self._out([np.nan] * 50, ts)])


@NEEDS_REAL
class ScoringSetRealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from prep import m1_dataset as md
        cls.D = md.Model1Data.load()

    def test_validation_set_is_exactly_the_committed_validation_set(self):
        from prep import m1_dataset as md
        s = ss.build(self.D, "validation")
        self.assertEqual(s["counts"], {"windows": 869, "event_associated": 150, "normal_reference": 411, "ignore": 308})
        self.assertEqual(s["labels"], rd.validation_labels(self.D))
        self.assertEqual(sorted(s["detection_rows"].tolist()), sorted(md.objective_rows(self.D, "validation_all").tolist()))
        self.assertEqual(s["event_rows"].tolist(), sorted(md.objective_rows(self.D, "validation_fault_eval").tolist()))
        self.assertTrue(s["contiguous_target_minutes"])
        self.assertTrue((np.diff(s["target_ts"]) > 0).all())

    def test_the_test_period_and_test_features_are_locked_without_a_token(self):
        with self.assertRaises(tf.FreezeError):
            ss.build(self.D, "test")
        with self.assertRaises(cf.RoleLocked):
            cf.build_case_matrix(self.D, "test")


class ValidationEnsembleResultTests(unittest.TestCase):
    """The committed secondary validation row and the runner's dry-run equivalence checks."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ENS_VAL, "ensemble_validation.json")) as fh:
            cls.r = json.load(fh)

    def test_dry_run_reproduced_every_committed_validation_result(self):
        c = self.r["dry_run_checks"]
        self.assertTrue(c["all_identical"])
        self.assertTrue(c["missing_seed_0_mask_0"]["all_identical"])
        self.assertEqual(c["missing_seed_0_mask_0"]["differing"], [])
        for s in rt.SEEDS:
            self.assertTrue(all(c[f"seed_{s}"].values()), s)

    def test_ensemble_row_is_consistent_with_its_predictions(self):
        bc = rt.read_preds(os.path.join(ENS_VAL, "predictions_v1_ensemble.csv"))
        self.assertEqual(len(bc[0]), 25)
        f = self.r["ensemble"]["fault"]
        self.assertAlmostEqual(f["accuracy"], float((bc[1] == bc[2]).mean()), places=12)
        d = self.r["ensemble"]["detection"]["ours"]
        self.assertEqual(d["normal_reference"]["windows"], 411)
        self.assertEqual(self.r["ensemble"]["calibration"]["n_windows"], 329)
        self.assertAlmostEqual(self.r["ensemble"]["calibration"]["rolled_fraction_above"], 0.1, delta=0.01)
        self.assertEqual(set(self.r["ensemble"]["detection"]), {"ours"})                 # no official labels on validation


if __name__ == "__main__":
    unittest.main()
