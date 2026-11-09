"""v1 runner on real D1 data: paired loader, tiny end-to-end two-stage run, determinism, leakage spies, reduced missing-data run, aggregation.

The tiny run (2 epochs x 2 batches per stage) proves the wiring only; it is not a training claim. Real fitting windows are used for gradient
steps, validation/calibration windows only for no-gradient scoring, and nothing from the test period.
"""
import json
import os
import re
import tempfile
import unittest
from unittest import mock

import _m1_model_helpers as MH  # noqa: F401
import numpy as np
import torch

import _m1_helpers as H
from model1_transformer import checkpoint as ck, missing_eval as me, paired, run_detection_v0 as rd, run_v1 as r1, training as T, training_v1 as tv
from model1_transformer.model import state_digest
from model1_transformer.spec import AdapterSpec
from prep import m1_dataset as md

V0_CKPT = os.path.join(rd.DEFAULT_CKPT, "seed_0", "s1_best.pt")
NEEDS_REAL = H.needs(H.CACHE_PRESENT and H.ARTIFACTS_PRESENT and os.path.isfile(V0_CKPT), "needs the D1 cache, the M1-03 artifacts and the v0 Stage 1 checkpoints")


@NEEDS_REAL
class PairedLoaderRealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = md.Model1Data.load()

    def test_one_epoch_covers_every_eligible_event_window_once_with_matching_normals(self):
        loader = paired.PairedLoader(self.D, batch_size=64, seed=0)
        self.assertEqual(len(loader), 9)
        starts_ev, starts_nm, sizes = [], [], []
        for b in loader:
            lab = b["det_label"].numpy()
            sizes.append(len(lab))
            self.assertEqual(set(b["role"]), {"fit"})
            self.assertEqual(set(np.array(b["category"])[lab == 1]), {"event_associated"})
            self.assertEqual(set(np.array(b["category"])[lab == 0]), {"normal_reference"})
            self.assertFalse(b["fault_label_ambiguous"].any())
            self.assertTrue(b["fault_class_present"][lab == 1].all() and (b["fault_class"][lab == 1] >= 0).all())
            self.assertTrue((b["fault_class"][lab == 0] == -1).all())
            starts_ev += b["start_index"][lab == 1].tolist()
            starts_nm += b["start_index"][lab == 0].tolist()
        self.assertEqual((len(starts_ev), len(set(starts_ev))), (523, 523))
        self.assertEqual(sizes, [128] * 8 + [22])
        self.assertNotIn(2550, starts_ev)                                                # the ambiguous conflicting-label window
        t = self.D.table
        all_fit = set(self.D.selection(["fit"], ["event_associated", "normal_reference"]).tolist())
        row_by_start = {int(t.start[r]): r for r in all_fit}
        self.assertTrue(set(starts_ev + starts_nm) <= set(row_by_start))               # every window is a fitting event/normal window
        self.assertEqual({str(t.role[row_by_start[s]]) for s in starts_ev + starts_nm}, {"fit"})
        self.assertEqual(len(starts_nm), 523)

    def test_deterministic_and_seed_dependent_and_normals_cycle(self):
        a = [b["start_index"].tolist() for b in paired.PairedLoader(self.D, 64, 3)]
        b = [b["start_index"].tolist() for b in paired.PairedLoader(self.D, 64, 3)]
        c = [b["start_index"].tolist() for b in paired.PairedLoader(self.D, 64, 4)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        loader = paired.PairedLoader(self.D, 64, 0)
        e1 = [b["start_index"].tolist() for b in loader]
        e2 = [b["start_index"].tolist() for b in loader]
        self.assertNotEqual(e1, e2)                                                      # a new shuffle every epoch
        normals = [s for ep in (e1, e2) for bl in ep for s in bl[len(bl) // 2:]]
        self.assertGreater(len(normals), 1000)                                           # more than half of the 1,312 normal windows drawn over two epochs

    def test_only_fitting_objectives_are_ever_used_by_the_loader(self):
        loader = paired.PairedLoader(self.D, 64, 0)
        self.assertEqual((loader.event_ds.name, loader.normal_ds.name), ("s2_fault_finetune", "s1_pretrain"))
        self.assertTrue(loader.event_ds.training and loader.normal_ds.training)


@NEEDS_REAL
class TinyRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = md.Model1Data.load()
        cls.cfg = T.load_train_config()
        cls.spec = AdapterSpec.from_d1(cls.D.schema, cls.D.applicable)
        cls.labels = rd.validation_labels(cls.D)
        cls.td = tempfile.TemporaryDirectory()
        cls.seen = []
        real = tv.check_batch_v1

        def spy(b, stage):
            cls.seen.append((stage, tuple(sorted(set(b["role"]))), tuple(sorted(set(b["category"])))))
            return real(b, stage)
        cls.ck_a, cls.out_a = os.path.join(cls.td.name, "ck_a"), os.path.join(cls.td.name, "res_a", "seed_0")
        with mock.patch.object(tv, "check_batch_v1", spy):
            cls.r = r1.run_seed(cls.D, cls.cfg, cls.spec, 0, cls.labels, cls.ck_a, cls.out_a, epochs=2, max_batches=2, log=lambda *_: None, n_boot=20)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_training_batches_are_fitting_windows_of_the_right_kind_only(self):
        s1 = {x for x in self.seen if x[0] == "s1"}
        s2 = {x for x in self.seen if x[0] == "s2"}
        self.assertEqual(s1, {("s1", ("fit",), ("normal_reference",))})
        self.assertEqual(s2, {("s2", ("fit",), ("event_associated", "normal_reference"))})

    def test_outputs_selection_and_provenance(self):
        run = self.r["run"]
        self.assertIn(run["selected"]["s1_epoch"], (1, 2))
        self.assertIn(run["selected"]["s2_epoch"], (1, 2))
        self.assertEqual(run["checkpoints"]["parameters"], 176521)
        self.assertEqual(run["checkpoints"]["s2_state_sha256"], run["checkpoints"]["s2_state_sha256_reloaded"])
        self.assertEqual(run["provenance"]["model"], "Model1V1")
        self.assertEqual(run["provenance"]["simulated_missing"]["rate"], 0.2)
        self.assertEqual(run["objectives_used"], list(r1.OBJECTIVES_USED))
        self.assertFalse(os.path.abspath(self.ck_a).startswith(r1.REPO))
        for n in ("histories.json", "metrics.json", "detection.json", "run_meta.json", "predictions.csv"):
            self.assertTrue(os.path.isfile(os.path.join(self.out_a, n)), n)
        h = self.r["histories"]
        self.assertTrue(all("val_masked_mse" in e for e in h["s1"]))
        self.assertTrue(all({"val_case_nll", "val_balanced_bce", "val_monitor", "train_l_fault", "train_l_det"} <= set(e) for e in h["s2"]))
        self.assertEqual(self.r["fault"]["n_cases"], 25)
        self.assertEqual([sum(x) for x in self.r["fault"]["confusion_matrix"]], [10, 6, 3, 5, 1])
        self.assertEqual(run["alignment"]["minute_union_sha256"], "004d53ea6f834f51ae571edb75c97ff6fd78a05b2cc98cdf7e7d87a7c9c6944d")

    def test_detection_structure_primary_and_diagnostics(self):
        d = self.r["detection"]
        self.assertEqual(set(d["validation"]), {"primary", "d2_no_rolling", "d1_p95.0", "d1_p97.0", "d1_p99.0", "d1_p99.5", "d1_spot"})
        self.assertEqual(set(d["d3_forecast_error_stage1"]["validation"]), {"m1_raw_max", "m2_max_normalised", "m3_topk_mean_normalised"})
        self.assertEqual(set(d["d3_forecast_error_stage1"]["validation"]["m2_max_normalised"]), {"90.0", "95.0", "97.0", "99.0", "99.5"})
        self.assertEqual(d["thresholds"]["primary_p90"], d["validation"]["primary"]["threshold"])
        self.assertEqual(d["calibration"]["n_windows"], 329)
        self.assertEqual(d["calibration"]["block_bootstrap_p90"]["block_length"], 7)
        self.assertEqual((d["calibration"]["ranks_used_p90"]["lower_rank_ascending"], d["calibration"]["ranks_used_p90"]["upper_rank_ascending"]), (296, 297))
        p = d["validation"]["primary"]
        self.assertEqual(p["normal_reference"]["windows"], 411)
        self.assertIn("overlapping_ignore_regions_total", p["intervals"])
        self.assertEqual(set(p["alert_minutes_by_category"]), {"event_associated", "normal_reference", "ignore"})

    def test_checkpoints_reload_exactly_and_pass_the_integrity_checks(self):
        prov = r1.provenance(self.D, self.spec, self.cfg)
        expect = {k: prov[k] for k in ("manifest_content_sha256", "fit_only_scaler_file_sha256", "spec_fingerprint", "train_config_sha256", "model")}
        m = tv.build_model_v1(self.spec, 0, torch.device("cpu"))
        meta = ck.load_checkpoint(os.path.join(self.ck_a, "seed_0", "s2_best"), m, expect=expect, stage="s2", seed=0)
        self.assertEqual(state_digest(m), self.r["run"]["checkpoints"]["s2_state_sha256"])
        self.assertEqual(meta["extra"]["s1_selected_epoch"], self.r["run"]["selected"]["s1_epoch"])
        from model1_transformer.model import Model1
        v0 = Model1.build(self.spec, seed=0)
        with self.assertRaises(ck.CheckpointError):
            ck.load_checkpoint(os.path.join(self.ck_a, "seed_0", "s2_best"), v0)         # a v1 checkpoint cannot be loaded as a v0 model

    def test_a_second_run_is_byte_identical_and_aggregation_records_it(self):
        out_b = os.path.join(self.td.name, "res_b", "seed_0")
        r1.run_seed(self.D, self.cfg, self.spec, 0, self.labels, os.path.join(self.td.name, "ck_b"), out_b, epochs=2, max_batches=2, log=lambda *_: None, n_boot=20)
        for n in ("metrics.json", "histories.json", "detection.json", "predictions.csv"):
            with open(os.path.join(self.out_a, n), "rb") as fa, open(os.path.join(out_b, n), "rb") as fb:
                self.assertEqual(fa.read(), fb.read(), n)
        root = os.path.join(self.td.name, "res_a")
        out = r1.aggregate(root=root, seeds=(0,), rerun_dir=out_b)
        self.assertTrue(out["reproducibility_rerun"]["all_identical"])
        self.assertEqual(set(out["fault_type"]["comparison"]), {"macro_f1", "weighted_f1", "accuracy"})
        self.assertIn("primary", out["detection"])
        self.assertIn("d4_v0", out["detection"])
        self.assertTrue(os.path.isfile(os.path.join(root, "validation_summary.json")))

    def test_the_reduced_missing_data_run_is_consistent_with_the_clean_run(self):
        out_dir = os.path.join(self.td.name, "res_a")
        scen = [("full", ()), ("drop_metric", ("metric",))]
        with mock.patch.object(me, "RATES", (0.0, 0.2)), mock.patch.object(me, "armor_scenarios", lambda: scen):
            res = r1.run_missing_seed(self.D, self.cfg, self.spec, 0, self.labels, ckpt_v1=self.ck_a, out_dir=out_dir, log=lambda *_: None, masks=1)
            agg = r1.missing_aggregate(root=out_dir, seeds=(0,))
        ref = res["clean_reference"]
        self.assertEqual(res["armor"]["full"]["models"], ref)                              # no masking = the clean condition
        self.assertEqual(res["random"]["0.0"]["0"], ref)                                    # rate 0 hides nothing
        self.assertTrue(res["armor"]["drop_metric"]["hides_all_vms"])
        self.assertFalse(res["armor"]["full"]["hides_all_vms"])
        self.assertEqual(set(ref), {"v1_primary", "v1_d3", "v0"})
        self.assertEqual(set(ref["v1_primary"]), {"recalibrated", "clean_thresholds"})
        self.assertEqual(ref["v1_primary"]["recalibrated"], ref["v1_primary"]["clean_thresholds"])
        self.assertEqual(set(agg["armor_scenarios"]), {"full", "drop_metric"})
        self.assertEqual(set(agg["random_rates"]), {"0.0", "0.2"})
        self.assertEqual(agg["random_rates"]["0.0"]["distinct_masks"], 1)
        self.assertTrue(agg["armor_scenarios"]["drop_metric"]["hides_all_vms"])


class StaticGuardTests(unittest.TestCase):
    def test_the_runner_names_only_the_allowed_objectives_and_no_test_or_official_data(self):
        from prep import m1_split as ms
        with open(r1.__file__) as fh:
            body = fh.read().split('"""', 2)[2]
        named = {k for k in ms.OBJECTIVES if re.search(r"[\"']%s[\"']" % k, body)}
        self.assertEqual(named, set(r1.OBJECTIVES_USED))
        self.assertIsNone(re.search(r"ad_cases|\.pkl|pre_interval\.json|torch\.optim|optimizer\.|\.backward\(", body))
        self.assertIsNone(re.search(r"""["'](test|embargo)["']""", body))
        self.assertFalse(os.path.abspath(r1.DEFAULT_CKPT).startswith(r1.REPO))
        with self.assertRaises(SystemExit):
            r1.main(["--seed", "0", "--ckpt-dir", os.path.join(r1.REPO, "armor-system", "ck")])
        with self.assertRaises(ValueError):
            T.validate_seed(5)

    def test_the_v1_primary_configuration_is_the_one_in_the_protocol(self):
        with open(r1.PROTOCOL) as fh:
            doc = fh.read()
        for s in ("empirical 90th percentile", "causal, 60 s", "paired batch", "64 fitting `normal_reference` windows", "176,521"):
            self.assertIn(s.lower(), doc.lower())
        self.assertEqual((r1.dv.P_PRIMARY, r1.dv.BLOCK, r1.dv.N_BOOT, r1.RATE), (90.0, 7, 1000, 0.2))


if __name__ == "__main__":
    unittest.main()
