"""Checks of the saved Model 1 v1 results (validation only): consistency, integrity, reproducibility and the leakage facts they record."""
import json
import os
import unittest

import _m1_model_helpers as MH  # noqa: F401
import numpy as np

import _m1_helpers as H
from model1_transformer import bootstrap_compare as bc, bootstrap_v1 as bv, run_detection_v0 as rd, run_experiment as rx, run_v1 as r1

R = r1.RESULTS
CK = os.path.join(r1.DEFAULT_CKPT, "seed_0", "s2_best.pt")
NEEDS_CK = H.needs(os.path.isfile(CK) and os.path.isfile(os.path.join(rd.DEFAULT_CKPT, "seed_0", "s1_best.pt")), "needs the v0 and v1 checkpoints outside the repository")


def jl(*p):
    with open(os.path.join(*p)) as fh:
        return json.load(fh)


class SavedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.per = {s: {n: jl(R, f"seed_{s}", f"{n}.json") for n in ("metrics", "detection", "run_meta", "histories")} for s in r1.SEEDS}
        cls.sm = jl(R, "validation_summary.json")

    def test_summary_means_are_the_means_of_the_per_seed_results(self):
        for k in ("macro_f1", "weighted_f1", "accuracy"):
            self.assertAlmostEqual(self.sm["fault_type"][k]["mean"], float(np.mean([self.per[s]["metrics"][k] for s in r1.SEEDS])), places=12)
            self.assertAlmostEqual(self.sm["fault_type"][k]["sd"], float(np.std([self.per[s]["metrics"][k] for s in r1.SEEDS], ddof=1)), places=12)
        for s in r1.SEEDS:
            cm = np.array(self.per[s]["metrics"]["confusion_matrix"])
            self.assertEqual(cm.sum(1).tolist(), [10, 6, 3, 5, 1])
        p = self.sm["detection"]["primary"]
        self.assertAlmostEqual(p["f1"]["mean"], float(np.mean([self.per[s]["detection"]["validation"]["primary"]["f1"] for s in r1.SEEDS])), places=12)

    def test_the_primary_detector_is_the_predeclared_rule_for_every_seed(self):
        for s in r1.SEEDS:
            d = self.per[s]["detection"]
            self.assertEqual(d["thresholds"]["primary_p90"], d["validation"]["primary"]["threshold"])
            self.assertEqual((d["calibration"]["n_windows"], d["calibration"]["block_bootstrap_p90"]["block_length"], d["calibration"]["block_bootstrap_p90"]["resamples"]), (329, 7, 1000))
            self.assertAlmostEqual(d["calibration"]["rolled_fraction_above_primary"], 0.1, delta=0.01)
            self.assertEqual(d["validation"]["primary"]["normal_reference"]["windows"], 411)
            self.assertEqual(sum(d["validation"]["primary"]["alert_minutes_by_category"].values()), d["validation"]["primary"]["n_alert_minutes"])
            self.assertEqual(set(d["validation"]), {"primary", "d2_no_rolling", "d1_p95.0", "d1_p97.0", "d1_p99.0", "d1_p99.5", "d1_spot"})

    def test_alignment_selection_and_parameters(self):
        digests = {self.per[s]["run_meta"]["alignment"]["minute_union_sha256"] for s in r1.SEEDS}
        self.assertEqual(digests, {"004d53ea6f834f51ae571edb75c97ff6fd78a05b2cc98cdf7e7d87a7c9c6944d"})
        for s in r1.SEEDS:
            m = self.per[s]["run_meta"]
            self.assertEqual(m["checkpoints"]["parameters"], 176521)
            self.assertTrue(1 <= m["selected"]["s1_epoch"] <= 100 and 1 <= m["selected"]["s2_epoch"] <= 100)
            self.assertEqual(m["checkpoints"]["s2_state_sha256"], m["checkpoints"]["s2_state_sha256_reloaded"])
            self.assertEqual(m["provenance"]["model"], "Model1V1")
            self.assertEqual(m["objectives_used"], list(r1.OBJECTIVES_USED))
            h2 = self.per[s]["histories"]["s2"]
            self.assertEqual(min(e["val_monitor"] for e in h2), h2[m["selected"]["s2_epoch"] - 1]["val_monitor"])        # the selected epoch has the lowest monitor
            h1 = self.per[s]["histories"]["s1"]
            self.assertEqual(min(e["val_masked_mse"] for e in h1), h1[m["selected"]["s1_epoch"] - 1]["val_masked_mse"])
        self.assertEqual(len({self.per[s]["run_meta"]["provenance"]["manifest_content_sha256"] for s in r1.SEEDS}), 1)

    def test_the_seed_zero_rerun_is_recorded_as_byte_identical(self):
        self.assertTrue(self.sm["reproducibility_rerun"]["all_identical"])
        self.assertEqual(self.sm["reproducibility_rerun"]["s2_state_sha256"], self.per[0]["run_meta"]["checkpoints"]["s2_state_sha256"])

    def test_missing_data_summary_structure(self):
        m = jl(R, "missing_summary.json")
        self.assertEqual(list(m["armor_scenarios"]), ["drop_log", "drop_log+metric", "drop_log+trace", "drop_metric", "drop_metric+trace", "drop_trace", "full"])
        self.assertEqual({n for n, s in m["armor_scenarios"].items() if s["hides_all_vms"]}, {"drop_metric", "drop_log+metric", "drop_metric+trace"})
        self.assertEqual({r: s["distinct_masks"] for r, s in m["random_rates"].items()}, {"0.0": 1, "0.1": 10, "0.2": 10, "0.3": 10, "0.4": 10})
        for r in ("0.1", "0.2", "0.3", "0.4"):
            self.assertEqual(m["random_rates"][r]["v1_primary"]["recalibrated"]["f1"]["values"].__len__(), 50)       # 10 masks x 5 seeds
        self.assertEqual(m["armor_scenarios"]["full"]["models"]["v1_primary"]["recalibrated"], m["clean_reference"]["v1_primary"]["recalibrated"])
        self.assertAlmostEqual(m["random_rates"]["0.0"]["v1_primary"]["recalibrated"]["f1"]["mean"], self.sm["detection"]["primary"]["f1"]["mean"], places=12)


class BootstrapResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = jl(os.path.dirname(R), "comparison_v1", "bootstrap_validation_v1_vs_xgboost.json")
        cls.sm = jl(R, "validation_summary.json")

    def test_observed_means_equal_the_summaries_and_the_design_is_the_paired_one(self):
        for k in ("macro_f1", "weighted_f1"):
            self.assertAlmostEqual(self.r["observed"]["transformer"][k], self.sm["fault_type"][k]["mean"], places=12)
        with open(os.path.join(bc.XGB_RES, "validation_summary.json")) as fh:
            x = json.load(fh)["mean_sd_across_seeds"]
        for k in ("macro_f1", "weighted_f1"):
            self.assertAlmostEqual(self.r["observed"]["xgboost"][k], x[k]["mean"], places=12)
        self.assertEqual((self.r["resamples"], self.r["bootstrap_seed"], self.r["support"]), (1000, bc.SEED, [10, 6, 3, 5, 1]))
        self.assertIn("NOT a probability", self.r["reading"]["fraction_positive"])
        for m in bc.METRICS:
            s = self.r["metrics"][m]
            self.assertLessEqual(s["difference_transformer_minus_xgboost"]["ci95_low"], s["difference_transformer_minus_xgboost"]["ci95_high"])
            self.assertAlmostEqual(s["fraction_difference_positive"] + s["fraction_difference_zero"] + s["fraction_difference_negative"], 1.0)

    def test_regeneration_is_identical_and_inputs_are_the_ten_prediction_files(self):
        mt = bc._metrics_module()
        cases, y, pv1, pxg, files = bc.load_inputs(tr_dir=bv.V1_RES)
        again = bc.bootstrap(y, pv1, pxg, mt=mt)
        for k in ("observed", "metrics", "resamples_without_class"):
            self.assertEqual(json.dumps(self.r[k], sort_keys=True), json.dumps(again[k], sort_keys=True))
        self.assertEqual(len(files), 10)
        for name, f in files.items():
            self.assertEqual(self.r["provenance"]["input_files_sha256"][name], bc.sha256_file(f))
            self.assertTrue(f.endswith("predictions.csv"))


@NEEDS_CK
class CheckpointIntegrityTests(unittest.TestCase):
    def test_v1_checkpoint_files_match_the_recorded_hashes_and_stay_outside_the_repo(self):
        for s in r1.SEEDS:
            m = jl(R, f"seed_{s}", "run_meta.json")["checkpoints"]
            self.assertEqual(r1.sha(os.path.join(m["dir"], "s1_best.pt")), m["s1_pt_sha256"])
            self.assertEqual(r1.sha(os.path.join(m["dir"], "s2_best.pt")), m["s2_pt_sha256"])
            self.assertFalse(os.path.abspath(m["dir"]).startswith(r1.REPO))

    def test_v0_stage1_checkpoints_are_unchanged(self):
        for s in r1.SEEDS:
            rec = jl(rx.RESULTS, f"seed_{s}", "run_meta.json")["checkpoints"]["s1_pt_sha256"]
            self.assertEqual(r1.sha(os.path.join(rd.DEFAULT_CKPT, f"seed_{s}", "s1_best.pt")), rec)


if __name__ == "__main__":
    unittest.main()
