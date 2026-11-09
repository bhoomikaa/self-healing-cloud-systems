"""Focused tests of the Model 1 v0 detection experiment: diagnostics, alignment, calibration binding, determinism, leakage guards."""
import json
import os
import re
import sys
import unittest
from unittest import mock

import _m1_model_helpers as MH  # noqa: F401
import numpy as np

import _m1_helpers as H

DET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "detection"))
if DET not in sys.path:
    sys.path.insert(0, DET)

import armor_scoring as sc  # noqa: E402
import interval_diagnostics as idg  # noqa: E402
from model1_transformer import run_detection_v0 as rd, threshold_diagnostics as tdg  # noqa: E402
from prep import detection_calibration as dc, m1_dataset as md  # noqa: E402

B = 2_000_000
CKPT = os.path.join(rd.DEFAULT_CKPT, "seed_0", "s1_best.pt")
NEEDS_REAL = H.needs(H.CACHE_PRESENT and H.ARTIFACTS_PRESENT and os.path.isfile(CKPT), "needs the D1 cache, the M1-03 artifacts and the Stage 1 checkpoints")
NEEDS_ARMOR = H.needs(os.path.isfile(rd.ARMOR_FILES["armor_model.py"]), "needs ARMOR's released source under armor-system/data/raw")


class IntervalDiagnosticTests(unittest.TestCase):
    def test_raw_groups_minus_singletons_equal_the_armor_intervals(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            alerts = np.sort(B + 60 * rng.choice(300, size=int(rng.integers(0, 60)), replace=False)).tolist()
            raw = idg.raw_alert_groups(alerts)
            self.assertEqual([g for g in raw if g[0] != g[1]], sc.get_pre_interval(alerts))
            self.assertEqual(idg.count_dropped_singletons(alerts), sum(g[0] == g[1] for g in raw))

    def test_singleton_count_examples(self):
        self.assertEqual(idg.count_dropped_singletons([B, B + 1200, B + 1260, B + 3000]), 2)
        self.assertEqual(idg.count_dropped_singletons([]), 0)
        self.assertEqual(idg.count_dropped_singletons([B, B, B]), 1)

    def test_delay_is_earliest_matching_interval_start_minus_label(self):
        iv = [(B + 100, B + 160), (B + 300, B + 420), (B + 9000, B + 9060)]
        d = idg.detection_delays(iv, [B + 350, B + 5000])
        self.assertEqual(d, [(B + 350, -250)])                                  # the earliest of the two reaching intervals starts at B+100, 250 s before the label
        self.assertEqual(idg.delay_summary(d)["n_negative"], 1)
        self.assertEqual(idg.delay_summary([])["median_s"], None)
        s = idg.delay_summary([(0, 120), (1, -60), (2, 300)])
        self.assertEqual((s["median_s"], s["min_s"], s["max_s"], s["n_hits"]), (120.0, -60.0, 300.0, 3))

    def test_coverage_and_padding(self):
        m = B + 60 * np.arange(20)
        iv = [(B + 300, B + 420)]
        self.assertEqual(idg.in_intervals(m, iv).sum(), 3)
        self.assertEqual(idg.in_intervals(m, iv, 60).sum(), 5)

    def test_interval_classes_are_exclusive_and_ignore_overlap_is_counted_separately(self):
        iv = [(B, B + 120), (B + 5000, B + 5120), (B + 9000, B + 9120)]
        labels = [B + 60]
        ign = [B + 60, B + 5060]                      # an ignore minute inside the hit interval and one inside the second interval
        c = idg.classify_intervals(iv, labels, ign)
        self.assertEqual((c["per_interval"], c["hit"], c["ignore_overlap"], c["normal_only"], c["overlap_ignore_total"]), (["hit", "ignore_overlap", "normal_only"], 1, 1, 1, 2))


class ThresholdDiagnosticTests(unittest.TestCase):
    def test_ranks_used_for_329_windows(self):
        r = tdg.percentile_ranks(329, 99.0)
        self.assertEqual((r["lower_rank_ascending"], r["upper_rank_ascending"], r["lower_rank_from_top"], r["upper_rank_from_top"]), (325, 326, 5, 4))
        self.assertAlmostEqual(r["weight_on_upper"], 0.72, places=9)
        self.assertEqual((tdg.percentile_ranks(329, 99.5)["lower_rank_from_top"], tdg.percentile_ranks(329, 99.5)["upper_rank_from_top"]), (3, 2))
        self.assertEqual(tdg.percentile_ranks(329, 95.0)["lower_rank_ascending"], 312)

    def test_ranks_reproduce_numpys_linear_percentile(self):
        x = np.sort(np.random.default_rng(0).normal(size=329))
        for p in (95.0, 97.0, 99.0, 99.5):
            r = tdg.percentile_ranks(329, p)
            manual = x[r["lower_rank_ascending"] - 1] * (1 - r["weight_on_upper"]) + x[r["upper_rank_ascending"] - 1] * r["weight_on_upper"]
            self.assertAlmostEqual(manual, np.percentile(x, p, method="linear"), places=12)

    def test_block_resample_structure(self):
        idx = tdg.block_resample_indices(329, 7, np.random.default_rng(0))
        self.assertEqual(len(idx), 329)
        self.assertTrue(((idx >= 0) & (idx < 329)).all())
        blocks = idx[:322].reshape(46, 7)
        self.assertTrue((np.diff(blocks, axis=1) == 1).all())                       # each block is 7 consecutive windows
        self.assertTrue((idx == tdg.block_resample_indices(329, 7, np.random.default_rng(0))).all())
        with self.assertRaises(ValueError):
            tdg.block_resample_indices(5, 7, np.random.default_rng(0))

    def test_block_bootstrap_is_reproducible_and_reports_all_methods(self):
        e = np.random.default_rng(2).gamma(2.0, 0.05, size=(120, 8))
        a = tdg.block_bootstrap_thresholds(e, None, block=7, n_boot=50, seed=3)
        b = tdg.block_bootstrap_thresholds(e, None, block=7, n_boot=50, seed=3)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertEqual(set(a["methods"]), set(dc.METHODS))
        self.assertEqual((a["n_blocks"], a["blocks_per_resample"], a["windows"]), (114, 18, 120))
        for m in dc.METHODS:
            r = a["methods"][m]
            self.assertLessEqual(r["ci95_low"], r["ci95_high"])
        full = dc.calibrate(e)["thresholds"]
        for m in dc.METHODS:
            self.assertAlmostEqual(a["methods"][m]["tau_full_sample"], full[m], places=12)

    def test_consecutive_fraction(self):
        self.assertEqual(tdg.consecutive_fraction([1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(tdg.consecutive_fraction([1, 2, 5, 6]), 2 / 3)


class EvaluateAlertsTests(unittest.TestCase):
    def setUp(self):
        n = 869
        self.ts = 1_650_000_000 + 60 * np.arange(n)
        self.cat = np.array(["normal_reference"] * n, dtype=object)
        self.cat[100:160] = "ignore"
        self.cat[160:166] = "event_associated"
        self.labels = [int(self.ts[163]) + 7]

    def flags(self, idx):
        f = np.zeros(len(self.ts), bool)
        f[idx] = True
        return f

    def test_a_hit_interval_and_its_metrics(self):
        r = rd.evaluate_alerts(self.flags([160, 161, 162]), self.ts, self.cat, self.labels)
        self.assertEqual((r["scored"], r["TP"], r["FP"], r["FN"], r["precision"], r["recall"], r["f1"]), (True, 1, 0, 0, 1.0, 1.0, 1.0))
        self.assertEqual((r["n_intervals"], r["n_alert_minutes"], r["singleton_alerts_dropped"]), (1, 3, 0))
        self.assertEqual(r["delay"]["n_hits"], 1)
        self.assertEqual(r["delay"]["median_s"], float(self.ts[160] - self.labels[0]))
        self.assertEqual(r["intervals"]["hit"], 1)
        self.assertEqual(r["normal_reference"]["alarm_windows"], 0)

    def test_false_alarm_classes_normal_rates_and_singletons(self):
        flags = self.flags([160, 161, 120, 121, 400, 401, 402, 700])      # hit | ignore-overlap | normal-only | singleton (dropped)
        r = rd.evaluate_alerts(flags, self.ts, self.cat, self.labels)
        self.assertEqual((r["n_intervals"], r["singleton_alerts_dropped"]), (3, 1))
        self.assertEqual(r["intervals"], {"hit": 1, "ignore_overlap_not_hit": 1, "normal_only_false_alarm": 1, "overlapping_ignore_regions_total": 1})
        self.assertEqual(r["armor_false_positive_intervals"], 2)
        self.assertEqual((r["TP"], r["FP"]), (1, 2))
        nr = r["normal_reference"]
        self.assertEqual((nr["windows"], nr["alarm_windows"]), (869 - 60 - 6, 4))                         # 400, 401, 402 and the singleton 700 are normal windows
        self.assertEqual(nr["inside_interval_windows"], 3 + 0)                                           # 400..402 are in a kept interval; the dropped singleton is not
        self.assertEqual(r["alert_minutes_by_category"], {"event_associated": 2, "normal_reference": 4, "ignore": 2})

    def test_no_alert_and_zero_hit_cases_are_reported_as_no_score(self):
        r = rd.evaluate_alerts(self.flags([]), self.ts, self.cat, self.labels)
        self.assertEqual((r["scored"], r["precision"], r["recall"], r["f1"], r["TP"], r["FP"], r["FN"]), (False, 0.0, 0.0, 0.0, 0, 0, 1))
        self.assertIn("ZeroDivisionError", r["reason"])
        r = rd.evaluate_alerts(self.flags([500, 501]), self.ts, self.cat, self.labels)
        self.assertEqual((r["scored"], r["f1"], r["FP"], r["TP"]), (False, 0.0, 1, 0))
        self.assertIn("NaN", r["reason"])

    def test_aggregate_counts_no_score_seeds_as_zero_with_sample_sd(self):
        mk = lambda f, scored: {"validation": {"methods": {m: {str(p): {"precision": f, "recall": f, "f1": f, "n_intervals": 1, "armor_false_positive_intervals": 1, "singleton_alerts_dropped": 0,   # noqa: E731
                                                                      "coverage_padded_300s": 0.1, "TP": 1, "scored": scored, "threshold": 1.0,
                                                                      "normal_reference": {"alarm_rate": 0.0, "inside_interval_rate": 0.0}, "intervals": {"normal_only_false_alarm": 0, "ignore_overlap_not_hit": 0, "overlapping_ignore_regions_total": 0},
                                                                      "delay": {"median_s": None}} for p in rd.PERCENTILES} for m in dc.METHODS}}}
        a = rd.aggregate([mk(1.0, True), mk(0.0, False), mk(1.0, True)])
        c = a["m2_max_normalised"]["99.0"]
        self.assertEqual(c["n_no_score_seeds"], 1)
        self.assertAlmostEqual(c["f1"]["mean"], 2 / 3)
        self.assertAlmostEqual(c["f1"]["sd"], float(np.std([1, 0, 1], ddof=1)))
        self.assertIsNone(c["delay_median_s_over_seeds_with_hits"])


class StaticGuardTests(unittest.TestCase):
    def test_only_calibration_and_validation_objectives_and_no_gradient_or_official_labels(self):
        with open(rd.__file__) as fh:
            src = fh.read()
        from prep import m1_split as ms
        named = {k for k in ms.OBJECTIVES if re.search(r"[\"']%s[\"']" % k, src)}
        self.assertEqual(named, set(rd.OBJECTIVES_USED))
        body = src.split('"""', 2)[2]
        self.assertIsNone(re.search(r"torch\.optim|optimizer\.|\.backward\(|\.step\(|training_dataset|training_loader|ad_cases|\.pkl|pre_interval\.json", body))
        self.assertIsNone(re.search(r"""["'](test|fit|embargo)["']""", body))
        self.assertEqual(rd.PRIMARY, ("m2_max_normalised", 99.0))
        self.assertEqual((rd.N_BOOT, rd.BLOCK, rd.DELAY, rd.IMPACT), (1000, 7, 600, 300))


@NEEDS_ARMOR
class TimestampAlignmentSourceTests(unittest.TestCase):
    def test_armor_assigns_the_target_sample_timestamp_to_each_score(self):
        import ast
        with open(rd.ARMOR_FILES["armor_model.py"]) as fh:
            src = fh.read()
        seg = next(ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "create_dataloader_AR")
        self.assertIn("window_size=6", seg)
        self.assertIn("series_len = window_size + 1", seg)
        self.assertRegex(seg, r"torch\.tensor\(series_sample\[-1\]\[0\]\)")                            # window timestamp = last (target) sample
        self.assertRegex(seg, r"torch\.stack\(\[torch\.as_tensor\(step\[2\]\) for step in series_sample\[:-1\]\]\)")   # inputs = the earlier samples
        with open(rd.ARMOR_FILES["deviation.py"]) as fh:
            dev = fh.read()
        self.assertIn("timestamp_list.extend(batch_ts", dev)
        with open(rd.ARMOR_FILES["anomaly_detector.py"]) as fh:
            ad = fh.read()
        self.assertIn('eval_slds_sum = eval_slds[["timestamp"]].copy()', ad)

    def test_model1_alert_time_is_the_target_sample(self):
        from model1_transformer import window_errors as we
        self.assertEqual(we.TARGET_POSITION, 6)


@NEEDS_REAL
class RealTableTests(unittest.TestCase):
    """Table/metadata checks only: no model is run."""

    @classmethod
    def setUpClass(cls):
        cls.D = md.Model1Data.load()

    def test_calibration_reference_is_exactly_329_normal_reference_windows(self):
        t = self.D.table
        rows = md.objective_rows(self.D, "s3_calibration_reference")
        self.assertEqual(len(rows), 329)
        self.assertEqual((set(t.role[rows]), set(t.category[rows]), set(t.case_index[rows].tolist())), ({"calibration"}, {"normal_reference"}, {-1}))
        with self.assertRaises(md.LeakageError):
            md.training_dataset(self.D, "s3_calibration_reference")                    # never a gradient objective

    def test_validation_scoring_set_is_869_contiguous_target_minutes(self):
        t = self.D.table
        rows = md.objective_rows(self.D, "validation_all")
        self.assertEqual(len(rows), 869)
        self.assertEqual({c: int((t.category[rows] == c).sum()) for c in ("event_associated", "normal_reference", "ignore")}, {"event_associated": 150, "normal_reference": 411, "ignore": 308})
        tgt = np.sort(t.T[rows][:, 6])
        self.assertTrue((np.diff(tgt) == 60).all())
        self.assertTrue((np.diff(t.T[rows], axis=1) == 60).all())                       # seven consecutive samples per window
        self.assertTrue((t.T[rows][:, 6] == self.D.ts[t.start[rows] + 6]).all())         # target = sample start + 6

    def test_validation_labels_are_the_25_documented_validation_incidents_inside_the_scored_span(self):
        labels = rd.validation_labels(self.D)
        self.assertEqual(len(labels), 25)
        self.assertEqual(labels, sorted(labels))
        lo, hi = int(self.D.boundaries[0]), int(self.D.boundaries[1])
        self.assertTrue(all(lo <= x < hi for x in labels))
        t = self.D.table
        rows = md.objective_rows(self.D, "validation_all")
        tgt = t.T[rows][:, 6]
        self.assertTrue(all(tgt.min() <= x <= tgt.max() for x in labels))


@NEEDS_REAL
class SeedPipelineTests(unittest.TestCase):
    """Seed 0 end to end with a reduced bootstrap: binding, determinism, unmodified checkpoint, agreement with the saved full run."""

    @classmethod
    def setUpClass(cls):
        from model1_transformer import training as T
        from model1_transformer.spec import AdapterSpec
        cls.D = md.Model1Data.load()
        cls.cfg = T.load_train_config()
        cls.spec = AdapterSpec.from_d1(cls.D.schema, cls.D.applicable)
        cls.labels = rd.validation_labels(cls.D)

    def run_seed0(self):
        with mock.patch.object(rd, "N_BOOT", 20):
            return rd._py(rd.run_seed(self.D, self.cfg, self.spec, 0, self.labels))

    def test_binding_determinism_checkpoint_unmodified_and_agreement_with_saved_results(self):
        before = rd.sha256_file(CKPT)
        a, b = self.run_seed0(), self.run_seed0()
        self.assertEqual(rd.sha256_file(CKPT), before)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        cal = a["calibration"]
        with open(os.path.join(rd.EXPERIMENT, "seed_0", "run_meta.json")) as fh:
            rec = json.load(fh)
        self.assertEqual(cal["binding"]["pt_file_sha256"], rec["checkpoints"]["s1_pt_sha256"])
        self.assertEqual(cal["binding"]["manifest_content_sha256"], self.D.manifest["content_sha256"])
        self.assertEqual(cal["n_windows"], 329)
        self.assertEqual(len(cal["node_median"]), 46)
        self.assertEqual(set(cal["ranks_used"]), {"95.0", "97.0", "99.0", "99.5"})
        self.assertEqual(a["validation"]["n_windows"], 869)
        prim = a["validation"]["methods"][rd.PRIMARY[0]]["99.0"]
        self.assertEqual(prim["threshold"], cal["thresholds_p99_primary"][rd.PRIMARY[0]])
        saved = os.path.join(rd.RESULTS, "seed_0.json")
        if os.path.isfile(saved):                                                        # everything except the (reduced) bootstrap equals the saved full run
            with open(saved) as fh:
                s = json.load(fh)
            for k in ("errors_sha256", "node_median", "node_iqr", "epsilon", "thresholds_p99_primary", "thresholds_sensitivity", "ranks_used", "binding"):
                self.assertEqual(json.dumps(s["calibration"][k], sort_keys=True), json.dumps(cal[k], sort_keys=True), k)
            self.assertEqual(json.dumps(s["validation"], sort_keys=True), json.dumps(a["validation"], sort_keys=True))

    def test_models_scored_here_are_stage_one_checkpoints_and_loading_checks_the_experiment_record(self):
        model, binding = rd.load_s1(self.D, self.cfg, self.spec, 3)
        self.assertEqual(binding["s1_selected_epoch"], 92)
        self.assertFalse(model.training)
        with mock.patch.object(rd, "EXPERIMENT", "/nonexistent"):
            with self.assertRaises(OSError):
                rd.load_s1(self.D, self.cfg, self.spec, 3)


if __name__ == "__main__":
    unittest.main()
