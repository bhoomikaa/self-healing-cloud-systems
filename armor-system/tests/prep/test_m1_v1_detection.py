"""v1 detection: thresholds, bootstrap, forecast-error scoring with node validity, selection monitor, outputs collector, missing-data masks."""
import os
import re
import sys
import types
import unittest

import _m1_model_helpers as MH
import numpy as np
import torch

import _m1_helpers as H
from model1_transformer import case_eval as ce, detection_v1 as dv, missing_eval as me, rolling, run_detection_v0 as rd, window_errors as we
from model1_transformer.model import Model1
from model1_transformer.model_v1 import Model1V1
from model1_transformer.sim_missing import MissingBlocks
from prep import detection_calibration as dc, m1_dataset as md

NEEDS_REAL = H.needs(H.CACHE_PRESENT and H.ARTIFACTS_PRESENT, "needs the D1 cache and the M1-03 artifacts")
NEEDS_ARMOR = H.needs(os.path.isfile(rd.ARMOR_FILES["anomaly_detector.py"]), "needs ARMOR's released source under armor-system/data/raw")


def synthetic_outputs(n=120, seed=0, role="validation", bump=None):
    """Outputs dict for n contiguous minutes: 40 ignore, 12 event, rest normal; detection logits noisy with an optional bump on the event windows."""
    rng = np.random.default_rng(seed)
    ts = 1_700_000_000 + 60 * np.arange(n)
    cat = np.array(["normal_reference"] * n, dtype=object)
    cat[20:60] = "ignore"
    cat[60:72] = "event_associated"
    det = rng.normal(0, 1, n)
    if bump:
        det[60:72] += bump
    errors = rng.gamma(2.0, 0.05, size=(n, 8))
    return {"fault_logits": rng.normal(size=(n, 5)), "det_logit": det, "errors": errors, "node_valid": np.ones((n, 8), bool), "target_ts": ts, "start_index": np.arange(n), "case_index": np.where(cat == "event_associated", 7, -1),
            "fault_class": np.where(cat == "event_associated", 2, -1), "ambiguous": np.zeros(n, bool), "timestamps": ts[:, None] + 60 * np.arange(-6, 1)[None], "role": np.array([role] * n), "category": cat}


class ThresholdTests(unittest.TestCase):
    def test_primary_threshold_is_the_90th_percentile_of_the_rolled_calibration_scores(self):
        cal = synthetic_outputs(329, 1, role="calibration")
        th, info = dv.supervised_thresholds(cal, seed=0, bootstrap=False)
        rolled = rolling.rolling_max_causal(cal["det_logit"], cal["target_ts"], cal["role"])
        self.assertEqual(th["primary_p90"], float(np.percentile(rolled, 90, method="linear")))
        self.assertEqual(th["d2_p90_unrolled"], float(np.percentile(cal["det_logit"], 90, method="linear")))
        for p in (95.0, 97.0, 99.0, 99.5):
            self.assertEqual(th[f"p{p}"], float(np.percentile(rolled, p, method="linear")))
        self.assertGreater(th["primary_p90"], th["d2_p90_unrolled"])                       # rolling can only raise scores
        self.assertEqual((info["ranks_used_p90"]["lower_rank_ascending"], info["ranks_used_p90"]["upper_rank_ascending"]), (296, 297))
        self.assertNotIn("spot", th)
        self.assertAlmostEqual(info["rolled_fraction_above_primary"], 0.1, delta=0.01)

    def test_unsorted_calibration_input_gives_the_same_thresholds(self):
        cal = synthetic_outputs(120, 2, role="calibration")
        perm = np.random.default_rng(1).permutation(120)
        shuf = {k: (v[perm] if isinstance(v, np.ndarray) and v.shape[:1] == (120,) else v) for k, v in cal.items()}
        self.assertEqual(dv.supervised_thresholds(cal, 0, bootstrap=False)[0], dv.supervised_thresholds(shuf, 0, bootstrap=False)[0])

    def test_the_spot_diagnostic_calls_arms_function_on_the_rolled_scores_with_its_parameters(self):
        seen = {}

        def fake(scores, q, level):
            seen.update(n=len(scores), q=q, level=level)
            return 1.25
        th, _ = dv.supervised_thresholds(synthetic_outputs(329, 3, role="calibration"), 0, fake, bootstrap=False)
        self.assertEqual((th["spot"], seen), (1.25, {"n": 329, "q": 0.1, "level": 0.95}))

    @NEEDS_ARMOR
    def test_real_armor_spot_runs_on_rolled_scores(self):
        from model1_transformer import v1_audit_spot as au
        _, get_threshold, _ = au.load_armor_spot()
        th, _ = dv.supervised_thresholds(synthetic_outputs(329, 4, role="calibration"), 0, get_threshold, bootstrap=False)
        self.assertTrue(np.isfinite(th["spot"]))

    def test_block_bootstrap_of_the_percentile_is_reproducible_and_well_formed(self):
        x = np.random.default_rng(5).normal(size=329)
        a, b = dv.block_bootstrap_percentile(x, 90.0, 7, 200, 11), dv.block_bootstrap_percentile(x, 90.0, 7, 200, 11)
        self.assertEqual(a, b)
        self.assertNotEqual(a, dv.block_bootstrap_percentile(x, 90.0, 7, 200, 12))
        self.assertLessEqual(a["ci95_low"], a["tau_full_sample"] + 1.0)
        self.assertLess(a["ci95_low"], a["ci95_high"])
        self.assertEqual((a["blocks_per_resample"], a["windows"], a["block_length"]), (47, 329, 7))
        self.assertEqual(a["tau_full_sample"], float(np.percentile(x, 90.0)))


class ValidationScoringTests(unittest.TestCase):
    def test_a_bump_on_the_event_windows_is_detected_and_diagnostics_exist(self):
        cal = synthetic_outputs(329, 1, role="calibration")
        val = synthetic_outputs(120, 2, bump=8.0)
        th, _ = dv.supervised_thresholds(cal, 0, bootstrap=False)
        labels = [int(val["target_ts"][65]) + 11]
        r = dv.supervised_validation(val, th, labels)
        self.assertEqual(set(r), {"primary", "d2_no_rolling", "d1_p95.0", "d1_p97.0", "d1_p99.0", "d1_p99.5"})
        self.assertEqual((r["primary"]["TP"], r["primary"]["scored"]), (1, True))
        self.assertEqual(r["primary"]["threshold"], th["primary_p90"])
        self.assertEqual(r["primary"]["alert_minutes_by_category"]["event_associated"], 12)

    def test_the_rolling_maximum_turns_an_isolated_exceedance_into_an_interval(self):
        cal = synthetic_outputs(329, 1, role="calibration")
        val = synthetic_outputs(120, 3)
        val["det_logit"][:] = -5.0
        val["det_logit"][64] = 50.0                                                      # one isolated exceeding minute
        th, _ = dv.supervised_thresholds(cal, 0, bootstrap=False)
        labels = [int(val["target_ts"][64])]
        r = dv.supervised_validation(val, th, labels)
        self.assertEqual((r["primary"]["n_alert_minutes"], r["primary"]["n_intervals"], r["primary"]["singleton_alerts_dropped"]), (2, 1, 0))
        self.assertEqual((r["d2_no_rolling"]["n_alert_minutes"], r["d2_no_rolling"]["n_intervals"], r["d2_no_rolling"]["singleton_alerts_dropped"]), (1, 0, 1))
        self.assertEqual(r["primary"]["delay"]["median_s"], float(val["target_ts"][64] - labels[0]))     # the interval starts at the exceeding minute: no shift

    def test_calibration_windows_are_never_scored_as_validation_by_the_guards(self):
        with self.assertRaises(we.ScoringRoleError):
            dv.collect_outputs(Model1.build(MH.synthetic_spec_and_applicable()[0], seed=0), [], ("test",))


class ForecastScoringTests(unittest.TestCase):
    def test_all_valid_matches_the_committed_calibration_module(self):
        rng = np.random.default_rng(0)
        e = rng.gamma(2.0, 0.05, size=(200, 10))
        valid = np.ones_like(e, bool)
        st, ref = dv.masked_fit_statistics(e, valid), dc.fit_statistics(e)
        for k in ("median", "iqr"):
            np.testing.assert_array_equal(st[k], ref[k])
        self.assertEqual(st["epsilon"], ref["epsilon"])
        mine, theirs = dv.forecast_scores(e, valid, st), dc.window_scores(e, ref)
        for k in theirs:
            np.testing.assert_array_equal(mine[k], theirs[k])

    def test_invalid_node_windows_are_excluded_and_absent_nodes_flagged(self):
        rng = np.random.default_rng(1)
        e = rng.gamma(2.0, 0.05, size=(300, 6))
        valid = np.ones_like(e, bool)
        valid[::3, 2] = False
        valid[:, 5] = False                                                              # node 5 never has an effective entry
        e[~valid] = 0.0
        st = dv.masked_fit_statistics(e, valid)
        self.assertFalse(st["node_valid"][5])
        self.assertEqual((st["median"][5], st["iqr"][5]), (0.0, 0.0))
        self.assertAlmostEqual(st["median"][2], float(np.median(e[valid[:, 2], 2])), places=12)    # zeros of invalid windows do not enter the statistics
        sc = dv.forecast_scores(e, valid, st)
        big = e.copy()
        big[5, 5] = 1e9                                                                  # an error at a node that is invalid everywhere cannot raise a score
        self.assertTrue(np.array_equal(dv.forecast_scores(big, valid, st)["m1_raw_max"], sc["m1_raw_max"]))
        none = np.zeros((1, 6), bool)
        self.assertTrue(np.isneginf(dv.forecast_scores(np.zeros((1, 6)), none, st)["m2_max_normalised"]).all())

    def test_thresholds_and_scoring_run_through_the_alert_scorer(self):
        cal, val = synthetic_outputs(329, 1, role="calibration"), synthetic_outputs(120, 2)
        val["errors"][62] += 5.0
        stats, th, info = dv.forecast_calibration(cal)
        self.assertEqual(set(th), {"m1_raw_max", "m2_max_normalised", "m3_topk_mean_normalised"})
        self.assertEqual(set(th["m2_max_normalised"]), {"90.0", "95.0", "97.0", "99.0", "99.5"})
        r = dv.forecast_validation(val, stats, th, [int(val["target_ts"][62])])
        self.assertEqual(r["m1_raw_max"]["99.0"]["alert_minutes_by_category"]["event_associated"], 1)
        self.assertEqual(info["n_windows"], 329)


class MonitorTests(unittest.TestCase):
    def test_selection_monitor_is_case_nll_plus_balanced_bce(self):
        val = synthetic_outputs(120, 5)
        m = dv.selection_monitor(val, ce)
        ev = val["category"] == "event_associated"
        probs = dv.softmax(val["fault_logits"][ev]).mean(0)                              # one case: mean probability of its windows
        self.assertAlmostEqual(m["case_nll"], -np.log(probs[2]), places=10)
        e, n = val["det_logit"][ev], val["det_logit"][val["category"] == "normal_reference"]
        manual = 0.5 * (np.mean(np.log1p(np.exp(-e))) + np.mean(np.log1p(np.exp(n))))
        self.assertAlmostEqual(m["balanced_bce"], manual, places=10)
        self.assertAlmostEqual(m["monitor"], m["case_nll"] + m["balanced_bce"], places=12)

    def test_ignore_windows_do_not_enter_the_monitor(self):
        val = synthetic_outputs(120, 5)
        base = dv.selection_monitor(val, ce)["monitor"]
        val["det_logit"][20:60] = 1e6                                                   # the ignore block
        val["fault_logits"][20:60] = 1e6
        self.assertEqual(dv.selection_monitor(val, ce)["monitor"], base)

    def test_bce_helper_is_stable_and_correct(self):
        self.assertAlmostEqual(float(dv.bce_with_logits(np.array([0.0]), 1.0)[0]), np.log(2), places=12)
        self.assertTrue(np.isfinite(dv.bce_with_logits(np.array([1e4, -1e4]), 1.0)).all())


class CollectorTests(unittest.TestCase):
    def test_outputs_match_the_window_error_collector_and_v0_models_have_no_detection_logit(self):
        spec, app, _ = MH.synthetic_spec_and_applicable()
        batches = []
        for s in range(2):
            b = MH.make_batch(app, B=3, seed=s, holes=0.1)
            b.update({"role": ["validation"] * 3, "category": ["event_associated"] * 3, "case_index": torch.tensor([5, 5, 6]), "start_index": torch.arange(3) + 10 * s,
                      "timestamps": 1_000_000 + 60 * (torch.arange(3)[:, None] + torch.arange(7)[None]) + 10_000 * s, "fault_class": torch.tensor([1, 1, 2]), "fault_label_ambiguous": torch.zeros(3, dtype=torch.bool)})
            batches.append(b)
        v1, v0 = MH.make_model(spec, seed=1, cfg=None) if False else Model1V1.build(spec, seed=1).eval(), Model1.build(spec, seed=1).eval()
        o1, o0 = dv.collect_outputs(v1, batches, ("validation",)), dv.collect_outputs(v0, batches, ("validation",))
        ref = we.collect_window_errors(v1, batches)
        self.assertTrue(np.array_equal(o1["errors"], ref["errors"]) and np.array_equal(o1["target_ts"], ref["target_ts"]))
        self.assertTrue(np.isfinite(o1["det_logit"]).all() and np.isnan(o0["det_logit"]).all())
        self.assertTrue(np.array_equal(o1["fault_logits"], o0["fault_logits"]))
        self.assertEqual(o1["fault_logits"].shape, (6, 5))
        for role in ("fit", "test", "embargo"):
            bad = [dict(b, role=[role] * 3) for b in batches]
            with self.assertRaises(we.ScoringRoleError):
                dv.collect_outputs(v1, bad, ("validation",))
        with self.assertRaises(we.ScoringRoleError):
            dv.collect_outputs(v1, batches, ("validation", "test"))


def fake_data(S=400, seed=0):
    spec, app, schema = MH.synthetic_spec_and_applicable()
    return types.SimpleNamespace(Xt=np.zeros((S, 46, 130), np.float32), schema=schema, applicable=app, observed=None), spec


class MissingMaskTests(unittest.TestCase):
    def test_armor_scenarios_are_the_six_drop_sets_plus_full_and_map_to_schema_channels(self):
        sc = me.armor_scenarios()
        self.assertEqual([n for n, _ in sc], ["full", "drop_log", "drop_metric", "drop_trace", "drop_log+metric", "drop_log+trace", "drop_metric+trace"])
        D, spec = fake_data(10)
        r = D.schema["modality_index_ranges"]
        for name, drop in sc:
            obs = me.scenario_observed(D, drop)
            hidden = np.nonzero(~obs[0, 0])[0].tolist()
            self.assertEqual(hidden, sorted({i for m in drop for i in r[m]}))
            self.assertTrue((obs == obs[0:1, 0:1]).all())                                # same for every node and every minute
        self.assertEqual((len(r["metric"]), len(r["trace"]), len(r["log"])), (110, 9, 11))

    def test_metric_scenarios_hide_all_vms_and_pod_only_scenarios_do_not(self):
        D, spec = fake_data(10)
        flags = {n: me.hides_all_vms(D, spec, me.scenario_observed(D, d)) for n, d in me.armor_scenarios()}
        self.assertEqual({n for n, f in flags.items() if f}, {"drop_metric", "drop_log+metric", "drop_metric+trace"})

    def test_random_masks_are_deterministic_paired_and_at_the_requested_rate(self):
        D, spec = fake_data(400)
        blocks = MissingBlocks(spec)
        a = me.random_observed(D, blocks, 2, 3, 0.2)
        self.assertTrue(np.array_equal(a, me.random_observed(D, blocks, 2, 3, 0.2)))
        self.assertFalse(np.array_equal(a, me.random_observed(D, blocks, 2, 4, 0.2)))
        self.assertFalse(np.array_equal(a, me.random_observed(D, blocks, 3, 3, 0.2)))
        self.assertTrue(me.random_observed(D, blocks, 0, 7, 0.0).all())                  # rate 0: nothing hidden whatever the mask index
        hidden = ~a
        self.assertFalse(hidden[:, ~D.applicable].any())                                 # natural absence is never hidden
        bapp = blocks.block_applicable(D.applicable)
        per_block = np.zeros((400, 46, 3), bool)
        for m in range(3):
            for n in range(46):
                cols = blocks.chan_block[n, m]
                if cols.any():
                    per_block[:, n, m] = hidden[:, n][:, cols].any(-1)
        self.assertAlmostEqual(float(per_block[:, bapp].mean()), 0.2, delta=0.01)
        for rate in (0.1, 0.3, 0.4):
            h = ~me.random_observed(D, blocks, 1, 0, rate)
            self.assertGreater(h.mean(), 0)

    def test_the_masked_view_leaves_the_original_untouched(self):
        D, spec = fake_data(10)
        obs = me.scenario_observed(D, ("log",))
        v = me.masked_view(D, obs)
        self.assertIsNone(D.observed)
        self.assertIs(v.observed, obs)
        with self.assertRaises(ValueError):
            me.masked_view(D, obs.astype(np.uint8))

    def test_mask_constants_follow_the_protocol(self):
        self.assertEqual((me.RATES, me.N_MASKS), ((0.0, 0.1, 0.2, 0.3, 0.4), 10))


@NEEDS_REAL
class RealMaskWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = md.Model1Data.load()

    def test_masked_windows_carry_the_mask_on_inputs_and_the_target_minute(self):
        from model1_transformer.spec import AdapterSpec
        spec = AdapterSpec.from_d1(self.D.schema, self.D.applicable)
        obs = me.random_observed(self.D, MissingBlocks(spec), 2, 1, 0.3)
        v = me.masked_view(self.D, obs)
        ds = md.evaluation_dataset(v, "validation_all")
        for i in (0, 100, 500):
            it = ds[i]
            s = int(v.table.start[ds.rows[i]])
            sl = obs[s:s + 7] & self.D.applicable[None]
            self.assertTrue(np.array_equal(it["input_mask"].numpy(), sl[:6]))
            self.assertTrue(np.array_equal(it["target_mask"].numpy(), sl[6]))                # the target minute follows the same mask
            self.assertTrue((it["inputs"].numpy()[~sl[:6]] == 0).all())
        self.assertIsNone(self.D.observed)

    def test_armor_drop_metric_removes_every_vm_entry_from_windows(self):
        from model1_transformer.spec import AdapterSpec
        spec = AdapterSpec.from_d1(self.D.schema, self.D.applicable)
        v = me.masked_view(self.D, me.scenario_observed(self.D, ("metric",)))
        it = md.evaluation_dataset(v, "validation_all")[10]
        vm = spec.node_type == spec.type_names.index("vm")
        self.assertFalse(it["input_mask"][:, torch.as_tensor(vm)].any() or it["target_mask"][torch.as_tensor(vm)].any())
        self.assertTrue(it["input_mask"][:, torch.as_tensor(~vm)].any())


class StaticTests(unittest.TestCase):
    def test_detection_and_missing_modules_have_no_gradient_test_or_official_label_access(self):
        for mod in (dv, me):
            with open(mod.__file__) as fh:
                body = fh.read().split('"""', 2)[2]
            self.assertIsNone(re.search(r"torch\.optim|\.backward\(|\.step\(|training_dataset|training_loader|ad_cases|\.pkl|pre_interval\.json", body), mod.__name__)
            self.assertIsNone(re.search(r"""["'](test|fit|embargo)["']""", body), mod.__name__)
        self.assertEqual((dv.P_PRIMARY, dv.P_D1, dv.P_D3, dv.D3_PRIMARY), (90.0, (95.0, 97.0, 99.0, 99.5), (90.0, 95.0, 97.0, 99.0, 99.5), ("m2_max_normalised", 99.0)))
        self.assertEqual((dv.BOOT_SEED, dv.N_BOOT, dv.BLOCK), (20261028, 1000, 7))


if __name__ == "__main__":
    unittest.main()
