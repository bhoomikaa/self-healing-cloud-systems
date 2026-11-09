"""v1 design audit checks: ARMOR's SPOT behaviour, the saved feasibility audit, ARMOR's six missing-data scenarios on our schema, protocol-vs-code facts."""
import ast
import itertools
import json
import os
import re
import sys
import unittest

import _m1_model_helpers as MH  # noqa: F401
import numpy as np

import _m1_helpers as H
from model1_transformer import run_detection_v0 as rd, v1_audit_spot as au
from model1_transformer.model import Model1

MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "model1_transformer"))
PROTOCOL = os.path.join(MODELS, "V1_PROTOCOL.md")
AUDIT = au.OUT
HAVE_ARMOR = all(os.path.isfile(p) for p in (au.SPOT_FILE, au.DETECTOR_FILE))
NEEDS_ARMOR = H.needs(HAVE_ARMOR, "needs ARMOR's released source under armor-system/data/raw (git-ignored)")


@NEEDS_ARMOR
class ArmorSpotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.SPOT, get_threshold, cls.prov = au.load_armor_spot()
        cls.get_threshold = staticmethod(get_threshold)

    def test_initialisation_internals_follow_the_armor_formula(self):
        x = np.random.default_rng(0).gamma(2.0, 3.0, size=329)
        r = au.spot_fit(self.SPOT, x)
        s = np.sort(x)
        t = s[int(0.95 * 329)]
        self.assertEqual(r["t"], t)
        self.assertEqual(r["Nt"], int((x > t).sum()))
        self.assertAlmostEqual(r["r"], 329 * 0.1 / r["Nt"], places=12)
        if r["gamma"] != 0:
            self.assertAlmostEqual(r["threshold"], t + (r["sigma"] / r["gamma"]) * (r["r"] ** (-r["gamma"]) - 1), places=9)
        else:
            self.assertAlmostEqual(r["threshold"], t - r["sigma"] * np.log(r["r"]), places=9)
        self.assertEqual(float(self.get_threshold(x, au.Q, au.LEVEL)), r["threshold"])

    def test_with_level_095_and_q_01_the_ratio_is_about_two_for_any_n(self):
        rng = np.random.default_rng(1)
        for n in (329, 1000, 2269):
            r = au.spot_fit(self.SPOT, rng.gamma(2.0, 3.0, size=n))
            self.assertAlmostEqual(r["r"], 2.0, delta=0.15)                          # so the 'extreme quantile' lies below the cut t

    def test_armor_fallbacks(self):
        const = np.full(50, 3.5)
        self.assertEqual(float(self.get_threshold(const, 0.1, 0.95)), 3.5)             # constant series -> that value
        x = np.zeros(20)
        x[-1] = 1.0                                                                     # no peak above t -> exception inside SPOT -> percentile fallback
        self.assertIn("error", au.spot_fit(self.SPOT, x))
        self.assertEqual(float(self.get_threshold(x, 0.1, 0.95)), float(sorted(x)[int(20 * 0.95)]))

    def test_source_hashes_are_the_pinned_ones(self):
        self.assertEqual(self.prov["anomaly_detector.py_sha256"], "5166023fd39aad6deb72ff528d2009631fae282373972adbec11adcda32414e3")
        self.assertEqual(self.prov["spot.py_sha256"], "164cdad8b4f56fe8155d9b5cd4bfaf8f1495491a0f58497baf407e718fb52cc5")

    def test_dependence_helpers(self):
        rng = np.random.default_rng(2)
        iid = rng.normal(size=400)
        ar = np.zeros(400)
        for i in range(1, 400):
            ar[i] = 0.8 * ar[i - 1] + rng.normal()
        self.assertGreater(au.effective_sample_size(iid), 0.7 * 400)
        self.assertLess(au.effective_sample_size(ar), 0.4 * 400)
        self.assertEqual(len(au.autocorrelation(iid)), 7)


@unittest.skipUnless(os.path.isfile(AUDIT), "the audit result file is part of the repository")
class SavedAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(AUDIT) as fh:
            cls.r = json.load(fh)

    def test_five_seeds_on_329_calibration_windows_with_the_documented_findings(self):
        self.assertEqual(sorted(self.r["seeds"]), ["0", "1", "2", "3", "4"])
        for s, d in self.r["seeds"].items():
            sp = d["spot_on_real_scores"]
            self.assertEqual((d["n_windows"], sp["n"], sp["Nt"]), (329, 329, 16))
            self.assertAlmostEqual(sp["r"], 32.9 / 16, places=9)
            self.assertEqual(sp["gamma"], 0.0)
            self.assertLess(sp["threshold"], d["empirical_percentiles"]["90"])
            self.assertGreater(d["calibration_windows_above_spot_threshold"], 0.2)
            self.assertTrue(120 < d["effective_sample_size"] < 180)
            for k in ("stability_block7", "stability_independent"):
                self.assertEqual(d[k]["fits_raising_an_exception"], 0)
                self.assertEqual(d[k]["gamma_exactly_zero_share"], 1.0)
                self.assertGreater(d[k]["threshold_iqr_over_median"], 0.6)
            self.assertLess(d["stability_block7"]["threshold_p2.5_p97.5"][0], 0)

    @NEEDS_ARMOR
    def test_provenance_matches_the_released_armor_source(self):
        _, _, prov = au.load_armor_spot()
        self.assertEqual(self.r["provenance"], prov)

    def test_the_audit_reads_calibration_windows_only(self):
        with open(au.__file__) as fh:
            body = fh.read().split('"""', 2)[2]
        self.assertIn('"s3_calibration_reference"', body)
        self.assertNotIn("validation_all", body)
        self.assertIsNone(re.search(r"training_loader|training_dataset|\.backward\(|\.step\(|ad_cases|\.pkl", body))
        self.assertIn('allowed_roles=("calibration",)', body)


@NEEDS_ARMOR
class ArmorMissingScenarioTests(unittest.TestCase):
    """ARMOR's six missing-data scenarios, rebuilt from its source, map onto our schema's modality channels."""

    def armor_function(self, path, name, ns):
        with open(path) as fh:
            src = fh.read()
        seg = next(ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(seg, path + "::" + name, "exec"), ns)                           # noqa: S102
        return ns[name]

    def test_six_drop_scenarios_plus_full_and_the_channel_mapping(self):
        arm = os.path.join(rd.REPO, "armor-system", "data", "raw", "armor_zenodo_21744694", "ARMOR")
        build = self.armor_function(os.path.join(arm, "main.py"), "build_missing_cases", {"combinations": itertools.combinations})
        mask_idx = self.armor_function(os.path.join(arm, "utils", "public_functions.py"), "get_channel_indices_to_mask", {})
        with open(os.path.join(arm, "config", "D1.yaml")) as fh:
            txt = fh.read()
        span = {k: ast.literal_eval(re.search(rf"^\s+{k}:\s*(\[.*\])\s*$", txt, flags=re.M).group(1)) for k in ("metric", "trace", "log")}
        cases = build(span, include_full=True, include_all_missing=False)
        self.assertEqual(len(cases), 7)
        self.assertEqual(cases[0], [])
        self.assertEqual({tuple(sorted(c)) for c in cases[1:]}, {("log",), ("metric",), ("trace",), ("log", "metric"), ("log", "trace"), ("metric", "trace")})
        self.assertNotIn(("log", "metric", "trace"), {tuple(sorted(c)) for c in cases})            # all three missing is excluded
        schema_path = os.path.join(rd.REPO, "armor-system", "data", "artifacts", "d1", "inspection", "schema.json")
        with open(schema_path) as fh:
            ranges = json.load(fh)["modality_index_ranges"]
        for m in ("metric", "trace", "log"):
            self.assertEqual(mask_idx(span, [m]), sorted(ranges[m]))                       # ARMOR's channel set == our schema's modality channels
        self.assertEqual((len(ranges["metric"]), len(ranges["trace"]), len(ranges["log"])), (110, 9, 11))
        self.assertEqual(mask_idx(span, ["metric", "log"]), sorted(ranges["metric"] + ranges["log"]))


class ProtocolFactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(PROTOCOL) as fh:
            cls.doc = fh.read()

    def test_parameter_counts_in_the_protocol_match_the_code(self):
        spec, _, _ = MH.synthetic_spec_and_applicable()
        m = Model1.build(spec, seed=0)
        base = m.n_parameters()
        head = (128 * 64 + 64) + (64 * 1 + 1)
        self.assertEqual((base, head, base + head), (168200, 8321, 176521))
        for s in ("168,200", "8,321", "176,521", "151,169", "8,581", "8,450"):
            self.assertIn(s, self.doc)
        pc = lambda mod: sum(p.numel() for p in mod.parameters())          # noqa: E731
        self.assertEqual((pc(m.input_adapter) + pc(m.core), pc(m.fault_head), pc(m.next_step_head)), (151169, 8581, 8450))

    def test_predeclaration_decisions_and_extensions_are_recorded_and_distinguished(self):
        self.assertIn("Model 1 v0 uses no simulated-missing training and is reported as the ablation. Model 1 v1 uses simulated-missing training and is the predeclared final model for held-out test evaluation, regardless of validation performance.", self.doc)
        for s in ("Explicit v1 extension", "amendment A-2", "extension A-3", "extension A-4", "binary detection head", "0.20", "q = 0.1", "level = 0.95", "D1:", "D2:", "D3:", "D4:",
                  "approved by the project lead", "B1 (resolved)", "empirical 90th percentile", "Option (c)", "rejected", "7-window moving-block bootstrap", "paired batch", "L_fault + L_det",
                  "Targets are never hidden", "Philox", "causal", "Dropping `metric` hides all six VMs", "clean thresholds", "two rows"):
            self.assertIn(s.lower(), self.doc.lower(), s)
        self.assertNotIn("[S]", self.doc)                                              # every former specified-by-audit item is now a confirmed decision
        self.assertNotIn("provisional", self.doc.lower().replace("not provisional", ""))

    def test_the_ranks_quoted_for_the_90th_percentile_are_right(self):
        from model1_transformer import threshold_diagnostics as tdg
        r = tdg.percentile_ranks(329, 90.0)
        self.assertEqual((r["lower_rank_ascending"], r["upper_rank_ascending"], r["lower_rank_from_top"], r["upper_rank_from_top"]), (296, 297, 34, 33))
        self.assertAlmostEqual(r["weight_on_upper"], 0.2, places=9)
        self.assertIn("ascending ranks 296 and 297, 34th and 33rd from the top, weight 0.2", self.doc)

    def test_the_design_note_really_has_no_binary_head_spot_or_rolling_maximum(self):
        with open(os.path.join(MODELS, "MODEL1_V0_DESIGN.md")) as fh:
            d = fh.read().lower()
        for w in ("binary", "spot", "evt", "rolling", "supervised detection"):
            self.assertNotIn(w, d, w)
        self.assertIn("simulated missing", d)

    def test_no_checkpoint_files_live_inside_the_repository(self):
        root = os.path.abspath(os.path.join(MODELS, "..", ".."))
        found = [os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs if f.endswith((".pt", ".pth", ".ckpt"))]
        self.assertEqual(found, [])                                                    # checkpoints stay outside Git


if __name__ == "__main__":
    unittest.main()
