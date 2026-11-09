import copy
import csv
import hashlib
import io
import json
import os
import unittest

import numpy as np

import _m1_helpers as H
from prep import m1_masks as mm, m1_split as ms, paths, preprocess as pp

CFG_FIELDS = ("tc", "embargo_s", "pre_guard_s", "post_guard_s")


def oracle_categories(ts, cases, cfg, table):
    """Independent, loop-based Policy B definition (slow and obvious) for every window in `table`."""
    out = []
    for s in table.start:
        T = ts[s:s + 7]
        t5 = T[5]
        ea = any(c - cfg.case_before_s <= t5 < c + cfg.case_after_s for c in cases["timestamp"])
        guard = any(any(c - cfg.case_before_s - cfg.pre_guard_s <= t < c + cfg.case_after_s + cfg.post_guard_s for t in T) for c in cases["timestamp"])
        out.append("event_associated" if ea else ("ignore" if guard else "normal_reference"))
    return np.array(out)


class SyntheticSplitTests(unittest.TestCase):
    def setUp(self):
        self.ts, self.cases, self.bounds, self.cfg = H.synthetic_timeline()
        self.table = ms.compute_split(self.ts, self.cases, self.bounds, self.cfg)

    def test_categories_match_an_independent_oracle(self):
        self.assertTrue((oracle_categories(self.ts, self.cases, self.cfg, self.table) == self.table.category).all())
        self.assertEqual(set(self.table.category), set(ms.CATEGORIES))

    def test_roles_follow_the_cutoff_rules(self):
        t, c = self.table, self.cfg
        fit, cal = t.role == "fit", t.role == "calibration"
        self.assertTrue((t.T[fit, -1] < c.tc - c.embargo_s).all())
        self.assertTrue((t.T[cal, 0] >= c.tc).all())
        self.assertTrue(fit.any() and cal.any())
        self.assertTrue(((t.role == "embargo") == ((t.period == "train") & ~fit & ~cal)).all())
        self.assertTrue((t.period[t.role == "validation"] == "validation").all())

    def test_no_window_crosses_a_boundary_and_leakage_checks_pass(self):
        self.assertTrue(ms.region_ok(self.table, self.bounds, self.cfg).all())
        checks = ms.leakage_checks(self.table, self.cases, self.bounds, self.cfg)
        self.assertTrue(all(c["ok"] for c in checks), checks)
        ms.assert_leakage_free(self.table, self.cases, self.bounds, self.cfg)

    def test_event_windows_carry_case_and_class(self):
        t = self.table
        ea = t.category == "event_associated"
        self.assertTrue((t.case_index[ea] >= 0).all())
        self.assertTrue((t.fault_class[ea] == self.cases["class_id"][t.case_index[ea]]).all())
        self.assertTrue((t.case_index[~ea] == -1).all() and (t.fault_class[~ea] == -1).all())

    def test_two_class_overlap_is_ambiguous(self):
        cases = copy.deepcopy(self.cases)
        cases["timestamp"] = cases["timestamp"].copy()
        cases["timestamp"][1] = cases["timestamp"][0] + 100       # two train cases 100 s apart, different classes
        t = ms.compute_split(self.ts, cases, self.bounds, self.cfg)
        self.assertTrue(t.ambiguous.any())
        a = t.ambiguous
        self.assertTrue((t.fault_class[a] == -1).all() and (t.case_index[a] == -1).all())
        self.assertTrue((t.category[a] == "event_associated").all())

    def test_embargo_removes_windows_between_regions(self):
        cfg = ms.SplitConfig(tc=self.cfg.tc, embargo_s=3000)
        t = ms.compute_split(self.ts, self.cases, self.bounds, cfg)
        self.assertGreater(int((t.role == "embargo").sum()), 0)
        self.assertEqual(set(np.intersect1d(t.T[t.role == "fit"], t.T[t.role == "calibration"])), set())


class ExclusionReasonTests(unittest.TestCase):
    def setUp(self):
        self.ts, self.cases, self.bounds, self.cfg = H.synthetic_timeline()
        cases = copy.deepcopy(self.cases)
        cases["timestamp"] = cases["timestamp"].copy()
        cases["timestamp"][1] = cases["timestamp"][0] + 100       # train cases 0 and 1: 100 s apart, different classes
        self.cases = cases
        self.table = ms.compute_split(self.ts, cases, self.bounds, self.cfg)

    def test_reason_is_set_exactly_for_overlapping_different_class_windows(self):
        t = self.table
        has = np.array([r is not None for r in t.exclusion_reason])
        self.assertTrue(t.ambiguous.any())
        self.assertTrue((has == t.ambiguous).all())
        self.assertTrue(all(r == "overlapping_cases_different_classes" for r in t.exclusion_reason[has]))
        self.assertEqual(ms.REASON_OVERLAP, "overlapping_cases_different_classes")

    def test_every_unlabelled_event_window_has_a_reason(self):
        t = self.table
        ea_unlabelled = (t.category == "event_associated") & (t.fault_class == -1)
        self.assertTrue(ea_unlabelled.any())
        self.assertTrue(all(r is not None for r in t.exclusion_reason[ea_unlabelled]))
        labelled = (t.category == "event_associated") & (t.fault_class >= 0)
        self.assertTrue(all(r is None for r in t.exclusion_reason[labelled]))
        self.assertTrue(all(r is None for r in t.exclusion_reason[t.category != "event_associated"]))

    def test_same_class_overlap_is_labelled_and_has_no_reason(self):
        cases = copy.deepcopy(self.cases)
        cases["class_id"] = np.zeros(8, np.int64)
        t = ms.compute_split(self.ts, cases, self.bounds, self.cfg)
        self.assertFalse(t.ambiguous.any())
        self.assertTrue(all(r is None for r in t.exclusion_reason))

    def test_missing_or_inconsistent_reasons_fail_the_invariants(self):
        def failing(mutate):
            t = ms.compute_split(self.ts, self.cases, self.bounds, self.cfg)
            t.exclusion_reason = t.exclusion_reason.copy()
            mutate(t)
            return {c["id"] for c in ms.leakage_checks(t, self.cases, self.bounds, self.cfg) if not c["ok"]}, t

        amb = int(np.nonzero(self.table.ambiguous)[0][0])
        bad, t = failing(lambda t: t.exclusion_reason.__setitem__(amb, None))
        self.assertIn("L9_every_unlabelled_event_window_has_an_exclusion_reason", bad)
        with self.assertRaises(ms.ManifestInvariantError):
            ms.assert_leakage_free(t, self.cases, self.bounds, self.cfg)
        other = int(np.nonzero(self.table.category == "normal_reference")[0][0])
        bad, _ = failing(lambda t: t.exclusion_reason.__setitem__(other, ms.REASON_OVERLAP))
        self.assertIn("L10_exclusion_reasons_are_documented_and_consistent", bad)
        bad, _ = failing(lambda t: t.exclusion_reason.__setitem__(amb, "made_up_reason"))
        self.assertIn("L10_exclusion_reasons_are_documented_and_consistent", bad)

    def test_counts_are_unchanged_by_the_reason(self):
        t = self.table
        self.assertEqual(int(((t.category == "event_associated") & t.ambiguous).sum()), int(t.ambiguous.sum()))


class LeakageDetectionTests(unittest.TestCase):
    """The invariants must fail loudly on a bad cutoff, not pass vacuously."""

    def setUp(self):
        self.ts, self.cases, self.bounds, self.cfg = H.synthetic_timeline()

    def failing(self, cases, cfg, table=None):
        table = table or ms.compute_split(self.ts, cases, self.bounds, cfg)
        return {c["id"] for c in ms.leakage_checks(table, cases, self.bounds, cfg) if not c["ok"]}, table

    def test_negative_embargo_shares_timestamps(self):
        bad, table = self.failing(self.cases, ms.SplitConfig(tc=self.cfg.tc, embargo_s=-600))
        self.assertIn("L1_no_shared_timestamp_fit_calibration", bad)
        with self.assertRaises(ms.ManifestInvariantError):
            ms.assert_leakage_free(table, self.cases, self.bounds, ms.SplitConfig(tc=self.cfg.tc, embargo_s=-600))

    def test_cutoff_inside_one_incident_group_fails(self):
        cases = copy.deepcopy(self.cases)
        cases["group_id"] = cases["group_id"].copy()
        cases["group_id"][2] = cases["group_id"][3] = 99           # the cases on both sides of Tc share a group
        bad, _ = self.failing(cases, self.cfg)
        self.assertIn("L3_no_incident_group_on_both_sides", bad)
        self.assertIn("L7_cutoff_lies_between_cases_of_different_groups", bad)

    def test_train_case_without_event_windows_fails_coverage(self):
        cases = copy.deepcopy(self.cases)
        cases["timestamp"] = cases["timestamp"].copy()
        cases["timestamp"][0] = self.ts[0] - 10 ** 6               # an extra case far from every sample
        bad, _ = self.failing(cases, self.cfg)
        self.assertIn("L4_every_train_case_on_exactly_one_side", bad)

    def test_corrupted_role_is_caught_by_region_check(self):
        table = ms.compute_split(self.ts, self.cases, self.bounds, self.cfg)
        table.role = table.role.copy()
        first_fit = int(np.nonzero(table.role == "fit")[0][0])
        table.role[first_fit] = "calibration"                      # a fit-time window mislabelled calibration
        bad, _ = self.failing(self.cases, self.cfg, table)
        self.assertIn("L5_no_window_crosses_a_region_boundary", bad)

    def test_validation_window_labelled_fit_is_caught(self):
        table = ms.compute_split(self.ts, self.cases, self.bounds, self.cfg)
        table.role = table.role.copy()
        table.role[int(np.nonzero(table.role == "validation")[0][0])] = "fit"
        bad, _ = self.failing(self.cases, self.cfg, table)
        self.assertTrue({"L5_no_window_crosses_a_region_boundary", "L6_roles_stay_inside_their_outer_period"} & bad)

    def test_window_bridging_a_gap_is_never_created(self):
        ts = self.ts.copy()
        ts[2000:] += 600                                           # a 10-minute gap inside the validation period
        table = ms.compute_split(ts, self.cases, self.bounds, self.cfg)
        self.assertTrue((np.diff(table.T, axis=1) == 60).all())


class ApprovedCountsTests(unittest.TestCase):
    def fake(self):
        return copy.deepcopy({**{k: v for k, v in ms.APPROVED_COUNTS.items()}})

    def test_identical_counts_pass(self):
        s = self.fake()
        self.assertEqual(ms.check_approved_counts(s), [])
        ms.assert_approved_counts(s)

    def test_any_difference_raises(self):
        for mutate in (lambda s: s["roles"]["fit"].update(windows=2869),
                       lambda s: s["roles"]["calibration"].update(cases=14),
                       lambda s: s["roles"]["validation"]["cases_by_class"].__setitem__(0, 9),
                       lambda s: s.update(fit_samples=2881),
                       lambda s: s["period_windows"].update(test=1)):
            s = self.fake()
            mutate(s)
            with self.assertRaises(ms.ApprovedCountsError):
                ms.assert_approved_counts(s)


class LabelAndCaseTableTests(unittest.TestCase):
    def test_raw_ids_map_consecutively_to_0_4(self):
        self.assertEqual(ms.raw_to_model_map(H.synthetic_schema()), {0: 0, 1: 1, 3: 2, 4: 3, 5: 4})

    def test_bad_category_map_is_rejected(self):
        s = H.synthetic_schema()
        s["category_ids"]["5"] = "Not A Class"
        with self.assertRaises(ValueError):
            ms.raw_to_model_map(s)
        s = H.synthetic_schema()
        s["category_ids"].pop("5")
        with self.assertRaises(ms.ManifestInvariantError):
            ms.raw_to_model_map(s)

    def rows(self):
        return [{"case_index": str(i), "timestamp": str(1000 + i * 1000), "category": "Node CPU", "group_id": str(i), "model1_split": "train"} for i in range(3)]

    def test_case_table_validation(self):
        t = ms.load_case_table(self.rows())
        self.assertEqual(t["class_id"].tolist(), [2, 2, 2])
        r = self.rows()
        r[1]["timestamp"] = "10"
        with self.assertRaises(ms.ManifestInvariantError):
            ms.load_case_table(r)
        r = self.rows()
        r[0]["category"] = "Bogus"
        with self.assertRaises(ms.ManifestInvariantError):
            ms.load_case_table(r)
        r = self.rows()
        r[2]["case_index"] = "7"
        with self.assertRaises(ms.ManifestInvariantError):
            ms.load_case_table(r)


class FitOnlyScalerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.n = 80
        self.ts = H.T_BASE + 60 * np.arange(self.n, dtype=np.int64)
        self.X = rng.normal(size=(self.n, 46, 130)).astype(np.float32)
        self.X[:, :6, 53:73] = 0                                   # VM trace/log blocks absent
        self.groups = H.synthetic_schema()["modality_index_ranges"]
        self.cfg = ms.SplitConfig(tc=int(self.ts[60]), embargo_s=360)       # fit samples: ts < ts[60] - 360 -> first 54

    def test_fit_set_is_exactly_the_fitting_samples(self):
        a = ms.fit_scaler_fit_only(self.X, self.ts, self.groups, self.cfg)
        self.assertEqual(int(a["n_fit_samples"]), 54)
        self.assertTrue((a["fit_timestamps"] == self.ts[:54]).all())
        ms.check_scaler_provenance(a, self.ts, self.cfg)

    def test_calibration_observations_do_not_influence_any_parameter(self):
        a = ms.fit_scaler_fit_only(self.X, self.ts, self.groups, self.cfg)
        X2 = self.X.copy()
        X2[54:] = np.random.default_rng(9).normal(size=X2[54:].shape) * 100     # embargo + calibration + later rows
        b = ms.fit_scaler_fit_only(X2, self.ts, self.groups, self.cfg)
        for k in ("mean", "scale", "constant", "absent_block", "absent_entry", "fit_timestamps"):
            self.assertTrue((a[k] == b[k]).all(), k)

    def test_scaler_arrays_are_finite_and_positive(self):
        a = ms.fit_scaler_fit_only(self.X, self.ts, self.groups, self.cfg)
        self.assertTrue(np.isfinite(a["mean"]).all() and np.isfinite(a["scale"]).all())
        self.assertTrue((a["scale"] > 0).all())
        Xt, _ = pp.transform(self.X, ms.scaler_from_arrays(a), self.groups)
        self.assertTrue(np.isfinite(Xt).all())

    def test_tampered_provenance_is_rejected(self):
        a = ms.fit_scaler_fit_only(self.X, self.ts, self.groups, self.cfg)
        bad = dict(a)
        bad["fit_timestamps"] = self.ts[:55]                        # includes a post-cutoff sample
        with self.assertRaises(ms.ManifestInvariantError):
            ms.check_scaler_provenance(bad, self.ts, self.cfg)
        bad = dict(a)
        bad["fit_timestamps"] = np.r_[a["fit_timestamps"][:-1], self.ts[70]]
        with self.assertRaises(ms.ManifestInvariantError):
            ms.check_scaler_provenance(bad, self.ts, self.cfg)
        bad = dict(a)
        bad["mean"] = a["mean"].copy()
        bad["mean"][0, 0] = np.nan
        with self.assertRaises(ms.ManifestInvariantError):
            ms.check_scaler_provenance(bad, self.ts, self.cfg)
        bad = dict(a)
        bad["scale"] = a["scale"].copy()
        bad["scale"][0, 0] = 0.0
        with self.assertRaises(ms.ManifestInvariantError):
            ms.check_scaler_provenance(bad, self.ts, self.cfg)


class DeterministicFileTests(unittest.TestCase):
    def test_npz_bytes_are_identical_across_calls_and_loadable(self):
        arrays = {"b": np.arange(6, dtype=np.float32).reshape(2, 3), "a": np.array(7, np.int64), "c": np.array([True, False])}
        b1, b2 = ms.npz_bytes(arrays), ms.npz_bytes(dict(reversed(list(arrays.items()))))
        self.assertEqual(b1, b2)
        z = np.load(io.BytesIO(b1))
        self.assertEqual(sorted(z.files), ["a", "b", "c"])
        self.assertEqual(int(z["a"]), 7)
        self.assertEqual(z["a"].ndim, 0)
        self.assertTrue((z["b"] == arrays["b"]).all())

    def test_manifest_text_is_stable_and_round_trips(self):
        obj = {"x": list(range(100)), "y": {"z": [1, 2, 3], "w": "t"}, "e": []}
        t = ms.dumps_manifest(obj)
        self.assertEqual(t, ms.dumps_manifest(copy.deepcopy(obj)))
        self.assertEqual(json.loads(t), obj)
        self.assertTrue(max(len(line) for line in t.splitlines()) < 120)


# ---------------------------------------------------------------------------------------------------------
# committed artifacts (no feature cache needed)
# ---------------------------------------------------------------------------------------------------------
MANIFEST_PATH = os.path.join(paths.ART, "splits", f"{ms.MANIFEST_NAME}.json")
SCALER_PATH = os.path.join(paths.ART, "preprocessing", f"{ms.SCALER_NAME}.npz")


@H.needs(H.ARTIFACTS_PRESENT, "M1-03 artifacts not generated (data/model1_pipeline.py)")
class CommittedManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = H.read_json(MANIFEST_PATH)
        cls.cases = ms.load_case_table(H.read_csv(os.path.join(paths.ART, "splits", "case_split_assignments.csv")))

    def test_hash_status_and_approved_inputs(self):
        m = self.m
        self.assertEqual(m["content_sha256"], ms.manifest_content_hash(m))
        self.assertEqual(m["status"], "approved")
        self.assertEqual(m["inputs"]["outer_split_manifest"]["content_sha256"], ms.APPROVED_OUTER_SHA256)
        self.assertEqual(m["inputs"]["preprocessing_config_sha256"], ms.APPROVED_PREPROCESSING_CONFIG_SHA256)
        self.assertEqual((m["protocol"]["Tc"], m["protocol"]["embargo_s"]), (1651637521, 360))
        self.assertEqual((m["protocol"]["policy"]["pre_guard_s"], m["protocol"]["policy"]["post_guard_s"]), (120, 300))

    def test_counts_equal_the_approved_design(self):
        self.assertEqual(ms.check_approved_counts(self.m["counts"]), [])
        c = self.m["counts"]["roles"]
        self.assertEqual([c[r]["cases"] for r in ("fit", "calibration", "validation", "test")], [88, 13, 25, 84])
        self.assertEqual([c[r]["windows"] for r in ("fit", "calibration")], [2868, 559])

    def test_every_invariant_is_recorded_as_passing(self):
        self.assertTrue(self.m["invariants"] and all(c["ok"] for c in self.m["invariants"]))

    def test_window_lists_are_consistent_with_counts_and_disjoint(self):
        w = self.m["windows"]
        seen = {}
        for r, per in w.items():
            for c, starts in per["start_index"].items():
                self.assertEqual(len(starts), self.m["counts"]["roles"][r][c], (r, c))
                self.assertEqual(ms.starts_digest(starts), per["start_index_sha256"][c])
                self.assertEqual(starts, sorted(set(starts)))
                for s in starts:
                    self.assertNotIn(s, seen, f"window {s} listed twice")
                    seen[s] = (r, c)
        self.assertEqual(len(seen), 2868 + 559 + 869 + 2868)

    def test_policy_b_category_totals_for_the_train_period(self):
        w = self.m["windows"]
        tot = {c: len(w["fit"]["start_index"][c]) + len(w["calibration"]["start_index"][c]) for c in ms.CATEGORIES}
        self.assertEqual(tot, {"normal_reference": 1641, "event_associated": 602, "ignore": 1184})

    def test_case_accounting_88_13_25_84_and_group_isolation(self):
        by = self.m["cases"]["by_role"]
        self.assertEqual({r: len(v) for r, v in by.items()}, {"fit": 88, "calibration": 13, "validation": 25, "test": 84})
        sets = {r: set(v) for r, v in by.items()}
        for a in sets:
            for b in sets:
                if a < b:
                    self.assertFalse(sets[a] & sets[b], (a, b))
        self.assertEqual(set().union(*sets.values()), set(range(210)))
        g = self.m["cases"]["groups_by_role"]
        self.assertFalse(set(g["fit"]) & set(g["calibration"]), "an incident group crosses the fit/calibration boundary")
        for row in self.m["cases"]["assignments"]:
            self.assertEqual(row["fault_class"], int(self.cases["class_id"][row["case_index"]]))
            self.assertIn(row["fault_class"], range(5))
            if row["role"] in ("fit", "calibration"):
                self.assertEqual(row["outer_split"], "train")
                other = "calibration" if row["role"] == "fit" else "fit"
                self.assertEqual(row["event_windows_by_role"][other], 0)
                self.assertGreater(row["event_windows_by_role"][row["role"]], 0)

    def test_exclusion_reason_is_recorded_for_the_ambiguous_window_only(self):
        r = ms.REASON_OVERLAP
        self.assertEqual(self.m["windows"]["fit"]["fault_label_exclusions"], {r: [2550]})
        for role in ("calibration", "validation", "test"):
            self.assertEqual(self.m["windows"][role]["fault_label_exclusions"], {r: []}, role)
        self.assertEqual(self.m["counts"]["fault_label_exclusions"], {r: 1})
        self.assertEqual(self.m["ambiguous_windows"][0]["fault_label_exclusion_reason"], r)
        self.assertEqual(self.m["protocol"]["fault_label_exclusion_reasons"], [r])
        self.assertEqual(self.m["counts"]["roles"]["fit"]["event_associated"], 524)      # category counts unchanged

    def test_ambiguous_window_is_recorded_and_excluded_from_the_fault_objective(self):
        amb = self.m["ambiguous_windows"]
        self.assertEqual(len(amb), 1)
        self.assertEqual(amb[0]["case_indices"], [78, 79])
        self.assertIn(amb[0]["start_index"], self.m["windows"]["fit"]["start_index"]["event_associated"])

    def test_objective_table_never_lets_a_gradient_objective_see_non_fit_data(self):
        for name, o in self.m["objectives"].items():
            if o["gradient"]:
                self.assertEqual(o["roles"], ["fit"], name)
            self.assertNotIn("embargo", o["roles"])
        for name, o in ms.OBJECTIVES.items():
            if o["gradient"]:
                self.assertEqual(o["roles"], ("fit",))
                self.assertNotIn("ignore", o["categories"], name)
        self.assertEqual(ms.OBJECTIVES["s1_pretrain"]["categories"], ("normal_reference",))
        self.assertEqual(ms.OBJECTIVES["s2_fault_finetune"]["categories"], ("event_associated",))


@H.needs(H.ARTIFACTS_PRESENT, "M1-03 artifacts not generated (data/model1_pipeline.py)")
class CommittedScalerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = H.read_json(MANIFEST_PATH)
        z = np.load(SCALER_PATH)
        cls.s = {k: z[k] for k in z.files}

    def test_file_hash_matches_manifest(self):
        self.assertEqual(hashlib.sha256(H.read_bytes(SCALER_PATH)).hexdigest(), self.m["fit_only_scaler"]["file_sha256"])
        for k, h in self.m["fit_only_scaler"]["array_sha256"].items():
            self.assertEqual(ms.array_digest(self.s[k]), h, k)

    def test_fit_samples_are_before_the_embargoed_cutoff_only(self):
        ft = self.s["fit_timestamps"]
        self.assertEqual(len(ft), 2880)
        self.assertEqual(int(self.s["n_fit_samples"]), 2880)
        self.assertLess(int(ft.max()), 1651637521 - 360)
        self.assertFalse(((ft >= 1651637521 - 360)).any())
        self.assertTrue((np.diff(ft) > 0).all())

    def test_finite_positive_and_layer1_identical_to_the_existing_scaler(self):
        self.assertTrue(np.isfinite(self.s["mean"]).all() and np.isfinite(self.s["scale"]).all())
        self.assertTrue((self.s["scale"] > 0).all())
        old = np.load(os.path.join(paths.ART, "preprocessing", "scaler_model1_train_period_v1.npz"))
        self.assertTrue((old["absent_block"] == self.s["absent_block"]).all())
        self.assertTrue((old["absent_entry"] == self.s["absent_entry"]).all())
        self.assertEqual((int(old["constant"].sum()), int(self.s["constant"].sum())), (4206, 4209))

    def test_declared_mask_gives_3152_with_the_committed_scaler(self):
        schema = H.read_json(os.path.join(paths.ART, "inspection", "schema.json"))
        app, _, rep = mm.applicable_entry_mask(schema, self.s["absent_entry"])
        mm.check_expected_counts(rep)
        self.assertEqual(int(app.sum()), 3152)


@H.needs(H.ARTIFACTS_PRESENT and H.CACHE_PRESENT, "needs the D1 feature cache and the M1-03 artifacts")
class RegenerationTests(unittest.TestCase):
    def test_fresh_generation_is_byte_identical_to_the_committed_artifacts(self):
        import sys
        sys.path.insert(0, paths.ROOT + "/data")
        import model1_pipeline
        text, sbytes, m = model1_pipeline.generate()
        self.assertEqual(text, H.read_text(MANIFEST_PATH))
        self.assertEqual(sbytes, H.read_bytes(SCALER_PATH))

    def test_manifest_windows_equal_windows_recomputed_from_the_cache(self):
        from prep import loader
        ts, _, _ = loader.load_arrays(mmap=True)
        m = H.read_json(MANIFEST_PATH)
        cases = ms.load_case_table(H.read_csv(os.path.join(paths.ART, "splits", "case_split_assignments.csv")))
        outer = H.read_json(os.path.join(paths.ART, "splits", "model1_split_101_25_84_v1.json"))
        table = ms.compute_split(ts, cases, ms.approved_boundaries(outer, cases))
        ms.verify_manifest_against_table(m, table)
        bad = copy.deepcopy(m)
        bad["windows"]["fit"]["start_index"]["normal_reference"].pop()
        with self.assertRaises(ms.ManifestInvariantError):
            ms.verify_manifest_against_table(bad, table)


if __name__ == "__main__":
    unittest.main()
