import copy
import unittest

import numpy as np

import _m1_helpers as H
from prep import m1_masks as mm


class DeclaredMaskTests(unittest.TestCase):
    def setUp(self):
        self.schema = H.synthetic_schema()
        self.absent = H.synthetic_absent_entry(self.schema)

    def test_counts_follow_the_approved_rule(self):
        app, node_type, rep = mm.applicable_entry_mask(self.schema, self.absent)
        self.assertEqual(rep["total"], 3152)
        self.assertEqual(rep["vm"], {"metric": 342})
        self.assertEqual(rep["pod"], {"metric": 2120, "trace": 360, "log": 330})
        self.assertEqual(node_type.tolist(), [0] * 6 + [1] * 40)
        mm.check_expected_counts(rep)

    def test_vm_uses_system_metrics_only_and_pod_uses_kpi_trace_log(self):
        app, _, _ = mm.applicable_entry_mask(self.schema, self.absent)
        names = [c["name"] for c in self.schema["channel_order"]]
        vm_cols = np.nonzero(app[0])[0]
        self.assertTrue(all(names[j].startswith("system.") for j in vm_cols))
        self.assertEqual(len(vm_cols), 57)
        pod_cols = np.nonzero(app[20])[0]                       # a pod with a log block
        self.assertFalse(any(names[j].startswith("system.") for j in pod_cols))
        self.assertEqual(len(pod_cols), 53 + 9 + 11)
        self.assertEqual(len(np.nonzero(app[6])[0]), 53 + 9)    # a pod whose log block is absent (layer 1)

    def test_mask_is_declared_not_inferred_from_values(self):
        """The mask is a function of names only: it cannot change with the data, so a zero never hides a channel."""
        a1, _, _ = mm.applicable_entry_mask(self.schema, self.absent)
        a2, _, _ = mm.applicable_entry_mask(copy.deepcopy(self.schema), self.absent.copy())
        self.assertTrue((a1 == a2).all())
        with self.assertRaises(TypeError):
            mm.applicable_entry_mask(self.schema, self.absent, X=np.zeros(3))      # no data argument exists

    def test_wrong_counts_fail_loudly(self):
        _, _, rep = mm.applicable_entry_mask(self.schema, self.absent)
        rep = copy.deepcopy(rep)
        rep["total"] += 1
        with self.assertRaises(mm.MaskError):
            mm.check_expected_counts(rep)
        rep2 = copy.deepcopy(rep)
        rep2["total"] -= 1
        rep2["pod"]["trace"] += 1
        with self.assertRaises(mm.MaskError):
            mm.check_expected_counts(rep2)


class InvalidSchemaTests(unittest.TestCase):
    def setUp(self):
        self.schema = H.synthetic_schema()
        self.absent = H.synthetic_absent_entry(self.schema)

    def expect(self, mutate, exc=mm.SchemaError):
        s = copy.deepcopy(self.schema)
        mutate(s)
        with self.assertRaises(exc):
            mm.declared_entry_mask(s)

    def test_missing_field(self):
        self.expect(lambda s: s.pop("node_order"))
        self.expect(lambda s: s.pop("modality_index_ranges"))

    def test_wrong_shape(self):
        self.expect(lambda s: s.update(tensor_shape_per_sample=[45, 130]))

    def test_duplicate_or_short_node_order(self):
        self.expect(lambda s: s["node_order"].__setitem__(1, s["node_order"][0]))
        self.expect(lambda s: s["node_order"].pop())

    def test_malformed_channel_entries(self):
        self.expect(lambda s: s["channel_order"][3].update(modality="bogus"))
        self.expect(lambda s: s["channel_order"][3].update(index=99))
        self.expect(lambda s: s["channel_order"][3].update(name=s["channel_order"][4]["name"]))
        self.expect(lambda s: s["channel_order"].pop())

    def test_modality_ranges_disagree_with_channels(self):
        self.expect(lambda s: s["modality_index_ranges"]["trace"].pop())
        self.expect(lambda s: s["modality_index_ranges"].pop("log"))

    def test_rule_that_selects_nothing_is_a_schema_mismatch(self):
        rule = copy.deepcopy(mm.D1_MASK_RULE)
        rule["node_types"][0]["modalities"]["metric"] = "no_such_prefix."
        with self.assertRaises(mm.MaskError):
            mm.declared_entry_mask(self.schema, rule)
        rule = copy.deepcopy(mm.D1_MASK_RULE)
        rule["node_types"].reverse()                     # default type first: ambiguous
        with self.assertRaises(mm.MaskError):
            mm.declared_entry_mask(self.schema, rule)

    def test_invalid_mask_arrays_fail(self):
        with self.assertRaises(mm.MaskError):
            mm.applicable_entry_mask(self.schema, self.absent.astype(np.uint8))
        with self.assertRaises(mm.MaskError):
            mm.applicable_entry_mask(self.schema, self.absent[:, :129])
        app, _, _ = mm.applicable_entry_mask(self.schema, self.absent)
        with self.assertRaises(mm.MaskError):
            mm.effective_mask(app, np.ones((6, 46, 130), np.uint8))
        with self.assertRaises(mm.MaskError):
            mm.effective_mask(app, np.ones((6, 46, 129), bool))
        with self.assertRaises(mm.MaskError):
            mm.mask_values(np.zeros((6, 46, 130)), np.ones((6, 46, 130), np.int8))
        with self.assertRaises(mm.MaskError):
            mm.mask_values(np.zeros((6, 46, 130)), np.ones((6, 46, 7), bool))


class UnavailableValuesTests(unittest.TestCase):
    def setUp(self):
        self.schema = H.synthetic_schema()
        self.app, _, _ = mm.applicable_entry_mask(self.schema, H.synthetic_absent_entry(self.schema))

    def test_unavailable_values_are_replaced_by_zero_before_model_use(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=(6, 46, 130)).astype(np.float32)
        v[:, ~self.app] = np.nan                                  # garbage (even NaN) where nothing is measured
        out = mm.mask_values(v, mm.effective_mask(self.app))
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue((out[:, ~self.app] == 0).all())
        self.assertTrue((out[:, self.app] == v[:, self.app]).all())

    def test_observed_false_removes_an_applicable_entry_but_zero_alone_does_not(self):
        v = np.ones((6, 46, 130), np.float32)
        obs = np.ones((6, 46, 130), bool)
        obs[2, 10, 5] = False
        eff = mm.effective_mask(self.app, obs)
        self.assertFalse(eff[2, 10, 5])
        self.assertTrue(eff[2, 10, 4])
        out = mm.mask_values(v, eff)
        self.assertEqual(out[2, 10, 5], 0)
        zeroed = v.copy()
        zeroed[:, 10, 4] = 0.0                                     # a real zero at an applicable entry stays available
        self.assertTrue(mm.effective_mask(self.app)[10, 4])
        self.assertTrue(mm.mask_values(zeroed, mm.effective_mask(self.app))[0, 10, 4] == 0)

    def test_effective_mask_without_observed_is_the_applicable_mask(self):
        self.assertTrue((mm.effective_mask(self.app) == self.app).all())


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.schema = H.synthetic_schema()
        self.app, _, _ = mm.applicable_entry_mask(self.schema, H.synthetic_absent_entry(self.schema))

    def test_clean_data_passes_every_period(self):
        X = np.zeros((10, 46, 130), np.float32)
        X[:, self.app] = 1.0
        rows = {"a": np.arange(10) < 5, "b": np.arange(10) >= 5}
        self.assertEqual(mm.check_mask_stability(X, self.app, rows, self.schema), {"a": 0, "b": 0})

    def test_violation_in_any_period_raises_with_entry_name(self):
        X = np.zeros((10, 46, 130), np.float32)
        X[7, 0, 3] = 0.5                                           # VM node-1, a kpi_container channel
        rows = {"fit": np.arange(10) < 5, "test": np.arange(10) >= 5}
        with self.assertRaises(mm.MaskError) as cm:
            mm.check_mask_stability(X, self.app, rows, self.schema)
        self.assertIn("test", str(cm.exception))
        self.assertIn("node-1", str(cm.exception))
        self.assertIn("kpi_container_m3", str(cm.exception))

    def test_empty_period_is_not_a_pass(self):
        X = np.zeros((10, 46, 130), np.float32)
        with self.assertRaises(mm.MaskError):
            mm.check_mask_stability(X, self.app, {"empty": np.zeros(10, bool)})

    def test_shape_mismatch_raises(self):
        with self.assertRaises(mm.MaskError):
            mm.check_mask_stability(np.zeros((3, 46, 129)), self.app)


@H.needs(H.CACHE_PRESENT, "D1 feature cache not built (pipeline.py --stage cache)")
class RealDataStabilityTests(unittest.TestCase):
    def test_mi6_zero_violations_in_every_period_and_3152_applicable(self):
        import json
        import os
        from prep import loader, paths
        schema = H.read_json(os.path.join(paths.ART, "inspection", "schema.json"))
        sc = np.load(os.path.join(paths.ART, "preprocessing", "scaler_model1_train_period_v1.npz"))
        app, _, rep = mm.applicable_entry_mask(schema, sc["absent_entry"])
        mm.check_expected_counts(rep)
        self.assertEqual(rep["total"], 3152)
        ts, X, _ = loader.load_arrays()
        out = mm.check_mask_stability(X, app, {"train": ts < 1651713858, "validation": (ts >= 1651713858) & (ts < 1651852800), "test": ts >= 1651852800}, schema)
        self.assertEqual(out, {"train": 0, "validation": 0, "test": 0})


if __name__ == "__main__":
    unittest.main()
