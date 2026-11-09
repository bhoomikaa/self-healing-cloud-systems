import copy
import types
import unittest

import numpy as np

import _m1_helpers as H
from prep import m1_case_features as cf, m1_masks as mm

try:
    from prep import m1_dataset as md
except ImportError:                                            # pragma: no cover
    md = None

NEEDS_REAL = H.needs(md is not None and H.CACHE_PRESENT and H.ARTIFACTS_PRESENT, "needs torch, the D1 feature cache and the M1-03 artifacts")


def schema_vm_last():
    """Synthetic schema whose 6 VM nodes are the LAST six nodes (as in real D1), with a consistent layer-1 mask."""
    s = H.synthetic_schema()
    absent = H.synthetic_absent_entry(s)
    order = list(range(6, 46)) + list(range(6))
    s = copy.deepcopy(s)
    s["node_order"] = [s["node_order"][i] for i in order]
    return s, absent[order]


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.schema, absent = schema_vm_last()
        self.app, self.node_type, _ = mm.applicable_entry_mask(self.schema, absent)
        self.layout = cf.group_layout(self.schema, self.app)

    def test_groups_come_from_node_names_not_position(self):
        (g0, n0, c0), (g1, n1, c1) = self.layout
        self.assertEqual((g0, g1), ("vm", "pod"))
        self.assertEqual(n0.tolist(), list(range(40, 46)))              # VMs are last here
        self.assertEqual(n1.tolist(), list(range(40)))
        self.assertTrue(all(self.schema["node_order"][i].startswith("node-") for i in n0))
        self.assertEqual((len(c0), len(c1)), (57, 73))

    def test_520_features_in_deterministic_order(self):
        names = cf.feature_names(self.schema, self.layout)
        self.assertEqual(len(names), 520)
        self.assertEqual(len(set(names)), 520)
        self.assertEqual(names, cf.feature_names(self.schema, cf.group_layout(self.schema, self.app)))
        self.assertTrue(names[0].startswith("vm|") and names[-1].startswith("pod|"))
        self.assertEqual([n.rsplit("|", 1)[1] for n in names[:4]], list(cf.STATS))
        self.assertTrue(all(n.split("|")[1].startswith("system.") for n in names if n.startswith("vm|")))
        self.assertFalse(any(n.split("|")[1].startswith("system.") for n in names if n.startswith("pod|")))

    def test_group_without_applicable_channels_fails_loudly(self):
        with self.assertRaises(mm.MaskError):
            cf.group_layout(self.schema, np.zeros_like(self.app))


class CaseVectorKnownAnswerTests(unittest.TestCase):
    def setUp(self):
        self.schema, absent = schema_vm_last()
        self.app, _, _ = mm.applicable_entry_mask(self.schema, absent)
        self.layout = cf.group_layout(self.schema, self.app)
        self.names = cf.feature_names(self.schema, self.layout)
        self.X = np.zeros((6, 46, 130), np.float32)

    def feat(self, group, channel_name, stat, vec):
        return float(vec[self.names.index(f"{group}|{channel_name}|{stat}")])

    def test_statistics_of_a_group_average(self):
        c = self.schema["channel_order"][80]["name"]                # a system.* channel: applicable to the 6 VMs
        self.X[:4, 40, 80] = [1.0, 2.0, 3.0, 4.0]                    # VM node-1 (index 40), minutes 0..3, then 0, 0
        v = cf.case_vector(self.X, np.arange(6), self.app, self.layout)
        avg = np.array([1, 2, 3, 4, 0, 0]) / 6.0                     # average over the 6 applicable VM nodes
        self.assertAlmostEqual(self.feat("vm", c, "mean", v), avg.mean())
        self.assertAlmostEqual(self.feat("vm", c, "std", v), avg.std())
        self.assertAlmostEqual(self.feat("vm", c, "max_abs", v), 4.0)  # not diluted by the group size
        self.assertAlmostEqual(self.feat("vm", c, "delta", v), avg[-1] - avg[0])

    def test_max_abs_uses_the_absolute_value(self):
        c = self.schema["channel_order"][80]["name"]
        self.X[2, 41, 80] = -7.5
        v = cf.case_vector(self.X, np.arange(6), self.app, self.layout)
        self.assertAlmostEqual(self.feat("vm", c, "max_abs", v), 7.5)

    def test_values_at_unavailable_entries_are_ignored(self):
        v0 = cf.case_vector(self.X, np.arange(6), self.app, self.layout)
        X = self.X.copy()
        X[:, 40, 5] = 99.0                                           # VM node on a pod-only (kpi_container) channel: not applicable
        X[:, 0, 90] = 99.0                                           # pod node on a system.* channel: not applicable
        X[:, 7, 70] = 99.0                                           # pod whose log block is absent (layer 1)
        self.assertTrue((cf.case_vector(X, np.arange(6), self.app, self.layout) == v0).all())

    def test_average_is_over_applicable_nodes_only(self):
        # log channel 70: 10 pods (indices 0..9 after reordering are pods 6..15 -> logs absent) are not applicable
        n_app = int(self.app[:40, 70].sum())
        self.assertEqual(n_app, 30)
        self.X[:, 20, 70] = 3.0                                      # a pod with logs
        c = self.schema["channel_order"][70]["name"]
        v = cf.case_vector(self.X, np.arange(6), self.app, self.layout)
        self.assertAlmostEqual(self.feat("pod", c, "mean", v), 3.0 / 30)

    def test_invalid_minute_index_fails(self):
        for bad in (np.array([3]), np.array([1, 1, 2]), np.array([2, 1, 0]), np.zeros((2, 2), int)):
            with self.assertRaises(ValueError):
                cf.case_vector(self.X, bad, self.app, self.layout)


class DeduplicationTests(unittest.TestCase):
    def test_overlapping_windows_count_each_minute_once(self):
        table = types.SimpleNamespace(start=np.array([10, 11, 12, 13, 14]))
        m = cf.case_minutes(table, [0, 1, 2, 3, 4])
        self.assertEqual(m.tolist(), list(range(10, 21)))            # 11 distinct minutes, not 5 x 7 = 35
        self.assertEqual(len(m), len(set(m.tolist())))
        self.assertEqual(cf.case_minutes(table, [0, 0, 1]).tolist(), list(range(10, 18)))   # repeated window rows add nothing

    def test_vector_does_not_depend_on_how_many_windows_cover_a_minute(self):
        rng = np.random.default_rng(0)
        schema, absent = schema_vm_last()
        app, _, _ = mm.applicable_entry_mask(schema, absent)
        layout = cf.group_layout(schema, app)
        X = np.where(app[None], rng.normal(size=(30, 46, 130)), 0).astype(np.float32)
        a = cf.case_vector(X, cf.case_minutes(types.SimpleNamespace(start=np.array([5, 6, 7])), [0, 1, 2]), app, layout)
        b = cf.case_vector(X, cf.case_minutes(types.SimpleNamespace(start=np.array([5, 6, 7, 7, 5])), [0, 1, 2, 3, 4]), app, layout)
        self.assertTrue((a == b).all())


class ClassWeightTests(unittest.TestCase):
    def test_inverse_frequency_from_labels_given(self):
        w = cf.inverse_frequency_weights(np.array([0, 0, 1, 2, 3, 4]))
        np.testing.assert_allclose(w, [6 / (5 * 2), 6 / 5, 6 / 5, 6 / 5, 6 / 5])

    def test_real_fit_distribution(self):
        w = cf.inverse_frequency_weights(np.repeat(np.arange(5), [33, 29, 9, 13, 4]))
        np.testing.assert_allclose(w, [88 / (5 * n) for n in (33, 29, 9, 13, 4)])

    def test_missing_class_is_an_error(self):
        with self.assertRaises(ValueError):
            cf.inverse_frequency_weights(np.array([0, 1, 2, 3]))


class RoleLockTests(unittest.TestCase):
    def test_only_fit_and_validation_are_allowed(self):
        for role in ("calibration", "test", "embargo", "anything"):
            with self.assertRaises(cf.RoleLocked):
                cf.build_case_matrix(None, role)                      # refused before any data is touched
        self.assertEqual(cf.ALLOWED_ROLES, ("fit", "validation"))


@NEEDS_REAL
class RealCaseMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = md.Model1Data.load()
        cls.fit = cf.build_case_matrix(cls.D, "fit")
        cls.val = cf.build_case_matrix(cls.D, "validation")
        cls.by_role = cls.D.manifest["cases"]["by_role"]

    def test_shapes_and_case_counts(self):
        self.assertEqual(self.fit["X"].shape, (88, 520))
        self.assertEqual(self.val["X"].shape, (25, 520))
        self.assertEqual((len(self.fit["y"]), len(self.val["y"])), (88, 25))
        self.assertEqual(self.fit["X"].dtype, np.float32)
        self.assertTrue(np.isfinite(self.fit["X"]).all() and np.isfinite(self.val["X"]).all())

    def test_exactly_one_row_per_case_in_deterministic_order(self):
        for m, role in ((self.fit, "fit"), (self.val, "validation")):
            ids = m["case_index"].tolist()
            self.assertEqual(ids, sorted(set(ids)))
            self.assertEqual(ids, self.by_role[role])
        again = cf.build_case_matrix(self.D, "fit")
        self.assertTrue((again["X"] == self.fit["X"]).all() and again["feature_names"] == self.fit["feature_names"])

    def test_calibration_and_held_out_cases_are_absent(self):
        used = set(self.fit["case_index"]) | set(self.val["case_index"])
        self.assertFalse(used & set(self.by_role["calibration"]))
        self.assertFalse(used & set(self.by_role["test"]))
        self.assertEqual(len(used), 113)
        self.assertEqual(len(self.by_role["calibration"]), 13)

    def test_labels_and_incident_groups(self):
        D = self.D
        for m in (self.fit, self.val):
            self.assertTrue((m["y"] == D.cases["class_id"][m["case_index"]]).all())
            self.assertTrue((m["group_id"] == D.cases["group_id"][m["case_index"]]).all())
            self.assertTrue(set(m["y"]) <= set(range(5)))
        self.assertEqual(np.bincount(self.fit["y"], minlength=5).tolist(), [33, 29, 9, 13, 4])
        self.assertEqual(np.bincount(self.val["y"], minlength=5).tolist(), [10, 6, 3, 5, 1])
        self.assertFalse(set(self.fit["group_id"]) & set(self.val["group_id"]))

    def test_windows_and_minutes_per_case(self):
        self.assertEqual(int(self.fit["n_windows"].sum()), 523)           # 524 event windows minus the ambiguous one
        self.assertEqual(int(self.val["n_windows"].sum()), 150)
        self.assertTrue(((self.fit["n_minutes"] >= 11) & (self.fit["n_minutes"] <= 12)).all())
        self.assertTrue((self.val["n_minutes"] == 12).all())

    def test_ambiguous_window_is_not_used(self):
        t = self.D.table
        rows = self.D.selection(["fit"], ["event_associated"], drop_ambiguous=True)
        self.assertNotIn(2550, set(int(s) for s in t.start[rows]))

    def test_node_groups_and_masks_on_real_schema(self):
        names = self.D.schema["node_order"]
        (g0, n0, c0), (g1, n1, c1) = self.fit["layout"]
        self.assertEqual((g0, len(n0), len(c0)), ("vm", 6, 57))
        self.assertEqual((g1, len(n1), len(c1)), ("pod", 40, 73))
        self.assertTrue(all(names[i].startswith("node-") for i in n0))
        self.assertFalse(any(names[i].startswith("node-") for i in n1))
        self.assertNotEqual(n0.tolist(), list(range(6)))                  # VMs are not the first six nodes

    def test_features_ignore_every_observation_outside_the_cases_own_minutes(self):
        """Perturb every sample that is not part of a fitting/validation case: the matrices must not change."""
        D = self.D
        t = D.table
        keep = set()
        for role in ("fit", "validation"):
            for r in D.selection([role], ["event_associated"], drop_ambiguous=True):
                keep.update(range(int(t.start[r]), int(t.start[r]) + 7))
        other = np.array([i for i in range(len(D.ts)) if i not in keep])
        Xt2 = D.Xt.copy()
        noise = np.random.default_rng(3).normal(size=Xt2[other].shape).astype(np.float32) * 50
        Xt2[other] = np.where(D.applicable[None], noise, 0)             # calibration, held-out and normal-reference samples all changed
        D2 = md.Model1Data(Xt2, D.ts, D.table, D.cases, D.manifest, D.scaler, D.applicable, (D.structural_absent, D.observed_all_zero), D.schema, None, D.cfg, D.boundaries)
        self.assertTrue((cf.build_case_matrix(D2, "fit")["X"] == self.fit["X"]).all())
        self.assertTrue((cf.build_case_matrix(D2, "validation")["X"] == self.val["X"]).all())

    def test_fit_matrix_does_not_depend_on_validation_samples(self):
        D = self.D
        t = D.table
        val_samples = sorted({i for r in D.selection(["validation"]) for i in range(int(t.start[r]), int(t.start[r]) + 7)})
        Xt2 = D.Xt.copy()
        Xt2[val_samples] = np.where(D.applicable[None], 5.0, 0)
        D2 = md.Model1Data(Xt2, D.ts, D.table, D.cases, D.manifest, D.scaler, D.applicable, (D.structural_absent, D.observed_all_zero), D.schema, None, D.cfg, D.boundaries)
        self.assertTrue((cf.build_case_matrix(D2, "fit")["X"] == self.fit["X"]).all())


if __name__ == "__main__":
    unittest.main()
