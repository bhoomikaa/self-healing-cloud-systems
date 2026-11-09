import unittest

import numpy as np

import _paths  # noqa: F401
from prep import detection_calibration as dc


class KnownAnswerTests(unittest.TestCase):
    """Hand-computed answers."""

    def setUp(self):
        # 5 windows x 3 nodes. node0: 1..5 (median 3, IQR 2); node1: 10..50 (median 30, IQR 20); node2: constant 7 (IQR 0)
        self.E = np.array([[1, 10, 7], [2, 20, 7], [3, 30, 7], [4, 40, 7], [5, 50, 7]], float)
        self.st = dc.fit_statistics(self.E)

    def test_median_iqr_epsilon_and_floor_report(self):
        np.testing.assert_allclose(self.st["median"], [3, 30, 7])
        np.testing.assert_allclose(self.st["iqr"], [2, 20, 0])
        self.assertAlmostEqual(self.st["epsilon"], 0.01 * 2.0)                     # 0.01 * median([2, 20, 0])
        self.assertEqual(self.st["floor_affected_nodes"], [{"node": 2, "iqr": 0.0}])
        self.assertEqual(self.st["n_windows"], 5)

    def test_normalisation_and_the_three_scores(self):
        new = np.array([[5, 50, 7], [3, 30, 7.1]], float)
        z = dc.normalise(new, self.st)
        np.testing.assert_allclose(z[0], [1, 1, 0], atol=1e-12)
        np.testing.assert_allclose(z[1], [0, 0, 0.1 / 0.02], atol=1e-9)           # a tiny change on a constant node is amplified by 1/epsilon
        sc = dc.window_scores(new, self.st)
        np.testing.assert_allclose(sc["m1_raw_max"], [50, 30])
        np.testing.assert_allclose(sc["m2_max_normalised"], [1, 5], atol=1e-9)
        np.testing.assert_allclose(sc["m3_topk_mean_normalised"], [2 / 3, 5 / 3], atol=1e-9)

    def test_floor_prevents_division_by_zero(self):
        z = dc.normalise(np.array([[3.0, 30.0, 7.0 + 1e-9]]), self.st)
        self.assertTrue(np.isfinite(z).all())

    def test_percentile_uses_linear_interpolation(self):
        scores = {m: np.array([1.0, 2, 3, 4, 100]) for m in dc.METHODS}
        tau = dc.thresholds(scores, 99.0)
        self.assertAlmostEqual(tau["m1_raw_max"], 4 + 0.96 * 96)                   # rank 0.99*(5-1) = 3.96 -> 96.16
        self.assertAlmostEqual(dc.thresholds(scores, 50.0)["m2_max_normalised"], 3.0)

    def test_single_node_model(self):
        e = np.arange(101, dtype=float)[:, None]                                    # one node, errors 0..100
        rep = dc.calibrate(e)
        self.assertEqual(rep["n_valid_nodes"], 1)
        self.assertEqual(rep["top_k_used"], 1)
        np.testing.assert_allclose(rep["stats"]["median"], [50])
        np.testing.assert_allclose(rep["stats"]["iqr"], [50])
        self.assertAlmostEqual(rep["thresholds"]["m1_raw_max"], 99.0)
        self.assertAlmostEqual(rep["thresholds"]["m2_max_normalised"], (99 - 50) / 50)
        self.assertAlmostEqual(rep["thresholds"]["m3_topk_mean_normalised"], rep["thresholds"]["m2_max_normalised"])

    def test_fewer_than_three_nodes_uses_all_of_them(self):
        e = np.array([[1.0, 3.0], [2.0, 6.0], [3.0, 9.0], [4.0, 12.0], [5.0, 15.0]])
        rep = dc.calibrate(e)
        self.assertEqual(rep["top_k_used"], 2)
        z = dc.normalise(e, rep["stats"])
        np.testing.assert_allclose(rep["scores"]["m3_topk_mean_normalised"], z.mean(1))

    def test_topk_is_three_with_many_nodes(self):
        rng = np.random.default_rng(0)
        e = rng.gamma(2.0, 1.0, size=(40, 10))
        rep = dc.calibrate(e)
        self.assertEqual(rep["top_k_used"], 3)
        z = dc.normalise(e, rep["stats"])
        np.testing.assert_allclose(rep["scores"]["m3_topk_mean_normalised"], np.sort(z, axis=1)[:, -3:].mean(1))
        np.testing.assert_allclose(rep["scores"]["m2_max_normalised"], z.max(1))
        np.testing.assert_allclose(rep["scores"]["m1_raw_max"], e.max(1))

    def test_all_constant_nodes_fall_back_to_the_absolute_floor(self):
        e = np.full((10, 4), 2.5)
        st = dc.fit_statistics(e)
        self.assertEqual(st["epsilon"], 1e-6)
        self.assertEqual(len(st["floor_affected_nodes"]), 4)
        rep = dc.calibrate(e)
        self.assertTrue(all(np.isfinite(v) for v in rep["thresholds"].values()))
        np.testing.assert_allclose(rep["scores"]["m2_max_normalised"], 0.0)

    def test_padded_nodes_are_ignored(self):
        rng = np.random.default_rng(1)
        e = rng.gamma(2.0, 1.0, size=(30, 5))
        padded = np.concatenate([e, np.full((30, 3), 1e9)], axis=1)               # garbage in padded nodes
        valid = np.array([True] * 5 + [False] * 3)
        a, b = dc.calibrate(e), dc.calibrate(padded, valid)
        for m in dc.METHODS:
            np.testing.assert_allclose(a["scores"][m], b["scores"][m])
            self.assertAlmostEqual(a["thresholds"][m], b["thresholds"][m])
        self.assertEqual(a["stats"]["epsilon"], b["stats"]["epsilon"])

    def test_sensitivity_table_and_monotonicity(self):
        rng = np.random.default_rng(2)
        rep = dc.calibrate(rng.gamma(2.0, 1.0, size=(300, 12)))
        self.assertEqual(sorted(rep["sensitivity"]), [95.0, 97.0, 99.0, 99.5])
        for m in dc.METHODS:
            vals = [rep["sensitivity"][p][m] for p in sorted(rep["sensitivity"])]
            self.assertEqual(vals, sorted(vals))
            self.assertEqual(rep["sensitivity"][99.0][m], rep["thresholds"][m])

    def test_every_method_has_its_own_threshold(self):
        rng = np.random.default_rng(3)
        rep = dc.calibrate(rng.gamma(2.0, 1.0, size=(200, 12)))
        self.assertEqual(set(rep["thresholds"]), set(dc.METHODS))
        self.assertEqual(len(set(rep["thresholds"].values())), 3)                  # no shared number across methods

    def test_apply_thresholds_uses_frozen_statistics(self):
        rng = np.random.default_rng(4)
        cal = rng.gamma(2.0, 1.0, size=(200, 6))
        rep = dc.calibrate(cal)
        hi = cal.copy()
        hi[:, 2] += 1000.0
        flags = dc.apply_thresholds(hi, rep["stats"], rep["thresholds"])
        self.assertTrue(all(f.all() for f in flags.values()))
        self.assertLess(dc.apply_thresholds(cal, rep["stats"], rep["thresholds"])["m2_max_normalised"].mean(), 0.05)


class NodeErrorTests(unittest.TestCase):
    def test_unavailable_entries_contribute_nothing(self):
        pred = np.zeros((2, 4)); target = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]])
        mask = np.array([[True, True, False, False], [False, False, False, False]])
        err, valid = dc.node_errors(pred, target, mask)
        np.testing.assert_allclose(err, [(1 + 4) / 2, 0.0])
        self.assertEqual(valid.tolist(), [True, False])
        target2 = target.copy(); target2[0, 2:] = 1e9                              # change only masked entries
        np.testing.assert_allclose(dc.node_errors(pred, target2, mask)[0], err)

    def test_bad_inputs_raise(self):
        with self.assertRaises(dc.CalibrationError):
            dc.node_errors(np.zeros((2, 3)), np.zeros((2, 3)), np.ones((2, 3), int))
        with self.assertRaises(dc.CalibrationError):
            dc.node_errors(np.zeros((2, 3)), np.zeros((2, 4)), np.ones((2, 3), bool))


class InvalidInputTests(unittest.TestCase):
    def test_invalid_errors_fail_loudly(self):
        for bad in (np.array([[1.0, np.nan]]), np.array([[1.0, -0.5]]), np.zeros((0, 3)), np.zeros(5), np.zeros((3, 0)), np.array([[np.inf, 1.0]])):
            with self.assertRaises(dc.CalibrationError):
                dc.fit_statistics(bad)
        with self.assertRaises(dc.CalibrationError):
            dc.fit_statistics(np.ones((3, 2)), node_valid=np.array([False, False]))
        with self.assertRaises(dc.CalibrationError):
            dc.fit_statistics(np.ones((3, 2)), node_valid=np.array([True]))
        with self.assertRaises(dc.CalibrationError):
            dc.thresholds({m: np.ones(3) for m in dc.METHODS}, 100.0)


if __name__ == "__main__":
    unittest.main()
