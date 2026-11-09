"""Gate for the real-data integration run: with M1_REQUIRE_REAL_DATA=1 every required input must exist and be valid."""
import os
import unittest

import _m1_helpers as H
from prep import m1_split as ms, paths


class RequiredResourcesTests(unittest.TestCase):
    def test_committed_m1_artifacts_exist(self):
        self.assertTrue(H.ARTIFACTS_PRESENT, "model1_calibration_split_v1.json is missing")
        self.assertTrue(os.path.exists(os.path.join(paths.ART, "preprocessing", f"{ms.SCALER_NAME}.npz")), "fit-only scaler is missing")
        for rel in ("inspection/schema.json", "splits/case_split_assignments.csv", "splits/model1_split_101_25_84_v1.json", "preprocessing/scaler_model1_train_period_v1.npz"):
            self.assertTrue(os.path.exists(os.path.join(paths.ART, rel)), rel)

    def test_real_data_prerequisites_when_required(self):
        if not H.REQUIRE_REAL:
            return                                              # an ordinary run may lack the git-ignored cache
        self.assertTrue(H.CACHE_PRESENT, "feature cache data/processed/d1_features_v1.npz is missing")
        self.assertTrue(os.path.isdir(paths.D1_DIR), "extracted D1 release is missing")
        try:
            import torch  # noqa: F401
        except ImportError:
            self.fail("torch is required for the DataLoader tests")

    def test_skip_gate_is_inactive_by_default_and_active_when_required(self):
        deco = H.needs(False, "x")

        @deco
        class T(unittest.TestCase):
            def test_a(self):
                pass

        skipped = getattr(T, "__unittest_skip__", False)
        self.assertEqual(skipped, not H.REQUIRE_REAL)


if __name__ == "__main__":
    unittest.main()
