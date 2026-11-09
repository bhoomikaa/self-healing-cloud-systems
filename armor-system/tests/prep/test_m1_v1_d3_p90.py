"""Checks of the D3 missing-data results at the 90th-percentile operating point (validation only, no retraining)."""
import json
import os
import unittest

import _m1_model_helpers as MH  # noqa: F401
import numpy as np

from model1_transformer import missing_d3_p90 as m9, run_v1 as r1

R = r1.RESULTS


def jl(*p):
    with open(os.path.join(*p)) as fh:
        return json.load(fh)


class D3P90ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.per = {s: jl(R, f"missing_d3_p90_seed_{s}.json") for s in r1.SEEDS}
        cls.sm = jl(R, "missing_d3_p90_summary.json")

    def test_clean_reference_equals_the_validation_d3_result_at_p90(self):
        for s in r1.SEEDS:
            ref = jl(R, f"seed_{s}", "detection.json")["d3_forecast_error_stage1"]["validation"][m9.METRIC][str(m9.PCT)]
            c = self.per[s]["clean_reference"]["recalibrated"]
            for k in ("f1", "TP", "FP", "FN", "precision", "recall"):
                self.assertAlmostEqual(c[k], ref[k], places=9)
            self.assertEqual(c, self.per[s]["clean_reference"]["clean_thresholds"])       # unmasked: both modes coincide

    def test_rate_zero_is_the_clean_reference_and_masks_are_the_same_as_the_99th_percentile_run(self):
        for s in r1.SEEDS:
            self.assertEqual(self.per[s]["random"]["0.0"]["0"], self.per[s]["clean_reference"])
            old = jl(R, f"missing_seed_{s}.json")
            self.assertEqual({r: len(v) for r, v in self.per[s]["random"].items()}, {r: len(v) for r, v in old["random"].items()})
            self.assertEqual({n: (a["dropped"], a["hides_all_vms"]) for n, a in self.per[s]["armor"].items()}, {n: (a["dropped"], a["hides_all_vms"]) for n, a in old["armor"].items()})

    def test_summary_is_the_pooled_mean_of_the_per_seed_cells(self):
        for r in ("0.1", "0.4"):
            vals = [c["recalibrated"]["f1"] for s in r1.SEEDS for c in self.per[s]["random"][r].values()]
            self.assertEqual(len(vals), 50)
            self.assertAlmostEqual(self.sm["random_rates"][r]["d3_p90"]["recalibrated"]["f1"]["mean"], float(np.mean(vals)), places=12)
        self.assertEqual({n for n, a in self.sm["armor_scenarios"].items() if a["hides_all_vms"]}, {"drop_metric", "drop_log+metric", "drop_metric+trace"})

    def test_summary_aggregation_is_reproducible(self):
        before = json.dumps(self.sm, sort_keys=True)
        self.assertEqual(json.dumps(m9.aggregate(), sort_keys=True), before)


if __name__ == "__main__":
    unittest.main()
