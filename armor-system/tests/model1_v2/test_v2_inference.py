"""v2 inference entry point vs the evaluation path on validation windows (frozen checkpoints; skipped without data or checkpoints)."""
import os
import sys
import unittest

ARMOR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in ("data", "models", os.path.join("evaluation", "detection")):
    if os.path.join(ARMOR, p) not in sys.path:
        sys.path.insert(0, os.path.join(ARMOR, p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from prep import m1_dataset as md, paths  # noqa: E402

CK = os.path.expanduser("~/298A_project/checkpoints/model1_v1/seed_0/s2_best.pt")


@unittest.skipUnless(os.path.exists(paths.CACHE_FILE) and os.path.exists(CK), "D1 cache or frozen checkpoints not present")
class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from model1_v2.inference import Model1V2
        torch.set_num_threads(1)
        cls.D = md.Model1Data.load()
        cls.m = Model1V2.load()
        cls.loader = md.evaluation_loader(md.evaluation_dataset(cls.D, "validation_all"), batch_size=64)

    def test_ensemble_scores_equal_the_evaluation_path(self):
        from model1_transformer import detection_v1 as dv, ensemble as en
        batch = next(iter(self.loader))
        got = self.m.predict_windows(batch)
        outs = [dv.collect_outputs(mm, [batch], ("validation",)) for mm in self.m.models]
        ref = np.mean([en.sigmoid(o["det_logit"]) for o in outs], axis=0)
        self.assertTrue(np.array_equal(got["det_prob"], ref))
        self.assertTrue(np.array_equal(got["fault_probs_per_seed"][0], dv.softmax(outs[0]["fault_logits"])))

    def test_threshold_is_the_committed_calibration_threshold(self):
        import json
        p = os.path.join(ARMOR, "models", "model1_transformer", "results", "ensemble_validation_v1", "ensemble_validation.json")
        with open(p) as fh:
            self.assertEqual(self.m.threshold, json.load(fh)["ensemble"]["calibration"]["threshold"])

    def test_case_fault_matches_frozen_ensemble_on_validation(self):
        import csv
        per_seed, cases = [], []
        for b in self.loader:
            ev = (np.array(b["category"]) == "event_associated") & (~b["fault_label_ambiguous"].numpy()) & (b["fault_class"].numpy() >= 0)
            if not ev.any():
                continue
            o = self.m.predict_windows({k: (v[torch.from_numpy(ev)] if isinstance(v, torch.Tensor) and v.shape[:1] == (len(ev),) else v) for k, v in b.items()})
            per_seed.append(o["fault_probs_per_seed"])
            cases.append(b["case_index"].numpy()[ev])
        res = self.m.case_fault(np.concatenate(per_seed, axis=1), np.concatenate(cases))
        p = os.path.join(ARMOR, "models", "model1_transformer", "results", "ensemble_validation_v1", "predictions_v1_ensemble.csv")
        with open(p, newline="") as fh:
            committed = {int(r["case_index"]): int(r["predicted_class"]) for r in csv.DictReader(fh)}
        self.assertEqual({int(c): int(k) for c, k in zip(res["case_index"], res["pred_class"])}, committed)


if __name__ == "__main__":
    unittest.main()
