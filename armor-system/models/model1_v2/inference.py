"""Model 1 v2 inference: the five frozen v1 Stage 2 checkpoints as one ensemble (V2_ENSEMBLE_PROTOCOL.md section 2).

    m = Model1V2.load()                                       # verifies every checkpoint against the test-freeze record
    out = m.predict_windows(batch)                            # batch: Model 1 window dict (inputs, input_mask, applicable_entry)
    flags = m.alert_flags(out["det_prob"], target_ts, period) # causal 60 s rolling max > the frozen calibration threshold
    cases = m.case_fault(out["fault_probs"], case_index)      # mean window softmax per case, averaged over seeds -> argmax

The detection threshold is the one computed from the 329 calibration windows (identical in the committed validation and test
records); it is a constant here, not refitted. Window construction and scaling are the frozen Model 1 data layer's job.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in ("data", "models", os.path.join("evaluation", "detection")):
    if os.path.join(REPO, "armor-system", _p) not in sys.path:
        sys.path.insert(0, os.path.join(REPO, "armor-system", _p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

CKPT_ROOT = os.path.expanduser("~/298A_project/checkpoints")
FREEZE = os.path.join(REPO, "armor-system", "models", "model1_transformer", "results", "test_freeze", "evidence_test_freeze.json")
THRESHOLD = 0.7831957951425433            # 90th percentile of the rolled ensemble score on the 329 calibration windows
CLASS_NAMES = ("Container Hardware", "Container Network", "Node CPU", "Node Disk", "Node Memory")
SEEDS = (0, 1, 2, 3, 4)


class Model1V2:
    def __init__(self, models, spec, threshold=THRESHOLD):
        self.models, self.spec, self.threshold = models, spec, threshold

    @classmethod
    def load(cls, ckpt_root=CKPT_ROOT, spec=None):
        """Load and hash-check the five frozen v1 S2 checkpoints. `spec` defaults to the D1 adapter spec from the verified data layer."""
        from model1_transformer import checkpoint as ck, test_freeze as tf, training_v1 as tv
        with open(FREEZE) as fh:
            frozen = {r["path"]: r["sha256"] for r in json.load(fh)["checkpoints"]["files"]}
        if spec is None:
            from model1_transformer.spec import AdapterSpec
            from prep import m1_dataset as md
            D = md.Model1Data.load()
            spec = AdapterSpec.from_d1(D.schema, D.applicable)
        models = []
        for s in SEEDS:
            rel = f"model1_v1/seed_{s}/s2_best"
            for ext in (".pt", ".json"):
                if tf.sha(os.path.join(ckpt_root, rel + ext)) != frozen[rel + ext]:
                    raise RuntimeError(f"{rel}{ext} differs from the frozen checkpoint")
            m = tv.build_model_v1(spec, s, torch.device("cpu"))
            ck.load_checkpoint(os.path.join(ckpt_root, rel), m, stage="s2", seed=s)
            m.eval()
            models.append(m)
        return cls(models, spec)

    @torch.no_grad()
    def predict_windows(self, batch):
        """Per window: mean over seeds of sigmoid(detection logit) and of the fault softmax; also the per-seed values."""
        from model1_transformer import detection_v1 as dv, ensemble as en
        det, fault = [], []
        for m in self.models:
            out = m(batch["inputs"], batch["input_mask"], applicable_entry=batch["applicable_entry"])
            det.append(en.sigmoid(out["detection_logit"].double().cpu().numpy()))           # the same functions as the evaluation path
            fault.append(dv.softmax(out["logits"].double().cpu().numpy()))
        det, fault = np.stack(det), np.stack(fault)
        return {"det_prob": det.mean(0), "fault_probs": fault.mean(0), "det_prob_per_seed": det, "fault_probs_per_seed": fault}

    def alert_flags(self, det_prob, target_ts, period):
        """Alerts per window (ascending target minutes): causal 60 s rolling max of the ensemble score above the frozen threshold."""
        from model1_transformer.rolling import rolling_max_causal
        return rolling_max_causal(det_prob, target_ts, period) > self.threshold

    def case_fault(self, fault_probs_per_seed, case_index):
        """Case-level fault type: per seed the mean window softmax per case, then the mean over seeds, argmax (ties: lowest id)."""
        p, c = np.asarray(fault_probs_per_seed, np.float64), np.asarray(case_index)
        cases = np.unique(c)
        probs = np.stack([np.stack([p[s][c == k].mean(0) for k in cases]) for s in range(len(p))]).mean(0)
        return {"case_index": cases, "probs": probs, "pred_class": probs.argmax(1), "pred_name": [CLASS_NAMES[i] for i in probs.argmax(1)]}
