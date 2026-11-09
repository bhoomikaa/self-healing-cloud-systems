"""Five-seed probability-averaged ensemble, the secondary row of V1_PROTOCOL.md section 11.3 (never chosen from, never the primary).

Fault type: the mean over seeds of each case's class probabilities (case_eval: mean window softmax per case), then argmax (ties go to the
lowest class id). Detection: the mean over seeds of sigmoid(detection logit) per window, then the causal 60 s rolling maximum, with the
threshold the empirical 90th percentile (linear) of the rolled ensemble score on the 329 calibration windows; then the ARMOR interval rules.
"""
import numpy as np

from . import detection_v1 as dv, run_detection_v0 as rd
from .rolling import rolling_max_causal

META = ("target_ts", "role", "category", "start_index")


class EnsembleError(ValueError):
    """The seeds' outputs do not describe the same cases or windows."""


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, np.float64)))


def fault(aggs):
    ref = aggs[0]
    for a in aggs[1:]:
        if a["case_index"].tolist() != ref["case_index"].tolist() or a["true_class"].tolist() != ref["true_class"].tolist():
            raise EnsembleError("seeds disagree on cases or labels")
    probs = np.mean([a["probs"] for a in aggs], axis=0)
    return {"case_index": ref["case_index"], "true_class": ref["true_class"], "probs": probs, "pred_class": probs.argmax(1).astype(np.int64), "n_windows": ref["n_windows"], "n_seeds": len(aggs)}


def detection_scores(outs):
    """Per-window mean of sigmoid(logit) over seeds, with the shared window metadata (time-ordered)."""
    s = [dv.sort_by_time(o) for o in outs]
    for o in s[1:]:
        for k in META:
            if not np.array_equal(o[k], s[0][k]):
                raise EnsembleError(f"seeds disagree on window {k}")
    if any(np.isnan(o["det_logit"]).any() for o in s):
        raise EnsembleError("a detection logit is missing (a v0 model has no detection head)")
    return {**{k: s[0][k] for k in META}, "det_prob": np.mean([sigmoid(o["det_logit"]) for o in s], axis=0), "n_seeds": len(s)}


def threshold(cal):
    rolled = rolling_max_causal(cal["det_prob"], cal["target_ts"], cal["role"])
    tau = float(np.percentile(rolled, dv.P_PRIMARY, method="linear"))
    return tau, {"n_windows": int(len(rolled)), "percentile": dv.P_PRIMARY, "rolled_fraction_above": float((rolled > tau).mean())}


def evaluate(ev, tau, labels):
    rolled = rolling_max_causal(ev["det_prob"], ev["target_ts"], ev["role"])
    return {"threshold": tau, **rd.evaluate_alerts(rolled > tau, ev["target_ts"], ev["category"], labels)}
