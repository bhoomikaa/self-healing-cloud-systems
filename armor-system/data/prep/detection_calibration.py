"""M1-03 detection-calibration utility (design note section 4.5). Pure NumPy; needs no model and no data.

Input: per-window, per-node prediction errors `e` of shape (W, N) from the calibration-reference windows.
It computes, with linear-interpolation percentiles throughout:
  per-node median and IQR; epsilon = max(1e-6, 0.01 * median(IQR over valid nodes)); the floor-affected nodes
  (IQR < epsilon); z = (e - median) / max(IQR, epsilon); and three window scores
  M1 raw max, M2 max normalised (primary), M3 mean of the min(3, N_v) largest z.
Each method gets its own percentile threshold. This module creates no threshold for real data: it is a function
of whatever errors are passed in, and the numbers it returns belong to exactly those errors (and, later, to the
exact S1 checkpoint that produced them).
"""
import numpy as np

EPS_ABS = 1e-6
EPS_REL = 0.01
TOP_K = 3
PERCENTILE = 99.0
SENSITIVITY_PERCENTILES = (95.0, 97.0, 99.0, 99.5)
METHODS = ("m1_raw_max", "m2_max_normalised", "m3_topk_mean_normalised")


class CalibrationError(ValueError):
    """Invalid calibration input."""


def node_errors(pred, target, mask):
    """Per-node masked mean squared error. pred/target (..., N, C), mask (..., N, C) bool -> (errors (..., N), node_valid (..., N)).

    Unavailable entries contribute nothing. A node with no effective entry is invalid (error 0, node_valid False).
    """
    pred, target, mask = np.asarray(pred, np.float64), np.asarray(target, np.float64), np.asarray(mask)
    if mask.dtype != np.bool_ or pred.shape != target.shape or mask.shape != pred.shape:
        raise CalibrationError("pred, target and a bool mask must share one shape")
    n = mask.sum(-1)
    sq = np.where(mask, (pred - target) ** 2, 0.0)
    err = np.divide(sq.sum(-1), n, out=np.zeros(n.shape), where=n > 0)
    return err, n > 0


def _check(errors, node_valid):
    e = np.asarray(errors, np.float64)
    if e.ndim != 2 or e.shape[0] == 0 or e.shape[1] == 0:
        raise CalibrationError(f"errors must be (windows, nodes) and non-empty, got {e.shape}")
    nv = np.ones(e.shape[1], bool) if node_valid is None else np.asarray(node_valid, bool)
    if nv.shape != (e.shape[1],) or not nv.any():
        raise CalibrationError("node_valid must be a (nodes,) bool vector with at least one valid node")
    if not np.isfinite(e[:, nv]).all():
        raise CalibrationError("errors of valid nodes must be finite")
    if (e[:, nv] < 0).any():
        raise CalibrationError("errors are squared errors and cannot be negative")
    return e, nv


def fit_statistics(errors, node_valid=None):
    """Per-node median / IQR, epsilon and the floor report from calibration-reference errors (W, N).

    Invalid nodes get median 0 and IQR 0 and are never counted. Returns a plain dict (JSON-friendly via tolist()).
    """
    e, nv = _check(errors, node_valid)
    med = np.zeros(e.shape[1])
    iqr = np.zeros(e.shape[1])
    q = np.percentile(e[:, nv], [25, 50, 75], axis=0, method="linear")
    med[nv], iqr[nv] = q[1], q[2] - q[0]
    eps = max(EPS_ABS, EPS_REL * float(np.median(iqr[nv])))
    floor = np.nonzero(nv & (iqr < eps))[0]
    return {"median": med, "iqr": iqr, "epsilon": eps, "node_valid": nv, "n_windows": int(e.shape[0]),
            "floor_affected_nodes": [{"node": int(i), "iqr": float(iqr[i])} for i in floor]}


def normalise(errors, stats):
    """z = (e - median) / max(IQR, epsilon), 0 for invalid nodes. errors (..., N)."""
    e = np.asarray(errors, np.float64)
    z = (e - stats["median"]) / np.maximum(stats["iqr"], stats["epsilon"])
    return np.where(stats["node_valid"], z, 0.0)


def window_scores(errors, stats, top_k=TOP_K):
    """The three window scores for errors (W, N): dict method -> (W,) array. Invalid nodes are ignored."""
    e, nv = _check(errors, stats["node_valid"])
    z = normalise(e, stats)
    k = min(top_k, int(nv.sum()))
    zv = np.where(nv, z, -np.inf)
    top = -np.sort(-zv, axis=1)[:, :k]
    return {"m1_raw_max": np.where(nv, e, -np.inf).max(1), "m2_max_normalised": zv.max(1), "m3_topk_mean_normalised": top.mean(1)}


def thresholds(scores, percentile=PERCENTILE):
    """Per-method threshold = percentile (linear interpolation) of that method's calibration scores."""
    if not 0 < percentile < 100:
        raise CalibrationError("percentile must be in (0, 100)")
    return {m: float(np.percentile(scores[m], percentile, method="linear")) for m in METHODS}


def sensitivity_table(scores, percentiles=SENSITIVITY_PERCENTILES):
    """{percentile: {method: threshold}} for validation-only sensitivity reporting (95/97/99/99.5 by default)."""
    return {float(p): thresholds(scores, p) for p in percentiles}


def calibrate(errors, node_valid=None, percentile=PERCENTILE, top_k=TOP_K):
    """Full S3 calibration report for calibration-reference errors (W, N). No model, no gradient."""
    stats = fit_statistics(errors, node_valid)
    scores = window_scores(errors, stats, top_k)
    return {"stats": stats, "scores": scores, "percentile": float(percentile), "top_k_used": min(top_k, int(stats["node_valid"].sum())),
            "thresholds": thresholds(scores, percentile), "sensitivity": sensitivity_table(scores),
            "n_windows": stats["n_windows"], "n_valid_nodes": int(stats["node_valid"].sum())}


def apply_thresholds(errors, stats, tau, top_k=TOP_K):
    """Alert flags per method for new errors (e.g. validation): {method: (W,) bool} using the frozen calibration stats and tau."""
    sc = window_scores(errors, stats, top_k)
    return {m: sc[m] > tau[m] for m in METHODS}
