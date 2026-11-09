"""v1 detection scoring (V1_PROTOCOL.md sections 5-6): outputs collector, primary detector, diagnostics D1-D3, selection monitor.

Pure functions over "outputs" dicts so that the clean run and the missing-data runs share one implementation. Only calibration and
validation windows can be scored (`ScoringRoleError` otherwise); nothing here takes a gradient. Alerts are scored with
`run_detection_v0.evaluate_alerts` (the ARMOR-equivalent interval builder and scorer, +/-300 s proxy, the 25 documented validation incidents).

Primary: supervised logit -> causal 60 s rolling maximum -> empirical 90th percentile of the rolled calibration scores.
D1: other thresholds on the same rolled score (SPOT with ARMOR's parameters, 95/97/99/99.5th percentiles). D2: no rolling maximum.
D3: forecast-error M1/M2/M3 of the Stage 1 checkpoint (v0 protocol, M2 at the 99th percentile primary, plus 90/95/97/99.5).
"""
import numpy as np
import torch

from prep import detection_calibration as dc

from . import run_detection_v0 as rd, threshold_diagnostics as tdg, window_errors as we
from .rolling import rolling_max_causal

BOOT_SEED, N_BOOT, BLOCK = 20261028, 1000, 7
P_PRIMARY = 90.0
P_D1 = (95.0, 97.0, 99.0, 99.5)
P_D3 = (90.0, 95.0, 97.0, 99.0, 99.5)
D3_PRIMARY = ("m2_max_normalised", 99.0)


def collect_outputs(model, loader, allowed_roles, device="cpu"):
    """One eval-mode, no-gradient pass: fault logits, detection logits (NaN for a v0 model), per-node forecast errors and window metadata."""
    bad = set(allowed_roles) - set(we.SCORABLE_ROLES)
    if bad:
        raise we.ScoringRoleError(f"roles {sorted(bad)} cannot be scored: only {we.SCORABLE_ROLES} are allowed")
    dev = torch.device(device)
    was = model.training
    model.eval()
    keys = ("fault_logits", "det_logit", "errors", "node_valid", "target_ts", "start_index", "case_index", "fault_class", "ambiguous", "timestamps")
    parts = {k: [] for k in keys}
    role, category = [], []
    with torch.no_grad():
        for b in loader:
            if not set(b["role"]) <= set(allowed_roles):
                raise we.ScoringRoleError(f"batch holds roles {sorted(set(b['role']))}; only {tuple(allowed_roles)} may be scored")
            out = model(b["inputs"].to(dev), b["input_mask"].to(dev), applicable_entry=b["applicable_entry"].to(dev))
            err, nv = we.window_node_errors(out["prediction"].cpu().numpy(), b["target"].numpy(), b["target_mask"].numpy())
            n = len(b["role"])
            det = out["detection_logit"].double().cpu().numpy() if "detection_logit" in out else np.full(n, np.nan)
            for k, v in (("fault_logits", out["logits"].double().cpu().numpy()), ("det_logit", det), ("errors", err), ("node_valid", nv), ("target_ts", b["timestamps"][:, 6].numpy()),
                         ("start_index", b["start_index"].numpy()), ("case_index", b["case_index"].numpy()), ("fault_class", b["fault_class"].numpy()),
                         ("ambiguous", b["fault_label_ambiguous"].numpy()), ("timestamps", b["timestamps"].numpy())):
                parts[k].append(v)
            role += list(b["role"])
            category += list(b["category"])
    model.train(was)
    if not parts["errors"]:
        raise ValueError("no window was scored")
    res = {k: np.concatenate(v) for k, v in parts.items()}
    res["role"], res["category"] = np.array(role), np.array(category)
    return res


def sort_by_time(out):
    order = np.argsort(out["target_ts"], kind="stable")
    n = len(order)
    return {k: (v[order] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in out.items()}


def bce_with_logits(logit, y):
    """Numerically stable binary cross-entropy per element."""
    l = np.asarray(logit, np.float64)
    return np.maximum(l, 0) - l * y + np.log1p(np.exp(-np.abs(l)))


def case_level_nll(agg):
    p = agg["probs"][np.arange(len(agg["case_index"])), agg["true_class"]]
    return float(-np.log(np.maximum(p, 1e-12)).mean())


def softmax(x):
    z = np.asarray(x, np.float64)
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def event_windows(val):
    return (val["category"] == "event_associated") & (~val["ambiguous"]) & (val["fault_class"] >= 0)


def selection_monitor(val, ce):
    """Stage 2 checkpoint selection (V1_PROTOCOL.md section 3): case-level fault NLL + balanced window BCE over validation event/normal windows. No detection F1 or threshold."""
    ev = event_windows(val)
    agg = ce.aggregate_cases(val["case_index"][ev], softmax(val["fault_logits"][ev]), val["fault_class"][ev])
    nll = case_level_nll(agg)
    e, n = val["category"] == "event_associated", val["category"] == "normal_reference"
    bce = 0.5 * (float(bce_with_logits(val["det_logit"][e], 1.0).mean()) + float(bce_with_logits(val["det_logit"][n], 0.0).mean()))
    return {"case_nll": nll, "balanced_bce": bce, "monitor": nll + bce, "agg": agg}


# --------------------------------------------------------------------------- supervised detectors (primary, D1, D2)
def block_bootstrap_percentile(x, p, block=BLOCK, n_boot=N_BOOT, seed=0):
    """Moving-block bootstrap of one percentile of a time-ordered score series (blocks of `block` consecutive windows)."""
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(seed)
    vals = np.array([np.percentile(x[tdg.block_resample_indices(len(x), block, rng)], p, method="linear") for _ in range(n_boot)])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"percentile": p, "block_length": block, "resamples": n_boot, "seed": seed, "windows": len(x), "blocks_per_resample": -(-len(x) // block), "tau_full_sample": float(np.percentile(x, p, method="linear")),
            "boot_mean": float(vals.mean()), "boot_sd": float(vals.std(ddof=1)), "ci95_low": float(lo), "ci95_high": float(hi)}


def supervised_thresholds(cal, seed, armor_get_threshold=None, bootstrap=True, n_boot=N_BOOT):
    """Thresholds from the (time-ordered) calibration windows. `armor_get_threshold` (ARMOR's own function) adds the SPOT diagnostic."""
    c = sort_by_time(cal)
    rolled = rolling_max_causal(c["det_logit"], c["target_ts"], c["role"])
    pct = lambda x, p: float(np.percentile(x, p, method="linear"))        # noqa: E731
    th = {"primary_p90": pct(rolled, P_PRIMARY), "d2_p90_unrolled": pct(c["det_logit"], P_PRIMARY), **{f"p{p}": pct(rolled, p) for p in P_D1}}
    if armor_get_threshold is not None:
        th["spot"] = float(armor_get_threshold(rolled, 0.1, 0.95))        # ARMOR's initialisation step and fallback; its "threshold <= 0 invalid" rule is not applied
    info = {"n_windows": int(len(rolled)), "ranks_used_p90": tdg.percentile_ranks(len(rolled), P_PRIMARY), "consecutive_window_fraction": tdg.consecutive_fraction(c["start_index"]),
            "rolled_fraction_above_primary": float((rolled > th["primary_p90"]).mean())}
    if bootstrap:
        info["block_bootstrap_p90"] = block_bootstrap_percentile(rolled, P_PRIMARY, BLOCK, n_boot, BOOT_SEED + seed)
    return th, info


def supervised_validation(val, th, labels):
    """Primary, D1 and D2 on the validation windows."""
    v = sort_by_time(val)
    rolled = rolling_max_causal(v["det_logit"], v["target_ts"], v["role"])
    ts, cat = v["target_ts"], v["category"]
    runs = {"primary": (rolled, th["primary_p90"]), "d2_no_rolling": (v["det_logit"], th["d2_p90_unrolled"]), **{f"d1_p{p}": (rolled, th[f"p{p}"]) for p in P_D1}}
    if "spot" in th:
        runs["d1_spot"] = (rolled, th["spot"])
    return {name: {"threshold": tau, **rd.evaluate_alerts(s > tau, ts, cat, labels)} for name, (s, tau) in runs.items()}


# --------------------------------------------------------------------------- forecast-error detectors (D3, v0 protocol)
def masked_fit_statistics(errors, valid):
    """Per-node median/IQR over the windows where the node has effective entries (identical to `dc.fit_statistics` when every node is always valid)."""
    e, v = np.asarray(errors, np.float64), np.asarray(valid, bool)
    any_valid = v.any(0)
    if not any_valid.any():
        raise ValueError("no node has an effective entry in any calibration window")
    med, iqr = np.zeros(e.shape[1]), np.zeros(e.shape[1])
    q = np.nanpercentile(np.where(v, e, np.nan)[:, any_valid], [25, 50, 75], axis=0, method="linear")
    med[any_valid], iqr[any_valid] = q[1], q[2] - q[0]
    eps = max(dc.EPS_ABS, dc.EPS_REL * float(np.median(iqr[any_valid])))
    return {"median": med, "iqr": iqr, "epsilon": eps, "node_valid": any_valid}


def forecast_scores(errors, valid, stats, top_k=dc.TOP_K):
    """M1 raw max, M2 max normalised, M3 mean of the top-k normalised errors, over each window's valid nodes (-inf when a window has none)."""
    e, v = np.asarray(errors, np.float64), np.asarray(valid, bool)
    ok = v & stats["node_valid"][None]
    z = (e - stats["median"]) / np.maximum(stats["iqr"], stats["epsilon"])
    zv, ev = np.where(ok, z, -np.inf), np.where(ok, e, -np.inf)
    k = np.minimum(top_k, ok.sum(1))
    top = -np.sort(-zv, axis=1)[:, :top_k]
    m3 = np.where(np.arange(top_k)[None] < k[:, None], top, 0.0).sum(1) / np.maximum(k, 1)
    return {"m1_raw_max": ev.max(1), "m2_max_normalised": zv.max(1), "m3_topk_mean_normalised": np.where(k > 0, m3, -np.inf)}


def forecast_calibration(cal, percentiles=P_D3):
    c = sort_by_time(cal)
    stats = masked_fit_statistics(c["errors"], c["node_valid"])
    sc = forecast_scores(c["errors"], c["node_valid"], stats)
    th = {m: {str(p): float(np.percentile(s[np.isfinite(s)], p, method="linear")) for p in percentiles} for m, s in sc.items()}
    return stats, th, {"n_windows": int(len(c["errors"])), "epsilon": stats["epsilon"], "nodes_without_any_valid_window": int((~stats["node_valid"]).sum())}


def forecast_validation(val, stats, th, labels):
    v = sort_by_time(val)
    sc = forecast_scores(v["errors"], v["node_valid"], stats)
    return {m: {p: {"threshold": tau, **rd.evaluate_alerts(sc[m] > tau, v["target_ts"], v["category"], labels)} for p, tau in per.items()} for m, per in th.items()}
