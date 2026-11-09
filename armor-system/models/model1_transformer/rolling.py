"""Causal trailing rolling maximum over consecutive minutes (V1_PROTOCOL.md section 5, amendment A-3). Pure NumPy.

s'_t = max(s_t, s_{t-60 s}) when the window whose target minute is t - 60 s exists AND belongs to the same period; otherwise s'_t = s_t
(a partial window). The output at minute t uses nothing later than t. Input order does not matter (sorted by time internally).
"""
import numpy as np


def rolling_max_causal(scores, target_ts, period, step=60):
    s, ts, per = np.asarray(scores, np.float64), np.asarray(target_ts, np.int64), np.asarray(period)
    if s.ndim != 1 or ts.shape != s.shape or per.shape != s.shape:
        raise ValueError("scores, target_ts and period must be 1-D arrays of equal length")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if len(np.unique(ts)) != len(ts):
        raise ValueError("duplicate target minutes: more than one window per minute")
    order = np.argsort(ts, kind="stable")
    ss, tt, pp = s[order], ts[order], per[order]
    out = ss.copy()
    prev_ok = (tt[1:] - tt[:-1] == step) & (pp[1:] == pp[:-1])
    out[1:] = np.where(prev_ok, np.maximum(ss[1:], ss[:-1]), ss[1:])
    res = np.empty_like(out)
    res[order] = out
    return res
