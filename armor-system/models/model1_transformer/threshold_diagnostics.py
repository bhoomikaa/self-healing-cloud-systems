"""Threshold diagnostics for the detection calibration (design note 4.5, Linear 298-51): ranks used and the 7-window block bootstrap.

Reuses `prep/detection_calibration` (statistics, normalisation, scores, thresholds); nothing here changes a threshold.
"""
import math

import numpy as np

from prep import detection_calibration as dc


def percentile_ranks(n, p):
    """Order statistics used by a linear-interpolation percentile of n values: ascending ranks (1-based), the same counted from the top, and the weight."""
    h = (n - 1) * p / 100.0
    lo, hi = int(math.floor(h)), int(math.ceil(h))
    return {"percentile": p, "n": n, "position": h, "lower_rank_ascending": lo + 1, "upper_rank_ascending": hi + 1,
            "lower_rank_from_top": n - lo, "upper_rank_from_top": n - hi, "weight_on_upper": h - lo}


def consecutive_fraction(start_index):
    """Share of adjacent windows (in the given order) that are consecutive minutes (start index differs by 1)."""
    s = np.asarray(start_index, np.int64)
    return float((np.diff(s) == 1).mean()) if len(s) > 1 else 1.0


def block_resample_indices(n, block, rng):
    """Moving-block resample: ceil(n / block) starts drawn with replacement from the n - block + 1 overlapping blocks, concatenated, cut to n."""
    if not 1 <= block <= n:
        raise ValueError(f"block length {block} must lie in 1..{n}")
    starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def block_bootstrap_thresholds(errors, node_valid=None, block=7, n_boot=1000, seed=0, percentile=99.0, top_k=3):
    """Per-method threshold uncertainty. `errors` (W, N) must be in TIME ORDER. Every resample is fully recalibrated (statistics, scores, threshold)."""
    errors = np.asarray(errors, np.float64)
    n = len(errors)
    rng = np.random.default_rng(seed)
    full_stats = dc.fit_statistics(errors, node_valid)
    full = dc.thresholds(dc.window_scores(errors, full_stats, top_k), percentile)
    taus = {m: np.empty(n_boot) for m in dc.METHODS}
    for b in range(n_boot):
        e = errors[block_resample_indices(n, block, rng)]
        st = dc.fit_statistics(e, node_valid)
        t = dc.thresholds(dc.window_scores(e, st, top_k), percentile)
        for m in dc.METHODS:
            taus[m][b] = t[m]
    out = {}
    for m in dc.METHODS:
        lo, hi = np.percentile(taus[m], [2.5, 97.5])
        out[m] = {"tau_full_sample": float(full[m]), "boot_mean": float(taus[m].mean()), "boot_sd": float(taus[m].std(ddof=1)), "ci95_low": float(lo), "ci95_high": float(hi)}
    return {"block_length": block, "resamples": n_boot, "seed": seed, "percentile": percentile, "windows": n, "n_blocks": n - block + 1, "blocks_per_resample": math.ceil(n / block), "methods": out}
