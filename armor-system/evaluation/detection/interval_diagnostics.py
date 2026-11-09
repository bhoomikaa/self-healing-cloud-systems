"""Diagnostics around ARMOR-style alert intervals (Linear 298-51): singleton count, detection delay, coverage, ignore-region classes.

Pure NumPy/stdlib. Uses the same closed-boundary overlap rule as `armor_scoring.eval_AD` (an interval [s, e] reaches a label t when
not (t - impact > e or t + impact < s)). Nothing here reads data: callers pass timestamps.
"""
import numpy as np


def raw_alert_groups(alerts, delay=600):
    """The (start, end) groups ARMOR's `get_pre_interval` forms BEFORE it drops those with start == end (same loop, same order rule)."""
    ts = list(alerts)
    if not ts:
        return []
    diffs = [0] + np.diff(ts).tolist()
    groups, start, end = [], None, None
    for t, diff in zip(ts, diffs):
        if start is None:
            start = int(t)
        if diff >= delay:
            groups.append((start, end))
            start = int(t)
        end = int(t)
    groups.append((start, end))
    return groups


def count_dropped_singletons(alerts, delay=600):
    """Number of alert groups ARMOR discards because start == end (a single alert minute, or only identical timestamps)."""
    return sum(1 for a, b in raw_alert_groups(alerts, delay) if a == b)


def label_matches(intervals, labels, impact=300):
    """{label: [intervals that reach it]} for the distinct labels, with ARMOR's closed-boundary rule."""
    return {t: [(s, e) for s, e in intervals if not (t - impact > e or t + impact < s)] for t in dict.fromkeys(labels)}


def detection_delays(intervals, labels, impact=300):
    """For every HIT label: start of the earliest reaching interval minus the label time, in seconds (negative = alert began before it)."""
    out = []
    for t, ivs in label_matches(intervals, labels, impact).items():
        if ivs:
            out.append((t, min(s for s, _ in ivs) - t))
    return sorted(out)


def delay_summary(delays):
    d = np.array([x for _, x in delays], np.float64)
    if len(d) == 0:
        return {"n_hits": 0, "median_s": None, "mean_s": None, "min_s": None, "max_s": None, "n_negative": 0}
    return {"n_hits": int(len(d)), "median_s": float(np.median(d)), "mean_s": float(d.mean()), "min_s": float(d.min()), "max_s": float(d.max()), "n_negative": int((d < 0).sum())}


def in_intervals(minutes, intervals, pad=0):
    """Boolean array: is each minute inside some interval, padded by `pad` seconds on both sides (closed)."""
    m = np.asarray(minutes, np.int64)
    inside = np.zeros(m.shape, bool)
    for s, e in intervals:
        inside |= (m >= s - pad) & (m <= e + pad)
    return inside


def classify_intervals(intervals, labels, ignore_minutes, impact=300):
    """Each interval once: 'hit' (reaches a label), else 'ignore_overlap' (contains the target minute of an ignore window), else 'normal_only'.

    Also returns `overlap_ignore_total`: intervals containing an ignore-window minute whether or not they are hits.
    """
    ign = np.asarray(ignore_minutes, np.int64)
    hit_ivs = {iv for ivs in label_matches(intervals, labels, impact).values() for iv in ivs}
    classes = []
    overlap_total = 0
    for s, e in intervals:
        has_ignore = bool(((ign >= s) & (ign <= e)).any())
        overlap_total += has_ignore
        classes.append("hit" if (s, e) in hit_ivs else ("ignore_overlap" if has_ignore else "normal_only"))
    return {"per_interval": classes, "hit": classes.count("hit"), "ignore_overlap": classes.count("ignore_overlap"), "normal_only": classes.count("normal_only"),
            "overlap_ignore_total": int(overlap_total)}
