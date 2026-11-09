"""Synthetic equivalence battery: ARMOR's ORIGINAL scoring functions vs the port in `armor_scoring.py`.

Everything here is synthetic (generated alert flags and event timestamps). It reads ARMOR's released SOURCE files from the
git-ignored raw folder (`get_pre_interval` is extracted verbatim from `anomaly_detector.py` with `ast`, because that module
imports ARMOR's model code; `eval_AD` is imported from `task_metrics.py`, which needs only numpy). It touches no label,
no sample and no result file of any period.
"""
import ast
import hashlib
import importlib.util
import os
import warnings

import numpy as np
import pandas as pd

from armor_scoring import eval_AD as port_eval_AD, eval_AD_detail as port_eval_detail, get_pre_interval as port_get_pre_interval

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ARMOR_DIR = os.path.join(REPO, "armor-system", "data", "raw", "armor_zenodo_21744694", "ARMOR")
DETECTOR = os.path.join(ARMOR_DIR, "models", "diagnosis_tasks", "anomaly_detector.py")
METRICS = os.path.join(ARMOR_DIR, "models", "diagnosis_tasks", "task_metrics.py")


def available():
    return os.path.isfile(DETECTOR) and os.path.isfile(METRICS)


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_original():
    """(original eval_AD, original get_pre_interval, provenance). The function sources are executed unchanged."""
    spec = importlib.util.spec_from_file_location("armor_task_metrics", METRICS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open(DETECTOR) as fh:
        src = fh.read()
    seg = next(ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "get_pre_interval")
    ns = {"np": np, "pd": pd}
    exec(compile(seg, DETECTOR + "::get_pre_interval", "exec"), ns)                        # noqa: S102  (ARMOR's own function source, unchanged)
    prov = {"anomaly_detector.py_sha256": _sha(DETECTOR), "task_metrics.py_sha256": _sha(METRICS), "get_pre_interval_source_sha256": hashlib.sha256(seg.encode()).hexdigest()}
    return mod.eval_AD, ns["get_pre_interval"], prov


def original_frame(timestamps, flags):
    """The DataFrame ARMOR passes to `get_pre_interval`: one row per scored minute with outlier 0/1 (row order kept)."""
    return pd.DataFrame({"timestamp": np.asarray(timestamps, dtype=np.int64), "outlier": np.asarray(flags, dtype=np.int64)})


def _capture(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return ("ok", fn(*a, **k))
        except Exception as e:                                                             # noqa: BLE001  (exceptions are part of the semantics)
            return ("raised", type(e).__name__, str(e))


def same(a, b):
    """Exact equality that treats NaN == NaN and compares types of the numeric results."""
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float) or isinstance(a, np.floating) or isinstance(b, np.floating):
        return type(a) is type(b) and ((np.isnan(a) and np.isnan(b)) or a == b)
    return a == b and type(a) is type(b)


def compare_case(orig_eval, orig_get, timestamps, flags, labels, delay=600, impact=300, as_lists=False):
    """Run both implementations on one synthetic case. Returns a dict of booleans (all must be True) and the interval count."""
    ts, fl = np.asarray(timestamps), np.asarray(flags)
    alerts = ts[fl == 1].tolist()
    oi = _capture(orig_get, original_frame(ts, fl), delay)
    pi = _capture(port_get_pre_interval, alerts, delay)
    out = {"intervals_identical": same(oi, pi)}
    iv = oi[1] if oi[0] == "ok" else []
    if as_lists:
        iv = [list(x) for x in iv]
    oe = _capture(orig_eval, iv, list(labels), impact)
    pe = _capture(port_eval_AD, iv, list(labels), impact)
    out["scores_identical"] = same(oe, pe)
    if oe[0] == "ok":                                                                       # the detail variant returns ARMOR's exact triple
        d = _capture(port_eval_detail, iv, list(labels), impact)
        out["detail_triple_identical"] = d[0] == "ok" and same((d[1]["precision"], d[1]["recall"], d[1]["f1"]), oe[1])
    out["n_intervals"] = len(iv)
    out["outcome"] = oe[0] if oe[0] != "ok" else "scored"
    return out


# (name, timestamps(minutes grid), flags, labels, delay, impact, as_lists)
def edge_cases():
    base = 1_000_000
    g = lambda n: [base + 60 * i for i in range(n)]            # noqa: E731
    def flags(n, on):
        f = [0] * n
        for i in on:
            f[i] = 1
        return f
    cases = []
    cases.append(("gap_660_between_alert_groups", g(40), flags(40, [0, 1, 11, 12]), [base + 100, base + 700], 600, 300))
    t = [base, base + 60, base + 60 + 599, base + 60 + 599 + 60]
    cases.append(("gap_exactly_599", t, [1, 1, 1, 1], [base], 600, 300))
    t = [base, base + 60, base + 60 + 600, base + 60 + 600 + 60]
    cases.append(("gap_exactly_600_splits", t, [1, 1, 1, 1], [base, base + 700], 600, 300))
    t = [base, base + 60, base + 60 + 601, base + 60 + 601 + 60]
    cases.append(("gap_601_splits", t, [1, 1, 1, 1], [base], 600, 300))
    cases.append(("singleton_dropped", g(10), flags(10, [3]), [base + 180], 600, 300))
    cases.append(("all_singletons_no_intervals", g(60), flags(60, [0, 20, 40]), [base + 10], 600, 300))
    cases.append(("singleton_between_groups", g(60), flags(60, [0, 1, 20, 40, 41]), [base, base + 2400], 600, 300))
    cases.append(("two_consecutive_alerts_kept", g(10), flags(10, [4, 5]), [base + 270], 600, 300))
    cases.append(("one_interval_matches_many_labels", g(40), flags(40, list(range(5, 15))), [base + 400, base + 500, base + 700, base + 900], 600, 300))
    cases.append(("one_label_matched_by_two_intervals", g(60), flags(60, [10, 11, 25, 26]), [base + 1000], 600, 300))
    cases.append(("repeated_timestamps_merge", [base, base, base, base + 60], [1, 1, 1, 1], [base], 600, 300))
    cases.append(("identical_repeats_only_dropped", [base, base, base], [1, 1, 1], [base], 600, 300))
    cases.append(("duplicate_labels_count_once", g(20), flags(20, [3, 4, 5]), [base + 200, base + 200, base + 5000], 600, 300))
    cases.append(("duplicate_intervals_count_once", g(10), flags(10, [1, 2]), [base + 5000], 600, 300))
    cases.append(("label_touches_interval_end_inclusive", g(20), flags(20, [5, 6]), [base + 360 + 300], 600, 300))
    cases.append(("label_one_second_after_reach", g(20), flags(20, [5, 6]), [base + 360 + 301], 600, 300))
    cases.append(("label_touches_interval_start_inclusive", g(20), flags(20, [5, 6]), [base + 300 - 300], 600, 300))
    cases.append(("label_one_second_before_reach", g(20), flags(20, [5, 6]), [base + 300 - 301], 600, 300))
    cases.append(("no_alerts_zero_division", g(10), [0] * 10, [base], 600, 300))
    cases.append(("no_labels_zero_division", g(10), flags(10, [2, 3]), [], 600, 300))
    cases.append(("all_false_alarms_zero_recall", g(20), flags(20, [2, 3, 10, 11]), [base + 100000], 600, 300))
    cases.append(("f1_from_rounded_precision_recall", g(200), flags(200, [0, 1, 20, 21, 40, 41, 60, 61, 80, 81, 100, 101]), [base, base + 1200, base + 2400], 600, 300))
    cases.append(("custom_delay_120", g(30), flags(30, [0, 1, 4, 5, 8]), [base + 30, base + 480], 120, 300))
    cases.append(("custom_impact_60", g(30), flags(30, [5, 6]), [base + 420], 600, 60))
    cases.append(("alert_every_minute", g(100), [1] * 100, [base + 600, base + 3000], 600, 300))
    return cases


def run_battery(n_random=400, seed=20261028):
    """Edge cases + seeded random sweep (+ unsorted timestamps and list-of-lists intervals). Returns a summary dict."""
    orig_eval, orig_get, prov = load_original()
    results, ok = [], True
    for name, ts, fl, labels, delay, impact in edge_cases():
        r = compare_case(orig_eval, orig_get, ts, fl, labels, delay, impact)
        results.append({"case": name, **r})
        ok &= all(v for k, v in r.items() if k.endswith("identical"))
    rng = np.random.default_rng(seed)
    n_scored = n_raised = 0
    for i in range(n_random):
        n = int(rng.integers(5, 200))
        base = int(rng.integers(1_000_000, 2_000_000))
        ts = base + 60 * np.arange(n)
        p = float(rng.choice([0.02, 0.05, 0.1, 0.25, 0.5, 0.9]))
        fl = (rng.random(n) < p).astype(int)
        k = int(rng.integers(0, 14))
        labels = (base + rng.integers(-700, 60 * n + 700, size=k)).tolist()
        if k and rng.random() < 0.3:
            labels.append(labels[0])                                                        # a repeated label
        delay = int(rng.choice([60, 120, 300, 600, 601, 900]))
        impact = int(rng.choice([60, 300, 301]))
        r = compare_case(orig_eval, orig_get, ts, fl, labels, delay, impact)
        ok &= all(v for kk, v in r.items() if kk.endswith("identical"))
        n_scored += r["outcome"] == "scored"
        n_raised += r["outcome"] == "raised"
    unsorted_ok = True                                                                      # ARMOR never sorts: both must agree on unsorted input too
    for i in range(60):
        n = int(rng.integers(5, 60))
        base = int(rng.integers(1_000_000, 2_000_000))
        ts = (base + 60 * rng.permutation(n)).astype(np.int64)
        fl = (rng.random(n) < 0.5).astype(int)
        r = compare_case(orig_eval, orig_get, ts, fl, [int(base + 60 * rng.integers(0, n))], 600, 300)
        unsorted_ok &= all(v for kk, v in r.items() if kk.endswith("identical"))
    lists = compare_case(orig_eval, orig_get, [1_000_000 + 60 * i for i in range(10)], [0, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1_000_100], as_lists=True)
    return {"all_identical": bool(ok and unsorted_ok and lists["scores_identical"]), "edge_cases": results, "random_cases": n_random, "random_seed": seed,
            "random_scored": int(n_scored), "random_raised_same_exception": int(n_raised), "unsorted_cases": 60, "unsorted_identical": bool(unsorted_ok),
            "list_of_lists_intervals_identical_error": bool(lists["scores_identical"]) and lists["outcome"] == "raised", "provenance": prov}
