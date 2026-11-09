"""ARMOR-equivalent alert-interval construction and event-overlap scoring (Linear 298-51 scoring harness).

A line-for-line port of two functions from ARMOR's released code (Zenodo record 21744694,
`ARMOR/models/diagnosis_tasks/anomaly_detector.py::get_pre_interval` and
`ARMOR/models/diagnosis_tasks/task_metrics.py::eval_AD`). Pure NumPy/stdlib: no torch, no dataset, no Model 1 code.

The ARMOR semantics are kept exactly, including the ones that look like quirks (all are listed in the harness evidence):
  * `get_pre_interval` takes the alert timestamps in the order given (it never sorts); a new interval starts when the gap to
    the previous alert is >= `delay` (600 s) - a gap of exactly 600 s splits, 599 s merges; an interval is (first alert,
    last alert); intervals with start == end (a single alert, or repeated identical timestamps only) are dropped.
  * `eval_AD` keys intervals and labels in dicts, so duplicated intervals or duplicated labels count once; a label is hit when
    an interval overlaps [t - impact, t + impact] with CLOSED boundaries; TP = labels hit (>= 1 interval), FP = intervals that hit
    no label, FN = labels not hit; precision and recall are rounded to 4 decimals and F1 is computed from the ROUNDED
    precision and recall; TP + FP = 0 or TP + FN = 0 raises ZeroDivisionError; the density mean over an empty set is NaN.
  * Interval lists must contain hashable pairs (tuples); a list of lists raises TypeError exactly as in ARMOR.

`eval_AD_detail` runs the same body and also returns the counts (TP, FP, FN, ...), which ARMOR computes but does not return.
"""
import numpy as np


def get_pre_interval(alert_timestamps, delay=600):
    """ARMOR `get_pre_interval` on the timestamps of the rows with outlier == 1 (in row order)."""
    ts = list(alert_timestamps)
    if not ts:
        return []
    diffs = [0] + np.diff(ts).tolist()
    pre_interval = []
    start_ts, end_ts = None, None
    for t, diff in zip(ts, diffs):
        if start_ts is None:
            start_ts = int(t)
        if diff >= delay:
            pre_interval.append((start_ts, end_ts))
            start_ts = int(t)
        end_ts = int(t)
    pre_interval.append((start_ts, end_ts))
    return [(item[0], item[1]) for item in pre_interval if item[0] != item[1]]


def eval_AD_detail(pre_interval, ad_cases_label, impact_window=5 * 60):
    """ARMOR `eval_AD` body. Returns a dict with precision, recall, f1 (exactly ARMOR's values) and the counts."""
    pre_dict = {key: set() for key in pre_interval}
    ad_dict = {key: set() for key in ad_cases_label}
    for s, e in pre_interval:
        for case_ts in ad_cases_label:
            case_s, case_e = case_ts - impact_window, case_ts + impact_window
            if not (case_s > e or case_e < s):
                pre_dict[(s, e)].add(case_ts)
                ad_dict[case_ts].add((s, e))

    TP = len([key for key, value in ad_dict.items() if len(value) > 0])
    FP = len([key for key, value in pre_dict.items() if len(value) == 0])
    FN = len([key for key, value in ad_dict.items() if len(value) == 0])
    precision = np.round(TP / (TP + FP), 4)
    recall = np.round(TP / (TP + FN), 4)
    f1 = np.round(2 * precision * recall / (precision + recall), 4)
    density = np.round(np.mean([len(value) for _, value in pre_dict.items() if len(value) > 0]), 2)
    return {"precision": precision, "recall": recall, "f1": f1, "density": density, "TP": TP, "FP": FP, "FN": FN,
            "n_intervals": len(pre_dict), "n_labels": len(ad_dict)}


def eval_AD(pre_interval, ad_cases_label, impact_window=5 * 60, verbose=False):
    """ARMOR `eval_AD`: returns (precision, recall, f1)."""
    r = eval_AD_detail(pre_interval, ad_cases_label, impact_window)
    if verbose:
        print("AD:")
        print(f'precision: {r["precision"]}, recall: {r["recall"]}, f1: {r["f1"]}, density: {r["density"]}')
    return r["precision"], r["recall"], r["f1"]
