"""Scoring sets for one evaluation period, built with the validation rules (Linear 298-51, TEST_PLAN.md sections 3-4).

For a period ('validation' or 'test'):
  detection_rows  every window of the period's role (all three categories, ambiguous windows kept), ordered by target minute;
                  `ignore` windows are scored but never counted as negatives (ARMOR interval rules, run_detection_v0.evaluate_alerts)
  event_rows      the approved non-ambiguous `event_associated` windows (case-level fault classification, case_eval protocol)
  labels          the documented incident times of the period's cases (case table), sorted: the "our scoring" population
The test period needs a TestRunToken (test_freeze.claim_test_run); without one this module refuses it. For validation the result
must equal the committed validation set exactly (869 target minutes 150/411/308, the 25 incident times of run_detection_v0).
"""
import numpy as np

from prep import m1_dataset as md

from . import test_freeze as tf

APPROVED = {"validation": {"windows": 869, "event_associated": 150, "normal_reference": 411, "ignore": 308, "cases": 25},
            "test": {"windows": 2868, "event_associated": 501, "cases": 84}}
CATEGORIES = ("event_associated", "normal_reference", "ignore")


class ScoringSetError(RuntimeError):
    """The scoring set differs from the approved construction."""


def build(D, period, token=None):
    if period not in APPROVED:
        raise ValueError(f"period must be one of {sorted(APPROVED)}")
    if period == "test":
        tf.check_token(token)
    t = D.table
    rows = D.selection([period])
    rows = rows[np.argsort(t.T[rows, -1], kind="stable")]
    target = t.T[rows, -1].astype(np.int64)
    cat = t.category[rows]
    counts = {"windows": int(len(rows)), **{c: int((cat == c).sum()) for c in CATEGORIES}}
    ev = D.selection([period], ["event_associated"], drop_ambiguous=True)
    cases = sorted(a["case_index"] for a in D.manifest["cases"]["assignments"] if a["role"] == period)
    if sorted(set(int(c) for c in t.case_index[ev])) != cases:
        raise ScoringSetError(f"{period}: cases with approved event windows differ from the manifest's cases")
    labels = sorted(int(D.cases["timestamp"][c]) for c in cases)
    exp = APPROVED[period]
    got = {k: (len(cases) if k == "cases" else counts[k]) for k in exp}
    if got != exp:
        raise ScoringSetError(f"{period}: counts {got} differ from the approved {exp}")
    if len(np.unique(target)) != len(target):
        raise ScoringSetError(f"{period}: two windows share a target minute")
    return {"period": period, "detection_rows": rows, "target_ts": target, "category": cat, "counts": counts, "event_rows": np.sort(ev), "cases": cases,
            "labels": labels, "contiguous_target_minutes": bool((np.diff(target) == D.cfg.step_s).all()), "n_ambiguous_event_windows_dropped": int((cat == "event_associated").sum() - len(ev))}


def loader(D, rows, name, batch_size=64):
    return md.evaluation_loader(md.Model1WindowDataset(D, rows, name), batch_size=batch_size)
