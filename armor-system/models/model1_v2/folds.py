"""Day-blocked development folds for the v2 CV (V2_ENSEMBLE_PROTOCOL.md section 3b). No test-period window ever enters a fold.

A fold is the verified `Model1Data` with a relabelled window table:
  training-fold windows  -> role "fit"         (reachable only through the frozen training_dataset / PairedLoader)
  held-out windows       -> role "validation"  (reachable only through evaluation_dataset("validation_all"))
  calibration windows    -> role "calibration" (unchanged; the 329 threshold windows)
Embargo and test-period rows are dropped from the table. The frozen datasets check that a window lies inside the time region of
its role (`m1_split_region_ok`), which assumes the approved roles; `fold_regions()` swaps in the fold's own region check for the
duration of a with-block and restores the original afterwards (no frozen file is modified).
"""
import contextlib
import copy

import numpy as np

from prep import m1_dataset as md, m1_split as ms

DAY1_END = 1651420740              # 2022-05-01 15:59:00 UTC: last sample of recorded day 1
DAY2_START = 1651507200            # 2022-05-02 16:00:00 UTC: first sample of recorded day 2
FOLDS = ("F1", "F2", "F3")
EXPECTED_CASES = {"F1": list(range(0, 45)), "F2": list(range(45, 88)), "F3": list(range(101, 126))}


class FoldError(RuntimeError):
    """A fold differs from the predeclared layout or would leak."""


def blocks(D):
    """{fold: (lo, hi)} time ranges [lo, hi) that each held-out block's windows (t0 .. t6) must lie in."""
    b1, b2 = D.boundaries
    return {"F1": (-ms.BIG, DAY1_END + 1), "F2": (DAY2_START, D.cfg.tc - D.cfg.embargo_s), "F3": (b1, b2)}


def window_block(D):
    """(n,) str: the fold block of every window of the approved table ('' for calibration, embargo and test windows)."""
    t = D.table
    t0, t6 = t.T[:, 0], t.T[:, -1]
    out = np.full(len(t0), "", dtype=object)
    fit = t.role == "fit"
    out[fit & (t6 <= DAY1_END)] = "F1"
    out[fit & (t0 >= DAY2_START)] = "F2"
    out[t.role == "validation"] = "F3"
    if (fit & (out == "")).any():
        raise FoldError("a fitting window lies in neither recorded day 1 nor recorded day 2")
    return out


def fold_cases(D):
    """{fold: sorted case indices} from the approved case-role assignment and the day boundary."""
    out = {f: [] for f in FOLDS}
    for a in D.manifest["cases"]["assignments"]:
        c, ts = int(a["case_index"]), int(D.cases["timestamp"][a["case_index"]])
        if a["role"] == "fit":
            out["F1" if ts <= DAY1_END else "F2"].append(c)
        elif a["role"] == "validation":
            out["F3"].append(c)
    out = {f: sorted(v) for f, v in out.items()}
    if out != EXPECTED_CASES:
        raise FoldError(f"fold case sets differ from the protocol: { {f: (v[0], v[-1], len(v)) for f, v in out.items()} }")
    return out


def build(D, held_out):
    """The fold data object for one held-out fold."""
    if held_out not in FOLDS:
        raise FoldError(f"held-out fold must be one of {FOLDS}")
    t = D.table
    blk = window_block(D)
    keep = (blk != "") | (t.role == "calibration")
    if (t.role[keep] == "test").any() or (t.T[keep, -1] >= D.boundaries[1]).any():
        raise FoldError("a test-period window would enter the fold")
    rows = np.nonzero(keep)[0]
    role = np.where(blk[rows] == held_out, "validation", np.where(blk[rows] != "", "fit", "calibration")).astype(t.role.dtype)
    table = ms.WindowTable(t.start[rows], t.T[rows], t.period[rows], role, t.category[rows], t.case_index[rows], t.fault_class[rows],
                           t.ambiguous[rows], t.exclusion_reason[rows], t.match[rows])
    fd = copy.copy(D)
    fd.table = table
    fd.fold = {"held_out": held_out, "training": [f for f in FOLDS if f != held_out], "block_of_row": blk[rows], "blocks": blocks(D),
               "cases": fold_cases(D), "source_rows": rows}
    return fd


def region_ok(fd, rows):
    """Fold version of the frozen region check: each window lies wholly inside the block(s) of its relabelled role."""
    t, f = fd.table, fd.fold
    rows = np.asarray(rows, np.int64)
    t0, t6, role = t.T[rows, 0], t.T[rows, -1], t.role[rows]
    ok = np.zeros(len(rows), bool)
    lo, hi = f["blocks"][f["held_out"]]
    ok |= (role == "validation") & (t0 >= lo) & (t6 < hi)
    for b in f["training"]:
        lo, hi = f["blocks"][b]
        ok |= (role == "fit") & (t0 >= lo) & (t6 < hi)
    cal = role == "calibration"
    ok[cal] = (t0[cal] >= fd.cfg.tc) & (t6[cal] < fd.boundaries[0])
    return bool(ok.all())


@contextlib.contextmanager
def fold_regions():
    """Inside the block the frozen datasets use the fold region check for fold data objects (and the original one otherwise)."""
    original = md.m1_split_region_ok

    def check(data, rows):
        return region_ok(data, rows) if hasattr(data, "fold") else original(data, rows)

    md.m1_split_region_ok = check
    try:
        yield
    finally:
        md.m1_split_region_ok = original


def heldout_labels(fd):
    """Incident labels of the held-out fold: the held-out cases' documented timestamps, sorted (as scoring_sets does for a period)."""
    return sorted(int(fd.cases["timestamp"][c]) for c in fd.fold["cases"][fd.fold["held_out"]])


def s2_weights(fd):
    """v1's case-balanced class weights from the training folds' eligible event windows (the frozen class, its own constructor)."""
    from model1_transformer.weighting import CaseBalancedWeights
    rows = md.objective_rows(fd, "s2_fault_finetune")
    t = fd.table
    if not (t.role[rows] == "fit").all():
        raise FoldError("an S2 window outside the training folds")
    return CaseBalancedWeights(t.case_index[rows], t.fault_class[rows])
