"""M1-03 fit / calibration split, Policy B window table, fit-only scaler and the versioned manifest.

Implements design note sections 5.3, 5.4 and 5.6 exactly (approved for Model 1 v0, 2026-10-27). Nothing here is
random. The approved outer 101/25/84 split and the existing scaler files are only read, never written.

Roles of a 7-step window (6 input minutes t0..t5, target minute t6):
  fit          t6 <  Tc - EMBARGO            (inside the approved train period)
  embargo      neither fit nor calibration   (never used)
  calibration  t0 >= Tc                      (inside the approved train period)
  validation   wholly inside [B1, B2)        (approved outer split)
  test         wholly inside [B2, inf)       (fixed structural checks only)
Categories (Policy B): event_associated / normal_reference / ignore, see `compute_split`.
"""
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass

import numpy as np

from . import preprocess as pp
from .splits import BEFORE, AFTER, CLASSES, content_hash as manifest_content_hash

MANIFEST_NAME = "model1_calibration_split_v1"
SCALER_NAME = "scaler_model1_fit_only_v1"
MANIFEST_VERSION = "1.0"

TC = 1651637521                       # calibration start, 2022-05-04T04:12:01Z (approved)
EMBARGO_S = 360                       # one window of history (approved)
PRE_GUARD_S, POST_GUARD_S = 120, 300  # Policy B (approved default)
WINDOW, HORIZON, STEP_S = 6, 1, 60
ROLES = ("fit", "embargo", "calibration", "validation", "test")
CATEGORIES = ("normal_reference", "event_associated", "ignore")
REASON_OVERLAP = "overlapping_cases_different_classes"      # t5 inside case windows of two different fault classes
EXCLUSION_REASONS = (REASON_OVERLAP,)                          # every documented fault-label exclusion reason
PERIODS = ("train", "validation", "test")
BIG = 10 ** 18

APPROVED_OUTER_SHA256 = "32ff03fcffca09f3d3953777a020514b4af9c882bc85e3dc1693e282607638ac"
APPROVED_PREPROCESSING_CONFIG_SHA256 = "e9aa3ccc776a8bc9940d151ba3feea3c79f502eff58e12c9946babe0aaefeb8c"

# Design note 5.4 / 5.6 expectations. Any difference raises ApprovedCountsError.
APPROVED_COUNTS = {
    "period_windows": {"train": 3427, "validation": 869, "test": 2868},
    "roles": {
        "fit": {"windows": 2868, "event_associated": 524, "normal_reference": 1312, "ignore": 1032, "cases": 88, "cases_by_class": [33, 29, 9, 13, 4]},
        "embargo": {"windows": 0, "event_associated": 0, "normal_reference": 0, "ignore": 0},
        "calibration": {"windows": 559, "event_associated": 78, "normal_reference": 329, "ignore": 152, "cases": 13, "cases_by_class": [4, 3, 1, 5, 0]},
        "validation": {"windows": 869, "event_associated": 150, "normal_reference": 411, "ignore": 308, "cases": 25, "cases_by_class": [10, 6, 3, 5, 1]},
        "test": {"windows": 2868, "event_associated": 501, "cases": 84, "cases_by_class": [36, 22, 8, 14, 4]},
    },
    "fit_samples": 2880, "calibration_region_samples": 565,
    "train_period_normal_reference": 1641, "train_period_event_associated": 602, "train_period_ignore": 1184,
    "cases_with_event_windows": {"fit": 88, "calibration": 13, "validation": 25, "test": 84},
}

# Which window categories each learning objective may use (design note 6 and 5.6). `gradient` objectives may
# only ever see fitting windows: calibration, validation and test observations never update weights.
OBJECTIVES = {
    "s1_pretrain": {"roles": ("fit",), "categories": ("normal_reference",), "gradient": True, "drop_ambiguous": False,
                    "note": "self-supervised next-step masked MSE; never sees event_associated or ignore windows"},
    "s2_fault_finetune": {"roles": ("fit",), "categories": ("event_associated",), "gradient": True, "drop_ambiguous": True,
                          "note": "5-class fault CE; weak labels (documented fault window, not minute-level truth); the 1 train window inside two different-class case windows is excluded"},
    "s1_early_stopping": {"roles": ("validation",), "categories": ("normal_reference",), "gradient": False, "drop_ambiguous": False,
                          "note": "validation masked MSE used for early stopping only; no weight update"},
    "s3_calibration_reference": {"roles": ("calibration",), "categories": ("normal_reference",), "gradient": False, "drop_ambiguous": False,
                                 "note": "no gradient: median, IQR, epsilon, thresholds of design note 4.5"},
    "calibration_diagnostics": {"roles": ("calibration",), "categories": ("event_associated",), "gradient": False, "drop_ambiguous": False,
                                "note": "diagnostics only; never used for fine-tuning"},
    "validation_fault_eval": {"roles": ("validation",), "categories": ("event_associated",), "gradient": False, "drop_ambiguous": True,
                              "note": "validation fault evaluation with case-level aggregation"},
    "validation_all": {"roles": ("validation",), "categories": CATEGORIES, "gradient": False, "drop_ambiguous": False,
                       "note": "validation detection and sensitivity analyses; ignore windows are not negatives"},
    "test_structural": {"roles": ("test",), "categories": CATEGORIES, "gradient": False, "drop_ambiguous": False,
                        "note": "fixed structural integrity checks only; no tuning, no threshold or model selection"},
}


class ManifestInvariantError(RuntimeError):
    """A leakage or structure invariant failed."""


class ApprovedCountsError(RuntimeError):
    """A count differs from the approved design (sections 5.4 and 5.6)."""


@dataclass(frozen=True)
class SplitConfig:
    tc: int = TC
    embargo_s: int = EMBARGO_S
    pre_guard_s: int = PRE_GUARD_S
    post_guard_s: int = POST_GUARD_S
    case_before_s: int = BEFORE
    case_after_s: int = AFTER
    window: int = WINDOW
    horizon: int = HORIZON
    step_s: int = STEP_S

    @property
    def n_steps(self):
        return self.window + self.horizon


@dataclass
class WindowTable:
    start: np.ndarray        # global index of the first input minute
    T: np.ndarray            # (n, n_steps) int64 timestamps
    period: np.ndarray       # str: train / validation / test
    role: np.ndarray         # str: fit / embargo / calibration / validation / test
    category: np.ndarray     # str: normal_reference / event_associated / ignore
    case_index: np.ndarray   # int64, -1 when none or ambiguous
    fault_class: np.ndarray  # int64 0..4, -1 when none or ambiguous
    ambiguous: np.ndarray    # bool: t5 inside two case windows of different classes
    exclusion_reason: np.ndarray  # object: None, or why fault_class is -1 although the window is event_associated
    match: np.ndarray        # (n, n_cases) bool: t5 inside that case's window


# ---------------------------------------------------------------------------------------------------------
# approved inputs
# ---------------------------------------------------------------------------------------------------------
def load_case_table(rows):
    """Parse `case_split_assignments.csv` rows (list of dicts) into arrays. Class ids follow `splits.CLASSES`."""
    try:
        cls = np.array([CLASSES.index(r["category"]) for r in rows], np.int64)
    except ValueError as e:
        raise ManifestInvariantError(f"unknown fault category in case table: {e}") from None
    t = {"timestamp": np.array([int(r["timestamp"]) for r in rows], np.int64), "class_id": cls,
         "group_id": np.array([int(r["group_id"]) for r in rows], np.int64),
         "outer_split": np.array([r["model1_split"] for r in rows]), "case_index": np.array([int(r["case_index"]) for r in rows], np.int64)}
    if (t["case_index"] != np.arange(len(rows))).any() or (np.diff(t["timestamp"]) <= 0).any():
        raise ManifestInvariantError("case table must be in strict chronological order with case_index 0..n-1")
    return t


def raw_to_model_map(schema):
    """Raw category ids {0,1,3,4,5} -> consecutive model ids 0..4 via `splits.CLASSES`."""
    m = {int(k): CLASSES.index(v) for k, v in schema["category_ids"].items()}
    if sorted(m.values()) != list(range(len(CLASSES))):
        raise ManifestInvariantError(f"raw category ids {m} do not map onto 0..{len(CLASSES) - 1}")
    return m


def approved_boundaries(outer, case_table):
    """(B1, B2) from the approved outer manifest, after checking hash, status and case ranges."""
    if outer.get("content_sha256") != APPROVED_OUTER_SHA256 or manifest_content_hash(outer) != APPROVED_OUTER_SHA256:
        raise ManifestInvariantError("approved outer split manifest hash does not match the approved value")
    if outer.get("status") != "approved" or not outer.get("usable"):
        raise ManifestInvariantError("outer split manifest is not approved/usable")
    names = list(outer["case_ranges"])
    if names != list(PERIODS):
        raise ManifestInvariantError(f"outer split names {names} != {list(PERIODS)}")
    for n in names:
        r = outer["case_ranges"][n]
        got = np.nonzero(case_table["outer_split"] == n)[0]
        if got.tolist() != list(range(r["start_index"], r["end_index_exclusive"])):
            raise ManifestInvariantError(f"case table split '{n}' disagrees with the approved manifest ranges")
    b = [x["boundary_ts"] for x in outer["sample_time_boundaries"]]
    if len(b) != 2:
        raise ManifestInvariantError("expected two sample-time boundaries")
    return int(b[0]), int(b[1])


# ---------------------------------------------------------------------------------------------------------
# window table (pure function of timestamps and cases)
# ---------------------------------------------------------------------------------------------------------
def period_bounds(boundaries):
    b1, b2 = boundaries
    return {"train": (-BIG, b1), "validation": (b1, b2), "test": (b2, BIG)}


def _starts(ts, lo, hi, cfg):
    pos = np.nonzero((ts >= lo) & (ts < hi))[0]
    allw = pp.make_windows(ts, pos, cfg.window, cfg.horizon, cfg.step_s)
    kept, dropped = pp.protect_boundaries(ts, allw, lo, hi, cfg.window, cfg.horizon, cfg.step_s)
    if len(dropped):
        raise ManifestInvariantError(f"{len(dropped)} windows cross a period boundary")
    return kept


def compute_split(ts, cases, boundaries, cfg=SplitConfig()):
    """Policy-B window table for the whole timeline.

    event_associated: t5 in [t_i - 59, t_i + 300) for some case i.
    normal_reference: all minutes outside every guard [t_i - 59 - pre, t_i + 300 + post) and not event_associated.
    ignore: everything else (never a confirmed negative).
    """
    ts = np.asarray(ts, np.int64)
    ct = np.asarray(cases["timestamp"], np.int64)
    ncase = len(ct)
    parts = []
    for name, (lo, hi) in period_bounds(boundaries).items():
        st = _starts(ts, lo, hi, cfg)
        parts.append((name, st))
    start = np.concatenate([st for _, st in parts])
    period = np.concatenate([np.full(len(st), n) for n, st in parts])
    T = ts[start[:, None] + np.arange(cfg.n_steps)]
    t0, t5, t6 = T[:, 0], T[:, cfg.window - 1], T[:, -1]
    match = np.zeros((len(start), ncase), bool)
    guard = np.zeros(len(start), bool)
    for i, c in enumerate(ct):
        match[:, i] = (t5 >= c - cfg.case_before_s) & (t5 < c + cfg.case_after_s)
        guard |= ((T >= c - cfg.case_before_s - cfg.pre_guard_s) & (T < c + cfg.case_after_s + cfg.post_guard_s)).any(1)
    ea = match.any(1)
    nr = ~guard & ~ea
    category = np.where(ea, "event_associated", np.where(nr, "normal_reference", "ignore"))
    cls = np.asarray(cases["class_id"], np.int64)
    last = ncase - 1 - np.argmax(match[:, ::-1], axis=1)
    cls_min = np.where(match, cls[None], 99).min(1)
    cls_max = np.where(match, cls[None], -1).max(1)
    ambiguous = ea & (cls_min != cls_max)
    case_index = np.where(ea & ~ambiguous, last, -1).astype(np.int64)
    fault_class = np.where(ea & ~ambiguous, cls[np.clip(last, 0, ncase - 1)], -1).astype(np.int64)
    fit_cut = cfg.tc - cfg.embargo_s
    role = np.where(period == "validation", "validation", np.where(period == "test", "test", "embargo"))
    role = np.where((period == "train") & (t6 < fit_cut), "fit", role)
    role = np.where((period == "train") & (t0 >= cfg.tc), "calibration", role)
    reason = np.full(len(start), None, dtype=object)
    reason[ambiguous] = REASON_OVERLAP
    return WindowTable(start, T, period, role, category, case_index, fault_class, ambiguous, reason, match)


def region_ok(table, boundaries, cfg=SplitConfig()):
    """(n,) bool: every window lies wholly inside the time region of its role."""
    b1, b2 = boundaries
    t0, t6 = table.T[:, 0], table.T[:, -1]
    lo = {"fit": -BIG, "calibration": cfg.tc, "validation": b1, "test": b2}
    hi = {"fit": cfg.tc - cfg.embargo_s, "calibration": b1, "validation": b2, "test": BIG}
    ok = np.ones(len(t0), bool)
    for r in lo:
        sel = table.role == r
        ok[sel] = (t0[sel] >= lo[r]) & (t6[sel] < hi[r])
    return ok


def case_roles(table, cases):
    """Role per case: train cases by the side holding their event windows, others by the outer split."""
    n = len(cases["timestamp"])
    out = np.array(["none"] * n, dtype=object)
    for r in ("fit", "calibration"):
        out[table.match[table.role == r].any(0) & (cases["outer_split"] == "train")] = r
    both = table.match[table.role == "fit"].any(0) & table.match[table.role == "calibration"].any(0)
    for r in ("validation", "test"):
        out[cases["outer_split"] == r] = r
    return out, both


def leakage_checks(table, cases, boundaries, cfg=SplitConfig()):
    """List of {id, ok, detail}. All must be ok; `assert_leakage_free` raises otherwise."""
    checks = []

    def add(cid, ok, detail):
        checks.append({"id": cid, "ok": bool(ok), "detail": detail})

    fit, cal = table.role == "fit", table.role == "calibration"
    shared = np.intersect1d(np.unique(table.T[fit]), np.unique(table.T[cal]))
    add("L1_no_shared_timestamp_fit_calibration", len(shared) == 0, f"{len(shared)} timestamps shared between fit and calibration windows")
    roles, both = case_roles(table, cases)
    add("L2_no_case_with_event_windows_on_both_sides", not both.any(), f"{int(both.sum())} cases have event windows in fit and calibration")
    train = cases["outer_split"] == "train"
    gid = cases["group_id"]
    bad_groups = sorted(int(g) for g in np.unique(gid[train]) if len({roles[i] for i in np.nonzero(train & (gid == g))[0]} - {"none"}) > 1)
    add("L3_no_incident_group_on_both_sides", not bad_groups, f"{len(bad_groups)} incident groups span fit and calibration: {bad_groups}")
    uncovered = [int(i) for i in np.nonzero(train & (roles == "none"))[0]]
    add("L4_every_train_case_on_exactly_one_side", not uncovered and not both.any(), f"train cases with no event window in fit or calibration: {uncovered}")
    ok = region_ok(table, boundaries, cfg)
    add("L5_no_window_crosses_a_region_boundary", ok[table.role != "embargo"].all(), f"{int((~ok[table.role != 'embargo']).sum())} windows reach outside their region")
    stray = int(((table.role == "fit") | (table.role == "calibration"))[table.period != "train"].sum()) + int(((table.role == "validation") & (table.period != "validation")).sum()) + int(((table.role == "test") & (table.period != "test")).sum())
    add("L6_roles_stay_inside_their_outer_period", stray == 0, f"{stray} windows have a role outside their approved period")
    ct = cases["timestamp"]
    order = np.nonzero(train)[0]
    k = int(np.searchsorted(ct[order], cfg.tc))
    admissible = 0 < k < len(order) and gid[order[k - 1]] != gid[order[k]]
    add("L7_cutoff_lies_between_cases_of_different_groups", admissible,
        "Tc lies between train cases %s and %s" % ((int(order[k - 1]), int(order[k])) if 0 < k < len(order) else ("-", "-")) + ("" if admissible else " that share a group (or Tc is outside the train cases)"))
    emb = table.role == "embargo"
    wrong = int((emb & ~((table.period == "train") & (table.T[:, -1] >= cfg.tc - cfg.embargo_s) & (table.T[:, 0] < cfg.tc))).sum())
    add("L8_embargo_windows_are_unused", wrong == 0, f"{int(emb.sum())} embargo windows, in no role; {wrong} of them are not between the fit and calibration definitions")
    ea = table.category == "event_associated"
    has_reason = np.array([r is not None for r in table.exclusion_reason], bool)
    unexplained = int((ea & (table.fault_class == -1) & ~has_reason).sum())
    add("L9_every_unlabelled_event_window_has_an_exclusion_reason", unexplained == 0, f"{unexplained} event_associated windows have fault_class -1 and no exclusion reason")
    inconsistent = int((has_reason & ~((table.fault_class == -1) & ea)).sum()) + sum(1 for r in table.exclusion_reason if r is not None and r not in EXCLUSION_REASONS)
    add("L10_exclusion_reasons_are_documented_and_consistent", inconsistent == 0, f"{inconsistent} windows carry an undocumented reason or a reason with a fault label")
    return checks


def assert_leakage_free(table, cases, boundaries, cfg=SplitConfig()):
    bad = [c for c in leakage_checks(table, cases, boundaries, cfg) if not c["ok"]]
    if bad:
        raise ManifestInvariantError("; ".join(f"{c['id']}: {c['detail']}" for c in bad))


# ---------------------------------------------------------------------------------------------------------
# counts
# ---------------------------------------------------------------------------------------------------------
def summarize(table, cases, ts, boundaries, cfg=SplitConfig()):
    """Counts in the exact shape of APPROVED_COUNTS (plus extra measured values)."""
    roles, _ = case_roles(table, cases)
    out = {"period_windows": {p: int((table.period == p).sum()) for p in PERIODS}, "roles": {}}
    cls = cases["class_id"]
    for r in ROLES:
        sel = table.role == r
        d = {"windows": int(sel.sum()), **{c: int((sel & (table.category == c)).sum()) for c in CATEGORIES}}
        if r != "embargo":
            idx = np.nonzero(roles == r)[0]
            d["cases"] = int(len(idx))
            d["cases_by_class"] = [int((cls[idx] == k).sum()) for k in range(len(CLASSES))]
            d["cases_with_event_windows"] = int(table.match[sel & (table.category == "event_associated")].any(0).sum())
        out["roles"][r] = d
    fit_cut = cfg.tc - cfg.embargo_s
    out["fit_samples"] = int((ts < fit_cut).sum())
    out["calibration_region_samples"] = int(((ts >= cfg.tc) & (ts < boundaries[0])).sum())
    train = table.period == "train"
    for c in CATEGORIES:
        out[f"train_period_{c}"] = int((train & (table.category == c)).sum())
    nr_fit, nr_cal = out["roles"]["fit"]["normal_reference"], out["roles"]["calibration"]["normal_reference"]
    out["fit_share_of_normal_reference"] = nr_fit / (nr_fit + nr_cal)
    out["cases_with_event_windows"] = {r: out["roles"][r]["cases_with_event_windows"] for r in ("fit", "calibration", "validation", "test")}
    out["ambiguous_windows"] = {p: int((table.ambiguous & (table.period == p)).sum()) for p in PERIODS}
    out["fault_label_exclusions"] = {reason: int(sum(1 for r in table.exclusion_reason if r == reason)) for reason in EXCLUSION_REASONS}
    return out


def check_approved_counts(summary):
    """Return a list of 'name: got != expected' strings (empty = identical to the approved design)."""
    bad = []

    def cmp(name, got, exp):
        if got != exp:
            bad.append(f"{name}: {got} != {exp}")

    for k in ("period_windows", "fit_samples", "calibration_region_samples", "train_period_normal_reference",
              "train_period_event_associated", "train_period_ignore", "cases_with_event_windows"):
        cmp(k, summary[k], APPROVED_COUNTS[k])
    for r, exp in APPROVED_COUNTS["roles"].items():
        for k, v in exp.items():
            cmp(f"roles.{r}.{k}", summary["roles"][r][k], v)
    return bad


def assert_approved_counts(summary):
    bad = check_approved_counts(summary)
    if bad:
        raise ApprovedCountsError("counts differ from the approved design: " + "; ".join(bad))


# ---------------------------------------------------------------------------------------------------------
# fit-only scaler
# ---------------------------------------------------------------------------------------------------------
def fit_only_mask(ts, cfg=SplitConfig()):
    return np.asarray(ts) < cfg.tc - cfg.embargo_s


def fit_scaler_fit_only(X, ts, groups, cfg=SplitConfig()):
    """`preprocess.fit_scaler` on exactly the fitting samples. Rows outside the mask are never read for estimation."""
    mask = fit_only_mask(ts, cfg)
    sc = pp.fit_scaler(X, mask, groups)
    arrays = {"mean": sc["mean"], "scale": sc["scale"], "constant": sc["constant"], "absent_block": sc["absent_block"],
              "absent_entry": sc["absent_entry"], "fit_timestamps": np.asarray(ts, np.int64)[mask],
              "n_fit_samples": np.array(sc["n_fit_samples"], np.int64)}
    return arrays


def scaler_from_arrays(arrays):
    """Scaler dict usable by `preprocess.transform` / `prepare_window`."""
    return {"mean": arrays["mean"], "scale": arrays["scale"], "constant": arrays["constant"], "absent_block": arrays["absent_block"],
            "absent_entry": arrays["absent_entry"], "n_fit_samples": int(arrays["n_fit_samples"])}


def check_scaler_provenance(arrays, ts, cfg=SplitConfig()):
    """Raise ManifestInvariantError unless the scaler was fit on exactly the permitted fitting samples."""
    ft = np.asarray(arrays["fit_timestamps"])
    expect = np.asarray(ts)[fit_only_mask(ts, cfg)]
    if ft.shape != expect.shape or (ft != expect).any():
        raise ManifestInvariantError("scaler fit_timestamps differ from the permitted fitting samples (ts < Tc - EMBARGO)")
    if int(arrays["n_fit_samples"]) != len(expect) or (len(ft) and ft.max() >= cfg.tc - cfg.embargo_s):
        raise ManifestInvariantError("scaler records a fit sample at or after Tc - EMBARGO")
    for k in ("mean", "scale"):
        if not np.isfinite(arrays[k]).all():
            raise ManifestInvariantError(f"scaler {k} is not finite")
    if not (arrays["scale"] > 0).all():
        raise ManifestInvariantError("scaler scale must be strictly positive")


# ---------------------------------------------------------------------------------------------------------
# deterministic files and hashes
# ---------------------------------------------------------------------------------------------------------
def npz_bytes(arrays):
    """Byte-identical NPZ: sorted members, stored (not compressed), fixed timestamps. `np.load` reads it normally."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            b = io.BytesIO()
            a = np.asarray(arrays[name])
            np.lib.format.write_array(b, a if a.flags.c_contiguous else a.copy(order="C"), allow_pickle=False)
            zi = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type, zi.external_attr = zipfile.ZIP_STORED, 0o600 << 16
            zf.writestr(zi, b.getvalue())
    return buf.getvalue()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def array_digest(a):
    a = np.asarray(a)
    a = a if a.flags.c_contiguous else a.copy(order="C")
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def feature_cache_digest(ts, X, chunk=256):
    """Content hash of the cached feature tensors (independent of how the NPZ container was compressed)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(ts, np.int64).tobytes())
    for i in range(0, len(X), chunk):
        h.update(np.ascontiguousarray(X[i:i + chunk]).tobytes())
    return h.hexdigest()


_INT_LIST = re.compile(r"\[\s*-?\d+(?:\s*,\s*-?\d+)*\s*\]")


def dumps_manifest(obj, per_row=16):
    """Stable JSON; long integer lists are wrapped `per_row` values to a line."""
    text = json.dumps(obj, indent=2, ensure_ascii=False)

    def fix(m):
        vals = re.findall(r"-?\d+", m.group(0))
        if len(vals) <= per_row:
            return "[" + ", ".join(vals) + "]"
        line_start = text.rfind("\n", 0, m.start()) + 1
        indent = " " * (len(text[line_start:m.start()]) - len(text[line_start:m.start()].lstrip()) + 2)
        rows = [", ".join(vals[i:i + per_row]) for i in range(0, len(vals), per_row)]
        return "[\n" + ",\n".join(indent + r for r in rows) + "\n" + indent[:-2] + "]"

    return _INT_LIST.sub(fix, text) + "\n"


def starts_digest(starts):
    return array_digest(np.asarray(starts, np.int64))


# ---------------------------------------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------------------------------------
def build_manifest(table, cases, ts, boundaries, *, scaler_arrays, scaler_file_sha256, scaler_file, outer, outer_file,
                   outer_case_csv_sha256, preprocessing_config_sha256, schema, cache_digest, train_period_scaler=None,
                   cfg=SplitConfig(), enforce_approved=True):
    """Assemble the manifest dict (without writing). Raises if an invariant or an approved count fails."""
    checks = leakage_checks(table, cases, boundaries, cfg)
    assert_leakage_free(table, cases, boundaries, cfg)
    summary = summarize(table, cases, ts, boundaries, cfg)
    if enforce_approved:
        assert_approved_counts(summary)
        if outer.get("content_sha256") != APPROVED_OUTER_SHA256:
            raise ApprovedCountsError("outer split hash differs from the approved value")
        if preprocessing_config_sha256 != APPROVED_PREPROCESSING_CONFIG_SHA256:
            raise ApprovedCountsError("preprocessing config hash differs from the approved value")
    check_scaler_provenance(scaler_arrays, ts, cfg)
    roles, _ = case_roles(table, cases)
    n_cases = len(cases["timestamp"])
    windows = {}
    for r in ("fit", "calibration", "validation", "test"):
        windows[r] = {"n_windows": int((table.role == r).sum()), "start_index": {}, "start_index_sha256": {}}
        for c in CATEGORIES:
            s = table.start[(table.role == r) & (table.category == c)]
            windows[r]["start_index"][c] = s.tolist()
            windows[r]["start_index_sha256"][c] = starts_digest(s)
    for r in windows:
        sel = table.role == r
        windows[r]["fault_label_exclusions"] = {reason: [int(s) for s in table.start[sel & np.array([x == reason for x in table.exclusion_reason], bool)]] for reason in EXCLUSION_REASONS}
    case_rows = []
    for i in range(n_cases):
        ev = {r: int((table.match[:, i] & (table.role == r) & (table.category == "event_associated")).sum()) for r in ("fit", "calibration", "validation", "test")}
        case_rows.append({"case_index": i, "timestamp": int(cases["timestamp"][i]), "fault_class": int(cases["class_id"][i]),
                          "class_name": CLASSES[int(cases["class_id"][i])], "group_id": int(cases["group_id"][i]),
                          "outer_split": str(cases["outer_split"][i]), "role": str(roles[i]), "event_windows_by_role": ev})
    role_cases = {r: [int(i) for i in np.nonzero(roles == r)[0]] for r in ("fit", "calibration", "validation", "test")}
    role_groups = {r: sorted({int(cases["group_id"][i]) for i in idx}) for r, idx in role_cases.items()}
    amb = np.nonzero(table.ambiguous)[0]
    prep_cfg = {"tc": cfg.tc, "embargo_s": cfg.embargo_s, "pre_guard_s": cfg.pre_guard_s, "post_guard_s": cfg.post_guard_s}
    sc_new_vs_old = None
    if train_period_scaler is not None:
        sc_new_vs_old = {"absent_block_identical": bool((train_period_scaler["absent_block"] == scaler_arrays["absent_block"]).all()),
                         "constant_entries_train_period_scaler": int(train_period_scaler["constant"].sum()),
                         "constant_entries_fit_only_scaler": int(scaler_arrays["constant"].sum())}
    m = {
        "manifest_version": MANIFEST_VERSION, "name": MANIFEST_NAME, "status": "approved",
        "approval": {"scope": "Model 1 v0 calibration split only; no threshold, model result or open evaluation definition is approved",
                     "approved_on": "2026-10-27", "approved_by": "project owner", "design_note": "armor-system/models/model1_transformer/MODEL1_V0_DESIGN.md sections 5.6 and 12"},
        "description": "Leakage-safe fit/calibration split inside the approved Model 1 train period, Policy B window categories, region and case accounting. Generated by code; do not edit by hand.",
        "protocol": {
            "Tc": cfg.tc, "Tc_utc": _utc(cfg.tc), "embargo_s": cfg.embargo_s, "fit_sample_rule": "ts < Tc - embargo_s",
            "calibration_window_rule": "t0 >= Tc", "fit_window_rule": "t6 < Tc - embargo_s",
            "window": {"inputs": cfg.window, "horizon": cfg.horizon, "step_s": cfg.step_s, "bridges_gaps": False},
            "policy": {"name": "B", "pre_guard_s": cfg.pre_guard_s, "post_guard_s": cfg.post_guard_s,
                       "case_window": f"[t_case - {cfg.case_before_s}, t_case + {cfg.case_after_s})", "event_associated": "t5 inside a case window",
                       "normal_reference": "all 7 minutes outside every guard region and not event_associated", "ignore": "everything else; never a confirmed negative"},
            "ambiguous_event_windows": "t5 inside case windows of two different fault classes: category stays event_associated for the counts, fault_class and case_index are -1, fault_label_exclusion_reason is 'overlapping_cases_different_classes' and the fault objective excludes them",
            "fault_label_exclusion_reasons": list(EXCLUSION_REASONS),
            "outer_period_boundaries": {"train_validation": boundaries[0], "validation_test": boundaries[1]},
            "random_numbers": "none used by the manifest or scaler generation",
        },
        "inputs": {"outer_split_manifest": {"file": outer_file, "name": outer["name"], "content_sha256": outer["content_sha256"]},
                   "case_assignments_csv_sha256": outer_case_csv_sha256, "preprocessing_config_sha256": preprocessing_config_sha256,
                   "channel_order_sha256": schema["channel_order_sha256"], "node_order_sha256": schema["node_order_sha256"],
                   "feature_cache_content_sha256": cache_digest, "n_samples": int(len(ts)),
                   "first_timestamp": int(ts[0]), "last_timestamp": int(ts[-1])},
        "label_map": {"classes": list(CLASSES), "raw_to_model": {str(k): v for k, v in raw_to_model_map(schema).items()}},
        "counts": summary,
        "windows": windows,
        "cases": {"by_role": role_cases, "groups_by_role": role_groups, "assignments": case_rows},
        "ambiguous_windows": [{"start_index": int(table.start[i]), "t5": int(table.T[i, cfg.window - 1]), "case_indices": [int(j) for j in np.nonzero(table.match[i])[0]],
                               "fault_label_exclusion_reason": table.exclusion_reason[i]} for i in amb],
        "fit_only_scaler": {"file": scaler_file, "file_sha256": scaler_file_sha256, "n_fit_samples": int(scaler_arrays["n_fit_samples"]),
                            "first_fit_timestamp": int(scaler_arrays["fit_timestamps"][0]), "last_fit_timestamp": int(scaler_arrays["fit_timestamps"][-1]),
                            "fit_timestamps_sha256": array_digest(scaler_arrays["fit_timestamps"]),
                            "array_sha256": {k: array_digest(scaler_arrays[k]) for k in ("mean", "scale", "constant", "absent_block", "absent_entry")},
                            "structural_absent_blocks": int(scaler_arrays["absent_block"].sum()), "structural_absent_entries": int(scaler_arrays["absent_entry"].sum()),
                            "constant_entries": int(scaler_arrays["constant"].sum()), "comparison_with_train_period_scaler": sc_new_vs_old,
                            "note": "ARMOR's own upstream pre-scaling is undocumented and cannot be undone by this scaler (design note R6)"},
        "objectives": {k: {**{kk: (list(v) if isinstance(v, tuple) else v) for kk, v in o.items()}} for k, o in OBJECTIVES.items()},
        "invariants": checks,
        "generation_parameters": prep_cfg,
    }
    m["content_sha256"] = manifest_content_hash(m)
    return m


def verify_manifest_against_table(manifest, table):
    """Raise ManifestInvariantError unless the manifest's window lists equal a freshly computed table."""
    for r, w in manifest["windows"].items():
        for c in CATEGORIES:
            fresh = table.start[(table.role == r) & (table.category == c)].tolist()
            if fresh != w["start_index"][c]:
                raise ManifestInvariantError(f"manifest windows differ from the data for role={r} category={c}")
        for reason, starts in w.get("fault_label_exclusions", {}).items():
            fresh = table.start[(table.role == r) & np.array([x == reason for x in table.exclusion_reason], bool)].tolist()
            if fresh != starts:
                raise ManifestInvariantError(f"manifest fault-label exclusions differ from the data for role={r} reason={reason}")


def _utc(ts):
    import datetime
    return datetime.datetime.fromtimestamp(int(ts), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
