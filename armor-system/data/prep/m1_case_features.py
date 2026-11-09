"""M1-03 follow-up (Linear 298-49): one deterministic feature vector per documented case, for the XGBoost control.

Uses only approved `event_associated` windows (the ambiguous-label window is excluded) of ONE role at a time
("fit" or "validation"). The calibration role and the held-out evaluation role are refused. Nothing is learned here:
the values come from `Model1Data.Xt`, which is already scaled with the fit-only scaler and masked.

Per case: the distinct one-minute samples touched by the case's windows (neighbouring windows share 6 of their 7
minutes, so each minute is used once). For every node group (VM, pod, taken from the declared node types, never from
node position) and every channel applicable to that group, four numbers over those minutes:
  mean    mean over time of the group average (average over the group's applicable nodes only)
  std     standard deviation over time (ddof=0) of that group average
  max_abs largest |value| over the group's applicable nodes and all minutes
  delta   group average at the last minute minus the first minute
Feature order: group, channel, statistic -> 2 groups x (57 + 73) channels x 4 statistics = 520 for D1.
"""
import numpy as np

from . import m1_masks as mm

ALLOWED_ROLES = ("fit", "validation")
STATS = ("mean", "std", "max_abs", "delta")
APPROVED_CASES = {"fit": 88, "validation": 25}
APPROVED_FEATURES = 520


class RoleLocked(PermissionError):
    """The requested partition may not be used for XGBoost features in this milestone."""


def group_layout(schema, applicable, rule=mm.D1_MASK_RULE):
    """[(group_name, node_indices, channel_indices)] with channels applicable to at least one node of the group."""
    _, node_type = mm.declared_entry_mask(schema, rule)
    applicable = mm.validate_bool_mask(applicable, (len(node_type), applicable.shape[1]), "applicable_entry")
    out = []
    for k, t in enumerate(rule["node_types"]):
        nodes = np.nonzero(node_type == k)[0]
        chans = np.nonzero(applicable[nodes].any(0))[0]
        if len(nodes) == 0 or len(chans) == 0:
            raise mm.MaskError(f"node group {t['name']!r} has no nodes or no applicable channels")
        out.append((t["name"], nodes, chans))
    return out


def feature_names(schema, layout):
    chan = [c["name"] for c in schema["channel_order"]]
    return [f"{g}|{chan[c]}|{s}" for g, _, chans in layout for c in chans for s in STATS]


def case_vector(Xt, minute_idx, applicable, layout):
    """Feature vector (n_features,) float64 for one case from the sorted distinct sample indices `minute_idx`."""
    minute_idx = np.asarray(minute_idx)
    if minute_idx.ndim != 1 or len(minute_idx) < 2 or (np.diff(minute_idx) <= 0).any():
        raise ValueError("minute_idx must be strictly increasing (distinct minutes) and hold at least two minutes")
    X = np.asarray(Xt)[minute_idx].astype(np.float64)                     # (m, nodes, channels)
    feats = []
    for _, nodes, chans in layout:
        app = applicable[np.ix_(nodes, chans)]                            # (n, c)
        sub = X[:, nodes][:, :, chans] * app[None]                        # masked entries contribute 0
        avg = sub.sum(1) / app.sum(0)[None]                               # (m, c): average over applicable nodes
        peak = np.abs(sub).max(axis=(0, 1))                               # (c,)
        stats = np.stack([avg.mean(0), avg.std(0), peak, avg[-1] - avg[0]], axis=1)      # (c, 4)
        feats.append(stats.reshape(-1))
    return np.concatenate(feats)


def case_minutes(table, rows, n_steps=7):
    """Distinct global sample indices (sorted) covered by the windows in `rows` (windows are consecutive minutes)."""
    idx = set()
    for r in rows:
        s = int(table.start[r])
        idx.update(range(s, s + n_steps))
    return np.array(sorted(idx), dtype=np.int64)


def build_case_matrix(data, role):
    """Case-level matrix for 'fit' or 'validation' from a loaded `Model1Data`.

    Returns a dict: X (n, 520) float32, y (n,) int64 in Model 1 class order, case_index, group_id, n_windows, n_minutes,
    feature_names, role, layout. Raises RoleLocked for any other role (calibration, held-out evaluation, embargo).
    """
    if role not in ALLOWED_ROLES:
        raise RoleLocked(f"role {role!r} is not available for XGBoost features; allowed: {ALLOWED_ROLES}")
    t = data.table
    rows = data.selection([role], ["event_associated"], drop_ambiguous=True)
    if not (t.role[rows] == role).all():
        raise RuntimeError("selected windows do not all belong to the requested role")
    by = {}
    for r in rows:
        by.setdefault(int(t.case_index[r]), []).append(int(r))
    if -1 in by:
        raise RuntimeError("a window without a case label survived the ambiguity filter")
    expected = [a["case_index"] for a in data.manifest["cases"]["assignments"] if a["role"] == role]
    if sorted(by) != expected:
        raise RuntimeError(f"cases with approved windows {len(by)} differ from the manifest's {len(expected)} cases for role {role!r}")
    layout = group_layout(data.schema, data.applicable)
    cases = sorted(by)
    X = np.stack([case_vector(data.Xt, case_minutes(t, by[c]), data.applicable, layout) for c in cases]).astype(np.float32)
    y = data.cases["class_id"][cases].astype(np.int64)
    return {"X": X, "y": y, "case_index": np.array(cases, np.int64), "group_id": data.cases["group_id"][cases].astype(np.int64),
            "n_windows": np.array([len(by[c]) for c in cases], np.int64), "n_minutes": np.array([len(case_minutes(t, by[c])) for c in cases], np.int64),
            "feature_names": feature_names(data.schema, layout), "role": role, "layout": layout}


def inverse_frequency_weights(y_fit, n_classes=5):
    """Per-class weights N / (K * n_k) from the FITTING labels only. A class with no fitting case is an error."""
    y_fit = np.asarray(y_fit)
    counts = np.bincount(y_fit, minlength=n_classes).astype(np.float64)
    if len(counts) != n_classes or (counts == 0).any():
        raise ValueError(f"every class needs at least one fitting case, got counts {counts.tolist()}")
    return len(y_fit) / (n_classes * counts)
