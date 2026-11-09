"""Fair case-level validation interface for the Transformer fault classifier (Linear 298-51).

Protocol (design note 7.1 and the XGBoost README "fair comparison rule"): the approved non-ambiguous `event_associated`
windows of each of the 25 validation cases -> one probability vector per window -> mean over the case's windows (once per
case, every window of a case equal) -> one predicted class per case (argmax; a tie goes to the lowest class id). Class
columns keep the five-class order 0 Container Hardware, 1 Container Network, 2 Node CPU, 3 Node Disk, 4 Node Memory.

`check_alignment` requires each case's telemetry-minute union (first sample, last sample, number of distinct one-minute
samples, contiguity) to equal the frozen XGBoost provenance (`observation_timing.json`, `case_features_provenance.json`)
exactly, so both models classify the same cases from the same observations.

Only the 'validation' role is accepted anywhere here; the held-out set is locked. This milestone implements and tests the
interface on synthetic predictions and on window METADATA; no model is run on real validation windows yet.
"""
import hashlib
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.abspath(os.path.join(HERE, "..", "model1_xgboost_control", "results", "xgb_control_v1"))
CLASS_NAMES = ("Container Hardware", "Container Network", "Node CPU", "Node Disk", "Node Memory")
STEP_S = 60


class RoleLocked(PermissionError):
    """Only validation windows may be scored here; calibration and the held-out set are locked."""


class AlignmentError(ValueError):
    """The case windows or their telemetry minutes differ from the frozen XGBoost provenance."""


def array_sha256(a):
    """Same formula as the XGBoost control's provenance (dtype, shape, bytes)."""
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def load_frozen(frozen_dir=FROZEN_DIR):
    with open(os.path.join(frozen_dir, "observation_timing.json")) as fh:
        timing = json.load(fh)
    with open(os.path.join(frozen_dir, "case_features_provenance.json")) as fh:
        prov = json.load(fh)
    return {"timing": {c["case_index"]: c for c in timing["per_case"] if c["role"] == "validation"}, "provenance": prov}


def window_metadata_from_table(data):
    """Validation windows of the approved `validation_fault_eval` selection, from the window TABLE only (no telemetry is read)."""
    from prep import m1_dataset as md
    rows = md.objective_rows(data, "validation_fault_eval")
    t = data.table
    if not (t.role[rows] == "validation").all() or not (t.category[rows] == "event_associated").all():
        raise RoleLocked("the validation_fault_eval selection holds a window outside validation/event_associated")
    if any(t.exclusion_reason[r] is not None or bool(t.ambiguous[r]) for r in rows):
        raise AlignmentError("the validation selection holds an ambiguous-label window")
    return {"case_index": t.case_index[rows].astype(np.int64), "start_index": t.start[rows].astype(np.int64), "timestamps": t.T[rows].astype(np.int64),
            "fault_class": t.fault_class[rows].astype(np.int64)}


def predict_window_probabilities(model, loader, device="cpu", role="validation"):
    """Softmax probabilities (float64, (W,5)) for every window of `loader`. Eval mode, no gradient, training mode restored."""
    if role != "validation":
        raise RoleLocked(f"role {role!r} is locked: only 'validation' windows may be scored in this milestone")
    dev = torch.device(device)
    was_training = model.training
    model.eval()
    out = {"case_index": [], "start_index": [], "timestamps": [], "fault_class": [], "probs": []}
    with torch.no_grad():
        for b in loader:
            if set(b["role"]) != {"validation"} or set(b["category"]) != {"event_associated"}:
                raise RoleLocked(f"batch holds roles {sorted(set(b['role']))} / categories {sorted(set(b['category']))}; expected validation event_associated only")
            if b["fault_label_ambiguous"].any() or not b["fault_class_present"].all():
                raise AlignmentError("an ambiguous or unlabelled window reached the validation scorer")
            logits = model(b["inputs"].to(dev), b["input_mask"].to(dev), applicable_entry=b["applicable_entry"].to(dev))["logits"]
            out["probs"].append(logits.double().softmax(-1).cpu().numpy())
            for k in ("case_index", "start_index", "timestamps", "fault_class"):
                out[k].append(b[k].numpy())
    model.train(was_training)
    if not out["probs"]:
        raise AlignmentError("no validation window was scored")
    return {k: np.concatenate(v) for k, v in out.items()}


def aggregate_cases(case_index, probs, fault_class=None, n_classes=5):
    """Mean probability once per case -> predicted class. Output rows are sorted by case_index whatever the input order."""
    case_index, probs = np.asarray(case_index, np.int64), np.asarray(probs, np.float64)
    if probs.ndim != 2 or probs.shape != (len(case_index), n_classes) or len(case_index) == 0:
        raise AlignmentError(f"probs must have shape (windows, {n_classes}) matching case_index, got {probs.shape}")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1 + 1e-9).any() or not np.allclose(probs.sum(1), 1.0, atol=1e-6):
        raise AlignmentError("window probabilities must be finite, in [0, 1] and sum to 1")
    cases = np.unique(case_index)
    mean = np.stack([probs[case_index == c].mean(0) for c in cases])
    out = {"case_index": cases, "probs": mean, "pred_class": mean.argmax(1).astype(np.int64), "n_windows": np.array([(case_index == c).sum() for c in cases], np.int64)}
    if fault_class is not None:
        fc = np.asarray(fault_class, np.int64)
        y = []
        for c in cases:
            u = np.unique(fc[case_index == c])
            if len(u) != 1:
                raise AlignmentError(f"case {c} has windows with several labels {u.tolist()}")
            y.append(u[0])
        out["true_class"] = np.array(y, np.int64)
    return out


def minute_unions(case_index, timestamps):
    """Per case (sorted): first sample, last sample, number of distinct one-minute samples, contiguity. timestamps (W,7)."""
    case_index, timestamps = np.asarray(case_index, np.int64), np.asarray(timestamps, np.int64)
    rows = []
    for c in np.unique(case_index):
        u = np.unique(timestamps[case_index == c])
        rows.append({"case_index": int(c), "first_sample_ts": int(u[0]), "last_sample_ts": int(u[-1]), "n_samples": int(len(u)),
                     "contiguous": bool((np.diff(u) == STEP_S).all())})
    return rows


def check_alignment(case_index, timestamps, fault_class, frozen, case_timestamps=None):
    """Raise AlignmentError unless the cases, labels and telemetry-minute unions equal the frozen XGBoost provenance exactly.

    `case_timestamps` (optional {case_index: documented case time}) is compared with the frozen `case_ts` when given.
    Returns a report with the per-case union digest.
    """
    prov, timing = frozen["provenance"], frozen["timing"]
    case_index, fault_class = np.asarray(case_index, np.int64), np.asarray(fault_class, np.int64)
    bad = []
    cases = np.unique(case_index)
    if cases.tolist() != prov["validation_case_index"]:
        bad.append(f"validation cases {cases.tolist()} differ from the frozen {prov['validation_case_index']}")
    else:
        if array_sha256(cases) != prov["array_sha256"]["validation_case_index"]:
            bad.append("validation_case_index array hash differs from the frozen provenance")
        y = np.array([np.unique(fault_class[case_index == c])[0] for c in cases], np.int64)
        if array_sha256(y) != prov["array_sha256"]["y_validation"]:
            bad.append("validation labels differ from the frozen provenance (y_validation hash)")
        dist = np.bincount(y, minlength=5).tolist()
        if dist != prov["class_distribution"]["validation"]:
            bad.append(f"class distribution {dist} != frozen {prov['class_distribution']['validation']}")
    if tuple(prov["class_order"]) != CLASS_NAMES:
        bad.append(f"frozen class order {prov['class_order']} != {list(CLASS_NAMES)}")
    unions = minute_unions(case_index, timestamps)
    nwin = {int(c): int((case_index == c).sum()) for c in cases}
    for u in unions:
        f = timing.get(u["case_index"])
        if f is None:
            bad.append(f"case {u['case_index']} is not a frozen validation case")
            continue
        for key in ("first_sample_ts", "last_sample_ts", "n_samples"):
            if u[key] != f[key]:
                bad.append(f"case {u['case_index']}: {key} {u[key]} != frozen {f[key]}")
        if not u["contiguous"]:
            bad.append(f"case {u['case_index']}: telemetry minutes are not consecutive one-minute samples")
        lo, hi = prov["windows_per_case"]["validation"]
        if not lo <= nwin[u["case_index"]] <= hi:
            bad.append(f"case {u['case_index']}: {nwin[u['case_index']]} windows outside the frozen range {[lo, hi]}")
        if case_timestamps is not None and int(case_timestamps[u["case_index"]]) != f["case_ts"]:
            bad.append(f"case {u['case_index']}: documented case time differs from the frozen one")
    if bad:
        raise AlignmentError("; ".join(bad[:8]) + (f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""))
    digest = hashlib.sha256(json.dumps(unions, sort_keys=True).encode()).hexdigest()
    return {"cases": len(cases), "windows": int(len(case_index)), "minute_union_sha256": digest, "validation_case_index_sha256": array_sha256(cases)}


def xgb_style_rows(agg, group_id=None, names=CLASS_NAMES):
    """Rows with the same fields as the XGBoost control's per-case prediction CSV, so one metrics code scores both models."""
    rows = []
    for i, c in enumerate(agg["case_index"]):
        r = {"case_index": int(c), "group_id": -1 if group_id is None else int(group_id[i]), "true_class": int(agg["true_class"][i]),
             "true_name": names[int(agg["true_class"][i])], "predicted_class": int(agg["pred_class"][i]), "predicted_name": names[int(agg["pred_class"][i])]}
        r.update({f"p_{k}": float(p) for k, p in enumerate(agg["probs"][i])})
        rows.append(r)
    return rows
