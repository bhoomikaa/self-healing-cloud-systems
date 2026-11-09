"""Per-window masked next-step prediction errors for anomaly scoring (design note 4.5; Linear 298-51 scoring harness).

Reuses `prep/detection_calibration.node_errors`: for every window and node, the error is the mean over that node's EFFECTIVE
entries (applicable and observed) of (prediction - target)^2 at the target minute, in the scaled space the model works in; a
node with no effective entry is invalid (error 0, node_valid False). The alert time of a window is the timestamp of its target
minute (the seventh sample).

`collect_window_errors` only accepts windows of the roles listed in `SCORABLE_ROLES` (calibration and validation); fitting,
embargo and held-out test windows raise. It runs the model in eval mode without gradients. This module is NOT executed on real
data in the scoring-harness milestone.
"""
import numpy as np
import torch

from prep import detection_calibration as dc

SCORABLE_ROLES = ("calibration", "validation")
TARGET_POSITION = 6                                   # timestamps (W, 7): six inputs, then the target minute


class ScoringRoleError(PermissionError):
    """A batch holds windows that may not be scored here (fitting, embargo or held-out test)."""


def window_node_errors(pred, target, mask):
    """(W, N, C) arrays -> (errors (W, N) float64, node_valid (W, N) bool). Thin wrapper over `node_errors`."""
    return dc.node_errors(pred, target, mask)


def collect_window_errors(model, loader, device="cpu", allowed_roles=SCORABLE_ROLES):
    """Run `model` over `loader` (eval mode, no gradient) and return per-window node errors with the window metadata.

    Returns {"errors": (W, N), "node_valid": (W, N), "target_ts": (W,), "start_index": (W,), "role": (W,), "category": (W,), "case_index": (W,)}
    in loader order. Training mode is restored afterwards.
    """
    bad = set(allowed_roles) - set(SCORABLE_ROLES)
    if bad:
        raise ScoringRoleError(f"roles {sorted(bad)} cannot be scored: only {SCORABLE_ROLES} are allowed")
    dev = torch.device(device)
    was_training = model.training
    model.eval()
    parts = {k: [] for k in ("errors", "node_valid", "target_ts", "start_index", "role", "category", "case_index")}
    with torch.no_grad():
        for b in loader:
            roles = set(b["role"])
            if not roles <= set(allowed_roles):
                raise ScoringRoleError(f"batch holds roles {sorted(roles)}; only {tuple(allowed_roles)} may be scored")
            pred = model(b["inputs"].to(dev), b["input_mask"].to(dev), applicable_entry=b["applicable_entry"].to(dev))["prediction"].cpu().numpy()
            err, nv = window_node_errors(pred, b["target"].numpy(), b["target_mask"].numpy())
            parts["errors"].append(err)
            parts["node_valid"].append(nv)
            parts["target_ts"].append(b["timestamps"][:, TARGET_POSITION].numpy())
            parts["start_index"].append(b["start_index"].numpy())
            parts["case_index"].append(b["case_index"].numpy())
            parts["role"].extend(b["role"])
            parts["category"].extend(b["category"])
    model.train(was_training)
    if not parts["errors"]:
        raise ValueError("no window was scored")
    out = {k: np.concatenate(v) for k, v in parts.items() if k not in ("role", "category")}
    out["role"], out["category"] = np.array(parts["role"]), np.array(parts["category"])
    return out


def alert_timestamps(target_ts, flags):
    """Ascending target-minute timestamps (python ints) of the windows flagged as alerts; the input of `get_pre_interval`.

    `target_ts` must hold distinct timestamps (one scored window per minute); a duplicate is an error, not silently merged.
    """
    ts, fl = np.asarray(target_ts, np.int64), np.asarray(flags, bool)
    if ts.ndim != 1 or fl.shape != ts.shape:
        raise ValueError("target_ts and flags must be 1-D arrays of equal length")
    if len(np.unique(ts)) != len(ts):
        raise ValueError("duplicate target minutes: more than one scored window per minute")
    return [int(t) for t in np.sort(ts[fl])]
