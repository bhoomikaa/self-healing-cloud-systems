"""M1-03 feature masks for Model 1 (design note sections 3.1-3.4 and 12.3).

Three layers (design note 3.2):
  1. `absent_entry`    (46, 130) existing, fit-region derived (`preprocess.fit_scaler`): the node-modality block does not exist.
  2. `applicable_entry`(46, 130) NEW, **declared** from node names and channel names, never inferred from zeros.
  3. `observed_entry`  (T, 46, 130) NEW, dynamic availability from the data source. All True for D1.

`effective = applicable & observed`. A zero value never makes an entry unavailable. Unavailable entries are
replaced by 0 before any model sees them. Every failure here raises; nothing is silently repaired.
"""
import numpy as np

from .preprocess import MODALITIES, N_CHANNELS, N_NODES


class SchemaError(ValueError):
    """schema.json (or an array checked against it) is inconsistent."""


class MaskError(ValueError):
    """A mask is malformed, or the data contradicts a declared mask."""


# A node type is the first entry whose `node_prefix` matches the node name (None = default, must be last).
# `modalities` maps a modality to the channel-name prefix a channel needs to be applicable ("" = every channel).
# A modality missing from the mapping is not used by that node type.
D1_MASK_RULE = {
    "name": "d1_node_type_by_channel_source_v1",
    "node_types": [
        {"name": "vm", "node_prefix": "node-", "modalities": {"metric": "system."}},
        {"name": "pod", "node_prefix": None, "modalities": {"metric": "kpi_container", "trace": "", "log": ""}},
    ],
    # Approved design note section 3.1 Finding 3. Any difference is a failure, not a warning.
    "expected_applicable": {"total": 3152, "vm": {"metric": 342}, "pod": {"metric": 2120, "trace": 360, "log": 330}},
}


def read_layout(schema, n_nodes=N_NODES, n_channels=N_CHANNELS):
    """Validate the schema's node/channel layout and return (node_names, channel_names, channel_modality list)."""
    try:
        shape = tuple(schema["tensor_shape_per_sample"])
        nodes = list(schema["node_order"])
        chans = list(schema["channel_order"])
        ranges = schema["modality_index_ranges"]
    except (KeyError, TypeError) as e:
        raise SchemaError(f"schema is missing a required field: {e}") from None
    if shape != (n_nodes, n_channels):
        raise SchemaError(f"schema tensor shape {shape} != ({n_nodes}, {n_channels})")
    if len(nodes) != n_nodes or len(set(nodes)) != n_nodes:
        raise SchemaError(f"node_order must hold {n_nodes} unique names (got {len(nodes)}, {len(set(nodes))} unique)")
    if len(chans) != n_channels:
        raise SchemaError(f"channel_order has {len(chans)} entries, expected {n_channels}")
    names, modality = [], []
    for i, c in enumerate(chans):
        if not isinstance(c, dict) or c.get("index") != i or "name" not in c or c.get("modality") not in MODALITIES:
            raise SchemaError(f"channel_order[{i}] is malformed: {c!r}")
        names.append(c["name"])
        modality.append(c["modality"])
    if len(set(names)) != n_channels:
        raise SchemaError("channel names are not unique")
    if set(ranges) != set(MODALITIES):
        raise SchemaError(f"modality_index_ranges keys {sorted(ranges)} != {sorted(MODALITIES)}")
    for m in MODALITIES:
        if sorted(ranges[m]) != [i for i, x in enumerate(modality) if x == m]:
            raise SchemaError(f"modality_index_ranges['{m}'] disagrees with channel_order")
    return nodes, names, modality


def declared_entry_mask(schema, rule=D1_MASK_RULE):
    """(46, 130) bool declared by `rule` from node and channel names, plus the (46,) node-type index per node."""
    nodes, names, modality = read_layout(schema)
    types = rule["node_types"]
    if types[-1]["node_prefix"] is not None or any(t["node_prefix"] is None for t in types[:-1]):
        raise MaskError("rule: only the last node type may be the default (node_prefix None)")
    node_type = np.empty(len(nodes), np.int64)
    for i, n in enumerate(nodes):
        node_type[i] = next(k for k, t in enumerate(types) if t["node_prefix"] is None or n.startswith(t["node_prefix"]))
    declared = np.zeros((len(nodes), len(names)), bool)
    for k, t in enumerate(types):
        for m, prefix in t["modalities"].items():
            if m not in MODALITIES:
                raise MaskError(f"rule: unknown modality {m!r}")
            cols = np.array([j for j, (nm, md) in enumerate(zip(names, modality)) if md == m and nm.startswith(prefix)], np.int64)
            if cols.size == 0:
                raise MaskError(f"rule: node type {t['name']!r}, modality {m!r}, prefix {prefix!r} selects no channel (schema mismatch)")
            declared[np.ix_(node_type == k, cols)] = True
    return declared, node_type


def validate_bool_mask(mask, shape, name):
    mask = np.asarray(mask)
    if mask.dtype != np.bool_:
        raise MaskError(f"{name} must be a bool array, got {mask.dtype}")
    if mask.shape != tuple(shape):
        raise MaskError(f"{name} has shape {mask.shape}, expected {tuple(shape)}")
    return mask


def applicable_entry_mask(schema, absent_entry, rule=D1_MASK_RULE):
    """Layer 2: declared rule AND NOT layer-1 absent_entry. Returns (applicable, node_type, report)."""
    declared, node_type = declared_entry_mask(schema, rule)
    absent_entry = validate_bool_mask(absent_entry, declared.shape, "absent_entry")
    applicable = declared & ~absent_entry
    _, _, modality = read_layout(schema)
    mod = np.array(modality)
    report = {"total": int(applicable.sum()), "declared_total": int(declared.sum()), "removed_by_absent_entry": int((declared & absent_entry).sum())}
    for k, t in enumerate(rule["node_types"]):
        report[t["name"]] = {m: int(applicable[np.ix_(node_type == k, mod == m)].sum()) for m in MODALITIES if m in t["modalities"]}
    return applicable, node_type, report


def check_expected_counts(report, rule=D1_MASK_RULE):
    """Raise MaskError unless the applicable-entry counts equal the approved ones (D1: 3,152)."""
    exp = rule.get("expected_applicable")
    if exp is None:
        return
    bad = [f"total {report['total']} != {exp['total']}"] if report["total"] != exp["total"] else []
    for t, mods in exp.items():
        if t == "total":
            continue
        for m, n in mods.items():
            if report[t][m] != n:
                bad.append(f"{t}/{m} {report[t][m]} != {n}")
    if bad:
        raise MaskError("applicable-entry counts differ from the approved design: " + "; ".join(bad))


def effective_mask(applicable, observed=None):
    """applicable (46,130) [& observed (..., 46, 130)] -> effective mask; observed None means all observed."""
    applicable = validate_bool_mask(applicable, (N_NODES, N_CHANNELS), "applicable_entry")
    if observed is None:
        return applicable
    observed = np.asarray(observed)
    if observed.dtype != np.bool_ or observed.shape[-2:] != applicable.shape:
        raise MaskError(f"observed_entry must be bool with trailing shape {applicable.shape}, got {observed.dtype} {observed.shape}")
    return observed & applicable


def mask_values(values, mask):
    """Replace every unavailable entry by 0 (a masked 0 is later told apart from a real 0 by the mask itself)."""
    mask = np.asarray(mask)
    if mask.dtype != np.bool_:
        raise MaskError(f"mask must be bool, got {mask.dtype}")
    try:
        return np.where(mask, values, np.zeros((), dtype=np.asarray(values).dtype))
    except ValueError as e:
        raise MaskError(f"mask shape {mask.shape} does not broadcast with values {np.shape(values)}") from e


def check_mask_stability(X, applicable, periods=None, schema=None, max_report=5):
    """MI-6: no non-zero raw value outside `applicable` in any period. Raises MaskError on a violation.

    `periods`: {name: row boolean mask}; default one period "all". Returns {period: 0} when clean.
    """
    applicable = validate_bool_mask(applicable, (N_NODES, N_CHANNELS), "applicable_entry")
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[1:] != applicable.shape:
        raise MaskError(f"data shape {X.shape} does not match mask {applicable.shape}")
    periods = periods or {"all": np.ones(len(X), bool)}
    names = read_layout(schema)[:2] if schema is not None else None
    out, problems = {}, []
    for pname, rows in periods.items():
        rows = np.asarray(rows, bool)
        if rows.shape != (len(X),) or not rows.any():
            raise MaskError(f"period '{pname}' selects no samples; a stability check on an empty period proves nothing")
        viol = (X[rows] != 0).any(axis=0) & ~applicable
        out[pname] = int(viol.sum())
        if viol.any():
            ex = [(names[0][i], names[1][c]) if names else (int(i), int(c)) for i, c in np.argwhere(viol)[:max_report]]
            problems.append(f"{pname}: {int(viol.sum())} entries non-zero outside the declared mask, e.g. {ex}")
    if problems:
        raise MaskError("MI-6 violated (the mask is not extended silently): " + " | ".join(problems))
    return out
