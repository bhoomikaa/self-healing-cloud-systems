"""Model 1 v2 serving contract: request validation for `model1.telemetry.v1` (no torch; see SERVING.md section 3).

A request carries T >= 7 consecutive one-minute samples of D1-layout features (46 nodes x 130 channels, in the schema's node and
channel order, *before* Model 1 scaling) plus an optional observed mask. Every window of 6 input minutes + 1 target minute inside
the segment is scored. Validation never repairs data: anything outside the contract is rejected with a code, a field and a message.
"""
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

REQUEST_SCHEMA = "model1.telemetry.v1"
RESPONSE_SCHEMA = "model1.prediction.v1"
SOURCE_KINDS = ("replay", "live")
STEP_S = 60
WINDOW = 6                      # input minutes per window; the 7th minute is the window's target (alert) minute
MIN_MINUTES = WINDOW + 1
MAX_MINUTES = 240               # four hours per request (about 20 MB of JSON); split longer streams
OOD_SCALED_ABS = 50.0           # |scaled value| above this is reported as a warning (D1 scaled values stay within about +/-14)

ARMOR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCHEMA_PATH = os.path.join(ARMOR, "data", "artifacts", "d1", "inspection", "schema.json")


class RequestError(ValueError):
    """The request violates the contract. `code` and `field` are stable identifiers for clients."""

    def __init__(self, code, field, message):
        super().__init__(f"{code} ({field}): {message}")
        self.code, self.field, self.message = code, field, message

    def as_dict(self):
        return {"code": self.code, "field": self.field, "message": self.message}


@dataclass
class Segment:
    source: dict
    timestamps: np.ndarray            # (T,) int64, step 60 s
    values: np.ndarray                # (T, 46, 130) float32, D1 pre-scaler features
    observed: np.ndarray              # (T, 46, 130) bool
    previous: dict = None             # {"target_ts", "detection_score"} of the window just before this segment, or None
    warnings: list = field(default_factory=list)


def load_schema(path=SCHEMA_PATH):
    with open(path) as fh:
        return json.load(fh)


def layout(schema):
    return {"nodes": len(schema["node_order"]), "channels": len(schema["channel_order"]),
            "node_order_sha256": schema["node_order_sha256"], "channel_order_sha256": schema["channel_order_sha256"]}


def _array(obj, name, shape, kind):
    try:
        a = np.asarray(obj, dtype=np.float64 if kind == "float" else object)
    except (ValueError, TypeError) as e:
        raise RequestError("malformed_array", name, f"not a rectangular array: {e}") from e
    if a.shape != shape:
        raise RequestError("bad_shape", name, f"shape {a.shape}, expected {shape}")
    return a


def validate(body, schema, applicable):
    """Parse and check a decoded JSON request. Returns a `Segment` or raises `RequestError`."""
    if not isinstance(body, dict):
        raise RequestError("not_an_object", "$", "the request body must be a JSON object")
    if body.get("schema_version") != REQUEST_SCHEMA:
        raise RequestError("bad_schema_version", "schema_version", f"expected {REQUEST_SCHEMA!r}, got {body.get('schema_version')!r}")
    src = body.get("source")
    if not isinstance(src, dict) or src.get("kind") not in SOURCE_KINDS or not isinstance(src.get("name"), str) or not src["name"]:
        raise RequestError("bad_source", "source", f"source must be an object with kind in {SOURCE_KINDS} and a non-empty name")
    lay = layout(schema)
    got = body.get("layout") or {}
    for k in ("node_order_sha256", "channel_order_sha256"):
        if got.get(k) != lay[k]:
            raise RequestError("layout_mismatch", f"layout.{k}", "the node/channel order fingerprint differs from the model's D1 schema (GET /v1/metadata)")
    ts = body.get("timestamps")
    if not isinstance(ts, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in ts):
        raise RequestError("bad_timestamps", "timestamps", "timestamps must be a list of integer epoch seconds")
    if not MIN_MINUTES <= len(ts) <= MAX_MINUTES:
        raise RequestError("bad_length", "timestamps", f"{len(ts)} minutes; a segment holds {MIN_MINUTES}..{MAX_MINUTES} minutes")
    ts = np.asarray(ts, np.int64)
    if not (np.diff(ts) == STEP_S).all():
        raise RequestError("not_consecutive", "timestamps", "timestamps must be strictly consecutive minutes (step 60 s, no gaps); split the stream at gaps")
    T, N, C = len(ts), lay["nodes"], lay["channels"]
    vals = body.get("values")
    if vals is None:
        raise RequestError("missing_values", "values", "values is required")
    v = _array(vals, "values", (T, N, C), "float")
    obs_raw = body.get("observed")
    if obs_raw is None:
        obs = np.ones((T, N, C), bool)
    else:
        o = _array(obs_raw, "observed", (T, N, C), "object")
        if not all(isinstance(x, bool) for x in o.ravel()):
            raise RequestError("bad_observed", "observed", "observed must hold only true/false")
        obs = o.astype(bool)
    if not np.isfinite(v).all():
        raise RequestError("non_finite", "values", "values must be finite numbers (send an unobserved entry as observed=false with value 0)")
    stray = obs & ~applicable[None] & (v != 0)
    if stray.any():
        t, n, c = (int(x[0]) for x in np.nonzero(stray))
        raise RequestError("value_at_structurally_absent_entry", "values",
                           f"{int(stray.sum())} non-zero observed values at entries the model never uses (first: minute {t}, node {schema['node_order'][n]}, "
                           f"channel {schema['channel_order'][c]['name']}); send 0 or observed=false there")
    prev = body.get("previous")
    if prev is not None:
        ok = isinstance(prev, dict) and isinstance(prev.get("target_ts"), int) and isinstance(prev.get("detection_score"), (int, float)) \
            and not isinstance(prev.get("detection_score"), bool) and math.isfinite(prev["detection_score"]) and 0.0 <= prev["detection_score"] <= 1.0
        if not ok:
            raise RequestError("bad_previous", "previous", "previous must be {target_ts: int, detection_score: number in [0, 1]}")
        if prev["target_ts"] != int(ts[WINDOW]) - STEP_S:
            raise RequestError("previous_not_adjacent", "previous.target_ts", f"must be the minute before this segment's first target minute ({int(ts[WINDOW]) - STEP_S})")
    seg = Segment(source=dict(src), timestamps=ts, values=v.astype(np.float32), observed=obs, previous=prev)
    if src["kind"] == "replay":
        seg.warnings.append({"code": "replay_data", "message": "source.kind=replay: recorded data replayed, not live telemetry"})
    return seg

