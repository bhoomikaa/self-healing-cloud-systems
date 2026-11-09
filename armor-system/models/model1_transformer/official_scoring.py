"""Official ARMOR population scorer wrapper (Linear 298-51, TEST_PLAN.md section 5).

Reads ARMOR's scoring labels (`ad_cases.pkl`: pinned by size and CRC32 in the D1 acquisition manifest, parsed with the harness's
pickle reader that forbids every global, kept only above ARMOR's AD split time = 98 labels) and ARMOR's released full-scenario alert
intervals (sha256 pinned). Both loaders need a TestRunToken: the labels lie entirely in the locked test period. Model 1 alerts are scored
with the same ARMOR-equivalent construction as validation (`run_detection_v0.evaluate_alerts`: 600 s interval merging, singletons
dropped, +/-300 s impact window), so only the label population differs. The reference row re-scores ARMOR's released intervals and
must reproduce ARMOR's published P 0.9245, R 1.0, F1 0.9608. No other ARMOR result file exists in the release (TEST_PLAN.md section 6).
"""
import io
import json
import zlib

from . import run_detection_v0 as rd, test_freeze as tf          # run_detection_v0 puts evaluation/detection on sys.path

import verify_armor_published_ad as va  # noqa: E402


class OfficialInputError(RuntimeError):
    """An official ARMOR input is missing or differs from its pinned checksum."""


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def load_labels(token, label_file=va.LABEL_FILE, split_file=va.SPLIT_FILE, manifest=va.ACQ_MANIFEST, expected=va.PUBLISHED_LABELS):
    tf.check_token(token)
    with open(manifest) as fh:
        pin = next(f for f in json.load(fh)["nested_d1_extraction"]["files"] if f["path"] == tf.ARMOR_LABELS_MANIFEST_PATH)
    b = _read(label_file)
    crc = "%08x" % zlib.crc32(b)
    if len(b) != pin["size"] or crc != pin["crc32"]:
        raise OfficialInputError("ad_cases.pkl differs from the size/CRC32 pinned in the acquisition manifest")
    with open(split_file) as fh:
        split = json.load(fh)
    split_ts = int(split["ad_split"]["timestamp"])
    labels = [int(x) for x in va._NoGlobals(io.BytesIO(b)).load() if int(x) > split_ts]          # ARMOR: [item for item in labels if item > split_ts]
    if expected is not None and len(labels) != expected:
        raise OfficialInputError(f"post-split official labels {len(labels)} != published {expected}")
    return {"labels": labels, "split_ts": split_ts, "provenance": {"bytes": len(b), "crc32": crc, "sha256": va.sha256_bytes(b), "split_file_content_sha256": split["content_sha256"]}}


def released_intervals(token, interval_file=va.INTERVAL_FILE, pinned=va.PINNED_INTERVAL_SHA256):
    tf.check_token(token)
    b = _read(interval_file)
    if va.sha256_bytes(b) != pinned:
        raise OfficialInputError("released interval file differs from the pinned sha256")
    return [tuple(int(v) for v in pair) for pair in json.loads(b)]


def score(flags, target_ts, category, labels):
    """Model 1 alerts (bool per target minute, ascending) against the official labels: the validation construction and diagnostics."""
    return rd.evaluate_alerts(flags, target_ts, category, labels)


def armor_reference(token, labels):
    """ARMOR's released full-scenario intervals scored with the port: the reproduction row (must equal the published numbers)."""
    iv = released_intervals(token)
    r = rd.score_events(iv, labels)
    r["matches_published"] = {k: r[k] == va.PUBLISHED[k] for k in va.PUBLISHED}
    r["published"] = dict(va.PUBLISHED)
    r["n_intervals"] = len(iv)
    return r
