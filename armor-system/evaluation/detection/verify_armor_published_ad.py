"""Baseline-only verification: reproduce ARMOR's PUBLISHED anomaly-detection metrics from its RELEASED alert intervals.

    .venv/bin/python -I armor-system/evaluation/detection/verify_armor_published_ad.py [--out FILE]

This is a baseline-harness verification of the scoring code, NOT a Model 1 evaluation and NOT a Model 1 test result. It imports no
Model 1 model, dataset, checkpoint, calibration or training code (only `armor_scoring.py`, NumPy and the standard library). It
reads exactly: ARMOR's released `pre_interval.json` (sha256 pinned below), ARMOR's own scoring labels `ad_cases.pkl` (size and
CRC32 pinned in the committed D1 acquisition manifest; read with a pickle reader that forbids every global), and the committed
split file for ARMOR's AD split time. It prints only the three aggregate metrics, PASS/FAIL/BLOCKED and provenance hashes:
no per-case predictions, no label values, no Model 1 output.

Ownership: the ARMOR baseline belongs to the baseline teammates (`armor-system/baselines/armor`). This harness only verifies that
the scoring port reproduces their published numbers and must be reviewed by them before it is used as the baseline reproduction.
"""
import argparse
import hashlib
import io
import json
import os
import pickle
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

from armor_scoring import eval_AD  # noqa: E402

RAW = os.path.join(REPO, "armor-system", "data", "raw", "armor_zenodo_21744694")
RELEASED = os.path.join(RAW, "ARMOR", "res", "D1", "20260113_101332", "scenarios", "full")
INTERVAL_FILE = os.path.join(RELEASED, "tmp", "pre_interval.json")
RES_JSON = os.path.join(RELEASED, "res.json")
LABEL_FILE = os.path.join(RAW, "extracted", "D1", "cases", "ad_cases.pkl")
SPLIT_FILE = os.path.join(REPO, "armor-system", "data", "artifacts", "d1", "splits", "official_armor_v1.json")
ACQ_MANIFEST = os.path.join(REPO, "armor-system", "data", "manifests", "armor_d1", "checksums.json")
ORIGINAL_METRICS = os.path.join(RAW, "ARMOR", "models", "diagnosis_tasks", "task_metrics.py")

PINNED_INTERVAL_SHA256 = "c8da7bfa44cf18c44403739b98de8e45e98d1f628ae500c3e0ccb57f46c0e9d7"      # Zenodo 21744694, ARMOR/res/D1/20260113_101332/scenarios/full/tmp/pre_interval.json
PUBLISHED = {"precision": 0.9245, "recall": 1.0, "f1": 0.9608}                                    # ARMOR res.json, "AD"
PUBLISHED_LABELS = 98                                                                             # published number of post-split official AD labels
IMPACT_WINDOW = 300


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"global {module}.{name} is not allowed in ad_cases.pkl")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    with open(path, "rb") as fh:
        return sha256_bytes(fh.read())


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def verify(interval_file=INTERVAL_FILE, label_file=LABEL_FILE, res_json=RES_JSON, split_file=SPLIT_FILE):
    prov = {"interval_file_sha256_pinned": PINNED_INTERVAL_SHA256, "script_sha256": sha256_file(__file__), "armor_scoring_py_sha256": sha256_file(os.path.join(HERE, "armor_scoring.py"))}
    out = {"verification": "ARMOR published anomaly-detection metrics reproduced from released alert intervals", "scope": "baseline-harness verification of the scoring port; not a Model 1 evaluation; no test-period Model 1 output exists or is used",
           "status": "BLOCKED", "reasons": [], "precision": None, "recall": None, "f1": None, "published": dict(PUBLISHED), "provenance": prov}
    for name, path in (("released interval file", interval_file), ("official scoring labels (ad_cases.pkl)", label_file), ("committed split file", split_file), ("D1 acquisition manifest", ACQ_MANIFEST)):
        if not os.path.isfile(path):
            out["reasons"].append(f"missing: {name}; nothing is substituted")
    if out["reasons"]:
        return out
    iv_bytes = _read(interval_file)
    prov["interval_file_sha256"] = sha256_bytes(iv_bytes)
    if prov["interval_file_sha256"] != PINNED_INTERVAL_SHA256:
        out["reasons"].append("released interval file checksum differs from the pinned value")
        return out
    lb = _read(label_file)
    with open(ACQ_MANIFEST) as fh:
        pin = next(f for f in json.load(fh)["nested_d1_extraction"]["files"] if f["path"] == "D1/cases/ad_cases.pkl")
    prov["ad_cases_pkl"] = {"bytes": len(lb), "crc32": "%08x" % zlib.crc32(lb), "sha256": sha256_bytes(lb), "pinned_bytes": pin["size"], "pinned_crc32": pin["crc32"]}
    if len(lb) != pin["size"] or prov["ad_cases_pkl"]["crc32"] != pin["crc32"]:
        out["reasons"].append("ad_cases.pkl differs from the size/CRC32 recorded in the D1 acquisition manifest")
        return out
    with open(split_file) as fh:
        split = json.load(fh)
    split_ts = int(split["ad_split"]["timestamp"])
    prov["split_file_content_sha256"], prov["armor_ad_split_timestamp"] = split["content_sha256"], split_ts
    prov["res_json_sha256"] = sha256_file(res_json) if os.path.isfile(res_json) else None
    if prov["res_json_sha256"]:
        with open(res_json) as fh:
            ad = json.load(fh)["AD"]
        if {k: ad[k] for k in PUBLISHED} != PUBLISHED:
            out["reasons"].append("the released res.json AD values differ from the published constants of this script")
    labels = [int(x) for x in _NoGlobals(io.BytesIO(lb)).load() if int(x) > split_ts]          # ARMOR: ad_cases_label = [item for item in labels if item > split_ts]
    if len(labels) != PUBLISHED_LABELS:
        out["reasons"].append(f"the post-split official label population differs from the published {PUBLISHED_LABELS}")
    intervals = [tuple(int(v) for v in pair) for pair in json.loads(iv_bytes)]                  # JSON gives lists; ARMOR's eval needs hashable pairs
    p, r, f = eval_AD(intervals, labels, IMPACT_WINDOW)
    out["precision"], out["recall"], out["f1"] = float(p), float(r), float(f)
    if os.path.isfile(ORIGINAL_METRICS):                                                        # cross-check with ARMOR's own eval_AD source
        import importlib.util
        spec = importlib.util.spec_from_file_location("armor_task_metrics", ORIGINAL_METRICS)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        prov["armor_task_metrics_py_sha256"] = sha256_file(ORIGINAL_METRICS)
        out["original_eval_AD_agrees"] = tuple(float(x) for x in mod.eval_AD(intervals, labels, IMPACT_WINDOW)) == (out["precision"], out["recall"], out["f1"])
        if not out["original_eval_AD_agrees"]:
            out["reasons"].append("ARMOR's original eval_AD disagrees with the port on the released intervals")
    same = {k: out[k] == PUBLISHED[k] for k in PUBLISHED}
    out["matches_published"] = same
    out["status"] = "PASS" if all(same.values()) and not out["reasons"] else "FAIL"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="also write the result JSON to this file")
    a = ap.parse_args(argv)
    res = verify()
    text = json.dumps(res, indent=2, sort_keys=True)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
