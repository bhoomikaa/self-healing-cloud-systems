"""One-shot test freeze for Model 1 (Linear 298-51): freeze record, verification, run marker and the gated test unlock.

    .venv/bin/python -I armor-system/models/model1_transformer/test_freeze.py --write     # write evidence_test_freeze.json (freeze commit)
    .venv/bin/python -I armor-system/models/model1_transformer/test_freeze.py --verify    # print the problems (none = frozen state matches disk)

The freeze record (`results/test_freeze/evidence_test_freeze.json`) holds the sha256 of every tracked file under the Model 1, XGBoost,
detection-scoring and data-preparation trees (plus the tests), the 20 Model 1 checkpoint files outside the repository, the feature cache,
the ARMOR files the test step reads (released intervals pinned by sha256; the official label file pinned by size and CRC32 from the
acquisition manifest, never opened here), the test plan and the protocol. It holds no hash of itself.

How the test locks are lifted. The existing locks stay in place and unchanged: `window_errors.SCORABLE_ROLES` (calibration, validation),
`case_eval.predict_window_probabilities` (validation), `m1_case_features.build_case_matrix` (fit, validation) and
`xgb_control.final_test()`. Those files are bound into earlier evidence files whose verification hashes the live files, so editing them
would invalidate that evidence. Instead, `unlocked(token)` adds the role 'test' to `window_errors.SCORABLE_ROLES` and
`m1_case_features.ALLOWED_ROLES` for the duration of a `with` block, and only for a token that `claim_test_run()` returns after the
freeze verified and the one-shot marker was written. No other code path can score a test window.
"""
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FREEZE_DIR = os.path.join(HERE, "results", "test_freeze")
FREEZE_FILE = os.path.join(FREEZE_DIR, "evidence_test_freeze.json")
FREEZE_REL = os.path.relpath(FREEZE_FILE, REPO)
TEST_OUT = os.path.join(HERE, "results", "test_v1")
TEST_OUT_REL = os.path.relpath(TEST_OUT, REPO) + "/"
CKPT_ROOT = os.path.expanduser("~/298A_project/checkpoints")
EXTERNAL_MARKER = os.path.join(CKPT_ROOT, "model1_test_run_marker.json")
REPO_MARKER = os.path.join(TEST_OUT, "TEST_RUN_MARKER.json")
TRACKED_PREFIXES = ("armor-system/models/model1_transformer/", "armor-system/models/model1_xgboost_control/", "armor-system/evaluation/detection/",
                    "armor-system/data/prep/", "armor-system/data/artifacts/d1/", "armor-system/data/manifests/armor_d1/", "armor-system/tests/prep/", "armor-system/tests/data/")
TRACKED_TOP = ("armor-system/data/m1_integration.py", "armor-system/data/model1_pipeline.py")
CHECKPOINTS = tuple((v, s, st) for v in ("model1_v1", "model1_v0") for s in range(5) for st in ("s1_best", "s2_best"))
RAW = os.path.join(REPO, "armor-system", "data", "raw", "armor_zenodo_21744694")
ARMOR_INTERVALS = os.path.join(RAW, "ARMOR", "res", "D1", "20260113_101332", "scenarios", "full", "tmp", "pre_interval.json")
ARMOR_LABELS_MANIFEST_PATH = "D1/cases/ad_cases.pkl"
ACQ_MANIFEST = os.path.join(REPO, "armor-system", "data", "manifests", "armor_d1", "checksums.json")
ARMOR_CODE = {"spot.py": os.path.join(RAW, "ARMOR", "utils", "spot.py"), "anomaly_detector.py": os.path.join(RAW, "ARMOR", "models", "diagnosis_tasks", "anomaly_detector.py"),
              "task_metrics.py": os.path.join(RAW, "ARMOR", "models", "diagnosis_tasks", "task_metrics.py")}
PROTOCOL_REL = "armor-system/models/model1_transformer/V1_PROTOCOL.md"
PLAN_REL = "armor-system/models/model1_transformer/TEST_PLAN.md"


class FreezeError(RuntimeError):
    """The frozen state does not match disk, or the one-shot test run was already claimed."""


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git(*a, repo=REPO):
    p = subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True)
    if p.returncode != 0:
        raise FreezeError(f"git {' '.join(a)} failed: {p.stderr.strip()}")
    return p.stdout


def tracked_files(repo=REPO):
    """Tracked files under the frozen trees, except the freeze record itself and the test output directory."""
    out = [p for p in _git("ls-files", "-z", repo=repo).split("\0") if p]
    keep = [p for p in out if (p.startswith(TRACKED_PREFIXES) or p in TRACKED_TOP) and p != FREEZE_REL and not p.startswith(TEST_OUT_REL)]
    return sorted(keep)


def set_hash(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(r["path"].encode() + b"\0" + r["sha256"].encode() + b"\n")
    return h.hexdigest()


def checkpoint_rows(root=CKPT_ROOT):
    rows = []
    for v, s, st in CHECKPOINTS:
        for ext in (".pt", ".json"):
            rel = os.path.join(v, f"seed_{s}", st + ext)
            rows.append({"path": rel, "sha256": sha(os.path.join(root, rel))})
    return rows


def label_file_pin(manifest=ACQ_MANIFEST):
    with open(manifest) as fh:
        pin = next(f for f in json.load(fh)["nested_d1_extraction"]["files"] if f["path"] == ARMOR_LABELS_MANIFEST_PATH)
    return {"path_in_manifest": ARMOR_LABELS_MANIFEST_PATH, "size": pin["size"], "crc32": pin["crc32"], "read_by_freeze": False}


def external_inputs():
    from prep import paths
    return {"feature_cache": {"path": os.path.relpath(paths.CACHE_FILE, REPO), "sha256": sha(paths.CACHE_FILE)},
            "armor_released_intervals_full": {"path": os.path.relpath(ARMOR_INTERVALS, REPO), "sha256": sha(ARMOR_INTERVALS)},
            "armor_official_labels": label_file_pin(),
            "armor_code": {k: sha(p) for k, p in ARMOR_CODE.items()}}


def build(repo=REPO, ckpt_root=CKPT_ROOT, extra=None):
    """The freeze record (no self-hash). `extra` adds facts such as the integration run and the validation ensemble files."""
    files = [{"path": p, "sha256": sha(os.path.join(repo, p))} for p in tracked_files(repo)]
    ck = checkpoint_rows(ckpt_root)
    doc = {"evidence_name": "evidence_test_freeze", "evidence_version": "1.0", "linear_issue": "298-51",
           "scope": "Frozen state for the one-shot Model 1 / v0 / XGBoost test evaluation. Written before any test-period window, test case label or official ARMOR label is read.",
           "test_plan": {"path": PLAN_REL, "sha256": sha(os.path.join(repo, PLAN_REL))}, "protocol": {"path": PROTOCOL_REL, "sha256": sha(os.path.join(repo, PROTOCOL_REL))},
           "tracked_prefixes": list(TRACKED_PREFIXES) + list(TRACKED_TOP), "tracked_files": files, "tracked_set_sha256": set_hash(files),
           "checkpoints": {"root": "~/298A_project/checkpoints", "files": ck, "set_sha256": set_hash(ck)},
           "external_inputs": external_inputs(),
           "locks": {"kept_unchanged": ["window_errors.SCORABLE_ROLES", "case_eval.predict_window_probabilities role check", "m1_case_features.build_case_matrix role check", "xgb_control.final_test"],
                     "lifted_by": "test_freeze.unlocked(token): adds 'test' to window_errors.SCORABLE_ROLES and m1_case_features.ALLOWED_ROLES inside a with block, for a token from claim_test_run() only"},
           "one_shot": {"markers": ["armor-system/models/model1_transformer/results/test_v1/TEST_RUN_MARKER.json", "~/298A_project/checkpoints/model1_test_run_marker.json"],
                        "rule": "the runner refuses to start if either marker exists; a marker is written (exclusive create) before any test window is read"},
           "base_commit": _git("rev-parse", "HEAD", repo=repo).strip(), "branch": _git("rev-parse", "--abbrev-ref", "HEAD", repo=repo).strip(),
           "environment": {"python": platform.python_version()},
           "test_data_accessed": False, "official_armor_labels_read": False}
    doc.update(extra or {})
    return doc


def verify(repo=REPO, ckpt_root=CKPT_ROOT, freeze_file=FREEZE_FILE, require_committed=True):
    """Problems (empty list = every frozen hash matches disk, the tree is clean and the freeze record is committed unchanged)."""
    if not os.path.isfile(freeze_file):
        return ["no freeze record: " + os.path.relpath(freeze_file, repo)]
    with open(freeze_file) as fh:
        fr = json.load(fh)
    bad = []
    if require_committed:
        rel = os.path.relpath(freeze_file, repo)
        if not _git("ls-files", rel, repo=repo).strip():
            bad.append("the freeze record is not committed")
        elif subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=repo).returncode != 0:
            bad.append("the freeze record differs from the committed one")
        dirty = [ln for ln in _git("status", "--porcelain", "--untracked-files=all", repo=repo).splitlines() if not ln[3:].startswith(TEST_OUT_REL)]
        if dirty:
            bad.append(f"working tree not clean: {dirty[:5]}")
    rec = {r["path"]: r["sha256"] for r in fr["tracked_files"]}
    now = tracked_files(repo)
    if sorted(rec) != now:
        bad.append(f"tracked file set changed: added {sorted(set(now) - set(rec))[:5]}, removed {sorted(set(rec) - set(now))[:5]}")
    for p in now:
        if p in rec and sha(os.path.join(repo, p)) != rec[p]:
            bad.append(f"changed since freeze: {p}")
    for r in fr["checkpoints"]["files"]:
        f = os.path.join(ckpt_root, r["path"])
        if not os.path.isfile(f) or sha(f) != r["sha256"]:
            bad.append(f"checkpoint differs or missing: {r['path']}")
    for key in ("test_plan", "protocol"):
        if sha(os.path.join(repo, fr[key]["path"])) != fr[key]["sha256"]:
            bad.append(f"{key} changed since freeze")
    ext = external_inputs()
    for k in ("feature_cache", "armor_released_intervals_full", "armor_official_labels", "armor_code"):
        if ext[k] != fr["external_inputs"][k]:
            bad.append(f"external input changed: {k}")
    return bad


class TestRunToken:
    """Proof that this process verified the freeze and owns the one-shot marker. Only `claim_test_run` and `resume_token` create one."""

    def __init__(self, run_id, freeze_sha256):
        self.run_id, self.freeze_sha256 = run_id, freeze_sha256


def _write_exclusive(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)          # fails if the file exists: a second run is refused
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")


def claim_test_run(repo=REPO, ckpt_root=CKPT_ROOT, freeze_file=FREEZE_FILE, markers=(REPO_MARKER, EXTERNAL_MARKER)):
    """Verify the freeze, then write both one-shot markers (exclusive create). Raises FreezeError on any problem or an existing marker."""
    existing = [m for m in markers if os.path.exists(m)]
    if existing:
        raise FreezeError(f"the one-shot test run was already started (marker {existing[0]}); a second test run is refused")
    bad = verify(repo, ckpt_root, freeze_file)
    if bad:
        raise FreezeError("freeze verification failed: " + "; ".join(bad[:8]))
    tok = TestRunToken(uuid.uuid4().hex, sha(freeze_file))
    doc = {"run_id": tok.run_id, "freeze_record_sha256": tok.freeze_sha256, "head": _git("rev-parse", "HEAD", repo=repo).strip(), "started_unix": int(time.time()),
           "note": "written before any test window was read; its presence refuses every later test run"}
    for m in markers:
        _write_exclusive(m, doc)
    return tok


def resume_token(run_id, repo=REPO, ckpt_root=CKPT_ROOT, freeze_file=FREEZE_FILE, markers=(REPO_MARKER, EXTERNAL_MARKER)):
    """For the runner's own worker processes: the freeze must still verify and both markers must name this run."""
    for m in markers:
        if not os.path.isfile(m):
            raise FreezeError(f"marker missing: {m}")
        with open(m) as fh:
            if json.load(fh)["run_id"] != run_id:
                raise FreezeError(f"marker {m} belongs to another run")
    bad = verify(repo, ckpt_root, freeze_file)
    if bad:
        raise FreezeError("freeze verification failed: " + "; ".join(bad[:8]))
    return TestRunToken(run_id, sha(freeze_file))


def check_token(token):
    if not isinstance(token, TestRunToken) or not token.run_id:
        raise FreezeError("the test role is locked: a TestRunToken from claim_test_run() is required")


@contextlib.contextmanager
def unlocked(token):
    """Inside the block the role 'test' is scorable (window_errors) and available for XGBoost case features; restored afterwards."""
    check_token(token)
    from prep import m1_case_features as cf

    from . import window_errors as we
    saved = (we.SCORABLE_ROLES, cf.ALLOWED_ROLES)
    we.SCORABLE_ROLES = tuple(saved[0]) + ("test",)
    cf.ALLOWED_ROLES = tuple(saved[1]) + ("test",)
    try:
        yield
    finally:
        we.SCORABLE_ROLES, cf.ALLOWED_ROLES = saved


def main(argv=None):
    import argparse
    sys.path[:0] = [os.path.join(REPO, "armor-system", "data"), os.path.join(REPO, "armor-system", "models")]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--extra", help="JSON file with additional facts to record (integration run, validation ensemble)")
    a = ap.parse_args(argv)
    if a.write:
        extra = None
        if a.extra:
            with open(a.extra) as fh:
                extra = json.load(fh)
        doc = build(extra=extra)
        os.makedirs(FREEZE_DIR, exist_ok=True)
        with open(FREEZE_FILE, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("wrote", FREEZE_REL, "tracked files", len(doc["tracked_files"]), "set", doc["tracked_set_sha256"])
    if a.verify:
        bad = verify()
        print("freeze verification:", bad or "OK")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
