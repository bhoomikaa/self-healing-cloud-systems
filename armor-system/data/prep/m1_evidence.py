"""M1-03 evidence record (`evidence_m1_03.json`) and its verification.

Scope: Linear 298-48 only. The historical 298-44 `evidence.json` is never read or rewritten here.

No circular hashing: the evidence file contains no hash of itself and no commit SHA that the commit holding the
file would change. Instead it records
  * `source_set_sha256`: a hash over the content of every M1-03 source file (code, tests, the two artifacts, the
    data-layer doc) - recomputable from any checkout, and equal in the working tree and in the commit that contains it;
  * `evidence_base_commit`: the commit the work was based on when the evidence was generated (normally the parent
    of the M1-03 commit; informational, the content claim rests on `source_set_sha256`);
  * `final_merge_commit`: null here, recorded in the Linear 298-48 comment when the PR merges.
"""
import hashlib
import json
import os

from .splits import content_hash as manifest_content_hash

EVIDENCE_NAME = "evidence_m1_03"
EVIDENCE_VERSION = "1.0"
SCOPE = "Linear 298-48 (Model 1 M1-03 data layer) only. The 298-44 data-preparation evidence is armor-system/data/artifacts/d1/evidence/evidence.json and is not touched."

# Repo-relative, stable. The evidence file itself and shared docs (START_HERE.md) are deliberately not in the set.
SOURCE_FILES = (
    "armor-system/data/model1_pipeline.py",
    "armor-system/data/m1_integration.py",
    "armor-system/data/prep/m1_masks.py",
    "armor-system/data/prep/m1_split.py",
    "armor-system/data/prep/m1_dataset.py",
    "armor-system/data/prep/m1_evidence.py",
    "armor-system/data/prep/detection_calibration.py",
    "armor-system/data/artifacts/d1/splits/model1_calibration_split_v1.json",
    "armor-system/data/artifacts/d1/preprocessing/scaler_model1_fit_only_v1.npz",
    "armor-system/models/model1_transformer/DATA_LAYER.md",
    "armor-system/tests/prep/_m1_helpers.py",
    "armor-system/tests/prep/test_m1_dataset.py",
    "armor-system/tests/prep/test_m1_detection_calibration.py",
    "armor-system/tests/prep/test_m1_evidence.py",
    "armor-system/tests/prep/test_m1_masks.py",
    "armor-system/tests/prep/test_m1_required_resources.py",
    "armor-system/tests/prep/test_m1_split.py",
)
EVIDENCE_PATH = "armor-system/data/artifacts/d1/evidence/evidence_m1_03.json"
MANIFEST_PATH = "armor-system/data/artifacts/d1/splits/model1_calibration_split_v1.json"
SCALER_PATH = "armor-system/data/artifacts/d1/preprocessing/scaler_model1_fit_only_v1.npz"


class EvidenceError(RuntimeError):
    pass


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def source_set(repo_root, files=SOURCE_FILES):
    """({path, bytes, sha256} list, set hash). Raises EvidenceError if a listed file is missing."""
    rows, h = [], hashlib.sha256()
    for rel in sorted(files):
        full = os.path.join(repo_root, rel)
        if not os.path.isfile(full):
            raise EvidenceError(f"M1-03 source file missing: {rel}")
        d = sha256_file(full)
        rows.append({"path": rel, "bytes": os.path.getsize(full), "sha256": d})
        h.update(rel.encode() + b"\0" + d.encode() + b"\n")
    return rows, h.hexdigest()


def build(repo_root, *, d1_verification, cache_rebuild, reproducibility, tests, command, environment, git):
    """Assemble the evidence dict from the facts the integration command measured."""
    manifest = read_json(os.path.join(repo_root, MANIFEST_PATH))
    rows, set_hash = source_set(repo_root)
    return {
        "evidence_name": EVIDENCE_NAME, "evidence_version": EVIDENCE_VERSION, "scope": SCOPE, "linear_issue": "298-48",
        "hashes": {
            "calibration_manifest_file_sha256": sha256_file(os.path.join(repo_root, MANIFEST_PATH)),
            "calibration_manifest_content_sha256": manifest["content_sha256"],
            "fit_only_scaler_file_sha256": sha256_file(os.path.join(repo_root, SCALER_PATH)),
            "feature_cache_content_sha256": manifest["inputs"]["feature_cache_content_sha256"],
            "outer_split_content_sha256": manifest["inputs"]["outer_split_manifest"]["content_sha256"],
            "preprocessing_config_sha256": manifest["inputs"]["preprocessing_config_sha256"],
        },
        "counts": {k: manifest["counts"]["roles"][k]["windows"] for k in ("fit", "calibration", "validation", "test")} | {"cases": {k: manifest["counts"]["roles"][k]["cases"] for k in ("fit", "calibration", "validation", "test")}},
        "d1_verification": d1_verification, "feature_cache_rebuild": cache_rebuild, "reproducibility": reproducibility,
        "integration_command": command, "tests": tests, "environment": environment,
        "commit_reference": {
            "scheme": "No self-referential hash. evidence_base_commit is the commit this work was based on when the evidence was generated; "
                      "source_set_sha256 identifies the exact content of the M1-03 files and is identical in the working tree and in the commit that adds them; "
                      "the final PR merge commit is recorded in the Linear 298-48 comment after merge.",
            "evidence_base_commit": git["head"], "branch": git["branch"], "working_tree_clean_when_generated": git["clean"],
            "source_set_sha256": set_hash, "source_files": rows,
            "final_merge_commit": None, "final_merge_commit_recorded_in": "Linear 298-48 comment (added when the PR merges)",
        },
    }


def verify(evidence, repo_root):
    """Return a list of problems (empty = the evidence matches the current tree and artifacts)."""
    bad = []
    if evidence.get("evidence_name") != EVIDENCE_NAME or evidence.get("evidence_version") != EVIDENCE_VERSION:
        bad.append("evidence name/version mismatch")
    cr = evidence.get("commit_reference", {})
    if cr.get("final_merge_commit") is not None and not isinstance(cr["final_merge_commit"], str):
        bad.append("final_merge_commit must be null or a SHA string")
    try:
        rows, set_hash = source_set(repo_root)
    except EvidenceError as e:
        return bad + [str(e)]
    if cr.get("source_set_sha256") != set_hash:
        bad.append("source_set_sha256 differs from the current M1-03 files (code, tests or artifacts changed after the evidence was generated)")
    recorded = {r["path"]: r["sha256"] for r in cr.get("source_files", [])}
    for r in rows:
        if recorded.get(r["path"]) != r["sha256"]:
            bad.append(f"changed since evidence: {r['path']}")
    m = read_json(os.path.join(repo_root, MANIFEST_PATH))
    h = evidence.get("hashes", {})
    exp = {"calibration_manifest_file_sha256": sha256_file(os.path.join(repo_root, MANIFEST_PATH)),
           "calibration_manifest_content_sha256": m["content_sha256"],
           "fit_only_scaler_file_sha256": sha256_file(os.path.join(repo_root, SCALER_PATH)),
           "feature_cache_content_sha256": m["inputs"]["feature_cache_content_sha256"],
           "outer_split_content_sha256": m["inputs"]["outer_split_manifest"]["content_sha256"],
           "preprocessing_config_sha256": m["inputs"]["preprocessing_config_sha256"]}
    if m.get("content_sha256") != manifest_content_hash(m):
        bad.append("calibration manifest content_sha256 does not match its body")
    for k, v in exp.items():
        if h.get(k) != v:
            bad.append(f"hash {k} differs from the artifact")
    t = evidence.get("tests", {})
    suites = t.get("suites") or [t]
    if not (isinstance(t.get("ran"), int) and t["ran"] > 0 and t.get("skipped") == 0 and t.get("failures") == 0 and t.get("errors") == 0):
        bad.append("recorded test run lacks a positive count or has skips, failures or errors")
    if any(not (isinstance(s.get("ran"), int) and s["ran"] > 0) for s in suites):
        bad.append("a recorded test suite ran no tests")
    base = cr.get("evidence_base_commit", "")
    if not (isinstance(base, str) and len(base) == 40 and all(c in "0123456789abcdef" for c in base)):
        bad.append("evidence_base_commit is not a 40-hex commit SHA")
    rep = evidence.get("reproducibility", {})
    if not (rep.get("runs", 0) >= 2 and rep.get("manifest_identical") and rep.get("scaler_identical") and rep.get("equal_to_committed")):
        bad.append("reproducibility check not recorded as identical")
    d1 = evidence.get("d1_verification", {})
    if d1.get("problems") != [] or not (isinstance(d1.get("inventory_files_checked"), int) and d1["inventory_files_checked"] > 0):
        bad.append("D1 verification missing, empty or recorded problems")
    if not evidence.get("feature_cache_rebuild", {}).get("rebuilt_equals_manifest"):
        bad.append("clean cache rebuild not recorded as equal to the manifest hash")
    return bad
