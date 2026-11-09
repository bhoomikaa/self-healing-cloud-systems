"""Evidence for the scoring harness (Linear 298-51): `evidence_armor_scoring_harness_v1.json`. Separate from every historical evidence file.

    .venv/bin/python -I armor-system/evaluation/detection/harness_evidence.py --write-evidence
    .venv/bin/python -I armor-system/evaluation/detection/harness_evidence.py --verify-evidence

Runs the synthetic equivalence battery and the published-baseline verification, runs the harness test files and the full
M1-03 integration command in clean processes, and records provenance (no self-referential hash: `source_set_sha256` covers
the files that produced the results). It never touches Model 1 checkpoints, calibration, validation or test data.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import equivalence as eq  # noqa: E402
import verify_armor_published_ad as vp  # noqa: E402

EVIDENCE = os.path.join(HERE, "evidence_armor_scoring_harness_v1.json")
REL = "armor-system/evaluation/detection/"
SOURCE_FILES = tuple(REL + f for f in ("armor_scoring.py", "equivalence.py", "verify_armor_published_ad.py", "harness_evidence.py", "DETECTION_PROTOCOL.md")) + (
    "armor-system/models/model1_transformer/window_errors.py",
    "armor-system/tests/prep/test_m1_armor_scoring.py", "armor-system/tests/prep/test_m1_armor_baseline_verify.py", "armor-system/tests/prep/test_m1_window_errors.py")
TEST_FILES = ("test_m1_armor_scoring.py", "test_m1_armor_baseline_verify.py", "test_m1_window_errors.py")
HISTORICAL = ("armor-system/data/artifacts/d1/evidence/evidence.json", "armor-system/data/artifacts/d1/evidence/evidence_m1_03.json",
              "armor-system/models/model1_xgboost_control/results/xgb_control_v1/evidence_xgb_control_v1.json", "armor-system/models/model1_transformer/results/core_v0/evidence_model1_core_v0.json",
              "armor-system/models/model1_transformer/results/experiment_v0/evidence_model1_experiment_v0.json")
AMBIGUITIES = [
    "ARMOR drops every alert interval with start == end: a single-minute alert (or repeated identical timestamps) is neither a hit nor a false alarm.",
    "ARMOR's get_pre_interval never sorts; it relies on ascending timestamps (the port reproduces this, including on unsorted synthetic input).",
    "A gap of exactly 600 s splits an interval; 599 s merges.",
    "TP and FN count labels but FP counts intervals; duplicated labels and duplicated intervals collapse to one.",
    "F1 is computed from precision and recall already rounded to 4 decimals; it can differ from the exact F1 in the fourth decimal.",
    "TP + FP = 0 or TP + FN = 0 raises ZeroDivisionError (no score instead of a score of 0); density is NaN when no interval matches.",
    "Interval lists must be tuples; ARMOR's own JSON gives lists, so the baseline script converts them explicitly.",
    "ARMOR's split time is the 60% sample quantile of all sample times; this harness uses the committed official_armor_v1.json value 1651852800 (the validation|test boundary of Model 1).",
    "The meaning of an ARMOR sample 'timestamp' (window start, end or sample time) is not stated in the released code; Model 1 alerts are placed at the window's target minute (an assumption).",
    "The origin of ad_cases.pkl (21 events without a case record, 7 cases missing from it) is unexplained; it is ARMOR's official scoring label set, not clean ground truth.",
    "ARMOR's detection threshold (SPOT on training-period scores) differs from Model 1's calibration percentile; only the interval/overlap scoring is shared.",
    "A D2 result file (pre_interval.json) exists in an earlier research scratch folder; it was not used and the D2 dataset was never downloaded.",
]


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def source_set(files=SOURCE_FILES):
    rows, h = [], hashlib.sha256()
    for rel in sorted(files):
        d = sha256_file(os.path.join(REPO, rel))
        rows.append({"path": rel, "bytes": os.path.getsize(os.path.join(REPO, rel)), "sha256": d})
        h.update(rel.encode() + b"\0" + d.encode() + b"\n")
    return rows, h.hexdigest()


def _run(cmd, env=None):
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env, timeout=3600)
    return p.returncode, p.stdout + p.stderr


def _git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def _unittest_file(name, env):
    rc, text = _run([".venv/bin/python", "-I", "-m", "unittest", "discover", "-s", "armor-system/tests/prep", "-p", name], env=env)
    last = text.strip().splitlines()[-1]
    return {"ran": int(re.search(r"Ran (\d+) tests", text).group(1)), "skipped": int((re.search(r"skipped=(\d+)", last) or [0, 0])[1]), "ok": rc == 0 and last.startswith("OK")}


def build():
    env = {**os.environ, "M1_REQUIRE_REAL_DATA": "1"}
    battery = eq.run_battery()
    published = vp.verify()
    tests = {n: _unittest_file(n, env) for n in TEST_FILES}
    rc, text = _run([".venv/bin/python", "-I", "armor-system/data/m1_integration.py"], env=env)
    m = re.search(r"SUCCESS in ([\d.]+) s: tests ran (\d+), skipped (\d+), failures (\d+), errors (\d+)", text)
    suites = {s: {"ran": int(a), "failures": int(f), "errors": int(e), "skipped": int(k)} for s, a, f, e, k in re.findall(r"^\s+(\w+): ran (\d+), failures (\d+), errors (\d+), skipped (\d+)", text, flags=re.M)}
    if rc != 0 or not m or int(m[3]) or int(m[4]) or int(m[5]) or "evidence verification: OK" not in text:
        raise SystemExit("integration command did not succeed with zero skips:\n" + text[-2500:])
    for n, t in tests.items():
        if not t["ok"] or t["skipped"]:
            raise SystemExit(f"test file {n} not clean: {t}")
    if not battery["all_identical"] or published["status"] != "PASS":
        raise SystemExit(f"harness verification failed: battery {battery['all_identical']}, published {published['status']}")
    rows, set_hash = source_set()
    hist = {p: {"sha256": sha256_file(os.path.join(REPO, p)), "unchanged_vs_HEAD": subprocess.run(["git", "diff", "--quiet", "HEAD", "--", p], cwd=REPO).returncode == 0} for p in HISTORICAL}
    if not all(h["unchanged_vs_HEAD"] for h in hist.values()):
        raise SystemExit(f"historical evidence changed: {hist}")
    return {"evidence_name": "evidence_armor_scoring_harness_v1", "linear_issue": "298-51",
            "scope": "scoring-harness verification only (ARMOR-equivalent interval construction and event scoring, per-window error extraction); no calibration, no Model 1 scoring, no validation or test evaluation; baseline-harness verification, not Model 1 test evidence",
            "ownership": "ARMOR baseline = baseline teammates (armor-system/baselines/armor); this harness needs their review before it is cited as the baseline reproduction",
            "command": ".venv/bin/python -I armor-system/evaluation/detection/harness_evidence.py --write-evidence",
            "synthetic_equivalence": battery, "published_verification": published, "scoring_ambiguities": AMBIGUITIES,
            "tests": {"harness_files": tests, "m1_integration": {"command": ".venv/bin/python -I armor-system/data/m1_integration.py", "tests_ran": int(m[2]), "skipped": int(m[3]), "failures": int(m[4]), "errors": int(m[5]), "suites": suites, "evidence_m1_03_verified": True}},
            "historical_evidence_unchanged": hist,
            "commit_reference": {"scheme": "no self-referential hash; source_set_sha256 covers the harness code, protocol and tests", "evidence_base_commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                                 "source_set_sha256": set_hash, "source_files": rows}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-evidence", action="store_true")
    ap.add_argument("--verify-evidence", action="store_true")
    a = ap.parse_args(argv)
    if a.write_evidence:
        ev = build()
        with open(EVIDENCE, "w") as fh:
            json.dump(ev, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
        print("wrote", os.path.relpath(EVIDENCE, REPO), ev["commit_reference"]["source_set_sha256"])
        return 0
    if a.verify_evidence:
        with open(EVIDENCE) as fh:
            ev = json.load(fh)
        bad = []
        if ev["commit_reference"]["source_set_sha256"] != source_set()[1]:
            bad.append("source_set_sha256 differs from the tree")
        if not eq.run_battery()["all_identical"]:
            bad.append("equivalence battery no longer identical")
        pv = vp.verify()
        if pv["status"] != "PASS" or (pv["precision"], pv["recall"], pv["f1"]) != (0.9245, 1.0, 0.9608):
            bad.append("published verification no longer PASS")
        print("evidence verification:", bad or "OK")
        return 1 if bad else 0
    ap.error("give --write-evidence or --verify-evidence")


if __name__ == "__main__":
    sys.exit(main())
