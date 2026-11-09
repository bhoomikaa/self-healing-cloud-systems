"""M1-03 real-data integration command (Linear 298-48). CPU only; never downloads anything (no D2).

    .venv/bin/python -I armor-system/data/m1_integration.py                    # verify everything, write nothing
    .venv/bin/python -I armor-system/data/m1_integration.py --write-evidence   # also (re)write evidence_m1_03.json
    .venv/bin/python -I armor-system/data/m1_integration.py --update           # also rewrite the M1-03 artifacts if they differ

Steps (any failure stops with a non-zero exit status):
  1. D1 source files re-hashed against the approved checksums (`file_inventory.csv`, `checksums.json`).
  2. Feature cache rebuilt from source into a fresh temporary directory; its content hash must equal the hash recorded
     in the committed calibration manifest. A missing or stale cache in `data/processed/` is replaced by the rebuild.
  3. M1-03 artifacts regenerated twice in independent processes; both runs must be byte-identical to each other and
     to the committed artifacts.
  4. The data and prep test suites run with skipping disabled (M1_REQUIRE_REAL_DATA=1); zero skips are required.
  5. Evidence (`evidence_m1_03.json`) written (with --write-evidence) or, if present, verified against the tree.
Run it from any directory; it adds `armor-system/data` to `sys.path` itself (`armor_system` is not a package).
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from prep import loader, m1_evidence as ev, m1_split as ms, paths  # noqa: E402

REPO = os.path.abspath(os.path.join(paths.ROOT, ".."))
COMMAND = ".venv/bin/python -I armor-system/data/m1_integration.py --write-evidence"
SUITES = (("data", "armor-system/tests/data"), ("prep", "armor-system/tests/prep"))


class StepFailed(RuntimeError):
    pass


def step(n, title):
    print(f"\n[{n}] {title}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def verify_d1():
    """Re-hash every D1 file in the inventory and every fetched ARMOR member against the approved checksums."""
    import validate_d1                                            # data/tools, put on sys.path by prep.paths
    mdir = paths.MANIFESTS_298_43
    if not os.path.isdir(paths.D1_DIR):
        raise StepFailed(f"D1 release not found at {paths.D1_DIR} (obtain it with: python3 -I armor-system/data/tools/fetch_d1.py)")
    if os.path.exists(os.path.join(paths.RAW, "extracted", "D2")):
        raise StepFailed("a D2 directory exists locally; D2 must not be downloaded for this project")
    inv = validate_d1.verify_inventory(paths.D1_DIR, os.path.join(mdir, "file_inventory.csv"))
    problems = list(inv["problems"])
    with open(os.path.join(mdir, "checksums.json")) as fh:
        members = json.load(fh)["fetched_members"]
    for m in members:
        p = os.path.join(paths.RAW, m["local_path"])
        if not os.path.isfile(p):
            problems.append(f"missing: {m['local_path']}")
        elif os.path.getsize(p) != m["size"] or sha256_file(p) != m["sha256"]:
            problems.append(f"changed: {m['local_path']}")
    with open(os.path.join(mdir, "file_inventory.csv"), newline="") as fh:
        n_inv = sum(1 for _ in csv.DictReader(fh))
    if inv["files_checked"] == 0 or inv["files_checked"] != n_inv or not members:
        raise StepFailed("D1 inventory or fetched-member list is empty; nothing was verified")
    print(f"  D1 inventory files: {inv['files_checked']}, ARMOR members: {len(members)}, problems: {problems or 'none'}")
    if problems:
        raise StepFailed("D1 source files do not match the approved checksums: " + "; ".join(problems))
    return {"inventory_files_checked": inv["files_checked"], "armor_members_checked": len(members), "problems": [], "d2_present": False,
            "sources": ["data/manifests/armor_d1/file_inventory.csv", "data/manifests/armor_d1/checksums.json"]}


def digest_of(npz_path):
    import numpy as np
    z = np.load(npz_path)
    return ms.feature_cache_digest(z["timestamps"], z["X"])


def rebuild_cache(expected):
    tmp = tempfile.mkdtemp(prefix="m1_cache_")
    try:
        t0 = time.time()
        out = loader.build_cache(out=os.path.join(tmp, "d1_features_v1.npz"))
        rebuilt = digest_of(out)
        print(f"  clean rebuild from source: {rebuilt} ({time.time() - t0:.1f} s)")
        on_disk = digest_of(paths.CACHE_FILE) if os.path.exists(paths.CACHE_FILE) else None
        if expected is not None and rebuilt != expected:
            raise StepFailed(f"rebuilt feature cache hash {rebuilt} != manifest {expected}")
        replaced = False
        if on_disk != rebuilt:
            os.makedirs(paths.CACHE_DIR, exist_ok=True)
            shutil.copyfile(out, paths.CACHE_FILE)
            replaced = True
            print("  data/processed/ cache was " + ("missing" if on_disk is None else "stale") + ": replaced with the clean rebuild")
        else:
            print("  data/processed/ cache equals the clean rebuild")
        return {"rebuilt_digest": rebuilt, "on_disk_digest_before": on_disk, "rebuilt_equals_manifest": expected is not None, "on_disk_was_current": on_disk == rebuilt, "replaced_on_disk_cache": replaced}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def regenerate(update):
    committed = {"manifest": os.path.join(paths.ART, "splits", f"{ms.MANIFEST_NAME}.json"), "scaler": os.path.join(paths.ART, "preprocessing", f"{ms.SCALER_NAME}.npz")}
    runs = []
    for i in (1, 2):
        out = tempfile.mkdtemp(prefix=f"m1_gen{i}_")
        p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "model1_pipeline.py"), "--out-dir", out], capture_output=True, text=True, cwd=tempfile.gettempdir())
        if p.returncode != 0:
            raise StepFailed("artifact generation failed:\n" + (p.stdout + p.stderr)[-1500:])
        runs.append({"manifest": read_bytes(os.path.join(out, "splits", f"{ms.MANIFEST_NAME}.json")), "scaler": read_bytes(os.path.join(out, "preprocessing", f"{ms.SCALER_NAME}.npz"))})
        shutil.rmtree(out, ignore_errors=True)
    mi, si = runs[0]["manifest"] == runs[1]["manifest"], runs[0]["scaler"] == runs[1]["scaler"]
    have = all(os.path.exists(p) for p in committed.values())
    eq = have and read_bytes(committed["manifest"]) == runs[0]["manifest"] and read_bytes(committed["scaler"]) == runs[0]["scaler"]
    h = {k: hashlib.sha256(runs[0][k]).hexdigest() for k in runs[0]}
    print(f"  run1 == run2: manifest {mi}, scaler {si}; equal to committed artifacts: {eq}\n  manifest sha256 {h['manifest']}\n  scaler   sha256 {h['scaler']}")
    if not (mi and si):
        raise StepFailed("two independent generation runs are not byte-identical")
    if not eq:
        if not update:
            raise StepFailed("committed M1-03 artifacts are missing or differ from a fresh deterministic regeneration (rerun with --update to rewrite them)")
        for k, p in committed.items():
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(runs[0][k])
        print("  --update: artifacts rewritten")
        eq = True
    return {"runs": 2, "manifest_identical": mi, "scaler_identical": si, "equal_to_committed": eq, "manifest_file_sha256": h["manifest"], "scaler_file_sha256": h["scaler"]}


def run_suite(name, rel):
    env = dict(os.environ, M1_REQUIRE_REAL_DATA="1")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-I", "-B", "-m", "unittest", "discover", "-s", os.path.join(REPO, rel), "-p", "test_*.py"], capture_output=True, text=True, cwd=REPO, env=env)
    out = p.stderr + p.stdout
    m = re.search(r"Ran (\d+) tests?", out)
    num = lambda key: int(x.group(1)) if (x := re.search(key + r"=(\d+)", out)) else 0  # noqa: E731
    res = {"suite": name, "directory": rel, "ran": int(m.group(1)) if m else 0, "failures": num("failures"), "errors": num("errors"), "skipped": num("skipped"),
           "result": "OK" if p.returncode == 0 else "FAILED", "seconds": round(time.time() - t0, 1)}
    print(f"  {name}: ran {res['ran']}, failures {res['failures']}, errors {res['errors']}, skipped {res['skipped']} -> {res['result']}")
    if p.returncode != 0 or res["ran"] == 0:
        res["result"] = "FAILED"
        print(out[-2500:])
    return res


def run_tests():
    suites = [run_suite(n, r) for n, r in SUITES]
    total = {k: sum(s[k] for s in suites) for k in ("ran", "failures", "errors", "skipped")}
    if any(s["result"] != "OK" or s["ran"] == 0 for s in suites):
        raise StepFailed("test suites failed")
    if total["skipped"]:
        raise StepFailed(f"{total['skipped']} tests were skipped; real-data tests must all run")
    return {"command": "M1_REQUIRE_REAL_DATA=1 .venv/bin/python -I -B -m unittest discover -s armor-system/tests/<data|prep> -p 'test_*.py'",
            "skipping_disabled": True, "suites": suites, **total}


def git(*a):
    p = subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO)
    if p.returncode != 0:
        raise StepFailed(f"git {' '.join(a)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def environment():
    import numpy
    import torch
    return {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}", "numpy": numpy.__version__, "torch": torch.__version__, "device": "cpu"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-evidence", action="store_true", help="write armor-system/data/artifacts/d1/evidence/evidence_m1_03.json")
    ap.add_argument("--update", action="store_true", help="rewrite the M1-03 artifacts if they differ from a fresh regeneration")
    args = ap.parse_args(argv)
    t0 = time.time()
    try:
        manifest_path = os.path.join(paths.ART, "splits", f"{ms.MANIFEST_NAME}.json")
        if not os.path.exists(manifest_path) and not args.update:
            raise StepFailed(f"required artifact missing: {os.path.relpath(manifest_path, REPO)} (run model1_pipeline.py or use --update)")
        expected = None                                   # first generation (--update): nothing to compare the rebuild with yet
        if os.path.exists(manifest_path):
            with open(manifest_path) as fh:
                expected = json.load(fh)["inputs"]["feature_cache_content_sha256"]
        step(1, "verify D1 source files against the approved checksums")
        d1 = verify_d1()
        step(2, "clean feature-cache rebuild from source")
        cache = rebuild_cache(expected)
        step(3, "regenerate M1-03 artifacts twice (independent processes) and compare")
        repro = regenerate(args.update)
        step(4, "run all tests with skipping disabled")
        tests = run_tests()
        step(5, "evidence")
        g = {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "clean": not git("status", "--porcelain")}
        evp = os.path.join(REPO, ev.EVIDENCE_PATH)
        if args.write_evidence:
            doc = ev.build(REPO, d1_verification=d1, cache_rebuild=cache, reproducibility=repro, tests=tests, command=COMMAND, environment=environment(), git=g)
            with open(evp, "w") as fh:
                json.dump(doc, fh, indent=2)
                fh.write("\n")
            print("  wrote", ev.EVIDENCE_PATH)
        if os.path.exists(evp):
            with open(evp) as fh:
                problems = ev.verify(json.load(fh), REPO)
            print("  evidence verification:", problems or "OK")
            if problems:
                raise StepFailed("evidence_m1_03.json does not match the tree: " + "; ".join(problems))
        else:
            raise StepFailed("evidence_m1_03.json is missing; run once with --write-evidence")
    except StepFailed as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1
    print(f"\nSUCCESS in {time.time() - t0:.1f} s: tests ran {tests['ran']}, skipped {tests['skipped']}, failures {tests['failures']}, errors {tests['errors']}")
    print("manifest file sha256 ", repro["manifest_file_sha256"], "\nscaler file sha256   ", repro["scaler_file_sha256"], "\nfeature cache sha256 ", cache["rebuilt_digest"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
