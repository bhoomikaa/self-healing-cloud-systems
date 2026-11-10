"""Compare the artifacts a pipeline run generated against the committed ones.

This is what makes a scheduled run worth anything. The DAG regenerates every
artifact in a disposable workspace; this script then checks that the result is
byte-identical to what is committed. Reproducibility is the claim, so a
byte-for-byte diff is the test.

Exits 1 and prints the drifted files if anything differs. Also writes a JSON
report so a red task can be diagnosed without re-reading the log.

Usage:
    python -I compare_artifacts.py --committed REPO --generated WORKSPACE --report OUT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# Artifacts that record *when* and *where* a run happened rather than what it
# produced. They differ on every run by design, so comparing them byte for byte
# would make the check always fail and therefore meaningless.
VOLATILE = (
    os.path.join("data", "artifacts", "d1", "evidence"),
)

# Subtrees that hold generated artifacts and nothing else.
COMPARED = (
    os.path.join("data", "artifacts", "d1"),
    os.path.join("data", "manifests", "armor_d1"),
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(root, subtree):
    base = os.path.join(root, subtree)
    if not os.path.isdir(base):
        return {}
    out = {}
    for dirpath, _dirnames, filenames in os.walk(base):
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if any(rel.startswith(v.replace(os.sep, "/") + "/") for v in VOLATILE):
                continue
            out[rel] = full
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--committed", required=True, help="armor-system directory of the read-only repository")
    ap.add_argument("--generated", required=True, help="armor-system directory of the run workspace")
    ap.add_argument("--report", required=True)
    a = ap.parse_args(argv)

    committed, generated = {}, {}
    for sub in COMPARED:
        committed.update(walk(a.committed, sub))
        generated.update(walk(a.generated, sub))

    same, drifted, missing, extra = [], [], [], []
    for rel in sorted(committed):
        if rel not in generated:
            missing.append(rel)
            continue
        c, g = sha256(committed[rel]), sha256(generated[rel])
        (same if c == g else drifted).append(
            rel if c == g else {"file": rel, "committed_sha256": c, "generated_sha256": g,
                                "committed_bytes": os.path.getsize(committed[rel]),
                                "generated_bytes": os.path.getsize(generated[rel])})
    extra = sorted(set(generated) - set(committed))

    report = {
        "compared_subtrees": [s.replace(os.sep, "/") for s in COMPARED],
        "excluded_as_volatile": [v.replace(os.sep, "/") for v in VOLATILE],
        "files_compared": len(committed),
        "identical": len(same),
        "drifted": drifted,
        "missing_from_run": missing,
        "not_committed": extra,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    with open(a.report, "w") as fh:
        json.dump(report, fh, indent=1)
        fh.write("\n")

    print(f"compared {len(committed)} committed artifacts: {len(same)} identical, "
          f"{len(drifted)} drifted, {len(missing)} missing, {len(extra)} not committed")
    print(f"report: {a.report}")

    if drifted or missing:
        for d in drifted[:25]:
            print(f"  DRIFT   {d['file']}  {d['committed_bytes']}B -> {d['generated_bytes']}B")
        if len(drifted) > 25:
            print(f"  ... and {len(drifted) - 25} more")
        for m in missing[:25]:
            print(f"  MISSING {m}")
        print("\nThe run did not reproduce the committed artifacts. Either the pipeline "
              "changed, the pinned dependency versions changed, or the committed "
              "artifacts were generated with a different environment.")
        return 1
    print("every committed artifact reproduced byte for byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
