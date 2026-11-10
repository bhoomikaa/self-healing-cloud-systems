"""Compare the artifacts a pipeline run generated against the committed ones.

This is what makes a scheduled run worth anything. The DAG regenerates every
artifact in a disposable workspace; this script then checks the result still
matches what is committed.

Byte-identity is reported but **not** the pass condition, because three things
change the bytes without changing the result:

1.  Line endings. This repository is checked out on Windows with
    `core.autocrlf=true`, so a committed JSON file on disk has CRLF while the
    container writes LF. Measured on `inspection/schema.json`: 17794 bytes
    against 16898, a difference of exactly 896, which is its line count.
2.  Floating point across platforms. The committed artifacts were produced on
    macOS arm64 with Python 3.14; the DAG runs Linux x86_64 with Python 3.12,
    on identical library versions. Reductions land a few ulp apart. Measured on
    `eda/eda_findings.json`: 9 values differ, worst relative gap 8.4e-15.
3.  Font rasterisation. matplotlib renders the same figure to different PNG
    bytes on Linux and macOS, about 19% larger, because the available fonts
    differ.

So the comparison is done on content. JSON and CSV are parsed and compared
value by value, exactly for structure, strings and integers, and within a
relative tolerance for floats. Figures are checked for existence and validity
rather than bytes, because cross-platform byte-identity is not achievable for
them. Anything else is compared byte for byte.

What still fails the run: a missing artifact, a structural change, a changed
string or integer, or a float that moved by more than the tolerance. Those mean
the pipeline changed, not the platform.

Usage:
    python -I compare_artifacts.py --committed REPO --generated WORKSPACE --report OUT.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import struct
import sys

# Floats may move this far, relatively, without it counting as a change. The
# platform noise measured above is ~1e-14; a real pipeline change moves numbers
# by far more than 1e-9, so this separates the two cleanly.
FLOAT_RTOL = 1e-9

# Artifacts that record *when* and *where* a run happened rather than what it
# produced. They differ on every run by design.
VOLATILE = ("data/artifacts/d1/evidence",)

# Subtrees that hold generated artifacts and nothing else.
COMPARED = ("data/artifacts/d1", "data/manifests/armor_d1")


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
            if any(rel.startswith(v + "/") for v in VOLATILE):
                continue
            out[rel] = full
    return out


# --------------------------------------------------------------------- values

def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare_values(a, b, path, problems, drift):
    """Walk two parsed structures together. `a` is committed, `b` is generated."""
    if type(a) is not type(b) and not (_is_number(a) and _is_number(b)):
        problems.append(f"{path}: type changed, {type(a).__name__} -> {type(b).__name__}")
        return
    if isinstance(a, dict):
        for k in a.keys() - b.keys():
            problems.append(f"{path}.{k}: missing from the run")
        for k in b.keys() - a.keys():
            problems.append(f"{path}.{k}: present in the run but not committed")
        for k in a.keys() & b.keys():
            compare_values(a[k], b[k], f"{path}.{k}", problems, drift)
    elif isinstance(a, list):
        if len(a) != len(b):
            problems.append(f"{path}: length changed, {len(a)} -> {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare_values(x, y, f"{path}[{i}]", problems, drift)
    elif _is_number(a):
        if a == b:
            return
        if isinstance(a, int) and isinstance(b, int):
            problems.append(f"{path}: integer changed, {a} -> {b}")
            return
        denom = max(abs(a), abs(b))
        rel = abs(a - b) / denom if denom else abs(a - b)
        if rel > FLOAT_RTOL:
            problems.append(f"{path}: {a!r} -> {b!r}, relative gap {rel:.3e} exceeds {FLOAT_RTOL:.0e}")
        else:
            drift.append(rel)
    elif a != b:
        problems.append(f"{path}: {a!r} -> {b!r}")


def compare_json(pa, pb, problems, drift):
    with open(pa, encoding="utf-8") as fh:
        a = json.load(fh)
    with open(pb, encoding="utf-8") as fh:
        b = json.load(fh)
    compare_values(a, b, "", problems, drift)


def _cells(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def compare_csv(pa, pb, problems, drift):
    a, b = _cells(pa), _cells(pb)
    if len(a) != len(b):
        problems.append(f"row count changed, {len(a)} -> {len(b)}")
        return
    for r, (ra, rb) in enumerate(zip(a, b)):
        if len(ra) != len(rb):
            problems.append(f"row {r}: column count changed, {len(ra)} -> {len(rb)}")
            continue
        for c, (x, y) in enumerate(zip(ra, rb)):
            if x == y:
                continue
            try:
                fx, fy = float(x), float(y)
            except ValueError:
                problems.append(f"row {r} col {c}: {x!r} -> {y!r}")
                continue
            denom = max(abs(fx), abs(fy))
            rel = abs(fx - fy) / denom if denom else abs(fx - fy)
            if rel > FLOAT_RTOL:
                problems.append(f"row {r} col {c}: {x} -> {y}, relative gap {rel:.3e}")
            else:
                drift.append(rel)


def png_size(path):
    """Width and height from the IHDR chunk, without an image library."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", head[16:24])


def compare_png(pa, pb, problems):
    sa, sb = png_size(pa), png_size(pb)
    if sb is None:
        problems.append("the run did not produce a valid PNG")
        return None
    if sa is not None and sa != sb:
        problems.append(f"figure dimensions changed, {sa[0]}x{sa[1]} -> {sb[0]}x{sb[1]}")
    return sb


def compare_text(pa, pb, problems):
    with open(pa, encoding="utf-8") as fh:
        a = fh.read().replace("\r\n", "\n")
    with open(pb, encoding="utf-8") as fh:
        b = fh.read().replace("\r\n", "\n")
    if a != b:
        problems.append("text content differs (line endings already normalised)")


# ----------------------------------------------------------------------- main

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

    results, failures = [], []
    identical_bytes = 0
    all_drift = []

    for rel in sorted(committed):
        pa = committed[rel]
        if rel not in generated:
            failures.append({"file": rel, "problems": ["missing from the run"]})
            results.append({"file": rel, "verdict": "missing"})
            continue
        pb = generated[rel]

        same_bytes = sha256(pa) == sha256(pb)
        if same_bytes:
            identical_bytes += 1
            results.append({"file": rel, "verdict": "identical"})
            continue

        problems, drift = [], []
        ext = os.path.splitext(rel)[1].lower()
        extra = {}
        try:
            if ext == ".json":
                compare_json(pa, pb, problems, drift)
            elif ext == ".csv":
                compare_csv(pa, pb, problems, drift)
            elif ext == ".png":
                dims = compare_png(pa, pb, problems)
                extra = {"dimensions": f"{dims[0]}x{dims[1]}" if dims else None,
                         "committed_bytes": os.path.getsize(pa), "generated_bytes": os.path.getsize(pb),
                         "note": "figures are not byte-comparable across platforms; fonts rasterise differently"}
            elif ext in (".md", ".txt"):
                compare_text(pa, pb, problems)
            else:
                problems.append("bytes differ and this file type has no content comparison")
        except Exception as exc:                                  # noqa: BLE001
            problems.append(f"could not be compared: {type(exc).__name__}: {exc}")

        all_drift.extend(drift)
        row = {"file": rel, "verdict": "changed" if problems else
               ("equivalent" if not drift else "within_tolerance"), **extra}
        if drift:
            row["values_within_tolerance"] = len(drift)
            row["worst_relative_gap"] = max(drift)
        results.append(row)
        if problems:
            failures.append({"file": rel, "problems": problems[:20],
                             "problem_count": len(problems)})

    verdicts = {}
    for r in results:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1

    report = {
        "compared_subtrees": list(COMPARED),
        "excluded_as_volatile": list(VOLATILE),
        "float_relative_tolerance": FLOAT_RTOL,
        "files_compared": len(committed),
        "identical_bytes": identical_bytes,
        "verdicts": verdicts,
        "worst_relative_gap": max(all_drift) if all_drift else 0.0,
        "values_within_tolerance": len(all_drift),
        "not_committed": sorted(set(generated) - set(committed)),
        "failures": failures,
        "files": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
        fh.write("\n")

    print(f"compared {len(committed)} committed artifacts")
    print(f"  byte-identical       {identical_bytes}")
    for v in ("equivalent", "within_tolerance", "changed", "missing"):
        if verdicts.get(v):
            print(f"  {v:<20} {verdicts[v]}")
    if all_drift:
        print(f"  worst relative gap   {max(all_drift):.3e} over {len(all_drift)} values "
              f"(tolerance {FLOAT_RTOL:.0e})")
    print(f"report: {a.report}")

    if failures:
        print("\nThe run did not reproduce the committed artifacts:")
        for f in failures[:15]:
            print(f"  {f['file']}  ({f.get('problem_count', len(f['problems']))} problem(s))")
            for p in f["problems"][:3]:
                print(f"      {p}")
        if len(failures) > 15:
            print(f"  ... and {len(failures) - 15} more files")
        print("\nThis is a real change: structure, a string, an integer, or a float beyond "
              f"{FLOAT_RTOL:.0e}. Line endings, cross-platform float noise and figure "
              "rendering are already accounted for.")
        return 1

    print("\nevery committed artifact reproduced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
