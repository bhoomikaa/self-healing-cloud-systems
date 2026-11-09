"""M1-03 artifact generation: fit-only scaler + calibration-split manifest for Model 1 (Linear 298-48). CPU only.

    .venv/bin/python -I armor-system/data/model1_pipeline.py                 # write the two artifacts
    .venv/bin/python -I armor-system/data/model1_pipeline.py --out-dir DIR   # write them somewhere else (determinism checks)
    .venv/bin/python -I armor-system/data/model1_pipeline.py --check         # regenerate in memory, compare, write nothing

Reads the git-ignored feature cache `data/processed/d1_features_v1.npz` (`pipeline.py --stage cache`) and the committed
artifacts under `data/artifacts/d1/`. It never touches an existing scaler or split artifact other than: only
`preprocessing/scaler_model1_fit_only_v1.npz` and `splits/model1_calibration_split_v1.json` are written.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from prep import loader, m1_masks, m1_split as ms, paths  # noqa: E402
from prep.common import read_json, sha256_file  # noqa: E402


def generate(cache=paths.CACHE_FILE, art_root=paths.ART):
    """Return (manifest_text, scaler_bytes, manifest, summary) computed from approved inputs. Writes nothing."""
    A = lambda *p: os.path.join(art_root, *p)  # noqa: E731
    schema = read_json(A("inspection", "schema.json"))
    outer_path = A("splits", "model1_split_101_25_84_v1.json")
    outer = read_json(outer_path)
    csv_path = A("splits", "case_split_assignments.csv")
    with open(csv_path, newline="") as fh:
        cases = ms.load_case_table(list(csv.DictReader(fh)))
    boundaries = ms.approved_boundaries(outer, cases)
    ts, X, _ = loader.load_arrays(cache)
    issues = ms.pp.validate_schema(X, ts, schema)
    if issues:
        raise SystemExit("feature cache fails schema validation: " + "; ".join(issues))
    groups = schema["modality_index_ranges"]
    cfg = ms.SplitConfig()
    arrays = ms.fit_scaler_fit_only(X, ts, groups, cfg)
    scaler_bytes = ms.npz_bytes(arrays)
    old = np.load(A("preprocessing", "scaler_model1_train_period_v1.npz"))
    applicable, _, report = m1_masks.applicable_entry_mask(schema, arrays["absent_entry"])
    m1_masks.check_expected_counts(report)
    m1_masks.check_mask_stability(X, applicable, {"fit": ts < cfg.tc - cfg.embargo_s, "calibration_region": (ts >= cfg.tc) & (ts < boundaries[0]),
                                                  "validation": (ts >= boundaries[0]) & (ts < boundaries[1]), "test": ts >= boundaries[1]}, schema)
    table = ms.compute_split(ts, cases, boundaries, cfg)
    manifest = ms.build_manifest(
        table, cases, ts, boundaries, scaler_arrays=arrays, scaler_file_sha256=ms.sha256_bytes(scaler_bytes),
        scaler_file=f"preprocessing/{ms.SCALER_NAME}.npz", outer=outer, outer_file="splits/model1_split_101_25_84_v1.json",
        outer_case_csv_sha256=sha256_file(csv_path), preprocessing_config_sha256=read_json(A("preprocessing", "preprocessing_config_v1.json"))["config_sha256"],
        schema=schema, cache_digest=ms.feature_cache_digest(ts, X), train_period_scaler=old, cfg=cfg)
    manifest["applicable_entry"] = {"rule": m1_masks.D1_MASK_RULE["name"], **report}
    manifest["content_sha256"] = ms.manifest_content_hash(manifest)
    return ms.dumps_manifest(manifest), scaler_bytes, manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=paths.ART, help="artifact root to write into (default: the committed artifacts)")
    ap.add_argument("--check", action="store_true", help="compare with the existing artifacts and write nothing")
    args = ap.parse_args(argv)
    text, sbytes, m = generate()
    mpath = os.path.join(args.out_dir, "splits", f"{ms.MANIFEST_NAME}.json")
    spath = os.path.join(args.out_dir, "preprocessing", f"{ms.SCALER_NAME}.npz")
    if args.check:
        same = os.path.exists(mpath) and open(mpath).read() == text and os.path.exists(spath) and open(spath, "rb").read() == sbytes
        print("artifacts identical to a fresh regeneration" if same else "artifacts DIFFER from a fresh regeneration")
        return 0 if same else 1
    for p, data in ((spath, sbytes), (mpath, text.encode())):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(data)
        print("wrote", os.path.relpath(p), len(data), "bytes sha256", ms.sha256_bytes(data))
    c = m["counts"]["roles"]
    print("manifest content_sha256", m["content_sha256"])
    print("fit", c["fit"]["windows"], "calibration", c["calibration"]["windows"], "validation", c["validation"]["windows"], "test", c["test"]["windows"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
