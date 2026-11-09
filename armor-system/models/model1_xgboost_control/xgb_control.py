"""XGBoost fault-classification control for Model 1 (Linear 298-49). CPU only, fixed configuration, five seeds.

    .venv/bin/python -I armor-system/models/model1_xgboost_control/xgb_control.py                    # write results
    .venv/bin/python -I armor-system/models/model1_xgboost_control/xgb_control.py --write-evidence   # + 2 clean reruns, tests, evidence file

Trains on the 88 FITTING cases only (calibration, validation and the held-out evaluation set never reach a fit, a
class weight or a feature decision) and evaluates the fixed configuration on the 25 validation cases. There is no
hyper-parameter search, no early stopping and no way to request the held-out set from this script.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "armor-system", "data"))
sys.path.insert(0, HERE)

import metrics as mt  # noqa: E402
from prep import m1_case_features as cf, m1_evidence as m1ev  # noqa: E402   (m1_dataset imports torch: only ever imported in the feature-dump process, see below)

CONFIG_PATH = os.path.join(HERE, "config_v1.json")
RESULTS_DIR = os.path.join(HERE, "results", "xgb_control_v1")
EVIDENCE_NAME = "evidence_xgb_control_v1.json"
K = 5
FORBIDDEN_KEY = re.compile(r"(^|_)test(_|$)")


class FinalTestLocked(PermissionError):
    pass


def final_test(*_a, **_k):
    """Future pathway (see README): blocked in this milestone."""
    raise FinalTestLocked("The held-out evaluation set is locked: it is evaluated once, together with Model 1, after both configurations are frozen.")


def canonical_sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def array_sha256(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def load_config(path=CONFIG_PATH):
    with open(path) as fh:
        cfg = json.load(fh)
    if cfg["xgboost_params"]["num_class"] != K or len(cfg["class_order"]) != K or tuple(cfg["class_order"]) != mt.CLASS_NAMES:
        raise ValueError("config class order differs from the Model 1 order")
    if cfg["early_stopping"] is not None or cfg["hyperparameter_search"]:
        raise ValueError("this control forbids early stopping and hyper-parameter search")
    if cfg["partitions"]["calibration_cases_used"] != 0 or cfg["partitions"]["held_out_cases_used"] != 0:
        raise ValueError("config must not use calibration or held-out cases")
    return cfg, canonical_sha256(cfg)


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(jsonable(obj), fh, indent=2)
        fh.write("\n")


def sha256_file(path):
    return m1ev.sha256_file(path)


def versions():
    import scipy
    import xgboost
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "xgboost": xgboost.__version__}


# ---------------------------------------------------------------------------------------------------------
# training and evaluation
# ---------------------------------------------------------------------------------------------------------
def train_predict(cfg, fit, val, class_weights, seed):
    """Fit on the fitting cases only; return validation class probabilities (n_val, 5) and a model digest."""
    import xgboost as xgb
    params = dict(cfg["xgboost_params"], seed=int(seed))
    dfit = xgb.DMatrix(fit["X"], label=fit["y"], weight=class_weights[fit["y"]])
    dval = xgb.DMatrix(val["X"])                                              # no labels, no weights: nothing from validation is fitted
    booster = xgb.train(params, dfit, num_boost_round=cfg["num_boost_round"], verbose_eval=False)
    proba = booster.predict(dval).astype(np.float64)
    if proba.shape != (len(val["X"]), K) or not np.isfinite(proba).all():
        raise RuntimeError("unexpected prediction shape or non-finite probabilities")
    return proba, hashlib.sha256(bytes(booster.save_raw("ubj"))).hexdigest()


def evaluate(y_true, y_pred):
    s = mt.summary(y_true, y_pred, K)
    s["macro_f1_excluding_node_memory"] = mt.macro_f1_excluding(y_true, y_pred, exclude=4, k=K)
    s["family_diagnostic"] = mt.family_diagnostic(y_true, y_pred)
    s["per_class_names"] = list(mt.CLASS_NAMES)
    return s


def mean_sd(values):
    v = np.array([x for x in values if x is not None], np.float64)
    if len(v) == 0:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "n": int(len(v)), "values": [float(x) for x in v]}


def aggregate(per_seed):
    """Mean +- sample SD (ddof=1) across seeds. SD = training randomness only."""
    agg = {k: mean_sd([m[k] for m in per_seed]) for k in ("accuracy", "macro_f1", "weighted_f1", "macro_f1_excluding_node_memory")}
    agg["per_class"] = {key: [mean_sd([m["per_class"][key][i] for m in per_seed]) for i in range(K)] for key in ("precision", "recall", "f1")}
    agg["per_class_support"] = per_seed[0]["per_class"]["support"]
    agg["mean_confusion_matrix"] = np.mean([np.array(m["confusion_matrix"]) for m in per_seed], axis=0).tolist()
    agg["family"] = {}
    for f in ("container", "node"):
        fam = [m["family_diagnostic"]["families"][f] for m in per_seed]
        agg["family"][f] = {"n_cases": fam[0]["n_cases"],
                            "exactly_correct": mean_sd([x["exactly_correct"] for x in fam]),
                            "wrong_inside_family": mean_sd([x["wrong_inside_family"] for x in fam]),
                            "wrong_outside_family": mean_sd([x["wrong_outside_family"] for x in fam]),
                            "family_level_accuracy": mean_sd([x["family_level_accuracy"]["value"] for x in fam]),
                            "conditional_within_family_accuracy": mean_sd([x["conditional_within_family_accuracy"]["value"] for x in fam])}
    return agg


def write_predictions(path, val, proba, pred):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["case_index", "group_id", "true_class", "true_name", "predicted_class", "predicted_name"] + [f"p_{i}" for i in range(K)])
        for i in range(len(pred)):
            w.writerow([int(val["case_index"][i]), int(val["group_id"][i]), int(val["y"][i]), mt.CLASS_NAMES[int(val["y"][i])],
                        int(pred[i]), mt.CLASS_NAMES[int(pred[i])]] + [f"{p:.9g}" for p in proba[i]])


def dump_features(dirpath):
    """Runs in its own process: loads the approved data (imports torch) and saves the two case matrices. Never imports xgboost.

    torch and xgboost each bundle an OpenMP runtime and abort when both are loaded into one macOS process, so the two
    halves of this experiment live in separate processes.
    """
    from prep import m1_dataset as md
    D = md.Model1Data.load()
    mats = {r: cf.build_case_matrix(D, r) for r in cf.ALLOWED_ROLES}
    timing = observation_timing(D)
    os.makedirs(dirpath, exist_ok=True)
    arrays = {f"{r}_{k}": mats[r][k] for r in mats for k in ("X", "y", "case_index", "group_id", "n_windows", "n_minutes")}
    np.savez(os.path.join(dirpath, "features.npz"), **arrays)
    write_json(os.path.join(dirpath, "features_meta.json"), {
        "feature_names": mats["fit"]["feature_names"],
        "inputs": {"calibration_manifest_content_sha256": D.manifest["content_sha256"], "fit_only_scaler_file_sha256": D.manifest["fit_only_scaler"]["file_sha256"],
                   "feature_cache_content_sha256": D.manifest["inputs"]["feature_cache_content_sha256"],
                   "outer_split_content_sha256": D.manifest["inputs"]["outer_split_manifest"]["content_sha256"]},
        "timing": timing})


def observation_timing(D):
    """Exact observation times behind every case vector, plus runtime isolation checks (raises on a violation).

    Offsets are seconds relative to the documented case time `t_case` (ARMOR's injection time; case times are not
    minute-aligned, sample times are). A prediction for a case can be produced no earlier than its last sample time.
    """
    t, cfg, (b1, b2) = D.table, D.cfg, D.boundaries
    rows, minute_sets = [], {}
    for role in cf.ALLOWED_ROLES:
        by = {}
        for r in D.selection([role], ["event_associated"], drop_ambiguous=True):
            by.setdefault(int(t.case_index[r]), []).append(int(r))
        minute_sets[role] = set()
        for c in sorted(by):
            ct = int(D.cases["timestamp"][c])
            idx = cf.case_minutes(t, by[c])
            ts = D.ts[idx]
            minute_sets[role].update(int(x) for x in ts)
            rows.append({"case_index": c, "role": role, "case_ts": ct, "first_sample_ts": int(ts[0]), "last_sample_ts": int(ts[-1]),
                         "first_offset_s": int(ts[0] - ct), "last_offset_s": int(ts[-1] - ct), "n_samples": int(len(ts)),
                         "n_before_injection": int((ts < ct).sum()), "n_at_or_after_injection": int((ts >= ct).sum()),
                         "earliest_prediction_ts": int(ts[-1]), "earliest_prediction_offset_s": int(ts[-1] - ct)})
    fit_m, val_m = minute_sets["fit"], minute_sets["validation"]
    checks = {"fit_samples_before_fit_cutoff": max(fit_m) < cfg.tc - cfg.embargo_s, "validation_samples_inside_validation_period": min(val_m) >= b1 and max(val_m) < b2,
              "samples_shared_by_fit_and_validation": len(fit_m & val_m), "samples_in_calibration_region": sum(1 for x in fit_m | val_m if cfg.tc <= x < b1),
              "samples_in_held_out_period": sum(1 for x in fit_m | val_m if x >= b2)}
    if not (checks["fit_samples_before_fit_cutoff"] and checks["validation_samples_inside_validation_period"] and checks["samples_shared_by_fit_and_validation"] == 0
            and checks["samples_in_calibration_region"] == 0 and checks["samples_in_held_out_period"] == 0):
        raise RuntimeError(f"observation isolation violated: {checks}")
    summ = {}
    for role in cf.ALLOWED_ROLES:
        rr = [r for r in rows if r["role"] == role]
        for k in ("first_offset_s", "last_offset_s", "n_before_injection", "n_at_or_after_injection", "earliest_prediction_offset_s"):
            v = [r[k] for r in rr]
            summ.setdefault(role, {})[k] = {"min": min(v), "median": float(np.median(v)), "max": max(v)}
    return {"unit": "seconds relative to the documented case time", "per_case": rows, "summary": summ, "isolation_checks": checks,
            "reading": ["Each case vector uses roughly 5-6 samples before the documented injection time and 5-6 samples at or after it (about t_case - 359 s to t_case + 359 s).",
                        "A case prediction cannot be produced before the last of those samples, about 4.7-6 minutes after the documented injection time.",
                        "The windows are anchored on the documented injection time, which a deployed system would not know. This is fault CLASSIFICATION of an already-localised incident window, not early incident detection.",
                        "A fair Transformer comparison must classify the same cases from the same approved windows (same samples, same last-sample availability, same case-level aggregation), with the same 88 fitting cases."]}


def load_features():
    tmp = tempfile.mkdtemp(prefix="xgb_features_")
    p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--dump-features", tmp], capture_output=True, text=True, cwd=tempfile.gettempdir())
    if p.returncode != 0:
        raise SystemExit("feature extraction failed:\n" + (p.stdout + p.stderr)[-1500:])
    z = np.load(os.path.join(tmp, "features.npz"))
    with open(os.path.join(tmp, "features_meta.json")) as fh:
        meta = json.load(fh)
    timing_meta = meta.pop("timing")
    out = {r: {k: z[f"{r}_{k}"] for k in ("X", "y", "case_index", "group_id", "n_windows", "n_minutes")} | {"feature_names": meta["feature_names"], "role": r} for r in cf.ALLOWED_ROLES}
    meta["timing"] = timing_meta
    return out["fit"], out["validation"], meta


def run_experiment(out_dir=RESULTS_DIR, cfg_path=CONFIG_PATH):
    cfg, cfg_sha = load_config(cfg_path)
    fit, val, meta = load_features()
    n_feat = fit["X"].shape[1]
    if (len(fit["y"]), len(val["y"])) != (cf.APPROVED_CASES["fit"], cf.APPROVED_CASES["validation"]) or n_feat != cf.APPROVED_FEATURES or val["X"].shape[1] != n_feat:
        raise RuntimeError(f"unexpected case or feature counts: fit {fit['X'].shape}, validation {val['X'].shape}")
    if set(fit["case_index"]) & set(val["case_index"]):
        raise RuntimeError("a case appears in both fit and validation")
    if set(fit["group_id"]) & set(val["group_id"]):
        raise RuntimeError("an incident group spans fit and validation")
    weights = cf.inverse_frequency_weights(fit["y"], K)
    fit_counts, val_counts = np.bincount(fit["y"], minlength=K), np.bincount(val["y"], minlength=K)
    maj = mt.majority_class(fit["y"], K)

    per_seed, seed_files = [], {}
    for seed in cfg["seeds"]:
        proba, model_sha = train_predict(cfg, fit, val, weights, seed)
        pred = proba.argmax(1)
        m = evaluate(val["y"], pred)
        m.update({"seed": int(seed), "model_ubj_sha256": model_sha})
        write_json(os.path.join(out_dir, "per_seed", f"seed_{seed}_metrics.json"), m)
        write_predictions(os.path.join(out_dir, "per_seed", f"seed_{seed}_predictions.csv"), val, proba, pred)
        per_seed.append(m)
        print(f"seed {seed}: macro-F1 {m['macro_f1']:.4f} accuracy {m['accuracy']:.4f} weighted-F1 {m['weighted_f1']:.4f}", flush=True)

    maj_pred = np.full(len(val["y"]), maj, np.int64)
    majority = evaluate(val["y"], maj_pred)
    majority.update({"predicted_class": maj, "predicted_name": mt.CLASS_NAMES[maj], "fit_class_counts": fit_counts.tolist(),
                     "rule": cfg["majority_control"]})
    write_json(os.path.join(out_dir, "majority_control.json"), majority)

    write_json(os.path.join(out_dir, "observation_timing.json"), meta["timing"])
    write_json(os.path.join(out_dir, "config_used.json"), {"config_sha256": cfg_sha, **cfg})
    write_json(os.path.join(out_dir, "case_features_provenance.json"), {
        "cases": {"fit": len(fit["y"]), "validation": len(val["y"]), "calibration_used": 0, "held_out_used": 0},
        "feature_matrix_shapes": {"fit": list(fit["X"].shape), "validation": list(val["X"].shape)},
        "feature_names_sha256": canonical_sha256(fit["feature_names"]), "first_features": fit["feature_names"][:8], "n_features": n_feat,
        "array_sha256": {"X_fit": array_sha256(fit["X"]), "y_fit": array_sha256(fit["y"]), "X_validation": array_sha256(val["X"]), "y_validation": array_sha256(val["y"]),
                         "fit_case_index": array_sha256(fit["case_index"]), "validation_case_index": array_sha256(val["case_index"])},
        "fit_case_index": fit["case_index"], "validation_case_index": val["case_index"],
        "class_order": list(mt.CLASS_NAMES), "class_distribution": {"fit": fit_counts.tolist(), "validation": val_counts.tolist()},
        "class_weights_from_fit_cases": weights.tolist(),
        "windows_per_case": {"fit": [int(fit["n_windows"].min()), int(fit["n_windows"].max())], "validation": [int(val["n_windows"].min()), int(val["n_windows"].max())]},
        "distinct_minutes_per_case": {"fit": [int(fit["n_minutes"].min()), int(fit["n_minutes"].max())], "validation": [int(val["n_minutes"].min()), int(val["n_minutes"].max())]},
        "incident_groups": {"fit": int(len(set(fit["group_id"]))), "validation": int(len(set(val["group_id"]))), "shared": 0},
        "inputs": meta["inputs"],
    })
    write_json(os.path.join(out_dir, "validation_summary.json"), {
        "experiment": cfg["name"], "config_sha256": cfg_sha, "evaluated_on": "25 validation cases (fitted on 88 fitting cases)",
        "seeds": cfg["seeds"], "xgboost_params": cfg["xgboost_params"], "num_boost_round": cfg["num_boost_round"], "versions": versions(),
        "primary_metric": "five-class macro-F1 (all five labels, zero_division = 0)",
        "per_seed": [{k: m[k] for k in ("seed", "accuracy", "macro_f1", "weighted_f1", "macro_f1_excluding_node_memory", "model_ubj_sha256")} | {"per_class_recall": m["per_class"]["recall"]} for m in per_seed],
        "mean_sd_across_seeds": aggregate(per_seed),
        "majority_control": {k: majority[k] for k in ("predicted_name", "accuracy", "macro_f1", "weighted_f1", "macro_f1_excluding_node_memory")} | {"per_class_recall": majority["per_class"]["recall"]},
        "validation_support": dict(zip(mt.CLASS_NAMES, val_counts.tolist())),
        "limitations": [
            "Seed SD measures training randomness only. It is not uncertainty across independent incidents; 25 validation cases give wide uncertainty.",
            "Node Memory has ONE validation case, so its recall can only be 0 or 1 and per-seed values are not a stable estimate.",
            "Node CPU has 3 validation cases. Macro-F1 therefore moves in large steps.",
            "Labels come from documented fault windows (event_associated), not verified minute-level anomalies.",
            "No configuration was tuned: parameters were fixed in advance. The held-out evaluation set was not used.",
        ],
    })
    return per_seed, majority


# ---------------------------------------------------------------------------------------------------------
# reproducibility, tests and evidence
# ---------------------------------------------------------------------------------------------------------
RESULT_FILES_EXCLUDED = {EVIDENCE_NAME}


def result_files(root):
    out = []
    for d, _, files in os.walk(root):
        for f in files:
            if f not in RESULT_FILES_EXCLUDED:
                out.append(os.path.relpath(os.path.join(d, f), root))
    return sorted(out)


def read_bytes(p):
    with open(p, "rb") as fh:
        return fh.read()


def compare_dirs(a, b):
    fa, fb = result_files(a), result_files(b)
    same = fa == fb and all(read_bytes(os.path.join(a, f)) == read_bytes(os.path.join(b, f)) for f in fa)
    return same, fa, fb


def forbidden_keys(obj, path=""):
    """Keys that would indicate held-out-set results in a saved JSON document."""
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if FORBIDDEN_KEY.search(str(k)):
                bad.append(f"{path}/{k}")
            bad += forbidden_keys(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += forbidden_keys(v, f"{path}[{i}]")
    return bad


def run_suites():
    env = dict(os.environ, M1_REQUIRE_REAL_DATA="1")
    suites, total = [], {"ran": 0, "failures": 0, "errors": 0, "skipped": 0}
    for name in ("data", "prep"):
        t0 = time.time()
        p = subprocess.run([sys.executable, "-I", "-B", "-m", "unittest", "discover", "-s", os.path.join(REPO, "armor-system", "tests", name), "-p", "test_*.py"],
                           capture_output=True, text=True, cwd=REPO, env=env)
        out = p.stderr + p.stdout
        m = re.search(r"Ran (\d+) tests?", out)
        num = lambda key: int(x.group(1)) if (x := re.search(key + r"=(\d+)", out)) else 0  # noqa: E731
        s = {"suite": name, "ran": int(m.group(1)) if m else 0, "failures": num("failures"), "errors": num("errors"), "skipped": num("skipped"),
             "result": "OK" if p.returncode == 0 else "FAILED", "seconds": round(time.time() - t0, 1)}
        print(f"  {name}: ran {s['ran']}, failures {s['failures']}, errors {s['errors']}, skipped {s['skipped']} -> {s['result']}")
        suites.append(s)
        for k in total:
            total[k] += s[k]
    if any(s["result"] != "OK" or s["ran"] == 0 for s in suites) or total["skipped"]:
        raise SystemExit("test suites failed or skipped tests")
    return {"suites": suites, **total, "skipping_disabled": True}


def git(*a):
    p = subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO)
    if p.returncode != 0:
        raise SystemExit(f"git {' '.join(a)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


SOURCE_FILES = (
    "armor-system/data/prep/m1_case_features.py",
    "armor-system/models/model1_xgboost_control/metrics.py",
    "armor-system/models/model1_xgboost_control/xgb_control.py",
    "armor-system/models/model1_xgboost_control/config_v1.json",
    "armor-system/models/model1_xgboost_control/requirements.txt",
    "armor-system/models/model1_xgboost_control/README.md",
    "armor-system/tests/prep/test_m1_xgb_features.py",
    "armor-system/tests/prep/test_m1_xgb_metrics.py",
    "armor-system/tests/prep/test_m1_xgb_control.py",
)
COMMANDS = {
    "run": ".venv/bin/python -I armor-system/models/model1_xgboost_control/xgb_control.py",
    "evidence": ".venv/bin/python -I armor-system/models/model1_xgboost_control/xgb_control.py --write-evidence",
    "tests": "M1_REQUIRE_REAL_DATA=1 .venv/bin/python -I -B -m unittest discover -s armor-system/tests/<data|prep> -p 'test_*.py'",
}


def build_evidence(results_dir, repro, suites, git_info, repo=REPO):
    rows, set_hash = m1ev.source_set(repo, SOURCE_FILES)
    prov = json.load(open(os.path.join(results_dir, "case_features_provenance.json")))
    summ = json.load(open(os.path.join(results_dir, "validation_summary.json")))
    return {
        "evidence_name": "evidence_xgb_control_v1", "evidence_version": "1.0", "linear_issue": "298-49",
        "scope": "XGBoost fault-classification control only. Historical 298-44, M1-02 and M1-03 evidence is not rewritten.",
        "config_sha256": summ["config_sha256"], "versions": summ["versions"], "seeds": summ["seeds"], "xgboost_params": summ["xgboost_params"],
        "dataset_hashes": prov["inputs"] | {"array_sha256": prov["array_sha256"], "feature_names_sha256": prov["feature_names_sha256"]},
        "cases": prov["cases"], "result_files_sha256": {f: sha256_file(os.path.join(results_dir, f)) for f in result_files(results_dir)},
        "reproducibility": repro, "suite_runs": suites, "commands": COMMANDS,
        "held_out_evaluation": "not run; no held-out predictions or metrics exist in any saved artifact",
        "commit_reference": {"scheme": "No self-referential commit SHA or hash. source_set_sha256 identifies the exact content of the XGBoost control files; the final merge commit is recorded in the Linear 298-49 comment after merge.",
                             "evidence_base_commit": git_info["head"], "branch": git_info["branch"], "working_tree_clean_when_generated": git_info["clean"],
                             "source_set_sha256": set_hash, "source_files": rows, "final_merge_commit": None},
    }


def verify_evidence(ev, results_dir, repo=REPO):
    bad = []
    rows, set_hash = m1ev.source_set(repo, SOURCE_FILES)
    if ev["commit_reference"]["source_set_sha256"] != set_hash:
        bad.append("source files changed after the evidence was generated")
    cur = {f: sha256_file(os.path.join(results_dir, f)) for f in result_files(results_dir)}
    if cur != ev["result_files_sha256"]:
        bad.append("result files differ from the hashes in the evidence")
    r = ev["reproducibility"]
    if not (r.get("runs", 0) >= 2 and r.get("identical_to_each_other") and r.get("identical_to_saved_results")):
        bad.append("reproducibility not recorded as identical")
    s = ev["suite_runs"]
    if not (s.get("ran", 0) > 0 and s.get("skipped") == 0 and s.get("failures") == 0 and s.get("errors") == 0):
        bad.append("test run has skips, failures, errors or no tests")
    if forbidden_keys(ev):
        bad.append("evidence contains held-out-style keys")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--dump-features", metavar="DIR", help=argparse.SUPPRESS)
    ap.add_argument("--write-evidence", action="store_true", help="two independent clean reruns + compare, run all tests, write the evidence file")
    args = ap.parse_args(argv)
    if args.dump_features:
        dump_features(args.dump_features)
        return 0
    run_experiment(args.out_dir)
    if not args.write_evidence:
        return 0
    if os.path.abspath(args.out_dir) != RESULTS_DIR:
        raise SystemExit("--write-evidence works on the default results directory")
    print("\nreproducibility: two clean reruns in independent processes")
    temps = []
    for i in (1, 2):
        d = tempfile.mkdtemp(prefix=f"xgb_rerun{i}_")
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--out-dir", d], capture_output=True, text=True, cwd=tempfile.gettempdir())
        if p.returncode != 0:
            raise SystemExit("rerun failed:\n" + (p.stdout + p.stderr)[-1500:])
        temps.append(d)
    same12, f1, _ = compare_dirs(temps[0], temps[1])
    same_saved, _, _ = compare_dirs(temps[0], args.out_dir)
    print(f"  rerun1 == rerun2: {same12}; equal to saved results: {same_saved} ({len(f1)} files)")
    if not (same12 and same_saved):
        raise SystemExit("reruns are not identical")
    repro = {"runs": 2, "identical_to_each_other": same12, "identical_to_saved_results": same_saved, "files_compared": len(f1),
             "per_seed_prediction_sha256": {f: sha256_file(os.path.join(temps[0], f)) for f in f1 if f.endswith("_predictions.csv")}}
    stale = os.path.join(args.out_dir, EVIDENCE_NAME)
    if os.path.exists(stale):
        os.remove(stale)                      # a previous evidence file describes older sources; it is rewritten below
    print("tests (skipping disabled):")
    suites = run_suites()
    g = {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "clean": not git("status", "--porcelain")}
    ev = build_evidence(args.out_dir, repro, suites, g)
    write_json(os.path.join(args.out_dir, EVIDENCE_NAME), ev)
    problems = verify_evidence(json.load(open(os.path.join(args.out_dir, EVIDENCE_NAME))), args.out_dir)
    print("evidence written; verification:", problems or "OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
