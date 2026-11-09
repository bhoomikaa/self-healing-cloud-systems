"""Model 1 v0 anomaly-detection experiment (Linear 298-51): calibrate on the 329 reference windows, score the 869 validation minutes.

    .venv/bin/python -I armor-system/models/model1_transformer/run_detection_v0.py --run              # five seeds -> results/detection_v0/
    .venv/bin/python -I armor-system/models/model1_transformer/run_detection_v0.py --write-evidence   # tests, integration, evidence file

Protocol: armor-system/evaluation/detection/V0_DETECTION_PREDECLARATION.md (committed before any score existed). v0 = the five selected
Stage 1 checkpoints, unmodified (file hashes are checked against the experiment records); no optimizer step, no tuning, no test data:
only the objectives `s3_calibration_reference` (calibration windows) and `validation_all` (validation windows) are ever built, both
non-gradient. Labels are the 25 documented validation incident timestamps; no official ARMOR label or interval file is opened.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DET = os.path.join(REPO, "armor-system", "evaluation", "detection")
sys.path[:0] = [os.path.join(REPO, "armor-system", "data"), os.path.join(REPO, "armor-system", "models"), DET]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import armor_scoring as sc  # noqa: E402
import interval_diagnostics as idg  # noqa: E402
from model1_transformer import checkpoint as ck, threshold_diagnostics as tdg, training as T, window_errors as we  # noqa: E402
from model1_transformer.model import state_digest  # noqa: E402
from model1_transformer.spec import AdapterSpec  # noqa: E402
from prep import detection_calibration as dc, m1_dataset as md, m1_evidence as m1ev  # noqa: E402

SEEDS = T.SEEDS
PERCENTILES = (95.0, 97.0, 99.0, 99.5)
PRIMARY = ("m2_max_normalised", 99.0)
BOOT_SEED, N_BOOT, BLOCK = 20261028, 1000, 7
DELAY, IMPACT = 600, 300
N_CAL, N_VAL, N_NORMAL, N_EVENT, N_IGNORE, N_CASES = 329, 869, 411, 150, 308, 25
DEFAULT_CKPT = os.path.expanduser("~/298A_project/checkpoints/model1_v0")
RESULTS = os.path.join(HERE, "results", "detection_v0")
EXPERIMENT = os.path.join(HERE, "results", "experiment_v0")
EVIDENCE = os.path.join(RESULTS, "evidence_model1_detection_v0.json")
OBJECTIVES_USED = ("s3_calibration_reference", "validation_all")
ARMOR_DIR = os.path.join(REPO, "armor-system", "data", "raw", "armor_zenodo_21744694", "ARMOR", "models")
ARMOR_FILES = {"anomaly_detector.py": os.path.join(ARMOR_DIR, "diagnosis_tasks", "anomaly_detector.py"), "armor_model.py": os.path.join(ARMOR_DIR, "unified_representation", "armor_model.py"),
               "deviation.py": os.path.join(ARMOR_DIR, "unified_representation", "deviation.py")}
SOURCE_FILES = tuple(f"armor-system/models/model1_transformer/{f}" for f in ("run_detection_v0.py", "threshold_diagnostics.py", "window_errors.py")) + (
    "armor-system/evaluation/detection/interval_diagnostics.py", "armor-system/evaluation/detection/armor_scoring.py", "armor-system/evaluation/detection/V0_DETECTION_PREDECLARATION.md",
    "armor-system/data/prep/detection_calibration.py", "armor-system/tests/prep/test_m1_detection_v0.py")
TEST_FILES = ("test_m1_detection_v0.py",)
HISTORICAL = ("armor-system/data/artifacts/d1/evidence/evidence.json", "armor-system/data/artifacts/d1/evidence/evidence_m1_03.json",
              "armor-system/models/model1_xgboost_control/results/xgb_control_v1/evidence_xgb_control_v1.json", "armor-system/models/model1_transformer/results/core_v0/evidence_model1_core_v0.json",
              "armor-system/models/model1_transformer/results/experiment_v0/evidence_model1_experiment_v0.json", "armor-system/evaluation/detection/evidence_armor_scoring_harness_v1.json",
              "armor-system/models/model1_transformer/results/comparison_v0/bootstrap_validation_v1.json")


def _py(o):
    """JSON-safe copy (numpy scalars/arrays to python)."""
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def digest(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def sha256_file(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def validation_labels(D):
    """The 25 documented validation incident timestamps (case table only; no other role's timestamp is selected)."""
    ts = [int(D.cases["timestamp"][a["case_index"]]) for a in D.manifest["cases"]["assignments"] if a["role"] == "validation"]
    if len(ts) != N_CASES:
        raise RuntimeError(f"expected {N_CASES} validation cases, found {len(ts)}")
    return sorted(ts)


def load_s1(D, cfg, spec, seed, ckpt_root=DEFAULT_CKPT):
    """The seed's selected Stage 1 checkpoint, verified against the data/scaler/architecture/config hashes AND the experiment record."""
    prov = ck.data_provenance(D, spec, cfg, repo=REPO)
    expect = {k: prov[k] for k in ("manifest_content_sha256", "fit_only_scaler_file_sha256", "feature_cache_content_sha256", "preprocessing_config_sha256", "spec_fingerprint", "train_config_sha256")}
    path = os.path.join(ckpt_root, f"seed_{seed}", "s1_best")
    with open(os.path.join(EXPERIMENT, f"seed_{seed}", "run_meta.json")) as fh:
        rec = json.load(fh)
    pt = sha256_file(path + ".pt")
    if pt != rec["checkpoints"]["s1_pt_sha256"]:
        raise RuntimeError(f"seed {seed}: the Stage 1 checkpoint file differs from the one recorded by the experiment")
    model = T.build_model(spec, seed, torch.device("cpu"))
    meta = ck.load_checkpoint(path, model, expect=expect, stage="s1", seed=seed)
    if meta["epoch"] != rec["selected"]["s1_epoch"]:
        raise RuntimeError(f"seed {seed}: checkpoint epoch differs from the experiment record")
    model.eval()
    return model, {"seed": seed, "s1_selected_epoch": meta["epoch"], "pt_file_sha256": pt, "state_sha256": meta["state_sha256"], "parameters": meta["parameters"], "git_head_at_training": meta["provenance"]["git_head"],
                   **{k: expect[k] for k in expect}}


def calibrate_seed(D, model, seed, binding):
    r = we.collect_window_errors(model, md.evaluation_loader(md.evaluation_dataset(D, "s3_calibration_reference"), batch_size=64), allowed_roles=("calibration",))
    n = len(r["errors"])
    if n != N_CAL or set(r["role"]) != {"calibration"} or set(r["category"]) != {"normal_reference"} or (r["case_index"] != -1).any() or not r["node_valid"].all():
        raise RuntimeError("the calibration reference set is not exactly the 329 calibration normal_reference windows with all 46 nodes valid")
    order = np.argsort(r["target_ts"], kind="stable")
    errors, start = r["errors"][order], r["start_index"][order]
    cal = dc.calibrate(errors)
    stats = cal["stats"]
    boot = tdg.block_bootstrap_thresholds(errors, None, block=BLOCK, n_boot=N_BOOT, seed=BOOT_SEED + seed, percentile=99.0)
    return {"binding": binding, "n_windows": n, "errors_sha256": digest(errors), "consecutive_window_fraction": tdg.consecutive_fraction(start),
            "node_median": stats["median"], "node_iqr": stats["iqr"], "epsilon": stats["epsilon"], "floor_affected_nodes": stats["floor_affected_nodes"],
            "thresholds_p99_primary": cal["thresholds"], "thresholds_sensitivity": {str(p): t for p, t in cal["sensitivity"].items()},
            "ranks_used": {str(p): tdg.percentile_ranks(n, p) for p in PERCENTILES},
            "top5_calibration_scores": {m: np.sort(cal["scores"][m])[::-1][:5] for m in dc.METHODS},
            "block_bootstrap_p99": boot, "_stats": stats, "_thresholds": {p: t for p, t in cal["sensitivity"].items()}}


def score_events(intervals, labels):
    """ARMOR scorer with explicit 'no score' handling (see the predeclaration): returns the metrics and whether ARMOR defines them."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                                          # ARMOR's 0/0 F1 is a NumPy NaN with a RuntimeWarning
            d = sc.eval_AD_detail(intervals, labels, IMPACT)
    except ZeroDivisionError:
        return {"scored": False, "reason": "no surviving alert interval (ARMOR raises ZeroDivisionError)", "precision": 0.0, "recall": 0.0, "f1": 0.0, "TP": 0, "FP": 0, "FN": len(set(labels))}
    p, r, f = float(d["precision"]), float(d["recall"]), float(d["f1"])
    if np.isnan(f):
        return {"scored": False, "reason": "precision = recall = 0 (ARMOR's F1 is NaN)", "precision": p, "recall": r, "f1": 0.0, "TP": d["TP"], "FP": d["FP"], "FN": d["FN"]}
    return {"scored": True, "precision": p, "recall": r, "f1": f, "TP": d["TP"], "FP": d["FP"], "FN": d["FN"]}


def evaluate_alerts(flags, target_ts, category, labels):
    """All validation metrics for one alert vector. `flags` (869,) bool aligned with `target_ts` (ascending)."""
    alerts = we.alert_timestamps(target_ts, flags)
    iv = sc.get_pre_interval(alerts, DELAY)
    kept = [g for g in idg.raw_alert_groups(alerts, DELAY) if g[0] != g[1]]
    if kept != iv:
        raise RuntimeError("interval diagnostics disagree with the ARMOR-equivalent construction")
    res = score_events(iv, labels)
    if res["TP"] != len([t for t, m in idg.label_matches(iv, labels, IMPACT).items() if m]):
        raise RuntimeError("hit count disagrees between the scorer and the diagnostics")
    normal = category == "normal_reference"
    ign = target_ts[category == "ignore"]
    cls = idg.classify_intervals(iv, labels, ign, IMPACT)
    res.update({"n_alert_minutes": len(alerts), "n_intervals": len(iv), "singleton_alerts_dropped": idg.count_dropped_singletons(alerts, DELAY),
                "delay": idg.delay_summary(idg.detection_delays(iv, labels, IMPACT)),
                "coverage_raw": float(idg.in_intervals(target_ts, iv).mean()), "coverage_padded_300s": float(idg.in_intervals(target_ts, iv, IMPACT).mean()),
                "normal_reference": {"windows": int(normal.sum()), "alarm_windows": int(flags[normal].sum()), "alarm_rate": float(flags[normal].mean()),
                                     "inside_interval_windows": int(idg.in_intervals(target_ts[normal], iv).sum()), "inside_interval_rate": float(idg.in_intervals(target_ts[normal], iv).mean())},
                "alert_minutes_by_category": {c: int(flags[category == c].sum()) for c in ("event_associated", "normal_reference", "ignore")},
                "intervals": {"hit": cls["hit"], "ignore_overlap_not_hit": cls["ignore_overlap"], "normal_only_false_alarm": cls["normal_only"], "overlapping_ignore_regions_total": cls["overlap_ignore_total"]},
                "armor_false_positive_intervals": cls["ignore_overlap"] + cls["normal_only"]})
    return res


def validate_seed(D, model, cal, labels):
    r = we.collect_window_errors(model, md.evaluation_loader(md.evaluation_dataset(D, "validation_all"), batch_size=64), allowed_roles=("validation",))
    order = np.argsort(r["target_ts"], kind="stable")
    errors, ts, cat = r["errors"][order], r["target_ts"][order], r["category"][order]
    counts = {c: int((cat == c).sum()) for c in ("event_associated", "normal_reference", "ignore")}
    if len(ts) != N_VAL or counts != {"event_associated": N_EVENT, "normal_reference": N_NORMAL, "ignore": N_IGNORE} or not (np.diff(ts) == 60).all() or not r["node_valid"].all():
        raise RuntimeError(f"validation scoring set is not the 869 contiguous target minutes (150/411/308): {len(ts)} {counts}")
    scores = dc.window_scores(errors, cal["_stats"])
    out = {"n_windows": len(ts), "windows_by_category": counts, "errors_sha256": digest(errors), "first_target_ts": int(ts[0]), "last_target_ts": int(ts[-1]), "n_labels": len(set(labels)), "methods": {}}
    for m in dc.METHODS:
        out["methods"][m] = {}
        for p in PERCENTILES:
            tau = cal["_thresholds"][p][m]
            out["methods"][m][str(p)] = {"threshold": tau, **evaluate_alerts(scores[m] > tau, ts, cat, labels)}
    return out


def run_seed(D, cfg, spec, seed, labels, ckpt_root=DEFAULT_CKPT):
    model, binding = load_s1(D, cfg, spec, seed, ckpt_root)
    before = state_digest(model)
    cal = calibrate_seed(D, model, seed, binding)
    val = validate_seed(D, model, cal, labels)
    if state_digest(model) != before or sha256_file(os.path.join(ckpt_root, f"seed_{seed}", "s1_best.pt")) != binding["pt_file_sha256"]:
        raise RuntimeError("the model or its checkpoint file changed during calibration/validation")
    cal = {k: v for k, v in cal.items() if not k.startswith("_")}
    return {"seed": seed, "calibration": cal, "validation": val}


FIELDS = ("precision", "recall", "f1", "n_intervals", "armor_false_positive_intervals", "singleton_alerts_dropped", "coverage_padded_300s")


def mean_sd(v):
    v = np.asarray(v, np.float64)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "values": [float(x) for x in v]}


def aggregate(per_seed):
    out = {}
    for m in dc.METHODS:
        out[m] = {}
        for p in PERCENTILES:
            cells = [s["validation"]["methods"][m][str(p)] for s in per_seed]
            a = {f: mean_sd([c[f] for c in cells]) for f in FIELDS}
            a["tp"] = [c["TP"] for c in cells]
            a["n_no_score_seeds"] = int(sum(not c["scored"] for c in cells))
            a["normal_reference_alarm_rate"] = mean_sd([c["normal_reference"]["alarm_rate"] for c in cells])
            a["normal_reference_inside_interval_rate"] = mean_sd([c["normal_reference"]["inside_interval_rate"] for c in cells])
            a["normal_only_false_alarm_intervals"] = mean_sd([c["intervals"]["normal_only_false_alarm"] for c in cells])
            a["ignore_overlap_not_hit_intervals"] = mean_sd([c["intervals"]["ignore_overlap_not_hit"] for c in cells])
            a["intervals_overlapping_ignore_total"] = mean_sd([c["intervals"]["overlapping_ignore_regions_total"] for c in cells])
            med = [c["delay"]["median_s"] for c in cells if c["delay"]["median_s"] is not None]
            a["delay_median_s_over_seeds_with_hits"] = mean_sd(med) if med else None
            a["threshold"] = mean_sd([c["threshold"] for c in cells])
            out[m][str(p)] = a
    return out


def run_all(ckpt_root=DEFAULT_CKPT, out_dir=RESULTS, log=print):
    torch.set_num_threads(1)
    D = md.Model1Data.load()
    cfg = T.load_train_config()
    spec = AdapterSpec.from_d1(D.schema, D.applicable)
    labels = validation_labels(D)
    os.makedirs(out_dir, exist_ok=True)
    per = []
    for s in SEEDS:
        r = run_seed(D, cfg, spec, s, labels, ckpt_root)
        per.append(_py(r))
        with open(os.path.join(out_dir, f"seed_{s}.json"), "w") as fh:
            json.dump(per[-1], fh, indent=1, sort_keys=True)
            fh.write("\n")
        c = r["validation"]["methods"][PRIMARY[0]][str(PRIMARY[1])]
        log(f"seed {s}: primary M2@99 tau {c['threshold']:.4f} P {c['precision']:.3f} R {c['recall']:.3f} F1 {c['f1']:.3f} intervals {c['n_intervals']} FP {c['armor_false_positive_intervals']} delay median {c['delay']['median_s']}")
    summary = {"experiment": "Model 1 v0 anomaly detection (ablation: no simulated-missing training), validation only", "primary": {"method": PRIMARY[0], "percentile": PRIMARY[1]},
               "seeds": list(SEEDS), "labels": f"{N_CASES} documented validation incident timestamps, +/-{IMPACT} s proxy hit rule", "interval_delay_s": DELAY,
               "aggregate_mean_sd_across_seeds": aggregate(per), "predeclaration": "armor-system/evaluation/detection/V0_DETECTION_PREDECLARATION.md",
               "limitations": ["25 documented incidents and a weak +/-300 s proxy hit rule (no incident end times).",
                               "Validation normal_reference windows chose the Stage 1 epoch, so normal-window false-alarm rates are slightly optimistic; thresholds come from the untouched calibration windows.",
                               "Alerts near official-AD-only events cannot be separated: the official label file (it holds test-period labels) is not opened; they count as false positives here.",
                               "The calibration region precedes validation (distribution drift unmeasured); the 99th percentile of 329 overlapping windows rests on the top 3-4 windows.",
                               "No-score seeds (no surviving interval, or P = R = 0) count as 0 in the means and are counted next to every mean."]}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(_py(summary), fh, indent=1, sort_keys=True)
        fh.write("\n")
    return per, summary


# ---------------------------------------------------------------- evidence
def _run(cmd, env=None):
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env, timeout=3600)
    return p.returncode, p.stdout + p.stderr


def _git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def write_evidence():
    env = {**os.environ, "M1_REQUIRE_REAL_DATA": "1"}
    status = _git("status", "--porcelain", "--untracked-files=all")
    tests = {}
    for n in TEST_FILES:
        rc, text = _run([".venv/bin/python", "-I", "-m", "unittest", "discover", "-s", "armor-system/tests/prep", "-p", n], env=env)
        last = text.strip().splitlines()[-1]
        tests[n] = {"ran": int(re.search(r"Ran (\d+) tests", text).group(1)), "skipped": int((re.search(r"skipped=(\d+)", last) or [0, 0])[1]), "ok": rc == 0 and last.startswith("OK")}
        if not tests[n]["ok"] or tests[n]["skipped"]:
            raise SystemExit(f"{n} not clean: {tests[n]}")
    rc, text = _run([".venv/bin/python", "-I", "armor-system/data/m1_integration.py"], env=env)
    m = re.search(r"SUCCESS in ([\d.]+) s: tests ran (\d+), skipped (\d+), failures (\d+), errors (\d+)", text)
    suites = {s: {"ran": int(a), "failures": int(f), "errors": int(e), "skipped": int(k)} for s, a, f, e, k in re.findall(r"^\s+(\w+): ran (\d+), failures (\d+), errors (\d+), skipped (\d+)", text, flags=re.M)}
    if rc != 0 or not m or int(m[3]) or int(m[4]) or int(m[5]) or "evidence verification: OK" not in text:
        raise SystemExit("integration did not succeed with zero skips:\n" + text[-2500:])
    seeds = {s: json.load(open(os.path.join(RESULTS, f"seed_{s}.json"))) for s in SEEDS}
    rows, h = [], hashlib.sha256()
    for rel in sorted(SOURCE_FILES):
        d = sha256_file(os.path.join(REPO, rel))
        rows.append({"path": rel, "sha256": d})
        h.update(rel.encode() + b"\0" + d.encode() + b"\n")
    hist = {p: {"sha256": sha256_file(os.path.join(REPO, p)), "unchanged_vs_HEAD": subprocess.run(["git", "diff", "--quiet", "HEAD", "--", p], cwd=REPO).returncode == 0} for p in HISTORICAL}
    if not all(x["unchanged_vs_HEAD"] for x in hist.values()):
        raise SystemExit(f"historical evidence changed: {hist}")
    ev = {"evidence_name": "evidence_model1_detection_v0", "evidence_version": "1.0", "linear_issue": "298-51",
          "scope": "Model 1 v0 (ablation, no simulated-missing training) calibration on the 329 calibration-reference windows and detection validation on the 869 validation minutes; no test data, no official ARMOR label or interval file, no retraining",
          "predeclaration": {"file": "armor-system/evaluation/detection/V0_DETECTION_PREDECLARATION.md", "sha256": sha256_file(os.path.join(DET, "V0_DETECTION_PREDECLARATION.md")),
                             "statement": "Model 1 v0 uses no simulated-missing training and is reported as the ablation. Model 1 v1 uses simulated-missing training and is the predeclared final model for held-out test evaluation, regardless of validation performance."},
          "timestamp_alignment": {"decision": "primary and only convention: alert time = target (seventh) sample timestamp; unambiguous in ARMOR's source, so no alternative offset was scored",
                                  "armor_source_sha256": {k: sha256_file(v) for k, v in ARMOR_FILES.items()},
                                  "armor_rule": "create_dataloader_AR(window_size=6): series_len = 7 samples, window timestamp = series_sample[-1][0] (the target sample); SLD appends it to each score; AD passes eval_slds_sum[['timestamp']] to get_pre_interval",
                                  "model1_rule": "window_errors.TARGET_POSITION = 6 (timestamps[:, 6])"},
          "configuration": {"seeds": list(SEEDS), "primary": {"method": PRIMARY[0], "percentile": PRIMARY[1]}, "sensitivity_percentiles": [95.0, 97.0, 99.5], "interval_delay_s": DELAY, "impact_window_s": IMPACT,
                            "calibration_windows": N_CAL, "validation_minutes": N_VAL, "normal_reference_windows": N_NORMAL, "documented_validation_incidents": N_CASES,
                            "block_bootstrap": {"block": BLOCK, "resamples": N_BOOT, "seed": f"{BOOT_SEED} + seed"}, "training_config_sha256": seeds[0]["calibration"]["binding"]["train_config_sha256"]},
          "dataset_hashes": {k: seeds[0]["calibration"]["binding"][k] for k in ("manifest_content_sha256", "fit_only_scaler_file_sha256", "feature_cache_content_sha256", "preprocessing_config_sha256", "spec_fingerprint")},
          "checkpoint_bindings": {str(s): {k: seeds[s]["calibration"]["binding"][k] for k in ("s1_selected_epoch", "pt_file_sha256", "state_sha256", "git_head_at_training")} for s in SEEDS},
          "calibration_summary": {str(s): {"errors_sha256": seeds[s]["calibration"]["errors_sha256"], "epsilon": seeds[s]["calibration"]["epsilon"], "thresholds_p99": seeds[s]["calibration"]["thresholds_p99_primary"],
                                          "floor_affected_nodes": len(seeds[s]["calibration"]["floor_affected_nodes"])} for s in SEEDS},
          "validation_errors_sha256": {str(s): seeds[s]["validation"]["errors_sha256"] for s in SEEDS},
          "objectives_used": list(OBJECTIVES_USED), "scorable_roles": list(we.SCORABLE_ROLES), "optimizer_steps_taken": 0, "checkpoints_modified": False,
          "results_file_sha256": {f: sha256_file(os.path.join(RESULTS, f)) for f in sorted(os.listdir(RESULTS)) if f.endswith(".json") and f != os.path.basename(EVIDENCE)},
          "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "torch_num_threads": torch.get_num_threads()},
          "tests": {"files": tests, "m1_integration": {"tests_ran": int(m[2]), "skipped": int(m[3]), "failures": int(m[4]), "errors": int(m[5]), "suites": suites, "evidence_m1_03_verified": True}},
          "historical_evidence_unchanged": hist,
          "commit_reference": {"scheme": "no self-referential hash; source_set_sha256 covers the code, protocol and tests that produced the results", "evidence_base_commit": _git("rev-parse", "HEAD"),
                               "branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "uncommitted_when_generated": [l[3:] for l in status.splitlines()], "source_set_sha256": h.hexdigest(), "source_files": rows}}
    with open(EVIDENCE, "w") as fh:
        json.dump(_py(ev), fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("wrote", os.path.relpath(EVIDENCE, REPO), h.hexdigest())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--write-evidence", action="store_true")
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    a = ap.parse_args(argv)
    if a.run:
        _, s = run_all(a.ckpt_dir)
        c = s["aggregate_mean_sd_across_seeds"][PRIMARY[0]][str(PRIMARY[1])]
        print({k: (round(c[k]["mean"], 4), round(c[k]["sd"], 4)) for k in ("precision", "recall", "f1")}, "no-score seeds", c["n_no_score_seeds"])
        return 0
    if a.write_evidence:
        write_evidence()
        return 0
    ap.error("give --run or --write-evidence")


if __name__ == "__main__":
    sys.exit(main())
