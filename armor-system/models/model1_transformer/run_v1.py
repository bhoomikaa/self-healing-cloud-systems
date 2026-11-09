"""Model 1 v1 experiment (Linear 298-51): train (S1 -> S2), calibrate, validate, missing-data runs, aggregation, evidence.

    .venv/bin/python -I armor-system/models/model1_transformer/run_v1.py --seed N              # one seed (five seeds in parallel processes)
    .venv/bin/python -I armor-system/models/model1_transformer/run_v1.py --missing --seed N    # missing-data experiments for one seed
    .venv/bin/python -I armor-system/models/model1_transformer/run_v1.py --aggregate [--rerun-dir DIR]
    .venv/bin/python -I armor-system/models/model1_transformer/run_v1.py --missing-aggregate
    .venv/bin/python -I armor-system/models/model1_transformer/run_v1.py --write-evidence

Protocol: V1_PROTOCOL.md (committed before any v1 code). Validation is used for checkpoint selection (Stage 1: clean validation normal
MSE; Stage 2: case-level fault NLL + balanced window BCE) and metrics only, never for a gradient update or a detector choice. Gradient steps
take only fitting windows (paired batches in Stage 2). Calibration windows give thresholds only. No test-period data and no official ARMOR
label or interval file is read. Checkpoints go OUTSIDE the repository (default ~/298A_project/checkpoints/model1_v1).
"""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "armor-system", "data"), os.path.join(REPO, "armor-system", "models"), os.path.join(REPO, "armor-system", "evaluation", "detection")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model1_transformer import case_eval as ce, checkpoint as ck, detection_v1 as dv, missing_eval as me, run_detection_v0 as rd, run_experiment as rx, training as T, training_v1 as tv, v1_audit_spot as au  # noqa: E402
from model1_transformer.model import state_digest  # noqa: E402
from model1_transformer.sim_missing import RATE, MissingBlocks  # noqa: E402
from model1_transformer.spec import AdapterSpec  # noqa: E402
from prep import m1_dataset as md  # noqa: E402

SEEDS = T.SEEDS
DEFAULT_CKPT = os.path.expanduser("~/298A_project/checkpoints/model1_v1")
RESULTS = os.path.join(HERE, "results", "experiment_v1")
PROTOCOL = os.path.join(HERE, "V1_PROTOCOL.md")
OBJECTIVES_USED = ("s1_pretrain", "s2_fault_finetune", "s1_early_stopping", "validation_all", "s3_calibration_reference")
EVIDENCE = os.path.join(RESULTS, "evidence_model1_v1.json")
SOURCE_FILES = tuple(f"armor-system/models/model1_transformer/{f}" for f in (
    "V1_PROTOCOL.md", "model_v1.py", "sim_missing.py", "paired.py", "rolling.py", "training_v1.py", "detection_v1.py", "missing_eval.py", "run_v1.py", "bootstrap_v1.py",
    "training_config_v1.json", "weighting.py", "case_eval.py", "checkpoint.py", "window_errors.py", "threshold_diagnostics.py", "run_detection_v0.py", "training.py")) + (
    "armor-system/evaluation/detection/armor_scoring.py", "armor-system/evaluation/detection/interval_diagnostics.py",
    "armor-system/tests/prep/test_m1_v1_components.py", "armor-system/tests/prep/test_m1_v1_detection.py", "armor-system/tests/prep/test_m1_v1_runner.py", "armor-system/tests/prep/test_m1_v1_results.py")
TEST_FILES = ("test_m1_v1_components.py", "test_m1_v1_detection.py", "test_m1_v1_runner.py", "test_m1_v1_results.py")
HISTORICAL = ("armor-system/data/artifacts/d1/evidence/evidence.json", "armor-system/data/artifacts/d1/evidence/evidence_m1_03.json",
              "armor-system/models/model1_xgboost_control/results/xgb_control_v1/evidence_xgb_control_v1.json", "armor-system/models/model1_transformer/results/core_v0/evidence_model1_core_v0.json",
              "armor-system/models/model1_transformer/results/experiment_v0/evidence_model1_experiment_v0.json", "armor-system/models/model1_transformer/results/detection_v0/evidence_model1_detection_v0.json",
              "armor-system/evaluation/detection/evidence_armor_scoring_harness_v1.json", "armor-system/models/model1_transformer/results/comparison_v0/bootstrap_validation_v1.json")
BRIEF = ("precision", "recall", "f1", "scored", "TP", "FP", "FN", "n_intervals", "armor_false_positive_intervals", "coverage_padded_300s", "n_alert_minutes")


def say(*a):
    print(*a, flush=True)


def jload(path):
    with open(path) as fh:
        return json.load(fh)


def rbytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def view(D, name):
    return md.evaluation_loader(md.evaluation_dataset(D, name), batch_size=64)


def provenance(D, spec, cfg):
    prov = ck.data_provenance(D, spec, cfg, repo=REPO)
    prov.update({"model": "Model1V1", "v1_protocol_sha256": sha(PROTOCOL),
                 "simulated_missing": {"rate": RATE, "unit": "(window, input minute, node, modality)", "generator": "numpy Philox keyed by (seed, stage, epoch, window start index)", "targets_hidden": False}})
    return prov


def run_seed(D, cfg, spec, seed, labels, ckpt_root=DEFAULT_CKPT, out_dir=None, device="cpu", epochs=None, max_batches=None, log=say, n_boot=dv.N_BOOT):
    t0 = time.time()
    dev = torch.device(device)
    mt = rx._xgb_metrics()
    frozen = ce.load_frozen()
    meta = ce.window_metadata_from_table(D)
    case_ts = {int(c): int(D.cases["timestamp"][c]) for c in np.unique(meta["case_index"])}
    ce.check_alignment(meta["case_index"], meta["timestamps"], meta["fault_class"], frozen, case_ts)          # before any training
    prov = provenance(D, spec, cfg)
    expect = {k: prov[k] for k in ("manifest_content_sha256", "fit_only_scaler_file_sha256", "spec_fingerprint", "train_config_sha256", "model")}
    ckdir = os.path.join(ckpt_root, f"seed_{seed}")
    out_dir = out_dir or os.path.join(RESULTS, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    val_s1, val_all, cal_loader = view(D, "s1_early_stopping"), view(D, "validation_all"), view(D, "s3_calibration_reference")
    p1, p2 = os.path.join(ckdir, "s1_best"), os.path.join(ckdir, "s2_best")

    # ---- Stage 1: next-step masked MSE on fitting normal windows with 20% simulated missing; clean validation normal MSE selects the epoch
    stop1 = T.EarlyStopper(cfg.s1.patience, "min")

    def cb1(epoch, entry, model):
        v = T.evaluate_masked_mse(model, val_s1, dev)
        entry["val_masked_mse"] = v
        stop = stop1.update(epoch, v)
        if stop1.best_epoch == epoch:
            ck.save_checkpoint(p1, model, stage="s1", seed=seed, epoch=epoch, provenance=prov, extra={"val_masked_mse": v, "selected_by": "lowest clean validation masked MSE"})
        log(f"[v1 seed {seed}] s1 epoch {epoch:3d} train {entry['loss']:.5f} val {v:.5f} best {stop1.best:.5f}@{stop1.best_epoch} ({time.time() - t0:.0f}s)")
        return stop

    _, h1 = tv.train_stage1_v1(D, cfg, seed, spec, device, cb1, epochs, max_batches)
    sel1 = stop1.best_epoch

    # ---- Stage 2: from the selected Stage 1 checkpoint; paired batches, L_fault + L_det; validation fault NLL + balanced BCE selects the epoch
    m = tv.build_model_v1(spec, seed, dev)
    ck.load_checkpoint(p1, m, expect=expect, stage="s1", seed=seed)
    stop2 = T.EarlyStopper(cfg.s2.patience, "min")

    def cb2(epoch, entry, model):
        mon = dv.selection_monitor(dv.collect_outputs(model, val_all, ("validation",), device), ce)
        s = mt.summary(mon["agg"]["true_class"], mon["agg"]["pred_class"])
        entry.update({"val_case_nll": mon["case_nll"], "val_balanced_bce": mon["balanced_bce"], "val_monitor": mon["monitor"], "val_macro_f1": s["macro_f1"], "val_accuracy": s["accuracy"]})   # F1/accuracy logged only
        stop = stop2.update(epoch, mon["monitor"])
        if stop2.best_epoch == epoch:
            ck.save_checkpoint(p2, model, stage="s2", seed=seed, epoch=epoch, provenance=prov, extra={"val_monitor": mon["monitor"], "val_case_nll": mon["case_nll"], "val_balanced_bce": mon["balanced_bce"],
                                                                                                     "selected_by": "lowest validation case-level fault NLL + balanced window BCE", "s1_selected_epoch": sel1})
        log(f"[v1 seed {seed}] s2 epoch {epoch:3d} train {entry['loss']:.4f} monitor {mon['monitor']:.4f} (nll {mon['case_nll']:.3f} bce {mon['balanced_bce']:.3f}) best {stop2.best:.4f}@{stop2.best_epoch} macroF1 {s['macro_f1']:.3f} ({time.time() - t0:.0f}s)")
        return stop

    _, h2, weights = tv.train_stage2_v1(m, D, cfg, seed, device, cb2, epochs, max_batches)
    sel2 = stop2.best_epoch

    # ---- Final evaluation of the selected checkpoints (validation and calibration windows only, no gradient)
    s1m, s2m = tv.build_model_v1(spec, seed, dev), tv.build_model_v1(spec, seed, dev)
    ck.load_checkpoint(p1, s1m, expect=expect, stage="s1", seed=seed)
    meta2 = ck.load_checkpoint(p2, s2m, expect=expect, stage="s2", seed=seed)
    val2, cal2 = dv.collect_outputs(s2m, val_all, ("validation",), device), dv.collect_outputs(s2m, cal_loader, ("calibration",), device)
    val1, cal1 = dv.collect_outputs(s1m, val_all, ("validation",), device), dv.collect_outputs(s1m, cal_loader, ("calibration",), device)
    if len(val2["errors"]) != rd.N_VAL or len(cal2["errors"]) != rd.N_CAL:
        raise RuntimeError("unexpected number of validation or calibration windows")
    ev = dv.event_windows(val2)
    align = ce.check_alignment(val2["case_index"][ev], val2["timestamps"][ev], val2["fault_class"][ev], frozen, case_ts)     # the scored windows themselves
    agg = ce.aggregate_cases(val2["case_index"][ev], dv.softmax(val2["fault_logits"][ev]), val2["fault_class"][ev])
    fault = mt.summary(agg["true_class"], agg["pred_class"])
    fault.update({"macro_f1_excluding_node_memory": mt.macro_f1_excluding(agg["true_class"], agg["pred_class"], 4), "val_case_nll": dv.case_level_nll(agg)})
    rows = ce.xgb_style_rows(agg, group_id=D.cases["group_id"][agg["case_index"]])
    pred_path = os.path.join(out_dir, "predictions.csv")
    with open(pred_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in r.items()})
    _, get_threshold, _ = au.load_armor_spot()
    sup_th, sup_info = dv.supervised_thresholds(cal2, seed, get_threshold, True, n_boot)
    sup_val = dv.supervised_validation(val2, sup_th, labels)
    f_stats, f_th, f_info = dv.forecast_calibration(cal1)
    f_val = dv.forecast_validation(val1, f_stats, f_th, labels)
    detection = {"primary_rule": "supervised logit -> causal 60 s rolling max -> empirical 90th percentile of rolled calibration scores", "thresholds": sup_th, "calibration": sup_info, "validation": sup_val,
                 "d3_forecast_error_stage1": {"calibration": {**f_info, "thresholds": f_th, "node_median": f_stats["median"], "node_iqr": f_stats["iqr"]}, "validation": f_val}}
    run = {"seed": seed, "selected": {"s1_epoch": sel1, "s2_epoch": sel2, "s1_val_masked_mse": stop1.best, "s2_val_monitor": stop2.best}, "epochs_run": {"s1": len(h1), "s2": len(h2)},
           "stopped_early": {"s1": bool(h1[-1].get("stopped_early")), "s2": bool(h2[-1].get("stopped_early"))},
           "checkpoints": {"dir": ckdir, "s1_pt_sha256": sha(p1 + ".pt"), "s2_pt_sha256": sha(p2 + ".pt"), "s2_state_sha256": meta2["state_sha256"], "s2_state_sha256_reloaded": state_digest(s2m),
                           "parameters": s2m.n_parameters()},
           "predictions_csv_sha256": sha(pred_path), "alignment": align, "class_weights": weights.summary()["class_weights"], "provenance": prov, "objectives_used": list(OBJECTIVES_USED),
           "hardware": rx.hardware(), "elapsed_seconds": round(time.time() - t0, 1)}
    for name, obj in (("histories.json", {"s1": h1, "s2": h2}), ("metrics.json", fault), ("detection.json", detection), ("run_meta.json", run)):
        with open(os.path.join(out_dir, name), "w") as fh:
            json.dump(rd._py(obj), fh, indent=1, sort_keys=True)
            fh.write("\n")
    pr = sup_val["primary"]
    log(f"[v1 seed {seed}] DONE s1@{sel1} s2@{sel2} macroF1 {fault['macro_f1']:.4f} | detection primary P {pr['precision']:.3f} R {pr['recall']:.3f} F1 {pr['f1']:.3f} FP {pr['armor_false_positive_intervals']} in {run['elapsed_seconds']}s")
    return {"fault": fault, "detection": detection, "run": run, "histories": {"s1": h1, "s2": h2}}


# ----------------------------------------------------------------------------------- aggregation
def cell_summary(cells):
    def ms(v):
        return rd.mean_sd(v)
    a = {f: ms([c[f] for c in cells]) for f in ("precision", "recall", "f1", "n_intervals", "armor_false_positive_intervals", "singleton_alerts_dropped", "coverage_padded_300s", "coverage_raw")}
    a["tp"] = [c["TP"] for c in cells]
    a["n_no_score_seeds"] = int(sum(not c["scored"] for c in cells))
    a["normal_reference_alarm_rate"] = ms([c["normal_reference"]["alarm_rate"] for c in cells])
    a["normal_reference_inside_interval_rate"] = ms([c["normal_reference"]["inside_interval_rate"] for c in cells])
    a["normal_only_false_alarm_intervals"] = ms([c["intervals"]["normal_only_false_alarm"] for c in cells])
    a["ignore_overlap_not_hit_intervals"] = ms([c["intervals"]["ignore_overlap_not_hit"] for c in cells])
    a["intervals_overlapping_ignore_total"] = ms([c["intervals"]["overlapping_ignore_regions_total"] for c in cells])
    med = [c["delay"]["median_s"] for c in cells if c["delay"]["median_s"] is not None]
    a["delay_median_s_over_seeds_with_hits"] = ms(med) if med else None
    a["delay_mean_s_over_seeds_with_hits"] = ms([c["delay"]["mean_s"] for c in cells if c["delay"]["mean_s"] is not None]) if med else None
    a["threshold"] = ms([c["threshold"] for c in cells])
    return a


def compare_rerun(root, rerun_dir, seed=0):
    names = ("metrics.json", "histories.json", "detection.json", "predictions.csv")
    same = {n: rbytes(os.path.join(root, f"seed_{seed}", n)) == rbytes(os.path.join(rerun_dir, n)) for n in names}
    a, b = (jload(os.path.join(d, "run_meta.json")) for d in (os.path.join(root, f"seed_{seed}"), rerun_dir))
    same.update({"selected_epochs": a["selected"] == b["selected"], **{k: a["checkpoints"][k] == b["checkpoints"][k] for k in ("s1_pt_sha256", "s2_pt_sha256", "s2_state_sha256")}})
    return {"seed": seed, "identical": same, "all_identical": all(same.values()), "s2_state_sha256": a["checkpoints"]["s2_state_sha256"]}


def aggregate(root=RESULTS, seeds=SEEDS, rerun_dir=None):
    per = {s: {n: jload(os.path.join(root, f"seed_{s}", f"{n}.json")) for n in ("metrics", "detection", "run_meta", "histories")} for s in seeds}
    names = ce.CLASS_NAMES
    fault = {k: rd.mean_sd([per[s]["metrics"][k] for s in seeds]) for k in ("macro_f1", "weighted_f1", "accuracy", "macro_f1_excluding_node_memory", "val_case_nll")}
    fault["per_class_recall"] = {names[k]: rd.mean_sd([per[s]["metrics"]["per_class"]["recall"][k] for s in seeds]) for k in range(5)}
    fault["mean_confusion_matrix"] = np.mean([per[s]["metrics"]["confusion_matrix"] for s in seeds], axis=0).tolist()
    xgb = jload(os.path.join(rx.XGB_DIR, "results", "xgb_control_v1", "validation_summary.json"))["mean_sd_across_seeds"]
    v0 = jload(os.path.join(rx.RESULTS, "validation_summary.json"))["mean_sd_across_seeds"]
    fault["comparison"] = {k: {"v1_mean": fault[k]["mean"], "v1_sd": fault[k]["sd"], "v0_mean": v0[k]["mean"], "v0_sd": v0[k]["sd"], "xgboost_mean": xgb[k]["mean"], "xgboost_sd": xgb[k]["sd"]} for k in ("macro_f1", "weighted_f1", "accuracy")}
    det = {"primary": cell_summary([per[s]["detection"]["validation"]["primary"] for s in seeds])}
    det["d1"] = {n: cell_summary([per[s]["detection"]["validation"][n] for s in seeds]) for n in sorted(per[seeds[0]]["detection"]["validation"]) if n.startswith("d1_")}
    det["d2_no_rolling"] = cell_summary([per[s]["detection"]["validation"]["d2_no_rolling"] for s in seeds])
    d3 = {}
    for m in per[seeds[0]]["detection"]["d3_forecast_error_stage1"]["validation"]:
        d3[m] = {p: cell_summary([per[s]["detection"]["d3_forecast_error_stage1"]["validation"][m][p] for s in seeds]) for p in per[seeds[0]]["detection"]["d3_forecast_error_stage1"]["validation"][m]}
    det["d3_forecast_error_stage1"] = d3
    with open(os.path.join(HERE, "results", "detection_v0", "summary.json")) as fh:
        det["d4_v0"] = json.load(fh)["aggregate_mean_sd_across_seeds"]
    det["calibration_threshold_bootstrap_p90"] = {str(s): per[s]["detection"]["calibration"]["block_bootstrap_p90"] for s in seeds}
    out = {"experiment": "Model 1 v1 (supervised detection head, 20% simulated missing), validation only", "seeds": list(seeds), "selected_epochs": {str(s): per[s]["run_meta"]["selected"] for s in seeds},
           "fault_type": fault, "detection": det, "protocol": "armor-system/models/model1_transformer/V1_PROTOCOL.md",
           "limitations": ["25 validation cases (one Node Memory case) and 25 documented incidents with a +/-300 s proxy hit rule.",
                           "Validation also selects the checkpoints (Stage 1: clean normal MSE; Stage 2: fault NLL + balanced BCE), so validation numbers are optimistic; thresholds come from calibration only.",
                           "v1 detection is supervised with fitting-region weak labels; D3 is the label-free like-for-like row. No ARMOR number is compared here (ARMOR's published numbers are on the test period).",
                           "A 90th-percentile operating point alerts on about 10% of calibration minutes by construction; coverage and false-positive intervals accompany every number."]}
    out["reproducibility_rerun"] = compare_rerun(root, rerun_dir) if rerun_dir else "not checked in this invocation"
    with open(os.path.join(root, "validation_summary.json"), "w") as fh:
        json.dump(rd._py(out), fh, indent=1, sort_keys=True)
        fh.write("\n")
    return out


# ----------------------------------------------------------------------------------- missing-data experiments
def brief(r):
    return {k: r[k] for k in BRIEF} | {"normal_alarm_rate": r["normal_reference"]["alarm_rate"]}


def load_models(D, cfg, spec, seed, ckpt_v1, ckpt_v0):
    prov = provenance(D, spec, cfg)
    expect = {k: prov[k] for k in ("manifest_content_sha256", "fit_only_scaler_file_sha256", "spec_fingerprint", "train_config_sha256", "model")}
    s1, s2 = tv.build_model_v1(spec, seed, torch.device("cpu")), tv.build_model_v1(spec, seed, torch.device("cpu"))
    ck.load_checkpoint(os.path.join(ckpt_v1, f"seed_{seed}", "s1_best"), s1, expect=expect, stage="s1", seed=seed)
    ck.load_checkpoint(os.path.join(ckpt_v1, f"seed_{seed}", "s2_best"), s2, expect=expect, stage="s2", seed=seed)
    v0, _ = rd.load_s1(D, cfg, spec, seed, ckpt_v0)
    return {"v1_primary": s2, "v1_d3": s1, "v0": v0}


def condition_outputs(models, view_D, device="cpu"):
    cal_loader, val_loader = view(view_D, "s3_calibration_reference"), view(view_D, "validation_all")
    return {k: (dv.collect_outputs(m, cal_loader, ("calibration",), device), dv.collect_outputs(m, val_loader, ("validation",), device)) for k, m in models.items()}


def score_condition(outs, clean, labels, seed):
    """Per model: recalibrated on the same masked calibration windows (primary) and with the clean thresholds (secondary)."""
    res = {}
    for key, (cal, val) in outs.items():
        if key == "v1_primary":
            th, _ = dv.supervised_thresholds(cal, seed, None, bootstrap=False)
            res[key] = {"recalibrated": brief(dv.supervised_validation(val, th, labels)["primary"]), "clean_thresholds": brief(dv.supervised_validation(val, clean[key]["th"], labels)["primary"])}
        else:
            stats, th, _ = dv.forecast_calibration(cal)
            m, p = dv.D3_PRIMARY
            res[key] = {"recalibrated": brief(dv.forecast_validation(val, stats, {m: {str(p): th[m][str(p)]}}, labels)[m][str(p)]),
                        "clean_thresholds": brief(dv.forecast_validation(val, clean[key]["stats"], {m: {str(p): clean[key]["th"][m][str(p)]}}, labels)[m][str(p)])}
    return res


def run_missing_seed(D, cfg, spec, seed, labels, ckpt_v1=DEFAULT_CKPT, ckpt_v0=rd.DEFAULT_CKPT, out_dir=RESULTS, log=say, masks=me.N_MASKS):
    t0 = time.time()
    torch.set_num_threads(1)
    models = load_models(D, cfg, spec, seed, ckpt_v1, ckpt_v0)
    clean_outs = condition_outputs(models, D)
    clean = {}
    for key, (cal, val) in clean_outs.items():
        if key == "v1_primary":
            clean[key] = {"th": dv.supervised_thresholds(cal, seed, None, bootstrap=False)[0]}
        else:
            stats, th, _ = dv.forecast_calibration(cal)
            clean[key] = {"stats": stats, "th": th}
    blocks = MissingBlocks(spec)
    res = {"seed": seed, "armor": {}, "random": {}, "clean_reference": score_condition(clean_outs, clean, labels, seed)}
    for name, drop in me.armor_scenarios():
        obs = me.scenario_observed(D, drop)
        outs = condition_outputs(models, me.masked_view(D, obs))
        res["armor"][name] = {"dropped": list(drop), "hides_all_vms": me.hides_all_vms(D, spec, obs), "models": score_condition(outs, clean, labels, seed)}
    for ri, rate in enumerate(me.RATES):
        res["random"][str(rate)] = {}
        for mi in range(1 if rate == 0.0 else masks):                           # rate 0: all masks are identical, one is enough
            outs = condition_outputs(models, me.masked_view(D, me.random_observed(D, blocks, ri, mi, rate)))
            res["random"][str(rate)][str(mi)] = score_condition(outs, clean, labels, seed)
        log(f"[missing seed {seed}] rate {rate} done ({time.time() - t0:.0f}s)")
    with open(os.path.join(out_dir, f"missing_seed_{seed}.json"), "w") as fh:
        json.dump(rd._py(res), fh, indent=1, sort_keys=True)
        fh.write("\n")
    return res


def missing_aggregate(root=RESULTS, seeds=SEEDS):
    per = {s: jload(os.path.join(root, f"missing_seed_{s}.json")) for s in seeds}
    models, modes = ("v1_primary", "v1_d3", "v0"), ("recalibrated", "clean_thresholds")
    fields = ("f1", "recall", "precision", "armor_false_positive_intervals", "coverage_padded_300s")

    def stat(cells):
        return {f: rd.mean_sd([c[f] for c in cells]) for f in fields} | {"n": len(cells), "n_no_score": int(sum(not c["scored"] for c in cells))}

    out = {"seeds": list(seeds), "armor_scenarios": {}, "random_rates": {}, "clean_reference": {m: {md_: stat([per[s]["clean_reference"][m][md_] for s in seeds]) for md_ in modes} for m in models}}
    for name in per[seeds[0]]["armor"]:
        out["armor_scenarios"][name] = {"dropped": per[seeds[0]]["armor"][name]["dropped"], "hides_all_vms": per[seeds[0]]["armor"][name]["hides_all_vms"],
                                        "models": {m: {md_: stat([per[s]["armor"][name]["models"][m][md_] for s in seeds]) for md_ in modes} for m in models}}
    for rate in per[seeds[0]]["random"]:
        out["random_rates"][rate] = {m: {md_: stat([c[m][md_] for s in seeds for c in per[s]["random"][rate].values()]) for md_ in modes} for m in models}
        out["random_rates"][rate]["distinct_masks"] = len(per[seeds[0]]["random"][rate])
    out["notes"] = ["Scenarios containing 'metric' hide all six VMs entirely (hides_all_vms) and are reported separately from the pod-only scenarios drop_log, drop_trace, drop_log+trace.",
                    "Random rates use 10 deterministic masks per rate (rate 0: one distinct mask); statistics pool masks and seeds. Recalibrated = thresholds refitted on the same masked calibration windows (primary); clean_thresholds = thresholds from the unmasked calibration windows."]
    with open(os.path.join(root, "missing_summary.json"), "w") as fh:
        json.dump(rd._py(out), fh, indent=1, sort_keys=True)
        fh.write("\n")
    return out


# ----------------------------------------------------------------------------------- evidence
def _run(cmd, env=None):
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env, timeout=7200)
    return p.returncode, p.stdout + p.stderr


def _git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def write_evidence(rerun_dir=None):
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
    per = {s: jload(os.path.join(RESULTS, f"seed_{s}", "run_meta.json")) for s in SEEDS}
    rows, h = [], hashlib.sha256()
    for rel in sorted(SOURCE_FILES):
        d = sha(os.path.join(REPO, rel))
        rows.append({"path": rel, "sha256": d})
        h.update(rel.encode() + b"\0" + d.encode() + b"\n")
    hist = {p: {"sha256": sha(os.path.join(REPO, p)), "unchanged_vs_HEAD": subprocess.run(["git", "diff", "--quiet", "HEAD", "--", p], cwd=REPO).returncode == 0} for p in HISTORICAL}
    if not all(x["unchanged_vs_HEAD"] for x in hist.values()):
        raise SystemExit(f"historical evidence changed: {hist}")
    v0_ck = {s: jload(os.path.join(rx.RESULTS, f"seed_{s}", "run_meta.json"))["checkpoints"]["s1_pt_sha256"] for s in SEEDS}
    ev = {"evidence_name": "evidence_model1_v1", "evidence_version": "1.0", "linear_issue": "298-51",
          "scope": "Model 1 v1 (binary detection head, 20% simulated missing, supervised detection; validation only): training, calibration, validation, missing-data experiments; no test data, no official ARMOR label or interval file",
          "predeclaration": {"protocol_file": "armor-system/models/model1_transformer/V1_PROTOCOL.md", "protocol_sha256": sha(PROTOCOL),
                             "statement": "Model 1 v0 uses no simulated-missing training and is reported as the ablation. Model 1 v1 uses simulated-missing training and is the predeclared final model for held-out test evaluation, regardless of validation performance."},
          "configuration": {"seeds": list(SEEDS), "training_config_sha256": per[0]["provenance"]["train_config_sha256"], "model": "Model1V1 (176,521 parameters)", "simulated_missing_rate": RATE,
                            "stage2_batch": "64 event + 64 normal windows", "loss": "L_fault + L_det (weight 1 each)", "selection": "Stage 1: clean validation normal masked MSE; Stage 2: validation case-level fault NLL + balanced window BCE",
                            "primary_detector": "supervised logit, causal 60 s rolling max, empirical 90th percentile of rolled calibration scores, ARMOR interval rules, 25 documented validation incidents",
                            "block_bootstrap": {"block": dv.BLOCK, "resamples": dv.N_BOOT, "seed": f"{dv.BOOT_SEED} + seed"}, "missing_data": {"rates": list(me.RATES), "masks_per_rate": me.N_MASKS, "armor_scenarios": [n for n, _ in me.armor_scenarios()]}},
          "dataset_hashes": {k: per[0]["provenance"][k] for k in ("manifest_content_sha256", "fit_only_scaler_file_sha256", "feature_cache_content_sha256", "preprocessing_config_sha256", "spec_fingerprint")},
          "alignment_with_xgboost": per[0]["alignment"],
          "per_seed": {str(s): {"selected": per[s]["selected"], "epochs_run": per[s]["epochs_run"], "checkpoints": per[s]["checkpoints"], "predictions_csv_sha256": per[s]["predictions_csv_sha256"],
                                "elapsed_seconds": per[s]["elapsed_seconds"], "git_head_at_training": per[s]["provenance"]["git_head"]} for s in SEEDS},
          "v0_stage1_checkpoint_hashes_unchanged": {str(s): sha(os.path.join(rd.DEFAULT_CKPT, f"seed_{s}", "s1_best.pt")) == v0_ck[s] for s in SEEDS},
          "hardware": per[0]["hardware"], "objectives_used": list(OBJECTIVES_USED), "test_data_accessed": False, "official_armor_labels_or_intervals_accessed": False,
          "results_file_sha256": {os.path.relpath(os.path.join(dp, f), RESULTS): sha(os.path.join(dp, f)) for dp, _, fs in os.walk(RESULTS) for f in sorted(fs) if f.endswith((".json", ".csv")) and f != os.path.basename(EVIDENCE)},
          "bootstrap_vs_xgboost_file_sha256": sha(os.path.join(HERE, "results", "comparison_v1", "bootstrap_validation_v1_vs_xgboost.json")),
          "reproducibility_rerun": compare_rerun(RESULTS, rerun_dir) if rerun_dir else "not provided",
          "tests": {"files": tests, "m1_integration": {"tests_ran": int(m[2]), "skipped": int(m[3]), "failures": int(m[4]), "errors": int(m[5]), "suites": suites, "evidence_m1_03_verified": True}},
          "historical_evidence_unchanged": hist,
          "commit_reference": {"scheme": "no self-referential hash; source_set_sha256 covers the code, protocol and tests that produced the results", "evidence_base_commit": _git("rev-parse", "HEAD"),
                               "branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "uncommitted_when_generated": [l[3:] for l in status.splitlines()], "source_set_sha256": h.hexdigest(), "source_files": rows}}
    with open(EVIDENCE, "w") as fh:
        json.dump(rd._py(ev), fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("wrote", os.path.relpath(EVIDENCE, REPO), h.hexdigest())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--missing", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--missing-aggregate", action="store_true")
    ap.add_argument("--write-evidence", action="store_true")
    ap.add_argument("--rerun-dir")
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir")
    a = ap.parse_args(argv)
    if a.aggregate:
        out = aggregate(rerun_dir=a.rerun_dir)
        print(json.dumps({k: out["fault_type"][k]["mean"] for k in ("macro_f1", "weighted_f1", "accuracy")}))
        return 0
    if a.missing_aggregate:
        missing_aggregate()
        return 0
    if a.write_evidence:
        write_evidence(a.rerun_dir)
        return 0
    if a.seed is None:
        ap.error("give --seed N, --aggregate, --missing-aggregate or --write-evidence")
    if os.path.abspath(a.ckpt_dir).startswith(REPO):
        ap.error("checkpoints must be saved outside the repository")
    T.validate_seed(a.seed)
    torch.set_num_threads(1)
    D = md.Model1Data.load()
    cfg = T.load_train_config()
    spec = AdapterSpec.from_d1(D.schema, D.applicable)
    labels = rd.validation_labels(D)
    if a.missing:
        run_missing_seed(D, cfg, spec, a.seed, labels, a.ckpt_dir)
    else:
        run_seed(D, cfg, spec, a.seed, labels, a.ckpt_dir, a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
