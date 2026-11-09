"""First fixed five-seed Model 1 experiment (Linear 298-51): S1 -> S2 per seed, validation for checkpoint selection and metrics only.

    .venv/bin/python -I armor-system/models/model1_transformer/run_experiment.py --seed 0     # one seed (run the five seeds in parallel processes)
    .venv/bin/python -I armor-system/models/model1_transformer/run_experiment.py --aggregate  # mean +- sample SD, XGBoost comparison, evidence file

Frozen decisions: `training_config_v1.json` and TRAINING.md section 7. Validation windows are only ever passed through
`predict_window_probabilities` / `evaluate_masked_mse` (eval mode, no gradient); gradient steps use `training_loader` over
fitting windows. Calibration and the held-out test partition are never loaded into a dataset. Checkpoints go OUTSIDE the
repository (default ~/298A_project/checkpoints/model1_v0). Per-seed results go to results/experiment_v0/seed_<n>/.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "armor-system", "data"), os.path.join(REPO, "armor-system", "models")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model1_transformer import case_eval as ce, checkpoint as ck, training as T, weighting as W  # noqa: E402
from model1_transformer.model import state_digest  # noqa: E402
from model1_transformer.spec import AdapterSpec  # noqa: E402
from prep import m1_dataset as md, m1_evidence as m1ev  # noqa: E402

OBJECTIVES_USED = ("s1_pretrain", "s2_fault_finetune", "s1_early_stopping", "validation_fault_eval")      # nothing else is ever constructed
DEFAULT_CKPT = os.path.expanduser("~/298A_project/checkpoints/model1_v0")
RESULTS = os.path.join(HERE, "results", "experiment_v0")
XGB_DIR = os.path.join(REPO, "armor-system", "models", "model1_xgboost_control")
EVIDENCE_NAME = "evidence_model1_experiment_v0.json"
SOURCE_FILES = tuple(f"armor-system/models/model1_transformer/{f}" for f in (
    "spec.py", "adapters.py", "core.py", "heads.py", "losses.py", "model.py", "weighting.py", "training.py", "checkpoint.py", "case_eval.py",
    "run_experiment.py", "training_config_v1.json", "TRAINING.md"))


def _xgb_metrics():
    spec = importlib.util.spec_from_file_location("xgb_metrics", os.path.join(XGB_DIR, "metrics.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def case_nll(agg):
    """Mean over cases of -log(mean window probability of the true class)."""
    p = agg["probs"][np.arange(len(agg["case_index"])), agg["true_class"]]
    return float(-np.log(np.maximum(p, 1e-12)).mean())


def mean_sd(values):
    v = np.asarray(values, np.float64)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v)), "values": [float(x) for x in v]}


def hardware():
    brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor()
    return {"cpu": brand, "platform": f"{platform.system()} {platform.machine()} {platform.release()}", "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "device": "cpu", "cuda_available": torch.cuda.is_available(), "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def _sha(path):
    return m1ev.sha256_file(path)


def run_seed(D, cfg, seed, ckpt_root=DEFAULT_CKPT, out_dir=None, device="cpu", epochs=None, max_batches=None, log=print):
    t0 = time.time()
    mt = _xgb_metrics()
    dev = torch.device(device)
    spec = AdapterSpec.from_d1(D.schema, D.applicable)
    frozen = ce.load_frozen()
    meta = ce.window_metadata_from_table(D)
    case_ts = {int(c): int(D.cases["timestamp"][c]) for c in np.unique(meta["case_index"])}
    ce.check_alignment(meta["case_index"], meta["timestamps"], meta["fault_class"], frozen, case_ts)          # before any training
    prov = ck.data_provenance(D, spec, cfg, repo=REPO)
    expect = {k: prov[k] for k in ("manifest_content_sha256", "fit_only_scaler_file_sha256", "spec_fingerprint", "train_config_sha256")}
    ckdir = os.path.join(ckpt_root, f"seed_{seed}")
    out_dir = out_dir or os.path.join(RESULTS, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    val_s1 = md.evaluation_loader(md.evaluation_dataset(D, "s1_early_stopping"), batch_size=cfg.s1.batch_size)
    val_s2 = md.evaluation_loader(md.evaluation_dataset(D, "validation_fault_eval"), batch_size=cfg.s2.batch_size)
    p1, p2 = os.path.join(ckdir, "s1_best"), os.path.join(ckdir, "s2_best")

    # ---- Stage 1: normal-reference masked MSE; validation masked MSE selects the checkpoint
    stop1 = T.EarlyStopper(cfg.s1.patience, "min")

    def cb1(epoch, entry, model):
        v = T.evaluate_masked_mse(model, val_s1, dev)
        entry["val_masked_mse"] = v
        stop = stop1.update(epoch, v)
        if stop1.best_epoch == epoch:
            ck.save_checkpoint(p1, model, stage="s1", seed=seed, epoch=epoch, provenance=prov, extra={"val_masked_mse": v, "selected_by": "lowest validation masked MSE"})
        log(f"[seed {seed}] s1 epoch {epoch:3d} train {entry['loss']:.5f} val {v:.5f} best {stop1.best:.5f}@{stop1.best_epoch} lr {entry['lr']:.2e} gn {entry['grad_norm_mean']:.2f} ({time.time() - t0:.0f}s)")
        return stop

    _, h1 = T.train_stage1(D, cfg, seed, spec=spec, device=device, on_epoch_end=cb1, epochs=epochs, max_batches_per_epoch=max_batches)
    sel1 = stop1.best_epoch

    # ---- Stage 2: from the selected S1 checkpoint; case-level validation NLL selects the checkpoint
    m = T.build_model(spec, seed, dev)
    ck.load_checkpoint(p1, m, expect=expect, stage="s1", seed=seed)
    weights = W.CaseBalancedWeights.from_model1_data(D)
    stop2 = T.EarlyStopper(cfg.s2.patience, "min")

    def cb2(epoch, entry, model):
        r = ce.predict_window_probabilities(model, val_s2, dev)
        agg = ce.aggregate_cases(r["case_index"], r["probs"], r["fault_class"])
        nll, s = case_nll(agg), mt.summary(agg["true_class"], agg["pred_class"])
        entry.update({"val_case_nll": nll, "val_macro_f1": s["macro_f1"], "val_accuracy": s["accuracy"]})            # F1/accuracy are logged, never used for selection
        stop = stop2.update(epoch, nll)
        if stop2.best_epoch == epoch:
            ck.save_checkpoint(p2, model, stage="s2", seed=seed, epoch=epoch, provenance=prov, extra={"val_case_nll": nll, "selected_by": "lowest validation case-level NLL", "s1_selected_epoch": sel1})
        log(f"[seed {seed}] s2 epoch {epoch:3d} train {entry['loss']:.5f} val nll {nll:.5f} (best {stop2.best:.5f}@{stop2.best_epoch}) macroF1 {s['macro_f1']:.3f} acc {s['accuracy']:.2f} ({time.time() - t0:.0f}s)")
        return stop

    _, h2, _ = T.train_stage2(m, D, cfg, seed, device=device, on_epoch_end=cb2, epochs=epochs, weights=weights, max_batches_per_epoch=max_batches)
    sel2 = stop2.best_epoch

    # ---- Evaluate the selected S2 checkpoint on the 25 validation cases
    final = T.build_model(spec, seed, dev)
    meta2 = ck.load_checkpoint(p2, final, expect=expect, stage="s2", seed=seed)
    r = ce.predict_window_probabilities(final, val_s2, dev)
    align = ce.check_alignment(r["case_index"], r["timestamps"], r["fault_class"], frozen, case_ts)               # the scored windows, not just the table
    agg = ce.aggregate_cases(r["case_index"], r["probs"], r["fault_class"])
    s = mt.summary(agg["true_class"], agg["pred_class"])
    s["macro_f1_excluding_node_memory"] = mt.macro_f1_excluding(agg["true_class"], agg["pred_class"], 4)
    s["val_case_nll"] = case_nll(agg)
    rows = ce.xgb_style_rows(agg, group_id=D.cases["group_id"][agg["case_index"]])
    pred_path = os.path.join(out_dir, "predictions.csv")
    with open(pred_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in row.items()})
    run = {"seed": seed, "selected": {"s1_epoch": sel1, "s2_epoch": sel2, "s1_val_masked_mse": stop1.best, "s2_val_case_nll": stop2.best},
           "epochs_run": {"s1": len(h1), "s2": len(h2)}, "stopped_early": {"s1": bool(h1[-1].get("stopped_early")), "s2": bool(h2[-1].get("stopped_early"))},
           "checkpoints": {"dir": ckdir, "s1_pt_sha256": _sha(p1 + ".pt"), "s2_pt_sha256": _sha(p2 + ".pt"), "s2_state_sha256": meta2["state_sha256"], "s2_state_sha256_reloaded": state_digest(final)},
           "predictions_csv_sha256": _sha(pred_path), "alignment": align, "class_weights": weights.summary()["class_weights"], "provenance": prov,
           "objectives_used": list(OBJECTIVES_USED), "hardware": hardware(), "elapsed_seconds": round(time.time() - t0, 1)}
    for name, obj in (("histories.json", {"s1": h1, "s2": h2}), ("metrics.json", s), ("run_meta.json", run)):
        with open(os.path.join(out_dir, name), "w") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
            fh.write("\n")
    log(f"[seed {seed}] DONE s1@{sel1} s2@{sel2} macroF1 {s['macro_f1']:.4f} wF1 {s['weighted_f1']:.4f} acc {s['accuracy']:.3f} in {run['elapsed_seconds']}s")
    return {"metrics": s, "run": run, "histories": {"s1": h1, "s2": h2}}


def compare_rerun(root, rerun_dir, seed=0):
    """Byte-compare a clean rerun of one seed with the recorded results (results files, selected epochs, checkpoint hashes)."""
    same = {n: open(os.path.join(root, f"seed_{seed}", n), "rb").read() == open(os.path.join(rerun_dir, n), "rb").read() for n in ("metrics.json", "histories.json", "predictions.csv")}
    a, b = (json.load(open(os.path.join(d, "run_meta.json"))) for d in (os.path.join(root, f"seed_{seed}"), rerun_dir))
    same.update({"selected_epochs": a["selected"] == b["selected"], **{k: a["checkpoints"][k] == b["checkpoints"][k] for k in ("s1_pt_sha256", "s2_pt_sha256", "s2_state_sha256")}})
    return {"seed": seed, "identical": same, "all_identical": all(same.values()), "s2_state_sha256": a["checkpoints"]["s2_state_sha256"]}


def aggregate(root=RESULTS, seeds=T.SEEDS, rerun_dir=None):
    per = {}
    for sd in seeds:
        d = os.path.join(root, f"seed_{sd}")
        per[sd] = {n: json.load(open(os.path.join(d, f"{n}.json"))) for n in ("metrics", "run_meta", "histories")}
    with open(os.path.join(XGB_DIR, "results", "xgb_control_v1", "validation_summary.json")) as fh:
        xgb = json.load(fh)["mean_sd_across_seeds"]
    names = ce.CLASS_NAMES
    summ = {k: mean_sd([per[s]["metrics"][k] for s in seeds]) for k in ("macro_f1", "weighted_f1", "accuracy", "macro_f1_excluding_node_memory", "val_case_nll")}
    summ["per_class_recall"] = {names[k]: mean_sd([per[s]["metrics"]["per_class"]["recall"][k] for s in seeds]) for k in range(5)}
    summ["mean_confusion_matrix"] = np.mean([per[s]["metrics"]["confusion_matrix"] for s in seeds], axis=0).tolist()
    comp = {k: {"transformer_mean": summ[k]["mean"], "transformer_sd": summ[k]["sd"], "xgboost_mean": xgb[k]["mean"], "xgboost_sd": xgb[k]["sd"], "difference_of_means": summ[k]["mean"] - xgb[k]["mean"]}
            for k in ("macro_f1", "weighted_f1", "accuracy")}
    out = {"experiment": "model1_transformer_v0_five_seed_validation", "seeds": list(seeds), "evaluated_on": "25 validation cases (checkpoint selection and metrics only)",
           "primary_metric": "five-class validation macro-F1", "mean_sd_across_seeds": summ, "comparison_with_frozen_xgboost": comp,
           "bootstrap_intervals": "pending: no case-level bootstrap exists for the XGBoost control; none is invented here",
           "selected_epochs": {str(s): per[s]["run_meta"]["selected"] for s in seeds},
           "limitations": ["25 validation cases; Node Memory has 1 case, so its recall is 0 or 1 and macro-F1 moves by up to 0.2 on that case alone.",
                           "Seed SD measures training randomness only, not uncertainty across incidents.",
                           "Validation also chose the stopping epoch, so these numbers are slightly optimistic; the locked test partition is the unbiased estimate.",
                           "Incident-conditioned classification of a known incident window, not early detection."]}
    with open(os.path.join(root, "validation_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    rows, set_hash = m1ev.source_set(REPO, SOURCE_FILES)
    ev = {"evidence_name": "evidence_model1_experiment_v0", "linear_issue": "298-51", "scope": "first fixed five-seed Transformer experiment; validation only; no calibration fitting, no test evaluation",
          "config": {"training_config_sha256": per[seeds[0]]["run_meta"]["provenance"]["train_config_sha256"], "file": "armor-system/models/model1_transformer/training_config_v1.json"},
          "dataset_hashes": {k: per[seeds[0]]["run_meta"]["provenance"][k] for k in ("manifest_content_sha256", "fit_only_scaler_file_sha256", "feature_cache_content_sha256", "preprocessing_config_sha256", "spec_fingerprint")},
          "alignment_with_xgboost": per[seeds[0]]["run_meta"]["alignment"],
          "per_seed": {str(s): {"selected": per[s]["run_meta"]["selected"], "epochs_run": per[s]["run_meta"]["epochs_run"], "checkpoints": per[s]["run_meta"]["checkpoints"],
                                "predictions_csv_sha256": per[s]["run_meta"]["predictions_csv_sha256"], "elapsed_seconds": per[s]["run_meta"]["elapsed_seconds"],
                                "git_head": per[s]["run_meta"]["provenance"]["git_head"]} for s in seeds},
          "hardware": per[seeds[0]]["run_meta"]["hardware"], "objectives_used": list(OBJECTIVES_USED),
          "reproducibility": compare_rerun(root, rerun_dir) if rerun_dir else "not checked in this invocation",
          "results_file_sha256": {f"seed_{s}/{n}": _sha(os.path.join(root, f"seed_{s}", n)) for s in seeds for n in ("metrics.json", "histories.json", "predictions.csv")} | {"validation_summary.json": _sha(os.path.join(root, "validation_summary.json"))},
          "commit_reference": {"scheme": "no self-referential hash; source_set_sha256 covers the code and config that produced the results", "source_set_sha256": set_hash, "source_files": rows,
                               "git_head_at_run": per[seeds[0]]["run_meta"]["provenance"]["git_head"]}}
    with open(os.path.join(root, EVIDENCE_NAME), "w") as fh:
        json.dump(ev, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out, ev


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir")
    ap.add_argument("--rerun-dir", help="with --aggregate: a clean rerun of seed 0 to byte-compare")
    a = ap.parse_args(argv)
    if a.aggregate:
        out, ev = aggregate(rerun_dir=a.rerun_dir)
        print(json.dumps({k: out["mean_sd_across_seeds"][k] for k in ("macro_f1", "weighted_f1", "accuracy")}, indent=1)[:1500])
        print("source_set_sha256", ev["commit_reference"]["source_set_sha256"])
        return 0
    if a.seed is None:
        ap.error("give --seed N or --aggregate")
    if os.path.abspath(a.ckpt_dir).startswith(REPO):
        ap.error("checkpoints must be saved outside the repository")
    T.validate_seed(a.seed)
    D = md.Model1Data.load()
    run_seed(D, T.load_train_config(), a.seed, a.ckpt_dir, a.out_dir, device=str(T.select_device("cpu")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
