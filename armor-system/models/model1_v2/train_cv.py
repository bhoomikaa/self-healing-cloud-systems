"""Train one v1-recipe model for one CV fold and seed (V2_ENSEMBLE_PROTOCOL.md section 3b) and score its held-out fold.

    .venv/bin/python -I armor-system/models/model1_v2/train_cv.py --fold F1 --seed 0 [--ckpt-root DIR] [--out DIR]

Stage 1 then Stage 2 with the frozen v1 training functions (`training_v1.train_stage1_v1 / train_stage2_v1`), fixed epochs
(77 / 35), no checkpoint selection. Checkpoints and window outputs go outside the repository (--ckpt-root); the per-seed
result JSON and case predictions go to --out. Only training-fold, held-out-fold and calibration windows exist in the fold data.
"""
import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "armor-system", p) for p in ("data", "models", os.path.join("evaluation", "detection"))]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model1_v2 import folds  # noqa: E402

S1_EPOCHS, S2_EPOCHS = 77, 35                     # medians of v1's selected epochs (run_meta.json, seeds 0-4)
CKPT_ROOT = os.path.expanduser("~/298A_project/checkpoints/model1_v2_cv")
OUT = os.path.join(HERE, "results", "cv")
KEYS = ("det_logit", "fault_logits", "target_ts", "start_index", "case_index", "fault_class", "ambiguous", "category", "role")


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    from model1_transformer import run_detection_v0 as rd
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rd._py(obj), fh, indent=1, sort_keys=True)
        fh.write("\n")


def outputs(model, fd, objective, role):
    from model1_transformer import detection_v1 as dv
    from prep import m1_dataset as md
    out = dv.sort_by_time(dv.collect_outputs(model, md.evaluation_loader(md.evaluation_dataset(fd, objective), batch_size=64), (role,)))
    return {k: out[k] for k in KEYS}


def single_seed_metrics(held, cal, labels, seed):
    """v1's primary detector and case-level fault type for one model (thresholds from its own calibration outputs)."""
    from model1_transformer import case_eval as ce, detection_v1 as dv, run_experiment as rx
    mt = rx._xgb_metrics()
    ev = dv.event_windows(held)
    agg = ce.aggregate_cases(held["case_index"][ev], dv.softmax(held["fault_logits"][ev]), held["fault_class"][ev])
    fault = mt.summary(agg["true_class"], agg["pred_class"])
    th, info = dv.supervised_thresholds(cal, seed, None, bootstrap=False)
    det = dv.supervised_validation(held, th, labels)["primary"]
    return agg, fault, {"threshold": th["primary_p90"], "calibration_windows": info["n_windows"], **det}


def write_predictions(path, agg, D):
    from model1_transformer import case_eval as ce
    rows = ce.xgb_style_rows(agg, group_id=D.cases["group_id"][agg["case_index"]])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in r.items()})


def run(fold, seed, ckpt_root=CKPT_ROOT, out_dir=OUT, s1_epochs=S1_EPOCHS, s2_epochs=S2_EPOCHS, max_batches=None, log=print):
    from model1_transformer import checkpoint as ck, run_v1 as r1, training as T, training_v1 as tv
    from model1_transformer.model import state_digest
    from model1_transformer.spec import AdapterSpec
    from prep import m1_dataset as md
    t0 = time.time()
    torch.set_num_threads(1)
    D = md.Model1Data.load()
    cfg = T.load_train_config()
    spec = AdapterSpec.from_d1(D.schema, D.applicable)
    fd = folds.build(D, fold)
    prov = {**r1.provenance(D, spec, cfg), "v2_cv": {"held_out": fold, "training_folds": fd.fold["training"], "s1_epochs": s1_epochs, "s2_epochs": s2_epochs}}
    expect = {k: prov[k] for k in ("manifest_content_sha256", "fit_only_scaler_file_sha256", "spec_fingerprint", "train_config_sha256", "model", "v2_cv")}
    ckdir = os.path.join(ckpt_root, fold, f"seed_{seed}")
    p1, p2 = os.path.join(ckdir, "s1_final"), os.path.join(ckdir, "s2_final")
    with folds.fold_regions():
        m1, h1 = tv.train_stage1_v1(fd, cfg, seed, spec, "cpu", None, s1_epochs, max_batches)
        ck.save_checkpoint(p1, m1, stage="s1", seed=seed, epoch=len(h1), provenance=prov, extra={"selected_by": "fixed epoch count (no selection)"})
        log(f"[v2 cv {fold} seed {seed}] s1 done ({len(h1)} epochs, {time.time() - t0:.0f}s)")
        m = tv.build_model_v1(spec, seed, torch.device("cpu"))
        ck.load_checkpoint(p1, m, expect=expect, stage="s1", seed=seed)
        weights = folds.s2_weights(fd)
        _, h2, _ = tv.train_stage2_v1(m, fd, cfg, seed, "cpu", None, s2_epochs, max_batches, weights)
        meta2 = ck.save_checkpoint(p2, m, stage="s2", seed=seed, epoch=len(h2), provenance=prov, extra={"selected_by": "fixed epoch count (no selection)"})
        m.eval()
        held = outputs(m, fd, "validation_all", "validation")
        cal = outputs(m, fd, "s3_calibration_reference", "calibration")
    if state_digest(m) != meta2["state_sha256"]:
        raise RuntimeError("the model changed while scoring")
    if len(cal["target_ts"]) != 329:
        raise RuntimeError("calibration must be the 329 approved windows")
    np.savez_compressed(os.path.join(ckdir, "outputs.npz"), **{f"held_{k}": v for k, v in held.items()}, **{f"cal_{k}": v for k, v in cal.items()})
    labels = folds.heldout_labels(fd)
    agg, fault, det = single_seed_metrics(held, cal, labels, seed)
    sd = os.path.join(out_dir, fold, f"seed_{seed}")
    write_predictions(os.path.join(sd, "predictions.csv"), agg, D)
    res = {"fold": fold, "seed": seed, "training_folds": fd.fold["training"], "held_out_cases": fd.fold["cases"][fold], "n_labels": len(labels),
           "windows": {"held_out": int(len(held["target_ts"])), "calibration": int(len(cal["target_ts"])),
                       "s1_training": int(len(md.objective_rows(fd, "s1_pretrain"))), "s2_training_events": int(len(md.objective_rows(fd, "s2_fault_finetune")))},
           "epochs": {"s1": len(h1), "s2": len(h2)}, "class_weights": weights.summary()["class_weights"], "fault": fault, "detection_primary": det,
           "checkpoints": {"s1_pt_sha256": sha(p1 + ".pt"), "s2_pt_sha256": sha(p2 + ".pt"), "s2_state_sha256": meta2["state_sha256"]},
           "outputs_npz_sha256": sha(os.path.join(ckdir, "outputs.npz")), "predictions_csv_sha256": sha(os.path.join(sd, "predictions.csv")),
           "histories": {"s1": h1, "s2": h2}, "elapsed_seconds": round(time.time() - t0, 1)}
    dump(os.path.join(sd, "result.json"), res)
    log(f"[v2 cv {fold} seed {seed}] done in {res['elapsed_seconds']}s: macro-F1 {fault['macro_f1']:.4f} acc {fault['accuracy']:.4f} | primary F1 {det['f1']:.4f} TP {det['TP']} FP {det['FP']}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", required=True, choices=folds.FOLDS)
    ap.add_argument("--seed", required=True, type=int, choices=range(5))
    ap.add_argument("--ckpt-root", default=CKPT_ROOT)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    run(a.fold, a.seed, a.ckpt_root, a.out)


if __name__ == "__main__":
    main()
