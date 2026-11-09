"""Paired case-level bootstrap of the five-seed Transformer vs the frozen XGBoost control (validation only; no retraining).

    .venv/bin/python -I armor-system/models/model1_transformer/bootstrap_compare.py

Reads ONLY the saved per-case validation predictions of both models (5 seeds each) and the frozen XGBoost provenance JSON.
It imports no dataset, no torch and no model code, so no calibration or held-out test observation can be reached.

Procedure: 1,000 resamples of the 25 validation cases with replacement, the SAME resampled cases for both models and all seeds
(numpy default_rng(SEED).integers). In each resample every seed of each model is scored with the frozen XGBoost metric code
(`metrics.py`: macro-F1 over ALL five labels, a class with an empty denominator counts 0, so a class absent from the resample
keeps its place in the macro average), the five seed values are averaged per model, and the Transformer-minus-XGBoost
difference of those means is recorded. Intervals are 95% percentile intervals (2.5th and 97.5th percentiles, linear interpolation).
"""
import csv
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
XGB_DIR = os.path.join(REPO, "armor-system", "models", "model1_xgboost_control")
XGB_RES = os.path.join(XGB_DIR, "results", "xgb_control_v1")
TR_RES = os.path.join(HERE, "results", "experiment_v0")
OUT = os.path.join(HERE, "results", "comparison_v0", "bootstrap_validation_v1.json")
SEEDS = (0, 1, 2, 3, 4)
N_BOOT, SEED = 1000, 20261028
CLASS_NAMES = ("Container Hardware", "Container Network", "Node CPU", "Node Disk", "Node Memory")
METRICS = ("macro_f1", "weighted_f1")


class ComparisonError(ValueError):
    """The two models' saved predictions do not describe the same cases, labels or class order."""


def _metrics_module():
    spec = importlib.util.spec_from_file_location("xgb_metrics", os.path.join(XGB_DIR, "metrics.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_predictions(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {"case_index": np.array([int(r["case_index"]) for r in rows]), "true": np.array([int(r["true_class"]) for r in rows]),
            "pred": np.array([int(r["predicted_class"]) for r in rows]), "true_name": [r["true_name"] for r in rows], "pred_name": [r["predicted_name"] for r in rows]}


def load_inputs(tr_dir=TR_RES, xgb_dir=XGB_RES, seeds=SEEDS):
    """Return (case_index, y_true, pred_tr (S,25), pred_xgb (S,25), input_files) after verifying that every file agrees on cases, labels, class order."""
    with open(os.path.join(xgb_dir, "case_features_provenance.json")) as fh:
        frozen = json.load(fh)
    if tuple(frozen["class_order"]) != CLASS_NAMES:
        raise ComparisonError(f"frozen class order {frozen['class_order']} != {list(CLASS_NAMES)}")
    files, tr, xg = {}, [], []
    for s in seeds:
        for tag, d, name, acc in (("transformer", tr_dir, os.path.join(f"seed_{s}", "predictions.csv"), tr), ("xgboost", xgb_dir, os.path.join("per_seed", f"seed_{s}_predictions.csv"), xg)):
            path = os.path.join(d, name)
            files[f"{tag}/seed_{s}"] = path
            acc.append(read_predictions(path))
    ref = tr[0]
    if ref["case_index"].tolist() != frozen["validation_case_index"]:
        raise ComparisonError("Transformer cases differ from the frozen XGBoost validation cases")
    for tag, group in (("transformer", tr), ("xgboost", xg)):
        for s, p in zip(seeds, group):
            if p["case_index"].tolist() != ref["case_index"].tolist():
                raise ComparisonError(f"{tag} seed {s}: case ids/order differ")
            if p["true"].tolist() != ref["true"].tolist():
                raise ComparisonError(f"{tag} seed {s}: true labels differ")
            if ((p["pred"] < 0) | (p["pred"] > 4)).any() or ((p["true"] < 0) | (p["true"] > 4)).any():
                raise ComparisonError(f"{tag} seed {s}: class id outside 0..4")
            if p["true_name"] != [CLASS_NAMES[k] for k in p["true"]] or p["pred_name"] != [CLASS_NAMES[k] for k in p["pred"]]:
                raise ComparisonError(f"{tag} seed {s}: class names do not follow the five-class order")
    if np.bincount(ref["true"], minlength=5).tolist() != frozen["class_distribution"]["validation"]:
        raise ComparisonError("validation class distribution differs from the frozen provenance")
    return ref["case_index"], ref["true"], np.stack([p["pred"] for p in tr]), np.stack([p["pred"] for p in xg]), files


def five_seed_means(mt, y, preds):
    """Mean over seeds of macro-F1 and weighted-F1 with the frozen five-class definition (absent classes count 0)."""
    vals = [mt.summary(y, p) for p in preds]
    return {m: float(np.mean([v[m] for v in vals])) for m in METRICS}


def interval(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {"mean": float(np.mean(x)), "ci95_low": float(lo), "ci95_high": float(hi)}


def bootstrap(y, pred_tr, pred_xgb, n_boot=N_BOOT, seed=SEED, mt=None):
    mt = mt or _metrics_module()
    n = len(y)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    res = {m: {"transformer": np.empty(n_boot), "xgboost": np.empty(n_boot)} for m in METRICS}
    absent = np.zeros(5, np.int64)
    for b in range(n_boot):
        yb = y[idx[b]]
        absent += np.bincount(yb, minlength=5) == 0
        for tag, preds in (("transformer", pred_tr), ("xgboost", pred_xgb)):
            v = five_seed_means(mt, yb, preds[:, idx[b]])
            for m in METRICS:
                res[m][tag][b] = v[m]
    out = {"observed": {tag: five_seed_means(mt, y, preds) for tag, preds in (("transformer", pred_tr), ("xgboost", pred_xgb))}, "metrics": {}}
    out["observed"]["difference"] = {m: out["observed"]["transformer"][m] - out["observed"]["xgboost"][m] for m in METRICS}
    for m in METRICS:
        d = res[m]["transformer"] - res[m]["xgboost"]
        out["metrics"][m] = {"transformer": interval(res[m]["transformer"]), "xgboost": interval(res[m]["xgboost"]), "difference_transformer_minus_xgboost": interval(d),
                             "fraction_difference_positive": float((d > 0).mean()), "fraction_difference_zero": float((d == 0).mean()), "fraction_difference_negative": float((d < 0).mean())}
    out["resamples_without_class"] = {CLASS_NAMES[k]: float(absent[k] / n_boot) for k in range(5)}
    return out


def main():
    mt = _metrics_module()
    cases, y, ptr, pxg, files = load_inputs()
    r = bootstrap(y, ptr, pxg, mt=mt)
    out = {"experiment": "paired case-level bootstrap, five-seed Transformer v0 vs frozen XGBoost control", "evaluated_on": "25 validation cases (saved predictions only)",
           "resamples": N_BOOT, "bootstrap_seed": SEED, "resampling": "25 cases with replacement; the same cases for both models and all five seeds; numpy default_rng(seed).integers(0, 25, (1000, 25))",
           "interval": "95% percentile interval (2.5th/97.5th percentile, linear interpolation)",
           "metric_definition": "frozen XGBoost control metrics.py: macro-F1 averages ALL five labels, zero-division counts 0, so a class absent from a resample keeps F1 0 for both models; weighted-F1 weights by resample support",
           "statistic": "mean over the five seeds of each model, then Transformer minus XGBoost", "class_order": list(CLASS_NAMES), "case_index": cases.tolist(),
           "support": np.bincount(y, minlength=5).tolist(), **r,
           "reading": {"fraction_positive": "share of resamples with a positive difference; NOT a probability that the Transformer is better",
                       "seeds": "the five seeds enter only as averaging; seed variance is not a source of uncertainty across incidents",
                       "downward_shift": "because an absent class keeps F1 0, resample means sit below the observed values (macro-F1 most), while the paired difference is much less affected",
                       "pending": "no paired test beyond this bootstrap; no multiple-comparison adjustment (two metrics reported)"},
           "limitations": ["25 validation cases: every case is 4% of the data, and case-level accuracy moves in steps of 0.04.",
                           "Node Memory has ONE validation case: it is missing from roughly 36-38% of resamples (analytically (24/25)^25 = 36%; F1 0 for both models there), and when present one flip changes its recall between 0 and 1 and macro-F1 by up to 0.2; Node CPU has 3 cases.",
                           "Both models were chosen/stopped with validation data (Transformer: early stopping on validation; XGBoost: fixed configuration, no selection), so the Transformer's validation numbers are optimistically biased and the comparison favours it slightly; the locked test partition is the unbiased estimate.",
                           "The bootstrap resamples cases, treats them as independent although 25 cases come from 23 incident groups, and does not include training randomness beyond averaging five seeds.",
                           "Percentile intervals from 25 cases are approximate; with a discrete metric some resamples tie exactly."],
           "provenance": {"input_files_sha256": {k: sha256_file(v) for k, v in sorted(files.items())}, "frozen_case_features_provenance_sha256": sha256_file(os.path.join(XGB_RES, "case_features_provenance.json")),
                          "code_sha256": {"bootstrap_compare.py": sha256_file(__file__), "xgb_metrics.py": sha256_file(os.path.join(XGB_DIR, "metrics.py"))},
                          "numpy": np.__version__, "python": sys.version.split()[0], "accessed": "saved validation predictions only; no dataset, model, calibration or test observation"}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    for m in METRICS:
        s = out["metrics"][m]
        print(m, {k: (round(v["mean"], 4), round(v["ci95_low"], 4), round(v["ci95_high"], 4)) if isinstance(v, dict) else round(v, 4) for k, v in s.items()})
    print("observed", {t: {k: round(v, 4) for k, v in d.items()} for t, d in out["observed"].items()})
    print("absent", out["resamples_without_class"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
