"""Paired case-level bootstrap of the five-seed v1 fault-type predictions vs the frozen XGBoost control (validation predictions only).

    .venv/bin/python -I armor-system/models/model1_transformer/bootstrap_v1.py

Reuses `bootstrap_compare` (same 1,000 paired resamples, fixed seed, frozen five-class metric definition, 95% percentile intervals): it reads
only the saved per-case validation predictions of both models and the frozen XGBoost provenance; no dataset, model, calibration or test
observation is reachable. The v1 predictions are results/experiment_v1/seed_N/predictions.csv.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "armor-system", "models"))

import numpy as np  # noqa: E402

from model1_transformer import bootstrap_compare as bc  # noqa: E402

V1_RES = os.path.join(HERE, "results", "experiment_v1")
OUT = os.path.join(HERE, "results", "comparison_v1", "bootstrap_validation_v1_vs_xgboost.json")


def main():
    mt = bc._metrics_module()
    cases, y, pv1, pxg, files = bc.load_inputs(tr_dir=V1_RES)
    r = bc.bootstrap(y, pv1, pxg, mt=mt)
    out = {"experiment": "paired case-level bootstrap, five-seed Model 1 v1 vs frozen XGBoost control", "evaluated_on": "25 validation cases (saved predictions only)", "resamples": bc.N_BOOT, "bootstrap_seed": bc.SEED,
           "resampling": "25 cases with replacement; the same cases for both models and all five seeds; numpy default_rng(seed).integers(0, 25, (1000, 25))", "interval": "95% percentile interval (2.5th/97.5th percentile, linear interpolation)",
           "metric_definition": "frozen XGBoost control metrics.py: macro-F1 averages ALL five labels, zero-division counts 0, so a class absent from a resample keeps F1 0 for both models; weighted-F1 weights by resample support",
           "statistic": "mean over the five seeds of each model, then v1 minus XGBoost", "class_order": list(bc.CLASS_NAMES), "case_index": cases.tolist(), "support": np.bincount(y, minlength=5).tolist(), **r,
           "reading": {"fraction_positive": "share of resamples with a positive difference; NOT a probability that v1 is better",
                       "seeds": "the five seeds enter only as averaging; seed variance is not a source of uncertainty across incidents",
                       "downward_shift": "an absent class keeps F1 0, so resample means sit below the observed values while the paired difference is much less affected",
                       "pending": "no multiple-comparison adjustment (two metrics reported)"},
           "limitations": ["25 validation cases; case-level accuracy moves in steps of 0.04.",
                           "Node Memory has ONE validation case: missing from roughly 36-38% of resamples (F1 0 for both models there); when present one flip moves its recall between 0 and 1.",
                           "v1's checkpoints were selected with validation data (Stage 2: fault NLL + balanced BCE) and XGBoost used a fixed configuration without selection: the comparison favours v1 slightly; the locked test partition is the unbiased estimate.",
                           "Cases are resampled as independent although the 25 cases come from 23 incident groups; percentile intervals from 25 cases are approximate."],
           "provenance": {"input_files_sha256": {k: bc.sha256_file(v) for k, v in sorted(files.items())}, "frozen_case_features_provenance_sha256": bc.sha256_file(os.path.join(bc.XGB_RES, "case_features_provenance.json")),
                          "code_sha256": {"bootstrap_compare.py": bc.sha256_file(bc.__file__), "bootstrap_v1.py": bc.sha256_file(__file__), "xgb_metrics.py": bc.sha256_file(os.path.join(bc.XGB_DIR, "metrics.py"))},
                          "numpy": np.__version__, "python": sys.version.split()[0], "accessed": "saved validation predictions only; no dataset, model, calibration or test observation"}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    for m in bc.METRICS:
        s = out["metrics"][m]
        print(m, {k: (round(v["mean"], 4), round(v["ci95_low"], 4), round(v["ci95_high"], 4)) if isinstance(v, dict) else round(v, 4) for k, v in s.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
