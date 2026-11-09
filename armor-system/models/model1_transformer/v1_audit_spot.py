"""v1 design audit: can 329 calibration-reference windows support ARMOR's SPOT thresholding? (Linear 298-51; analysis only, no training.)

    .venv/bin/python -I armor-system/models/model1_transformer/v1_audit_spot.py

ARMOR's ORIGINAL `SPOT` class (initialisation step only, as ARMOR uses it) and `get_threshold` are loaded verbatim from the released
source with `ast`. Inputs are the v0 forecast-error M2 scores of the 329 CALIBRATION windows of each of the five seeds (existing
checkpoints, eval mode, calibration role only); no validation or test window is read and nothing is trained. The audit measures
(1) the dependence between adjacent calibration windows, (2) what SPOT does on the real calibration scores, and (3) how stable it is
under a 7-window block bootstrap versus a plain (independence-assuming) bootstrap.
"""
import ast
import hashlib
import json
import os
import sys
import warnings
from math import floor, log

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "armor-system", "data"), os.path.join(REPO, "armor-system", "models"), os.path.join(REPO, "armor-system", "evaluation", "detection")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ARMOR = os.path.join(REPO, "armor-system", "data", "raw", "armor_zenodo_21744694", "ARMOR")
SPOT_FILE = os.path.join(ARMOR, "utils", "spot.py")
DETECTOR_FILE = os.path.join(ARMOR, "models", "diagnosis_tasks", "anomaly_detector.py")
OUT = os.path.join(HERE, "results", "v1_audit", "spot_feasibility_v0_calibration_scores.json")
Q, LEVEL = 0.1, 0.95                                   # ARMOR D1 config: downstream_param.AD.q and .level
N_BOOT, BLOCK, SEED = 500, 7, 20261028


def load_armor_spot():
    """(SPOT class, get_threshold function, source hashes): ARMOR's code executed unchanged."""
    with open(SPOT_FILE) as fh:
        src = fh.read()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "SPOT")
    seg = ast.get_source_segment(src, cls)
    ns = {"np": np, "pd": pd, "minimize": minimize, "log": log, "floor": floor, "tqdm": None, "plt": None}
    exec(compile(seg, SPOT_FILE + "::SPOT", "exec"), ns)                                   # noqa: S102
    with open(DETECTOR_FILE) as fh:
        dsrc = fh.read()
    fns = [ast.get_source_segment(dsrc, n) for n in ast.parse(dsrc).body if isinstance(n, ast.FunctionDef) and n.name in ("run_spot", "get_threshold")]
    ns2 = {"SPOT": ns["SPOT"], "np": np}
    for f in fns:
        exec(compile(f, DETECTOR_FILE + "::get_threshold", "exec"), ns2)                   # noqa: S102
    def file_sha(path):
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    prov = {"spot.py_sha256": file_sha(SPOT_FILE), "anomaly_detector.py_sha256": file_sha(DETECTOR_FILE), "SPOT_class_source_sha256": hashlib.sha256(seg.encode()).hexdigest()}
    return ns["SPOT"], ns2["get_threshold"], prov


def spot_fit(SPOT, scores, q=Q, level=LEVEL):
    """ARMOR's initialisation step, exposing its internals. Returns a dict (error text instead of a threshold when the fit raises)."""
    s = np.asarray(scores, np.float64)
    out = {"n": int(len(s))}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SPOT(q)
            m.fit(s, [])
            m.initialize(level=level, verbose=False)
        g, sg, _ = m._grimshaw()
        out.update({"t": float(m.init_threshold), "Nt": int(m.Nt), "r": float(m.n * m.proba / m.Nt), "gamma": float(g), "sigma": float(sg), "threshold": float(m.extreme_quantile)})
    except Exception as e:                                                                  # noqa: BLE001  (ARMOR falls back to a percentile on any exception)
        out.update({"error": f"{type(e).__name__}: {e}"})
    return out


def autocorrelation(x, max_lag=7):
    x = np.asarray(x, np.float64) - np.mean(x)
    den = float((x * x).sum())
    return [float((x[:-k] * x[k:]).sum() / den) for k in range(1, max_lag + 1)]


def effective_sample_size(x):
    """n / (1 + 2 * sum of leading positive autocorrelations)."""
    rho = autocorrelation(x, max_lag=len(x) // 4)
    s = 0.0
    for r in rho:
        if r <= 0:
            break
        s += r
    return float(len(x) / (1 + 2 * s))


def block_resample(n, block, rng):
    starts = rng.integers(0, n - block + 1, size=-(-n // block))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def stability(SPOT, scores, block, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows, n = [], len(scores)
    for _ in range(n_boot):
        idx = block_resample(n, block, rng) if block > 1 else rng.integers(0, n, size=n)
        rows.append(spot_fit(SPOT, scores[idx]))
    ok = [r for r in rows if "threshold" in r]
    th = np.array([r["threshold"] for r in ok])
    q90 = float(np.percentile(scores, 90))
    return {"block": block, "resamples": n_boot, "fits_raising_an_exception": len(rows) - len(ok), "Nt_min_median_max": [int(min(r["Nt"] for r in ok)), float(np.median([r["Nt"] for r in ok])), int(max(r["Nt"] for r in ok))],
            "gamma_negative_share": float(np.mean([r["gamma"] < 0 for r in ok])), "gamma_exactly_zero_share": float(np.mean([r["gamma"] == 0 for r in ok])), "gamma_median": float(np.median([r["gamma"] for r in ok])),
            "threshold_median": float(np.median(th)), "threshold_p2.5_p97.5": [float(x) for x in np.percentile(th, [2.5, 97.5])], "threshold_iqr_over_median": float((np.percentile(th, 75) - np.percentile(th, 25)) / np.median(th)),
            "threshold_below_initial_t_share": float(np.mean([r["threshold"] < r["t"] for r in ok])), "full_sample_empirical_p90": q90}


def main():
    import torch
    from model1_transformer import run_detection_v0 as rd, training as T, window_errors as we
    from model1_transformer.spec import AdapterSpec
    from prep import detection_calibration as dc, m1_dataset as md
    torch.set_num_threads(1)
    SPOT, get_threshold, prov = load_armor_spot()
    D = md.Model1Data.load()
    cfg = T.load_train_config()
    spec = AdapterSpec.from_d1(D.schema, D.applicable)
    loader = md.evaluation_loader(md.evaluation_dataset(D, "s3_calibration_reference"), batch_size=64)
    res = {"purpose": "feasibility of ARMOR SPOT thresholding on the 329 calibration-reference windows (v0 forecast-error M2 scores); no validation or test window read",
           "armor_parameters": {"q": Q, "level": LEVEL, "initial_threshold_index": "S[int(level * n)] (ascending order statistic)", "peaks": "scores strictly above t, minus t", "quantile": "t + (sigma/gamma) * ((n*q/Nt)^(-gamma) - 1)",
                                "streaming": "not used by ARMOR (SPOT.fit(init, []) then initialize only)", "fallback_on_exception": "sorted(scores)[int(len * level)]", "invalid_if": "threshold <= 0 (ARMOR AD)"},
           "armor_regime_note": "with level 0.95 and q 0.1, r = n*q/Nt is about 2 whatever n is, so the extreme quantile lies BELOW the peaks-over-threshold cut t (about the 90th percentile)",
           "provenance": prov, "seeds": {}}
    for s in T.SEEDS:
        model, _ = rd.load_s1(D, cfg, spec, s)
        r = we.collect_window_errors(model, loader, allowed_roles=("calibration",))
        order = np.argsort(r["target_ts"], kind="stable")
        errors = r["errors"][order]
        st = dc.fit_statistics(errors)
        scores = np.asarray(dc.window_scores(errors, st)["m2_max_normalised"], np.float64)
        direct = spot_fit(SPOT, scores)
        res["seeds"][str(s)] = {"n_windows": int(len(scores)), "autocorrelation_lag1_to_7": autocorrelation(scores), "effective_sample_size": effective_sample_size(scores),
                                "spot_on_real_scores": direct, "armor_get_threshold": float(get_threshold(scores, Q, LEVEL)),
                                "calibration_windows_above_spot_threshold": float((scores > direct["threshold"]).mean()) if "threshold" in direct else None, "empirical_percentiles": {str(p): float(np.percentile(scores, p)) for p in (90, 95, 99)},
                                "stability_block7": stability(SPOT, scores, BLOCK, N_BOOT, SEED + s), "stability_independent": stability(SPOT, scores, 1, N_BOOT, SEED + s)}
        d = res["seeds"][str(s)]
        print(f"seed {s}: ESS {d['effective_sample_size']:.0f}/329, lag1 {d['autocorrelation_lag1_to_7'][0]:.2f}, Nt {direct.get('Nt')}, r {direct.get('r', float('nan')):.2f}, "
              f"SPOT thr {direct.get('threshold', float('nan')):.1f} (p90 {d['empirical_percentiles']['90']:.1f}), block-boot IQR/median {d['stability_block7']['threshold_iqr_over_median']:.2f} vs independent {d['stability_independent']['threshold_iqr_over_median']:.2f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
