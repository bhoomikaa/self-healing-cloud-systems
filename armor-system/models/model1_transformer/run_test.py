"""One-shot test runner for Model 1 v1, v0 and the XGBoost control (Linear 298-51, TEST_PLAN.md).

    .venv/bin/python -I armor-system/models/model1_transformer/run_test.py --mode validation --out DIR [--missing-check]   # dry run, repeatable
    .venv/bin/python -I armor-system/models/model1_transformer/run_test.py --mode test                                    # ONCE, after the freeze commit

--mode test verifies the freeze record against disk (test_freeze.verify: tracked files, checkpoints, feature cache, ARMOR inputs, plan,
protocol, clean tree), writes the one-shot markers (exclusive create; an existing marker refuses the run) and only then reads a test
window or label, through test_freeze.unlocked(token). Output goes to results/test_v1/ and its subfolders only. Worker processes
(missing data per seed, XGBoost features, XGBoost training: torch and xgboost cannot share a macOS process) re-verify the freeze and
the marker's run id.

Attempt 2 (TEST_INCIDENT_attempt1.md): attempt 1 stopped after core seeds 0-3 when its session ended, and its XGBoost-features worker
had been refused by the output check (which then accepted results/test_v1/ itself only, not its subfolders). Attempt 2 uses its own
marker names (MARKERS; refused if either exists), starts only on an empty results/test_v1/, and after completion compares seeds 0-3
byte for byte with results/test_v1_attempt1_aborted/ and records the outcome in results/test_v1/attempt1_consistency.json.

--mode validation runs the identical code on the validation period (no freeze, no marker, no official labels, which are all
test-period) into DIR, and checks it against the committed validation results: v1 predictions and detection, v0 predictions,
D4 (detection_v0), the XGBoost predictions, and with --missing-check the seed 0 missing-data cells (ARMOR scenarios, mask 0 per rate).

Rows (TEST_PLAN.md): fault type F1-F6, detection D0/DE/D1-D4 with our scoring, the official ARMOR rows A1 (v1 primary, ensemble as a
secondary line) and A2 (D3 at the 99th and 90th percentile) plus the released-interval reference, and the missing-data rows.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "armor-system", "data"), os.path.join(REPO, "armor-system", "models"), os.path.join(REPO, "armor-system", "evaluation", "detection")]

import numpy as np  # noqa: E402

from model1_transformer import test_freeze as tf  # noqa: E402   (no torch: the XGBoost worker imports this module too)

SEEDS = (0, 1, 2, 3, 4)
PY = os.path.join(REPO, ".venv", "bin", "python")
V1_RES = os.path.join(HERE, "results", "experiment_v1")
V0_RES = os.path.join(HERE, "results", "experiment_v0")
D4_RES = os.path.join(HERE, "results", "detection_v0")
XGB_RES = os.path.join(REPO, "armor-system", "models", "model1_xgboost_control", "results", "xgb_control_v1")
D3_P = ("90.0", "99.0")
D4_P = "99.0"
M2 = "m2_max_normalised"
MARKERS = (os.path.join(tf.TEST_OUT, "TEST_RUN_MARKER_attempt2.json"), os.path.join(tf.CKPT_ROOT, "model1_test_run_marker_attempt2.json"))
ATTEMPT1_DIR = os.path.join(HERE, "results", "test_v1_attempt1_aborted")
ATTEMPT1_FILES = ("result.json", "predictions_v1.csv", "predictions_v0.csv")


def say(*a):
    print(*a, flush=True)


def dump(path, obj):
    from model1_transformer import run_detection_v0 as rd
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rd._py(obj), fh, indent=1, sort_keys=True)
        fh.write("\n")


def jload(path):
    with open(path) as fh:
        return json.load(fh)


def rbytes(path):
    with open(path, "rb") as fh:
        return fh.read()


# ------------------------------------------------------------------------------------------------ setup (torch processes)
def setup(period, token):
    import torch
    from model1_transformer import scoring_sets as ss, training as T
    from model1_transformer.spec import AdapterSpec
    from prep import m1_dataset as md
    torch.set_num_threads(1)
    D = md.Model1Data.load()
    cfg = T.load_train_config()
    spec = AdapterSpec.from_d1(D.schema, D.applicable)
    sset = ss.build(D, period, token)
    labels = {"ours": sset["labels"]}
    if period == "test":
        from model1_transformer import official_scoring as osc
        labels["official"] = osc.load_labels(token)["labels"]
    return D, cfg, spec, sset, labels


def load_models(D, cfg, spec, seed, which=("v1_s2", "v1_s1", "v0_s1", "v0_s2")):
    """Checkpoints verified against the data/architecture/config provenance AND their experiment records (file sha256, epoch)."""
    import torch
    from model1_transformer import checkpoint as ck, run_detection_v0 as rd, run_v1 as r1, training as T, training_v1 as tv
    out, binding = {}, {}
    p1 = r1.provenance(D, spec, cfg)
    e1 = {k: p1[k] for k in ("manifest_content_sha256", "fit_only_scaler_file_sha256", "spec_fingerprint", "train_config_sha256", "model")}
    p0 = ck.data_provenance(D, spec, cfg, repo=REPO)
    e0 = {k: p0[k] for k in ("manifest_content_sha256", "fit_only_scaler_file_sha256", "feature_cache_content_sha256", "preprocessing_config_sha256", "spec_fingerprint", "train_config_sha256")}
    for key in which:
        ver, st = key.split("_")
        root, res, build, expect = ((r1.DEFAULT_CKPT, V1_RES, tv.build_model_v1, e1) if ver == "v1" else (rd.DEFAULT_CKPT, V0_RES, T.build_model, e0))
        path = os.path.join(root, f"seed_{seed}", f"{st}_best")
        rec = jload(os.path.join(res, f"seed_{seed}", "run_meta.json"))
        pt = tf.sha(path + ".pt")
        if pt != rec["checkpoints"][f"{st}_pt_sha256"]:
            raise RuntimeError(f"{key} seed {seed}: checkpoint file differs from the experiment record")
        m = build(spec, seed, torch.device("cpu"))
        meta = ck.load_checkpoint(path, m, expect=expect, stage=st, seed=seed)
        if meta["epoch"] != rec["selected"][f"{st}_epoch"]:
            raise RuntimeError(f"{key} seed {seed}: checkpoint epoch differs from the experiment record")
        m.eval()
        out[key], binding[key] = m, {"pt_sha256": pt, "state_sha256": meta["state_sha256"], "epoch": meta["epoch"]}
    return out, binding


def collect(model, D, sset, role):
    from model1_transformer import detection_v1 as dv, scoring_sets as ss
    return dv.collect_outputs(model, ss.loader(D, sset["detection_rows"], f"{sset['period']}_scoring"), (role,))


def collect_cal(model, D):
    from model1_transformer import detection_v1 as dv, run_v1 as r1
    return dv.collect_outputs(model, r1.view(D, "s3_calibration_reference"), ("calibration",))


# ------------------------------------------------------------------------------------------------ fault type
def fault_eval(out, D, sset, mt):
    from model1_transformer import case_eval as ce, detection_v1 as dv
    ev = dv.event_windows(out)
    if set(int(x) for x in out["start_index"][ev]) != set(int(x) for x in D.table.start[sset["event_rows"]]):
        raise RuntimeError("scored event windows differ from the scoring set's approved event windows")
    agg = ce.aggregate_cases(out["case_index"][ev], dv.softmax(out["fault_logits"][ev]), out["fault_class"][ev])
    return agg, fault_metrics_from_agg(agg, mt), ce.minute_unions(out["case_index"][ev], out["timestamps"][ev])


def fault_metrics_from_agg(agg, mt):
    from model1_transformer import detection_v1 as dv
    m = mt.summary(agg["true_class"], agg["pred_class"])
    m.update({"macro_f1_excluding_node_memory": mt.macro_f1_excluding(agg["true_class"], agg["pred_class"], 4), "case_nll": dv.case_level_nll(agg),
              "family_diagnostic": mt.family_diagnostic(agg["true_class"], agg["pred_class"])})
    return m


def write_predictions(path, agg, D):
    from model1_transformer import case_eval as ce
    rows = ce.xgb_style_rows(agg, group_id=D.cases["group_id"][agg["case_index"]])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in r.items()})


# ------------------------------------------------------------------------------------------------ core: one seed
def core_seed(D, cfg, spec, seed, sset, labels, out_dir, spot, mt):
    from model1_transformer import detection_v1 as dv, run_detection_v0 as rd
    from model1_transformer.model import state_digest
    role = sset["period"]
    models, binding = load_models(D, cfg, spec, seed)
    before = {k: state_digest(m) for k, m in models.items()}
    cal = {k: collect_cal(models[k], D) for k in ("v1_s2", "v1_s1", "v0_s1")}
    ev = {k: collect(m, D, sset, role) for k, m in models.items()}
    if any(state_digest(m) != before[k] for k, m in models.items()):
        raise RuntimeError("a model changed during scoring")
    for o in ev.values():
        t = dv.sort_by_time(o)
        if not np.array_equal(t["target_ts"], sset["target_ts"]):
            raise RuntimeError("scored windows differ from the scoring set")
    sd = os.path.join(out_dir, f"seed_{seed}")
    fault, unions = {}, {}
    aggs = {}
    for name, key in (("v1", "v1_s2"), ("v0", "v0_s2")):
        aggs[name], fault[name], unions[name] = fault_eval(ev[key], D, sset, mt)
        write_predictions(os.path.join(sd, f"predictions_{name}.csv"), aggs[name], D)
    if unions["v1"] != unions["v0"]:
        raise RuntimeError("v1 and v0 classified different case windows")
    sup_th, sup_info = dv.supervised_thresholds(cal["v1_s2"], seed, spot, True)
    d3_stats, d3_th, d3_info = dv.forecast_calibration(cal["v1_s1"])
    d4_stats, d4_th, d4_info = dv.forecast_calibration(cal["v0_s1"], rd.PERCENTILES)
    det = {}
    for lab, ls in labels.items():
        det[lab] = {"supervised": dv.supervised_validation(ev["v1_s2"], sup_th, ls),
                    "d3_forecast_error_stage1": dv.forecast_validation(ev["v1_s1"], d3_stats, d3_th, ls),
                    "d4_v0": dv.forecast_validation(ev["v0_s1"], d4_stats, d4_th, ls)}
    res = {"seed": seed, "binding": binding, "fault": fault, "case_minute_unions": unions["v1"],
           "calibration": {"supervised": {"thresholds": sup_th, **sup_info}, "d3": {**d3_info, "thresholds": d3_th}, "d4": {**d4_info, "thresholds": d4_th}},
           "detection": det}
    dump(os.path.join(sd, "result.json"), res)
    return res, aggs["v1"], cal["v1_s2"], ev["v1_s2"]


def run_core(period, token, out_dir, log=say):
    from model1_transformer import ensemble as en, run_experiment as rx, v1_audit_spot as au
    D, cfg, spec, sset, labels = setup(period, token)
    mt = rx._xgb_metrics()
    _, spot, _ = au.load_armor_spot()
    per, aggs, cals, evs = {}, [], [], []
    for s in SEEDS:
        t0 = time.time()
        r, a, c, e = core_seed(D, cfg, spec, s, sset, labels, out_dir, spot, mt)
        per[s] = r
        aggs.append(a), cals.append(c), evs.append(e)
        log(f"[{period} core] seed {s} done in {time.time() - t0:.0f}s: v1 macro-F1 {r['fault']['v1']['macro_f1']:.4f}, primary F1 {r['detection']['ours']['supervised']['primary']['f1']:.4f}")
    ens_agg = en.fault(aggs)
    write_predictions(os.path.join(out_dir, "ensemble", "predictions_v1_ensemble.csv"), ens_agg, D)
    cal_e, ev_e = en.detection_scores(cals), en.detection_scores(evs)
    tau, info = en.threshold(cal_e)
    ens = {"fault": fault_metrics_from_agg(ens_agg, mt), "detection": {lab: en.evaluate(ev_e, tau, ls) for lab, ls in labels.items()},
           "calibration": {"threshold": tau, **info}, "definition": "V1_PROTOCOL.md section 11.3 (secondary row; never chosen from)"}
    dump(os.path.join(out_dir, "ensemble", "result.json"), ens)
    meta = {"period": period, "scoring_set": {k: sset[k] for k in ("counts", "contiguous_target_minutes", "n_ambiguous_event_windows_dropped")} | {"n_cases": len(sset["cases"]), "n_labels": {k: len(v) for k, v in labels.items()}}}
    if period == "test":
        from model1_transformer import official_scoring as osc
        meta["armor_released_reference"] = osc.armor_reference(token, labels["official"])
    dump(os.path.join(out_dir, "core_meta.json"), meta)
    return per, ens, meta


# ------------------------------------------------------------------------------------------------ missing data (one seed per process)
def brief(r):
    from model1_transformer import run_v1 as r1
    return r1.brief(r)


def score_condition(cal, ev, clean, seed, labels):
    from model1_transformer import detection_v1 as dv, run_detection_v0 as rd
    out = {}
    th_sup = dv.supervised_thresholds(cal["v1_s2"], seed, None, bootstrap=False)[0]
    d3 = dv.forecast_calibration(cal["v1_s1"])[:2]
    v0 = dv.forecast_calibration(cal["v0_s1"], rd.PERCENTILES)[:2]
    for lab, ls in labels.items():
        cell = {}
        cell["v1_primary"] = {"recalibrated": brief(dv.supervised_validation(ev["v1_s2"], th_sup, ls)["primary"]),
                              "clean_thresholds": brief(dv.supervised_validation(ev["v1_s2"], clean["sup"], ls)["primary"])}
        for p in D3_P:
            cell[f"d3_p{p[:2]}"] = {mode: brief(dv.forecast_validation(ev["v1_s1"], st, {M2: {p: th[M2][p]}}, ls)[M2][p]) for mode, (st, th) in (("recalibrated", d3), ("clean_thresholds", clean["d3"]))}
        cell["v0_p99"] = {mode: brief(dv.forecast_validation(ev["v0_s1"], st, {M2: {D4_P: th[M2][D4_P]}}, ls)[M2][D4_P]) for mode, (st, th) in (("recalibrated", v0), ("clean_thresholds", clean["v0"]))}
        out[lab] = cell
    return out


def run_missing_seed(period, token, seed, out_dir, masks, log=say):
    from model1_transformer import detection_v1 as dv, missing_eval as me, run_detection_v0 as rd, scoring_sets as ss
    from model1_transformer.sim_missing import MissingBlocks
    D, cfg, spec, sset, labels = setup(period, token)
    models, binding = load_models(D, cfg, spec, seed, ("v1_s2", "v1_s1", "v0_s1"))

    def outs(view):
        return ({k: collect_cal(m, view) for k, m in models.items()}, {k: dv.collect_outputs(m, ss.loader(view, sset["detection_rows"], "scoring"), (period,)) for k, m in models.items()})

    cal0, ev0 = outs(D)
    clean = {"sup": dv.supervised_thresholds(cal0["v1_s2"], seed, None, bootstrap=False)[0], "d3": dv.forecast_calibration(cal0["v1_s1"])[:2], "v0": dv.forecast_calibration(cal0["v0_s1"], rd.PERCENTILES)[:2]}
    ours = {"ours": labels["ours"]}
    res = {"seed": seed, "binding": binding, "clean_reference": score_condition(cal0, ev0, clean, seed, labels), "armor": {}, "random": {}}
    for name, drop in me.armor_scenarios():
        obs = me.scenario_observed(D, drop)
        c, e = outs(me.masked_view(D, obs))
        res["armor"][name] = {"dropped": list(drop), "hides_all_vms": me.hides_all_vms(D, spec, obs), "models": score_condition(c, e, clean, seed, labels)}
    blocks = MissingBlocks(spec)
    for ri, rate in enumerate(me.RATES):
        res["random"][str(rate)] = {}
        for mi in range(1 if rate == 0.0 else masks):
            c, e = outs(me.masked_view(D, me.random_observed(D, blocks, ri, mi, rate)))
            res["random"][str(rate)][str(mi)] = score_condition(c, e, clean, seed, ours)
        log(f"[{period} missing seed {seed}] rate {rate} done")
    dump(os.path.join(out_dir, "missing", f"seed_{seed}.json"), res)
    return res


# ------------------------------------------------------------------------------------------------ XGBoost (two processes)
def xgb_features(period, token, feat_dir):
    """torch process: case matrices for 'fit' and the period, plus each case's minute union (alignment with Model 1)."""
    from prep import m1_case_features as cf, m1_dataset as md
    D = md.Model1Data.load()
    if period == "test":
        tf.check_token(token)
        with tf.unlocked(token):
            mats = {r: cf.build_case_matrix(D, r) for r in ("fit", "test")}
    else:
        mats = {r: cf.build_case_matrix(D, r) for r in ("fit", "validation")}
    os.makedirs(feat_dir, exist_ok=True)
    keys = ("X", "y", "case_index", "group_id", "n_windows", "n_minutes")
    np.savez(os.path.join(feat_dir, "features.npz"), **{f"{'fit' if r == 'fit' else 'eval'}_{k}": mats[r][k] for r in mats for k in keys})
    t = D.table
    unions = []
    for c in mats[period]["case_index"]:
        rows = [int(r) for r in D.selection([period], ["event_associated"], drop_ambiguous=True) if t.case_index[r] == c]
        ts = D.ts[cf.case_minutes(t, rows)]
        unions.append({"case_index": int(c), "first_sample_ts": int(ts[0]), "last_sample_ts": int(ts[-1]), "n_samples": int(len(ts)), "contiguous": bool((np.diff(ts) == 60).all())})
    with open(os.path.join(feat_dir, "unions.json"), "w") as fh:
        json.dump(unions, fh)


def xgb_train(feat_dir, out_dir):
    """xgboost process: the frozen configuration, trained on the 88 fitting cases, predicting the period's cases (five seeds)."""
    sys.path.insert(0, os.path.join(REPO, "armor-system", "models", "model1_xgboost_control"))
    import xgb_control as xc
    cfg, cfg_sha = xc.load_config()
    z = np.load(os.path.join(feat_dir, "features.npz"))
    fit = {k: z[f"fit_{k}"] for k in ("X", "y")}
    ev = {k: z[f"eval_{k}"] for k in ("X", "y", "case_index", "group_id")}
    if len(fit["y"]) != 88:
        raise RuntimeError("XGBoost must be fitted on the 88 fitting cases")
    w = xc.cf.inverse_frequency_weights(fit["y"], xc.K)
    per = []
    for seed in cfg["seeds"]:
        proba, model_sha = xc.train_predict(cfg, fit, ev, w, seed)
        pred = proba.argmax(1)
        m = xc.evaluate(ev["y"], pred)
        m.update({"seed": int(seed), "model_ubj_sha256": model_sha})
        xc.write_predictions(os.path.join(out_dir, "xgboost", f"seed_{seed}_predictions.csv"), ev, proba, pred)
        per.append(m)
    maj = xc.mt.majority_class(fit["y"], xc.K)
    majority = xc.evaluate(ev["y"], np.full(len(ev["y"]), maj, np.int64))
    majority.update({"predicted_class": int(maj), "predicted_name": xc.mt.CLASS_NAMES[maj]})
    xc.write_json(os.path.join(out_dir, "xgboost", "result.json"), {"config_sha256": cfg_sha, "per_seed": per, "mean_sd_across_seeds": xc.aggregate(per), "majority_control": majority})


# ------------------------------------------------------------------------------------------------ assembly (no torch, no xgboost)
def read_preds(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return np.array([int(r["case_index"]) for r in rows]), np.array([int(r["true_class"]) for r in rows]), np.array([int(r["predicted_class"]) for r in rows])


def assemble(out_dir, period):
    from model1_transformer import bootstrap_compare as bc, run_detection_v0 as rd, run_v1 as r1
    mt = bc._metrics_module()
    per = {s: jload(os.path.join(out_dir, f"seed_{s}", "result.json")) for s in SEEDS}
    preds = {}
    for name, f in (("v1", "predictions_v1.csv"), ("v0", "predictions_v0.csv")):
        preds[name] = [read_preds(os.path.join(out_dir, f"seed_{s}", f)) for s in SEEDS]
    preds["xgboost"] = [read_preds(os.path.join(out_dir, "xgboost", f"seed_{s}_predictions.csv")) for s in SEEDS]
    cases, y = preds["v1"][0][0], preds["v1"][0][1]
    for name, ps in preds.items():
        for c, t, _ in ps:
            if c.tolist() != cases.tolist() or t.tolist() != y.tolist():
                raise RuntimeError(f"{name}: cases or labels differ from v1")
    stack = {k: np.stack([p[2] for p in v]) for k, v in preds.items()}

    def relabel(r, a, b):
        txt = json.dumps(r).replace("transformer", a).replace("xgboost", b)
        return json.loads(txt)

    boot = {"v1_vs_xgboost": relabel(bc.bootstrap(y, stack["v1"], stack["xgboost"], mt=mt), "v1", "xgboost"),
            "v1_vs_v0": relabel(bc.bootstrap(y, stack["v1"], stack["v0"], mt=mt), "v1", "v0"),
            "design": {"resamples": bc.N_BOOT, "seed": bc.SEED, "cases": len(y), "support": np.bincount(y, minlength=5).tolist(), "statistic": "mean over five seeds per model, then difference"}}
    names = bc.CLASS_NAMES

    def fault_summary(ms):
        a = {k: rd.mean_sd([m[k] for m in ms]) for k in ("macro_f1", "weighted_f1", "accuracy", "macro_f1_excluding_node_memory")}
        a["per_class"] = {key: {names[k]: rd.mean_sd([m["per_class"][key][k] for m in ms]) for k in range(5)} for key in ("precision", "recall", "f1")}
        a["mean_confusion_matrix"] = np.mean([m["confusion_matrix"] for m in ms], axis=0).tolist()
        return a

    xr = jload(os.path.join(out_dir, "xgboost", "result.json"))
    fault = {"v1": fault_summary([per[s]["fault"]["v1"] for s in SEEDS]), "v0": fault_summary([per[s]["fault"]["v0"] for s in SEEDS]),
             "xgboost": fault_summary(xr["per_seed"]), "xgboost_majority_control": {k: xr["majority_control"][k] for k in ("accuracy", "macro_f1", "weighted_f1")},
             "v1_ensemble": jload(os.path.join(out_dir, "ensemble", "result.json"))["fault"]}
    det = {}
    labs = sorted(per[0]["detection"])
    for lab in labs:
        d = {"v1_primary": r1.cell_summary([per[s]["detection"][lab]["supervised"]["primary"] for s in SEEDS]),
             "d2_no_rolling": r1.cell_summary([per[s]["detection"][lab]["supervised"]["d2_no_rolling"] for s in SEEDS]),
             "d1": {n: r1.cell_summary([per[s]["detection"][lab]["supervised"][n] for s in SEEDS]) for n in sorted(per[0]["detection"][lab]["supervised"]) if n.startswith("d1_")}}
        for key in ("d3_forecast_error_stage1", "d4_v0"):
            d[key] = {m: {p: r1.cell_summary([per[s]["detection"][lab][key][m][p] for s in SEEDS]) for p in per[0]["detection"][lab][key][m]} for m in per[0]["detection"][lab][key]}
        d["v1_ensemble"] = jload(os.path.join(out_dir, "ensemble", "result.json"))["detection"][lab]
        det[lab] = d
    out = {"period": period, "fault_type": fault, "bootstrap": boot, "detection": det, "core_meta": jload(os.path.join(out_dir, "core_meta.json"))}
    if os.path.isdir(os.path.join(out_dir, "missing")):
        out["missing_data"] = missing_summary(out_dir)
    dump(os.path.join(out_dir, "summary.json"), out)
    return out


def missing_summary(out_dir):
    from model1_transformer import run_detection_v0 as rd
    per = [jload(os.path.join(out_dir, "missing", f)) for f in sorted(os.listdir(os.path.join(out_dir, "missing")))]
    fields = ("f1", "recall", "precision", "armor_false_positive_intervals", "coverage_padded_300s")

    def stat(cells):
        return {f: rd.mean_sd([c[f] for c in cells]) for f in fields} | {"n": len(cells), "n_no_score": int(sum(not c["scored"] for c in cells))}

    def models_stat(cells_by_seed):
        first = cells_by_seed[0]
        return {lab: {m: {md_: stat([c[lab][m][md_] for c in cells_by_seed]) for md_ in first[lab][m]} for m in first[lab]} for lab in first}

    out = {"seeds": [p["seed"] for p in per], "clean_reference": models_stat([p["clean_reference"] for p in per]), "armor_scenarios": {}, "random_rates": {}}
    for name in per[0]["armor"]:
        out["armor_scenarios"][name] = {"hides_all_vms": per[0]["armor"][name]["hides_all_vms"], "models": models_stat([p["armor"][name]["models"] for p in per])}
    for rate in per[0]["random"]:
        out["random_rates"][rate] = {"distinct_masks": len(per[0]["random"][rate]), "models": models_stat([c for p in per for c in p["random"][rate].values()])}
    return out


# ------------------------------------------------------------------------------------------------ validation dry-run checks
def validation_checks(out_dir, missing_check=False):
    """Equality of the runner's validation outputs with the committed validation results (the same code paths the test run uses)."""
    chk = {}
    for s in SEEDS:
        r = jload(os.path.join(out_dir, f"seed_{s}", "result.json"))
        v1det = jload(os.path.join(V1_RES, f"seed_{s}", "detection.json"))
        d4 = jload(os.path.join(D4_RES, f"seed_{s}.json"))["validation"]["methods"]
        chk[f"seed_{s}"] = {
            "v1_predictions_byte_identical": rbytes(os.path.join(out_dir, f"seed_{s}", "predictions_v1.csv")) == rbytes(os.path.join(V1_RES, f"seed_{s}", "predictions.csv")),
            "v0_predictions_byte_identical": rbytes(os.path.join(out_dir, f"seed_{s}", "predictions_v0.csv")) == rbytes(os.path.join(V0_RES, f"seed_{s}", "predictions.csv")),
            "v1_supervised_detection_identical": _same(r["detection"]["ours"]["supervised"], v1det["validation"]),
            "v1_thresholds_identical": _same(r["calibration"]["supervised"]["thresholds"], v1det["thresholds"]),
            "d3_identical": _same(r["detection"]["ours"]["d3_forecast_error_stage1"], v1det["d3_forecast_error_stage1"]["validation"]),
            "d4_identical_to_detection_v0": _same(r["detection"]["ours"]["d4_v0"], d4),
            "xgboost_predictions_byte_identical": rbytes(os.path.join(out_dir, "xgboost", f"seed_{s}_predictions.csv")) == rbytes(os.path.join(XGB_RES, "per_seed", f"seed_{s}_predictions.csv"))}
    if missing_check:
        new = jload(os.path.join(out_dir, "missing", "seed_0.json"))
        old, old9 = jload(os.path.join(V1_RES, "missing_seed_0.json")), jload(os.path.join(V1_RES, "missing_d3_p90_seed_0.json"))
        mapping = (("v1_primary", "v1_primary"), ("d3_p99", "v1_d3"), ("v0_p99", "v0"))
        ok = {}
        for name in old["armor"]:
            for a, b in mapping:
                ok[f"armor/{name}/{a}"] = _same(new["armor"][name]["models"]["ours"][a], old["armor"][name]["models"][b])
            ok[f"armor/{name}/d3_p90"] = _same(new["armor"][name]["models"]["ours"]["d3_p90"], old9["armor"][name]["d3_p90"])
        for rate in old["random"]:
            for a, b in mapping:
                ok[f"random/{rate}/0/{a}"] = _same(new["random"][rate]["0"]["ours"][a], old["random"][rate]["0"][b])
            ok[f"random/{rate}/0/d3_p90"] = _same(new["random"][rate]["0"]["ours"]["d3_p90"], old9["random"][rate]["0"])
        chk["missing_seed_0_mask_0"] = {"cells_compared": len(ok), "all_identical": all(ok.values()), "differing": [k for k, v in ok.items() if not v]}
    flat = [v for k, d in chk.items() if k.startswith("seed_") for v in d.values()] + ([chk["missing_seed_0_mask_0"]["all_identical"]] if missing_check else [])
    chk["all_identical"] = all(flat)
    return chk


def _same(a, b):
    from model1_transformer import run_detection_v0 as rd
    return json.dumps(rd._py(a), sort_keys=True) == json.dumps(rd._py(b), sort_keys=True)


# ------------------------------------------------------------------------------------------------ orchestration
def _worker(args, run_id, log_path):
    cmd = [PY, "-I", os.path.abspath(__file__), *args] + (["--run-id", run_id] if run_id else [])
    fh = open(log_path, "w")
    return subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT), fh


def _wait(procs):
    bad = []
    for name, (p, fh) in procs.items():
        if p.wait() != 0:
            bad.append(name)
        fh.close()
    if bad:
        raise RuntimeError(f"worker(s) failed: {bad}")


def out_ok(mode, path):
    """Test output must be results/test_v1/ or a subfolder of it; validation output must not be there."""
    p = os.path.abspath(path)
    inside = p == tf.TEST_OUT or p.startswith(tf.TEST_OUT + os.sep)
    return inside if mode == "test" else not inside


def worker_specs(period, out_dir, masks, missing_seeds):
    """Arguments of every worker process the orchestrator launches, by name."""
    feat = os.path.join(out_dir, "xgb_features")
    specs = {f"missing_{s}": ["--worker", "missing", "--mode", period, "--seed", str(s), "--out", out_dir, "--masks", str(masks)] for s in missing_seeds}
    specs["xgb_features"] = ["--worker", "xgb-features", "--mode", period, "--out", feat]
    specs["xgb_train"] = ["--worker", "xgb-train", "--mode", period, "--features", feat, "--out", out_dir]
    return specs


def start_test_run(markers=None):
    """Attempt 2: refuse if an attempt-2 marker exists or results/test_v1/ is not empty, then verify the freeze and write the markers."""
    markers = markers or MARKERS
    if os.path.isdir(tf.TEST_OUT) and os.listdir(tf.TEST_OUT):
        raise tf.FreezeError(f"{tf.TEST_OUT} is not empty; a test run starts only on an empty output directory")
    return tf.claim_test_run(markers=markers)


def compare_attempt1(out_dir, archive=ATTEMPT1_DIR, seeds=(0, 1, 2, 3)):
    """Byte-for-byte comparison of this run's seed files with the aborted attempt 1 (recorded, never used to change anything)."""
    files = {}
    for s in seeds:
        for f in ATTEMPT1_FILES:
            a, b = os.path.join(out_dir, f"seed_{s}", f), os.path.join(archive, f"seed_{s}", f)
            files[f"seed_{s}/{f}"] = {"identical": os.path.isfile(a) and os.path.isfile(b) and rbytes(a) == rbytes(b),
                                      "sha256_this_run": tf.sha(a) if os.path.isfile(a) else None, "sha256_attempt1": tf.sha(b) if os.path.isfile(b) else None}
    return {"archive": os.path.relpath(archive, REPO), "files": files, "all_identical": all(v["identical"] for v in files.values())}


def orchestrate(period, out_dir, run_id, token, masks, missing, missing_seeds):
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    specs = worker_specs(period, out_dir, masks, missing_seeds if missing else ())
    feat = os.path.join(out_dir, "xgb_features")
    procs = {name: _worker(specs[name], run_id, os.path.join(out_dir, "logs", f"{name}.log")) for name in specs if name.startswith("missing_")}
    procs["xgb_features"] = _worker(specs["xgb_features"], run_id, os.path.join(out_dir, "logs", "xgb_features.log"))
    run_core(period, token, out_dir)
    _wait({"xgb_features": procs.pop("xgb_features")})
    _wait({"xgb_train": _worker(specs["xgb_train"], run_id, os.path.join(out_dir, "logs", "xgb_train.log"))})
    v1_unions = jload(os.path.join(out_dir, "seed_0", "result.json"))["case_minute_unions"]
    if jload(os.path.join(feat, "unions.json")) != v1_unions:
        raise RuntimeError("XGBoost and Model 1 classified different case windows")
    _wait(procs)
    return assemble(out_dir, period)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("validation", "test"), required=True)
    ap.add_argument("--out")
    ap.add_argument("--worker", choices=("missing", "xgb-features", "xgb-train"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--features")
    ap.add_argument("--masks", type=int, default=10)
    ap.add_argument("--run-id")
    ap.add_argument("--no-missing", action="store_true")
    ap.add_argument("--missing-check", action="store_true", help="validation only: missing-data run for seed 0 with one mask per rate, compared with the committed cells")
    a = ap.parse_args(argv)
    for path in (a.out, a.features):
        if path and not out_ok(a.mode, path):
            ap.error(f"test output goes to {tf.TEST_OUT} or its subfolders only" if a.mode == "test" else "the test output directory is reserved for the test run")
    if a.worker:
        token = tf.resume_token(a.run_id, markers=MARKERS) if a.mode == "test" else None
        if a.worker == "missing":
            with tf.unlocked(token) if token else _null():
                run_missing_seed(a.mode, token, a.seed, a.out, a.masks)
        elif a.worker == "xgb-features":
            xgb_features(a.mode, token, a.out)
        else:
            xgb_train(a.features, a.out)
        return 0
    if a.mode == "test":
        token = start_test_run()                          # verifies the freeze and writes the attempt-2 markers BEFORE any test window is read
        out_dir = tf.TEST_OUT
        with tf.unlocked(token):
            orchestrate("test", out_dir, token.run_id, token, 10, True, SEEDS)
        cons = compare_attempt1(out_dir)
        dump(os.path.join(out_dir, "attempt1_consistency.json"), cons)
        say("seeds 0-3 identical to attempt 1:", cons["all_identical"])
        say("test run complete:", out_dir)
        return 0
    if not a.out:
        ap.error("--mode validation needs --out")
    out_dir = os.path.abspath(a.out)
    missing = a.missing_check
    orchestrate("validation", out_dir, None, None, 1, missing, (0,) if missing else ())
    chk = validation_checks(out_dir, missing)
    dump(os.path.join(out_dir, "validation_checks.json"), chk)
    say("validation dry run checks:", "all identical" if chk["all_identical"] else json.dumps(chk, indent=1))
    return 0 if chk["all_identical"] else 1


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    sys.exit(main())
