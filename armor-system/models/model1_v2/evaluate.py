"""v2 evaluation (V2_ENSEMBLE_PROTOCOL.md sections 3-6): frozen-v1 validation reproduction, CV ensembles, acceptance, ablations,
figures and the evidence file. Reads validation, calibration and (for AB2 only) fit windows; never a test-period window.

    .venv/bin/python -I armor-system/models/model1_v2/evaluate.py validation      # frozen checkpoints: reproduction + AB1/AB2 (torch)
    .venv/bin/python -I armor-system/models/model1_v2/evaluate.py cv              # CV ensembles, E1-E3, AB1/AB3, tables (no training)
    .venv/bin/python -I armor-system/models/model1_v2/evaluate.py figures
    .venv/bin/python -I armor-system/models/model1_v2/evaluate.py evidence
"""
import contextlib
import csv
import itertools
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "armor-system", p) for p in ("data", "models", os.path.join("evaluation", "detection"))]

import numpy as np  # noqa: E402

from model1_v2 import folds, train_cv  # noqa: E402

RES = os.path.join(HERE, "results")
V1 = os.path.join(REPO, "armor-system", "models", "model1_transformer")
ENS_VAL = os.path.join(V1, "results", "ensemble_validation_v1")
SEEDS = (0, 1, 2, 3, 4)
SCRATCH = os.path.expanduser("~/298A_project/checkpoints/model1_v2_cv/validation_frozen_outputs.npz")
NAMES = ("Container Hardware", "Container Network", "Node CPU", "Node Disk", "Node Memory")


def jload(path):
    with open(path) as fh:
        return json.load(fh)


def dump(path, obj):
    train_cv.dump(path, obj)


def same(a, b):
    from model1_transformer import run_detection_v0 as rd
    return json.loads(json.dumps(rd._py(a), sort_keys=True)) == json.loads(json.dumps(rd._py(b), sort_keys=True))


def per_label(flags, ts, labels):
    import armor_scoring as sc
    import interval_diagnostics as idg
    from model1_transformer import window_errors as we
    iv = sc.get_pre_interval(we.alert_timestamps(ts, flags), 600)
    return {int(t): bool(v) for t, v in idg.label_matches(iv, labels, 300).items()}


def ens_detection(outs, cals, labels, with_hits=False):
    """V1_PROTOCOL 11.3 ensemble (ensemble.py) over the given per-seed outputs."""
    from model1_transformer import ensemble as en
    from model1_transformer.rolling import rolling_max_causal
    ev, cal = en.detection_scores(outs), en.detection_scores(cals)
    tau, info = en.threshold(cal)
    r = en.evaluate(ev, tau, labels)
    if with_hits:
        r["_hits"] = per_label(rolling_max_causal(ev["det_prob"], ev["target_ts"], ev["role"]) > tau, ev["target_ts"], labels)
    return r, info


def single_hits(out, cal, labels, seed):
    from model1_transformer import detection_v1 as dv
    from model1_transformer.rolling import rolling_max_causal
    th, _ = dv.supervised_thresholds(cal, seed, None, bootstrap=False)
    v = dv.sort_by_time(out)
    return per_label(rolling_max_causal(v["det_logit"], v["target_ts"], v["role"]) > th["primary_p90"], v["target_ts"], labels)


def agg_of(out):
    from model1_transformer import case_eval as ce, detection_v1 as dv
    ev = dv.event_windows(out)
    return ce.aggregate_cases(out["case_index"][ev], dv.softmax(out["fault_logits"][ev]), out["fault_class"][ev])


def fault_summary(agg):
    from model1_transformer import run_experiment as rx
    mt = rx._xgb_metrics()
    s = mt.summary(agg["true_class"], agg["pred_class"])
    tc, pc = agg["true_class"], agg["pred_class"]
    s["exact_correct"] = int((tc == pc).sum())
    s["node_exact_correct"], s["node_cases"] = int(((tc >= 2) & (tc == pc)).sum()), int((tc >= 2).sum())
    s["container_exact_correct"], s["container_cases"] = int(((tc < 2) & (tc == pc)).sum()), int((tc < 2).sum())
    return s


def brief(d):
    return {k: d[k] for k in ("precision", "recall", "f1", "TP", "FP", "FN", "armor_false_positive_intervals", "n_intervals")} | {"normal_alarm_rate": d["normal_reference"]["alarm_rate"]}


def subsets_table(n_seeds, fn):
    """AB1: for k = 1..n, mean/SD over all C(n, k) seed subsets of fn(subset) -> (det F1, macro-F1)."""
    out = {}
    for k in range(1, n_seeds + 1):
        vals = [fn(list(c)) for c in itertools.combinations(range(n_seeds), k)]
        f1, mf = np.array([v[0] for v in vals]), np.array([v[1] for v in vals])
        out[str(k)] = {"n_subsets": len(vals), "det_f1_mean": float(f1.mean()), "det_f1_sd": float(f1.std(ddof=1)) if len(vals) > 1 else 0.0,
                       "det_f1_min": float(f1.min()), "det_f1_max": float(f1.max()), "macro_f1_mean": float(mf.mean()), "macro_f1_sd": float(mf.std(ddof=1)) if len(vals) > 1 else 0.0}
    return out


@contextlib.contextmanager
def fit_scorable():
    """AB2 only: lets the frozen collector score fitting windows (in-sample pool). Restored afterwards. Never the test role."""
    from model1_transformer import window_errors as we
    saved = we.SCORABLE_ROLES
    we.SCORABLE_ROLES = tuple(saved) + ("fit",)
    try:
        yield
    finally:
        we.SCORABLE_ROLES = saved


# ------------------------------------------------------------------------------------------------ validation (frozen v1 checkpoints)
def validation():
    import torch
    from model1_transformer import detection_v1 as dv, ensemble as en, run_experiment as rx, run_test as rt, scoring_sets as ss, training as T
    from model1_transformer.spec import AdapterSpec
    from prep import m1_dataset as md
    torch.set_num_threads(1)
    D = md.Model1Data.load()
    cfg = T.load_train_config()
    spec = AdapterSpec.from_d1(D.schema, D.applicable)
    sset = ss.build(D, "validation")
    labels = sset["labels"]
    mt = rx._xgb_metrics()
    evs, cals, fits, aggs, singles = [], [], [], [], []
    fit_rows = md.objective_rows(D, "s1_pretrain")
    for s in SEEDS:
        models, binding = rt.load_models(D, cfg, spec, s, ("v1_s2",))
        m = models["v1_s2"]
        ev, cal = dv.sort_by_time(rt.collect(m, D, sset, "validation")), dv.sort_by_time(rt.collect_cal(m, D))
        with fit_scorable():
            fit = dv.sort_by_time(dv.collect_outputs(m, md.evaluation_loader(md.Model1WindowDataset(D, fit_rows, "ab2_fit_normal")), ("fit",)))
        agg, fault, _ = rt.fault_eval(ev, D, sset, mt)
        th, _ = dv.supervised_thresholds(cal, s, None, bootstrap=False)
        det = dv.supervised_validation(ev, th, labels)["primary"]
        singles.append({"seed": s, "binding": binding["v1_s2"], "fault": fault_summary(agg), "detection_primary": brief(det),
                        "pred": agg["pred_class"].tolist(), "hits": {str(k): v for k, v in single_hits(ev, cal, labels, s).items()}})
        evs.append(ev), cals.append(cal), fits.append(fit), aggs.append(agg)
    v1_f1 = jload(os.path.join(V1, "results", "experiment_v1", "validation_summary.json"))["detection"]["primary"]["f1"]["values"]
    if [x["detection_primary"]["f1"] for x in singles] != v1_f1:
        raise RuntimeError("single-seed validation detection differs from v1's committed validation_summary.json")
    committed = jload(os.path.join(ENS_VAL, "ensemble_validation.json"))["ensemble"]
    ens_agg = en.fault(aggs)
    tmp = os.path.join(RES, "validation", "predictions_v2_ensemble.csv")
    rt.write_predictions(tmp, ens_agg, D)
    with open(tmp, "rb") as a, open(os.path.join(ENS_VAL, "predictions_v1_ensemble.csv"), "rb") as b:
        pred_identical = a.read() == b.read()
    det, cinfo = ens_detection(evs, cals, labels, with_hits=True)
    hits = det.pop("_hits")
    fault = rt.fault_metrics_from_agg(ens_agg, mt)
    repro = {"predictions_byte_identical": pred_identical, "detection_equal": same(det, committed["detection"]["ours"]), "fault_equal": same(fault, committed["fault"]),
             "threshold_equal": cinfo["n_windows"] == committed["calibration"]["n_windows"] and en.threshold(en.detection_scores(cals))[0] == committed["calibration"]["threshold"]}
    if not all(repro.values()):
        raise RuntimeError(f"frozen-v1 ensemble does not reproduce the committed validation ensemble: {repro}")
    # AB2: in-sample pool = 329 calibration + 1,312 fit normal windows
    from model1_transformer.rolling import rolling_max_causal
    ce_, fe_ = en.detection_scores(cals), en.detection_scores(fits)
    pool = {k: np.concatenate([ce_[k], fe_[k]]) for k in ("det_prob", "target_ts", "role")}
    tau2 = float(np.percentile(rolling_max_causal(pool["det_prob"], pool["target_ts"], pool["role"]), 90.0, method="linear"))
    ab2 = {"pool_windows": int(len(pool["det_prob"])), "threshold": tau2, "threshold_calibration_only": en.threshold(ce_)[0],
           "detection": brief(en.evaluate(en.detection_scores(evs), tau2, labels))}
    # AB1
    def fn(idx):
        d, _ = ens_detection([evs[i] for i in idx], [cals[i] for i in idx], labels)
        return d["f1"], fault_summary(en.fault([aggs[i] for i in idx]))["macro_f1"]
    res = {"scope": "validation period, frozen v1 checkpoints (25 cases / 25 incidents); not deciding (seen when v2 was chosen)",
           "reproduction_of_committed_ensemble": repro, "single_seeds": singles,
           "single_seed_mean": {"det_f1": float(np.mean([x["detection_primary"]["f1"] for x in singles])), "det_f1_sd": float(np.std([x["detection_primary"]["f1"] for x in singles], ddof=1)),
                                "TP": float(np.mean([x["detection_primary"]["TP"] for x in singles])), "FP_intervals": float(np.mean([x["detection_primary"]["armor_false_positive_intervals"] for x in singles])),
                                "exact_correct": float(np.mean([x["fault"]["exact_correct"] for x in singles])), "macro_f1": float(np.mean([x["fault"]["macro_f1"] for x in singles]))},
           "ensemble": {"detection": brief(det), "calibration": cinfo, "fault": fault_summary(ens_agg), "pred": ens_agg["pred_class"].tolist(), "hits": {str(k): v for k, v in hits.items()}},
           "case_index": ens_agg["case_index"].tolist(), "true_class": ens_agg["true_class"].tolist(), "labels": [int(x) for x in labels],
           "ab1_ensemble_size": subsets_table(5, fn), "ab2_in_sample_pool": ab2}
    dump(os.path.join(RES, "validation", "validation_frozen.json"), res)
    print("validation reproduction:", repro, "| ensemble F1", det["f1"], "| single mean", res["single_seed_mean"]["det_f1"], "| AB2 F1", ab2["detection"]["f1"])
    return res


# ------------------------------------------------------------------------------------------------ CV
def load_fold(fold, ckpt_root=train_cv.CKPT_ROOT, out_dir=train_cv.OUT):
    outs, cals, results = [], [], []
    for s in SEEDS:
        z = np.load(os.path.join(ckpt_root, fold, f"seed_{s}", "outputs.npz"), allow_pickle=True)
        outs.append({k[5:]: z[k] for k in z.files if k.startswith("held_")})
        cals.append({k[4:]: z[k] for k in z.files if k.startswith("cal_")})
        results.append(jload(os.path.join(out_dir, fold, f"seed_{s}", "result.json")))
    return outs, cals, results


def cv():
    from model1_transformer import ensemble as en
    from prep import m1_dataset as md
    D = md.Model1Data.load()
    summary, tables = {}, {}
    for f in folds.FOLDS:
        outs, cals, results = load_fold(f)
        labels = sorted(int(D.cases["timestamp"][c]) for c in folds.EXPECTED_CASES[f])
        aggs = [agg_of(o) for o in outs]
        singles = []
        for s, (o, c, r) in enumerate(zip(outs, cals, results)):
            fs = fault_summary(aggs[s])
            if abs(fs["macro_f1"] - r["fault"]["macro_f1"]) > 1e-12:
                raise RuntimeError(f"{f} seed {s}: recomputed fault differs from the run's result.json")
            singles.append({"seed": s, "fault": fs, "detection_primary": brief(r["detection_primary"]), "hits": single_hits(o, c, labels, s), "pred": aggs[s]["pred_class"].tolist()})
        det, cinfo = ens_detection(outs, cals, labels, with_hits=True)
        hits = det.pop("_hits")
        ens_agg = en.fault(aggs)
        ens_fault = fault_summary(ens_agg)
        mean = lambda key, sub: float(np.mean([x[sub][key] for x in singles]))      # noqa: E731
        def fn(idx):
            d, _ = ens_detection([outs[i] for i in idx], [cals[i] for i in idx], labels)
            return d["f1"], fault_summary(en.fault([aggs[i] for i in idx]))["macro_f1"]
        summary[f] = {"held_out_cases": len(labels), "single_seeds": [{k: v for k, v in x.items() if k not in ("hits", "pred")} for x in singles],
                      "single_mean": {"det_f1": mean("f1", "detection_primary"), "det_f1_sd": float(np.std([x["detection_primary"]["f1"] for x in singles], ddof=1)),
                                      "TP": mean("TP", "detection_primary"), "FN": mean("FN", "detection_primary"), "FP_intervals": mean("armor_false_positive_intervals", "detection_primary"),
                                      "exact_correct": mean("exact_correct", "fault"), "node_exact_correct": mean("node_exact_correct", "fault"),
                                      "container_exact_correct": mean("container_exact_correct", "fault"), "macro_f1": mean("macro_f1", "fault"), "weighted_f1": mean("weighted_f1", "fault")},
                      "ensemble": {"detection": brief(det), "calibration": cinfo, "fault": ens_fault},
                      "ab1_ensemble_size": subsets_table(5, fn)}
        tables[f] = {"labels": labels, "single_hits": [x["hits"] for x in singles], "ens_hits": hits,
                     "case_index": ens_agg["case_index"].tolist(), "true": ens_agg["true_class"].tolist(), "single_pred": [x["pred"] for x in singles], "ens_pred": ens_agg["pred_class"].tolist()}
    dec = [summary[f] for f in ("F1", "F2")]
    e1 = all(x["ensemble"]["detection"]["f1"] > x["single_mean"]["det_f1"] for x in dec)
    miss_e, miss_s = sum(x["ensemble"]["detection"]["FN"] for x in dec), sum(x["single_mean"]["FN"] for x in dec)
    fp_e, fp_s = sum(x["ensemble"]["detection"]["armor_false_positive_intervals"] for x in dec), sum(x["single_mean"]["FP_intervals"] for x in dec)
    ok_e, ok_s = sum(x["ensemble"]["fault"]["exact_correct"] for x in dec), sum(x["single_mean"]["exact_correct"] for x in dec)
    acc = {"E1_ensemble_f1_above_single_mean_in_F1_and_F2": {"F1": [dec[0]["ensemble"]["detection"]["f1"], dec[0]["single_mean"]["det_f1"]],
                                                            "F2": [dec[1]["ensemble"]["detection"]["f1"], dec[1]["single_mean"]["det_f1"]], "pass": e1},
           "E2_missed_and_fp_guard": {"ensemble_missed": miss_e, "single_mean_missed": miss_s, "ensemble_fp_intervals": fp_e, "single_mean_fp_intervals": fp_s,
                                      "pass": miss_e <= miss_s and fp_e <= fp_s + 2},
           "E3_fault_guard": {"ensemble_exact_correct": ok_e, "single_mean_exact_correct": ok_s, "pass": ok_e >= ok_s - 1}}
    acc["accepted"] = all(v["pass"] for v in acc.values() if isinstance(v, dict))
    acc["decision"] = ("v2 = five-seed ensemble of the frozen v1 checkpoints" if acc["accepted"] else "v2 not accepted: v1 stays unchanged")
    # AB3: F3 fixed-epoch models vs frozen, validation-selected v1 on the same 25 cases
    val = jload(os.path.join(RES, "validation", "validation_frozen.json"))
    ab3 = {"F3_fixed_epochs_single_mean": summary["F3"]["single_mean"], "frozen_v1_validation_selected_single_mean": val["single_seed_mean"],
           "F3_fixed_epochs_ensemble": summary["F3"]["ensemble"]["detection"] | {"macro_f1": summary["F3"]["ensemble"]["fault"]["macro_f1"]},
           "frozen_v1_ensemble": val["ensemble"]["detection"] | {"macro_f1": val["ensemble"]["fault"]["macro_f1"]}}
    rep = jload(os.path.join(os.path.expanduser("~/298A_project/checkpoints/model1_v2_cv_repeat/results"), "F3", "seed_0", "result.json"))
    za = np.load(os.path.join(train_cv.CKPT_ROOT, "F3", "seed_0", "outputs.npz"), allow_pickle=True)
    zb = np.load(os.path.expanduser("~/298A_project/checkpoints/model1_v2_cv_repeat/F3/seed_0/outputs.npz"), allow_pickle=True)
    det_check = {"outputs_arrays_identical": all(np.array_equal(za[k], zb[k]) for k in za.files) and set(za.files) == set(zb.files),
                 "s2_state_sha256_identical": rep["checkpoints"]["s2_state_sha256"] == jload(os.path.join(train_cv.OUT, "F3", "seed_0", "result.json"))["checkpoints"]["s2_state_sha256"]}
    dump(os.path.join(RES, "cv_summary.json"), {"protocol": "V2_ENSEMBLE_PROTOCOL.md (commit 2ac3ab7)", "folds": summary, "acceptance": acc, "ab3": ab3, "determinism_F3_seed0": det_check})
    write_tables(tables, val)
    print(json.dumps(acc, indent=1), "\ndeterminism", det_check)
    return acc


def write_tables(tables, val):
    d = os.path.join(RES, "tables")
    os.makedirs(d, exist_ok=True)
    tables = dict(tables)
    tables["validation_frozen"] = {"labels": val["labels"], "single_hits": [{int(k): v for k, v in x["hits"].items()} for x in val["single_seeds"]],
                                   "ens_hits": {int(k): v for k, v in val["ensemble"]["hits"].items()}, "case_index": val["case_index"], "true": val["true_class"],
                                   "single_pred": [x["pred"] for x in val["single_seeds"]], "ens_pred": val["ensemble"]["pred"]}
    for name, t in tables.items():
        with open(os.path.join(d, f"{name}_incidents.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["label_ts"] + [f"hit_seed_{s}" for s in SEEDS] + ["single_hits_of_5", "hit_ensemble"])
            for t_ in t["labels"]:
                h = [int(x[int(t_)]) for x in t["single_hits"]]
                w.writerow([t_] + h + [sum(h), int(t["ens_hits"][int(t_)])])
        with open(os.path.join(d, f"{name}_cases.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["case_index", "true"] + [f"pred_seed_{s}" for s in SEEDS] + ["correct_of_5", "pred_ensemble", "ensemble_correct"])
            for i, c in enumerate(t["case_index"]):
                p = [t["single_pred"][s][i] for s in SEEDS]
                w.writerow([c, NAMES[t["true"][i]]] + [NAMES[x] for x in p] + [sum(x == t["true"][i] for x in p), NAMES[t["ens_pred"][i]], int(t["ens_pred"][i] == t["true"][i])])


# ------------------------------------------------------------------------------------------------ figures
SURFACE, INK, INK2, GRID, C1, C2 = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"


def _axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cvs, val = jload(os.path.join(RES, "cv_summary.json")), jload(os.path.join(RES, "validation", "validation_frozen.json"))
    d = os.path.join(RES, "figures")
    os.makedirs(d, exist_ok=True)
    groups = [("CV F1\n(day 1, 45)", cvs["folds"]["F1"]), ("CV F2\n(day 2, 43)", cvs["folds"]["F2"]), ("CV F3\n(validation, 25)", cvs["folds"]["F3"])]
    vsingle = [x["detection_primary"]["f1"] for x in val["single_seeds"]]

    # Figure 1: single-seed vs ensemble incident F1
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150, facecolor=SURFACE)
    _axes(ax)
    rows = [(n, [x["detection_primary"]["f1"] for x in g["single_seeds"]], g["ensemble"]["detection"]["f1"]) for n, g in groups]
    rows.append(("Validation, frozen v1\n(25, not deciding)", vsingle, val["ensemble"]["detection"]["f1"]))
    for i, (n, singles, ens) in enumerate(rows):
        ax.scatter([i - 0.12] * len(singles), singles, s=42, color=C1, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="single seed (5)" if i == 0 else None)
        ax.hlines(np.mean(singles), i - 0.26, i + 0.02, color=C1, linewidth=2, zorder=2, label="single-seed mean" if i == 0 else None)
        ax.scatter([i + 0.14], [ens], s=70, marker="D", color=C2, edgecolor=SURFACE, linewidth=1.5, zorder=4, label="v2 ensemble" if i == 0 else None)
        ax.annotate(f"{ens:.3f}", (i + 0.14, ens), xytext=(8, -3), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5, color=INK)
    ax.set_ylabel("Incident F1 (ARMOR rules)", color=INK2, fontsize=9)
    ax.set_title("Detection: single v1 seeds vs v2 five-seed ensemble", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig1_detection_f1_single_vs_ensemble.png"), facecolor=SURFACE)
    plt.close(fig)

    # Figure 2: ensemble size (AB1)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150, facecolor=SURFACE)
    _axes(ax)
    series = [("CV F1", cvs["folds"]["F1"]["ab1_ensemble_size"]), ("CV F2", cvs["folds"]["F2"]["ab1_ensemble_size"]), ("CV F3", cvs["folds"]["F3"]["ab1_ensemble_size"]),
              ("Validation (frozen)", val["ab1_ensemble_size"])]
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")
    for (n, t), c in zip(series, colors):
        k = [int(x) for x in t]
        m = [t[str(x)]["det_f1_mean"] for x in k]
        ax.plot(k, m, color=c, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, label=n)
        ax.annotate(n, (k[-1], m[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.8, 5.9)
    ax.set_xlabel("Seeds averaged (k); mean over all C(5, k) subsets", color=INK2, fontsize=9)
    ax.set_ylabel("Incident F1", color=INK2, fontsize=9)
    ax.set_title("Ablation AB1: detection F1 by ensemble size", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig2_ensemble_size.png"), facecolor=SURFACE)
    plt.close(fig)

    # Figure 3: fault type exact-correct cases, single-seed mean vs ensemble
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150, facecolor=SURFACE)
    _axes(ax)
    rows = [(n, g["single_mean"]["exact_correct"] / g["held_out_cases"], g["ensemble"]["fault"]["exact_correct"] / g["held_out_cases"]) for n, g in groups]
    rows.append(("Validation, frozen v1\n(25, not deciding)", val["single_seed_mean"]["exact_correct"] / 25, val["ensemble"]["fault"]["exact_correct"] / 25))
    x = np.arange(len(rows))
    ax.bar(x - 0.17, [r[1] for r in rows], width=0.32, color=C1, label="single-seed mean", zorder=3)
    ax.bar(x + 0.17, [r[2] for r in rows], width=0.32, color=C2, label="v2 ensemble", zorder=3)
    for i, r in enumerate(rows):
        for dx, v in ((-0.17, r[1]), (0.17, r[2])):
            ax.annotate(f"{v:.2f}", (i + dx, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Case accuracy (exact class)", color=INK2, fontsize=9)
    ax.set_title("Fault type: single v1 seeds vs v2 ensemble", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig3_fault_accuracy.png"), facecolor=SURFACE)
    plt.close(fig)
    print("figures written to", d)


# ------------------------------------------------------------------------------------------------ evidence
def evidence():
    import platform

    def sha(p):
        return train_cv.sha(p)

    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()

    code = {os.path.relpath(p, REPO): sha(p) for p in sorted(
        [os.path.join(HERE, f) for f in ("__init__.py", "folds.py", "train_cv.py", "evaluate.py")]
        + [os.path.join(REPO, "armor-system", "tests", "model1_v2", "test_v2_folds.py")]
        + [os.path.join(V1, f) for f in ("ensemble.py", "detection_v1.py", "training_v1.py", "rolling.py", "case_eval.py", "run_test.py")])}
    runs = {}
    for f in folds.FOLDS:
        for s in SEEDS:
            r = jload(os.path.join(train_cv.OUT, f, f"seed_{s}", "result.json"))
            runs[f"{f}/seed_{s}"] = {"checkpoints": r["checkpoints"], "outputs_npz_sha256": r["outputs_npz_sha256"], "predictions_csv_sha256": r["predictions_csv_sha256"],
                                     "epochs": r["epochs"], "windows": r["windows"], "elapsed_seconds": r["elapsed_seconds"]}
    cvs = jload(os.path.join(RES, "cv_summary.json"))
    val = jload(os.path.join(RES, "validation", "validation_frozen.json"))
    fr = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"), "-I", os.path.join(V1, "test_freeze.py"), "--verify"], capture_output=True, text=True).stdout.strip()
    import torch
    doc = {"evidence_name": "evidence_model1_v2", "linear_issue": "298-51", "protocol": {"path": "armor-system/models/model1_post_test_analysis/V2_ENSEMBLE_PROTOCOL.md", "commit": "2ac3ab7"},
           "head": git("rev-parse", "HEAD"), "code_sha256": code,
           "folds": {f: {"held_out_cases": folds.EXPECTED_CASES[f]} for f in folds.FOLDS} | {"day1_end": folds.DAY1_END, "day2_start": folds.DAY2_START, "fixed_epochs": {"s1": train_cv.S1_EPOCHS, "s2": train_cv.S2_EPOCHS}},
           "cv_runs": runs, "checkpoint_root": "~/298A_project/checkpoints/model1_v2_cv (outside the repository)",
           "determinism_F3_seed0": cvs["determinism_F3_seed0"], "frozen_v1_validation_reproduction": val["reproduction_of_committed_ensemble"],
           "frozen_v1_checkpoints": {f"seed_{x['seed']}": x["binding"] for x in val["single_seeds"]},
           "acceptance": cvs["acceptance"], "result_files_sha256": {os.path.relpath(p, REPO): sha(p) for p in sorted(
               [os.path.join(RES, "cv_summary.json"), os.path.join(RES, "validation", "validation_frozen.json"), os.path.join(RES, "validation", "predictions_v2_ensemble.csv")])},
           "freeze_verification": fr, "test_data_accessed": False,
           "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(), "torch_num_threads": 1}}
    dump(os.path.join(RES, "evidence_model1_v2.json"), doc)
    print("evidence written; freeze:", fr)


if __name__ == "__main__":
    {"validation": validation, "cv": cv, "figures": figures, "evidence": evidence}[sys.argv[1]]()
