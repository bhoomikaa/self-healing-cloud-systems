"""Post-hoc reconstruction of per-window scores and per-incident detection outcomes for the frozen Model 1 test run.

Read-only: frozen checkpoints, the frozen scoring code (run_test / detection_v1 / run_detection_v0), the attempt-2 run's token
(resume_token re-verifies the freeze and both attempt-2 markers). Nothing is trained, no threshold is changed, nothing is written to
results/test_v1/. Every reconstructed row is compared with the saved test result (exact equality after the same JSON serialisation),
so the per-incident lists below are the decomposition of the frozen numbers, not new numbers.

    .venv/bin/python -I reconstruct.py OUT_DIR [SEEDS]
"""
import json
import os
import sys
import time

REPO = os.path.expanduser("~/298A_project/team-repo")
sys.path[:0] = [os.path.join(REPO, "armor-system", p) for p in ("data", "models", os.path.join("evaluation", "detection"))]

import numpy as np  # noqa: E402

from model1_transformer import run_test as rt, test_freeze as tf  # noqa: E402

RUN_ID = "dabf6b59c185451e9b3e2a2f930eba30"
TEST_DIR = tf.TEST_OUT
DELAY, IMPACT = 600, 300
M2 = "m2_max_normalised"


def ranks(x):
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    return ((cum - cnt + 1 + cum) / 2.0)[inv]


def auc_roc(score, y):
    y = np.asarray(y, bool)
    r = ranks(np.asarray(score, np.float64))
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auc_pr(score, y):
    """Average precision (step-wise, ties grouped at distinct thresholds; sklearn's definition)."""
    s, y = np.asarray(score, np.float64), np.asarray(y, np.float64)
    o = np.argsort(-s, kind="mergesort")
    s, y = s[o], y[o]
    idx = np.r_[np.where(np.diff(s))[0], len(y) - 1]
    tps = np.cumsum(y)[idx]
    prec, rec = tps / (idx + 1), tps / y.sum()
    return float(((rec - np.r_[0.0, rec[:-1]]) * prec).sum())


def ranking(score, cat):
    keep = (cat == "event_associated") | (cat == "normal_reference")
    y = cat[keep] == "event_associated"
    s = score[keep]
    return {"auc_roc": auc_roc(s, y), "auc_pr": auc_pr(s, y), "n_event": int(y.sum()), "n_normal": int((~y).sum()), "prevalence": float(y.mean()),
            "n_nonfinite": int((~np.isfinite(s)).sum())}


def per_label(flags, ts, cat, labels):
    import armor_scoring as sc
    import interval_diagnostics as idg
    from model1_transformer import window_errors as we
    iv = sc.get_pre_interval(we.alert_timestamps(ts, flags), DELAY)
    m = idg.label_matches(iv, labels, IMPACT)
    lab = {int(t): {"hit": bool(v), "delay_s": (int(min(s for s, _ in v) - t) if v else None)} for t, v in m.items()}
    cls = idg.classify_intervals(iv, labels, ts[cat == "ignore"], IMPACT)
    ivs = []
    for (s, e), c in zip(iv, cls["per_interval"]):
        inside = (ts >= s) & (ts <= e)
        ivs.append({"start": int(s), "end": int(e), "class": c, "minutes": {k: int((inside & (cat == k)).sum()) for k in ("event_associated", "normal_reference", "ignore")}})
    return {"labels": lab, "intervals": ivs}


def same(row, saved):
    from model1_transformer import run_detection_v0 as rd
    return json.loads(json.dumps(rd._py(row), sort_keys=True)) == saved


def main():
    from model1_transformer import detection_v1 as dv, missing_eval as me, official_scoring as osc, run_detection_v0 as rd, run_v1 as r1, scoring_sets as ss
    from model1_transformer.rolling import rolling_max_causal
    out_dir = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(rt.SEEDS)
    os.makedirs(out_dir, exist_ok=True)
    tok = tf.resume_token(RUN_ID, markers=rt.MARKERS)                     # freeze verifies, both attempt-2 markers name this run
    with tf.unlocked(tok):
        D, cfg, spec, sset_t, labels_t = rt.setup("test", tok)
        sset_v = ss.build(D, "validation")
        labels_v = {"ours": sset_v["labels"]}
        armor_iv = osc.released_intervals(tok)
        import interval_diagnostics as idg
        armor_hits = {int(t): bool(v) for t, v in idg.label_matches(armor_iv, labels_t["official"], IMPACT).items()}
        meta = {"run_id": RUN_ID, "freeze_record_sha256": tok.freeze_sha256, "labels": {k: [int(x) for x in v] for k, v in labels_t.items()},
                "validation_labels": [int(x) for x in labels_v["ours"]], "test_cases": [int(c) for c in sset_t["cases"]],
                "armor_released_intervals": [list(x) for x in armor_iv], "armor_hits_official": {str(k): v for k, v in armor_hits.items()},
                "cases": {k: [int(x) if not isinstance(x, str) else x for x in D.cases[k].tolist()] for k in ("timestamp", "class_id", "group_id", "case_index")}}
        with open(os.path.join(out_dir, "meta.json"), "w") as fh:
            json.dump(meta, fh)
        for seed in seeds:
            t0 = time.time()
            saved = rt.jload(os.path.join(TEST_DIR, f"seed_{seed}", "result.json"))
            saved_miss = rt.jload(os.path.join(TEST_DIR, "missing", f"seed_{seed}.json"))
            models, _ = rt.load_models(D, cfg, spec, seed, ("v1_s2", "v1_s1"))
            cal = {k: rt.collect_cal(m, D) for k, m in models.items()}
            ev = {"test": {k: dv.sort_by_time(rt.collect(m, D, sset_t, "test")) for k, m in models.items()},
                  "validation": {k: dv.sort_by_time(rt.collect(m, D, sset_v, "validation")) for k, m in models.items()}}
            sup_th, _ = dv.supervised_thresholds(cal["v1_s2"], seed, None, bootstrap=False)
            d3_stats, d3_th, _ = dv.forecast_calibration(cal["v1_s1"])
            thr_ok = {"primary_p90": sup_th["primary_p90"] == saved["calibration"]["supervised"]["thresholds"]["primary_p90"],
                      "d3_m2": d3_th[M2] == saved["calibration"]["d3"]["thresholds"][M2]}
            res = {"seed": seed, "thresholds": {"primary_p90": sup_th["primary_p90"], "d3_m2": d3_th[M2]}, "threshold_matches_saved": thr_ok,
                   "ranking": {}, "detail": {}, "reproduces_saved": {}, "missing": {}}
            arrays = {}
            for period, e in ev.items():
                s2, s1 = e["v1_s2"], e["v1_s1"]
                ts, cat = s2["target_ts"], s2["category"]
                if not np.array_equal(ts, s1["target_ts"]):
                    raise RuntimeError("S1/S2 window order differs")
                rolled = rolling_max_causal(s2["det_logit"], ts, s2["role"])
                m2 = dv.forecast_scores(s1["errors"], s1["node_valid"], d3_stats)[M2]
                res["ranking"][period] = {"v1_logit": ranking(s2["det_logit"], cat), "v1_logit_rolled": ranking(rolled, cat), "d3_m2": ranking(m2, cat)}
                arrays.update({f"{period}_ts": ts, f"{period}_cat": cat, f"{period}_case": s2["case_index"], f"{period}_logit": s2["det_logit"], f"{period}_rolled": rolled, f"{period}_m2": m2})
                if period != "test":
                    continue
                for lab, ls in labels_t.items():
                    rows = {"primary": (rolled > sup_th["primary_p90"], saved["detection"][lab]["supervised"]["primary"], sup_th["primary_p90"]),
                            "d3_m2_p90": (m2 > d3_th[M2]["90.0"], saved["detection"][lab]["d3_forecast_error_stage1"][M2]["90.0"], d3_th[M2]["90.0"]),
                            "d3_m2_p99": (m2 > d3_th[M2]["99.0"], saved["detection"][lab]["d3_forecast_error_stage1"][M2]["99.0"], d3_th[M2]["99.0"])}
                    for name, (flags, sv, tau) in rows.items():
                        res["reproduces_saved"][f"{lab}/{name}"] = same({"threshold": tau, **rd.evaluate_alerts(flags, ts, cat, ls)}, sv)
                        res["detail"][f"{lab}/{name}"] = per_label(flags, ts, cat, ls)
            np.savez_compressed(os.path.join(out_dir, f"scores_seed_{seed}.npz"), **arrays)
            # ARMOR whole-modality scenarios: recalibrated thresholds (the reported missing-data rows)
            for name, drop in me.armor_scenarios():
                if name == "full":
                    continue
                obs = me.scenario_observed(D, drop)
                view = me.masked_view(D, obs)
                c = {k: rt.collect_cal(m, view) for k, m in models.items()}
                e = {k: dv.sort_by_time(dv.collect_outputs(m, ss.loader(view, sset_t["detection_rows"], "scoring"), ("test",))) for k, m in models.items()}
                th = dv.supervised_thresholds(c["v1_s2"], seed, None, bootstrap=False)[0]
                st, d3t, _ = dv.forecast_calibration(c["v1_s1"])
                ts, cat = e["v1_s2"]["target_ts"], e["v1_s2"]["category"]
                rolled = rolling_max_causal(e["v1_s2"]["det_logit"], ts, e["v1_s2"]["role"])
                m2 = dv.forecast_scores(e["v1_s1"]["errors"], e["v1_s1"]["node_valid"], st)[M2]
                sc = {"hides_all_vms": me.hides_all_vms(D, spec, obs), "detail": {}, "reproduces_saved": {}}
                for lab, ls in labels_t.items():
                    sv = saved_miss["armor"][name]["models"][lab]
                    for row, flags, svr in (("v1_primary", rolled > th["primary_p90"], sv["v1_primary"]["recalibrated"]),
                                            ("d3_p90", m2 > d3t[M2]["90.0"], sv["d3_p90"]["recalibrated"])):
                        sc["reproduces_saved"][f"{lab}/{row}"] = same(r1.brief(rd.evaluate_alerts(flags, ts, cat, ls)), svr)
                        sc["detail"][f"{lab}/{row}"] = per_label(flags, ts, cat, ls)
                res["missing"][name] = sc
            with open(os.path.join(out_dir, f"recon_seed_{seed}.json"), "w") as fh:
                json.dump(res, fh)
            ok = all(res["reproduces_saved"].values()) and all(thr_ok.values()) and all(all(v["reproduces_saved"].values()) for v in res["missing"].values())
            print(f"seed {seed} done in {time.time() - t0:.0f}s; reproduces saved: {ok}", flush=True)


if __name__ == "__main__":
    main()
