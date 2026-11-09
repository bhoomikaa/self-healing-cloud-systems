"""Post-test analysis tables (no torch, no test-window access): failure taxonomy, ranking diagnostics, detection gap, calibration pools.

Inputs: results/test_v1/ (saved predictions and results) and the reconstruction output of reconstruct.py (per-incident detail that
was checked to reproduce the saved detection rows exactly).

    .venv/bin/python -I analyze.py RECON_DIR OUT_JSON
"""
import csv
import json
import os
import sys
from collections import defaultdict

REPO = os.path.expanduser("~/298A_project/team-repo")
sys.path[:0] = [os.path.join(REPO, "armor-system", p) for p in ("data", "models")]

import numpy as np  # noqa: E402

TEST = os.path.join(REPO, "armor-system", "models", "model1_transformer", "results", "test_v1")
SEEDS = (0, 1, 2, 3, 4)
FAMILY = {0: "container", 1: "container", 2: "node", 3: "node", 4: "node"}
VM_HIDING = ("drop_metric", "drop_log+metric", "drop_metric+trace")
LATE_S = 120


def read_csv(path):
    with open(path, newline="") as fh:
        return {int(r["case_index"]): (int(r["true_class"]), int(r["predicted_class"]), r["true_name"], r["predicted_name"]) for r in csv.DictReader(fh)}


def seeds_word(n, total=5):
    return "all 5" if n == total else ("one" if n == 1 else f"some ({n}/{total})")


def mean_sd(v):
    v = np.asarray(v, np.float64)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "values": [round(float(x), 4) for x in v]}


def main():
    from prep import loader
    recon_dir, out_path = sys.argv[1], sys.argv[2]
    meta = json.load(open(os.path.join(recon_dir, "meta.json")))
    rec = {s: json.load(open(os.path.join(recon_dir, f"recon_seed_{s}.json"))) for s in SEEDS}
    labels_csv = loader.read_labels()["cases"]
    case_ts = meta["cases"]["timestamp"]
    assert [c["timestamp"] for c in labels_csv] == case_ts, "case order differs between cases.csv and the case table"
    names = {}
    out = {"reproduction": {s: {"rows": all(r["reproduces_saved"].values()), "thresholds": all(r["threshold_matches_saved"].values()),
                                "missing": all(all(v["reproduces_saved"].values()) for v in r["missing"].values())} for s, r in rec.items()}}

    # ------------------------------------------------------------------ fault type
    preds = {"v1": {s: read_csv(os.path.join(TEST, f"seed_{s}", "predictions_v1.csv")) for s in SEEDS},
             "v0": {s: read_csv(os.path.join(TEST, f"seed_{s}", "predictions_v0.csv")) for s in SEEDS},
             "xgboost": {s: read_csv(os.path.join(TEST, "xgboost", f"seed_{s}_predictions.csv")) for s in SEEDS},
             "ensemble": {0: read_csv(os.path.join(TEST, "ensemble", "predictions_v1_ensemble.csv"))}}
    for p in preds["v1"][0].values():
        names[p[0]] = p[2]
    cases = sorted(preds["v1"][0])
    fault = {}
    for model, per in preds.items():
        obs = {"across": defaultdict(list), "within": defaultdict(list)}
        for s, rows in per.items():
            for c, (t, p, _, _) in rows.items():
                if t != p:
                    obs["across" if FAMILY[t] != FAMILY[p] else "within"][c].append({"seed": s, "true": names[t], "pred": names[p]})
        fault[model] = {}
        for kind, d in obs.items():
            n_obs = sum(len(v) for v in d.values())
            fault[model][kind] = {"units": len(d), "observations": n_obs, "unit_pct_of_84_cases": 100 * len(d) / 84,
                                  "obs_pct_of_case_predictions": 100 * n_obs / (84 * len(per)),
                                  "seed_pattern": {"all": sum(len(v) == len(per) for v in d.values()), "some": sum(1 < len(v) < len(per) for v in d.values()), "one": sum(len(v) == 1 for v in d.values())} if len(per) > 1 else None,
                                  "cases": {str(c): {"n_seeds": len(v), "true": v[0]["true"], "preds": sorted({x["pred"] for x in v}), "root_cause": labels_csv[c]["cmdb_id"], "level": labels_csv[c]["level"]} for c, v in sorted(d.items(), key=lambda kv: -len(kv[1]))}}
        # confusion pairs (observations)
        pairs = defaultdict(int)
        for s, rows in per.items():
            for c, (t, p, _, _) in rows.items():
                if t != p:
                    pairs[f"{names[t]} -> {names[p]}"] += 1
        fault[model]["confusion_pairs"] = dict(sorted(pairs.items(), key=lambda kv: -kv[1]))
    # cases wrong for every model in every seed
    allwrong = [c for c in cases if all(rows[c][0] != rows[c][1] for per in preds.values() for rows in per.values())]
    fault["wrong_in_every_model_and_seed"] = [{"case": c, "true": names[preds["v1"][0][c][0]], "root_cause": labels_csv[c]["cmdb_id"]} for c in allwrong]
    out["fault"] = fault

    # ------------------------------------------------------------------ detection, our labels, v1 primary (+ D3 rows)
    ts2case = defaultdict(list)
    for c in meta["test_cases"]:
        ts2case[case_ts[c]].append(c)
    true_cls = {c: preds["v1"][0][c][0] for c in cases}

    def case_info(t):
        cs = ts2case.get(int(t), [])
        return [{"case": c, "class": names[true_cls[c]], "root_cause": labels_csv[c]["cmdb_id"], "level": labels_csv[c]["level"]} for c in cs]

    det = {}
    for row in ("primary", "d3_m2_p90", "d3_m2_p99"):
        missed, late, delays = defaultdict(list), defaultdict(list), defaultdict(list)
        fp = []
        for s in SEEDS:
            d = rec[s]["detail"][f"ours/{row}"]
            for t, v in d["labels"].items():
                if not v["hit"]:
                    missed[t].append(s)
                else:
                    delays[t].append(v["delay_s"])
                    if v["delay_s"] > LATE_S:
                        late[t].append(s)
            fp += [dict(iv, seed=s) for iv in d["intervals"] if iv["class"] != "hit"]
        # FP episodes: overlapping FP intervals (any seed) merged into one unit
        fp.sort(key=lambda x: x["start"])
        episodes = []
        for iv in fp:
            if episodes and iv["start"] <= episodes[-1]["end"]:
                e = episodes[-1]
                e["end"] = max(e["end"], iv["end"])
                e["members"].append(iv)
            else:
                episodes.append({"start": iv["start"], "end": iv["end"], "members": [iv]})
        for e in episodes:
            cl = {m["class"] for m in e["members"]}
            e["class"] = "normal_only" if cl == {"normal_only"} else ("ignore_overlap" if cl == {"ignore_overlap"} else "mixed")
            e["seeds"] = sorted({m["seed"] for m in e["members"]})
            e["minutes"] = {k: sum(m["minutes"][k] for m in e["members"]) for k in ("event_associated", "normal_reference", "ignore")}
        by_cls = defaultdict(int)
        for m in fp:
            by_cls[m["class"]] += 1
        miss_by_class = defaultdict(lambda: {"units": 0, "obs": 0})
        for t, ss in missed.items():
            for ci in case_info(t):
                miss_by_class[ci["class"]]["units"] += 1
                miss_by_class[ci["class"]]["obs"] += len(ss)
        late_by_class = defaultdict(int)
        for t in late:
            for ci in case_info(t):
                late_by_class[ci["class"]] += 1
        det[row] = {
            "missed": {"units": len(missed), "observations": sum(map(len, missed.values())), "unit_pct_of_84": 100 * len(missed) / 84, "obs_pct_of_420": 100 * sum(map(len, missed.values())) / 420,
                       "seed_pattern": {"all": sum(len(v) == 5 for v in missed.values()), "some": sum(1 < len(v) < 5 for v in missed.values()), "one": sum(len(v) == 1 for v in missed.values())},
                       "by_class_units": dict(miss_by_class), "labels": {str(t): {"seeds": v, "cases": case_info(t)} for t, v in sorted(missed.items(), key=lambda kv: -len(kv[1]))}},
            "late": {"units": len(late), "observations": sum(map(len, late.values())), "hits_total_obs": sum(map(len, delays.values())),
                     "seed_pattern": {"all": sum(len(v) == 5 for v in late.values()), "some": sum(1 < len(v) < 5 for v in late.values()), "one": sum(len(v) == 1 for v in late.values())},
                     "by_class_units": dict(late_by_class),
                     "labels": {str(t): {"seeds": v, "delays_s": delays[t], "cases": case_info(t)} for t, v in sorted(late.items(), key=lambda kv: -len(kv[1]))}},
            "fp": {"interval_observations": len(fp), "by_interval_class": dict(by_cls), "episodes": len(episodes),
                   "episodes_by_class": {k: sum(e["class"] == k for e in episodes) for k in ("normal_only", "ignore_overlap", "mixed")},
                   "episode_seed_pattern": {"all": sum(len(e["seeds"]) == 5 for e in episodes), "some": sum(1 < len(e["seeds"]) < 5 for e in episodes), "one": sum(len(e["seeds"]) == 1 for e in episodes)},
                   "episodes_list": [{"start": e["start"], "end": e["end"], "class": e["class"], "seeds": e["seeds"], "n_intervals": len(e["members"]), "minutes": e["minutes"]} for e in episodes]}}
    out["detection_ours"] = det

    # ------------------------------------------------------------------ VM-hiding-only failures (v1 primary, recalibrated, our labels)
    vm = defaultdict(list)
    for s in SEEDS:
        clean = rec[s]["detail"]["ours/primary"]["labels"]
        miss_non_vm = set()
        for name, sc in rec[s]["missing"].items():
            if not sc["hides_all_vms"]:
                miss_non_vm |= {t for t, v in sc["detail"]["ours/v1_primary"]["labels"].items() if not v["hit"]}
        for name in VM_HIDING:
            sc = rec[s]["missing"][name]
            assert sc["hides_all_vms"]
            for t, v in sc["detail"]["ours/v1_primary"]["labels"].items():
                if clean[t]["hit"] and not v["hit"] and t not in miss_non_vm:
                    vm[t].append({"seed": s, "scenario": name})
    vm_cls = defaultdict(int)
    for t in vm:
        for ci in case_info(t):
            vm_cls[ci["class"] + " / " + ci["level"]] += 1
    out["vm_hiding_only"] = {"units": len(vm), "observations": sum(map(len, vm.values())),
                             "seed_pattern": {"all": sum(len({x["seed"] for x in v}) == 5 for v in vm.values()), "some": sum(1 < len({x["seed"] for x in v}) < 5 for v in vm.values()), "one": sum(len({x["seed"] for x in v}) == 1 for v in vm.values())},
                             "by_class_level_units": dict(vm_cls),
                             "labels": {str(t): {"n_obs": len(v), "seeds": sorted({x["seed"] for x in v}), "scenarios": sorted({x["scenario"] for x in v}), "cases": case_info(t)} for t, v in sorted(vm.items(), key=lambda kv: -len(kv[1]))}}

    # ------------------------------------------------------------------ ranking diagnostics
    rk = {}
    for period in ("validation", "test"):
        rk[period] = {}
        for score in ("v1_logit", "v1_logit_rolled", "d3_m2"):
            rk[period][score] = {m: mean_sd([rec[s]["ranking"][period][score][m] for s in SEEDS]) for m in ("auc_pr", "auc_roc")}
            r0 = rec[0]["ranking"][period][score]
            rk[period][score].update({"n_event": r0["n_event"], "n_normal": r0["n_normal"], "prevalence": r0["prevalence"], "n_nonfinite": [rec[s]["ranking"][period][score]["n_nonfinite"] for s in SEEDS]})
    out["ranking"] = rk

    # ------------------------------------------------------------------ detection gap vs ARMOR's released intervals (official labels)
    armor_hit = {int(k): v for k, v in meta["armor_hits_official"].items()}
    all_ts = np.array(case_ts)

    def official_info(t):
        d = np.abs(all_ts - t)
        c = int(d.argmin())
        exact = [i for i in range(len(case_ts)) if case_ts[i] == t]
        if exact:
            return {"cases": exact, "class": names.get(meta["cases"]["class_id"][exact[0]], str(meta["cases"]["class_id"][exact[0]])), "nearest_s": 0, "root_cause": labels_csv[exact[0]]["cmdb_id"]}
        return {"cases": [], "class": "no case record", "nearest_s": int(d[c]), "nearest_case_class": names.get(meta["cases"]["class_id"][c], "?")}
    gap = {}
    for d3row in ("d3_m2_p99", "d3_m2_p90"):
        miss_both = defaultdict(list)
        for s in SEEDS:
            prim = rec[s]["detail"]["official/primary"]["labels"]
            d3 = rec[s]["detail"][f"official/{d3row}"]["labels"]
            for t in prim:
                if armor_hit[int(t)] and not prim[t]["hit"] and not d3[t]["hit"]:
                    miss_both[int(t)].append(s)
        byc_any, byc_all = defaultdict(int), defaultdict(int)
        for t, ss in miss_both.items():
            cl = official_info(t)["class"]
            byc_any[cl] += 1
            if len(ss) == 5:
                byc_all[cl] += 1
        gap[f"primary_and_{d3row}"] = {"armor_hits": sum(armor_hit.values()), "n_official": len(armor_hit), "missed_by_both_in_any_seed": len(miss_both),
                                       "missed_by_both_in_all_5_seeds": sum(len(v) == 5 for v in miss_both.values()),
                                       "observations": sum(map(len, miss_both.values())),
                                       "by_class_any_seed": dict(byc_any), "by_class_all_seeds": dict(byc_all),
                                       "labels": {str(t): {"seeds": v, **official_info(t)} for t, v in sorted(miss_both.items())}}
    off_cls = defaultdict(int)
    for t in armor_hit:
        off_cls[official_info(t)["class"]] += 1
    gap["official_labels_by_class"] = dict(off_cls)
    out["detection_gap"] = gap

    # ------------------------------------------------------------------ calibration pools
    from prep import paths
    z = np.load(paths.CACHE_FILE)
    ts, is_test = z["timestamps"], z["is_test_file"]
    split = json.load(open(os.path.join(REPO, "armor-system", "data", "artifacts", "d1", "splits", "official_armor_v1.json")))["ad_split"]["timestamp"]
    n329 = 329
    pos = {p: (n329 - 1) * p / 100 for p in (90.0, 95.0, 97.0, 99.0, 99.5)}
    out["calibration"] = {"model1_supervised_windows": n329, "model1_d3_windows": n329,
                          "order_statistics": {str(p): {"position": v, "upper_rank_from_top": n329 - int(np.ceil(v)), "windows_above_or_at_upper": n329 - int(np.floor(v))} for p, v in pos.items()},
                          "armor_init_samples_train_file_le_split": int(((~is_test) & (ts <= split)).sum()), "armor_split_ts": split,
                          "all_samples_le_split": int((ts <= split).sum())}
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"reproduction": out["reproduction"]}, indent=0))


if __name__ == "__main__":
    main()
