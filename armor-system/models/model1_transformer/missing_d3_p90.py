"""D3 (v1 Stage 1 forecast error) missing-data experiments at the 90th-percentile operating point (Linear 298-51, validation only, no retraining).

    .venv/bin/python -I armor-system/models/model1_transformer/missing_d3_p90.py --seed N
    .venv/bin/python -I armor-system/models/model1_transformer/missing_d3_p90.py --aggregate

run_v1.py reports D3 under missing data with the v0 rule (M2 at the 99th percentile). The primary detector uses the 90th, so this reruns D3
on exactly the same masks (missing_eval: ARMOR scenarios, random rates 0..40%, 10 masks, the same Philox keys) with M2 at the 90th percentile
of the calibration scores, recalibrated on the masked calibration windows (primary) and with the clean thresholds (secondary). The v1 Stage 1
checkpoints are only loaded. Nothing in run_v1.py or its committed results changes. No test-period data, no official ARMOR file.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "armor-system", "data"), os.path.join(REPO, "armor-system", "models"), os.path.join(REPO, "armor-system", "evaluation", "detection")]

import torch  # noqa: E402

from model1_transformer import checkpoint as ck, detection_v1 as dv, missing_eval as me, run_detection_v0 as rd, run_v1 as r1, training as T, training_v1 as tv  # noqa: E402
from model1_transformer.sim_missing import MissingBlocks  # noqa: E402
from model1_transformer.spec import AdapterSpec  # noqa: E402
from prep import m1_dataset as md  # noqa: E402

RESULTS = r1.RESULTS
METRIC, PCT = "m2_max_normalised", 90.0
MODES = ("recalibrated", "clean_thresholds")
FIELDS = ("f1", "recall", "precision", "armor_false_positive_intervals", "coverage_padded_300s")


def load_stage1(D, cfg, spec, seed, ckpt_v1):
    prov = r1.provenance(D, spec, cfg)
    expect = {k: prov[k] for k in ("manifest_content_sha256", "fit_only_scaler_file_sha256", "spec_fingerprint", "train_config_sha256", "model")}
    m = tv.build_model_v1(spec, seed, torch.device("cpu"))
    ck.load_checkpoint(os.path.join(ckpt_v1, f"seed_{seed}", "s1_best"), m, expect=expect, stage="s1", seed=seed)
    return m


def outputs(model, view_D):
    return (dv.collect_outputs(model, r1.view(view_D, "s3_calibration_reference"), ("calibration",), "cpu"),
            dv.collect_outputs(model, r1.view(view_D, "validation_all"), ("validation",), "cpu"))


def score(outs, clean, labels):
    cal, val = outs
    stats, th, _ = dv.forecast_calibration(cal)
    p = str(PCT)
    return {"recalibrated": r1.brief(dv.forecast_validation(val, stats, {METRIC: {p: th[METRIC][p]}}, labels)[METRIC][p]),
            "clean_thresholds": r1.brief(dv.forecast_validation(val, clean["stats"], {METRIC: {p: clean["th"][METRIC][p]}}, labels)[METRIC][p])}


def run_seed(D, cfg, spec, seed, labels, ckpt_v1=r1.DEFAULT_CKPT, out_dir=RESULTS, log=r1.say, masks=me.N_MASKS):
    torch.set_num_threads(1)
    model = load_stage1(D, cfg, spec, seed, ckpt_v1)
    clean_outs = outputs(model, D)
    stats, th, _ = dv.forecast_calibration(clean_outs[0])
    clean = {"stats": stats, "th": th}
    blocks = MissingBlocks(spec)
    res = {"seed": seed, "model": "v1_d3", "metric": METRIC, "percentile": PCT, "armor": {}, "random": {}, "clean_reference": score(clean_outs, clean, labels)}
    for name, drop in me.armor_scenarios():
        obs = me.scenario_observed(D, drop)
        res["armor"][name] = {"dropped": list(drop), "hides_all_vms": me.hides_all_vms(D, spec, obs), "d3_p90": score(outputs(model, me.masked_view(D, obs)), clean, labels)}
    for ri, rate in enumerate(me.RATES):
        res["random"][str(rate)] = {}
        for mi in range(1 if rate == 0.0 else masks):
            res["random"][str(rate)][str(mi)] = score(outputs(model, me.masked_view(D, me.random_observed(D, blocks, ri, mi, rate))), clean, labels)
        log(f"[d3 p90 seed {seed}] rate {rate} done")
    with open(os.path.join(out_dir, f"missing_d3_p90_seed_{seed}.json"), "w") as fh:
        json.dump(rd._py(res), fh, indent=1, sort_keys=True)
        fh.write("\n")
    return res


def aggregate(root=RESULTS, seeds=T.SEEDS):
    per = {s: r1.jload(os.path.join(root, f"missing_d3_p90_seed_{s}.json")) for s in seeds}

    def stat(cells):
        return {f: rd.mean_sd([c[f] for c in cells]) for f in FIELDS} | {"n": len(cells), "n_no_score": int(sum(not c["scored"] for c in cells))}

    out = {"seeds": list(seeds), "model": "v1_d3", "metric": METRIC, "percentile": PCT, "armor_scenarios": {}, "random_rates": {},
           "clean_reference": {md_: stat([per[s]["clean_reference"][md_] for s in seeds]) for md_ in MODES}}
    for name in per[seeds[0]]["armor"]:
        out["armor_scenarios"][name] = {"dropped": per[seeds[0]]["armor"][name]["dropped"], "hides_all_vms": per[seeds[0]]["armor"][name]["hides_all_vms"],
                                        "d3_p90": {md_: stat([per[s]["armor"][name]["d3_p90"][md_] for s in seeds]) for md_ in MODES}}
    for rate in per[seeds[0]]["random"]:
        out["random_rates"][rate] = {"d3_p90": {md_: stat([c[md_] for s in seeds for c in per[s]["random"][rate].values()]) for md_ in MODES},
                                     "distinct_masks": len(per[seeds[0]]["random"][rate])}
    out["notes"] = ["Same masks, seeds and checkpoints as missing_summary.json; only the D3 operating point differs (M2 at the 90th percentile instead of the 99th).",
                    "Recalibrated = thresholds and fit statistics refitted on the masked calibration windows (primary); clean_thresholds = from the unmasked calibration windows.",
                    "Scenarios containing 'metric' hide all six VMs and are reported separately from the pod-only scenarios."]
    with open(os.path.join(root, "missing_d3_p90_summary.json"), "w") as fh:
        json.dump(rd._py(out), fh, indent=1, sort_keys=True)
        fh.write("\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--ckpt-dir", default=r1.DEFAULT_CKPT)
    a = ap.parse_args(argv)
    if a.aggregate:
        aggregate()
        return 0
    if a.seed is None:
        ap.error("give --seed N or --aggregate")
    if os.path.abspath(a.ckpt_dir).startswith(REPO):
        ap.error("checkpoints must be outside the repository")
    T.validate_seed(a.seed)
    D = md.Model1Data.load()
    cfg = T.load_train_config()
    spec = AdapterSpec.from_d1(D.schema, D.applicable)
    run_seed(D, cfg, spec, a.seed, rd.validation_labels(D), a.ckpt_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
