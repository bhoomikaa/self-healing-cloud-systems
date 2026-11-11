"""Build the Model 1 v1 training record and learning curves.

Reads the per-epoch histories written during training,
`model1_transformer/results/experiment_v1/seed_*/histories.json`, which hold
both stages for all five seeds: loss, validation metrics, learning rate and
gradient-norm statistics per epoch.

This lives outside `model1_transformer/` on purpose. That tree is frozen: its
tracked file set is hashed into `results/test_freeze/evidence_test_freeze.json`,
so adding anything under it, even a plot, changes the set hash and fails the
freeze check that the one-shot test results rest on. Reading from the frozen
tree is fine; writing into it is not.

    python -I make_training_record.py [--out DIR]

Writes training_record.json and figures/learning_curves.png.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.abspath(os.path.join(HERE, ".."))
HIST_DIR = os.path.join(MODELS, "model1_transformer", "results", "experiment_v1")
CKPT_ROOT = os.environ.get("MODEL1_CHECKPOINT_ROOT",
                           os.path.expanduser("~/298A_project/checkpoints"))
SEEDS = range(5)

# The quantity each stage was selected on. Stage 1 pretrains the next-step head
# on masked MSE; Stage 2 trains detection and fault classification together, and
# is selected on the combined validation monitor.
STAGE_METRICS = {
    "s1": [("loss", "train loss"), ("val_masked_mse", "validation masked MSE")],
    "s2": [("loss", "train loss"), ("val_monitor", "validation monitor"),
           ("val_balanced_bce", "validation balanced BCE"),
           ("val_case_nll", "validation case NLL")],
}


def load_histories():
    out = {}
    for s in SEEDS:
        p = os.path.join(HIST_DIR, f"seed_{s}", "histories.json")
        if not os.path.isfile(p):
            sys.exit(f"missing history: {p}")
        with open(p, encoding="utf-8") as fh:
            out[s] = json.load(fh)
    return out


def load_selection():
    rows = []
    for s in SEEDS:
        p = os.path.join(CKPT_ROOT, "model1_v1", f"seed_{s}", "s2_best.json")
        if not os.path.isfile(p):
            return None
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        e = d.get("extra", {})
        rows.append({"seed": s, "stage1_selected_epoch": e.get("s1_selected_epoch"),
                     "stage2_selected_epoch": d.get("epoch"),
                     "selected_by": e.get("selected_by"),
                     "val_monitor": e.get("val_monitor"),
                     "val_balanced_bce": e.get("val_balanced_bce"),
                     "val_case_nll": e.get("val_case_nll"),
                     "parameters": d.get("parameters")})
    return rows


def series(hist, stage, key):
    xs, ys = [], []
    for row in hist.get(stage, []):
        if key in row and row[key] is not None:
            xs.append(row["epoch"])
            ys.append(row[key])
    return xs, ys


def figure(hists, selection, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("s1", k, t) for k, t in STAGE_METRICS["s1"]] + \
             [("s2", k, t) for k, t in STAGE_METRICS["s2"]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.ravel()
    sel = {r["seed"]: r for r in (selection or [])}

    for ax, (stage, key, title) in zip(axes, panels):
        for s in SEEDS:
            xs, ys = series(hists[s], stage, key)
            if xs:
                ax.plot(xs, ys, lw=1.2, alpha=0.85, label=f"seed {s}")
        # Mark where each run was actually kept, so the curve and the frozen
        # checkpoint can be read against each other.
        if sel:
            ekey = "stage1_selected_epoch" if stage == "s1" else "stage2_selected_epoch"
            for s in SEEDS:
                e = sel.get(s, {}).get(ekey)
                if e:
                    ax.axvline(e, color=f"C{s}", ls=":", lw=0.8, alpha=0.6)
        ax.set_title(f"Stage {stage[-1]} — {title}", fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=8, ncol=2)

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle("Model 1 v1 learning curves — five seeds, both stages. "
                 "Dotted lines mark the selected epoch of each run.", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args(argv)

    hists = load_histories()
    selection = load_selection()

    per_seed = []
    for s in SEEDS:
        h = hists[s]
        row = {"seed": s, "stage1_epochs": len(h.get("s1", [])),
               "stage2_epochs": len(h.get("s2", []))}
        for stage in ("s1", "s2"):
            for key, _ in STAGE_METRICS[stage]:
                xs, ys = series(h, stage, key)
                if ys:
                    row[f"{stage}_{key}_first"] = round(ys[0], 6)
                    row[f"{stage}_{key}_last"] = round(ys[-1], 6)
                    row[f"{stage}_{key}_min"] = round(min(ys), 6)
        per_seed.append(row)

    record = {
        "what": "Model 1 v1 training record and learning curves.",
        "source": {
            "histories": "armor-system/models/model1_transformer/results/experiment_v1/seed_*/histories.json",
            "selection": "the five frozen Stage 2 checkpoints",
            "note": ("Written outside model1_transformer/ because that tree is frozen: its "
                     "tracked file set is hashed into evidence_test_freeze.json, so writing "
                     "anything there fails the freeze check."),
        },
        "learning_curves": {
            "available": True,
            "figure": "results/figures/learning_curves.png",
            "epochs_recorded": {f"seed_{s}": {"stage1": len(hists[s].get("s1", [])),
                                              "stage2": len(hists[s].get("s2", []))}
                                for s in SEEDS},
        },
        "per_seed": per_seed,
        "selection": selection,
        "compute": {
            "hardware": "Apple M4, CPU only, 1 torch thread, deterministic algorithms on",
            "python": "3.14.4", "torch": "2.14.0",
            "total_cpu_hours_recorded": 14.01,
            "source": "elapsed_seconds across 20 runs in models/model1_transformer/results/",
        },
    }

    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, "training_record.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
        fh.write("\n")

    png = os.path.join(a.out, "figures", "learning_curves.png")
    figure(hists, selection, png)

    for s in SEEDS:
        n1, n2 = len(hists[s].get("s1", [])), len(hists[s].get("s2", []))
        sel = next((r for r in (selection or []) if r["seed"] == s), {})
        print(f"  seed {s}: stage 1 {n1:>3} epochs (kept {sel.get('stage1_selected_epoch', '?')}), "
              f"stage 2 {n2:>3} epochs (kept {sel.get('stage2_selected_epoch', '?')})")
    print(f"wrote {jpath}")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
