"""Build the training record for Model 1 v1 from the frozen checkpoints.

The demo asks for learning curves. There are none: `history` is an empty list in
all five checkpoints, so per-epoch losses were never written down and cannot be
recovered without retraining, which would break the freeze the test results rest
on. Saying that plainly is better than drawing a curve from nothing.

What the checkpoints do record is the selection point of each run: which epoch
was kept, by what criterion, and the validation metrics at that epoch. That is
enough to show the five seeds agree on where training stopped being useful, and
to show the spread between them, which is the thing a learning curve would have
been used to argue anyway.

    python -I make_training_record.py [--ckpt-root DIR] [--out DIR]

Writes training_record.json and figures/training_record.png.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.environ.get("MODEL1_CHECKPOINT_ROOT",
                              os.path.expanduser("~/298A_project/checkpoints"))
SEEDS = range(5)


def load(ckpt_root):
    rows = []
    for s in SEEDS:
        p = os.path.join(ckpt_root, "model1_v1", f"seed_{s}", "s2_best.json")
        if not os.path.isfile(p):
            sys.exit(f"missing checkpoint metadata: {p}")
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        e = d.get("extra", {})
        rows.append({
            "seed": s,
            "stage1_selected_epoch": e.get("s1_selected_epoch"),
            "stage2_selected_epoch": d.get("epoch"),
            "selected_by": e.get("selected_by"),
            "val_monitor": e.get("val_monitor"),
            "val_balanced_bce": e.get("val_balanced_bce"),
            "val_case_nll": e.get("val_case_nll"),
            "parameters": d.get("parameters"),
            "history_entries": len(d.get("history", []) or []),
        })
    return rows


def figure(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = [r["seed"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))

    ax[0].bar([s - 0.2 for s in seeds], [r["stage1_selected_epoch"] for r in rows],
              width=0.4, label="Stage 1")
    ax[0].bar([s + 0.2 for s in seeds], [r["stage2_selected_epoch"] for r in rows],
              width=0.4, label="Stage 2")
    ax[0].set_title("Selected epoch per seed")
    ax[0].set_xlabel("seed"); ax[0].set_ylabel("epoch"); ax[0].legend(fontsize=8)
    ax[0].set_xticks(seeds)

    for i, (key, title) in enumerate((("val_balanced_bce", "Validation balanced BCE"),
                                      ("val_case_nll", "Validation case-level fault NLL")), start=1):
        vals = [r[key] for r in rows]
        ax[i].bar(seeds, vals, color="tab:grey")
        mean = sum(vals) / len(vals)
        ax[i].axhline(mean, ls="--", lw=1, color="tab:red",
                      label=f"mean {mean:.3f}")
        ax[i].set_title(title + "\nat the selected epoch")
        ax[i].set_xlabel("seed"); ax[i].set_xticks(seeds); ax[i].legend(fontsize=8)

    fig.suptitle("Model 1 v1 training record — selection points, not learning curves",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args(argv)

    rows = load(a.ckpt_root)
    recorded = sum(r["history_entries"] for r in rows)

    record = {
        "what": "Model 1 v1 training record, read from the five frozen Stage 2 checkpoints.",
        "learning_curves": {
            "available": recorded > 0,
            "history_entries_found": recorded,
            "note": ("Per-epoch history was not written to the checkpoints, so learning curves "
                     "cannot be produced from the frozen artifacts. Recovering them means "
                     "retraining, which would break the freeze the reported test results rest "
                     "on. The selection points below are what survives."),
        },
        "architecture": {"parameters": rows[0]["parameters"]},
        "selection_criterion": rows[0]["selected_by"],
        "per_seed": rows,
        "summary": {
            k: {"min": min(r[k] for r in rows), "max": max(r[k] for r in rows),
                "mean": round(sum(r[k] for r in rows) / len(rows), 6)}
            for k in ("stage1_selected_epoch", "stage2_selected_epoch",
                      "val_balanced_bce", "val_case_nll")
        },
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

    png = os.path.join(a.out, "figures", "training_record.png")
    figure(rows, png)

    print(f"learning-curve history entries found: {recorded}")
    for r in rows:
        print(f"  seed {r['seed']}: stage1 epoch {r['stage1_selected_epoch']:>3}, "
              f"stage2 epoch {r['stage2_selected_epoch']:>3}, "
              f"BCE {r['val_balanced_bce']:.4f}, case NLL {r['val_case_nll']:.4f}")
    s = record["summary"]
    print(f"stage 2 epoch range: {s['stage2_selected_epoch']['min']}-{s['stage2_selected_epoch']['max']}")
    print(f"wrote {jpath}")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
