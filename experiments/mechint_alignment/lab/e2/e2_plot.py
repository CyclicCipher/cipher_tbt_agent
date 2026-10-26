"""E2 learning curves: per rule, held-out accuracy against instances seen, one line per run; plus bits per transition
and changed-entry accuracy against transitions.   python -m lab.e2.e2_plot out.png runs/e2/A runs/e2/B ..."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from worlds.village_events import EVENTS, UNCHECKABLE

RULES = [e for e in EVENTS if e not in UNCHECKABLE and e not in ("villager_death",)]


def main(out, runs):
    data = {r: [json.loads(x) for x in open(f"{r}/log.jsonl")] for r in runs}
    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    axes = axes.ravel()
    for i, e in enumerate(RULES):
        ax = axes[i]
        for r, rs in data.items():
            xs = [x[e]["seen"] for x in rs if x[e]["seen"] > 0]
            ys = [x[e]["acc"] for x in rs if x[e]["seen"] > 0]
            ax.plot(xs, ys, label=r.split("/")[-1])
        ax.set(title=e, xscale="log", ylim=(-0.03, 1.03))
        ax.axhline(0.9, c="gray", lw=0.5, ls="--")
    ax = axes[len(RULES)]
    for r, rs in data.items():
        ax.plot([x["transitions"] for x in rs[1:]], [x["_overall"]["bits_per_transition"] for x in rs[1:]], label=r.split("/")[-1])
    ax.set(title="bits per held-out transition", xscale="log", yscale="log", xlabel="transitions")
    ax = axes[len(RULES) + 1]
    for r, rs in data.items():
        ax.plot([x["transitions"] for x in rs[1:]], [x["_overall"]["changed_acc"] for x in rs[1:]], label=r.split("/")[-1])
    ax.set(title="accuracy on changed entries", xscale="log", xlabel="transitions")
    ax.legend(fontsize=7)
    for ax in axes[len(RULES) + 2:]:
        ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.suptitle("E2: held-out accuracy per rule vs instances of the rule seen (log)")
    fig.tight_layout()
    fig.savefig(out, dpi=90)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
