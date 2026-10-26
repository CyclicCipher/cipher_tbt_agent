"""E4/E5: gold delivered per consecutive held-out episode for the acting agents (plan in the learned world model).
python -m lab.e4.e4_plot out.png"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("layouts 0-29", [("E4b: learn each refutation (<= 30 updates), random interleaving", "runs/e4/E4b_trend3.jsonl"),
                      ("E4c: protected (slow copy plans, kind-balanced interleaving, <= 5 updates)", "runs/e4/E4c_protected.jsonl"),
                      ("E4d: E4c + sleep on own refutations after each episode", "runs/e4/E4d_daynight.jsonl")]),
    ("layouts 100-129 (replication)", [("E4c: protected, no sleep", "runs/e4/E4c_rep.jsonl"),
                                       ("E4d: + sleep after each episode", "runs/e4/E4d_rep.jsonl"),
                                       ("E5b: E4d's agent with a world model from random play only", "runs/e5/act_b.jsonl")]),
]


def main(out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    for ax, (title, runs) in zip(axes, PANELS):
        for name, path in runs:
            eps = [e for e in (json.loads(x) for x in open(path)) if "ret" in e]
            ax.plot(range(1, len(eps) + 1), [e["delivered"] for e in eps], marker=".", label=name)
        ax.axhline(14.0, ls="--", c="gray", lw=0.8, label="scripted task policy, ~14 gold")
        ax.set(title=f"gold delivered per episode, {title}", xlabel="consecutive held-out episode")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=100)


if __name__ == "__main__":
    main(sys.argv[1])
