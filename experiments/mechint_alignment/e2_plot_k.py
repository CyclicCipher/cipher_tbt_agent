"""E2 arms K/K2/K3: a new rule (button_off) learned from k refutations by a trained world model, held-out accuracy of the
new rule and of an old one (theft) against fine-tuning updates.   python e2_plot_k.py out.png"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [("K: k=1, always rehearse, lr 5e-4, stream rows", "runs/e2/K/button_off.jsonl", 1),
        ("K2a: k=1, retire, kinds pool, lr 5e-4", "runs/e2/K2/a_k1_kinds_retire.jsonl", None),
        ("K3a: k=1, retire, kinds pool, lr 1e-4", "runs/e2/K3/a_k1_kinds_retire.jsonl", None),
        ("K3b: k=1, retire, stream rows, lr 1e-4", "runs/e2/K3/b_k1_stream_retire.jsonl", None),
        ("K3d: k=4, retire, kinds pool, lr 1e-4", "runs/e2/K3/d_k4_kinds_retire.jsonl", None)]


def main(out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, path, k in RUNS:
        rows = [json.loads(x) for x in open(path)]
        if k is not None:
            rows = [r for r in rows if r["k"] == k]
        ax[0].plot([r["update"] for r in rows], [r["acc"] for r in rows], marker=".", label=name)
        ax[1].plot([r["update"] for r in rows], [r["others"]["theft"] for r in rows], marker=".", label=name)
    ax[0].set(title="new rule (button_off), held-out accuracy", xlabel="fine-tuning updates", ylim=(-0.03, 1.03))
    ax[1].set(title="old rule (theft), held-out accuracy", xlabel="fine-tuning updates", ylim=(-0.03, 1.03))
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=100)


if __name__ == "__main__":
    main(sys.argv[1])


def k4(out, runs=(("K4: kinds by probe progress", "runs/e2/K4/progress.jsonl"), ("K4u: kinds uniform", "runs/e2/K4/uniform.jsonl"))):
    """K4: label-free sleep on J2's model, held-out accuracy per rule against sleep updates."""
    rules = ("button_off", "theft", "interact_full", "robot_burn", "push", "deliver")
    fig, axes = plt.subplots(1, len(rules), figsize=(18, 3.4))
    for name, path in runs:
        rows = [json.loads(x) for x in open(path)]
        for ax, e in zip(axes, rules):
            ax.plot([r["update"] for r in rows], [r["acc"][e] for r in rows], marker=".", label=name)
            ax.set(title=e, xlabel="sleep updates", ylim=(-0.03, 1.03))
    axes[0].legend(fontsize=7)
    fig.suptitle("K4: label-free sleep on a trained world model (J2): held-out accuracy per rule")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
