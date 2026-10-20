"""E4: return and gold delivered per consecutive held-out episode, learning while acting (E4b) vs the protected loop
(E4c).   python e4_plot.py out.png"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [("E4b: refutations rehearsed until explained (<= 30 updates), random interleaving",
         "runs/e4/E4b_trend3.jsonl"),
        ("E4c: protected (slow copy plans, kind-balanced interleaving, <= 5 updates)", "runs/e4/E4c_protected.jsonl")]


def main(out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for name, path in RUNS:
        eps = [json.loads(x) for x in open(path)]
        eps = [e for e in eps if "ret" in e]
        ax[0].plot(range(1, len(eps) + 1), [e["ret"] for e in eps], marker=".", label=name)
        ax[1].plot(range(1, len(eps) + 1), [e["delivered"] for e in eps], marker=".", label=name)
    ax[0].axhline(13.1, ls="--", c="gray", label="scripted task policy (same seeds)")
    ax[0].set(title="return per episode", xlabel="consecutive held-out episode")
    ax[1].set(title="gold delivered per episode", xlabel="consecutive held-out episode")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=100)


if __name__ == "__main__":
    main(sys.argv[1])
