"""E12a figure: move and turn accuracy against transitions seen in a new world, for the ideal observer and E8c's
learned world models (small multiples: rule x split).

    python e12_plot.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = [("ideal", "ideal observer", "#2a78d6"), ("L_Ac", "adaptation, calibrated prior (L-Ac)", "#eb6834"),
          ("L_A", "adaptation, hand-set prior (L-A)", "#1baf7a"), ("L_A0", "one model for all worlds (L-A0)", "#9a9893")]
CHUNKS = 20                                                     # the first 160 transitions; all curves are flat after


def main():
    r = json.load(open("runs/e12/ideal_vs_model.json"))
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True, sharey=True, facecolor=SURFACE)
    x = [8 * (k + 1) for k in range(CHUNKS)]
    for j, split in enumerate(("test", "heldout")):
        for i, rule in enumerate(("move", "turn")):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            for key, label, colour in SERIES:
                ax.plot(x, r[split][key]["acc"][rule][:CHUNKS], color=colour, lw=2, label=label)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(color=GRID, lw=0.8)
            ax.tick_params(colors=MUTED, labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.set_title(f"{rule} predictions, {'new worlds' if split == 'test' else 'held-out physics'}",
                         fontsize=10, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel("accuracy", color=MUTED, fontsize=9)
            if i == 1:
                ax.set_xlabel("transitions seen in this world", color=MUTED, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.15, hspace=0.3, wspace=0.08)
    fig.savefig("runs/e12/ideal_vs_model.png", dpi=130, facecolor=SURFACE)
    print("saved runs/e12/ideal_vs_model.png")


if __name__ == "__main__":
    main()
