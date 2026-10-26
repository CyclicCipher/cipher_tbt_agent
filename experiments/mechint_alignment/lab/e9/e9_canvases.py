"""E9 figure: the final canvases, side by side with the starting canvas, per layout.

Rows: the canvas at the start, then the final canvas of each requested arm. Columns: layouts. Colour 0 (blank) is a
light neutral; colours 1-3 are categorical. A dot marks a cell the agent painted; dashed outlines mark the original
nature patches (stripes, rule 90), a dotted outline the noisy screen.

    python -m lab.e9.e9_canvases --run runs/e9c --seeds 100-107 --arms maximizer,appreciator
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import numpy as np                                 # noqa: E402
from matplotlib.patches import Patch, Rectangle    # noqa: E402
from matplotlib.lines import Line2D                # noqa: E402

from worlds import canvas as cv                                # noqa: E402

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
COLOURS = ["#e6e5e0", "#2a78d6", "#eb6834", "#1baf7a"]   # blank, then the reference palette's first three slots


def draw(ax, colour, painted, layout_world):
    n = colour.shape[0]
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle((c + 0.06, n - 1 - r + 0.06), 0.88, 0.88, color=COLOURS[colour[r, c]], lw=0))
            if painted is not None and painted[r, c]:
                ax.plot(c + 0.5, n - 1 - r + 0.5, "o", ms=2.6, color=INK)
    for kind, style in ((cv.STRIPES, (0, (3, 2))), (cv.RULE90, (0, (3, 2))), (cv.SCREEN, (0, (1, 1.5)))):
        r0, c0 = layout_world.patches[kind]
        h, w = {cv.STRIPES: cv.STRIPES_SIZE, cv.RULE90: cv.RULE90_SIZE, cv.SCREEN: cv.SCREEN_SIZE}[kind]
        ax.add_patch(Rectangle((c0, n - r0 - h), w, h, fill=False, ec=MUTED, lw=1.0, ls=style))
    ax.set_xlim(-0.2, n + 0.2)
    ax.set_ylim(-0.2, n + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/e9c")
    p.add_argument("--seeds", default="100-107")
    p.add_argument("--arms", default="maximizer,appreciator")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds, arms = list(range(lo, hi + 1)), a.arms.split(",")
    rows = ["start"] + arms
    fig, axes = plt.subplots(len(rows), len(seeds), figsize=(1.9 * len(seeds), 2.05 * len(rows) + 0.6),
                             facecolor=SURFACE, squeeze=False)
    for j, s in enumerate(seeds):
        world = cv.Canvas(s)
        for i, row in enumerate(rows):
            if row == "start":
                draw(axes[i, j], world.colour, None, world)
            else:
                z = np.load(f"{a.run}/{row}_{s}.npz")
                draw(axes[i, j], z["colour"], z["painted"], world)
            if i == 0:
                axes[i, j].set_title(f"layout {s}", fontsize=9, color=MUTED)
        for i, row in enumerate(rows):
            if j == 0:
                axes[i, 0].text(-0.8, cv.N / 2, row, rotation=90, ha="center", va="center", fontsize=10, color=INK)
    handles = [Patch(color=COLOURS[k], label="blank" if k == 0 else f"colour {k}") for k in range(4)]
    handles += [Line2D([], [], marker="o", ls="", ms=4, color=INK, label="painted by the agent"),
                Patch(fill=False, ec=MUTED, ls=(0, (3, 2)), label="original nature (stripes, rule 90)"),
                Patch(fill=False, ec=MUTED, ls=(0, (1, 1.5)), label="noisy screen")]
    ncol = len(handles) if len(seeds) >= 6 else 4                       # narrow figures: the legend wraps
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=8, labelcolor=INK)
    fig.subplots_adjust(left=0.03 if len(seeds) >= 6 else 0.06, right=0.995, top=0.94,
                        bottom=0.07 if ncol == len(handles) else 0.12, wspace=0.08, hspace=0.12)
    out = a.out or f"{a.run}/canvases.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print("saved", out)


if __name__ == "__main__":
    main()
