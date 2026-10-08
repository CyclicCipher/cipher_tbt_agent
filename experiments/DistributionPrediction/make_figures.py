"""Recreate the figures of the handwritten notes (TRANSCRIPT.md) from the RULES the notes describe.

Figures 1 and 2 run the drawing rule the notes give on page 2: "I drew one mark after another, moving in a spiral,
towards whatever section of the frame was nearest and not already filled to the appropriate density." Each is shown
twice: as a static picture (all you have without time) and coloured by drawing order (what watching the drawing
reveals).

Figures 3 and 4: a horseshoe of noisy points that is in fact the start of a logarithmic spiral, and two continuations
that both fit it — the spiral (the notes' "But it isn't") and a circle (the notes' "donut").

Usage:  python experiments/DistributionPrediction/make_figures.py      (writes figures/*.png; numpy + matplotlib)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
PENCIL = "#555555"


# ------------------------------------------------------------------------------------------------- the drawing rule
def draw_marks(n, grid, target, rng, start, turn=0.45, step=0.045, jitter=0.012):
    """The notes' rule. The frame [0,1]^2 is split into grid x grid sections; a section is FULL once it holds `target`
    marks. The pen turns by `turn` radians per mark (the spiral) and steps `step`; if the next spot is outside the frame
    or in a full section, the pen heads instead for the nearest section that is not yet full. Returns marks in order."""
    counts = np.zeros((grid, grid), dtype=int)
    centres = (np.stack(np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij"), -1) + 0.5) / grid
    pos, heading = np.array(start, float), rng.uniform(0, 2 * np.pi)
    marks = []

    def cell(p):
        return tuple(np.clip((p * grid).astype(int), 0, grid - 1))

    for i in range(n):
        heading += turn * (1 - 0.5 * i / max(n, 1))             # the turn eases: the spiral opens outward
        cand = pos + step * np.array([np.cos(heading), np.sin(heading)]) + rng.normal(0, jitter, 2)
        if not (0 <= cand[0] < 1 and 0 <= cand[1] < 1) or counts[cell(cand)] >= target:
            open_ = np.argwhere(counts < target)
            if len(open_) == 0:
                break
            d = np.linalg.norm(centres[open_[:, 0], open_[:, 1]] - pos, axis=1)
            goal = centres[tuple(open_[np.argmin(d)])]
            v = goal - pos
            heading = np.arctan2(v[1], v[0])
            cand = pos + min(step, np.linalg.norm(v)) * v / (np.linalg.norm(v) + 1e-9) + rng.normal(0, jitter, 2)
        pos = np.clip(cand, 0.001, 0.999)
        counts[cell(pos)] += 1
        marks.append(pos.copy())
    return np.array(marks)


def pencil(ax, pts, rng, colours=None, length=0.018, lw=1.1):
    """Each point as a short stroke at a random angle, like a pencil mark."""
    ang = rng.uniform(0, np.pi, len(pts))
    d = 0.5 * length * np.stack([np.cos(ang), np.sin(ang)], 1)
    for k, (p, q) in enumerate(zip(pts - d, pts + d)):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=PENCIL if colours is None else colours[k], lw=lw,
                solid_capstyle="round")


def frame(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def by_order(ax, pts, rng, cmap="viridis"):
    c = plt.get_cmap(cmap)(np.linspace(0, 1, len(pts)))
    ax.plot(pts[:, 0], pts[:, 1], color="#bbbbbb", lw=0.6, zorder=0)
    pencil(ax, pts, rng, c)


# ------------------------------------------------------------------------------------------------------- figures
def fig1(rng):
    pts = draw_marks(45, grid=16, target=1, rng=rng, start=(0.62, 0.74), turn=1.0, step=0.05)
    f, ax = plt.subplots(1, 2, figsize=(9, 4.6))
    frame(ax[0], "Fig (1), static: only WHERE the marks are")
    pencil(ax[0], pts, np.random.default_rng(1))
    frame(ax[1], "the same marks in drawing order (dark → light)")
    by_order(ax[1], pts, np.random.default_rng(1))
    f.suptitle("A patch of marks: a distribution without time, a rule with it", fontsize=11)
    f.tight_layout()
    f.savefig(OUT / "fig1_marks.png", dpi=150)
    plt.close(f)


def fig2(rng):
    pts = draw_marks(12 * 12 * 5, grid=12, target=5, rng=rng, start=(0.5, 0.5), turn=0.35, step=0.03)
    uni = np.random.default_rng(7).uniform(0, 1, (len(pts), 2))
    f, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    frame(ax[0], f"Fig (2): the same RULE, {len(pts)} marks, static")
    pencil(ax[0], pts, np.random.default_rng(2), length=0.012, lw=0.8)
    frame(ax[1], f"{len(uni)} marks placed uniformly at random, static")
    pencil(ax[1], uni, np.random.default_rng(3), length=0.012, lw=0.8)
    frame(ax[2], "the rule's marks in drawing order (dark → light)")
    by_order(ax[2], pts, np.random.default_rng(2))
    f.suptitle("Spread evenly enough, a rule looks like noise from the picture alone; the order still gives it away",
               fontsize=11)
    f.tight_layout()
    f.savefig(OUT / "fig2_dispersed.png", dpi=150)
    plt.close(f)


SPIRAL = dict(cx=0.5, cy=0.42, r0=0.12, b=0.12, seen=1.3 * np.pi, noise=0.022)


def spiral(theta, s=SPIRAL):
    r = s["r0"] * np.exp(s["b"] * theta)
    return np.stack([s["cx"] + r * np.cos(theta), s["cy"] + r * np.sin(theta)], 1)


def horseshoe(rng, n=40):
    th = np.sort(rng.uniform(0, SPIRAL["seen"], n))
    return spiral(th) + rng.normal(0, SPIRAL["noise"], (n, 2))


def fit_circle(p):
    """Algebraic (Kasa) circle fit; returns centre, radius, RMS distance of the points to the circle."""
    A = np.c_[2 * p, np.ones(len(p))]
    sol, *_ = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)
    c = sol[:2]
    r = np.sqrt(sol[2] + c @ c)
    return c, r, float(np.sqrt(np.mean((np.linalg.norm(p - c, axis=1) - r) ** 2)))


def fit_log_spiral(p):
    """Grid search over the centre; for each, a straight-line fit of log r against the unwrapped angle. Returns
    (centre, log r0, b, theta range, RMS radial error)."""
    best = None
    for cx in np.linspace(0.2, 0.8, 61):
        for cy in np.linspace(0.12, 0.72, 61):
            d = p - [cx, cy]
            th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
            if th[-1] < th[0]:
                continue
            lr = np.log(np.linalg.norm(d, axis=1))
            b, a = np.polyfit(th, lr, 1)
            err = np.exp(a + b * th) - np.linalg.norm(d, axis=1)
            rms = float(np.sqrt(np.mean(err ** 2)))
            if best is None or rms < best[-1]:
                best = (np.array([cx, cy]), a, b, (th[0], th[-1]), rms)
    return best


def fig3_and_4(rng):
    p = horseshoe(rng)
    f, ax = plt.subplots(figsize=(4.8, 4.8))
    frame(ax, 'Fig 3: "Seems like a horseshoe."')
    pencil(ax, p, np.random.default_rng(4))
    f.tight_layout()
    f.savefig(OUT / "fig3_horseshoe.png", dpi=150)
    plt.close(f)

    c, r, rms_c = fit_circle(p)
    cs, a, b, (t0, t1), rms_s = fit_log_spiral(p)
    n = len(p)
    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    for k, (title, rms, kparams) in enumerate([
            (f"continuation 1: a log spiral (b = {b:.2f})", rms_s, 4), (f"continuation 2: a circle — the donut", rms_c, 3)]):
        a_ = ax[k]
        a_.set_xlim(-0.15, 1.15)
        a_.set_ylim(-0.25, 1.05)
        a_.set_aspect("equal")
        a_.set_xticks([])
        a_.set_yticks([])
        a_.axvspan(-0.15, 1.15, color="#f3f3f3", zorder=-2)
        pencil(a_, p, np.random.default_rng(4))
        if k == 0:
            th = np.linspace(t0, t0 + 4 * np.pi, 600)
            q = cs + np.exp(a + b * th)[:, None] * np.stack([np.cos(th), np.sin(th)], 1)
            seen = (th >= t0) & (th <= t1)
            a_.plot(q[seen, 0], q[seen, 1], color="#2b6cb0", lw=1.5)
            a_.plot(q[~seen, 0], q[~seen, 1], color="#2b6cb0", lw=1.5, ls="--")
        else:
            th = np.linspace(0, 2 * np.pi, 400)
            a_.plot(c[0] + r * np.cos(th), c[1] + r * np.sin(th), color="#c05621", lw=1.5, ls="--")
        bic = n * np.log(rms ** 2) + kparams * np.log(n)
        a_.set_title(f"{title}\nfit to the {n} seen points: RMS {rms:.4f}, BIC-like score {bic:.1f}", fontsize=9.5)
    f.suptitle('Fig 4: the same seen points, two continuations out of distribution (dashed) — "But it isn\'t."',
               fontsize=11)
    f.tight_layout()
    f.savefig(OUT / "fig4_extrapolations.png", dpi=150)
    plt.close(f)
    return rms_s, rms_c


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig1(np.random.default_rng(11))
    fig2(np.random.default_rng(12))
    rms_s, rms_c = fig3_and_4(np.random.default_rng(13))
    print(f"wrote {sorted(x.name for x in OUT.glob('*.png'))}; horseshoe fits: spiral RMS {rms_s:.4f}, circle {rms_c:.4f}")
