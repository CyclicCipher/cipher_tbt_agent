"""Part 24 (docs/AGENT.md): does learning to learn emerge in the PFC column, from ordinary column learning?

Episodes of two kinds, each ambiguous on its first step, with the opposite answer:
- move: a 3 x 3 block that the keys move one cell (3 pixels). Right model: rigid. A recolour of the cells it touched
  also explains the first move.
- toggle: a 12 x 6 block that any key moves between two places 3 pixels apart, back and forth. Right model: a recolour
  (a colour swap on fixed cells). A rigid move also explains the first toggle.
New colour and place every episode. Measure: in each episode's first 10 actions, the steps on which the responsible
model mispredicted the contested cells. Arms: PERSIST keeps the PFC column across episodes (everything else is new);
RESET starts every episode with a new PFC.
    python -m arc_agent.test_pfc      (from experiments/symbolics, with venv312)
"""
import json
import time

import numpy as np

from arc_agent.brain import Brain
from arc_agent.pfc import PFC

EPISODES, ACTIONS, SEEDS = 12, 8, 5
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Episode:
    def __init__(self, kind, rng):
        self.kind, self.colour = kind, int(rng.integers(1, 14))
        self.w, self.h = (3, 3) if kind == "move" else (12, 12)
        self.x, self.y = int(rng.integers(24, 36)), int(rng.integers(24, 36))
        self.home = self.x

    def frame(self):
        g = np.zeros((64, 64), np.int8)
        g[self.y:self.y + self.h, self.x:self.x + self.w] = self.colour
        return g

    def step(self, a):
        if self.kind == "move":
            dx, dy = MOVES[a]
            self.x = min(60, max(0, self.x + 3 * dx))
            self.y = min(60, max(0, self.y + 3 * dy))
        else:
            self.x = self.home + 3 if self.x == self.home else self.home
        return self.frame()


def episode(kind, pfc, rng):
    ep, brain = Episode(kind, rng), Brain(pfc=pfc)
    g = ep.frame()
    brain.start(g, True)
    errors, steps = 0, 0
    key = int(rng.integers(1, 5))                       # part 24b: one key per episode
    for t in range(ACTIONS):
        a = key
        groups = [gid for gid, gr in pfc.groups.items() if gr["members"][0] in {c.cid for c in brain.columns}]
        resp = None
        if groups:
            gr = pfc.groups[groups[-1]]
            resp = next((c for c in gr["members"] if pfc.responsible(c, "key")), None)
        g2 = ep.step(a)
        brain.observe(g, (a, None, None), g2, ("key", a))
        if resp is not None:
            steps += 1
            lp = brain.last_evidence.get(resp)
            errors += lp is None or lp < -1e-6 * 0 + brain.last_cells.get(resp, 0) * np.log(0.99) - 1e-9
        g = g2
    return errors, steps


def run(arm, seed):
    rng = np.random.default_rng(seed)
    kinds = [str(k) for k in rng.choice(["move", "toggle"], EPISODES)]
    pfc = PFC(seed)
    out = []
    for e, kind in enumerate(kinds):
        if arm == "RESET":
            pfc = PFC(seed * 100 + e)
        pfc.groups, pfc.member = {}, {}                 # last episode's regions are gone; the learning stays
        out.append(episode(kind, pfc, rng) + (kind,))
    return out, pfc.attended()


def main():
    t0 = time.time()
    res = {}
    for arm in ("PERSIST", "RESET"):
        runs = [run(arm, s) for s in range(SEEDS)]
        res[arm] = runs
        err = np.array([[e for e, _, _ in r] for r, _ in runs], float)
        early, late = err[:, :3].mean(), err[:, -3:].mean()
        print(f"{arm:8s} errors per episode (mean over {SEEDS} seeds): {np.round(err.mean(0), 2).tolist()} | "
              f"episodes 1-3 {early:.2f}, 10-12 {late:.2f}, ratio {late / max(early, 1e-9):.2f} | "
              f"attended {[a for _, a in runs]}", flush=True)
    print(f"total {time.time() - t0:.1f} s")
    json.dump({k: [[list(x) for x in r] for r, _ in v] for k, v in res.items()},
              open("arc_agent/runs/test_pfc_b.json", "w"), indent=1, default=int)


if __name__ == "__main__":
    main()
