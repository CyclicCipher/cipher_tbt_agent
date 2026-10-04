"""Why does the GCML planner lose success on two-part goals in the product world? (CURRICULUM_LESS_COCONUT_AND_SEARCH.md
§15.3.) Deterministic inverse-model chains (z_t = W·(goal − state), no noise, 8 steps) with the true-coordinate code and
W fitted as in `run_geometry.py`, on 40 held-out problems per goal mode per world; results grouped by the number of
coordinate AXES the goal differs in (a one-part goal at distance 4 in a 4 × 4 part always needs exactly 2; a two-part
goal 2–4). Reports success (> 0.5 of the mass on the target at some step), the peak target mass, and the largest single
node's mass after steps 1 and 4 (how concentrated the state stays).
    python experiments/neural_turing_architecture/search_bench/probe_axes.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from mock_task import TaskConfig  # noqa: E402
from structured_task import StructuredWalk  # noqa: E402
from searchers import ALGOS  # noqa: E402
from gcml import transitions_search, InverseModel, goal_delta  # noqa: E402


def main(worlds=6, n_test=40):
    torch.set_num_threads(4)
    res = defaultdict(lambda: defaultdict(list))
    for d in (64, 256):
        cfg = TaskConfig(N=256, k=8, d=d, beta=16.0, gate=0.3, value_noise=0.1)
        for world in range(worlds):
            def make(ps, mode="one"):
                return StructuredWalk(cfg, world, ps, graph="grid2", keys="global", goal_mode=mode)
            gen = torch.Generator().manual_seed(world * 31 + d + 11)
            X, Z, X2 = transitions_search(lambda ps: make(ps, "one"), [world * 1000 + j for j in range(24)],
                                          ALGOS["mcts_hybrid"], 512, gen)
            W = InverseModel("ridge", "coord").fit(make(world * 1000 + 999), X, Z, X2)
            for mode in ("one", "both"):
                for j in range(n_test):
                    t = make(world * 1000 + 500 + j, mode)
                    axes = int(((t.xy[t.target] - t.xy[t.start]).abs() > 0).sum())
                    Xs, best, conc = t.p0[None], 0.0, []
                    with torch.no_grad():
                        for _ in range(8):
                            Xs = t.step(Xs, W.propose(t, Xs, goal_delta(t, Xs, "coord")))
                            conc.append(float(Xs[0, :cfg.N].max()))
                            best = max(best, float(Xs[0, t.target]))
                    res[(d, mode)][axes].append((best > 0.5, best, conc[0], conc[3]))
    print("| d | goal | axes | n | solved | peak target mass | max node mass after step 1 | after step 4 |")
    print("|---|---|---|---|---|---|---|---|")
    for (d, mode), by in sorted(res.items()):
        for axes, v in sorted(by.items()):
            n = len(v)
            print(f"| {d} | {mode} | {axes} | {n} | {sum(x[0] for x in v) / n:.2f} | {sum(x[1] for x in v) / n:.2f} | "
                  f"{sum(x[2] for x in v) / n:.2f} | {sum(x[3] for x in v) / n:.2f} |")


if __name__ == "__main__":
    main()
