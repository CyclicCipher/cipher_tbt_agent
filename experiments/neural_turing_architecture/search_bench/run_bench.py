"""Runs the search algorithms on the mock task over a grid and prints success rates (planning doc §11).

    python experiments/neural_turing_architecture/search_bench/run_bench.py --d 16 256 --budget 512 4096 \
        --noise 0.1 0.3 --prior 0.0 --seeds 40 --out runs/grid1.json
CPU only; one process per worker, one thread each.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from mock_task import TaskConfig, GraphWalk, Budget, Sim, Solved, BudgetExhausted, _unit  # noqa: E402
from searchers import ALGOS  # noqa: E402


def run_one(job):
    cfg, seed, name, budget = job
    torch.set_num_threads(1)
    task = GraphWalk(TaskConfig(**cfg), seed)
    sim = Sim(task, Budget(budget))
    gen = torch.Generator().manual_seed(seed * 7919 + sum(map(ord, name)))
    t0 = time.time()
    solved = False
    try:
        ALGOS[name](sim, gen)
    except Solved:
        solved = True
    except BudgetExhausted:
        pass
    return dict(cfg=cfg, seed=seed, algo=name, budget=budget, solved=solved, used=sim.budget.used,
                secs=round(time.time() - t0, 3))


def sanity(cfg, seeds):
    """The right thought at each node must solve every instance in exactly L steps -- the task is solvable."""
    for seed in range(seeds):
        t = GraphWalk(TaskConfig(**cfg), seed)
        p = t.p0[None]
        z_star = t.keys[torch.arange(t.cfg.N), t.best_edge] @ t.P
        for s in range(t.cfg.L):
            p = t.step(p, z_star[p[:, :t.cfg.N].argmax(-1)])
        assert bool(t.success(p).all()), f"seed {seed}: the oracle chain does not solve the task"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="+", default=[16, 256])
    ap.add_argument("--budget", type=int, nargs="+", default=[512, 4096])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--prior", type=float, nargs="+", default=[0.0])
    ap.add_argument("--hack", type=float, nargs="+", default=[0.0], help="value_hack: weight of the exploitable spurious value term")
    ap.add_argument("--regime", nargs="+", default=["tight"], help="tight: beta 8, gate 0.5; loose: beta 16, gate 0.3")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--algos", nargs="+", default=list(ALGOS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    regimes = {"tight": dict(beta=8.0, gate=0.5), "loose": dict(beta=16.0, gate=0.3)}
    cells = [dict(d=d, value_noise=nz, prior_cos=pc, value_hack=hk, **regimes[rg]) for rg in a.regime for d in a.d
             for nz in a.noise for pc in a.prior for hk in a.hack]
    for c in cells:
        sanity(c, min(a.seeds, 10))
    jobs = [(c, s, name, b) for c in cells for b in a.budget for name in a.algos for s in range(a.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(run_one, jobs, chunksize=4))
    print(f"{len(jobs)} episodes in {time.time() - t0:.0f}s\n")
    for c in cells:
        print(f"beta={c['beta']} gate={c['gate']}  d={c['d']}  value_noise={c['value_noise']}  prior_cos={c['prior_cos']}  value_hack={c['value_hack']}")
        print("   " + f"{'algorithm':<16}" + "".join(f"{'B=' + str(b):>10}" for b in a.budget))
        for name in a.algos:
            row = []
            for b in a.budget:
                r = [x["solved"] for x in res if x["cfg"] == c and x["budget"] == b and x["algo"] == name]
                row.append(f"{sum(r) / len(r):>10.2f}")
            print(f"   {name:<16}" + "".join(row))
        print()
    if a.out:
        out = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(args=vars(a), results=res), open(out, "w"), indent=0)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
