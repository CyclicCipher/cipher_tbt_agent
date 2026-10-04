"""Tests G0–G4 of GCML's inverse model on the structure dial (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §12).

One job = one world (a variant, a thought dimension d, a world seed):
  G0  action-effect invariance of every state code in that world;
  G1  inverse models fitted from three kinds of experience (random thoughts, demonstrations, the transitions the
      mcts_hybrid search simulated on 16 TRAINING problems of the same world) -- Hebbian, ridge, MLP -- and, for ridge,
      at growing amounts of data; how many transitions carried any signal;
  G2  direction quality on held-out problems: from every node, does one step along the proposal progress? cosine with
      the right thought? -- for each inverse model (goal-directed and value-directed), against the value gradient (cost
      3), the exact inverse J⁺ (a reference; cost ~3·min(m, n)) and a random thought;
  G3/G4  search on held-out problems at several budgets: the inverse-model algorithms (models fitted on the search
      experience) against grad_greedy, mcts_hybrid and mcts_guided.
    python experiments/neural_turing_architecture/search_bench/run_gcml.py --variants V0 V2coord --d 64 256 --worlds 6
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from mock_task import TaskConfig, Budget, Sim, Solved, BudgetExhausted, _unit  # noqa: E402
from structured_task import StructuredWalk, VARIANTS  # noqa: E402
from searchers import ALGOS  # noqa: E402
import searchers_gcml  # noqa: E402,F401  (registers the GCML algorithms)
from gcml import (transitions_random, transitions_demo, transitions_search, InverseModel, goal_delta, value_delta,  # noqa: E402
                  jacobian_inverse, invariance, progress, nodes_toward)

SEARCH_ALGOS = ["grad_greedy", "mcts_hybrid", "mcts_guided", "gcml_greedy", "gcml_greedy_v", "invnet_greedy",
                "invnet_greedy_v", "mcts_gcml", "mcts_gcml_v", "mcts_invnet", "mcts_invnet_v", "mcts_hybrid_gcml"]


def job(a):
    variant, d, world, o = a
    torch.set_num_threads(1)
    V = VARIANTS[variant]
    code = V["code"]
    cfg = TaskConfig(d=d, beta=16.0, gate=0.3, value_noise=o["noise"])

    def make(ps):
        return StructuredWalk(cfg, world_seed=world, problem_seed=ps, graph=V["graph"], keys=V["keys"])

    gen = torch.Generator().manual_seed(world * 31 + d)
    t0 = make(world * 1000 + 999)
    out = dict(variant=variant, d=d, world=world)
    out["invariance"] = {cn: invariance(t0, cn, gen=gen) for cn in t0.codes}
    # -- G1: experience and inverse models --------------------------------------------------------------------------------
    t1 = time.time()
    train = [world * 1000 + j for j in range(o["n_train"])]
    test = [world * 1000 + 500 + j for j in range(o["n_test"])]
    data = {"random": transitions_random(t0, o["n_rand"], gen), "demo": transitions_demo(t0, o["n_demo"], gen),
            "search": transitions_search(make, train, ALGOS["mcts_hybrid"], o["train_budget"], gen)}
    models = {}
    for src in data:
        for kind in ("hebb", "ridge", "mlp"):
            models[(src, kind)] = InverseModel(kind, code, steps=o["mlp_steps"], seed=world).fit(t0, *data[src])
    for src in ("random", "search"):                                       # ridge vs amount of data
        n_all = data[src][0].shape[0]
        for n in (256, 1024, 4096):
            if n < n_all:
                idx = torch.randperm(n_all, generator=gen)[:n]
                models[(src, f"ridge@{n}")] = InverseModel("ridge", code).fit(t0, *(x[idx] for x in data[src]))
    out["n_used"] = {f"{s}/{k}": [m.n_used, m.n_seen] for (s, k), m in models.items()}
    out["fit_secs"] = round(time.time() - t1, 1)
    # -- G2: direction quality ----------------------------------------------------------------------------------------------
    acc = defaultdict(list)
    for ps in test:
        t = make(ps)
        nodes = nodes_toward(t, t.target)
        X = t.onehot(nodes)
        zs = t.right_thought(nodes, t.target)

        def score(name, z):
            acc[name + "|progress"].append(float(progress(t, X, z, t.target).mean()))
            acc[name + "|cos"].append(float((_unit(z) * _unit(zs)).sum(1).mean()))

        gd, vd = goal_delta(t, X, code), value_delta(t, X, code)
        for (src, kind), m in models.items():
            score(f"W[{src},{kind}]·(goal−state)", m.propose(t, X, gd))
            score(f"W[{src},{kind}]·∇V", m.propose(t, X, vd))
        z0 = _unit(torch.randn(len(nodes), d, generator=gen))
        zr = z0.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(t.value(t.step(X, zr)).sum(), zr)
        score("value gradient ∇_z V (cost 3)", _unit(g))
        score("exact inverse J⁺·(goal−state) (reference)", jacobian_inverse(t, X, gd, code, z0))
        score("random thought", z0)
    out["direction"] = {k: sum(v) / len(v) for k, v in acc.items()}
    # -- G3/G4: search ----------------------------------------------------------------------------------------------------
    res = []
    for ps in test:
        t = make(ps)
        t.inv = {"lin": models[("search", "ridge")], "mlp": models[("search", "mlp")]}
        t.inv_code = code
        for name in o["algos"]:
            for B in o["budgets"]:
                g2 = torch.Generator().manual_seed(ps * 7919 + sum(map(ord, name)) + B)
                sim = Sim(t, Budget(B))
                solved = False
                try:
                    ALGOS[name](sim, g2)
                except Solved:
                    solved = True
                except BudgetExhausted:
                    pass
                res.append(dict(problem=ps, algo=name, budget=B, solved=solved, used=sim.budget.used))
    out["search"] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--d", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--worlds", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--n_train", type=int, default=16)
    ap.add_argument("--n_test", type=int, default=5)
    ap.add_argument("--n_rand", type=int, default=4096)
    ap.add_argument("--n_demo", type=int, default=4096)
    ap.add_argument("--train_budget", type=int, default=512)
    ap.add_argument("--mlp_steps", type=int, default=1500)
    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--algos", nargs="+", default=SEARCH_ALGOS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    o = dict(noise=a.noise, n_train=a.n_train, n_test=a.n_test, n_rand=a.n_rand, n_demo=a.n_demo,
             train_budget=a.train_budget, mlp_steps=a.mlp_steps, budgets=a.budgets, algos=a.algos)
    jobs = [(v, d, w, o) for v in a.variants for d in a.d for w in range(a.worlds)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        outs = list(ex.map(job, jobs))
    print(f"{len(jobs)} worlds in {time.time() - t0:.0f}s")
    if a.out:
        p = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(args=vars(a), worlds=outs), open(p, "w"), indent=0)
        print(f"-> {p}")


if __name__ == "__main__":
    main()
