"""The geometry test (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §14 pre-registration, results §15).

Can a state code in which thoughts ADD be LEARNED from experience (GCML's eq 11), does such a code let the inverse
model generalise COMPOSITIONALLY, and is COMMUTATIVITY the limit?
Worlds (all with global action keys, the loose regime, value noise 0.1):
  grid    8 × 8 grid (N 64, k 4)                         -- commuting actions
  grid2   product of two 4 × 4 grids (N 256, k 8)        -- a compositional world; TRAINING problems change ONE part,
                                                            test problems change one part ("one") or BOTH ("both")
  perm    the group S5, 4 generators (N 120, k 4)        -- non-commuting: the negative control
Codes: raw (one-hot), sr (successor-representation eigenmap), coord (true coordinates, where they exist), learned
(`gcml.learn_code_als` on the search experience plus 4,096 random-thought transitions; no coordinates given). Per world and code: a ridge inverse model W
fitted on the same search experience; one-step progress of W·(goal − state) from every node of held-out problems; the
k-step probe on value-gradient chains; the GCML planner (`gcml_greedy`) at budgets 64 and 256, against grad_greedy
(64, 256) and mcts_hybrid (256) on the same problems; where coordinates exist, the R² between them and the learned code.
    python experiments/neural_turing_architecture/search_bench/run_geometry.py --worlds grid grid2 perm --d 64 256 --seeds 6
    (§15's follow-ups: --worlds grid --m 2;  --worlds grid2 --m 4;  --worlds grid2 --m 4 --n_rand 16384)
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

from mock_task import TaskConfig, Budget, Sim, Solved, BudgetExhausted, _unit  # noqa: E402
from structured_task import StructuredWalk  # noqa: E402
from searchers import ALGOS  # noqa: E402
import searchers_gcml  # noqa: E402,F401
from gcml import (transitions_search, transitions_random, InverseModel, learn_code_als, value_gradient_chains, kstep_probe,  # noqa: E402
                  goal_delta, progress, nodes_toward)

SHAPE = {"grid": (64, 4), "grid2": (256, 8), "perm": (120, 4)}


def run_algo(t, name, B, seed):
    g = torch.Generator().manual_seed(seed)
    sim = Sim(t, Budget(B))
    try:
        ALGOS[name](sim, g)
    except Solved:
        return True
    except BudgetExhausted:
        pass
    return False


def r2(A, B):
    """Share of B's variance a linear map (with bias) from A explains."""
    A1 = torch.cat([A, torch.ones(len(A), 1)], 1)
    fit = A1 @ torch.linalg.lstsq(A1, B).solution
    return float(1 - ((B - fit) ** 2).sum() / ((B - B.mean(0)) ** 2).sum())


def job(a):
    world_type, d, world, o = a
    torch.set_num_threads(1)
    N, k = SHAPE[world_type]
    cfg = TaskConfig(N=N, k=k, d=d, beta=16.0, gate=0.3, value_noise=o["noise"])
    extra = {}

    def make(ps, mode="any"):
        return StructuredWalk(cfg, world, ps, graph=world_type, keys="global", goal_mode=mode, extra_codes=extra)

    train_mode = "one" if world_type == "grid2" else "any"
    test_modes = ("one", "both") if world_type == "grid2" else ("any",)
    gen = torch.Generator().manual_seed(world * 31 + d + 11)
    t1 = time.time()
    train = [world * 1000 + j for j in range(o["n_train"])]
    X, Z, X2 = transitions_search(lambda ps: make(ps, train_mode), train, ALGOS["mcts_hybrid"], o["train_budget"], gen)
    t0 = make(world * 1000 + 999, train_mode)
    R = transitions_random(t0, o["n_rand"], gen)                    # exploration: even coverage for the code
    extra["learned"] = learn_code_als(t0, *(torch.cat([u, v]) for u, v in zip((X, Z, X2), R)), m=o["m"], seed=world)
    t0 = make(world * 1000 + 999, train_mode)
    codes = [c for c in ("raw", "sr", "coord", "learned") if c in t0.codes]
    W = {c: InverseModel("ridge", c).fit(t0, X, Z, X2) for c in codes}
    tr = value_gradient_chains(lambda ps: make(ps, train_mode), train[:o["n_chain"]], gen)
    te = value_gradient_chains(lambda ps: make(ps, train_mode), [world * 1000 + 600 + j for j in range(o["n_chain"])], gen)
    out = dict(world_type=world_type, d=d, world=world, n_transitions=int(X.shape[0]), codes={}, baselines={})
    if "coord" in t0.codes:                                         # does the learned code span the true coordinates?
        out["r2"] = {"coord<-learned": r2(t0.codes["learned"], t0.codes["coord"]),
                     "learned<-coord": r2(t0.codes["coord"], t0.codes["learned"])}
    for c in codes:
        r = dict(probe=kstep_probe(t0, c, tr, te))
        for mode in test_modes:
            prog, cos, plan = [], [], {B: [] for B in o["budgets"]}
            for j in range(o["n_test"]):
                ps = world * 1000 + 500 + j
                t = make(ps, mode)
                nodes = nodes_toward(t, t.target)
                Xn = t.onehot(nodes)
                z = W[c].propose(t, Xn, goal_delta(t, Xn, c))
                prog.append(float(progress(t, Xn, z, t.target).mean()))
                cos.append(float((z * _unit(t.right_thought(nodes, t.target))).sum(1).mean()))
                t.inv, t.inv_code = {"lin": W[c]}, c
                for B in o["budgets"]:
                    plan[B].append(run_algo(t, "gcml_greedy", B, ps * 13 + B))
            r[mode] = dict(progress=sum(prog) / len(prog), cos=sum(cos) / len(cos),
                           planner={B: sum(v) / len(v) for B, v in plan.items()})
        out["codes"][c] = r
    for mode in test_modes:
        res = {}
        for name, B in [("grad_greedy", b) for b in o["budgets"]] + [("mcts_hybrid", max(o["budgets"]))]:
            res[f"{name}@{B}"] = sum(run_algo(make(world * 1000 + 500 + j, mode), name, B, (world * 1000 + 500 + j) * 13 + B)
                                     for j in range(o["n_test"])) / o["n_test"]
        out["baselines"][mode] = res
    out["secs"] = round(time.time() - t1, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", default=["grid", "grid2", "perm"])
    ap.add_argument("--d", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--n_train", type=int, default=24)
    ap.add_argument("--n_test", type=int, default=6)
    ap.add_argument("--n_chain", type=int, default=24)
    ap.add_argument("--train_budget", type=int, default=512)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n_rand", type=int, default=4096)
    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    o = dict(noise=a.noise, n_train=a.n_train, n_test=a.n_test, n_chain=a.n_chain, train_budget=a.train_budget, m=a.m,
             n_rand=a.n_rand, budgets=a.budgets)
    jobs = [(w, d, s, o) for w in a.worlds for d in a.d for s in range(a.seeds)]
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
