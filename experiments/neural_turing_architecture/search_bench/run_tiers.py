"""The tier test, Part A -- tiers 0 and 1 in thought space (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §17.6 pre-registration,
results §17.7).

Worlds (global action keys, the loose regime, value noise 0.1):
  grid      8 × 8 grid (N 64, k 4): actions commute -- the reference
  heading   8 × 8 grid × 4 headings (N 256, k 6): forward / back / step left / step right in the frame of the heading,
            turn left / right -- movement with heading, actions do not commute. Goals "pos" (same heading) and "any"
Codes: raw, sr, the geometric code ("coord" on the grid; "allo" = position + heading vector on the heading world), and a
learned ADDITIVE code (`gcml.learn_code_als`, m = the true dimension: 2 / 4). Inverse models, each fitted on the search
experience of 24 training problems ("any" goals):
  global|code   one ridge W for every state (tier 0, GCML as published)
  gated|code    one ridge W per heading, mixed by the state's heading marginal (tier 1, the factor GIVEN)
  net|geo       a state-conditioned network W(c, Δc) (tier 1, NOT told the factor)
Per model: the k-step probe (with that model), one-step progress and cosine with the right thought from every node of
held-out problems, the GCML planner (`gcml_greedy`) at budgets 64 / 256; baselines grad_greedy and mcts_hybrid. Also
the R² between the learned code and the true position / heading.
    python experiments/neural_turing_architecture/search_bench/run_tiers.py --worlds grid heading --d 64 256 --seeds 6
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

from mock_task import TaskConfig, _unit  # noqa: E402
from structured_task import StructuredWalk  # noqa: E402
from searchers import ALGOS  # noqa: E402
import searchers_gcml  # noqa: E402,F401
from gcml import (transitions_search, transitions_random, InverseModel, GatedInverse, learn_code_als,  # noqa: E402
                  value_gradient_chains, kstep_probe, goal_delta, progress, nodes_toward)
from run_geometry import run_algo, r2  # noqa: E402

SHAPE = {"grid": (64, 4), "heading": (256, 6)}
GEO = {"grid": "coord", "heading": "allo"}
M_TRUE = {"grid": 2, "heading": 4}


def job(a):
    world_type, d, world, o = a
    torch.set_num_threads(1)
    N, k = SHAPE[world_type]
    cfg = TaskConfig(N=N, k=k, d=d, beta=16.0, gate=0.3, value_noise=o["noise"])
    extra = {}

    def make(ps, mode="any"):
        return StructuredWalk(cfg, world, ps, graph=world_type, keys="global", goal_mode=mode, extra_codes=extra)

    test_modes = ("pos", "any") if world_type == "heading" else ("any",)
    geo = GEO[world_type]
    gen = torch.Generator().manual_seed(world * 31 + d + 17)
    t1 = time.time()
    train = [world * 1000 + j for j in range(o["n_train"])]
    X, Z, X2 = transitions_search(lambda ps: make(ps), train, ALGOS["mcts_hybrid"], o["train_budget"], gen)
    t0 = make(world * 1000 + 999)
    R = transitions_random(t0, o["n_rand"], gen)
    extra["learned"] = learn_code_als(t0, *(torch.cat([u, v]) for u, v in zip((X, Z, X2), R)), m=M_TRUE[world_type],
                                      seed=world)
    t0 = make(world * 1000 + 999)
    Lc = t0.codes["learned"]
    fit = {"position": r2(Lc, t0.xy)}
    if world_type == "heading":
        fit["heading"] = r2(Lc, t0.codes["allo"][:, 2:])
    models = {f"global|{c}": (c, (lambda c=c: InverseModel("ridge", c))) for c in ("raw", "sr", geo, "learned")}
    if world_type == "heading":
        models.update({f"gated|{c}": (c, (lambda c=c: GatedInverse(c))) for c in ("sr", geo, "learned")})
    models[f"net|{geo}"] = (geo, lambda: InverseModel("mlp", geo, steps=o["mlp_steps"], seed=world))
    tr = value_gradient_chains(lambda ps: make(ps), train[:o["n_chain"]], gen)
    te = value_gradient_chains(lambda ps: make(ps), [world * 1000 + 600 + j for j in range(o["n_chain"])], gen)
    out = dict(world_type=world_type, d=d, world=world, n_transitions=int(X.shape[0]), r2=fit, models={}, baselines={})
    for name, (c, factory) in models.items():
        W = factory().fit(t0, X, Z, X2)
        r = dict(probe=kstep_probe(t0, c, tr, te, model=factory()))
        for mode in test_modes:
            prog, cos, plan = [], [], {B: [] for B in o["budgets"]}
            for j in range(o["n_test"]):
                ps = world * 1000 + 500 + j
                t = make(ps, mode)
                nodes = nodes_toward(t, t.target)
                Xn = t.onehot(nodes)
                z = W.propose(t, Xn, goal_delta(t, Xn, c))
                prog.append(float(progress(t, Xn, z, t.target).mean()))
                cos.append(float((z * _unit(t.right_thought(nodes, t.target))).sum(1).mean()))
                t.inv, t.inv_code = {"lin": W}, c
                for B in o["budgets"]:
                    plan[B].append(run_algo(t, "gcml_greedy", B, ps * 13 + B))
            r[mode] = dict(progress=sum(prog) / len(prog), cos=sum(cos) / len(cos),
                           planner={B: sum(v) / len(v) for B, v in plan.items()})
        out["models"][name] = r
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
    ap.add_argument("--worlds", nargs="+", default=["grid", "heading"])
    ap.add_argument("--d", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--n_train", type=int, default=24)
    ap.add_argument("--n_test", type=int, default=6)
    ap.add_argument("--n_chain", type=int, default=24)
    ap.add_argument("--train_budget", type=int, default=512)
    ap.add_argument("--n_rand", type=int, default=4096)
    ap.add_argument("--mlp_steps", type=int, default=1500)
    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    o = {k: v for k, v in vars(a).items() if k not in ("worlds", "d", "seeds", "workers", "out")}
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
