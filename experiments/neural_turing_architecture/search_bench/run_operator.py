"""The operator-code test (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §18.3 pre-registration, results §18.4): tier 2 for
continuous thoughts.

Per world (grid, heading, perm = S₅; global keys, the loose regime): training transitions from the `mcts_hybrid` search
on 24 training problems plus 4,096 random thoughts; held-out transitions from 8 other problems plus 1,024 random
thoughts. Models: the learned ADDITIVE code (§15's ALS, m = the true dimension) with c′ = c + V·z; the learned OPERATOR
code (`operator_code.OperatorModel`, K = 8, m ∈ {true, 8, 16}); the constructed code with only the gate and operators
learned. Measured per model: held-out unexplained share of the code's change, path integration along key-thought walks
(all / wall-free), R² against position and heading (heading world), prototype cleanliness. Planners on 6 held-out
problems per goal mode (heading: "pos" and "any"), budgets 64 / 256: `gcml_greedy` on the additive code (tier 0),
`op_look_own`, `op_look_sr`, `sim_look_sr` (the operator code at the m with the lowest held-out error), `grad_greedy`,
`mcts_hybrid`.
    python experiments/neural_turing_architecture/search_bench/run_operator.py --worlds heading --d 64 256 --seeds 6 --out runs/operator_heading.json
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

from mock_task import TaskConfig  # noqa: E402
from structured_task import StructuredWalk  # noqa: E402
from searchers import ALGOS  # noqa: E402
import searchers_gcml  # noqa: E402,F401
from gcml import transitions_search, transitions_random, InverseModel, learn_code_als  # noqa: E402
from operator_code import OperatorModel, AdditiveModel, path_integration, prototype_report  # noqa: E402
from matrix_code import whiten, constructed  # noqa: E402
from run_geometry import run_algo, r2  # noqa: E402

SHAPE = {"grid": (64, 4), "heading": (256, 6), "perm": (120, 4)}
M_TRUE = {"grid": 2, "heading": 4, "perm": 4}
PLANNERS = ["gcml_greedy", "op_look_own", "op_look_sr", "sim_look_sr", "grad_greedy", "mcts_hybrid"]


def job(a):
    world_type, d, world, o = a
    torch.set_num_threads(1)
    N, k = SHAPE[world_type]
    cfg = TaskConfig(N=N, k=k, d=d, beta=16.0, gate=0.3, value_noise=o["noise"])
    extra = {}

    def make(ps, mode="any"):
        return StructuredWalk(cfg, world, ps, graph=world_type, keys="global", goal_mode=mode, extra_codes=extra)

    t1 = time.time()
    gen = torch.Generator().manual_seed(world * 31 + d + 23)
    t0 = make(world * 1000 + 999)
    X, Z, X2 = transitions_search(lambda ps: make(ps), [world * 1000 + j for j in range(o["n_train"])],
                                  ALGOS["mcts_hybrid"], o["train_budget"], gen)
    R = transitions_random(t0, o["n_rand"], gen)
    X, Z, X2 = (torch.cat([u, v]) for u, v in zip((X, Z, X2), R))
    Xh, Zh, X2h = transitions_search(lambda ps: make(ps), [world * 1000 + 700 + j for j in range(o["n_held"])],
                                     ALGOS["mcts_hybrid"], o["train_budget"], gen)
    Rh = transitions_random(t0, o["n_rand"] // 4, gen)
    Xh, Zh, X2h = (torch.cat([u, v]) for u, v in zip((Xh, Zh, X2h), Rh))

    extra["add"] = learn_code_als(t0, X, Z, X2, m=M_TRUE[world_type], seed=world)
    t0 = make(world * 1000 + 999)
    models = {"additive": AdditiveModel(extra["add"]).fit(X, Z, X2)}
    for m in sorted({M_TRUE[world_type], 8, 16}):
        models[f"operator m={m}"] = OperatorModel(N, d, m, K=o["K"]).fit(X, Z, X2, steps=o["steps"], seed=world)
    Ec = whiten(constructed(world_type, t0))
    models["operator, constructed code"] = OperatorModel(N, d, Ec.shape[1], K=o["K"]).fit(X, Z, X2, E_fixed=Ec,
                                                                                          steps=o["steps"], seed=world)
    out = dict(world_type=world_type, d=d, world=world, n_transitions=int(X.shape[0]), models={}, planners={})
    for name, mod in models.items():
        E = mod.E
        held = (mod.unexplained(Xh, Zh, X2h) if isinstance(mod, OperatorModel) else
                float((((Xh[:, :N] @ E) + torch.nn.functional.normalize(Zh, dim=1) @ mod.V - X2h[:, :N] @ E) ** 2).sum()
                      / ((X2h[:, :N] @ E - Xh[:, :N] @ E) ** 2).sum()))
        acc, acc_clean = path_integration(t0, E, mod.predict, torch.Generator().manual_seed(world))
        r = dict(m=int(E.shape[1]), held_unexplained=held, pi=acc, pi_clean=acc_clean)
        if world_type == "heading":
            r["r2"] = {"position": r2(E, t0.xy), "heading": r2(E, t0.codes["allo"][:, 2:])}
        if isinstance(mod, OperatorModel):
            r["train_unexplained"] = mod.train_unexplained
            r["prototypes"] = prototype_report(t0, mod)
        out["models"][name] = r
    learned = [n for n in models if n.startswith("operator m=")]
    best = min(learned, key=lambda n: out["models"][n]["held_unexplained"])
    out["planner_model"] = best
    W = InverseModel("ridge", "add").fit(t0, X, Z, X2)
    modes = ("pos", "any") if world_type == "heading" else ("any",)
    for mode in modes:
        res = {}
        for name in PLANNERS:
            for B in o["budgets"]:
                if name == "mcts_hybrid" and B != max(o["budgets"]):
                    continue
                hits = 0
                for j in range(o["n_test"]):
                    ps = world * 1000 + 500 + j
                    t = make(ps, mode)
                    t.opm, t.metric = models[best], t.codes["sr"]
                    t.inv, t.inv_code = {"lin": W}, "add"
                    hits += run_algo(t, name, B, ps * 13 + B)
                res[f"{name}@{B}"] = hits / o["n_test"]
        out["planners"][mode] = res
    out["secs"] = round(time.time() - t1, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", default=["grid", "heading", "perm"])
    ap.add_argument("--d", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--n_train", type=int, default=24)
    ap.add_argument("--n_held", type=int, default=8)
    ap.add_argument("--n_test", type=int, default=6)
    ap.add_argument("--n_rand", type=int, default=4096)
    ap.add_argument("--train_budget", type=int, default=512)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    o = {k: v for k, v in vars(a).items() if k not in ("worlds", "d", "seeds", "seed0", "workers", "out")}
    jobs = [(w, d, s, o) for w in a.worlds for d in a.d for s in range(a.seed0, a.seed0 + a.seeds)]
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
