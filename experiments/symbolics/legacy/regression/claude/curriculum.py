"""Experiment B's runner (docs/CURRICULUM.md): the curriculum, the wake-sleep cycles, the held-out evaluation.
Every stage is resumable and each run stays under 10 minutes; tasks are fitted in parallel (8 processes).
    python -m regression.claude.curriculum simulate         the 48 curriculum tasks, 50 simulations each
    python -m regression.claude.curriculum roughness        each task's simulator roughness (the solved threshold)
    python -m regression.claude.curriculum wake 1           fit every curriculum task (cycle 1: primitives, guide)
    python -m regression.claude.curriculum sleep 1          learn the library, fantasies, case memory
    ... wake 2, sleep 2, wake 3, sleep 3                    (the library after sleep 3 is frozen)
    python -m regression.claude.curriculum heldout          simulate the held-out tasks (only after the freeze)
    python -m regression.claude.curriculum evaluate         arms A0-A4 and the references, resumable
    python -m regression.claude.curriculum report
Experiment B2 (selection with the noise level known): --variant nf, after `roughness_heldout`; its cycles, library and
results go to runs/curriculum_nf; the simulations are shared with B.
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

from regression.claude import library as LIB
from regression.claude.guide import CaseMemory, Guide
from regression.claude.symreg2 import GRID, EXP_GRID, Finder, PRIMITIVES
from regression.claude.tasks import curriculum, heldout

DATA = "runs/curriculum"                          # simulations and roughness, shared by every variant
N_TRAIN, SOLVED, N_FANTASY = 50, 0.02, 100
NOISE_FLOOR = 0.002                                 # B2: sigma = max(roughness, NOISE_FLOOR) * std(training y)


def variant():
    return os.environ.get("CURRICULUM_VARIANT", "")


def out():
    return DATA + (f"_{variant()}" if variant() else "")


def noise_for(rough, y):
    return max(rough, NOISE_FLOOR) * float(np.std(y)) if variant() in ("nf", "v3") else None


def finder_opts(arm="A4"):
    """B3 (variant v3): finder version 3, minus one extension in the ablation arms."""
    if variant() != "v3":
        return {}
    opts = dict(outer=True, affine=True, relational=True)
    if arm.startswith("A2-"):
        opts[{"inv": "outer", "aff": "affine", "rel": "relational"}[arm[3:]]] = False
    return opts


def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def roughness():
    """Per task: 61-point sweeps through the box centre along each input (feasible points only); roughness = std of
    the residual of a degree-8 polynomial fit, over the std of the task's training y; the task's floor = the largest."""
    path = f"{DATA}/roughness.json"
    if os.path.exists(path):
        print("roughness exists")
        return
    tasks, data = curriculum(), load(f"{DATA}/curriculum_data.pkl")
    sweeps, jobs = {}, []
    for k, t in enumerate(tasks):
        mid = (t.lo + t.hi) / 2
        for i in range(len(t.names)):
            X = np.repeat(mid[None], 61, 0)
            X[:, i] = np.linspace(t.lo[i], t.hi[i], 61)
            X = X[t.feasible(X)]
            sweeps[(t.name, i)] = X
            jobs += [("c", k, x) for x in X]
    with Pool(8) as pool:
        ys = pool.map(run_sim, jobs, chunksize=8)
    out, at = {}, 0
    for t in tasks:
        floors = []
        for i in range(len(t.names)):
            X = sweeps[(t.name, i)]
            y = np.array(ys[at:at + len(X)])
            at += len(X)
            if len(X) >= 15:
                u = (X[:, i] - X[:, i].mean()) / X[:, i].std()
                floors.append(float(np.std(y - np.polyval(np.polyfit(u, y, 8), u)) / data[t.name]["y"].std()))
        out[t.name] = max(floors) if floors else 0.0
    json.dump(out, open(path, "w"), indent=1)
    for name, v in out.items():
        print(f"   {name:34s} roughness {v:.4f}")


def run_sim(job):
    kind, k, x = job
    tasks = curriculum() if kind == "c" else heldout()
    return tasks[k](x)


# ---- curriculum data --------------------------------------------------------------------------------------------------
def simulate():
    path = f"{DATA}/curriculum_data.pkl"
    if os.path.exists(path):
        print("curriculum data exist")
        return
    t0 = time.time()
    tasks, data = curriculum(), {}
    for k, t in enumerate(tasks):
        rng = np.random.default_rng([7, k])
        X = t.sample(rng, N_TRAIN)
        data[t.name] = dict(X=X, M=t.margins(X), domain=t.domain(rng))
    jobs = [("c", k, x) for k, t in enumerate(tasks) for x in data[t.name]["X"]]
    with Pool(8) as pool:
        ys = pool.map(run_sim, jobs, chunksize=4)
    at = 0
    for t in tasks:
        data[t.name]["y"] = np.array(ys[at:at + N_TRAIN])
        at += N_TRAIN
    save(data, path)
    print(f"{len(jobs)} curriculum simulations in {time.time() - t0:.0f}s")


# ---- fitting one task, any arm ----------------------------------------------------------------------------------------
def fit_one(names, mnames, lo, hi, X, y, M, domain, use_guide, lib_specs, prior, memory, noise=None, opts=None):
    """Returns (finder, guide or None). use_guide: G1 + G2 (+ G3 when a memory is given)."""
    prior = dict(prior or {})
    g = None
    hints, start, start_tr = {}, None, None
    if use_guide:
        g = Guide(X, y, lo, hi)
        if memory is not None:
            for name, b in memory.boosts(g.signature(len(names))).items():
                prior[name] = prior.get(name, 1.0) * b
        if g.additive:
            start, _ = g.decompose(X, y, lambda i: Finder([names[i]], library=lib_specs, prior=prior, k_max=3),
                                   domain=domain)
            start_tr = g.transform
        hints = g.hints()
    f = Finder(names, mnames, library=lib_specs, prior=prior, **(opts or {})).fit(
        X, y, M, domain, hints=hints, start=start, start_transform=start_tr, noise=noise)
    return f, g


def templates_used(rec):
    return sorted({b.template for b in rec["state"].blocks})


def wake_task(args):
    k, cycle = args
    warnings.simplefilter("ignore")
    task = curriculum()[k]
    path = f"{out()}/cycle{cycle}/{task.name}.pkl"
    if os.path.exists(path):
        return task.name, "cached"
    data = load(f"{DATA}/curriculum_data.pkl")[task.name]
    lib, prior, memory = ([], {}, None) if cycle == 1 else load(f"{out()}/sleep{cycle - 1}.pkl")[:3]
    t0 = time.time()
    rough = json.load(open(f"{DATA}/roughness.json"))[task.name]
    f, g = fit_one(task.names, task.mnames, task.lo, task.hi, data["X"], data["y"], data["M"], data["domain"], True,
                   [t.spec() for t in lib], prior, memory, noise_for(rough, data["y"]), finder_opts())
    pred = f.predict_one(f.best, data["X"], data["M"])
    nrmse = float(np.sqrt(np.mean((pred - data["y"]) ** 2)) / data["y"].std())
    rec = f.best
    result = dict(name=task.name, nrmse_train=nrmse, formula=f.formula(), terms=f.n_terms(), seconds=time.time() - t0,
                  used=templates_used(rec), signature=g.signature(len(task.names)),
                  record=dict(state=rec["state"], transform=rec["transform"]), pairs=list(getattr(g, "pairs", [])),
                  outer=str(rec["transform"]))
    save(result, path)
    return task.name, f"NRMSE(train) {nrmse:.4f} terms {f.n_terms()} {time.time() - t0:.0f}s  {f.formula()[:110]}"


def wake(cycle, budget=540):
    t0 = time.time()
    tasks = curriculum()
    todo = [(k, cycle) for k, t in enumerate(tasks) if not os.path.exists(f"{out()}/cycle{cycle}/{t.name}.pkl")]
    with Pool(8) as pool:
        for name, msg in pool.imap_unordered(wake_task, todo):
            print(f"cycle {cycle} {name:34s} {msg}", flush=True)
            if time.time() - t0 > budget:
                pool.terminate()
                print("budget reached; run again to continue", flush=True)
                return
    done = sum(os.path.exists(f"{out()}/cycle{cycle}/{t.name}.pkl") for t in tasks)
    print(f"cycle {cycle}: {done}/{len(tasks)} tasks done, {time.time() - t0:.0f}s")


# ---- sleep ------------------------------------------------------------------------------------------------------------
def fantasies(lib, rng, n=N_FANTASY):
    """Synthetic tasks: 1-4 library templates (or primitives if the library is empty), random bindings and
    parameters, random boxes in [0.2, 2], evaluated exactly. Returns (signature, templates used) pairs."""
    from regression.claude.symreg2 import Block, Context
    specs = [t.spec() for t in lib]
    prims = [dict(name="pow", factors=[("pow", 0)], slots=1, theta=np.array([1.0]), free=np.array([True]), group=False),
             dict(name="log", factors=[("log", 0)], slots=1, theta=np.zeros(0), free=np.zeros(0, bool), group=False),
             dict(name="exp", factors=[("exp", 0)], slots=1, theta=np.array([1.0]), free=np.array([True]), group=False)]
    pool = specs + prims
    out = []
    while len(out) < n:
        d = int(rng.integers(1, 4))
        lo = rng.uniform(0.2, 1.0, d)
        hi = lo * rng.uniform(1.5, 4.0, d)
        X = rng.uniform(lo, hi, (50, d))
        ctx = Context(X, np.zeros((50, 0)), {"X": X, "M": np.zeros((50, 0))})
        y, used = np.full(50, rng.uniform(0, 2)), []
        for _ in range(int(rng.integers(1, 5))):
            s = pool[int(rng.integers(len(pool)))]
            if s["group"] and d < 2 or s["slots"] > d or (s["group"] and s["slots"] == 2 and d < 3):
                continue
            if s["group"] and s["slots"] == 2:                   # a relational template: all input pairs
                bind = [tuple(p) for p in __import__("itertools").combinations(range(d), 2)]
            elif s["group"]:
                bind = [(v,) for v in range(d)]
            else:
                bind = [tuple(rng.choice(d, s["slots"], replace=False))]
            theta = np.array(s["theta"], float).copy()
            for j in np.nonzero(s["free"])[0]:
                theta[j] = rng.choice(GRID)
            b = Block(s["factors"], bind, theta, s["free"], s["name"])
            with np.errstate(all="ignore"):
                C = b.columns(X, np.zeros((50, 0)), ctx)
            if not np.all(np.isfinite(C)):
                continue
            y = y + C @ rng.lognormal(0, 0.5, C.shape[1])
            used.append(s["name"])
        if not used or not np.all(np.isfinite(y)) or y.std() == 0:
            continue
        g = Guide(X, y, lo, hi)
        out.append((g.signature(d), used))
    return out


def sleep(cycle):
    tasks = curriculum()
    results = {t.name: load(f"{out()}/cycle{cycle}/{t.name}.pkl") for t in tasks}
    prev_lib = [] if cycle == 1 else load(f"{out()}/sleep{cycle - 1}.pkl")[0]
    corpus = LIB.Corpus()
    floor = json.load(open(f"{DATA}/roughness.json"))
    solved = [r for r in results.values() if r["nrmse_train"] <= max(SOLVED, 2 * floor[r["name"]])]
    for r in solved:
        corpus.add(r["record"], N_TRAIN)
    t0 = time.time()
    lib, uses, log = LIB.learn(corpus, prev_lib)
    prior = {k: u + 1.0 for k, u in uses.items()}
    memory = CaseMemory()
    for r in solved:
        memory.add(r["signature"], r["used"])
    for sig, used in fantasies(lib, np.random.default_rng([cycle, 99])):
        memory.add(sig, used)
    save((lib, prior, memory, dict(log=log, solved=len(solved), n_tasks=len(tasks))), f"{out()}/sleep{cycle}.pkl")
    print(f"sleep {cycle}: {len(solved)}/{len(tasks)} tasks solved (train NRMSE <= max({SOLVED}, 2 x roughness)); "
          f"library learning {time.time() - t0:.0f}s")
    for entry in log:
        print(f"   {entry}")
    print(f"   uses: { {k: v for k, v in uses.items() if v} }")


# ---- held-out ---------------------------------------------------------------------------------------------------------
SEEDS, BUDGETS, N_POOL, N_TEST, N_EX = (0, 1, 2), (30, 100), 100, 500, 200


def outside(t, lo, hi):
    return lambda x: bool(np.any((x < t.lo - 1e-12) | (x > t.hi + 1e-12)))


def simulate_heldout():
    assert os.path.exists(f"{out()}/sleep3.pkl"), "the library must be frozen (sleep 3) before the held-out tasks exist"
    path = f"{DATA}/heldout_data.pkl"
    if os.path.exists(path):
        print("held-out data exist")
        return
    t0 = time.time()
    tasks, data, jobs = heldout(), {}, []
    for k, t in enumerate(tasks):
        rng = np.random.default_rng([11, k])
        sets = {f"train{s}": t.sample(np.random.default_rng([s, k, 5]), N_POOL) for s in SEEDS}
        sets["test"] = t.sample(rng, N_TEST)
        ok_ex = t.extrap_ok or outside(t, *t.extrap)
        sets["extrap"] = t.sample(rng, N_EX, *t.extrap, extra_ok=ok_ex)
        sets["far"] = t.sample(rng, N_EX, *t.far, extra_ok=outside(t, *t.far))
        data[t.name] = dict(sets=sets, domain=t.domain(rng))
        jobs += [("h", k, x) for key in sets for x in sets[key]]
    with Pool(8) as pool:
        ys = pool.map(run_sim, jobs, chunksize=8)
    at = 0
    for t in tasks:
        for key, X in data[t.name]["sets"].items():
            data[t.name][f"y_{key}"] = np.array(ys[at:at + len(X)])
            at += len(X)
    save(data, path)
    print(f"{len(jobs)} held-out simulations in {time.time() - t0:.0f}s")


def limit_check(task, model_predict, y_train):
    """The formula's value in a limit absent from the training data, against theory (docs/CURRICULUM.md)."""
    from regression.claude.tasks import euler_bernoulli
    if task.limit == "slender":
        nseg = len(task.names)
        load_ = "uniform" if "uniform" in task.name else "tip"
        c = euler_bernoulli(nseg, 4.0, "cantilever", load_)
        X = np.full((1, nseg), 0.15)
        return float(model_predict(X)[0] / (c @ X[0] ** -3.0))           # 1 = the textbook limit
    if task.limit == "small_hole":
        X = np.array([[0.02, 0.5], [0.02, 0.6]])
        return float(np.mean((model_predict(X) - 1) / (3 * math.pi * 0.02 ** 2)))   # 1 = the dilute limit
    if task.limit == "no_hole":
        names = task.names
        base = (task.lo + task.hi) / 2
        X = np.repeat(base[None], 3, 0)
        X[:, names.index("r")] = 0.01
        if "xc" in names:
            X[:, names.index("xc")] = np.linspace(task.lo[names.index("xc")], task.hi[names.index("xc")], 3)
        return float(np.max(np.abs(model_predict(X) - 1)) / y_train.std())  # 0 = no effect without a hole
    return None


def evaluate_one(args):
    k, arm, n, seed = args
    warnings.simplefilter("ignore")
    from regression.claude.symreg import SymbolicRegressor
    from regression.claude.surrogates import GP
    task = heldout()[k]
    data = load(f"{DATA}/heldout_data.pkl")[task.name]
    lib, prior, memory = load(f"{out()}/sleep3.pkl")[:3]
    S = data["sets"]
    X, y = S[f"train{seed}"][:n], data[f"y_train{seed}"][:n]
    M = task.margins(X)
    t0 = time.time()
    rec = dict(task=task.name, arm=arm, n=n, seed=seed)
    if arm == "A0":
        m = SymbolicRegressor(task.names).fit(X, y)
        pred = lambda Z: m.predict(Z)
        rec.update(formula=m.formula(), terms=m.n_terms())
    elif arm == "GP":
        m = GP(task.lo, task.hi).fit(X, y)
        pred = lambda Z: m.predict(Z)
    elif arm.startswith("law"):
        cols = (lambda Z: np.column_stack([np.ones(len(Z)), Z ** -3.0])) if arm == "law_bending" else \
               (lambda Z: np.column_stack([np.ones(len(Z)), Z ** -3.0, Z ** -1.0]))
        c = np.linalg.lstsq(cols(X), y, rcond=None)[0]
        pred = lambda Z: cols(Z) @ c
    else:
        use_guide = arm in ("A2", "A4") or arm.startswith("A2-")
        specs = [t.spec() for t in lib] if arm in ("A3", "A4") else None
        rough = json.load(open(f"{DATA}/roughness_heldout.json"))[task.name] if variant() in ("nf", "v3") else 0.0
        f, g = fit_one(task.names, task.mnames, task.lo, task.hi, X, y, M, data["domain"], use_guide, specs,
                       prior if specs else None, memory if arm == "A4" else None, noise_for(rough, y), finder_opts(arm))
        rec.update(outer=str(f.best["transform"]), pairs=[list(p) for p in getattr(g, "pairs", [])] if g else [])
        pred = lambda Z: f.predict(Z, task.margins(Z))
        rec.update(formula=f.formula(), terms=f.n_terms(), frontier=len(f.frontier),
                   spread_ex=f.spread(S["extrap"], task.margins(S["extrap"]), y.std()),
                   spread_far=f.spread(S["far"], task.margins(S["far"]), y.std()))
    rec["seconds"] = time.time() - t0
    for key in ("test", "extrap", "far"):
        p = pred(S[key])
        yk = data[f"y_{key}"]
        rec[f"nrmse_{key}"] = float(np.sqrt(np.mean((p - yk) ** 2)) / yk.std()) if np.all(np.isfinite(p)) else float("inf")
    rec["limit"] = limit_check(task, pred, y)
    return rec


ARMS = ("A0", "A1", "A2", "A3", "A4", "GP")


def arms():
    if variant() == "v3":
        return ("A1", "A2", "A3", "A4", "A2-inv", "A2-aff", "A2-rel")
    return ("A1", "A2", "A3", "A4") if variant() == "nf" else ARMS


def roughness_heldout():
    """As `roughness`, for the held-out tasks (B2): sweeps through the box centre, over the std of the seed-0 pool's y."""
    path = f"{DATA}/roughness_heldout.json"
    if os.path.exists(path):
        print("held-out roughness exists")
        return
    tasks, data = heldout(), load(f"{DATA}/heldout_data.pkl")
    sweeps, jobs = {}, []
    for k, t in enumerate(tasks):
        mid = (t.lo + t.hi) / 2
        for i in range(len(t.names)):
            X = np.repeat(mid[None], 61, 0)
            X[:, i] = np.linspace(t.lo[i], t.hi[i], 61)
            X = X[t.feasible(X)]
            sweeps[(t.name, i)] = X
            jobs += [("h", k, x) for x in X]
    with Pool(8) as pool:
        ys = pool.map(run_sim, jobs, chunksize=8)
    res, at = {}, 0
    for t in tasks:
        floors = []
        for i in range(len(t.names)):
            X = sweeps[(t.name, i)]
            y = np.array(ys[at:at + len(X)])
            at += len(X)
            if len(X) >= 15:
                u = (X[:, i] - X[:, i].mean()) / X[:, i].std()
                floors.append(float(np.std(y - np.polyval(np.polyfit(u, y, 8), u)) / data[t.name]["y_train0"].std()))
        res[t.name] = max(floors) if floors else 0.0
    json.dump(res, open(path, "w"), indent=1)
    for name, v in res.items():
        print(f"   {name:26s} roughness {v:.4f}")


def evaluate(budget=540):
    path = f"{out()}/heldout_results.jsonl"
    have = {(r["task"], r["arm"], r["n"], r["seed"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    tasks = heldout()
    jobs = [(k, arm, n, s) for k, t in enumerate(tasks) for n in BUDGETS for s in SEEDS for arm in arms()
            if (t.name, arm, n, s) not in have]
    if not variant():
        jobs += [(k, arm, n, s) for k, t in enumerate(tasks) if t.family == "stepped" for n in BUDGETS for s in SEEDS
                 for arm in ("law_bending", "law_bending_shear") if (t.name, arm, n, s) not in have]
    t0 = time.time()
    with Pool(8) as pool:
        for rec in pool.imap_unordered(evaluate_one, jobs):
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{rec['task']:24s} {rec['arm']:18s} n={rec['n']:3d} s{rec['seed']}: in {rec['nrmse_test']:.4f} "
                  f"ex {rec['nrmse_extrap']:.4f} far {rec['nrmse_far']:.4f}  {rec['seconds']:.0f}s", flush=True)
            if time.time() - t0 > budget:
                pool.terminate()
                print("budget reached; run again to continue")
                return
    print(f"evaluation complete, {time.time() - t0:.0f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("simulate", "roughness", "roughness_heldout", "wake", "sleep", "heldout",
                                      "evaluate", "report"))
    p.add_argument("cycle", nargs="?", type=int, default=1)
    p.add_argument("--variant", default="")
    a = p.parse_args()
    os.environ["CURRICULUM_VARIANT"] = a.variant          # read by the worker processes too
    if a.stage == "simulate":
        simulate()
    elif a.stage == "roughness":
        roughness()
    elif a.stage == "roughness_heldout":
        roughness_heldout()
    elif a.stage == "wake":
        wake(a.cycle)
    elif a.stage == "sleep":
        sleep(a.cycle)
    elif a.stage == "heldout":
        simulate_heldout()
    elif a.stage == "evaluate":
        evaluate()
    else:
        from regression.claude.curriculum_report import report
        report(out())


if __name__ == "__main__":
    main()
