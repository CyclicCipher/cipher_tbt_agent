"""Tier 2: which improvement operator gives a learned policy the most useful targets? (CURRICULUM_LESS_COCONUT_AND_SEARCH.md
§11.6 plan, §16 results.)

One WORLD (a `StructuredWalk` random graph with per-node keys -- §11's mock, the case without shared structure) is
fixed; PROBLEMS (start, target) vary, with the target 1-4 steps away (uniform), horizon 8. A policy network
π(z | state, target) -- a Gaussian over the thought z ∈ R^d -- and a value network V(state, target) ∈ (0, 1) are
LEARNED from scratch; nothing is mocked except the dynamics (exact, as a real network's are) and the verifier.
Every arm spends the same number of thought-steps per training problem (`--budget`; a gradient through a step
costs 3) and the same number of network updates per iteration, and differs only in the improvement operator:

  rloo          S0 -- sample chains from π (Gaussian noise), binary verifier reward, REINFORCE with a leave-one-out
                baseline (Soft Tokens, Hard Truths)
  bestofn       S1 -- sample chains from π, distil the verified ones (expert iteration)
  pi_grad       restarted chains, each thought a policy sample refined by one step along ∇_z V (the learned value)
  pi_mcts       the §11 hybrid tree (learned-subspace samples + gradient refinement + a policy sample) with the policy
                mean as an extra candidate, on the learned value; the solution found is distilled
  bptt          Q3: no search -- the answer's log-likelihood back-propagated through the exact dynamics into π
                (Coconut trained without its curriculum, P1(b)); log of the mean over stopping steps of p_t[target]
  answer_opt    §10.5: the answer's gradient as a TRAINING-TIME search proposal -- Adam on the chain's thoughts toward
                the answer, started at policy samples and held near them by a trust-region penalty; solutions distilled

Value targets for every arm come from the transitions it simulated, by one of two rules (`--vlabel`): `path` -- a
state on a found solution path gets γ^(steps to success), any state the verifier accepts 1, every other simulated
state 0; `bellman` -- each simulated state with simulated successors gets γ·max over them (1 if verified), leaves
bootstrapped from the current V (the tree's own backup as a target). §16 found `path` a confound (a tree explores many
states and expands few, so its value collapses to 0) and reports `bellman`.
Measured on 80 held-out (start, target) pairs (20 per distance, never trained on): the AMORTISED policy (its mean, no
search) and the policy + value inside `pi_mcts` at a test budget; the amortised policy on 80 of its own TRAINING problems
(fitting vs generalising); V at the start state per distance; the training solve rate; all against thought-steps spent.
    python experiments/neural_turing_architecture/search_bench/train_loop.py --d 64 --seed0 0 --seeds 1 --out runs/train_d64_s0.json
    python experiments/neural_turing_architecture/search_bench/train_loop.py --d 64 --seeds 3 --arms pi_mcts pi_grad answer_opt --vlabel bellman --out runs/train_bellman_d64.json
    (§16 ran d ∈ {64, 256}, seeds 0-2, both label rules, in calls short enough for one CPU session each;
    `summarize_train.py` merges them)
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from mock_task import TaskConfig, Budget, Sim, Solved, BudgetExhausted, _unit  # noqa: E402
from structured_task import StructuredWalk  # noqa: E402
from searchers import mcts, exp_hybrid  # noqa: E402

GAMMA = 0.7                                                   # the value target's discount per step (the mock's)


# ── the networks ────────────────────────────────────────────────────────────────────────────────────────────────────────
def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Nets:
    def __init__(self, N, H, d, sigma, seed, lr=1e-3):
        torch.manual_seed(seed)
        self.N, self.d, self.sigma = N, d, sigma
        self.pi, self.v = mlp(2 * N + H, d), mlp(2 * N + H, 1)
        with torch.no_grad():
            self.pi[-1].weight.mul_(0.1), self.pi[-1].bias.zero_()             # untrained policy ≈ isotropic samples
            self.v[-1].bias.fill_(-3.0)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=lr)

    def inp(self, X, tgt):
        return torch.cat([X, torch.nn.functional.one_hot(tgt, self.N).float()], -1)

    def mu(self, X, tgt):
        return self.pi(self.inp(X, tgt))

    def value(self, X, tgt):
        return torch.sigmoid(self.v(self.inp(X, tgt))).squeeze(-1)

    def noise(self, shape, gen):
        return self.sigma * torch.randn(shape, generator=gen) / math.sqrt(self.d)       # ‖noise‖ ≈ sigma


class Learned:
    """A problem seen through the learned networks: the true dynamics and verifier, the learned value and policy in place
    of the mock's -- the interface the §11 searchers use."""

    def __init__(self, base: StructuredWalk, nets: Nets):
        self.base, self.nets, self.cfg, self.p0, self.target = base, nets, base.cfg, base.p0, base.target

    def tg(self, X):
        return torch.full((X.shape[0],), self.target, dtype=torch.long)

    def step(self, X, z):
        return self.base.step(X, z)

    def success(self, X):
        return self.base.success(X)

    def value(self, X):
        if X.requires_grad:
            return self.nets.value(X, self.tg(X))
        with torch.no_grad():
            return self.nets.value(X, self.tg(X))

    @torch.no_grad()
    def prior_mean_of(self, X):
        return self.nets.mu(X, self.tg(X))

    @torch.no_grad()
    def prior_sample(self, X, m, gen, widen=1.0):
        mu = self.nets.mu(X, self.tg(X)).repeat_interleave(m, 0)
        return _unit(mu + widen * self.nets.noise(mu.shape, gen))


# ── recording what a search simulated, and the solution it found ────────────────────────────────────────────────────────
class TraceSim(Sim):
    def __init__(self, task, budget):
        super().__init__(task, budget)
        self.log = []

    def step(self, p, z, check=True):
        self.budget.spend(p.shape[0])
        with torch.no_grad():
            q = self.task.step(p, z)
        self.log.append((p.detach(), _unit(z).detach(), q))
        if check and bool(self.task.success(q).any()):
            raise Solved
        return q

    def value_grad(self, p, z):
        self.budget.spend(3 * p.shape[0])
        z = z.detach().clone().requires_grad_(True)
        q = self.task.step(p.detach(), z)
        self.log.append((p.detach(), _unit(z).detach(), q.detach()))
        if bool(self.task.success(q.detach()).any()):
            raise Solved
        (g,) = torch.autograd.grad(self.task.value(q).sum(), z)
        return g, q.detach()


def _key(x):
    return x.numpy().tobytes()


def solution_path(log, task):
    """Back-track a verified state to the start through the logged transitions (the dynamics are deterministic, so a
    logged state is reproduced bit-for-bit by the transition that made it). Returns [(X_t, z_t)] or None."""
    made, win = {}, None
    for P, Z, Q in log:
        ok = task.success(Q)
        for r in range(P.shape[0]):
            made.setdefault(_key(Q[r]), (P[r], Z[r]))
            if win is None and bool(ok[r]):
                win = (P[r], Z[r])
    if win is None:
        return None
    start, chain, X = _key(task.p0), [win], win[0]
    while _key(X) != start:
        if len(chain) > task.cfg.T or _key(X) not in made:
            return None
        chain.append(made[_key(X)])
        X = chain[-1][0]
    chain.reverse()
    with torch.no_grad():                                                           # replay: must verify
        Xr = task.p0[None]
        for _, z in chain:
            Xr = task.step(Xr, z[None])
        if not bool(task.success(Xr).all()):
            return None
    return chain


def value_labels(task, states, path, cap, gen):
    """γ^(steps to success) on the solution path, 1 for any verified state, 0 otherwise (negatives subsampled)."""
    S = torch.cat(states) if states else task.p0[None]
    y = task.success(S).float()
    pos_X, pos_y = [S[y > 0]], [y[y > 0]]
    if path:
        k = len(path)
        pos_X.append(torch.stack([x for x, _ in path]))
        pos_y.append(torch.tensor([GAMMA ** (k - j) for j in range(k)]))
    neg = S[y == 0]
    if neg.shape[0] > cap:
        neg = neg[torch.randperm(neg.shape[0], generator=gen)[:cap]]
    X = torch.cat(pos_X + [neg])
    return X, torch.cat(pos_y + [torch.zeros(neg.shape[0])])


def bellman_labels(task, trans, cap, gen):
    """The alternative value target (`--vlabel bellman`): for every simulated state that has simulated successors,
    γ·max over them (1 for a verified state), leaves bootstrapped from the current V -- the tree's own backup, used as
    a target. Unlike `value_labels`, an explored state off the solution path is not counted a failure."""
    idx, states, kids = {}, [], {}

    def node(x):
        k = _key(x)
        if k not in idx:
            idx[k] = len(states)
            states.append(x)
        return idx[k]
    for P, Q in trans:
        for r in range(P.shape[0]):
            kids.setdefault(node(P[r]), set()).add(node(Q[r]))
    S = torch.stack(states)
    ok = task.success(S).tolist()
    with torch.no_grad():
        val = task.value(S).tolist()
    val = [1.0 if o else v for o, v in zip(ok, val)]
    parents = [q for q in kids if not ok[q]]
    for _ in range(task.cfg.T):
        for q in parents:
            val[q] = GAMMA * max(val[c] for c in kids[q])
    lab = parents + [i for i, o in enumerate(ok) if o]
    if len(lab) > 2 * cap:
        lab = [lab[i] for i in torch.randperm(len(lab), generator=gen)[:2 * cap].tolist()]
    return S[lab], torch.tensor([val[i] for i in lab])


# ── the improvement operators ───────────────────────────────────────────────────────────────────────────────────────────
def sample_chains(task, G, gen, keep_graph=False):
    """G chains from π; returns states (G, T+1, ·), raw thoughts, means (with graph if asked), first success step."""
    nets, T = task.nets, task.cfg.T
    X = task.p0.expand(G, -1)
    Xs, Zr, Mu = [X], [], []
    first = torch.full((G,), -1)
    for s in range(T):
        with torch.set_grad_enabled(keep_graph):
            mu = nets.mu(X, task.tg(X))
        zr = mu.detach() + nets.noise(mu.shape, gen)
        with torch.no_grad():
            X = task.step(X, zr)
        ok = task.success(X) & (first < 0)
        first[ok] = s + 1
        Xs.append(X), Zr.append(zr), Mu.append(mu)
    return torch.stack(Xs, 1), torch.stack(Zr, 1), torch.stack(Mu, 1), first


def chains_to_data(task, Xs, Zr, first, gen, cap):
    out_paths, states = [], [Xs[:, 1:].reshape(-1, Xs.shape[-1])]
    for g in range(Xs.shape[0]):
        k = int(first[g])
        if k > 0:
            out_paths.append([(Xs[g, j], _unit(Zr[g, j])) for j in range(k)])
    best = min(out_paths, key=len) if out_paths else None
    Xv, yv = value_labels(task, states, best, cap, gen)
    for pth in out_paths:                                                             # every success labels its chain
        if pth is not best:
            k = len(pth)
            Xv = torch.cat([Xv, torch.stack([x for x, _ in pth])])
            yv = torch.cat([yv, torch.tensor([GAMMA ** (k - j) for j in range(k)])])
    return out_paths, Xv, yv


def _chain_trans(Xs):
    return [(Xs[:, :-1].reshape(-1, Xs.shape[-1]), Xs[:, 1:].reshape(-1, Xs.shape[-1]))]


def op_rloo(task, B, gen, cap):
    G = max(2, B // task.cfg.T)
    Xs, Zr, Mu, first = sample_chains(task, G, gen, keep_graph=True)
    R = (first > 0).float()
    A = R - (R.sum() - R) / (G - 1)
    T = task.cfg.T
    alive = (torch.arange(T)[None] < torch.where(first > 0, first, torch.full_like(first, T))[:, None]).float()
    logp = -(task.nets.d / (2 * task.nets.sigma ** 2)) * ((Zr - Mu) ** 2).sum(-1)          # (G, T)
    loss = -(A[:, None] * logp * alive).sum() / G
    paths, Xv, yv = chains_to_data(task, Xs, Zr, first, gen, cap)
    return dict(pg_loss=loss, distil=[], Xv=Xv, yv=yv, solved=bool(R.any()), trans=_chain_trans(Xs))


def op_bestofn(task, B, gen, cap):
    Xs, Zr, _, first = sample_chains(task, max(1, B // task.cfg.T), gen)
    paths, Xv, yv = chains_to_data(task, Xs, Zr, first, gen, cap)
    return dict(distil=[pr for p in paths for pr in p], Xv=Xv, yv=yv, solved=bool(paths), trans=_chain_trans(Xs))


def _answer_loss(task, Z, Xstart=None):
    """−log of the mean over stopping steps of p_t[target] through the exact dynamics; Z (R, T, d) raw thoughts."""
    X = task.p0.expand(Z.shape[0], -1) if Xstart is None else Xstart
    ps, Xs = [], []
    for s in range(Z.shape[1]):
        X = task.step(X, Z[:, s])
        ps.append(X[:, task.target]), Xs.append(X)
    return -torch.log(torch.stack(ps, 1).mean(1).clamp_min(1e-30)), torch.stack(Xs, 1)


def op_bptt(task, B, gen, cap):
    nets, T = task.nets, task.cfg.T
    G = max(1, B // (3 * T))
    X = task.p0.expand(G, -1)
    ps, Xs, Zs = [], [X.detach()], []
    for _ in range(T):
        z = nets.mu(X, task.tg(X)) + nets.noise((G, nets.d), gen)
        X = task.step(X, z)
        ps.append(X[:, task.target]), Xs.append(X.detach()), Zs.append(z.detach())
    loss = -torch.log(torch.stack(ps, 1).mean(1).clamp_min(1e-30)).mean()
    Xs, Zr = torch.stack(Xs, 1), torch.stack(Zs, 1)
    ok = task.success(Xs[:, 1:].reshape(-1, Xs.shape[-1])).reshape(G, T)
    first = torch.where(ok.any(1), ok.float().argmax(1) + 1, torch.full((G,), -1))
    paths, Xv, yv = chains_to_data(task, Xs, Zr, first, gen, cap)
    return dict(pg_loss=loss, distil=[], Xv=Xv, yv=yv, solved=bool(paths), trans=_chain_trans(Xs))


def op_answer_opt(task, B, gen, cap, R=2, lr=0.3, trust=1.0, iters=12):
    """Trust region: a penalty trust·Σ_t ‖z_t − z_t⁰‖² keeps the optimised chain near the policy's own sample z⁰."""
    nets, T, d = task.nets, task.cfg.T, task.nets.d
    Xs_all, found, trans = [], None, []
    while B >= 4 * R * T and found is None:
        Xs0, Z0, _, first = sample_chains(task, R, gen)
        B -= R * T                                                                    # the initial chains' steps
        Xs_all.append(Xs0[:, 1:].reshape(-1, Xs0.shape[-1]))
        trans += _chain_trans(Xs0)
        if bool((first > 0).any()):                                                  # a policy sample already solves
            r = int((first > 0).float().argmax())
            found = [(Xs0[r, j], _unit(Z0[r, j])) for j in range(int(first[r]))]
            break
        Z = Z0.clone().requires_grad_(True)
        opt = torch.optim.Adam([Z], lr=lr / math.sqrt(d))
        for _ in range(iters):
            if B < 3 * R * T:
                break
            B -= 3 * R * T
            loss, Xs = _answer_loss(task, Z)
            Xd = Xs.detach()
            Xs_all.append(Xd.reshape(-1, Xd.shape[-1]))
            trans += _chain_trans(torch.cat([task.p0.expand(R, 1, -1), Xd], 1))
            ok = task.success(Xd.reshape(-1, Xd.shape[-1])).reshape(R, T)
            if bool(ok.any()):
                r = int(ok.any(1).float().argmax())
                k = int(ok[r].float().argmax()) + 1
                Xp = torch.cat([task.p0[None], Xd[r, :k - 1]])
                found = [(Xp[j], _unit(Z[r, j].detach())) for j in range(k)]
                break
            pen = trust * ((Z - Z0) ** 2).sum((1, 2))
            opt.zero_grad()
            (loss + pen).sum().backward()
            opt.step()
    Xv, yv = value_labels(task, Xs_all, found, cap, gen)
    return dict(distil=found or [], Xv=Xv, yv=yv, solved=found is not None, trans=trans)


def _search(task, B, gen, cap, algo):
    sim = TraceSim(task, Budget(B))
    try:
        algo(sim, gen)
    except (Solved, BudgetExhausted):
        pass
    path = solution_path(sim.log, task)
    Xv, yv = value_labels(task, [Q for _, _, Q in sim.log], path, cap, gen)
    return dict(distil=path or [], Xv=Xv, yv=yv, solved=path is not None, trans=[(P, Q) for P, _, Q in sim.log])


def algo_pi_grad(sim, gen, alpha=1.0, chains=4):
    t, T = sim.task, sim.task.cfg.T
    while sim.budget.left >= 4 * T:
        C = max(1, min(chains, sim.budget.left // (4 * T)))
        X = t.p0.expand(C, -1)
        for _ in range(T):
            z0 = t.prior_sample(X, 1, gen)
            g, _ = sim.value_grad(X, z0)
            X = sim.step(X, _unit(z0 + alpha * _unit(g)))


def exp_pi_hybrid(sim, X, gen, shared):
    return torch.cat([exp_hybrid(sim, X, gen, shared), _unit(sim.task.prior_mean_of(X))], 0)


def algo_pi_mcts(sim, gen):
    mcts(sim, gen, exp_pi_hybrid)


OPS = {"rloo": op_rloo, "bestofn": op_bestofn, "bptt": op_bptt, "answer_opt": op_answer_opt,
       "pi_grad": lambda t, B, g, c: _search(t, B, g, c, algo_pi_grad),
       "pi_mcts": lambda t, B, g, c: _search(t, B, g, c, algo_pi_mcts)}


# ── evaluation ──────────────────────────────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def amortised(task):
    X = task.p0[None]
    for _ in range(task.cfg.T):
        X = task.step(X, task.nets.mu(X, task.tg(X)))
        if bool(task.success(X).any()):
            return True
    return False


def searched(task, B, seed):
    sim = Sim(task, Budget(B))
    try:
        algo_pi_mcts(sim, torch.Generator().manual_seed(seed))
    except Solved:
        return True
    except BudgetExhausted:
        pass
    return False


# ── one run ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def run(a):
    arm, d, seed, o = a
    torch.set_num_threads(1)
    world = seed
    cfgs = {L: TaskConfig(N=64, k=4, d=d, beta=16.0, gate=0.3, L=L, T=8) for L in (1, 2, 3, 4)}

    def make(ps, L):
        return StructuredWalk(cfgs[L], world, ps, graph="random", keys="node")

    evalset = [(L, 10 ** 6 + 100 * L + j) for L in (1, 2, 3, 4) for j in range(o["n_eval"])]
    held = {(make(ps, L).start, make(ps, L).target) for L, ps in evalset}
    base0 = make(0, 1)
    nets = Nets(64, base0.H, d, o["sigma"], seed=1000 + seed)
    gen = torch.Generator().manual_seed(seed)
    buf_d, buf_v = [], []                                    # replay: (X, target, z) and (X, target, y)
    log, ps_next, steps, solved_hist, path_lens = [], 0, 0, [], []

    def evaluate(it):
        res = {}
        for L in (1, 2, 3, 4):
            probs = [Learned(make(ps, L), nets) for LL, ps in evalset if LL == L]
            seen = [Learned(make(ps, LL), nets) for LL, ps in trainset if LL == L]
            with torch.no_grad():
                v0 = float(torch.stack([t.value(t.p0[None])[0] for t in probs]).mean())
            res[L] = dict(amortised=sum(map(amortised, probs)) / len(probs),
                          amortised_train=sum(map(amortised, seen)) / max(1, len(seen)),
                          searched=sum(searched(t, o["eval_budget"], 7 * j + L) for j, t in enumerate(probs)) / len(probs),
                          v_start=v0)
        recent = solved_hist[-o["batch"] * 5:]
        fit = None
        if buf_d:                                            # how well the policy reproduces its distilled targets
            with torch.no_grad():
                idx = torch.randperm(len(buf_d))[:512]
                X = torch.stack([buf_d[i][0] for i in idx])
                tg = torch.tensor([buf_d[i][1] for i in idx])
                z = torch.stack([buf_d[i][2] for i in idx])
                fit = float(torch.nn.functional.cosine_similarity(nets.mu(X, tg), z, dim=1).mean())
        log.append(dict(iter=it, thought_steps=steps, train_solved=sum(recent) / max(1, len(recent)), eval=res,
                        distil_fit=fit, path_len=sum(path_lens[-80:]) / max(1, len(path_lens[-80:]))))

    trainset = []                                            # the first training problems: is the policy FITTING?
    t0 = time.time()
    evaluate(0)
    per_step = max(1, o["batch"] // o["updates"])
    for it in range(1, o["iters"] + 1):
        pg = []
        for _ in range(o["batch"]):
            while True:                                      # a training problem: uniform distance, not a held-out pair
                L = 1 + int(torch.randint(4, (1,), generator=gen))
                base = make(ps_next, L)
                ps_next += 1
                if (base.start, base.target) not in held:
                    break
            if len(trainset) < 4 * o["n_eval"]:
                trainset.append((L, ps_next - 1))
            task = Learned(base, nets)
            out = OPS[arm](task, o["budget"], gen, o["neg_cap"])
            if o["vlabel"] == "bellman":
                out["Xv"], out["yv"] = bellman_labels(task, out["trans"], o["neg_cap"], gen)
            steps += o["budget"]
            solved_hist.append(out["solved"])
            if out["distil"]:
                path_lens.append(len(out["distil"]))
            if "pg_loss" in out:                             # on-policy arms (rloo, bptt): a step every few problems,
                pg.append(out["pg_loss"])                    # so they get as many updates as the distilling arms
                if len(pg) == per_step:
                    nets.opt_pi.zero_grad()
                    torch.stack(pg).mean().backward()
                    torch.nn.utils.clip_grad_norm_(nets.pi.parameters(), 1.0)
                    nets.opt_pi.step()
                    pg = []
            for X, z in out["distil"]:
                buf_d.append((X, base.target, z))
            buf_v.append((out["Xv"], torch.full((out["Xv"].shape[0],), base.target), out["yv"]))
        del buf_d[:-o["buf_distil"]]
        del buf_v[:-o["buf_value"]]
        if buf_d:                                            # distillation of found solutions
            for _ in range(o["updates"]):
                idx = torch.randint(len(buf_d), (min(256, len(buf_d)),), generator=gen)
                X = torch.stack([buf_d[i][0] for i in idx])
                tg = torch.tensor([buf_d[i][1] for i in idx])
                z = torch.stack([buf_d[i][2] for i in idx])
                loss = ((nets.mu(X, tg) - z) ** 2).sum(-1).mean()
                nets.opt_pi.zero_grad()
                loss.backward()
                nets.opt_pi.step()
        Xv = torch.cat([b[0] for b in buf_v])
        tv = torch.cat([b[1] for b in buf_v])
        yv = torch.cat([b[2] for b in buf_v])
        for _ in range(o["updates"]):                        # the value, on every arm's own simulated states
            idx = torch.randint(Xv.shape[0], (min(256, Xv.shape[0]),), generator=gen)
            loss = torch.nn.functional.binary_cross_entropy(nets.value(Xv[idx], tv[idx]), yv[idx])
            nets.opt_v.zero_grad()
            loss.backward()
            nets.opt_v.step()
        if it % o["eval_every"] == 0:
            evaluate(it)
    return dict(arm=arm, d=d, seed=seed, secs=round(time.time() - t0, 1), log=log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(OPS))
    ap.add_argument("--d", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--eval_budget", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--updates", type=int, default=8)
    ap.add_argument("--neg_cap", type=int, default=64)
    ap.add_argument("--vlabel", choices=["path", "bellman"], default="path")
    ap.add_argument("--buf_distil", type=int, default=4096)
    ap.add_argument("--buf_value", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    o = {k: v for k, v in vars(a).items() if k not in ("arms", "d", "seeds", "seed0", "workers", "out")}
    jobs = [(arm, d, s, o) for d in a.d for arm in a.arms for s in range(a.seed0, a.seed0 + a.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        outs = list(ex.map(run, jobs))
    print(f"{len(jobs)} runs in {time.time() - t0:.0f}s")
    for r in outs:
        last = r["log"][-1]
        print(r["arm"], r["d"], r["seed"], f"{r['secs']}s", "train", round(last["train_solved"], 2),
              {L: (e["amortised"], e["searched"]) for L, e in last["eval"].items()})
    if a.out:
        p = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(args=vars(a), runs=outs), open(p, "w"), indent=0)
        print(f"-> {p}")


if __name__ == "__main__":
    main()
