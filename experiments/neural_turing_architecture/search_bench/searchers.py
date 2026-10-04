"""The search algorithms compared on the mock task (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §10–§11).

Every algorithm is `fn(sim, gen) -> None`: it searches from the task's start state until it has spent its budget
(`BudgetExhausted`) or a simulated state satisfies the verifier (`Solved`, raised by the Sim). Same simulator, same
mocked value, same prior, same budget for all.

Whole-chain (open-loop) methods
  random_shooting   chains sampled from the prior (best-of-N over whole chains)
  cem               cross-entropy method over the T × d chain (TD-MPC2-style population planning, D6)
  grad_trajopt      gradient ascent on Σ_t V(p_t) through the exact dynamics, several restarts (trajectory optimisation)
  smc / smc_grad    sequential Monte Carlo over chains, twisted by V; proposals from the prior, or prior + a value-gradient
                    step (D5)
  grad_greedy       one normalised value-gradient step per thought, no search (the floor for "gradients alone")
One-step lookahead, receding horizon (the v0/v1 skeleton as first written): at each real thought, K candidates
(16–128, widening with budget), children scored by V, Sequential Halving with short prior rollouts on the survivors,
execute the winner, never backtrack (restart from the start if a pass fails). Candidate generators:
  look_iso          EZ-V2 as published: prior samples + a widened prior (D0)
  look_grad         + candidates along ∇_z V(step(p, z)) at several step sizes (D1)
  look_langevin     + short noisy gradient-ascent chains (D2)
  look_guided       + samples in a subspace learned from the value gradients seen so far (D3, Guided-ES style)
  look_randsub      + samples in a FIXED random 8-dim subspace — a search confined to a manifold it did not learn (D4's risk)
  look_oracle       + samples in the true effective subspace (an upper reference, not an algorithm)
A real tree (MCTS-style best-first): nodes keep their state; selection by best-reachable value + a UCT bonus;
progressive widening adds children to a node as it is visited (ceil(√(N+1))); each expansion adds a few children from
a generator; the backup is the discounted best value below (the dynamics are deterministic); it backtracks freely.
  mcts_iso          expansion by prior samples + one widened sample (EZ-V2's candidates in a real tree)
  mcts_grad         expansion along one value gradient (two step sizes and the direction) + one prior sample
  mcts_langevin     expansion by two short noisy gradient-ascent chains + one prior sample
  mcts_guided       expansion in the subspace spanned by the tree's recent value gradients + the gradient + one sample
  mcts_hybrid       guided subspace samples, each refined by two noisy gradient steps (direction + precision)
"""
from __future__ import annotations

import math

import torch

from mock_task import Sim, Solved, _unit

ALGOS = {}


def algo(name):
    def deco(fn):
        ALGOS[name] = fn
        return fn
    return deco


def _expand(p, m):
    return p.expand(m, -1) if p.shape[0] == 1 else p.repeat_interleave(m, 0)


# ── whole-chain methods ─────────────────────────────────────────────────────────────────────────────────────────────────
@algo("random_shooting")
def random_shooting(sim: Sim, gen):
    t, T = sim.task, sim.task.cfg.T
    while sim.budget.left >= T:
        C = min(256, sim.budget.left // T)
        p = t.p0.expand(C, -1)
        for _ in range(T):
            p = sim.step(p, t.prior_sample(p, 1, gen))


@algo("cem")
def cem(sim: Sim, gen, elite_frac=0.125):
    t, T, d = sim.task, sim.task.cfg.T, sim.task.cfg.d
    M = 64 if sim.budget.total >= 2048 else 32
    E = max(2, int(M * elite_frac))
    mean, std = torch.zeros(T, d), torch.ones(T, d)
    while sim.budget.left >= M * T:
        Z = mean + std * torch.randn(M, T, d, generator=gen)
        p, score = t.p0.expand(M, -1), torch.full((M,), -math.inf)
        for s in range(T):
            p = sim.step(p, Z[:, s])
            score = torch.maximum(score, t.value(p))
        el = Z[score.topk(E).indices]
        mean = 0.3 * mean + 0.7 * el.mean(0)
        std = (0.3 * std + 0.7 * el.std(0)).clamp_min(0.05)


@algo("grad_trajopt")
def grad_trajopt(sim: Sim, gen, R=2):
    t, T, d = sim.task, sim.task.cfg.T, sim.task.cfg.d
    while sim.budget.left >= 3 * R * T:                       # restart blocks of R chains
        Z = (torch.randn(R, T, d, generator=gen) / math.sqrt(d)).requires_grad_(True)
        opt = torch.optim.Adam([Z], lr=0.3 / math.sqrt(d))
        for _it in range(40):
            if sim.budget.left < 3 * R * T:
                return
            sim.budget.spend(3 * R * T)
            p, total = t.p0.expand(R, -1), 0.0
            for s in range(T):
                p = t.step(p, Z[:, s])
                if bool(t.success(p.detach()).any()):
                    raise Solved
                total = total + t.value(p)
            opt.zero_grad()
            (-total.sum()).backward()
            opt.step()


def _smc(sim: Sim, gen, grad: bool, tau=0.05, alpha=2.0):
    t, T = sim.task, sim.task.cfg.T
    per = 4 if grad else 1
    Np = max(8, sim.budget.total // (T * per))
    while sim.budget.left >= Np * per:
        p = t.p0.expand(Np, -1).clone()
        for _ in range(T):
            z = t.prior_sample(p, 1, gen)
            if grad:
                g, _ = sim.value_grad(p, z)
                z = _unit(z + alpha * _unit(g))
            v0 = t.value(p)
            p = sim.step(p, z)
            w = (t.value(p) - v0) / tau
            idx = torch.multinomial((w - w.max()).exp(), Np, replacement=True, generator=gen)
            p = p[idx]


@algo("smc")
def smc(sim, gen):
    _smc(sim, gen, grad=False)


@algo("smc_grad")
def smc_grad(sim, gen):
    _smc(sim, gen, grad=True)


@algo("grad_greedy")
def grad_greedy(sim: Sim, gen):
    t, T, d = sim.task, sim.task.cfg.T, sim.task.cfg.d
    while sim.budget.left >= 4 * T:
        p = t.p0[None]
        for _ in range(T):
            m = t.prior_mean_of(p)
            base = m if m is not None else _unit(torch.randn(1, d, generator=gen))
            g, _ = sim.value_grad(p, base)
            p = sim.step(p, _unit(g))


# ── receding-horizon tree search: the v0/v1 skeleton with pluggable candidate generators ────────────────────────────────
def _base_points(t, p, gen, state):
    m = t.prior_mean_of(p)
    pts = [m if m is not None else _unit(torch.randn(1, t.cfg.d, generator=gen))]
    if "winner" in state:
        pts.append(state["winner"][None])
    return torch.cat(pts, 0)


def prop_iso(sim, p, K, gen, state):
    t = sim.task
    return torch.cat([t.prior_sample(p, K - 4, gen), t.prior_sample(p, 4, gen, widen=4.0)], 0)


def prop_grad(sim, p, K, gen, state):
    t = sim.task
    base = _base_points(t, p, gen, state)
    g, _ = sim.value_grad(_expand(p, base.shape[0]), base)
    gh = _unit(g)
    grad_c = torch.cat([_unit(base + eta * gh) for eta in (0.5, 1.0, 2.0)] + [gh], 0)
    return torch.cat([grad_c, prop_iso(sim, p, max(4, K - grad_c.shape[0]), gen, state)], 0)


def prop_langevin(sim, p, K, gen, state, steps=3, alpha=1.0, sigma=0.3):
    t = sim.task
    k = K // 2
    z = t.prior_sample(p, k, gen)
    for _ in range(steps):
        g, _ = sim.value_grad(_expand(p, k), z)
        z = _unit(z + alpha * _unit(g) + sigma * _unit(torch.randn(z.shape, generator=gen)))
    return torch.cat([z, prop_iso(sim, p, K - k, gen, state)], 0)


def _subspace_samples(U, n, gen, scale=1.0):
    a = torch.randn(n, U.shape[1], generator=gen)
    return _unit(a @ U.T * scale)


def prop_guided(sim, p, K, gen, state, m=8):
    t = sim.task
    base = _base_points(t, p, gen, state)
    g, _ = sim.value_grad(_expand(p, base.shape[0]), base)
    buf = state.setdefault("dirs", [])
    buf.extend(list(_unit(g)))
    if "winner" in state:
        buf.append(state["winner"])
    del buf[:-m]
    U = torch.linalg.qr(torch.stack(buf, 1))[0]                                  # (d, ≤m) orthonormal
    sub = _subspace_samples(U, K - 6, gen)
    return torch.cat([sub, _unit(g), prop_iso(sim, p, 4, gen, state)], 0)


def prop_oracle(sim, p, K, gen, state):
    sub = _subspace_samples(sim.task.P.T, K - 4, gen)
    return torch.cat([sub, prop_iso(sim, p, 4, gen, state)], 0)


def prop_randsub(sim, p, K, gen, state, m=8):
    if "R" not in state:
        state["R"] = torch.linalg.qr(torch.randn(sim.task.cfg.d, m, generator=gen))[0]
    sub = _subspace_samples(state["R"], K - 4, gen)
    return torch.cat([sub, prop_iso(sim, p, 4, gen, state)], 0)


def tree(sim: Sim, gen, proposal, K=16, h_roll=2):
    t, T = sim.task, sim.task.cfg.T
    while sim.budget.left > 0:                                                    # restart from the start if a pass fails
        p, state = t.p0[None], {}
        for i in range(T):
            steps_left = T - i
            b_step = max(K + 8, sim.budget.left // steps_left)
            b_end = sim.budget.used + b_step
            Ki = int(min(128, max(K, b_step // 8)))                 # widen with budget, as more simulations would
            cand = proposal(sim, p, Ki, gen, state)
            children = sim.step(_expand(p, cand.shape[0]), cand)
            score = t.value(children).clone()
            alive = torch.arange(cand.shape[0])
            h = min(h_roll, steps_left - 1)
            rounds = max(1, math.ceil(math.log2(cand.shape[0])))
            if h > 0:
                for r in range(rounds):
                    if alive.numel() <= 1:
                        break
                    alive = alive[score[alive].topk(max(1, alive.numel() // 2)).indices]
                    room = (b_end - sim.budget.used) // max(1, rounds - r)
                    reps = room // (alive.numel() * h)
                    if reps < 1:
                        continue
                    q = children[alive].repeat_interleave(reps, 0)
                    best = t.value(q)
                    for _ in range(h):
                        q = sim.step(q, t.prior_sample(q, 1, gen))
                        best = torch.maximum(best, t.value(q))
                    score[alive] = 0.5 * score[alive] + 0.5 * best.view(alive.numel(), reps).mean(1)
            w = int(alive[score[alive].argmax()])
            state["winner"] = cand[w]
            p = children[w:w + 1]


for _name, _prop in [("look_iso", prop_iso), ("look_grad", prop_grad), ("look_langevin", prop_langevin),
                     ("look_guided", prop_guided), ("look_randsub", prop_randsub), ("look_oracle", prop_oracle)]:
    ALGOS[_name] = (lambda prop: (lambda sim, gen: tree(sim, gen, prop)))(_prop)


# ── a real tree: MCTS-style best-first search with progressive widening and backtracking ─────────────────────────────────
class _Node:
    __slots__ = ("X", "v", "best", "N", "children", "depth", "parent")

    def __init__(self, X, v, depth, parent):
        self.X, self.v, self.best, self.N = X, v, v, 0
        self.children, self.depth, self.parent = [], depth, parent


def exp_iso(sim, X, gen, shared, k=4):
    t = sim.task
    return torch.cat([t.prior_sample(X, k - 1, gen), t.prior_sample(X, 1, gen, widen=4.0)], 0)


def _exp_base(t, X, gen):
    m = t.prior_mean_of(X)
    noise = _unit(torch.randn(1, t.cfg.d, generator=gen))
    return noise if m is None else _unit(m + 0.5 * noise)


def exp_grad(sim, X, gen, shared):
    t = sim.task
    base = _exp_base(t, X, gen)
    g, _ = sim.value_grad(X, base)
    gh = _unit(g)
    shared.setdefault("dirs", []).append(gh[0])
    return torch.cat([_unit(base + 1.0 * gh), _unit(base + 2.0 * gh), gh, exp_iso(sim, X, gen, shared, k=1)], 0)


def exp_langevin(sim, X, gen, shared, chains=2, steps=3, alpha=1.0, sigma=0.3):
    t = sim.task
    z = torch.cat([_exp_base(t, X, gen) for _ in range(chains)], 0)
    for _ in range(steps):
        g, _ = sim.value_grad(_expand(X, chains), z)
        z = _unit(z + alpha * _unit(g) + sigma * _unit(torch.randn(z.shape, generator=gen)))
    shared.setdefault("dirs", []).extend(list(_unit(g)))
    return torch.cat([z, exp_iso(sim, X, gen, shared, k=1)], 0)


def exp_guided(sim, X, gen, shared, m=8):
    t = sim.task
    base = _exp_base(t, X, gen)
    g, _ = sim.value_grad(X, base)
    buf = shared.setdefault("dirs", [])
    buf.append(_unit(g)[0])
    del buf[:-m]
    U = torch.linalg.qr(torch.stack(buf, 1))[0]
    return torch.cat([_subspace_samples(U, 3, gen), _unit(g), exp_iso(sim, X, gen, shared, k=1)], 0)


def mcts(sim: Sim, gen, expand, c_uct=0.3, pw_c=1.0, pw_a=0.5, gamma=0.9):
    t, T = sim.task, sim.task.cfg.T
    root = _Node(t.p0[None], float(t.value(t.p0[None])), 0, None)
    shared = {}
    while True:
        node = root
        while node.depth < T:                                             # selection with progressive widening
            if len(node.children) < math.ceil(pw_c * (node.N + 1) ** pw_a):
                break
            lg = math.log(node.N + 1)
            node = max(node.children, key=lambda ch: ch.best + c_uct * math.sqrt(lg / (ch.N + 1)))
        if node.depth >= T:                                               # horizon reached without success: a dead end
            node.best = -1.0
        else:
            cand = expand(sim, node.X, gen, shared)
            kids = sim.step(_expand(node.X, cand.shape[0]), cand)         # raises Solved on success
            vals = t.value(kids)
            for j in range(cand.shape[0]):
                node.children.append(_Node(kids[j:j + 1], float(vals[j]), node.depth + 1, node))
        while node is not None:                                           # backup: best reachable value, discounted
            node.N += 1
            if node.children:
                node.best = max(node.v if node.depth < T else -1.0, gamma * max(ch.best for ch in node.children))
            node = node.parent


def exp_hybrid(sim, X, gen, shared, m=8, k=2, steps=2, alpha=1.0, sigma=0.3):
    """Direction from the learned subspace (guided), precision from short noisy gradient refinement (Langevin)."""
    t = sim.task
    base = _exp_base(t, X, gen)
    g, _ = sim.value_grad(X, base)
    buf = shared.setdefault("dirs", [])
    buf.append(_unit(g)[0])
    del buf[:-m]
    U = torch.linalg.qr(torch.stack(buf, 1))[0]
    z = _subspace_samples(U, k, gen)
    for _ in range(steps):
        gz, _ = sim.value_grad(_expand(X, k), z)
        z = _unit(z + alpha * _unit(gz) + sigma * _unit(torch.randn(z.shape, generator=gen)))
    return torch.cat([z, _unit(g), exp_iso(sim, X, gen, shared, k=1)], 0)


for _name, _exp in [("mcts_iso", exp_iso), ("mcts_grad", exp_grad), ("mcts_langevin", exp_langevin),
                    ("mcts_guided", exp_guided), ("mcts_hybrid", exp_hybrid)]:
    ALGOS[_name] = (lambda ex: (lambda sim, gen: mcts(sim, gen, ex)))(_exp)
