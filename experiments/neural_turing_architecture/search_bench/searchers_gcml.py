"""Search algorithms that use GCML's inverse model (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §12.3 G3, G4).

They read the task's fitted inverse models from `task.inv` (a dict: "lin" -> ridge W, "mlp" -> state-conditioned W(c))
and the state code from `task.inv_code`. A proposal costs NOTHING beyond the step that evaluates it (a matrix-vector
product), where a value-gradient proposal costs 3 thought-steps -- GCML's case is cost.
  gcml_greedy[_v]       G4, GCML's own planner: z_t = W·Δc + noise, iterated, restarted in parallel batches;
                        Δc = goal − state (goal known) or the value's gradient in code space (`_v`, no goal)
  invnet_greedy[_v]     the same with the state-conditioned inverse network
  mcts_gcml[_v]         the real tree with expansions from the inverse model (3 noised proposals + 1 prior sample)
  mcts_invnet[_v]       the same with the inverse network
  mcts_hybrid_gcml      the hybrid expansion (learned subspace + gradient refinement) plus 2 inverse-model proposals
"""
from __future__ import annotations

import torch

from mock_task import Sim, _unit
from searchers import ALGOS, mcts, exp_iso, exp_hybrid
from gcml import goal_delta, value_delta


def _delta(t, X, use_value):
    return value_delta(t, X, t.inv_code) if use_value else goal_delta(t, X, t.inv_code)


def _propose(t, X, which, use_value, gen, k, sigma):
    z = t.inv[which].propose(t, X, _delta(t, X, use_value))                       # (C, d)
    z = z.repeat_interleave(k, 0)
    return _unit(z + sigma * _unit(torch.randn(z.shape, generator=gen)))


def _greedy(sim: Sim, gen, which, use_value, sigma=0.3, chains=32):
    t, T = sim.task, sim.task.cfg.T
    while sim.budget.left >= T:
        C = max(1, min(chains, sim.budget.left // T))
        X = t.p0.expand(C, -1)
        for _ in range(T):
            X = sim.step(X, _propose(t, X, which, use_value, gen, 1, sigma))


def _exp(which, use_value, k=3, sigma=0.3):
    def expand(sim, X, gen, shared):
        return torch.cat([_propose(sim.task, X, which, use_value, gen, k, sigma), exp_iso(sim, X, gen, shared, k=1)], 0)
    return expand


def exp_hybrid_gcml(sim, X, gen, shared):
    return torch.cat([exp_hybrid(sim, X, gen, shared), _propose(sim.task, X, "lin", False, gen, 2, 0.3)], 0)


for _w, _wn in (("lin", "gcml"), ("mlp", "invnet")):
    for _v, _vn in ((False, ""), (True, "_v")):
        ALGOS[f"{_wn}_greedy{_vn}"] = (lambda w, v: (lambda sim, gen: _greedy(sim, gen, w, v)))(_w, _v)
        ALGOS[f"mcts_{_wn}{_vn}"] = (lambda w, v: (lambda sim, gen: mcts(sim, gen, _exp(w, v))))(_w, _v)
ALGOS["mcts_hybrid_gcml"] = lambda sim, gen: mcts(sim, gen, exp_hybrid_gcml)
