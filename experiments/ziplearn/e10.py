"""E10 — acting to learn (DESIGN.md §16, OPEN-11): is "the plan bits an action's description would save" a good reason
to try that action?

The E9 environment (7 actions over 6-digit sequences, V = 5). The agent starts knowing 4 actions (3 observations each:
rot_left, swap_pairs, inc, negate) and never having seen 3 (reverse, rot_right, swap_halves). It faces a stream of
goals, each reachable in <= 3 actions using ALL seven; some goals cannot be reached with the known four at all, others
only by long detours. Before planning each goal the agent may spend up to T TRIES: execute an action to see what it
does (which also moves it). Every executed action, try or plan step, is an observation that updates that action's
description. Success = the goal is reached within a step budget (tries + plan steps). Four policies for the tries:
  none            no tries (E9 with partial knowledge);
  random          T random actions;
  least observed  the T actions with the fewest observations (count-based novelty);
  price           the action with the largest EXPECTED PLAN-BIT SAVING for the current goal -- over the hypotheses its
                  partial description still allows (sampled), what would the best plan cost if the action worked
                  that way, against the best plan without it -- and only if the saving exceeds the try's own cost.
Measures (pre-registered): goals reached within the budget; mean steps to goal (failures count as the budget); goals
until every action is described. Pass: price reaches at least as many goals as random and least-observed with fewer
mean steps; refute: price is worse than random.

    python experiments/ziplearn/e10.py           (CPU, ~1-2 min) -> runs/e10/e10.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "transformers"))
from ziplearner import TwoLayer, PositionPerm   # noqa: E402
import h1_lid as H                              # noqa: E402

L, V = H.L, H.V
ACTIONS = list(H.NAMES)
KNOWN = ["rot_left", "swap_pairs", "inc", "negate"]
BITS = math.log2(len(ACTIONS))                  # one action's name; also the price of a try


def true_apply(name, s):
    return tuple(int(v) for v in H.PRIMS[name](torch.tensor(s)).tolist())


def hyp_apply(pi, a, b, s):
    return tuple((a * s[pi[j]] + b) % V for j in range(L))


class Agent:
    def __init__(self, rng):
        self.rng = rng
        self.models = {a: TwoLayer(L, V) for a in ACTIONS}
        self.n_obs = {a: 0 for a in ACTIONS}

    def observe(self, a, s, s2):
        self.models[a].observe_demo(np.array(s), np.array(s2))
        self.n_obs[a] += 1

    def effect(self, a, s):
        if self.n_obs[a] == 0:                                          # never seen: no description at all -- the
            return None                                                 # cheapest structure would be "identity",
        out = self.models[a].predict_seq(np.array(s))                   # which is a guess with zero evidence
        return None if any(v is None for v in out) else tuple(int(v) for v in out)

    def resolved(self, a):
        return self.n_obs[a] > 0 and all(self.effect(a, tuple(int(v) for v in self.rng.integers(0, V, L))) is not None
                                         for _ in range(2))

    def hypotheses(self, a, M=8):
        """What action `a` might do, given what has been seen of it: (pi, a, b) triples. Fully described: one.
        Partly: completions of the candidate sources of the runs within 2 bits of the best, sampled. Never seen:
        random permutations with the identity value map."""
        learner = self.models[a]
        if self.n_obs[a] == 0:
            return [(tuple(self.rng.permutation(L)), 1, 0) for _ in range(M)]
        best = learner.best()
        best_price = min(bits + learner.bits_value_name + pos.cost + learner.bits_pos_name
                         for (_n, _ab, bits), *positions in learner.runs for pos in positions)
        hyps = []
        for (name, (va, vb), bits), *positions in learner.runs:
            for pos in positions:
                price = bits + learner.bits_value_name + pos.cost + learner.bits_pos_name
                if price > best_price + 2.0:
                    continue
                if isinstance(pos, PositionPerm):
                    if pos.dead:
                        continue
                    cands = [sorted(c) for c in pos.cand]
                    # random completions that are bijections
                    for _ in range(4 * M):
                        pi, used = [], set()
                        for j in range(L):
                            opts = [i for i in cands[j] if i not in used]
                            if not opts:
                                break
                            pick = opts[int(self.rng.integers(len(opts)))]
                            pi.append(pick)
                            used.add(pick)
                        if len(pi) == L:
                            hyps.append((tuple(pi), va, vb))
                else:
                    hyps.append((tuple(range(L)), va, vb))
        hyps = list(dict.fromkeys(hyps))
        if len(hyps) > M:
            idx = self.rng.choice(len(hyps), M, replace=False)
            hyps = [hyps[i] for i in idx]
        return hyps or [(tuple(self.rng.permutation(L)), 1, 0) for _ in range(M)]


def plan(apply_fn, start, goal, max_depth):
    if start == goal:
        return []
    seen = {start: []}
    frontier = deque([start])
    while frontier:
        s = frontier.popleft()
        if len(seen[s]) >= max_depth:
            continue
        for a in ACTIONS:
            n = apply_fn(a, s)
            if n is None or n in seen:
                continue
            seen[n] = seen[s] + [a]
            if n == goal:
                return seen[n]
            frontier.append(n)
    return None


def plan_bits(agent, s, g, override=None, depth=3):
    """Bits of the best plan under the agent's descriptions, with one action's effect optionally overridden by a
    hypothesis; a large constant if no plan exists."""
    def apply_fn(a, st):
        if override and a == override[0]:
            pi, va, vb = override[1]
            return hyp_apply(pi, va, vb, st)
        return agent.effect(a, st)
    w = plan(apply_fn, s, g, depth)
    return (len(w) * BITS) if w is not None else (depth + 2) * BITS


def choose_try(policy, agent, s, g, rng, tries_left, recent_goals=(), horizon=1):
    unknown = [a for a in ACTIONS if not agent.resolved(a)]
    if policy == "none" or tries_left == 0:
        return None, None
    if policy == "random":
        return ACTIONS[int(rng.integers(len(ACTIONS)))], None
    if policy == "least observed":
        return min(ACTIONS, key=lambda a: (agent.n_obs[a], rng.random())), None
    if policy == "least observed (until known)":
        return (min(unknown, key=lambda a: (agent.n_obs[a], rng.random())) if unknown else None), None
    # price: expected plan-bit saving over the action's remaining hypotheses, minus the try's cost.
    #   "price (this goal)"  -- the saving on the current goal only (myopic);
    #   "price (horizon)"    -- the saving per goal, averaged over a sample of goals seen so far, times the goals
    #                           still to come: a description is an asset that pays on every plan that uses it.
    if not unknown:
        return None, None
    sample = [(s, g)] if policy == "price (this goal)" else [(s, g)] + list(recent_goals)
    H = 1 if policy == "price (this goal)" else max(1, horizon)
    best, best_gain = None, 0.0
    gains = {}
    for a in unknown:
        hyps = agent.hypotheses(a, M=6)
        per_goal = []
        for (ss, gg) in sample:
            base = plan_bits(agent, ss, gg)
            per_goal.append(float(np.mean([max(0.0, base - plan_bits(agent, ss, gg, (a, h))) for h in hyps])))
        gains[a] = float(np.mean(per_goal)) * H - BITS                  # the try itself costs one action
        if gains[a] > best_gain:
            best, best_gain = a, gains[a]
    return best, gains


def run_stream(policy, goals, T, budget, seed):
    rng = np.random.default_rng(seed)
    agent = Agent(rng)
    for a in KNOWN:                                                     # the known actions: 3 observations each
        for _ in range(3):
            s = tuple(int(v) for v in rng.integers(0, V, L))
            agent.observe(a, s, true_apply(a, s))
    reached, steps_all, first_all_known = 0, [], None
    tries_spent, tries_on_unknown = 0, 0
    for i, (start, goal) in enumerate(goals):
        s, steps, tries = start, 0, 0
        recent = goals[max(0, i - 3):i]                                  # the goals seen just before this one
        while steps < budget:
            a, _g = choose_try(policy, agent, s, goal, rng, T - tries, recent, len(goals) - i)
            if a is not None:
                s2 = true_apply(a, s)
                was_unknown = not agent.resolved(a)
                agent.observe(a, s, s2)
                tries_spent += 1
                tries_on_unknown += int(was_unknown)
                s, steps, tries = s2, steps + 1, tries + 1
                if s == goal:
                    break
                continue
            w = plan(lambda act, st: agent.effect(act, st), s, goal, 4)
            if w is None:
                break
            for act in w:
                s2 = true_apply(act, s)
                agent.observe(act, s, s2)
                s, steps = s2, steps + 1
                if s == goal or steps >= budget:
                    break
            break
        ok = s == goal
        reached += int(ok)
        steps_all.append(steps if ok else budget)
        if first_all_known is None and all(agent.resolved(a) for a in ACTIONS):
            first_all_known = i + 1
    return dict(reached=reached / len(goals), mean_steps=float(np.mean(steps_all)), goals_until_all_known=first_all_known,
                tries=tries_spent, tries_on_unknown=tries_on_unknown, described=sum(agent.resolved(a) for a in ACTIONS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", type=int, default=40)
    ap.add_argument("--tries", type=int, default=2)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--streams", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e10"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    streams = []
    for _ in range(args.streams):
        goals = []
        while len(goals) < args.goals:
            start = tuple(int(v) for v in rng.integers(0, V, L))
            word = [ACTIONS[int(i)] for i in rng.integers(0, len(ACTIONS), int(rng.integers(1, 4)))]
            g = start
            for a in word:
                g = true_apply(a, g)
            if g != start:
                goals.append((start, g))
        streams.append(goals)
    print(f"E10: {args.streams} streams of {args.goals} goals; known at start {KNOWN}; up to {args.tries} tries per goal; "
          f"step budget {args.budget}\n")
    print(f"{'policy':<30}{'reached':>9}{'mean steps':>12}{'goals until all 7 described':>30}{'tries':>8}{'on unknown':>12}")
    res = {}
    for policy in ("none", "random", "least observed", "least observed (until known)", "price (this goal)", "price (horizon)"):
        rows = [run_stream(policy, goals, args.tries, args.budget, args.seed + i) for i, goals in enumerate(streams)]
        agg = dict(reached=float(np.mean([r["reached"] for r in rows])), mean_steps=float(np.mean([r["mean_steps"] for r in rows])),
                   goals_until_all_known=[r["goals_until_all_known"] for r in rows],
                   tries=float(np.mean([r["tries"] for r in rows])), tries_on_unknown=float(np.mean([r["tries_on_unknown"] for r in rows])),
                   described=float(np.mean([r["described"] for r in rows])))
        res[policy] = agg
        gk = agg["goals_until_all_known"]
        gk_s = f"{np.mean([g for g in gk if g is not None]):.1f}" if any(g is not None for g in gk) else "never"
        print(f"{policy:<30}{agg['reached']:>9.3f}{agg['mean_steps']:>12.2f}{gk_s:>30}{agg['tries']:>8.1f}{agg['tries_on_unknown']:>12.1f}")
    def judge(p):
        r, lo, lk = res["random"], res["least observed"], res["least observed (until known)"]
        ok = all(p["reached"] >= q["reached"] and p["mean_steps"] < q["mean_steps"] for q in (r, lo, lk))
        return "PASS" if ok else "REFUTED" if p["reached"] < r["reached"] else "INCONCLUSIVE"
    verdict = {k: judge(res[k]) for k in ("price (this goal)", "price (horizon)")}
    print("\nE10 verdict: price (this goal) " + verdict["price (this goal)"] + "; price (horizon) " + verdict["price (horizon)"])
    res["verdict"] = verdict
    json.dump(res, open(out / "e10.json", "w"), indent=1)


if __name__ == "__main__":
    main()
