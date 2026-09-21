"""E10b — acting to learn, with useless and noisy actions (DESIGN.md §16, OPEN-11's discriminating test).

E10's environment plus four distractor actions the goals never need: two NO-OPS (they do nothing; one observation
describes them as the identity) and two NOISY actions (a fresh random permutation of the positions every time they
are used; no description ever fits, and the rate price says so through the exception rate). Goals are generated
with the seven real actions only. Tries cost a step each, as before.

Policies: none; random; least observed until known, with "known" = a description exists, whatever its exception
rate (count-based novelty with a stopping rule -- it stops on a noisy action after one try, calling it "identity with
exceptions"); least observed until known, with "known" = a description with exception rate < 0.5 (novelty that
refuses to call noise known -- and therefore keeps trying it: the trap); price (horizon), whose hypotheses for an
action are what its evidence allows -- and for an action whose best description is refuted more often than not,
nothing, so its expected saving is zero and it is left alone.

Pass (pre-registered): price reaches at least as many goals as every novelty policy with fewer mean steps, and
spends fewer tries on the noisy actions than the epsilon-aware novelty policy. Refute: price worse than random.

    python experiments/ziplearn/e10b.py          (CPU, ~6 min; run detached) -> runs/e10b/e10b.json
"""
from __future__ import annotations

import argparse
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
REAL = list(H.NAMES)
NOOPS = ["noop_a", "noop_b"]
NOISY = ["noise_a", "noise_b"]
ACTIONS = REAL + NOOPS + NOISY
KNOWN = ["rot_left", "swap_pairs", "inc", "negate"]
BITS = math.log2(len(ACTIONS))


class World:
    def __init__(self, rng):
        self.rng = rng

    def apply(self, name, s):
        if name in NOOPS:
            return tuple(s)
        if name in NOISY:
            pi = self.rng.permutation(L)
            return tuple(int(s[pi[j]]) for j in range(L))
        return tuple(int(v) for v in H.PRIMS[name](torch.tensor(s)).tolist())

    def apply_real(self, name, s):
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

    def exception_rate(self, a):
        _n, _ab, pos = self.models[a].best()
        n = pos.n_right + pos.n_wrong
        return pos.n_wrong / n if n else 0.0

    def effect(self, a, s):
        if self.n_obs[a] == 0:
            return None
        out = self.models[a].predict_seq(np.array(s))
        return None if any(v is None for v in out) else tuple(int(v) for v in out)

    def described(self, a):
        return self.n_obs[a] > 0 and all(self.effect(a, tuple(int(v) for v in self.rng.integers(0, V, L))) is not None
                                         for _ in range(2))

    def known(self, a, eps_aware):
        return self.described(a) and (not eps_aware or self.exception_rate(a) < 0.5)

    def hypotheses(self, a, M=6):
        """What action `a` might do, given its evidence. Never seen: random permutations. Partly seen: completions of
        the candidate sources of the runs near the best price. Refuted more often than not: NOTHING -- an action no
        description fits cannot be planned with, so it can save no bits."""
        learner = self.models[a]
        if self.n_obs[a] == 0:
            return [(tuple(self.rng.permutation(L)), 1, 0) for _ in range(M)]
        if self.n_obs[a] >= 2 and self.exception_rate(a) >= 0.5:
            return []
        best_price = min(bits + learner.bits_value_name + pos.cost + learner.bits_pos_name
                         for (_n, _ab, bits), pid, pperm in learner.runs for pos in (pid, pperm))
        hyps = []
        for (name, (va, vb), bits), pid, pperm in learner.runs:
            for pos in (pid, pperm):
                if bits + learner.bits_value_name + pos.cost + learner.bits_pos_name > best_price + 2.0:
                    continue
                if isinstance(pos, PositionPerm):
                    if pos.dead:
                        continue
                    cands = [sorted(c) for c in pos.cand]
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
            hyps = [hyps[i] for i in self.rng.choice(len(hyps), M, replace=False)]
        return hyps


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
    def apply_fn(a, st):
        if override and a == override[0]:
            pi, va, vb = override[1]
            return hyp_apply(pi, va, vb, st)
        e = agent.effect(a, st)
        return None if (e is None or agent.exception_rate(a) >= 0.5) else e      # noise is not planned with
    w = plan(apply_fn, s, g, depth)
    return (len(w) * BITS) if w is not None else (depth + 2) * BITS


def choose_try(policy, agent, s, g, rng, tries_left, recent_goals, horizon):
    if policy == "none" or tries_left == 0:
        return None
    if policy == "random":
        return ACTIONS[int(rng.integers(len(ACTIONS)))]
    if policy.startswith("least observed"):
        eps_aware = "eps" in policy
        unknown = [a for a in ACTIONS if not agent.known(a, eps_aware)]
        return min(unknown, key=lambda a: (agent.n_obs[a], rng.random())) if unknown else None
    unknown = [a for a in ACTIONS if not agent.known(a, True)]
    if not unknown:
        return None
    sample = [(s, g)] + list(recent_goals)
    best, best_gain = None, 0.0
    for a in unknown:
        hyps = agent.hypotheses(a)
        if not hyps:
            continue
        per_goal = []
        for (ss, gg) in sample:
            base = plan_bits(agent, ss, gg)
            per_goal.append(float(np.mean([max(0.0, base - plan_bits(agent, ss, gg, (a, h))) for h in hyps])))
        gain = float(np.mean(per_goal)) * max(1, horizon) - BITS
        if gain > best_gain:
            best, best_gain = a, gain
    return best


def run_stream(policy, goals, T, budget, seed):
    rng = np.random.default_rng(seed)
    world, agent = World(np.random.default_rng(seed + 1000)), Agent(rng)
    for a in KNOWN:
        for _ in range(3):
            s = tuple(int(v) for v in rng.integers(0, V, L))
            agent.observe(a, s, world.apply(a, s))
    reached, steps_all = 0, []
    tries = {"real": 0, "noop": 0, "noisy": 0}
    for i, (start, goal) in enumerate(goals):
        s, steps, t_used = start, 0, 0
        recent = goals[max(0, i - 3):i]
        while steps < budget:
            a = choose_try(policy, agent, s, goal, rng, T - t_used, recent, len(goals) - i)
            if a is not None:
                s2 = world.apply(a, s)
                agent.observe(a, s, s2)
                tries["noop" if a in NOOPS else "noisy" if a in NOISY else "real"] += 1
                s, steps, t_used = s2, steps + 1, t_used + 1
                if s == goal:
                    break
                continue
            w = plan(lambda act, st: (None if agent.exception_rate(act) >= 0.5 else agent.effect(act, st)), s, goal, 4)
            if w is None:
                break
            for act in w:
                s2 = world.apply(act, s)
                agent.observe(act, s, s2)
                s, steps = s2, steps + 1
                if s == goal or steps >= budget:
                    break
            break
        ok = s == goal
        reached += int(ok)
        steps_all.append(steps if ok else budget)
    return dict(reached=reached / len(goals), mean_steps=float(np.mean(steps_all)), tries=tries,
                described_real=sum(agent.known(a, True) for a in REAL))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", type=int, default=40)
    ap.add_argument("--tries", type=int, default=2)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--streams", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e10b"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    world = World(rng)
    streams = []
    for _ in range(args.streams):
        goals = []
        while len(goals) < args.goals:
            start = tuple(int(v) for v in rng.integers(0, V, L))
            word = [REAL[int(i)] for i in rng.integers(0, len(REAL), int(rng.integers(1, 4)))]
            g = start
            for a in word:
                g = world.apply_real(a, g)
            if g != start:
                goals.append((start, g))
        streams.append(goals)
    print(f"E10b: {args.streams} streams of {args.goals} goals; actions = 7 real + {len(NOOPS)} no-op + {len(NOISY)} noisy; "
          f"known at start {KNOWN}; up to {args.tries} tries per goal; budget {args.budget}\n")
    print(f"{'policy':<36}{'reached':>9}{'mean steps':>12}{'tries real/noop/noisy':>24}{'real described':>16}")
    res = {}
    for policy in ("none", "random", "least observed (until known)", "least observed (until known, eps-aware)", "price (horizon)"):
        rows = [run_stream(policy, goals, args.tries, args.budget, args.seed + i) for i, goals in enumerate(streams)]
        agg = dict(reached=float(np.mean([r["reached"] for r in rows])), mean_steps=float(np.mean([r["mean_steps"] for r in rows])),
                   tries={k: float(np.mean([r["tries"][k] for r in rows])) for k in ("real", "noop", "noisy")},
                   described_real=float(np.mean([r["described_real"] for r in rows])))
        res[policy] = agg
        tr = agg["tries"]
        print(f"{policy:<36}{agg['reached']:>9.3f}{agg['mean_steps']:>12.2f}{tr['real']:>10.1f}/{tr['noop']:<4.1f}/{tr['noisy']:<7.1f}"
              f"{agg['described_real']:>16.1f}")
    p = res["price (horizon)"]
    novelty = [res["least observed (until known)"], res["least observed (until known, eps-aware)"]]
    ok = all(p["reached"] >= q["reached"] and p["mean_steps"] < q["mean_steps"] for q in novelty + [res["random"]]) and \
        p["tries"]["noisy"] < res["least observed (until known, eps-aware)"]["tries"]["noisy"]
    verdict = "PASS" if ok else "REFUTED" if p["reached"] < res["random"]["reached"] else "INCONCLUSIVE"
    print(f"\nE10b verdict: {verdict}")
    res["verdict"] = verdict
    json.dump(res, open(out / "e10b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
