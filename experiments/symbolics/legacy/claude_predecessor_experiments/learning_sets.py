"""Learning sets (docs/AGENT.md, part 18): a prior over goal types learned across games, Harlow's (1949) result in the
form of a Bayesian overhypothesis (Kemp, Perfors & Tenenbaum 2007).

Each game has new colours and one of two goal families: push (the box onto the single target cell) or reach (the body
onto it, with the box still present and irrelevant). The agent (part 17's DE17) starts every game empty. Only the goal
prior differs between arms: FLAT, the description-length prior; LS, a prior over goal *types* learned from earlier
games. A type abstracts away the colours: (cell, the kind's role, the colour's feature) or (at, the kind's role).
    python -m arc_agent.claude.learning_sets      (from experiments/symbolics)
"""
import json
import time
from collections import Counter, deque

import numpy as np

from arc_agent.claude.efference import D
from arc_agent.claude.homeostatic_agent import AFF, PMIN, GoalModel, fkey, things, run
from arc_agent.overhypothesis import ALPHAS, Overhypothesis
from arc_agent.claude.push_block_rebuild import DIRS, H, W

GAMES, SEEDS, LEVELS, CAP = 12, 3, 3, 300
LS_STUCK = 580          # part 19's diagnosis: LS's steps with the box stuck before first WINs (a replay of part 18)


# ---- the games ------------------------------------------------------------------------------------------------------
class GameWorld:
    """One game: fixed colours and goal family; a random layout per level (body, box, target, two distractors)."""
    def __init__(self, rng, spec):
        self.rng, self.spec, self.t = rng, spec, 0
        self.reset()

    def reset(self):
        while True:
            cells = [(int(c % W), int(c // W)) for c in self.rng.choice(W * H, 5, replace=False)]
            self.player, self.box, self.target, d1, d2 = cells
            self.deco = {d1: self.spec["deco"][0], d2: self.spec["deco"][1]}
            if 1 <= self.box[0] <= W - 2 and 1 <= self.box[1] <= H - 2 and self.optimum() is not None:
                break
        self.t = 0
        return self.frame()

    def snapshot(self):
        return (self.player, self.box, self.target, dict(self.deco))

    def restore(self, snap):
        self.player, self.box, self.target, self.deco = snap[0], snap[1], snap[2], dict(snap[3])
        self.t = 0
        return self.frame()

    def won(self, player, box):
        return (box if self.spec["family"] == "push" else player) == self.target

    def frame(self):
        g = np.zeros((H, W), int)
        for (x, y), c in self.deco.items():
            g[y, x] = c
        g[self.target[1], self.target[0]] = self.spec["target"]
        g[self.box[1], self.box[0]] = self.spec["box"]
        g[self.player[1], self.player[0]] = self.spec["body"]
        return g

    def move(self, player, box, a):
        dx, dy = DIRS[a]
        nx, ny = player[0] + dx, player[1] + dy
        if not (0 <= nx < W and 0 <= ny < H):
            return player, box
        if (nx, ny) == box:
            bx, by = box[0] + dx, box[1] + dy
            if 0 <= bx < W and 0 <= by < H:
                return (nx, ny), (bx, by)
            return player, box
        return (nx, ny), box

    def step(self, a):
        self.t += 1
        self.player, self.box = self.move(self.player, self.box, a)
        return self.frame(), self.won(self.player, self.box)

    def optimum(self):
        start = (self.player, self.box)
        seen, q = {start: 0}, deque([start])
        while q:
            p, b = q.popleft()
            if self.won(p, b):
                return seen[(p, b)]
            for a in range(4):
                nxt = self.move(p, b, a)
                if nxt not in seen:
                    seen[nxt] = seen[(p, b)] + 1
                    q.append(nxt)
        return None


def game_specs(rng):
    specs = []
    for _ in range(GAMES):
        body, box, target, d1, d2 = (int(c) for c in rng.choice(np.arange(1, 14), 5, replace=False))
        specs.append(dict(family="push" if rng.random() < 0.5 else "reach", body=body, box=box, target=target, deco=(d1, d2)))
    return specs


# ---- goal types and the learned prior -------------------------------------------------------------------------------
def roles(agent, aff, th):
    out = {}
    for t in th:
        if t.shape in out:
            continue
        if t.frame is not None:
            m = agent.choice(t.frame)
            out[t.shape] = "body" if m == "action" else "pushed" if m.startswith("push") else "other"
        else:
            ap = aff.posterior(t.shape)
            out[t.shape] = "pushed" if ap.get("pushable", 0) > 0.5 else "other"
    return out


def htype(h, rl, counts):
    if h[0] == "cell":
        feat = "background" if h[2] == 0 else "unique" if counts.get(h[2]) == 1 else "other"
        return ("cell", rl.get(h[1], "other"), feat)
    return ("at", rl.get(h[1], "other"))


def make_prior(agent, aff, type_counts, n_games):
    """P(h) = flat(h) * (games whose goal had h's type + the type's flat mass) / ((games + 1) * the type's flat mass)."""
    def prior_fn(hs, th, static):
        n = Counter(h[0] for h in hs)
        flat = {h: 0.5 / n[h[0]] for h in hs}
        if n_games == 0:
            return flat
        rl = roles(agent, aff, th)
        counts = Counter(int(v) for v in static.ravel() if v >= 0)
        types = {h: htype(h, rl, counts) for h in hs}
        mass = Counter()
        for h in hs:
            mass[types[h]] += flat[h]
        return {h: flat[h] * (type_counts[types[h]] + mass[types[h]]) / ((n_games + 1) * mass[types[h]]) for h in hs}
    return prior_fn


def play(seed, learn, oh=False, viable=False):
    rng = np.random.default_rng(seed)
    specs, type_counts, out = game_specs(rng), Counter(), []
    over = Overhypothesis() if oh else None                 # part 19
    one_cell = ((0, 0),)
    for g, spec in enumerate(specs):
        world = GameWorld(np.random.default_rng(1000 * seed + g), spec)
        start = {}
        if oh:                                              # O1, O2: the priors this game starts from
            start = dict(actions_certain=all(max(over.prior(("action", a), D).values()) >= 1 - PMIN for a in range(4)),
                         p_pushable=over.prior(("aff", one_cell), AFF)["pushable"])
        eff_prior = (lambda a: over.prior(("action", a), D)) if oh else None
        aff_prior = (lambda shape: over.prior(("aff", shape[0]), AFF)) if oh else None
        memo, stuck, keep = {}, [], {}
        box_kind, body_kind = (((0, 0),), (spec["box"],)), (((0, 0),), (spec["body"],))
        true_kind, true_role = (box_kind, "pushed") if spec["family"] == "push" else (body_kind, "body")
        h_true = ("cell", true_kind, spec["target"])
        state = {}

        def factory(agent, aff):
            fn = make_prior(agent, aff, type_counts, g if learn else 0)
            state.update(agent=agent, aff=aff, fn=fn)
            return GoalModel(charge_new=True, prior_fn=fn)

        def probe(agent, goal, aff, win):
            state["goal"] = goal
            key = (world.player, world.box, world.target)   # measured from the true game; the agent never sees it
            if key not in memo:
                memo[key] = world.optimum() is None
            stuck.append(memo[key] and not win)
            if "p_true" not in state:                       # L3: once the true kind has its role
                th = things(agent)
                if roles(agent, aff, th).get(true_kind) == true_role:
                    hs = goal.hypotheses(th, agent.static)
                    pri = state["fn"](hs, th, agent.static)
                    z = sum(pri.values())
                    state["p_true"] = pri.get(h_true, 0.0) / z
                    post = goal.posterior(th, agent.static)
                    state["rank_true"] = sorted(post, key=lambda h: -post[h]).index(h_true) + 1 if h_true in post else None

        r = run({}, 1000 * seed + g, True, True, True, commit=True, macro=True, common=True, world=world,
                goal_factory=factory, levels=LEVELS, cap=CAP, probe=probe, aff_prior=aff_prior, eff_prior=eff_prior,
                viable=viable, keep=keep)
        if oh:                                              # what this game teaches about actions and objects
            eff_, aff_, agent_ = keep["eff"], keep["aff"], keep["agent"]
            for i, f in enumerate(agent_.frames):
                if agent_.choice(i) == "action":
                    for a in range(4):
                        d, p, _ = eff_.intended(fkey(f), a)
                        if eff_.moves[(fkey(f), a)] and p >= 1 - PMIN:
                            over.add(("action", a), d)
            for shape in aff_.lw:
                post = aff_.posterior(shape)
                v = max(post, key=post.get)
                if post[v] >= 1 - PMIN:
                    over.add(("aff", shape[0]), v)
        fw = r["first_win"] or CAP
        pre = stuck[:fw]
        goal, agent, aff = state["goal"], state["agent"], state["aff"]
        th = things(agent)
        post = goal.posterior(th, agent.static)
        h_map = max(post, key=post.get)
        rl, counts = roles(agent, aff, th), Counter(int(v) for v in agent.static.ravel() if v >= 0)
        t_map = htype(h_map, rl, counts)
        if r["levels"]:                                     # what this game teaches about goal types
            type_counts[t_map] += 1
        out.append(dict(game=g, family=spec["family"], first_win=r["first_win"], levels=[(l["steps"], l["optimum"], l["first_attempt"]) for l in r["levels"]],
                        p_true=state.get("p_true"), rank_true=state.get("rank_true"), map_correct=h_map == h_true, map_type=list(t_map),
                        kinds_until_win=dict(Counter(r["kinds_until_win"])), steps=r["total_steps"],
                        stuck_before_win=sum(pre), stuck_events=sum(1 for i, x in enumerate(pre) if x and (i == 0 or not pre[i - 1])),
                        ms_per_step=1000 * r["seconds"] / max(r["total_steps"], 1), **start))
    if oh:
        alphas = {slot: ALPHAS[int(np.argmax(over.alpha_posterior(slot, D if slot[0] == "action" else AFF)))] for slot in over.draws}
        out[-1]["alpha_map"] = {str(k): v for k, v in alphas.items()}
    return out


def main():
    t0 = time.time()
    res = {}
    for arm, learn in (("FLAT", False), ("LS", True)):
        res[arm] = [play(seed, learn) for seed in range(SEEDS)]
        print(f"{arm}: {time.time() - t0:.0f} s", flush=True)
        for s, games in enumerate(res[arm]):
            print(f"   seed {s}: first WIN by game {[g['first_win'] for g in games]}", flush=True)
            print(f"           families {[g['family'][0] for g in games]}, goal identified {[int(g['map_correct']) for g in games]}", flush=True)
    print()
    summary = {}
    for arm in res:
        games = [g for run_ in res[arm] for g in run_]
        fw = lambda lo, hi: [g["first_win"] if g["first_win"] is not None else CAP for g in games if lo <= g["game"] <= hi]
        early, late = np.mean(fw(0, 2)), np.mean(fw(6, 11))
        later = [lv for g in games for lv in g["levels"][1:]]
        pre = Counter()
        pre_late = Counter()
        for g in games:
            pre.update(g["kinds_until_win"])
            if g["game"] >= 6:
                pre_late.update(g["kinds_until_win"])
        p_true = [g["p_true"] for g in games if g["game"] >= 6 and g["p_true"] is not None]
        ranks = [g["rank_true"] for g in games if g["game"] >= 6 and g["rank_true"] is not None]
        summary[arm] = dict(first_win_games_1_3=float(early), first_win_games_7_12=float(late), ratio=float(late / early),
                            levels_2_3_first_attempt=float(np.mean([lv[2] for lv in later])) if later else None,
                            p_true_games_7_12=float(np.mean(p_true)) if p_true else None,
                            median_rank_true_games_7_12=float(np.median(ranks)) if ranks else None,
                            explore_share_before_win=pre["explore"] / max(sum(pre.values()), 1),
                            explore_share_before_win_games_7_12=pre_late["explore"] / max(sum(pre_late.values()), 1),
                            breakdown_before_win=dict(pre), breakdown_games_7_12=dict(pre_late),
                            goals_identified=float(np.mean([g["map_correct"] for g in games])))
        print(f"{arm}: " + ", ".join(f"{k} {v if not isinstance(v, float) else round(v, 3)}" for k, v in summary[arm].items()), flush=True)
    print(f"\ntotal {time.time() - t0:.1f} s")
    json.dump(dict(runs=res, summary=summary), open("runs/learning_sets.json", "w"), indent=1, default=str)


def part19():
    t0 = time.time()
    res = {"LS18": json.load(open("runs/learning_sets.json"))["runs"]["LS"]}     # part 18 as recorded, before the fix
    for arm, oh, viable in (("LS", False, False), ("OH", True, False), ("LSV", False, True), ("OHV", True, True)):
        res[arm] = []
        for seed in range(SEEDS):
            res[arm].append(play(seed, True, oh, viable))
            if time.time() - t0 > 570:                      # the 9.5-minute guard
                print("guard: stopped", flush=True)
                break
        print(f"{arm}: {time.time() - t0:.0f} s", flush=True)
        for s, games in enumerate(res[arm]):
            print(f"   seed {s}: first WIN by game {[g['first_win'] for g in games]}", flush=True)
        if time.time() - t0 > 570:
            break
    print()
    summary = {}
    for arm in res:
        games = [g for run_ in res[arm] for g in run_]
        fw = lambda g: g["first_win"] if g["first_win"] is not None else CAP
        mean = lambda xs: float(np.mean(xs)) if xs else None
        push, reach = [g for g in games if g["family"] == "push"], [g for g in games if g["family"] == "reach"]
        later = [lv for g in games for lv in g["levels"][1:]]
        pre = Counter()
        for g in games:
            pre.update(g["kinds_until_win"])
        late = [g for g in games if g["game"] >= 6]
        summary[arm] = dict(
            first_win_games_1_3=mean([fw(g) for g in games if g["game"] <= 2]),
            first_win_games_7_12=mean([fw(g) for g in late]),
            first_win_push=mean([fw(g) for g in push]), first_win_reach=mean([fw(g) for g in reach]),
            push_games_1_6=mean([fw(g) for g in push if g["game"] <= 5]), push_games_7_12=mean([fw(g) for g in push if g["game"] >= 6]),
            stuck_steps_before_win=sum(g.get("stuck_before_win", 0) for g in games) if arm != "LS18" else LS_STUCK,
            stuck_events=sum(g.get("stuck_events", 0) for g in games) if arm != "LS18" else 20,
            games_not_won=sum(g["first_win"] is None for g in games),
            levels_2_3_first_attempt=mean([lv[2] for lv in later]),
            explore_share_before_win=pre["explore"] / max(sum(pre.values()), 1),
            actions_certain_at_start_games_7_12=mean([g["actions_certain"] for g in late if "actions_certain" in g]),
            p_pushable_at_start_games_7_12=mean([g["p_pushable"] for g in late if "p_pushable" in g]),
            ms_per_step=mean([g["ms_per_step"] for g in games if "ms_per_step" in g]),
            alpha_map=[g["alpha_map"] for g in games if "alpha_map" in g])
        print(f"{arm}: " + ", ".join(f"{k} {round(v, 3) if isinstance(v, float) else v}" for k, v in summary[arm].items()), flush=True)
    print()
    print(f"total {time.time() - t0:.1f} s")
    json.dump(dict(runs=res, summary=summary), open("runs/learning_sets_part19.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    import sys
    part19() if "--part19" in sys.argv else main()
