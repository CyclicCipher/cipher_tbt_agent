"""ZipLearner plays the ARC-AGI-3 replica games (src/tasks/games): the interface, and the first agent behind it.

The interface (DESIGN.md §16 applied to a game): a frame is the state; the game's available actions are the actions;
the score and WIN / GAME_OVER signals are the only feedback; nothing about any game is known in advance. Everything
the agent learns is a LOCAL RULE learned by counting with the rate price -- the same core loop as everywhere:

  world model   one description per action: "the new colour of a cell is a function of the window around it" -- a
                table over windows, the same table at every cell (§5's tied blocks: translation invariance). Two
                window sizes (radius 1 and 2) compete on price, so the description that compresses best decides how
                much context the rule needs (a push needs more than a step). A window never seen predicts "unchanged"
                and is counted as UNKNOWN -- generalise, then correct.
  goal model    the windows present around the cells that changed at a transition that raised the score (win keys,
                per action) or ended the game (death keys). A (frame, action) is predicted to win / kill when a
                window of the frame is a known win / death key for that action.
  explorer      `explore`: the plan worth the most bits -- unknown windows to learn, frames never visited --
                exploration as the price of ignorance (E27). It is the only chooser of actions here.
  planning      NOT DESIGNED HERE. E27's breadth-first search and E29's imagination (a search procedure written
                around the model) were deleted on 2026-09-21: thinking belongs inside the looped block, as its own
                computation over the context window (DESIGN §18). Until that is built the games are explored, not solved
                on purpose; the goal model is kept because its keys are what the block will read.
  loop          choose, execute one step, observe (the transition trains the world and goal models), re-plan when
                the observation differs from the prediction.

No coordinates (ACTION6) yet: the games here do not use it. The frame is cropped to the bounding box of its non-zero
cells plus a margin, fixed per level -- a constant background costs nothing and is not looked at.
"""
from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
from ziplearner import flag_bits   # noqa: E402
from tasks.core import GameAction, GameState   # noqa: E402

V = 16                              # ARC colours
BORDER = -1


def entropy(p):
    return 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


class LocalRule:
    """One action's effect as a table from windows to the centre cell's new colour, shared by every cell. The
    description is the table over the cells of the window the MASK keeps; the evidence (`full`, counts over whole
    windows) is kept so that a SLEEP pass can try smaller masks -- fewer context cells -- and adopt one when the
    merged table is cheaper: the same knowledge in fewer, more general entries (E24)."""

    def __init__(self, radius):
        self.r = radius
        n = (2 * radius + 1) ** 2
        self.mask = np.ones(n, dtype=bool)
        self.centre = n // 2
        self.full = {}                                  # whole window -> {colour: count}: the evidence
        self.table = {}                                 # masked window -> {colour: count}: the description
        self.majority = {}                              # masked window -> the colour predicted (the entry's majority)
        self.stats = {}                                 # masked window -> [n_right, n_wrong]
        self.cost = 0.0

    def _project(self, full_key):
        return np.frombuffer(full_key, dtype=np.int16)[self.mask].tobytes()

    def _rows(self, frame):
        """Every cell's window as a row: (whole windows, masked windows), each (H*W, cells), in raster order."""
        r = self.r
        H, W = frame.shape
        pad = np.full((H + 2 * r, W + 2 * r), BORDER, dtype=np.int16)
        pad[r:r + H, r:r + W] = frame
        full = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(pad, (2 * r + 1, 2 * r + 1)).reshape(H * W, -1))
        return full, np.ascontiguousarray(full[:, self.mask])

    def windows(self, frame):
        """(i, j) -> (whole-window key, masked key)."""
        H, W = frame.shape
        full, masked = self._rows(frame)
        return {(t // W, t % W): (full[t].tobytes(), masked[t].tobytes()) for t in range(H * W)}

    def observe(self, before, after):
        win = self.windows(before)
        for (i, j), (fk, k) in win.items():
            y = int(after[i, j])
            f = self.full.setdefault(fk, {})
            f[y] = f.get(y, 0) + 1
            e = self.table.get(k)
            if e is None:                               # a new window: its outcome is a parameter
                self.cost += math.log2(V)
                self.table[k] = {y: 1}
                self.majority[k] = y
                self.stats[k] = [0, 0]
                continue
            pred = self.majority[k]
            st = self.stats[k]
            if pred == y:
                self.cost += flag_bits(False, st[0], st[1])
                st[0] += 1
            else:
                self.cost += flag_bits(True, st[0], st[1]) + math.log2(V - 1)
                st[1] += 1
            e[y] = e.get(y, 0) + 1
            self.majority[k] = max(e, key=e.get)

    def predict(self, frame):
        """The predicted frame and the number of cells whose (masked) window was never seen (left unchanged)."""
        _full, masked = self._rows(frame)
        out = frame.copy().reshape(-1)
        unknown = 0
        majority = self.majority
        for t in range(out.size):
            y = majority.get(masked[t].tobytes())
            if y is None:
                unknown += 1
            else:
                out[t] = y
        return out.reshape(frame.shape), unknown

    # -- the sleep pass ----------------------------------------------------------------------------------------
    def _merged(self, mask):
        merged = {}
        for fk, counts in self.full.items():
            k = np.frombuffer(fk, dtype=np.int16)[mask].tobytes()
            m = merged.setdefault(k, {})
            for y, c in counts.items():
                m[y] = m.get(y, 0) + c
        return merged

    @staticmethod
    def _price(merged):
        """Two-part code of the evidence under a table: one parameter per entry, exceptions by their rate."""
        bits = 0.0
        for counts in merged.values():
            n = sum(counts.values())
            wrong = n - max(counts.values())
            bits += math.log2(V) + n * entropy(wrong / n) + wrong * math.log2(V - 1)
        return bits

    @staticmethod
    def _wrong(merged):
        return sum(sum(c.values()) - max(c.values()) for c in merged.values())

    def sleep(self):
        """Drop context cells while the merged table gets cheaper AND explains the evidence no worse -- no new
        exceptions (greedy, the centre always kept). The first run let the price alone decide and it merged away the
        rare cells that change, since forgetting a rare change costs fewer bits than an entry; the planner lives on
        exactly those cells. Compact without losing what was known. Returns (bits before, bits after, cells kept)."""
        if not self.full:
            return 0.0, 0.0, int(self.mask.sum())
        merged0 = self._merged(self.mask)
        before, wrong0 = self._price(merged0), self._wrong(merged0)
        mask, price = self.mask.copy(), before
        while True:
            best_mask, best_price = None, price
            for c in np.flatnonzero(mask):
                if c == self.centre:
                    continue
                cand = mask.copy()
                cand[c] = False
                m = self._merged(cand)
                pr = self._price(m)
                if pr < best_price and self._wrong(m) <= wrong0:
                    best_mask, best_price = cand, pr
            if best_mask is None:
                break
            mask, price = best_mask, best_price
        self.mask = mask
        merged = self._merged(mask)
        self.table = merged
        self.majority = {k: max(c, key=c.get) for k, c in merged.items()}
        self.stats = {k: [max(c.values()), sum(c.values()) - max(c.values())] for k, c in merged.items()}
        self.cost = price
        return before, price, int(mask.sum())


class ActionModel:
    """Two window sizes compete on price; the cheapest description predicts."""

    def __init__(self, radii=(1, 2)):
        self.rules = {r: LocalRule(r) for r in radii}
        self.n_obs = 0

    def observe(self, before, after):
        for rule in self.rules.values():
            rule.observe(before, after)
        self.n_obs += 1

    def best(self):
        return min(self.rules.values(), key=lambda rule: (round(rule.cost + math.log2(len(self.rules)), 9), rule.r))

    def predict(self, frame):
        if self.n_obs == 0:
            return None, None
        return self.best().predict(frame)

    def sleep(self):
        return {r: rule.sleep() for r, rule in self.rules.items()}

    def exception_rate(self):
        st = self.best().stats.values()
        right, wrong = sum(s[0] for s in st), sum(s[1] for s in st)
        return wrong / (right + wrong) if right + wrong else 0.0


class GoalModel:
    """Win and death keys: the windows around the cells that changed at a scoring / fatal transition. Each key keeps
    a count of the times it fired and the outcome followed, and the times it fired and it did not (E19: hindsight
    as exception accounting on the goal model): a key that is wrong more often than right stops predicting."""

    def __init__(self, radius=1):
        self.r = radius
        self.win, self.death = {}, {}                   # action -> {window: [confirmed, refuted]}
        self.probe = LocalRule(radius)

    def record(self, action, before, after, outcome, recent=()):
        """Keys = the windows (in the frame before the move) around the cells that changed -- or, when the after-frame
        is not observable (a win moves to the next level), around the cells the model predicted to change and the
        cells that changed in the PREVIOUS transition: where things were happening. Never every window: a goal key
        that matches everywhere is a goal that fires everywhere (the first run's failure)."""
        changed = set(zip(*np.nonzero(before != after))) | set(recent)
        if not changed:
            return
        win = self.probe.windows(before)
        keys = {win[(int(i), int(j))][0] for i, j in changed}
        target = (self.win if outcome == "win" else self.death).setdefault(action, {})
        for k in keys:
            target.setdefault(k, [0, 0])[0] += 1

    def live(self, which, action):
        keys = (self.win if which == "win" else self.death).get(action, {})
        return {k for k, (c, r) in keys.items() if c > r}    # trusted while confirmed more often than refuted

    def predicts(self, which, action, frame):
        keys = self.live(which, action)
        if not keys:
            return False
        return any(fk in keys for fk, _m in self.probe.windows(frame).values())

    def refute(self, which, action, frame):
        """Hindsight: this (frame, action) was predicted to win / kill and did not -- every key that fired is refuted."""
        keys = (self.win if which == "win" else self.death).get(action, {})
        for fk, _m in self.probe.windows(frame).values():
            if fk in keys:
                keys[fk][1] += 1

    def known(self):
        return any(self.live("win", a) for a in self.win)


class Player:
    """The loop of §16 on a game: describe, act, learn from what happened. The only chooser is `explore` (E27): no
    planner lives here any more (module docstring)."""

    def __init__(self, actions, max_depth=14, max_nodes=4000):
        self.actions = list(actions)
        self.models = {a: ActionModel() for a in self.actions}
        self.goal = GoalModel()
        self.max_depth, self.max_nodes = max_depth, max_nodes
        self._cache = {}                                                 # (frame key, action) -> the model's prediction
        self.visited = set()
        self.last_changed = set()
        self.plan = []
        self.expect_win = self.expect_win_at_end = False
        self.last_choice = None
        self.remaining = 100                                             # the level's remaining action budget
        self.trans = {}                                                  # (frame key, action) -> the observed next frame
        self.stats = dict(predictions=0, correct=0, unknown_cells=0, cells=0, replans=0, explore_plans=0, goal_plans=0,
                          calls=0, thoughts=0)

    def frame_key(self, frame):
        return frame.tobytes()

    def sleep(self):
        """The sleep pass at a checkpoint: every action's rules drop the context cells they do not need."""
        report = {}
        for a, m in self.models.items():
            if m.n_obs:
                report[a.name] = m.sleep()
        self._cache = {}
        self.stats["sleeps"] = self.stats.get("sleeps", 0) + 1
        return report

    def observe(self, action, before, after, outcome):
        self._cache = {}
        if outcome != "win":                                  # a winning move's after-frame is the next level: not seen
            self.models[action].observe(before, after)
            self.trans[(self.frame_key(before), action)] = after.copy()    # a transition seen is a transition known
        if outcome in ("win", "death"):
            self.goal.record(action, before, after, outcome, recent=self.last_changed)
        self.last_changed = set(zip(*np.nonzero(before != after)))
        self.visited.add(self.frame_key(after))

    def predict(self, action, frame):
        """The observed successor when this exact transition has been seen (exact, nothing to learn); otherwise the
        model's prediction and how many windows it had never seen (cached per frame while the model is unchanged)."""
        key = (self.frame_key(frame), action)
        known = self.trans.get(key)
        if known is not None:
            return known, 0
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        pred, unknown = self.models[action].predict(frame)
        self._cache[key] = (pred, unknown)
        return pred, unknown

    def explore(self, frame):
        """E27: exploration as the price of ignorance over the whole reachable set the model can predict. Every
        frame the search reaches is valued in bits -- the windows that would be learned by acting there (unknown
        windows x log2 V, for actions whose rule is not mostly refuted: the noisy-TV filter) plus one entry's worth
        for a frame never visited -- less the plan's cost (its length x log2 actions). The best value wins; a
        visited frame with nothing to learn is worth nothing and is never a target. Plans are whole paths, not the
        first step to the nearest unknown."""
        bits_action = math.log2(max(2, len(self.actions)))
        learnable = [a for a in self.actions if self.models[a].n_obs == 0 or self.models[a].exception_rate() < 0.5]
        start_key = self.frame_key(frame)
        seen = {start_key: []}
        frontier = deque([(frame, [])])
        candidates = []                                                  # (plan, info bits, reaches a new frame)
        nodes = 0
        while frontier and nodes < self.max_nodes:
            f, path = frontier.popleft()
            for a in self.actions:
                nodes += 1
                if self.goal.predicts("death", a, f):
                    continue
                pred, unknown = self.predict(a, f)
                if pred is None or unknown:                          # doing THIS here is information: a complete plan
                    if a in learnable:
                        candidates.append((path + [a], (unknown if pred is not None else f.size) * math.log2(V), False))
                    if pred is None:
                        continue
                elif (self.frame_key(f), a) not in self.trans and a in learnable:
                    # predicted from a rule but never observed here: worth testing in proportion to the rule's own
                    # exception rate -- untested predictions are the model's remaining uncertainty once nothing is
                    # unknown, and they are what keeps the explorer from oscillating between two known frames
                    candidates.append((path + [a], max(0.02, self.models[a].exception_rate()) * math.log2(V), False))
                k = self.frame_key(pred)
                if k in seen:
                    continue
                seen[k] = path + [a]
                candidates.append((path + [a], 0.0, k not in self.visited))
                if len(path) + 1 < self.max_depth:
                    frontier.append((pred, path + [a]))
        if not candidates:
            return None, None
        # Under an unknown goal every never-visited frame is equally likely to hold it, so reaching one is worth the
        # bits the rest of the level would cost, spread over the unvisited frames in reach: the last unvisited frame
        # is worth everything, and a far one is still worth the walk when nothing nearer is new.
        n_new = sum(1 for _p, _i, new in candidates if new)
        novelty_bits = (self.remaining * bits_action / max(1, n_new)) if not self.goal.known() else math.log2(V)

        def value(c):
            path, info, new = c
            return info + (novelty_bits if new else 0.0) - len(path) * bits_action
        best = max(candidates, key=value)                                 # the budget is spent either way: the best
        return best[0], "explore"                                        # plan is taken even at a negative value

    def choose(self, frame):
        if self.plan:
            a = self.plan.pop(0)
            self.expect_win = (not self.plan) and self.expect_win_at_end
            self.last_choice = dict(kind="continuing", plan=[a.name] + [x.name for x in self.plan])
            return a
        plan, kind = self.explore(frame)
        if plan is None:                                             # nothing known and nothing new: any action
            plan, kind = [self.actions[len(self.visited) % len(self.actions)]], "blind"
        self.stats["goal_plans" if kind == "goal" else "explore_plans"] += 1
        self.plan = plan[1:]
        self.expect_win_at_end = kind == "goal"
        self.expect_win = kind == "goal" and not self.plan
        self.last_choice = dict(kind=kind, plan=[a.name for a in plan])
        return plan[0]


def crop_box(frame):
    nz = np.argwhere(frame != 0)
    if len(nz) == 0:
        return (0, 0, frame.shape[0], frame.shape[1])
    (i0, j0), (i1, j1) = nz.min(0), nz.max(0)
    m = 1
    return (max(0, i0 - m), max(0, j0 - m), min(frame.shape[0], i1 + 1 + m), min(frame.shape[1], j1 + 1 + m))


def play(env, budget_per_level=200, max_levels=None, seed=0, verbose=False, sleep=False, trace=None, return_player=False,
         max_depth=14, max_nodes=4000):
    """Drive one Environment with a Player. Returns per-level results and the player's statistics. `trace`, if a list,
    receives one record per step: what the player saw, chose, expected and got."""
    fd = env.reset()
    actions = [a for a in env.game.available_actions()]
    player = Player(actions, max_depth=max_depth, max_nodes=max_nodes)
    results = []
    level = env.level
    used = 0
    box = crop_box(np.array(fd.grid))
    n_levels = env.game.level_count if max_levels is None else min(max_levels, env.game.level_count)
    while env.state not in (GameState.WIN,) and level < n_levels:
        frame = np.array(fd.grid, dtype=np.int16)[box[0]:box[2], box[1]:box[3]]
        budget = budget_per_level[level] if isinstance(budget_per_level, (list, tuple)) else budget_per_level
        player.remaining = max(1, budget - used)
        a = player.choose(frame)
        pred, _unknown = player.predict(a, frame)
        rec = None
        if trace is not None:
            rec = dict(level=level, step=used, action=a.name, choice=player.last_choice, unknown=_unknown,
                       frame=frame.tolist(), predicted=(pred.tolist() if pred is not None else None),
                       fired_win=bool(player.goal.predicts("win", a, frame)), visited_before=None, outcome="none")
            trace.append(rec)
        fd = env.step(a)
        used += 1
        after_full = np.array(fd.grid, dtype=np.int16)
        if env.level != level or env.state == GameState.WIN:              # the level was completed by this action;
            if rec is not None:
                rec["outcome"] = "win"
            guess = pred if pred is not None else frame                    # the next frame is another level, so the
            player.observe(a, frame, guess, "win")                          # changed cells are the model's prediction
            results.append(dict(level=level, solved=True, actions=used))
            if verbose:
                print(f"   level {level} solved in {used} actions")
            if sleep:                                                          # the checkpoint: sleep before the next level
                rep = player.sleep()
                results[-1]["sleep"] = {a: {str(r): dict(bits_before=round(b, 1), bits_after=round(af, 1), cells=c)
                                            for r, (b, af, c) in rr.items()} for a, rr in rep.items()}
            level, used = env.level, 0
            player.plan, player.visited, player.trans = [], set(), {}
            if env.state != GameState.WIN:
                box = crop_box(after_full)
            continue
        after = after_full[box[0]:box[2], box[1]:box[3]]
        if after.shape != frame.shape:                                      # the playfield grew: re-crop next step
            box = crop_box(after_full)
            after = after_full[box[0]:box[2], box[1]:box[3]]
        if rec is not None:
            rec["observed"] = after.tolist()
            rec["visited_before"] = player.frame_key(after) in player.visited
            rec["outcome"] = "death" if env.state == GameState.GAME_OVER else "none"
        if env.state == GameState.GAME_OVER:
            player.observe(a, frame, after, "death")
            fd = env.step(GameAction.RESET)
            used += 1
            player.plan = []
            continue
        player.observe(a, frame, after, "none")
        if player.expect_win:                                              # a goal-directed plan ended without a score
            player.goal.refute("win", a, frame)                            # -- hindsight: its keys were wrong here
            player.stats["goal_refutations"] = player.stats.get("goal_refutations", 0) + 1
            player.expect_win = False
        if pred is not None and pred.shape == after.shape:
            player.stats["predictions"] += 1
            player.stats["correct"] += int(np.array_equal(pred, after))
            player.stats["cells"] += int(after.size)
            player.stats["unknown_cells"] += int(_unknown or 0)
            if not np.array_equal(pred, after):
                player.plan = []
                player.stats["replans"] += 1
        if used >= budget:
            results.append(dict(level=level, solved=False, actions=used))
            if verbose:
                print(f"   level {level} NOT solved in {used} actions")
            break
    for r in results:
        r["model"] = {a.name: {"radius": player.models[a].best().r if player.models[a].n_obs else None,
                               "windows": len(player.models[a].best().table) if player.models[a].n_obs else 0}
                      for a in player.actions}
    if return_player:
        return results, player.stats, player
    return results, player.stats
