# Handoff — Phase 2 Step B: the rollout planner OVER the multi-column spine

You are continuing the **TBT-faithful refactor** of Cipher's TBT agent (target: ARC-AGI-3). This doc is the
concrete, runnable handoff. **Read these two first** (they hold the full plan + history + research):
- Memory: `project_reorient_and_reconnect.md` (flagged READ-FIRST in `MEMORY.md`) — the whole plan, Phase 2, and
  the Step-B design + corrections.
- `src/tbt/RESEARCH.md` **R10** — the TBT research (Monty/Thousand-Brains-Systems + the dopamine active-inference
  model) that grounds this refactor, with sources.

## The discipline (non-negotiable — Cipher is exacting about these)
- **The bitter lesson.** No hand-coded rules, no role-branches, no typed sub-goals (cover/reach/affordance). Every
  capability EMERGES from a learned model + value. Every time we hand-coded (BFS, typed sub-goals) we failed; every
  time we used general column-learning we won.
- **ONE model, full multi-column.** NEVER strip the thalamus / basal ganglia / multiple columns. We NEED multiple
  communicating columns (capacity / 2^K-avoidance, different reference frames, compositionality) for bigger
  problems. The rollout planner is the PLANNER **over** that multi-column world model — not a replacement for it.
- **Fewer scripts at the end** than the start (this refactor deletes more than it adds).
- **Never run training on this machine** (CPU demos/tests are fine; large training is the user's GPU job).
- Always use the venv. Pause and answer questions directly; don't barrel past them.

## Where things stand (2026-06-27)
- **Step A DONE + committed (`82b9bcd`):** the cortical column IS the forward model — the conditional-dynamics
  faculty (`observe_effect` / `learn_dynamics` / `predict_effect`) was folded into `tbt/column.py`; `tbt/dynamics.py`
  and `tbt/forward.py` are deleted (24 → 22 agent files). Behaviour identical; full suite passes.
- **The agent is autonomous:** F's cold-start (`perception/learn.py` `WorldLearner` + `tbt/agent.py`
  `Agent.explore_and_learn`) learns the whole world model from pixels + the sparse score, no injected roles —
  `demos/cold_start.py` solves LockPath 4/4 + MultiKey 2/2 from scratch.
- **The CURRENT planner** (`tbt/neocortex.py` driven by `perception/control.py` `NeocortexPlanner`, with the old
  `tbt/planner.py` still present) uses TYPED sub-goals over a discrete cell-graph. It solves Sokoban 4/4 / LockPath
  4/4 / MultiKey 2/2 but CANNOT do CollectAll or Toggle. **Keep it working until the rollout replaces it.**
- **Step B prototype VALIDATED (4/5 games):** the unified rollout achiever below solves **LockPath 4/4, Sokoban
  3/3, CollectAll 3/3 (the typed planner can't), MultiKey 2/2** — reach + cover + collect + the key→door affordance
  + hazard-avoidance ALL emerge from one achiever + SIGNED value. Only **Toggle** fails (its door is invisible
  while open → needs episodic memory of the switch→door effect; do it last).

## Step B — the build (the rollout planner OVER the multi-column spine)
The prototype is a **single-column stand-in** that proved the PLANNING mechanism. The integration must make its
hand-coded factoring **emerge from the spine** (Cipher insisted — a single-column planner can't scale):

| prototype (hand-coded, single forward model) | faithful integration (multi-column) |
|---|---|
| `step()` = one monolithic forward model | forward model = **compose the active columns' predictions via the thalamus/CMP** (agent column → move; focus-mover column → push, *voted against* the absolute map = the egocentric⊗absolute lateral vote; dynamics column → effects) |
| perception picks "the focus mover" | the **basal ganglia gates** the active object-column (emergent focus) |
| perception hand-sequences sub-goals | the **task column** holds sub-goals; the **thalamus routes** the active goal-state (top-down CMP) |
| `(agent, focus-mover, doors)` tuple | the **factored joint of the active columns** — same shape, but factors are columns ⇒ +object = +column (additive), not ×state (2^K) |

**Build order (each step test-gated; keep the agent working; validate via `src/tests/test_agent.py`):**
1. **Extract the general rollout achiever → `tbt/neocortex.py`.** It takes a *forward-model callable* `step(state,
   action) -> (next_state, reward, done)` + does the prioritized-sweep (signed `tbt/reward.py`) + returns the
   greedy action. Decouples planning from the world model. (Signed value is FREE — `reward.py` already supports it
   with `beta=0, optimistic=False`; the prototype uses negative reward for death.)
2. **Make the forward model multi-column** — compose per-object column predictions through the thalamus (agent-move
   ⊗ focus-mover push voted vs the absolute map ⊗ dynamics-column effects). This is where "+object = +column".
3. **BG gates the active object-column + sub-goal** (emergent focus, not hand-picked).
4. **Wire into the one `Agent`** (perception builds the columns + drives the achiever), validate the FULL suite,
   then **DELETE `tbt/planner.py` + `perception/control.py`'s old path + the `openers` machinery**. (This is the
   net file reduction.)
5. **Toggle** last — needs the learned door-position (the door is invisible while open) + memory of switch→door.

## The validated prototype (copy to your scratchpad, run, confirm 4/5, then extend into `tbt/`)
Run: `PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe <path>.py` from the repo root. Expected: LockPath
L0–L3 won, Sokoban L0–L2 won, CollectAll L0–L2 won, MultiKey L0–L1 won.

```python
import sys, random
from collections import deque, defaultdict
sys.path.insert(0, "src")
from tasks import Environment
from tasks.games import LockPath, Sokoban, CollectAll, MultiKey
from tasks.games.lockpath import (C_AGENT, C_WALL, C_GOAL, C_KEY, C_DOOR, C_BLOCK, C_PAD, C_HAZARD,
                                   _LEVELS as LP)
from tasks.games.sokoban import _LEVELS as SK
from tasks.games.collectall import C_ITEM, _LEVELS as CA
from tasks.games.multikey import _LEVELS as MK
from perception.scene import WorldModel, Perception
from tbt.agent import Agent
from tbt.reward import RewardModel

DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


class RolloutPlanner:
    """Single-column STAND-IN proving the planning. ONE _achieve (prioritized-sweep over a factored state through
    a forward model + SIGNED value) does reach + cover + collect + the affordance + hazard-avoidance — all emerge.
    Integration: replace the hand-coded layout/forward-model/focus with the multi-column spine (see the table)."""

    def __init__(self, world, seed=0):
        self.world, self.seed = world, seed
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed); self._focus = None

    def new_level(self):
        self._focus = None

    def on_death(self):
        self._focus = None

    def _layout(self, scene):
        w, by = self.world, scene.by_color
        agent = scene.body_pos
        non_bg = {p for cells in by.values() for p in cells} | {agent}
        xs = [x for x, _ in non_bg]; ys = [y for _, y in non_bg]
        cells = {(x, y) for x in range(min(xs), max(xs) + 1) for y in range(min(ys), max(ys) + 1)}
        door_cols = ((set().union(*w.effects.values()) if w.effects else set()) |
                     (set().union(*w.adds.values()) if w.adds else set())) - w.pushable
        walls = {p for c in w.blocking if c not in door_cols for p in by.get(c, ())}
        deaths = {p for c in w.death for p in by.get(c, ())}
        movers = {p for c in w.pushable for p in by.get(c, ())}
        door_at = {p: c for c in door_cols for p in by.get(c, ())}
        opener = {p: set(rem) for tc, rem in w.effects.items() for p in by.get(tc, ())}
        closer = {p: set(add) for tc, add in w.adds.items() for p in by.get(tc, ())}
        removed0 = frozenset(c for c in door_cols if not by.get(c))
        return cells, walls, deaths, movers, door_at, opener, closer, removed0

    def _free(self, t, cells, walls, others, door_at, removed):
        return (t in cells and t not in walls and t not in others
                and not (t in door_at and door_at[t] not in removed))

    def _achieve(self, target, mover, agent, lay):
        cells, walls, deaths, movers, door_at, opener, closer, removed0 = lay
        others = movers - ({mover} if mover else set())

        def step(state, a):
            ag, mv, removed = state
            dx, dy = DELTAS[a]; t = (ag[0] + dx, ag[1] + dy)
            if not self._free(t, cells, walls, others, door_at, removed):
                return state, 0.0, False
            nmv = mv
            if mv is not None and t == mv:
                b = (t[0] + dx, t[1] + dy)
                if not self._free(b, cells, walls, others, door_at, removed):
                    return state, 0.0, False
                nmv = b
            nrem = (set(removed) | opener.get(t, set())) - closer.get(t, set())
            if t in deaths:
                return (t, nmv, frozenset(nrem)), -1.0, True
            done = (nmv == target) if mover else (t == target)
            return (t, nmv, frozenset(nrem)), (1.0 if done else 0.0), done

        start = (agent, mover, removed0)
        T, preds, R, term = {}, defaultdict(list), {}, set()
        seen, q = {start}, deque([start])
        while q:
            s = q.popleft(); row = []
            for a in range(4):
                ns, r, d = step(s, a)
                row.append(ns)
                if r:
                    R[ns] = r
                if d:
                    term.add(ns)
                if ns not in seen:
                    seen.add(ns); q.append(ns)
                preds[ns].append(s)
            T[s] = [] if s in term else row
        rm = RewardModel(max(2, len(T)), gamma=0.95, beta=0.0, prioritized=True, optimistic=False)
        rm.R_ext = R
        for ts in term:
            rm._push(ts, 1.0)
        rm.budget = 6 * len(T)
        rm.plan(T, preds, start)
        if not T[start]:
            return self.rng.randrange(4)
        vals = [rm.V[ns] for ns in T[start]]
        m = max(vals)
        return self.rng.choice([a for a, v in enumerate(vals) if v == m])

    def act(self, scene, explore=0.0):
        agent = scene.body_pos
        if agent is None:
            return self.rng.randrange(4)
        lay = self._layout(scene)
        movers = lay[3]
        req_visible = {p for c in self.world.required_absent for p in scene.by_color.get(c, ())}
        for C in sorted(scene.req_cells):
            if C in req_visible:
                free = [m for m in movers if m not in scene.req_cells]
                if free:
                    if self._focus not in free:
                        self._focus = min(free, key=lambda m: abs(m[0] - C[0]) + abs(m[1] - C[1]))
                    return self._achieve(C, self._focus, agent, lay)
                return self._achieve(C, None, agent, lay)
        goal = next(iter(scene.goal_cells), None)
        if goal is None:
            return self.rng.randrange(4)
        return self._achieve(goal, None, agent, lay)


def _w(**kw):
    base = dict(body=C_AGENT, pushable=set(), blocking={C_WALL}, death=set(), effects={}, adds={},
                harmful=set(), goal_colors={C_GOAL}, required_absent=set())
    base.update(kw); return WorldModel(**base)


SUITE = [
    ("LockPath", LockPath, LP, _w(pushable={C_BLOCK}, death={C_HAZARD}, effects={C_KEY: {C_DOOR}}, required_absent={C_PAD})),
    ("Sokoban", Sokoban, SK, _w(pushable={C_BLOCK}, required_absent={C_PAD})),
    ("CollectAll", CollectAll, CA, _w(goal_colors=set(), required_absent={C_ITEM})),
    ("MultiKey", MultiKey, MK, _w(death={C_HAZARD}, effects={4: {5}, 9: {10}})),
]
for name, cls, levels, world in SUITE:
    print(f"{name}:")
    for lvl in range(len(levels)):
        a = Agent(Perception(world), RolloutPlanner(world, seed=0))
        out = a.play(Environment(cls(levels=[levels[lvl]])), max_steps=400)
        print(f"  L{lvl}: won={out.won} actions={out.actions}")
```

## Run / validate
- Tests: `PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe -m pytest src/tests/ -q`
- Demos: from `src/`, `PYTHONIOENCODING=utf-8 PYTHONPATH=. ../venv/Scripts/python.exe -m demos.cold_start`
- Acceptance (bitter-lesson gate): `grep -niE 'grid|colou?r|door|pad|key|fire|cover|pushable|blocking|subgoal'
  src/tbt/<changed>.py` should return only disclaimers / generic terms — NO task-structural code in `tbt/`.

## Key source files
- `tbt/neocortex.py` — the planner (CURRENT: typed sub-goals + thalamus routing; TARGET: the rollout achiever).
- `tbt/column.py` — the cortical column (now also the forward model: the dynamics faculty + the SR-frame + recurrence).
- `tbt/thalamus.py` / `tbt/basal_ganglia.py` / `tbt/reward.py` — the spine (CMP routing / gate / SIGNED value).
- `perception/scene.py` — `StateEncoder` (scene → planner inputs) + `WorldModel` (the decoded roles) + `build_world`.
- `perception/control.py` — `NeocortexPlanner` (the adapter; will build the multi-column forward model).
- `perception/learn.py` — `WorldLearner` (F's cold-start).
- `tbt/planner.py` — the OLD enumeration planner (to DELETE once the rollout replaces it).
- `src/tests/test_agent.py` — the one-agent suite (Sokoban + LockPath + cold-start). Add CollectAll/Toggle here.

When Step B is done, **delete this handoff doc** (it's scaffolding) and revert the CLAUDE.md pointer.
