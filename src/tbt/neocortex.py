"""The neocortex — the multi-column control loop (task ⊕ space), columns coordinating by CMP messages.

This is the §5/§6 control loop made real: a TASK column sequences sub-goals and a SPATIAL column navigates to
them, joined ONLY by messages through the thalamus — never fused into one fat column (the 2^K conjunctive
explosion, architecture doc §2). Every inter-column message is in the **Cortical Messaging Protocol**
(RESEARCH.md R9 / Thousand Brains Project): a `(content, pose, confidence)` triple — *what*, *where in the
receiver's reference frame*, *how sure*. Columns exchange BELIEFS, never raw input.

  top-down  (task → space):  the active sub-goal's content → the goal-state (a node) bound to it  [thalamus.read_location]
  bottom-up (space → task):  the reached node → which sub-goal (if any) is bound there  [thalamus.read]

The spatial column navigates by SR prioritized-replay (`reward.py`, Mattar–Daw gain×need) — value propagated FROM
the goal-state, the architecture's vector-nav, NOT exhaustive search. "Navigate" generalises to *move any ENTITY
to a node*: the AGENT moves itself; a MOVER moves by the agent positioning behind it and pushing — the egocentric
relational affordance bound to the absolute map (the relational mechanic FACTORED, never an agent×mover joint
search). Each navigation is one entity over cells (N²), K sub-goals sequenced additively.

Everything domain-specific is OPAQUE: sub-goals are `(content, node, is_mover)` and the cell graph `T`/`preds` and
entity positions are handed in by perception. No grid / colour / door here. Lateral VOTING (the egocentric ⊗
absolute consensus under ambiguity) and the BG gate as the learned sub-goal selector are next; the spine + the
factored relational navigator are here.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from tbt.column import CorticalColumn
from tbt.reward import RewardModel
from tbt.thalamus import Thalamus


@dataclass
class CMP:
    """A Cortical Messaging Protocol message — the one currency between columns (R9)."""
    content: object              # WHAT — a sub-goal id / object label
    pose: object                 # WHERE — a node in the receiver column's reference frame
    confidence: float = 1.0


class Neocortex:
    """Task ⊕ space columns, joined by CMP messages; spatial nav by SR prioritized-replay; relational mechanics
    FACTORED into the agent + one mover (egocentric ⊗ absolute), never the joint."""

    def __init__(self, gamma: float = 0.9, seed: int = 0):
        self.thal = Thalamus()
        self.rm = RewardModel(1, gamma=gamma, beta=0.0, prioritized=True, optimistic=False)   # the SR value field
        self.rng = random.Random(seed)
        self.space = self.task = None                          # the columns are (re)bound per level by bind()
        self.reset()

    def reset(self):
        self.R, self._sub, self._done, self._active, self._focus, self._inv = None, [], set(), None, None, {}

    def bind(self, subgoals, space_col, cid):
        """`subgoals = [(content, node, is_mover), …]`, ordered (terminal last). `space_col` is the spatial
        column (perception-built, one per level); `cid` maps a node (cell) to its symbol in that column. Build
        the task column (one content code per sub-goal), then bind each content ⊗ its goal-state's place into the
        thalamus register — the shared blackboard the columns read both ways. `is_mover` = the sub-goal is met by
        a MOVER on the node (a relational push), not the agent on it."""
        self.space = space_col
        self.task = CorticalColumn(n_entities=max(1, len(subgoals)))
        self._sub, self._done, self._active, self._focus = list(subgoals), set(), None, None
        by_sym = {i: n for n, i in cid.items()}                # symbol → node
        self._inv = {f: by_sym[s] for s, f in space_col.loc.items()}   # place-frame index → node (what read_location
        self.R = (self.thal.bind(self.task, self.space, [(c, cid[n]) for c, n, _ in subgoals])   # returns); robust to
                  if subgoals else None)                                                          # loc ≠ identity

    # ---- the CMP channels (through the thalamus) ----------------------------------------------------------
    def _goal_node(self, content):
        idx = self.thal.read_location(self.R, self.task, self.space, content)   # top-down: the bound goal-state
        return self._inv.get(idx) if idx is not None else None

    # ---- the spatial column's navigation: obstacle-aware SR prioritized-replay (not BFS) ------------------
    def _filter(self, T, obstacles):
        """The cell graph with `obstacles` removed (a move into one becomes a self-loop) — the column routing
        AROUND parked/other movers, which the static map does not know about."""
        T2, preds2 = {}, defaultdict(list)
        for c, row in T.items():
            if c in obstacles:
                continue
            r2 = [nb if nb not in obstacles else c for nb in row]
            T2[c] = r2
            for nb in r2:
                preds2[nb].append(c)
        return T2, preds2

    def _sr(self, T, preds, start, goal):
        """Greedy on the SR value field, value propagated from `goal` by prioritized sweeping. Returns
        `(action, next_node)`, or `(None, start)` if already there / unreachable."""
        if start == goal or start not in T or goal not in T:
            return None, start
        rm = self.rm
        rm.R_ext = {goal: 1.0}; rm.V.clear(); rm.queue.clear()
        rm._push(goal, 1.0); rm.budget = 4 * len(T)
        rm.plan(T, preds, start)
        vals = [rm.V[nx] for nx in T[start]]
        m = max(vals)
        a = self.rng.choice([i for i, v in enumerate(vals) if v == m])
        return a, T[start][a]

    # ---- the affordance: open a door by reaching its trigger (means-ends, emergent from learned effects) --
    def _component(self, T, start):
        """The cells reachable from `start` in graph T — BFS over the per-action neighbour lists (a wall is a
        self-loop, so it lands back on a seen cell and stops)."""
        seen, stack = {start}, [start]
        while stack:
            for nb in T.get(stack.pop(), ()):
                if nb not in seen:
                    seen.add(nb); stack.append(nb)
        return seen

    def _unblock(self, Tf, agent, goal, openers):
        """If `goal` is UNREACHABLE in `Tf`, return the nearest reachable trigger that would open a door (any door
        — re-evaluated every step, so they open one at a time = factored, never 2^doors); else None. `openers =
        {shut-door cell: [trigger cells]}` is perception's door↔opener join, so WHICH trigger opens WHICH door IS
        the learned effect — reaching a 'key' to clear a 'door' EMERGES here, with no hardcoded key/door/fire."""
        comp = self._component(Tf, agent)
        if goal in comp:
            return None                                        # already reachable — no door to open
        cand = [t for trig in openers.values() for t in trig if t in comp]   # openers we can reach right now
        if not cand:
            return None
        return min(cand, key=lambda t: abs(t[0] - agent[0]) + abs(t[1] - agent[1]))

    # ---- the egocentric ⊗ absolute factored relational navigator -----------------------------------------
    def _push(self, mover, target, agent, movers, T):
        """Move `mover` one step toward `target`, FACTORED: (1) SR-nav the mover toward the target [absolute],
        (2) SR-nav the agent to the cell behind the mover [absolute], (3) when there, move into it [egocentric
        affordance — the learned push]. Other movers are walls in both navigations; the agent×mover joint is
        never searched."""
        a_b, n = self._sr(*self._filter(T, set(movers) - {mover}), mover, target)
        if a_b is None or n == mover:
            return self.rng.randrange(len(T[agent]))
        behind = (2 * mover[0] - n[0], 2 * mover[1] - n[1])    # the node opposite the mover's next node
        if agent != behind:
            a, _ = self._sr(*self._filter(T, set(movers)), agent, behind)   # agent-nav, all movers are walls
            return a if a is not None else self.rng.randrange(len(T[agent]))
        for a, nb in enumerate(T[behind]):                     # at the push-node → the action onto the mover
            if nb == mover:
                return a
        return self.rng.randrange(len(T[agent]))

    # ---- one step of the loop: acknowledge, sequence, route a goal-state, navigate ------------------------
    def act(self, agent, movers, T, openers=None):
        """`agent`/`movers` = current cells; `T` = the cell graph {cell: [neighbour per action]} (a wall = a
        self-loop). `openers = {shut-door cell: [trigger cells]}` (perception's door↔opener join): when the active
        goal-state sits behind a shut door, the loop reaches an opener FIRST — the affordance sub-goal, EMERGENT
        from the learned effect, factored one door at a time. `preds` is not an argument — `_filter` recomputes it."""
        for c, node, im in self._sub:                          # bottom-up CMP: which sub-goals are now satisfied?
            if c not in self._done and ((im and node in movers) or (not im and agent == node)):
                self._done.add(c)
        remaining = [(c, node, im) for c, node, im in self._sub if c not in self._done]
        if not remaining:
            return self.rng.randrange(len(T[agent]))
        c, node, is_mover = remaining[0]                       # ordered reveal: the first unmet sub-goal (terminal last)
        if c != self._active:                                  # a new sub-goal → drop the committed mover
            self._active, self._focus = c, None
        goal = self._goal_node(c)                              # top-down CMP: the goal-state
        if goal is None:                                       # thalamus read missed → fall back to the bound node
            goal = node
        Tf, pf = self._filter(T, set(movers))                  # the agent's navigable graph (other entities = walls)
        if openers:                                            # AFFORDANCE: is the goal-state behind a shut door?
            trig = self._unblock(Tf, agent, goal, openers)
            if trig is not None:                               # yes → reach a reachable opener first (emergent sub-goal)
                a, _ = self._sr(Tf, pf, agent, trig)
                return a if a is not None else self.rng.randrange(len(T[agent]))
        if not is_mover:                                       # the AGENT reaches the goal-state
            a, _ = self._sr(Tf, pf, agent, goal)
            return a if a is not None else self.rng.randrange(len(T[agent]))
        occ = {nd for cc, nd, im in self._sub if im and cc in self._done}   # parked movers (done sub-goals)
        free = [m for m in movers if m not in occ]
        if self._focus not in free:                            # COMMIT one mover per sub-goal (re-picking flips it)
            if not free:
                return self.rng.randrange(len(T[agent]))
            self._focus = min(free, key=lambda m: abs(m[0] - goal[0]) + abs(m[1] - goal[1]))
        return self._push(self._focus, goal, agent, movers, T)
