"""The neocortex — the ROLLOUT PLANNER over the multi-column world model (active inference, signed value).

This is the §5/§6 control loop, Phase-2 form: ONE general achiever plans by ROLLING a forward model and scoring
the reached states by SIGNED value — reach, cover, collect, the key→door affordance and hazard-avoidance all
EMERGE from that one loop, with NO typed sub-goals, NO role branches, NO cell/colour here (the bitter lesson).

The PLANNER, not the world model. It is handed a forward-model callable `step(state, action) -> (next_state,
signed_reward, done)` — the COMPOSED prediction of the active columns (perception builds it over the spine: the
spatial map column's agent-move, the basal-ganglia-gated focus object-column's egocentric push voted against the
absolute map, the dynamics column's effects) — and it:

  achieve  : roll `step` from a start state (reachable-state sweep), value the graph by prioritized replay
             (`reward.py`, Mattar–Daw gain×need) over SIGNED value (death = a terminal −1, success = +1), and
             return the greedy action. Active inference: pick the action that closes the gap to the desired state.

The thalamus + basal ganglia STAY — they ARE the multi-column world model the rollout plans over, never stripped:
  route_goals / goal_node : top-down CMP — the task column holds the sub-goals, the thalamus routes the active
                            one's goal-state into the spatial column's reference frame (R9 / Thousand Brains).
  gate_focus              : the basal ganglia gates the active object-column (the emergent focus), so WHICH object
                            the rollout factors on is selected by learned value (Go/NoGo + dopamine), not hand-picked.

Everything here is OPAQUE — states are tuples, actions are indices, the goal-state is a node handed in by
perception. Adding an object adds a column for the gate to select among (additive), never a dimension of the
rolled state (no 2^K). The signed value is FREE (`reward.py` with beta=0, optimistic=False).
"""

from __future__ import annotations

import random
from collections import defaultdict, deque

from tbt.column import CorticalColumn
from tbt.basal_ganglia import BasalGanglia
from tbt.reward import RewardModel
from tbt.thalamus import Thalamus


class Neocortex:
    """The rollout achiever over the multi-column spine (thalamus + basal ganglia retained as the world model)."""

    def __init__(self, n_columns: int = 8, gamma: float = 0.95, seed: int = 0):
        self.thal = Thalamus()                                  # top-down goal-state routing (CMP)
        self.bg = BasalGanglia(n_columns=n_columns, seed=seed)  # gates the active object-column (focus)
        self.rm = RewardModel(1, gamma=gamma, beta=0.0, prioritized=True, optimistic=False)   # SIGNED value field
        self.rng = random.Random(seed)
        self.root_value = 0.0                                   # the planned value of the last chosen action (so a
        #                                                        caller can arbitrate pragmatic vs epistemic plans)

    def reset(self):
        self.bg.gate_reset()                                    # focus-gate affinity is within-level only

    # ---- the general rollout achiever: roll the forward model, value it (signed), act --------------------
    def achieve(self, step, start, n_actions, max_states=None):
        """Roll the forward-model callable `step` from `start` — a breadth-first sweep over the reachable factored
        states — then value that graph by prioritized replay over SIGNED reward (terminals seed it; an aversive
        terminal carries −1, a success +1) and return the greedy action. The achiever is domain-blind: reach,
        cover, collect, the affordance and hazard-avoidance are not cases here — they are what rolling a model and
        following signed value DOES. `start` has no successors ⇒ a random move (nothing to plan).

        `max_states` BOUNDS the rollout for large / time-evolving worlds (Tetris: you can stack pieces almost
        forever without ever hitting a terminal, so enumerate-to-terminal explodes). Capping the frontier keeps it
        tractable; per-step re-planning then keeps the goal within the cap as progress is made. None (the default)
        = unbounded — byte-identical to before for the small factored worlds whose graph never reaches the cap."""
        T, preds, R, term = {}, defaultdict(list), {}, set()
        seen, q = {start}, deque([start])
        while q:
            s = q.popleft()
            row = []
            for a in range(n_actions):
                ns, r, done = step(s, a)
                row.append(ns)
                if r:
                    R[ns] = r                                  # signed: success > 0, aversive < 0
                if done:
                    term.add(ns)
                    T[ns] = []                                 # a terminal is a leaf — always recorded, never expanded
                elif ns not in seen and (max_states is None or len(seen) < max_states):
                    seen.add(ns)
                    q.append(ns)
                preds[ns].append(s)
            T[s] = row                                         # s was popped ⇒ non-terminal (terminals aren't enqueued)
        rm = self.rm
        rm.R_ext = R
        rm.V.clear()
        rm.queue.clear()
        for ts in term:
            rm._push(ts, 1.0)                                  # back up FROM every terminal (reverse replay)
        rm.budget = 6 * len(T)
        rm.plan(T, preds, start)
        if not T.get(start):
            self.root_value = 0.0
            return self.rng.randrange(n_actions)
        vals = [rm.V[ns] for ns in T[start]]
        m = max(vals)
        self.root_value = m                                    # expose the chosen action's value (caller may arbitrate)
        return self.rng.choice([a for a, v in enumerate(vals) if v == m])

    # ---- top-down CMP: the task column's sub-goals → goal-states in the spatial column (the thalamus) ----
    def route_goals(self, subgoals, space_col):
        """`subgoals = [(content, node), …]` (content = a sub-goal id, node = its goal-state in `space_col`'s
        frame). Build the task column (one content code per sub-goal) and bind each content ⊗ its goal node's
        place into the thalamus register — the shared blackboard. Returns `(task_col, R, inv)` for `goal_node`."""
        task_col = CorticalColumn(n_entities=max(1, len(subgoals)))
        inv = {i: s for s, i in space_col.loc.items()}         # frame index → node (read_location returns an index)
        R = self.thal.bind(task_col, space_col, subgoals) if subgoals else None
        return task_col, R, inv

    def goal_node(self, R, task_col, space_col, inv, content, fallback):
        """Top-down read: the goal-state node bound to `content`, via the thalamus (transpose bind). Falls back to
        the directly-bound node if the read misses — the CMP channel proposes, the binding guarantees."""
        if R is None:
            return fallback
        idx = self.thal.read_location(R, task_col, space_col, content)
        return inv.get(idx, fallback) if idx is not None else fallback

    # ---- the basal ganglia gates the active object-column (the emergent focus) ---------------------------
    def gate_focus(self, options, values):
        """Select the focus object-column among `options` by learned value (the BG gate: Go disinhibits the
        highest critic-value + affinity, dopamine reinforces it). So WHICH object the rollout factors on EMERGES
        from value, not a hand-picked nearest. Returns the chosen option (or None if there are none)."""
        if not options:
            return None
        return options[self.bg.gate(list(options), list(values))]
