"""The planner -- one active-inference value that EXPLORES to learn and EXPLOITS to score (the harness dissolved).

This is the harness dissolved. `perception/control.py`'s `NeocortexPlanner` reaches the same achiever, but only
after a fat role-aware glue layer (a `_Layout` of decoded body/pushable/blocking roles, a hand-coded `DELTAS` move
set, an SR-frame `CorticalColumn` over enumerated cells) and a SEPARATE `_explore` branch with an epsilon. None of
that survives the rebuild: real games announce no roles, the action set and its effects are LEARNED, and exploration
is not a special mode -- it is the epistemic half of one value.

So `act` minimizes expected free energy (active inference): value = PRAGMATIC (reach the rewarding state) + EPISTEMIC
(resolve the agent's uncertainty). No epsilon, no hand-set switch; the arbitration emerges from comparing the value
of the two, which carry three drives:

  * MOTOR BABBLING (operator uncertainty) -- an action whose operator was never observed is maximally uncertain, so
    optimism under uncertainty (R-MAX: value = the max return 1/(1-gamma)) makes the unknown the most attractive
    target. The agent tries each action ONCE to learn its operator, then the optimism is spent (self-limiting). This
    is developmental motor babbling; reafference (`events.py`) and the bottom-up `retina.salient_cells` ("what
    moved") confirm the effect -> `objects` -> `forward` learns the operator. Babbling, salience and operator-learning
    are one bootstrap loop.
  * GOAL (pragmatic) -- the score-rewarded configuration (`goal.py`) is a +1 terminal. The pragmatic rollout treats
    unexplored arrangements as TRAVERSABLE (the agent trusts its learned operators to path toward a known goal
    through cells it has not personally visited -- the action-cheapest choice; a misprediction is corrected by the
    prediction error). So a known goal is pursued directly.
  * NOVELTY (epistemic) -- when nothing pragmatic has value (no goal reachable, no untried action), the agent heads
    to the nearest arrangement it has never visited: an unvisited config is a small-reward TERMINAL TARGET, so the
    planner routes to the frontier rather than drifting to wherever it can SEE the most novelty. This is directed
    frontier exploration -- discover the score in far fewer actions than a random walk.

`act` therefore runs the pragmatic plan first; if it has value (a goal or an untried action is reachable) it is
taken (exploit / babble), otherwise the epistemic plan is taken (explore). Both are the one reused `Neocortex.achieve`
-- the arbitration is the active-inference argmax, biologically the locus-coeruleus phasic/tonic shift (engage a
reachable reward, else range for the unknown) emerging from the value rather than a coded mode. The SAME epistemic
currency ("act where you will learn the most") is what a saccade/attention policy will spend, so exploration with the
body and attention with the sensors are one quantity -- the seam this shares with the retina.

DOMAIN-GENERAL (the Danganronpa litmus): `act` consumes `(self pose, [(identity, pose), ...], available actions)` and
nothing about a grid, colours, or a mechanic. Nothing is reimplemented. Pure stdlib over the reused achiever.
"""

from __future__ import annotations

from .neocortex import Neocortex


class Planner:
    """Active-inference action selection over the learned forward model + goal, with directed exploration folded into
    the value (no epsilon, no separate explore branch). Hold the (mutable, still-learning) `forward` and `goal`
    models; `act(self_pose, others, actions)` returns the action key to take, where `actions` is the env's available
    set (its membership is given; its EFFECTS are learned). `cap` bounds the rollout (compute, which is cheap);
    `novelty` is the epistemic frontier reward (only used to choose AMONG frontiers, so its size is not delicate)."""

    def __init__(self, forward, goal, cap: int = 600, gamma: float = 0.95, novelty: float = 0.05, seed: int = 0):
        self.forward = forward
        self.goal = goal
        self.cap = cap
        self.novelty = novelty
        self._optimism = 1.0 / (1.0 - gamma)                   # R-MAX optimistic value for a never-tried operator
        self.neo = Neocortex(gamma=gamma, seed=seed)

    def reset(self):
        self.neo.reset()

    def act(self, self_pose, others, actions):
        """The action to take from `self_pose` given the other objects `others = [(identity, pose), ...]` and the
        env's `actions`. Plans pragmatically (reach the goal, or babble an untried operator) and, only if that has no
        value, epistemically (reach the nearest unvisited arrangement). Returns the action key, or None if `actions`
        is empty."""
        if not actions:
            return None
        others = [(ident, (round(px), round(py))) for ident, (px, py) in others]
        start = (round(self_pose[0]), round(self_pose[1]))
        forward, goal, optimism, novelty = self.forward, self.goal, self._optimism, self.novelty

        def pragmatic(state, a):
            key = actions[a]
            if forward.delta(key) is None:                     # operator never observed -> TRY it (motor babbling):
                return ("?", a), optimism, True                # optimism under uncertainty makes the unknown attractive
            nxt = forward.predict(state, key)
            if goal.is_goal(nxt, others):                      # the learned goal -> pragmatic terminal (+1): exploit
                return nxt, 1.0, True
            return nxt, 0.0, False                             # unexplored OR visited: traversable toward the goal

        a = self.neo.achieve(pragmatic, start, len(actions), max_states=self.cap)
        if self.neo.root_value > 1e-6:                         # a goal or an untried operator is reachable -> pursue it
            return actions[a]

        def epistemic(state, a):                               # nothing to exploit -> seek the nearest new arrangement
            nxt = forward.predict(state, actions[a])
            if goal.visits(nxt, others) == 0:                  # an unvisited config is a FRONTIER to go see (a target,
                return nxt, novelty, True                      # so the agent routes TO it) -- visited = traversable
            return nxt, 0.0, False

        return actions[self.neo.achieve(epistemic, start, len(actions), max_states=self.cap)]
