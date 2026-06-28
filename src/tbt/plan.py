"""The planner -- one active-inference value over the whole object configuration; the self EMERGES, never labelled.

This is the harness dissolved AND the self dissolved. There is no controllable agent here: the rollout state is the
configuration of ALL objects, and `step` applies EVERY object's learned operator under the hypothesised action. Most
objects are unaffected by a given action (their operator is identity, or action-independent); the action-sensitive
one(s) move. So when the planner follows value, it ends up moving whatever the actions control -- the self-model
*falls out of planning*, it is not a variable.

`act` minimizes expected free energy (active inference): value = PRAGMATIC (reach the rewarding configuration) +
EPISTEMIC (resolve uncertainty), no epsilon, no explore/exploit switch. Three drives, arbitrated by value:

  * MOTOR BABBLING -- an action never TAKEN is maximally uncertain (its effect on the world is unknown), so optimism
    under uncertainty (R-MAX) makes it the top target; each action is tried once, then the optimism is spent.
  * NOVELTY -- a configuration never visited is a frontier TARGET to route to (directed exploration).
  * GOAL -- the score-rewarded configuration is a +1 terminal (exploit). It outweighs novelty, so a known goal is
    pursued; with none yet, novelty drives discovery.

The pragmatic rollout treats unexplored configurations as traversable (trust the learned operators). The search is
the reused `Neocortex.achieve`. DOMAIN-GENERAL: `act` consumes objects, their per-object forward models, the available
actions, and which have been tried -- nothing about a grid, colours, a self, or a mechanic. Pure stdlib.
"""

from __future__ import annotations

from .goal import config_state
from .neocortex import Neocortex


class Planner:
    """Active-inference action selection over the object configuration. Holds the goal model and the achiever; `act`
    is handed the current objects, the per-object forward models, the available actions, and the set of actions tried
    so far (for babbling). `cap` bounds the rollout (compute, cheap); `novelty` is the epistemic frontier reward."""

    def __init__(self, goal, cap: int = 600, gamma: float = 0.95, novelty: float = 0.05, seed: int = 0):
        self.goal = goal
        self.cap = cap
        self.novelty = novelty
        self._optimism = 1.0 / (1.0 - gamma)                   # R-MAX optimistic value for a never-tried action
        self.neo = Neocortex(gamma=gamma, seed=seed)

    def reset(self):
        self.neo.reset()

    def act(self, objects, forwards, actions, tried):
        """The action to take. `objects = {id: (pose, size)}`, `forwards = {id: ForwardModel}`, `actions` = available
        action keys, `tried` = actions taken so far. Returns the action key, or None if no actions are available."""
        if not actions:
            return None
        goal, optimism, novelty = self.goal, self._optimism, self.novelty
        sizes = {oid: size for oid, (_pose, size) in objects.items()}
        # keep REAL poses in the rollout (predict adds integer deltas, so they stay exact) -- rounding here would make
        # the rollout's config_state disagree with perception's, and the goal would never be recognised mid-rollout.
        start = tuple(sorted((oid, pose) for oid, (pose, _size) in objects.items()))

        def advance(state, key):
            return tuple(sorted(
                (oid, forwards[oid].predict(pose, key) if oid in forwards else pose) for oid, pose in state))

        def to_objects(state):
            return {oid: (pose, sizes.get(oid, 0)) for oid, pose in state}

        def pragmatic(state, a):                               # reach the goal / try an untried action
            key = actions[a]
            if key not in tried:                               # an action never taken -> TRY it (motor babbling)
                return ("?", a), optimism, True
            nxt = advance(state, key)
            if goal.is_goal(to_objects(nxt)):                  # the learned goal configuration -> exploit
                return nxt, 1.0, True
            return nxt, 0.0, False                             # unexplored OR visited: traversable toward the goal

        a = self.neo.achieve(pragmatic, start, len(actions), max_states=self.cap)
        if self.neo.root_value > 1e-6:                         # a goal or an untried action is reachable -> pursue it
            return actions[a]

        def epistemic(state, a):                               # nothing to exploit -> seek the nearest new configuration
            nxt = advance(state, actions[a])
            if goal.visits(to_objects(nxt)) == 0:              # an unvisited configuration is a frontier TARGET
                return nxt, novelty, True
            return nxt, 0.0, False

        return actions[self.neo.achieve(epistemic, start, len(actions), max_states=self.cap)]
