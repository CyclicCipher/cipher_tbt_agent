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

    def __init__(self, goal, cap: int = 600, gamma: float = 0.95, novelty: float = 0.05,
                 contact: float = 0.5, seed: int = 0):
        self.goal = goal
        self.cap = cap
        self.novelty = novelty
        self.contact = contact                                 # salience: reward reaching an un-contacted object
        self.neo = Neocortex(gamma=gamma, seed=seed)

    def reset(self):
        self.neo.reset()

    def act(self, objects, forwards, actions, curiosity, context=None, contents=None, coord_actions=None):
        """The action to take. `objects = {id: (pose, size)}`, `forwards = {id: ForwardModel}`, `actions` = available
        action keys, `curiosity = {action: 0..1}` = how much there is still to LEARN about each action, `context(oid,
        pose, action_key) -> ctx` supplies each rolled-forward move's sensed context (so the rollout routes AROUND walls
        by value), `contents = {id: content}` adds each object's CONTENT to the rolled state (so a goal can be a state
        CHANGE, not only a spatial arrangement), and `coord_actions` = the keys that are PARAMETERIZED by a target (the
        click, ACTION6). A coordinate action is expanded into one choice PER perceived object -- the planner picks WHICH
        object to act on (the goal-state generator), the targeted object getting context 'clicked' -- so the click's
        effect is the same `(action, context)` operator and the parameter is chosen by value. Returns a plain action key,
        or `(coord_action, target_id)` for a click, or None if nothing pragmatic/salient is reachable."""
        if not actions:
            return None
        goal, contact = self.goal, self.contact
        coord = set(coord_actions or ())

        # The action set, with each PARAMETERIZED (coordinate) action expanded into "act on object X" for each perceived
        # object X -- the meaningful targets a goal-state generator proposes, not raw pixels.
        expanded = [a for a in actions if a not in coord]
        expanded += [(ca, oid) for ca in actions if ca in coord for oid in objects]

        def ecur(ea):                                          # how much there is to LEARN about this (expanded) action
            if isinstance(ea, tuple):
                ca, oid = ea
                fm = forwards.get(oid)
                return fm.curiosity(ca, "clicked") if fm is not None else 1.0   # have we tried clicking this object?
            return curiosity.get(ea, 1.0)

        # 1. CURIOSITY (motor babbling): an action still worth learning is just TRIED -- no rollout needed.
        curious = [ea for ea in expanded if ecur(ea) > 0.1]
        if curious:
            return max(curious, key=ecur)                     # practise the most-uncertain action (or click target)

        # rollout setup (reached only once the operators are learned). The state is (id, pose, content): an operator
        # moves the POSE and/or transforms the CONTENT -- movement, in-place change, and a click are ONE mechanism, so
        # the self, the goal, and the plan are KIND-agnostic. Keep REAL poses (predict adds integer deltas, exact).
        sizes = {oid: size for oid, (_pose, size) in objects.items()}
        cont = contents or {}
        aware = contents is not None
        start = tuple(sorted((oid, pose, cont.get(oid)) for oid, (pose, _size) in objects.items()))

        def advance(state, ea):
            key = ea[0] if isinstance(ea, tuple) else ea
            target = ea[1] if isinstance(ea, tuple) else None
            out = []
            for oid, pose, content in state:
                if oid in forwards:
                    fm = forwards[oid]
                    if target is not None:                    # a coordinate action: the targeted object is 'clicked'
                        ctx = "clicked" if oid == target else None
                    else:
                        ctx = context(oid, pose, key) if context else None
                    pose = fm.predict(pose, key, ctx)
                    content = fm.next_content(content, key, ctx)
                out.append((oid, pose, content))
            return tuple(sorted(out))

        def to_objects(state):
            return {oid: (pose, sizes.get(oid, 0)) for oid, pose, _c in state}

        def to_contents(state):
            return {oid: c for oid, _pose, c in state} if aware else None

        # 2. EXPLOIT a learned goal -- only roll if a goal has actually been seen (else this is wasted work).
        if goal.goals:
            def pragmatic(state, a):
                nxt = advance(state, expanded[a])
                if goal.is_goal(to_objects(nxt), to_contents(nxt)):    # the learned goal configuration -> exploit
                    return nxt, 1.0, True
                return nxt, 0.0, False                         # traversable toward the goal
            a = self.neo.achieve(pragmatic, start, len(expanded), max_states=self.cap)
            if self.neo.root_value > 1e-6:
                return expanded[a]

        # 3. SALIENT: reach an un-contacted object -- only roll if some object-pair is not yet contacted.
        szs = [size for _pose, size in objects.values()]
        pairs = {(min(szs[i], szs[j]), max(szs[i], szs[j])) for i in range(len(szs)) for j in range(i + 1, len(szs))}
        if pairs - goal.contacts:
            def salient(state, a):
                nxt = advance(state, expanded[a])
                if goal.new_contact(to_objects(nxt)):          # reach an un-contacted object (goals live there)
                    return nxt, contact, True
                return nxt, 0.0, False                         # else traversable -- a DISTANT contact stays reachable
            a = self.neo.achieve(salient, start, len(expanded), max_states=self.cap)
            if self.neo.root_value > 1e-6:
                return expanded[a]

        return None                                            # nothing to learn / exploit / contact -> caller's Lévy
