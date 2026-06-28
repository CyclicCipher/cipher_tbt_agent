"""The playing agent -- the continuous online loop, assembled with NO privileged self.

Thin by construction (it wires faculties). Each step it PERCEIVES the objects (`perceive.py`, tracked, no self),
LEARNS a per-object operator for EVERY tracked object (so the controllable one's operator becomes action-sensitive
and the rest stay identity/autonomous -- the self EMERGES, it is never named), LEARNS the goal from the score
(`goal.py`, over the whole configuration), and PLANS the next action (`plan.py`: babble -> explore -> exploit, over
configurations). The learned GOAL persists across levels (cross-level transfer of what scores); per-level the object
ids reset, so operators are re-learned by a cheap babble (instance-stable operator transfer is a later refinement).

One continuous interaction, no episodes. Generic over an env exposing `reset()/step(action)` and a frame with
`grid / score / level / available / is_win() / action_counter`. Pure stdlib.
"""

from __future__ import annotations

import random

from .agent import Outcome
from .events import EventSegmenter
from .forward import ForwardModel
from .goal import GoalModel
from .perceive import ObjectField
from .plan import Planner
from .retina import salient_cells


class Player:
    """The assembled agent, self-free. `act(grid, actions, score)` returns the next action; `run(env)` drives it.
    `reset()` is a new GAME (fresh learner); `new_level()` keeps the learned goal and re-localises."""

    def __init__(self, cap: int = 800, gamma: float = 0.95, novelty: float = 0.05, seed: int = 0):
        # cap bounds the directed rollout (compute). A live 64x64 game passes a board-sized cap so directed routing can
        # reach a VISIBLE distant object; the Lévy fallback covers the rest. Small scenes are bounded by reachability.
        self.cap, self.gamma, self.novelty, self.seed = cap, gamma, novelty, seed
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)
        self.field = ObjectField()
        self.goal = GoalModel()
        self.events = EventSegmenter()
        self.planner = Planner(self.goal, cap=self.cap, gamma=self.gamma, novelty=self.novelty, seed=self.seed)
        self.forwards: dict = {}                              # object_id -> ForwardModel (per-object operators)
        self._prev_objects = self._last = self._prev_frame = None
        self._prev_cells: dict = {}                           # last frame's per-object cells (for the move's context)
        self._prev_score = 0
        self._run = (None, 0)                                 # current Lévy run: (action, steps remaining)

    def new_level(self):
        """A level cleared: keep the learned goal (transfer); re-localise and re-learn operators (a cheap babble)."""
        self.planner.reset()
        self.events = EventSegmenter()
        self.field.reset()
        self.forwards = {}
        self._prev_objects = self._last = self._prev_frame = None
        self._prev_cells = {}
        self._run = (None, 0)

    def _levy(self, actions):
        """Heavy-tailed (Lévy) random search -- commit to one direction for a heavy-tailed number of steps (mostly
        short, occasionally long), the optimal memoryless search for sparse, unknown, far targets. Used only when the
        planner has nothing to learn/exploit/contact to route toward."""
        key, rem = self._run
        if key not in actions or rem <= 0:
            length = min(int(self.rng.paretovariate(1.5)), 25)   # Lévy-ish run length: usually small, rarely long
            key, rem = self.rng.choice(list(actions)), length
        self._run = (key, rem - 1)
        return key

    def _predictor(self, action):
        """Where each tracked object was expected to go under `action` (path integration via its learned operator) --
        used by perception to tell objects apart when they touch (identity from dynamics, not appearance)."""
        fwd = self.forwards
        def predict(oid, pose):
            fm = fwd.get(oid)
            return fm.predict(pose, action, None) if (fm is not None and action is not None) else pose
        return predict

    def _context(self, oid, pose, action, grid, cells):
        """The sensed context the move would enter: the MATERIAL at its destination cell -- another object's value (a
        wall/obstacle/target) -- or None for open background. The operator is conditioned on this, so a blocked move is
        a context-gated effect (not a binary obstacle flag). Reads the frame at pose + the action's base operator,
        skipping the object's OWN cells (a thing is not an obstacle to itself); off-grid -> a boundary context."""
        fm = self.forwards.get(oid)
        d = fm.delta(action, None) if fm is not None else None
        if not d or d == (0, 0):
            return None
        h = len(grid); w = len(grid[0]) if h else 0
        tx, ty = round(pose[0]) + d[0], round(pose[1]) + d[1]
        if not (0 <= ty < h and 0 <= tx < w):
            return "edge"
        if (tx, ty) in cells.get(oid, ()):                   # own cell -> not an obstacle to itself
            return None
        for o, cs in cells.items():
            if o != oid and (tx, ty) in cs:
                return grid[ty][tx]                          # destination occupied by another object -> its material
        return None                                          # open background

    def act(self, grid, actions, score):
        """Perceive the objects, learn each one's operator + the goal from the transition into this frame, then plan."""
        if not actions:
            return None
        objects = self.field.perceive(grid, self._predictor(self._last))
        if self._prev_objects is not None and self._last is not None:
            # the transitions of objects present in both frames, each with the context its move entered
            trans = [(oid, self._prev_objects[oid][0], pose,
                      self._context(oid, self._prev_objects[oid][0], self._last, self._prev_frame, self._prev_cells))
                     for oid, (pose, _size) in objects.items() if oid in self._prev_objects]
            # REAFFERENCE: a boundary is the change the operators CANNOT explain, not merely a big change. Subtract each
            # tracked object's predicted motion (the cells it vacates + enters) from the observed change; the residual is
            # the exafferent (world-caused) magnitude. Before operators exist the residual == the raw change (bootstrap),
            # and as operators sharpen the residual shrinks for explained moves -- the two co-bootstrap.
            observed = salient_cells(self._prev_frame, grid)
            explained = set()
            for (oid, _p0, _p1, ctx) in trans:
                fm = self.forwards.get(oid)
                if fm is not None:
                    pc = self._prev_cells.get(oid, set())
                    explained |= pc | fm.predict_cells(pc, self._last, ctx)
            boundary = self.events.is_boundary(len(observed - explained))
            if not boundary:
                for (oid, p0, p1, ctx) in trans:         # learn each object's operator (the self emerges from these)
                    self.forwards.setdefault(oid, ForwardModel()).observe(p0, self._last, p1, ctx)
            self.goal.observe(objects, score - self._prev_score)
        # how much there is still to LEARN about each action (learning progress; 1.0 if untried) -- the curiosity drive
        curiosity = {a: max((fm.curiosity(a) for fm in self.forwards.values()), default=1.0) for a in actions}
        context = lambda oid, pose, key: self._context(oid, pose, key, grid, self.field.cells)
        action = self.planner.act(objects, self.forwards, actions, curiosity, context=context)
        if action is None:                                   # nothing to route toward -> heavy-tailed (Lévy) search
            action = self._levy(actions)
        else:
            self._run = (None, 0)                            # a directed plan took over -> end the Lévy run
        self._prev_objects, self._last, self._prev_score = objects, action, score
        self._prev_frame, self._prev_cells = grid, dict(self.field.cells)
        return action

    def run(self, env, max_steps: int = 2000):
        self.reset()
        frame = env.reset()
        while frame.action_counter < max_steps and not frame.is_win():
            action = self.act(frame.grid, frame.available, frame.score)
            if action is None:
                break
            nxt = env.step(action)
            if nxt.level != frame.level:
                self.new_level()
            frame = nxt
        return Outcome(won=frame.is_win(), levels=frame.score, actions=frame.action_counter)
