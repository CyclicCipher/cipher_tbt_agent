"""ColumnAgent — a sensorimotor cortical column on the ARC-AGI-3 replica, WITHOUT the symbolic priors.

Agency is the EFFERENCE COPY, not a perceptual guess: the agent knows the action it issued, so its
position is the column's internal L6 location, PATH-INTEGRATED by the L5 displacement operator that the
action selects, and GROUNDED to the body landmark (the token whose motion the efference copy predicts —
von Holst & Mittelstaedt's reafference). Whatever motion the self-prediction does NOT explain is
exafference = an environmental effect. The flat reward model (reward.py) plans over the column's learned
transitions toward the goal COLOUR it infers from the sparse score.

What is LEARNED (no domain priors): the sensorimotor map (action -> displacement, i.e. the L5 operator),
which colours block / kill / are the goal, all from experience. What is GIVEN (architecture, not a domain
prior): the efference copy — that the agent has privileged access to its own motor command. The known
FLAT ceiling (scaling_probe.py) still applies to the conjunctive levels; the multi-scale grid / hierarchy
is the next layer.
"""

from __future__ import annotations

import os
import random
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "RecurrentWorldModel")))
from tbt.reward import RewardModel                                # noqa: E402  flat prioritized-sweeping planner
from tbt.l6_grid import L6_GridLocation                           # noqa: E402  the real L6 grid (location signal)
from tbt.l5_displacement import L5_Displacement                   # noqa: E402  the real L5 operators

from arc_agi_3 import GameAction, GameState                       # noqa: E402
from arc_agi_3.agents import Agent                                # noqa: E402

from .perceive import active_cells, bounding_box, detect_motion, modal_background  # noqa: E402

_MOVES = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
_EMPTY = frozenset()
_NUM_COLORS = 16


class ColumnAgent(Agent):
    def __init__(self, seed: int = 0, torus: int = 24):
        self.seed, self.torus = seed, torus
        self.reset()

    def reset(self) -> None:
        self.rng = random.Random(self.seed)
        self.L6 = L6_GridLocation(torus_size=self.torus, scales=(11, 13, 17), place_k=1)
        self.L5 = L5_Displacement()                              # action -> learned path-integration operator
        self.N = self.L6.N
        # efference-copy agency / sensorimotor calibration
        self.body_color = None
        self.body_evidence: dict[int, int] = {}
        self.disp: dict[GameAction, tuple[int, int]] = {}        # learned action -> body displacement
        self.T_int = None                                       # action -> per-location next loc (from L5)
        # learned dynamics (no priors)
        self.blocker_colors: set[int] = set()
        self.deadly_colors: set[int] = set()
        self.goal_colors: set[int] = set()
        # tracking
        self.prev_grid = self.prev_cells = self.prev_body = self.prev_target = None
        self.prev_action = None
        self.prev_score = self.prev_level = 0
        self._new_level()

    def _new_level(self) -> None:
        self.rm = RewardModel(16, prioritized=True)             # fresh value for the new layout

    # -- grid index helpers (loc = x*N + y, matching L6's pos = (idx//N, idx%N)) -----------------------
    def _loc(self, x, y):
        return x * self.N + y

    def _xy(self, loc):
        return (loc // self.N, loc % self.N)

    def _color_at(self, grid, pos):
        x, y = pos
        return grid[y][x]

    def _body_pos(self, cells):
        if self.body_color is None:
            return None
        for p, c in cells.items():
            if c == self.body_color:
                return p
        return None

    # -- build the L5 path-integration operators once the sensorimotor map is calibrated ----------------
    def _build_operators(self) -> None:
        L = self.L6.L
        Pall = self.L6.Pall                                     # (L, L) place code per location
        self.T_int = {}
        for a in _MOVES:
            dx, dy = self.disp[a]
            edges = []
            for x in range(self.N):
                for y in range(self.N):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.N and 0 <= ny < self.N:
                        edges.append((self._loc(x, y), self._loc(nx, ny)))
            self.L5.learn(a, Pall, edges)                      # M_a = Σ place(t)⊗place(s)
            nxt = (self.L5.ops[a] @ Pall.t()).argmax(0)        # path-integrate every location at once
            self.T_int[a] = nxt                                # (L,) tensor: loc -> next loc

    def _calibrated(self) -> bool:
        return self.body_color is not None and all(a in self.disp for a in _MOVES)

    # -- learn from the previous step (efference copy -> reafference / exafference) ---------------------
    def _learn(self, cells) -> None:
        if self.prev_cells is None or self.prev_action not in _MOVES:
            return
        moved = detect_motion(self.prev_cells, cells)          # {colour: (dx, dy)}
        for c in moved:
            self.body_evidence[c] = self.body_evidence.get(c, 0) + 1
        if self.body_evidence:
            self.body_color = max(self.body_evidence, key=self.body_evidence.get)
        if self.body_color is not None:
            if self.body_color in moved:
                self.disp[self.prev_action] = moved[self.body_color]   # reafference: action -> displacement
            elif self.prev_target is not None and self.prev_grid is not None:
                tc = self._color_at(self.prev_grid, self.prev_target)  # commanded move, body stayed -> blocker
                if tc != self.body_color:
                    self.blocker_colors.add(tc)
        # exafference (motion my self-prediction does not explain) = environmental effect — for later
        # dynamics learning (key->door, push); v1 learns blockers/goal/death from outcomes directly.

    # -- transitions + reward over the active region ---------------------------------------------------
    def _build_model(self, grid, cells, body):
        x0, y0, x1, y1 = bounding_box({**cells, body: 0})
        blocked = self.blocker_colors | self.deadly_colors

        def passable(x, y):
            return x0 <= x <= x1 and y0 <= y <= y1 and grid[y][x] not in blocked

        T, preds = {}, {}
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                s = ((x, y), _EMPTY)
                row = []
                for a in _MOVES:
                    nx, ny = self._xy(int(self.T_int[a][self._loc(x, y)]))   # L5/L6 path integration
                    nxt = ((nx, ny), _EMPTY) if passable(nx, ny) else s
                    row.append(nxt)
                    preds.setdefault(nxt, []).append(s)
                T[s] = row
        goal_states = [((x, y), _EMPTY) for (x, y), c in cells.items() if c in self.goal_colors]
        return T, preds, goal_states

    # -- the agent contract ----------------------------------------------------------------------------
    def choose_action(self, frame):
        grid = frame.grid

        if frame.state == GameState.GAME_OVER:                 # death = the entered colour kills
            if self.prev_target is not None and self.prev_grid is not None:
                self.deadly_colors.add(self._color_at(self.prev_grid, self.prev_target))
            self._new_level()
            self.prev_grid = self.prev_cells = self.prev_body = self.prev_target = None
            self.prev_action, self.prev_score, self.prev_level = GameAction.RESET, frame.score, frame.level
            return GameAction.RESET, None

        bg = modal_background(grid)
        cells = active_cells(grid, bg)
        self._learn(cells)

        if frame.score > self.prev_score:                      # completed a level → infer the goal colour
            if self.prev_target is not None and self.prev_grid is not None:
                self.goal_colors.add(self._color_at(self.prev_grid, self.prev_target))
            self._new_level()
        elif frame.level != self.prev_level:
            self._new_level()

        body = self._body_pos(cells)
        # still calibrating who I am / what my actions do → move to generate reafference (efference copy)
        if body is None or not self._calibrated():
            return self._commit(grid, cells, body, self.rng.choice(_MOVES), frame)
        if self.T_int is None:
            self._build_operators()

        T, preds, goal_states = self._build_model(grid, cells, body)
        current = (body, _EMPTY)
        if current not in T:
            return self._commit(grid, cells, body, self.rng.choice(_MOVES), frame)

        self.rm.visits[current] += 1
        self.rm.R_ext = {s: 1.0 for s in goal_states}
        for s in goal_states:
            self.rm._push(s, 1.0)
        action = _MOVES[self.rm.act(current, T, preds, self.rng)]
        return self._commit(grid, cells, body, action, frame)

    def _commit(self, grid, cells, body, action, frame):
        target = None
        if body is not None and action in self.disp:           # efference-copy prediction of my next cell
            dx, dy = self.disp[action]
            target = (body[0] + dx, body[1] + dy)
        self.prev_grid, self.prev_cells, self.prev_body, self.prev_target = grid, cells, body, target
        self.prev_action, self.prev_score, self.prev_level = action, frame.score, frame.level
        return action, None
