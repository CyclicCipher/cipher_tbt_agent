"""The discovered schema: the agent's growing theory of the game.

This is the regime (paper) / world-model (MuZero) / generative model (active
inference). It is grown by induction from transitions and the score signal, and
carried forward across levels (transport).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

from tasks.core import GameAction

from .perceptor import Grid, Pos, color_at, detect_move, find_cells, modal_background


@dataclass
class WorldModel:
    background: Optional[int] = None
    agent_color: Optional[int] = None
    move_model: Dict[GameAction, Pos] = field(default_factory=dict)   # action -> (dx, dy)
    tries: Dict[GameAction, int] = field(default_factory=dict)
    fail_positions: Dict[GameAction, Set[Pos]] = field(default_factory=dict)
    blocker_colors: Set[int] = field(default_factory=set)
    goal_colors: Set[int] = field(default_factory=set)
    # Causal rules: contacting `trigger` color opens (turns to background) these colors.
    contact_effect: Dict[int, Set[int]] = field(default_factory=dict)
    contacted: Set[int] = field(default_factory=set)         # colors the agent has moved onto
    pushable_colors: Set[int] = field(default_factory=set)   # colors that translate on contact
    # Block cells discovered (by hitting a deadlock + reset) to strand the level — the
    # learned "mistake": positions a block must not be wandered into again. Per-level.
    dead_block_cells: Set[Pos] = field(default_factory=set)
    # Context-dependent goal (a QKG triplet F_τ(C), learned from the sparse win signal):
    win_contexts: List[FrozenSet[int]] = field(default_factory=list)         # colors present at wins
    reach_no_win_contexts: List[FrozenSet[int]] = field(default_factory=list)  # goal reached, no win

    def agent_pos(self, grid: Grid) -> Optional[Pos]:
        if self.agent_color is None:
            return None
        cells = find_cells(grid, self.agent_color)
        return cells[0] if cells else None

    def present_colors(self, grid: Grid) -> FrozenSet[int]:
        cols = {c for row in grid for c in row}
        cols.discard(self.background)
        if self.agent_color is not None:
            cols.discard(self.agent_color)
        return frozenset(cols)

    def required_absent(self) -> Set[int]:
        """The discovered context condition on the goal: colors whose *presence* blocks
        a win — present in some goal-reach that did NOT win, yet absent in every actual
        win. This is the goal triplet's F_τ(C), induced from the sparse win signal."""
        if not self.win_contexts or not self.reach_no_win_contexts:
            return set()
        blocking = set().union(*self.reach_no_win_contexts)
        for won in self.win_contexts:
            blocking -= set(won)
        return blocking

    def goal_sufficient(self, grid: Grid) -> bool:
        """Would reaching the goal-color actually win right now (its context met)?"""
        return not (self.required_absent() & self.present_colors(grid))

    def resolved(self, action: GameAction) -> bool:
        """Resolved once we know its effect, or it's a no-op *everywhere* — failed to
        move from several distinct positions, not merely blocked against one wall."""
        return action in self.move_model or len(self.fail_positions.get(action, ())) >= 4

    def update(self, old: Grid, action: GameAction, new: Grid, score_delta: int) -> None:
        if self.background is None:
            self.background = modal_background(old)
        self.tries[action] = self.tries.get(action, 0) + 1

        before = self.agent_pos(old)
        mv = detect_move(old, new, self.background, self.agent_color)
        if mv is not None:
            color, p, q = mv
            if self.agent_color is None:
                self.agent_color, before = color, p
            if color == self.agent_color:
                self.move_model[action] = (q[0] - p[0], q[1] - p[1])
                self._learn_contact(old, new, p, q)
                # Reached the goal-color but the level did NOT complete -> the goal is
                # context-dependent (a negative example for its F_τ(C)).
                if old[q[1]][q[0]] in self.goal_colors and score_delta == 0:
                    self.reach_no_win_contexts.append(self.present_colors(old))
        else:
            # The agent did not move: record *where* it failed (to tell a blocked
            # direction from a genuine no-op), and if we know the delta, the cell we
            # tried to enter is a blocker.
            if before is not None:
                self.fail_positions.setdefault(action, set()).add(before)
            if action in self.move_model and before is not None:
                dx, dy = self.move_model[action]
                c = color_at(old, before[0] + dx, before[1] + dy)
                if c is not None and c != self.background and c != self.agent_color:
                    self.blocker_colors.add(c)

        if score_delta > 0:
            self.infer_goal(old, action)

    def _learn_contact(self, old: Grid, new: Grid, p: Pos, q: Pos) -> None:
        """Surprise-driven causal induction.

        Core Knowledge used: *persistence* — predict that nothing changes but the
        agent's own move, so any other change is a surprise; *contact* — attribute
        that surprise to the color the agent just moved onto. A surprise of the
        form "color X's cells turned to background" becomes a rule: contacting the
        contacted color opens color X.
        """
        bg = self.background
        contact = old[q[1]][q[0]]                    # what the agent moved onto
        if contact == bg or contact == self.agent_color:
            return
        self.blocker_colors.discard(contact)         # the agent passed onto it
        self.contacted.add(contact)
        # Push (the other causal form): contact made the object translate by the agent's delta.
        dx, dy = q[0] - p[0], q[1] - p[1]
        bx, by = q[0] + dx, q[1] + dy
        if 0 <= by < len(old) and 0 <= bx < len(old[0]):
            if old[by][bx] == bg and new[by][bx] == contact:
                self.pushable_colors.add(contact)
        opens = set()
        for y in range(len(old)):
            for x in range(len(old[0])):
                o, n = old[y][x], new[y][x]
                if o == n or (x, y) == p or (x, y) == q:
                    continue                          # the agent's own move, not a surprise
                if n == bg and o not in (bg, self.agent_color, contact):
                    opens.add(o)                      # color o was opened by the contact
        if opens:
            self.contact_effect.setdefault(contact, set()).update(opens)

    def learn_death(self, old: Grid, action: GameAction) -> None:
        """The agent died: mark whatever it moved onto as a deadly blocker to avoid."""
        before = self.agent_pos(old)
        if before is None or action not in self.move_model:
            return
        dx, dy = self.move_model[action]
        c = color_at(old, before[0] + dx, before[1] + dy)
        if c is not None and c != self.background and c != self.agent_color:
            self.blocker_colors.add(c)
            self.contacted.add(c)
            self.contact_effect.pop(c, None)

    def infer_goal(self, old: Grid, action: GameAction) -> None:
        """The score went up: whatever the agent moved onto is goal-related."""
        if self.background is None:
            self.background = modal_background(old)
        before = self.agent_pos(old)
        if before is None or action not in self.move_model:
            return
        dx, dy = self.move_model[action]
        c = color_at(old, before[0] + dx, before[1] + dy)
        if c is not None and c != self.background and c != self.agent_color:
            self.goal_colors.add(c)
            self.win_contexts.append(self.present_colors(old))   # positive example for F_τ(C)
