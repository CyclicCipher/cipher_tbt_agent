"""Phase-2 prior-minimal world model — REUSES the Phase-1 machinery and swaps only the *seeded*
priors for discovered ones, one at a time.

`DiscoveredWorldModel` subclasses the proven `agent/wm/` `WorldModel`. The Phase-1 agent loop,
planner, exploration, contact/push/open *instance* discovery, blockers, goal, context conditions and
deadlock learning are all inherited unchanged — only the induction's seeded priors are replaced.

Increment A (this): **agency.** Phase-1 `detect_move` hardcodes "the agent is the single cell that
translates" (requires exactly one gained + one lost cell). Here agency is *discovered*
(`discover_dynamics`: the action-controlled, localized colour — no single-cell assumption); the rest
of the induction is the reused Phase-1 logic applied to that discovered agent. Later increments
generalize the contact rule-types via `volume/relation`.
"""

from __future__ import annotations

from collections import defaultdict

from agent.wm.perceptor import color_at, modal_background
from agent.wm.world_model import WorldModel

from .perceive import discover_dynamics

_BUFFER = 300


class DiscoveredWorldModel(WorldModel):
    def __init__(self):
        super().__init__()
        self._buffer = []                       # transitions, only until agency is discovered

    def update(self, old, action, new, score_delta) -> None:
        if self.background is None:
            self.background = modal_background(old)
        self.tries[action] = self.tries.get(action, 0) + 1

        # --- Increment A: DISCOVER agency (replaces detect_move's single-cell prior) ---
        if self.agent_color is None:
            self._buffer.append((old, action, new))
            if len(self._buffer) > _BUFFER:
                self._buffer = self._buffer[-_BUFFER:]
            ac, _ = discover_dynamics(self._buffer)
            if ac is not None:
                self.agent_color = ac

        # --- reuse the Phase-1 induction, applied to the discovered agent ---
        before = self.agent_pos(old)
        after = self.agent_pos(new)
        if self.agent_color is not None and before is not None:
            if after is not None and after != before:                 # the agent moved
                self.move_model[action] = (after[0] - before[0], after[1] - before[1])
                self._learn_contact(old, new, before, after)
                if old[after[1]][after[0]] in self.goal_colors and score_delta == 0:
                    self.reach_no_win_contexts.append(self.present_colors(old))
            else:                                                      # blocked / no-op
                self.fail_positions.setdefault(action, set()).add(before)
                if action in self.move_model:
                    dx, dy = self.move_model[action]
                    c = color_at(old, before[0] + dx, before[1] + dy)
                    if c is not None and c != self.background and c != self.agent_color:
                        self.blocker_colors.add(c)

        if score_delta > 0:
            self.infer_goal(old, action)

    def _learn_contact(self, old, new, p, q) -> None:
        """Increment B: general residual-effect learning — replaces Phase-1's two hand-coded
        detectors (a `push` template tied to the cell `q+Δ`, and an `open` scan for colour→bg).

        The residual = changed cells outside the agent's own move (`p`, `q`). We read the effect off
        its *signature*, not its geometry: the contacted colour reappearing elsewhere ⇒ it was
        *displaced* (pushable); another colour vanishing with no reappearance ⇒ it was *removed* (an
        opening). Same `pushable_colors` / `contact_effect` fields the inherited planner consumes —
        only the detection is now general, not templated to the agent's direction."""
        contact = old[q[1]][q[0]]
        if contact == self.background or contact == self.agent_color:
            return
        self.blocker_colors.discard(contact)
        self.contacted.add(contact)

        lost = defaultdict(list)
        gained = defaultdict(list)
        for y in range(len(old)):
            row_o, row_n = old[y], new[y]
            for x in range(len(row_o)):
                if (x, y) == p or (x, y) == q:
                    continue
                o, n = row_o[x], row_n[x]
                if o != n:
                    lost[o].append((x, y))
                    gained[n].append((x, y))

        if contact in gained:                         # the contacted colour moved → pushable
            self.pushable_colors.add(contact)
        for color in lost:                            # a colour vanished on contact → an opening
            if color in (self.background, self.agent_color, contact):
                continue
            if color not in gained:
                self.contact_effect.setdefault(contact, set()).add(color)
