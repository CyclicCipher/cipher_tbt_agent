"""agent.py — the live entry point of the TBT agent, and the ROOT of the reachability graph (RULES.md #2).

STARTING OVER (2026-07-09): this is the skeleton. Everything the agent uses MUST be reachable from here by import, or the
reachability test goes red (RULES.md #2 — the rule that would have caught the basal-ganglia orphaning). Build the loop
VERTICAL from here (RULES.md #4): the next step is the thinnest end-to-end slice that wins one trivial replica level.
"""

from __future__ import annotations


class Agent:
    """The one agent — it will drive a live game loop (perceive → predict → plan → act → win) over the column, built up
    one vertical slice at a time. Skeleton: no loop yet; the first slice wires perception + action to win a trivial level.
    As each mechanism is added it is IMPORTED from here (directly or transitively), which is what keeps it 'wired' and off
    the orphan list — a mechanism is not 'done' until the agent plays more than it did before (RULES.md #3)."""

    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = int(n_actions)
        self.seed = int(seed)

    def step(self, observation):
        """One turn: an observation in, an action out. SKELETON — the first vertical slice fills this in. It RAISES rather
        than returning a silent no-op, so a half-built loop fails loudly instead of pretending to work (RULES.md #3)."""
        raise NotImplementedError("Agent.step: skeleton — build the first vertical slice (STATUS.md 'Next').")
