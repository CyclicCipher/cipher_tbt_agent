"""test_rule_proposal.py — the REFUTATION fixture, and what it disproves about rule proposal.

`tasks.refutation.capture` stops a game at the moment the agent's reward model is shown to be wrong, on either of two
triggers, and returns the live agent with everything it had learned. Domain-free: it reads the agent's own beliefs and its own
epistemic state, never the game.

The fixture exists to DISPROVE proposers, not to score them. It is very easy to tune a proposer until it reports the truth on
one captured moment, and that validates nothing — filters fitted to one example, inside a rule grammar chosen to fit that
example, is a schema rather than a discovery. So the games span different rule SHAPES, and these tests record where the
current proposer stops being able to state the rule at all.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                          # noqa: E402
from tasks.games.collectall import CollectAll        # noqa: E402
from tasks.games.lockpath import LockPath            # noqa: E402
from tasks.games.sokoban import Sokoban              # noqa: E402
from tasks.games.toggle import Toggle                # noqa: E402
from tasks.refutation import capture                 # noqa: E402


def _agent():
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _context(shot):
    a = shot["agent"]
    objs = a.transduce(shot["frame"].grid)
    return a, objs, a._positions(objs), a.self_color()


def test_a_believed_win_condition_can_be_caught_being_refuted():
    """LockPath L2 refutes the win condition learned on L0/L1: "reach the goal" holds again and nothing is paid, so its
    contingency falls. Nothing in the detector reads a goal tile — it watches the agent's own reward model being wrong."""
    shot = capture(_agent, LockPath, budget=400)
    assert shot is not None and shot["trigger"] == "refuted"
    assert shot["after"] < shot["before"], "a refutation IS the believed condition losing weight"
    assert shot["condition"] == 3, f"the refuted belief is 'reach the goal', got {shot['condition']}"


def test_a_game_that_never_pays_still_yields_a_moment():
    """THE REASON THERE ARE TWO TRIGGERS. Refutation presupposes a belief, and a game that has never paid gives the agent
    nothing to lose — so on exactly the games where a rule most needs proposing, a refutation detector is silent. Sokoban
    never pays, and is caught anyway by EXHAUSTION: `_unlearned_cells` is empty (the model is confident about every feature
    it can perceive) while no reward has arrived. The implicit expectation being violated is not "condition X pays" but the
    more basic one that a world you have learned is a world you can succeed in."""
    shot = capture(_agent, Sokoban, budget=400)
    assert shot is not None and shot["trigger"] == "exhausted"
    assert shot["frame"].score == 0, "Sokoban pays nothing, which is the point"
    assert not shot["agent"]._unlearned_cells(_context(shot)[2]), "and there was nothing left to learn"


def test_the_proposer_is_silent_exactly_when_it_is_needed():
    """THE FINDING THE FIXTURE WAS BUILT FOR. At LockPath's refutation the agent has not yet seen the block move, so
    `_movers` is empty — and the proposer iterates movers. It produces NOTHING at the one instant a hypothesis is called for,
    and could only ever speak after exploration had already discovered the mover, which is after the damage is done."""
    shot = capture(_agent, LockPath, budget=400)
    a, objs, pos, sc = _context(shot)
    assert not a._movers, "at refutation the block has not been seen to move"
    assert a._propose_rule(objs, pos, sc) == [], "so the proposer, keyed on movers, has nothing to say"


def test_the_pair_grammar_cannot_state_a_tour():
    """THE SCHEMA, showing through — and the reason a proposer that scores perfectly on LockPath has not generalised.
    CollectAll's win is `not items`: EVERY item visited, each consumed on contact. That is a property of a SET, with no
    target for anything to be placed on, whereas every proposal this proposer can emit is an `(a, b)` co-location. No
    ranking rescues a grammar that cannot express the rule."""
    shot = capture(_agent, CollectAll, budget=400)
    a, objs, pos, sc = _context(shot)
    game = shot["game"]
    assert hasattr(game, "items") and isinstance(game.items, (set, frozenset)), (
        "the win is a property of a SET of cells — `not items` — not an arrangement of two objects")
    assert all(isinstance(c, tuple) and len(c) == 2 for c in a._propose_rule(objs, pos, sc)), (
        "yet every proposal is ONE relation between TWO features, which cannot say 'all of them'")


def test_toggle_needs_no_hypothesis_because_the_agent_simply_wins_it():
    """Recorded because it corrects an assumption rather than confirming one. Toggle was built as an adversarial test for
    hardcoded subgoal vocabularies — a switch that CLOSES the door on the short path, which a fire/cover/goal/collect
    vocabulary cannot express. This agent has no such vocabulary, and wins it outright, so there is no moment to capture."""
    assert capture(_agent, Toggle, budget=400) is None
