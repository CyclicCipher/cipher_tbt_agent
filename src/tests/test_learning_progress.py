"""test_learning_progress.py — LEARNABLE NOVELTY (epiplexity) as the epistemic reward, read off the agent's OWN forward model.

`reward.LearningProgress` scores the model PREQUENTIALLY (predict, score, then learn) and rewards prediction-error REDUCTION,
which is what makes it epiplexity rather than raw novelty (`reference_learnable_novelty`; Finzi 2026 Def 8/eq 8): noise keeps
the loss high forever and pays NOTHING (the noisy TV), a mastered or empty world keeps it low and pays nothing (the dark room),
and only a sustained fall — real structure being absorbed — pays.

The integration test is the point of the rebuild: the loss being scored is the AGENT'S OWN model (operator ⊕ L5 transform ⊕
occasions, composed by `WorldModel`), not a shadow predictor over frames. A second model would have to re-learn what this one
already knows, and its errors would not be the errors that make plans fail.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                        # noqa: E402
from tbt.reward import LearningProgress            # noqa: E402
from tasks.games.lockpath import LockPath          # noqa: E402
from tasks.harness import Environment              # noqa: E402


def _feed(losses):
    lp = LearningProgress()
    return lp, [lp.observe(x) for x in losses]


def test_a_noisy_world_pays_almost_nothing_next_to_a_learnable_one():
    """THE noisy-TV test: a model that can never predict the stream keeps a high, flat loss, so the area above its settled
    floor ~cancels. Raw prediction error would rank noise MAXIMAL, which is exactly the pathology. (The residual is finite
    sample — the fluctuations cancel only in the limit — so this is stated as the ratio it really is.)"""
    rng = np.random.default_rng(0)
    noisy, _ = _feed([0.8 + 0.2 * rng.random() for _ in range(600)])
    learning, _ = _feed([max(0.0, 1.0 - t / 300.0) for t in range(600)])
    assert learning.total() > 10 * noisy.total(), (
        f"a learnable world must dominate noise, got learning {learning.total():.2f} vs noise {noisy.total():.2f}")


def test_a_simple_world_pays_far_less_than_a_complex_one():
    """The other half of the definition, and the one the earlier estimator got BACKWARDS: epiplexity is low for SIMPLE data
    too (mastered in a few steps ⇒ little area), and high only for structure that takes sustained learning to absorb."""
    simple, _ = _feed([1.0 if t < 20 else 0.0 for t in range(600)])          # instantly mastered
    complex_, _ = _feed([max(0.0, 1.0 - t / 600.0) for t in range(600)])     # absorbed slowly, all the way
    assert complex_.total() > 3 * simple.total(), (
        f"complex-and-learnable must dominate simple, got {complex_.total():.2f} vs {simple.total():.2f}")


def test_a_dark_room_pays_nothing():
    """THE dark-room test: a world the model predicts perfectly from the start (nothing ever happens) has no structure left to
    extract, so the loss is flat-low and the reward is ~0. A surprise-MINIMISER would find this maximally attractive."""
    lp, _ = _feed([0.0] * 600)
    assert lp.total() < 1e-6, f"a dark room must earn no learnable novelty, got {lp.total():.6f}"


def test_only_a_sustained_fall_pays_and_it_decays_at_mastery():
    """A model genuinely absorbing structure: the loss falls from wholly-surprised to mastered. That pays — and the reward
    DECAYS to ~0 once mastered, so a solved mechanic stops paying and the drive moves on rather than looping."""
    losses = [max(0.0, 1.0 - t / 300.0) for t in range(600)]         # falls to 0 by t=300, flat thereafter
    lp, prog = _feed(losses)
    assert lp.total() > 5.0, f"a learning model must earn real learnable novelty, got {lp.total():.3f}"
    peak, final = max(prog), float(np.mean(prog[-100:]))
    assert final < peak * 0.1, f"the reward must decay at mastery (peak {peak:.3f}, final {final:.4f})"


def test_the_agents_own_model_shows_a_prequential_learning_curve():
    """THE REBUILD: the loss being scored comes from the agent's OWN forward model, on a real game. Playing LockPath, the
    model's prequential error FALLS (it learns the body's motion and what the walls do) and the epistemic reward accumulates
    — with no second model of the frames anywhere in the loop."""
    game = LockPath()
    env = Environment(game)
    fd = env.reset()
    agent = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    losses = []
    original = agent._model_loss

    def spy(*args, **kwargs):
        loss = original(*args, **kwargs)
        losses.append(loss)
        return loss

    agent._model_loss = spy
    for _ in range(200):
        action, coords = agent.step(fd)
        fd = env.step(action, coords)
        if fd.is_win():
            break

    assert len(losses) > 40, "the agent must have scored its own model over many transitions"
    early = float(np.mean(losses[:40]))
    late = float(np.mean(losses[-40:]))
    assert early > 0.0, "the model must start out surprised by the world"
    assert late < early, f"the agent's OWN model must learn (loss {early:.3f} → {late:.3f})"
    assert agent.progress.total() > 0.0, "learning must accumulate epistemic reward"


# ── the CONJUNCTIVE win condition (`GoalMemory.goals`) ──────────────────────────────────────────────────────────────────
# LockPath L2's win is "the block is on the pad AND the agent is on the goal". A linear delta rule cannot represent an AND
# (the XOR problem), so the conjunction is offered as a CONFIGURAL cue alongside the elements and they compete.

def test_an_element_that_pays_alone_is_learned_alone():
    """The easy case must not regress: where one condition IS the win condition, it is what `goals()` reports."""
    from tbt.reward import GoalMemory
    gm = GoalMemory()
    for _ in range(5):
        gm.credit({3}, 1.0)
    assert gm.goals() == {3}


def test_a_conjunction_is_learned_and_the_misleading_element_is_demoted():
    """The L2 shape. Reaching the goal ALONE pays nothing and happens often; reaching it with the block on the pad pays. The
    elemental "on the goal" is driven down by the unpaid visits, while the configural cue — present only when the reward
    arrives — keeps its weight, so `goals()` reports BOTH conjuncts and a planner can satisfy them together."""
    from tbt.reward import GoalMemory
    gm = GoalMemory()
    pair = (6, 7)                                             # block 6 resting on pad 7
    for _ in range(12):
        gm.credit({3}, 0.0)                                   # on the goal, pad uncovered ⇒ nothing
        gm.credit({3, pair, frozenset({3, pair})}, 1.0)       # both hold ⇒ the win
    assert gm.goals() == {3, pair}, f"the conjunction must be discovered, got {gm.goals()}"
    assert gm.w[3] < gm.w[frozenset({3, pair})], "the misleading element must rank below the configural cue"
