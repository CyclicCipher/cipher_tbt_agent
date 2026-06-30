"""FM1 -- the generative forward model as a COLUMN capability (FORWARD_MODEL_PLAN). The column reads L4's
feature-at-location field and L5's per-location operator predicts the next field from the local feature-context +
action -- the TEM objective ('predict the next observation at each location') at cell grain, NOT a raw-pixel harness.

The test: a HIDDEN deterministic cellular automaton (a frontier that spreads under the action). The column learns it
from a few transitions and predicts held-out steps EXACTLY -- and the prediction is ACTION-CONDITIONED (spread right
vs left). Plus: an unseen context defaults to no-change; the field is L4 feature ids (the seating contract)."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.agent import Agent  # noqa: E402
from tbt.column import CorticalColumn  # noqa: E402

H, W = 5, 14


def grow(frame, action):
    """The HIDDEN rule the column must learn: a background cell turns 1 if its neighbour in the action's direction is
    1 (action 0 spreads a 1-region RIGHTWARD, action 1 LEFTWARD). Deterministic + local."""
    out = [row[:] for row in frame]
    for y in range(H):
        for x in range(W):
            if frame[y][x] == 0:
                if action == 0 and x > 0 and frame[y][x - 1] == 1:
                    out[y][x] = 1
                if action == 1 and x < W - 1 and frame[y][x + 1] == 1:
                    out[y][x] = 1
    return out


def _bar(x):
    g = [[0] * W for _ in range(H)]
    for y in (1, 2, 3):
        g[y][x] = 1
    return g


def _rollout(start, action, steps):
    frames = [start]
    for _ in range(steps):
        frames.append(grow(frames[-1], action))
    return frames


def test_field_is_l4_feature_ids():
    """The seating contract: the field the model predicts in is L4's feature-at-location (encoded ids), not raw
    pixels -- so the SAME operator handles richer L4 features later without change."""
    col = CorticalColumn(n_entities=64)
    field = col.feature_field(_bar(3))
    flat = {v for row in field for v in row}
    assert flat == {col.L4.encode((0,)), col.L4.encode((1,))}     # exactly the two colours, as L4 ids


def test_column_learns_a_hidden_ca_and_predicts_exactly():
    """Train L5's per-location operator on a few transitions; predict held-out steps EXACTLY (deterministic local rule
    -> single-entry rule; coverage from the repeating frontier pattern)."""
    col = CorticalColumn(n_entities=64)
    frames = _rollout(_bar(3), action=0, steps=7)
    fields = [col.feature_field(f) for f in frames]
    for i in range(4):                                            # learn transitions 0..3
        col.observe_field(fields[i], 0, fields[i + 1])
    for i in (4, 5, 6):                                           # predict held-out steps 4..6
        assert col.predict_field(fields[i], 0) == fields[i + 1], f"step {i} mispredicted"


def test_prediction_is_action_conditioned():
    """The same context predicts DIFFERENTLY per action -- the model found the action's control of the dynamics
    (the cn04 lesson: action-conditioning is what the tabular loop could not see)."""
    col = CorticalColumn(n_entities=64)
    right = _rollout(_bar(7), action=0, steps=5)                  # spreads right
    left = _rollout(_bar(7), action=1, steps=5)                   # spreads left
    rf = [col.feature_field(f) for f in right]
    lf = [col.feature_field(f) for f in left]
    for i in range(3):
        col.observe_field(rf[i], 0, rf[i + 1])
        col.observe_field(lf[i], 1, lf[i + 1])
    field = col.feature_field(_bar(7))
    pr = col.predict_field(field, 0)
    pl = col.predict_field(field, 1)
    assert pr == rf[1]                                            # action 0 -> grew right
    assert pl == lf[1]                                            # action 1 -> grew left
    assert pr != pl                                               # the action genuinely steers the prediction


def test_unseen_context_defaults_to_no_change():
    """A location whose (neighbourhood, action) was never observed keeps its current feature -- the safe default that
    makes the rule a GENERALISING base over the discrete edges, not a guess."""
    col = CorticalColumn(n_entities=64)
    field = col.feature_field(_bar(3))
    assert col.predict_field(field, 9) == field                  # action 9 never trained -> identity


# ---- FM2: the dense predict-then-compare loop in the agent --------------------------------------------
def _marker_frame(pos, w=8):
    """A 3xw frame with a single marker (colour 1) at column `pos` -- a deterministic moving-dynamics scene."""
    g = [[0] * w for _ in range(3)]
    g[1][pos] = 1
    return g


def test_fm2_dense_loop_learns_dynamics_online():
    """FM2: given frames, the agent does dense field predict-then-compare; the per-location forward model learns the
    (moving-marker) dynamics ONLINE, so its error on changed cells DROPS = learning progress."""
    agent = Agent(n_actions=1, seed=0)
    pos = 0
    agent.step(("s",), 0.0, frame=_marker_frame(pos))            # prime (no prediction yet)
    errs = []
    for _ in range(24):                                          # 3 wrap-around cycles (w=8)
        pos = (pos + 1) % 8
        agent.step(("s",), 0.0, frame=_marker_frame(pos))
        errs.append(agent.field_error)
    assert agent.col.L5.field_rule                               # the per-location rule was learned online
    early, late = sum(errs[:8]) / 8, sum(errs[-8:]) / 8         # full cycles (w=8): robust to where the wrap lands
    assert late < early, f"no learning progress: early {early:.2f} late {late:.2f}"
    assert late < 0.15, f"learnable motion not mastered: late cycle error {late:.2f}"


def test_fm2_epistemic_value_winds_down_when_mastered():
    """The dense error feeds learning-progress (reward's slow−fast EWMA). Once the LEARNABLE part is mastered (only
    the non-local wrap stays a small irreducible error), learning progress winds DOWN — NOT stuck high like the
    raw-error noisy-TV trap. The one unpredictable wrap per cycle is correctly treated as bounded noise."""
    agent = Agent(n_actions=1, seed=0)
    pos = 0
    agent.step(("s",), 0.0, frame=_marker_frame(pos))
    errs = []
    for _ in range(48):
        pos = (pos + 1) % 8
        agent.step(("s",), 0.0, frame=_marker_frame(pos))
        errs.append(agent.field_error)
    assert sum(errs[-8:]) / 8 < 0.15                            # the learnable motion is mastered (last cycle avg low)
    assert agent.reward.epistemic_value(("s",)) < 0.1          # learning progress wound down (noise-robust)


def test_fm2_backward_compatible_without_frame():
    """No frame → the field path is skipped; the agent runs exactly as before (binary state surprise)."""
    agent = Agent(n_actions=4, seed=0)
    agent.step(("a",), 0.0)
    agent.step(("b",), 0.0)
    assert agent._prev_field is None and agent.field_error == 0.0


# ---- FM3: the forward model DRIVES action selection (epistemic rollout) -------------------------------
def test_fm3_drives_toward_least_understood_action():
    """The forward model contributes a per-action epistemic value (learning potential). With one action's effect
    learned and another unseen, the agent is DRIVEN to the unseen one -- action-space curiosity, forward-model-led."""
    agent = Agent(n_actions=2, seed=0)
    f0 = agent.col.feature_field(_marker_frame(3))
    f1 = agent.col.feature_field(_marker_frame(4))
    for _ in range(4):
        agent.col.observe_field(f0, 0, f1)                       # action 0's effect on f0 learned (unambiguous)
    epi = {a: e for a, (_p, e) in agent._field_plan(f0).items()}   # action 1 never seen -> higher learning potential
    assert epi[1] > epi[0]
    assert agent._choose(("s",), field=f0) == 1                 # the tabular value is flat -> the forward model decides


def test_fm3_epistemic_winds_down_when_actions_learned():
    """When BOTH actions' effects are pinned, the epistemic drive collapses -> it hands off to the pragmatic value
    (the bounded bonus no longer overrides a learned reward). The noise-robust wind-down, in the planner."""
    agent = Agent(n_actions=2, seed=0)
    f0 = agent.col.feature_field(_marker_frame(3))
    fr = agent.col.feature_field(_marker_frame(4))               # action 0 -> right
    fl = agent.col.feature_field(_marker_frame(2))               # action 1 -> left
    for _ in range(4):
        agent.col.observe_field(f0, 0, fr)
        agent.col.observe_field(f0, 1, fl)                       # both actions learned on f0
    epi = {a: e for a, (_p, e) in agent._field_plan(f0).items()}
    assert epi[0] < 0.5 and epi[1] < 0.5                         # both understood -> low epistemic -> pragmatic takes over


def test_fm3_rollout_depth_runs():
    """A shallow rollout (depth>1) composes via field_step without error -- the rollout machinery for FM4."""
    agent = Agent(n_actions=2, seed=0)
    f0 = agent.col.feature_field(_marker_frame(3))
    agent.col.observe_field(f0, 0, agent.col.feature_field(_marker_frame(4)))
    plan = agent._field_plan(f0, depth=2)
    assert set(plan) == {0, 1} and all(isinstance(v, tuple) and len(v) == 2 for v in plan.values())


# ---- FM4: the goal in feature space -- the agent PLANS toward a scoring configuration -----------------
def _hbar(size, w=12):
    g = [[0] * w for _ in range(3)]
    for x in range(size):
        g[1][x] = 1
    return g


def test_fm4_value_directs_planning_toward_target():
    """The FM4 contract: with the dynamics learned (action 0 grows, action 1 shrinks a bar) AND a field value that
    knows the TARGET configuration is rewarding, the forward-model rollout DIRECTS the agent toward the target from
    EITHER side -- grow when below, shrink when above. Planning in feature space toward the goal."""
    agent = Agent(n_actions=2, seed=0)
    agent.field_bin = 1                                          # this tiny env needs size resolution (real games generalise)
    TARGET = 7
    for size in range(2, 11):                                    # learn the dynamics for both actions
        f = agent.col.feature_field(_hbar(size))
        for _ in range(3):
            agent.col.observe_field(f, 0, agent.col.feature_field(_hbar(size + 1)))
            agent.col.observe_field(f, 1, agent.col.feature_field(_hbar(size - 1)))
    for _ in range(40):                                          # learn the value: the TARGET configuration is rewarding
        agent.field_value.update(agent.field_features(agent.col.feature_field(_hbar(TARGET))), 1.0)
    assert agent._choose(("s",), field=agent.col.feature_field(_hbar(TARGET - 1))) == 0   # below the target -> grow
    assert agent._choose(("s",), field=agent.col.feature_field(_hbar(TARGET + 1))) == 1   # above the target -> shrink


def test_fm4_value_learns_from_score_in_loop():
    """The TD wiring: a sparse score entering a configuration RAISES the learned value of the configuration left --
    so the generalising goal-in-feature-space is acquired online from the score (no hand-coded goal)."""
    agent = Agent(n_actions=2, seed=0)
    agent.field_bin = 1
    feats5 = agent.field_features(agent.col.feature_field(_hbar(5)))
    before = agent.field_value.value(feats5)
    for _ in range(15):
        agent.new_episode()
        agent.step(("s",), 0.0, frame=_hbar(5))                  # in bar(5)
        agent.step(("s",), 1.0, frame=_hbar(6))                  # entering bar(6) scores -> TD credits bar(5)
    assert agent.field_value.value(feats5) > before


# ---- obstacle handling is NATIVE to the forward model (step 1: the `barriers` faculty deprecated) -----
def test_forward_model_predicts_a_blocked_move_as_no_change():
    """A move BLOCKED by a wall is predicted as NO CHANGE; a move into free space changes the field. So the planner
    sees a blocked move makes no progress (and, once learned, offers no epistemic gain) -- obstacle handling falls
    out of the generative prediction, no recognition faculty needed. (The full navigate-AROUND-an-obstacle behaviour
    needs the spatial value -- step 2; this is the prediction primitive it stands on.)"""
    col = CorticalColumn(n_entities=64)
    W, WALL = 10, 7

    def frame(p):                                                # agent (2) at column p; a wall (1) fixed at WALL, row 1
        g = [[0] * W for _ in range(3)]
        g[1][WALL] = 1
        g[1][p] = 2
        return g

    seq, p = [], 2                                               # the agent moves right until the wall blocks it
    for _ in range(8):
        seq.append(p)
        p = p + 1 if p + 1 != WALL else p
    fields = [col.feature_field(frame(pp)) for pp in seq]
    for i in range(len(seq) - 1):
        col.observe_field(fields[i], 0, fields[i + 1])           # learn: free move -> shifts; blocked move -> no change
    free = col.feature_field(frame(3))
    assert col.predict_field(free, 0) != free                    # free space: the move changes the field
    blocked = col.feature_field(frame(WALL - 1))
    assert col.predict_field(blocked, 0) == blocked              # against the wall: predicted NO CHANGE (blocked)


def test_fm4_end_to_end_scores_more_than_random():
    """END TO END: a dynamics env (grow/shrink a bar) scoring at a target the random walk struggles to reach (5
    directed grows from the start). The agent learns the dynamics + the field value from the sparse score and PLANS
    there, scoring far more than an undirected random policy."""
    import random as _rng
    W, TARGET = 12, 7

    def run(agent, steps=400):
        size = 2
        rng = _rng.Random(1)

        def frame():
            g = [[0] * W for _ in range(3)]
            for x in range(size):
                g[1][x] = 1
            return g

        comps = 0
        for _ in range(steps):
            a = agent.step(("s",), 0.0, frame=frame()) if agent is not None else rng.randrange(2)
            size = min(size + 1, W) if a == 0 else max(size - 1, 1) if a == 1 else size
            if size == TARGET:
                comps += 1
                if agent is not None:
                    agent.step(("s",), 1.0, frame=frame())
                    agent.new_episode()
                size = 2
        return comps

    ag = Agent(n_actions=2, seed=0)
    ag.field_bin = 1
    fm4 = run(ag)
    rnd = sum(run(None) for _ in range(3)) / 3
    assert fm4 > 1.5 * rnd and fm4 > 20, f"FM4 planned {fm4} vs random {rnd:.1f}"
