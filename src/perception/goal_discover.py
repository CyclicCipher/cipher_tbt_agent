"""Win-condition discovery (F) — learn the goal AND its conjunctive context from the sparse score alone.

Real ARC-AGI-3 gives no goal, only a level-completion signal, and the win is generally CONJUNCTIVE (reach the
goal AND clear some sub-condition). We reuse the predecessor wm agent's proven, *learned* goal-discovery
(world_model.py: infer_goal / required_absent), now built on E's object perception and feeding the value model
(reward.py):

  - the GOAL colour is whatever the body reaches when the score rises;
  - the CONTEXT that gates it is found by DIFFERENCING — a colour present when the goal was reached WITHOUT a
    win, yet absent in every actual win, is a condition that must be cleared first. In LockPath that resolves to
    the pad colour (present while a pad is uncovered, gone once a block covers it). No colour is assumed.

This removes the last hand-coding in the agent's perception (`pad = colour 7`): the pad falls out of the score.
"""

from __future__ import annotations

try:                                                   # runnable as a module (-m) or directly
    from .object_perceiver import ObjectPerceiver
    from .objects import segment
except ImportError:
    from object_perceiver import ObjectPerceiver
    from objects import segment


class GoalModel:
    """The learned win-condition. `goal_colors` = reached when the score rises; `required_absent()` = the
    colours that must be cleared for a goal-reach to actually win, induced from win vs reach-but-no-win
    contexts. Feeds reward.py: a state wins iff the goal is reached and no required-absent colour remains."""

    def __init__(self):
        self.goal_colors = set()
        self.win_contexts = []          # frozenset of colours present at each win (positive examples)
        self.reach_no_win = []          # frozenset present when the goal was reached but no win (negatives)

    def observe_win(self, present, goal_color):
        self.goal_colors.add(goal_color)
        self.win_contexts.append(frozenset(present))

    def observe_reach_no_win(self, present):
        self.reach_no_win.append(frozenset(present))

    def required_absent(self):
        """Colours present in some failed goal-reach yet absent in every win — the context that gates the goal
        (the conjunctive win's other half). The QKG triplet's F_tau(C), induced from the sparse win signal."""
        if not self.win_contexts or not self.reach_no_win:
            return set()
        blocking = set().union(*self.reach_no_win)
        for won in self.win_contexts:
            blocking -= won
        return blocking

    def wins_now(self, present):
        """Would reaching the goal win in this state? Only if no required-absent colour is still present."""
        return bool(self.goal_colors) and not (self.required_absent() & set(present))


def learn(episodes=60, max_steps=200, seed=0):
    """Drive a GoalModel from real play (oracle-hinted so levels get won, with random moves for the negatives
    — reaching the goal BEFORE covering a pad, which is what forces the win to be a CONJUNCTION). Body identity
    comes from E (ObjectPerceiver); the goal/context come from the score. (The play loop mirrors
    dynamics_perceive.collect — integration will share one loop across the dynamics + goal learners.)"""
    import random
    rng = random.Random(seed)
    perc, goal = ObjectPerceiver(), GoalModel()
    for ep in range(episodes):
        env = Environment(LockPath()); frame = env.reset()
        for _ in range(max_steps):
            if frame.state != GameState.NOT_FINISHED:
                break
            moves = [a for a in frame.available_actions if a.is_movement]
            sol = None
            if rng.random() < 0.8:
                saved = _capture(env.game); sol = solve_level(env.game); _restore(env.game, saved)
            action = sol[0] if (sol and sol[0] in moves) else rng.choice(moves)
            prev = frame; frame = env.step(action)
            if not action.is_movement:
                continue
            if prev.state == GameState.NOT_FINISHED and frame.level == prev.level:
                perc.observe(prev.grid, action.delta, frame.grid)   # E: keep the body identity current
            body = perc.body_color
            if body is None:
                continue
            objs = segment(prev.grid)
            bcell = next((o for o in objs if o.color == body), None)
            if bcell is None:
                continue
            bx, by = next(iter(bcell.cells))                        # 1-cell body on the replica
            dx, dy = action.delta
            tx, ty = bx + dx, by + dy
            if not (0 <= ty < len(prev.grid) and 0 <= tx < len(prev.grid[0])):
                continue
            stepped_on = prev.grid[ty][tx]
            present = {o.color for o in objs} - {body}             # object colours present (bg already excluded)
            scored = frame.score > prev.score and frame.state != GameState.GAME_OVER
            if scored:
                goal.observe_win(present, stepped_on)
            elif stepped_on in goal.goal_colors:
                goal.observe_reach_no_win(present)
    return goal


if __name__ == "__main__":
    import os, sys
    from tasks import Environment, GameState                              # noqa: E402
    from tasks.games import LockPath                                      # noqa: E402
    from tasks.oracle import _capture, _restore, solve_level             # noqa: E402

    print("win-condition discovery (F): learn the goal + its conjunctive context from the sparse score\n")
    g = learn()
    print(f"  goal colour(s):       {sorted(g.goal_colors)}        (reached when the score rises)")
    print(f"  required-absent:      {sorted(g.required_absent())}        (must be cleared to win = the pad)")
    print(f"  evidence:             {len(g.win_contexts)} wins, {len(g.reach_no_win)} goal-reaches-without-win")
    print("\n  the win is CONJUNCTIVE: reach the goal AND clear the required-absent colour. Both learned from the")
    print("  score alone -- `pad = colour 7` is no longer hand-coded. This feeds reward.py (value + planning).")
