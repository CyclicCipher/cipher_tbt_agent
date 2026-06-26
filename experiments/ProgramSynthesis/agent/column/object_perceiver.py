"""Object-level perception (E) — track objects across frames and LEARN the body + object roles from play.

Sits on objects.segment / object_motion. No semantic priors:
  - the BODY is the efference copy — the object that translates by the issued action's delta (and does so on
    almost every move, so it dominates the evidence even though a pushed piece briefly shares its delta);
  - a PUSHABLE object is a non-body object that translates by the action's delta — i.e. the body shoved it.
Both fall straight out of motion, so the replica's `block = colour 6` hand-coding goes away. The one object role
this CAN'T settle is the 'pad'/target, because that one is defined by the SCORE, not by motion — that's F.
Generic over grid size + palette, so the same perceiver serves the replica and a real 64×64 game.
"""

from __future__ import annotations

from collections import defaultdict

try:                                                   # runnable as a module (-m) or directly
    from .objects import object_motion, segment
except ImportError:
    from objects import object_motion, segment


class ObjectPerceiver:
    def __init__(self):
        self.body_evidence = defaultdict(int)
        self.push_evidence = defaultdict(int)
        self.entered = set()                           # colours the body has walked ONTO (walkable)
        self.failed = set()                            # colours the body could NOT enter
        self.body_color = None

    def observe(self, prev_grid, delta, cur_grid):
        """One transition: segment both frames, read the motion, accumulate body + pushable evidence.
        Returns (prev_objs, cur_objs, moved) so a caller can also reason over the perceived scene."""
        prev_objs, cur_objs = segment(prev_grid), segment(cur_grid)
        moved = object_motion(prev_objs, cur_objs)

        for obj, d in moved:                           # efference copy: moved by the issued delta
            if d == delta:
                self.body_evidence[obj.color] += 1
        if self.body_evidence:
            self.body_color = max(self.body_evidence, key=self.body_evidence.get)

        for obj, d in moved:                           # pushable: a NON-body object shoved along the move
            if obj.color != self.body_color and d == delta:
                self.push_evidence[obj.color] += 1

        # walkable vs obstacle: did the body enter the cell it stepped toward, or fail to?
        if self.body_color is not None:
            bobj = next((o for o in prev_objs if o.color == self.body_color), None)
            if bobj is not None:
                bx, by = next(iter(bobj.cells))
                tx, ty = bx + delta[0], by + delta[1]
                if 0 <= ty < len(prev_grid) and 0 <= tx < len(prev_grid[0]):
                    tcolor = prev_grid[ty][tx]
                    if any(o.color == self.body_color and d == delta for o, d in moved):
                        self.entered.add(tcolor)       # walked onto it
                    else:
                        self.failed.add(tcolor)        # could not enter it
        return prev_objs, cur_objs, moved

    @property
    def pushable(self):
        return {c for c in self.push_evidence if c != self.body_color}

    @property
    def walkable(self):
        return set(self.entered)                       # colours the body can move onto

    @property
    def blocking(self):
        return self.failed - self.entered - self.pushable


if __name__ == "__main__":
    import os, random, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from arc_agi_3 import Environment, GameState                              # noqa: E402
    from arc_agi_3.games import LockPath                                      # noqa: E402
    from arc_agi_3.oracle import _capture, _restore, solve_level             # noqa: E402

    print("object-level perception (E): learn the body + pushable piece from play, no colour priors\n")
    rng = random.Random(0)
    p = ObjectPerceiver()
    pushes = 0
    for ep in range(40):                               # oracle-hinted so blocks actually get pushed
        env = Environment(LockPath()); frame = env.reset()
        for _ in range(200):
            if frame.state != GameState.NOT_FINISHED:
                break
            moves = [a for a in frame.available_actions if a.is_movement]
            sol = None
            if rng.random() < 0.8:
                saved = _capture(env.game); sol = solve_level(env.game); _restore(env.game, saved)
            action = sol[0] if (sol and sol[0] in moves) else rng.choice(moves)
            prev = frame; frame = env.step(action)
            if frame.state == GameState.NOT_FINISHED and frame.level == prev.level and action.is_movement:
                _, _, moved = p.observe(prev.grid, action.delta, frame.grid)
                if any(c != p.body_color and d == action.delta for c, d in
                       ((o.color, d) for o, d in moved)):
                    pushes += 1

    print(f"  learned body colour:      {p.body_color}   (the agent: moves by the action's delta every step)")
    print(f"  learned pushable colour:  {sorted(p.pushable)}   (the block: only moves when shoved; {pushes} pushes seen)")
    print(f"  body evidence:            {dict(p.body_evidence)}")
    print(f"  push evidence:            {dict(p.push_evidence)}")
    print("\n  both discovered from object motion alone — `block = colour 6` is no longer hand-coded.")
    print("  the remaining role, the pad/target, is defined by the score, so it belongs to F (value).")
