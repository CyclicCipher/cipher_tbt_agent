"""Recurrent L6 — track WHERE you are with no position observation (the unified recurrence, architecture §14).

The column's static `place_code` is a lookup; `loc_move`/`loc_sense`/`loc_where` drive L6 as a DYNAMIC
path-integrated belief — the same selective gated recurrence as the language SSM, on the location code:

  loc_reset(start)   begin dead reckoning at a known origin.
  loc_move(action)   PREDICT — path-integrate by the L5 displacement operator (efference copy). No observation
                     — this is the update that works when position is NOT visible (partial observability).
  loc_sense(node)    CORRECT — selectively snap toward a sighted landmark (the SSM decay gate).
  loc_where()        read out the believed node.

Validated here on an open grid:
  (1) DEAD RECKONING — from a known start, path-integrate a random walk of UNBLOCKED moves; the belief tracks
      the true cell with NO position observation.
  (2) THE REAFFERENCE RULE — integrating a BLOCKED move (one that didn't happen) desyncs the belief; gating
      loc_move on actual motion (the reafference) keeps it exact. Motivates the partial-obs agent's move check.
  (3) CORRECTION — start the belief LOST (wrong cell), path-integrate, then sense one landmark → the gate snaps
      it back to truth. This is how a drifting belief is re-anchored from a sighting.

Run:  python -m demos.recurrent_location      (run from src/ with PYTHONPATH=src)
"""

from __future__ import annotations

import os
import random
import sys


from tbt.column import CorticalColumn                            # noqa: E402
from tbt.reward import MOVES                                     # noqa: E402

from demos.control_loop import _cell, learn_grid             # noqa: E402


def _step(pos, j, N):
    dx, dy = MOVES[j]
    nx, ny = pos[0] + dx, pos[1] + dy
    return (nx, ny) if 0 <= nx < N and 0 <= ny < N else None     # None = blocked (off the grid)


def dead_reckoning(N=7, steps=200, seed=0, integrate_blocked=False):
    """Path-integrate a random walk; report how often the belief equals the true cell."""
    col = learn_grid(CorticalColumn(n_entities=N * N, seed=seed), N)
    rng = random.Random(seed)
    pos = (N // 2, N // 2)
    col.loc_reset(_cell(*pos, N))
    correct = total = 0
    for _ in range(steps):
        j = rng.randrange(len(MOVES))
        nxt = _step(pos, j, N)
        if nxt is None:                                          # blocked move
            if integrate_blocked:
                col.loc_move(j)                                  # WRONG: integrate a move that didn't happen
                total += 1; correct += (col.loc_where() == _cell(*pos, N))
            continue                                             # the reafference rule: don't integrate
        pos = nxt
        belief = col.loc_move(j)                                 # PREDICT — no position observation used
        correct += (belief == _cell(*pos, N)); total += 1
    return correct, total


def correction(N=7, seed=0):
    """Start LOST, path-integrate a few moves, then sense a landmark → the gate re-anchors the belief."""
    col = learn_grid(CorticalColumn(n_entities=N * N, seed=seed), N)
    true = (1, 1)
    col.loc_reset(_cell(5, 5, N))                                # believe we are at (5,5) — wrong
    lost = col.loc_where() != _cell(*true, N)
    for j in (3, 1):                                             # path-integrate from the (wrong) belief
        nxt = _step(true, j, N)
        if nxt:
            true = nxt; col.loc_move(j)
    drifted = col.loc_where() != _cell(*true, N)                 # still wrong (we started lost)
    col.loc_sense(_cell(*true, N), keep=0.0)                     # sight a landmark at the true cell
    fixed = col.loc_where() == _cell(*true, N)                   # the gate snaps the belief to truth
    return lost, drifted, fixed


if __name__ == "__main__":
    print("recurrent L6 — track position with no position observation (the unified recurrence)\n")
    for N in (7, 12):
        c, t = dead_reckoning(N=N)
        cb, tb = dead_reckoning(N=N, integrate_blocked=True)
        print(f"  N={N:>2}  dead-reckoning (reafference-gated): {c}/{t} = {100*c/t:.0f}%"
              f"   |   if blocked moves ARE integrated: {cb}/{tb} = {100*cb/max(tb,1):.0f}%")
    lost, drifted, fixed = correction()
    print(f"\n  correction: started lost={lost}, still drifted after path-integration={drifted}, "
          f"re-anchored by one sighting={fixed}")
    print("\n  L6 path-integrates position from moves alone (no position observation); the reafference rule")
    print("  (don't integrate a blocked move) keeps it exact; one landmark sighting re-anchors a lost belief.")
