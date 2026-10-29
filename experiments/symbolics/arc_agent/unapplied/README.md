# Parts 30–32: drafted, never applied (v1 frozen 2026-10-29)

The arc agent (v1) was frozen on 2026-10-29, when the user started `arc_agent_v2`. Parts 30–32 had been pre-registered
in `docs/AGENT.md` and drafted as splice scripts, but were never applied to v1. They are kept here as reference
machinery for v2. Nothing imports them, and the splice scripts name paths in a session scratch folder that no longer
exists.

- **Part 30, a learned frame: contents permuted among fixed cells** (`permute_frame.py`, `patch_part30.py`,
  `edit30.py`).
  - **What it is:** a Permute frame whose operators are learned tokens. Each token's permutation is narrowed by every
    observation it explains: for each cell, the intersection of the cells its colour may have come from, then the
    nearest assignment within that set.
  - **Rivalry between sets of columns** (`edit30.py`, in the PFC): a single Permute column competes against the
    fragments the same change was also explained by. Each side is scored on its joint prediction, and only the side
    ahead is planned.
  - **Status:** only a first smoke test ran, on a scratch copy. W17 scored 0.255, before this redesign, because the
    ring was explained by translation and recolour fragments and no Permute column was ever recruited. The redesign
    was never run.
- **Part 31, the transform is the difference** (`displayed.py`, `conjugate.py`, `patch_part31.py`):
  - goals from displayed copies of a column's shape (up to a quarter-turn and a colour map);
  - action models lent across directions by conjugating with the frame's quarter-turns.
- **Part 32, a reward model across games** (`patch_part32.py`): a goal column that predicts the type of a game's WIN
  from the game's situation.
