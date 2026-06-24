# arc_agi_3 — an offline replica of the ARC-AGI-3 environment

A faithful, in-process replica of the **ARC-AGI-3 interactive reasoning benchmark**
(arcprize.org/arc-agi/3, docs.arcprize.org). The harness and object model match the
real benchmark closely enough that an agent written against this ports to the real
REST API with minimal change; original games stand in for the proprietary ones so
it is runnable entirely offline.

## What ARC-AGI-3 is

Unlike ARC-AGI-1/2 (static input→output grid puzzles), ARC-AGI-3 is **interactive**:
an agent is dropped into an unknown turn-based game with **no instructions, no win
condition, no language** — only pixels and a score. It must explore, infer the goal
on the fly, build a world model, and learn. It measures *skill-acquisition
efficiency over time*, the quantity Chollet defines as intelligence. The real suite
is 6 games × 8–10 levels, each level introducing a new mechanic; environments are
constrained to **Core Knowledge priors only** (objectness, geometry, basic physics,
agentness).

## Fidelity — what is faithful vs approximated

**Faithful (the object model / harness):**
- Observation is a *frame* = list of 1..N grids, each up to **64×64**, integer color
  values **0..15**, indexed `grid[y][x]` with `(0,0)` top-left.
- Action space: `RESET`, `ACTION1..4` (up/down/left/right), `ACTION5` (game-specific
  interact), `ACTION6` (complex — requires `(x,y)` in 0..63), `ACTION7` (undo).
- Every frame advertises `available_actions`; submitting anything else raises
  `ActionNotAvailable` (the real API's HTTP 400). On `GAME_OVER`, only `RESET` is legal.
- Lifecycle `NOT_PLAYED → NOT_FINISHED → WIN | GAME_OVER`.
- Scoring: `score` = levels completed; `action_counter` (total actions) is the
  tiebreaker, rewarding efficient exploration. `Scorecard` aggregates across games.
- Levels introduce mechanics progressively; later levels compose earlier ones.

**Approximated (necessarily):** the real games `ft09/ls20/vc33` (and the private
`sp80/lp85/as66`) are proprietary and only playable via ARC's servers — they are not
published as code. So the *games* here are **original**, built to the same design
rules, not reverse-engineered copies. The harness is the faithful part; the games are
faithful *in spirit*.

## Layout

```
arc_agi_3/
├── core.py        GameAction, GameState, FrameData, Grid/Frame types, GRID_SIZE
├── game.py        Game ABC — the contract a game implements
├── harness.py     Environment (lifecycle/gating/scoring) + Scorecard + GameResult
├── agents.py      Agent ABC, RandomAgent, run_episode()
└── games/
    └── lockpath.py  LockPath ("lp01") — 4 levels, original
tests/test_arc_agi_3.py   lifecycle, gating, a BFS solver proving every level → WIN
```

## Quick start

```python
from arc_agi_3 import Environment, RandomAgent, run_episode
from arc_agi_3.games import LockPath

env = Environment(LockPath())
result = run_episode(env, RandomAgent(seed=0), max_actions=2000)
print(result)            # GameResult(game_id='lp01', won=..., levels_completed=..., ...)
```

Write your own agent by subclassing `Agent` and implementing `choose_action(frame)`.

## LockPath ("lp01") — the bundled game

A grid world using only Core Knowledge priors. Each level adds one mechanic:

| Level | Mechanic | Prior exercised |
|---|---|---|
| L0 | navigate to the goal | agentness, geometry |
| L1 | key opens a door | causality/objectness |
| L2 | push a block onto a pad | basic physics |
| **L3** | **key+door AND block+pad together (+ a hazard)** | **composition** |

**Why L3 matters here.** L3 composes two mechanics that were each introduced *in
isolation* (L1, L2) into a single puzzle that requires both. That is a small,
in-environment instance of the RecurrentWorldModel **A∘B compositional-generalization
test** — and, in Chollet's terms, exactly the **program-centric abstraction** assay:
can a system recombine separately-learned atoms to solve a novel composite? See
`../chollet_connection.md`.

## Next steps (not yet built)

- More games, especially ones that use `ACTION6` (coordinate clicking) and `ACTION5`
  (interact), to exercise the full action space.
- An optional thin HTTP shim mirroring the real REST endpoints, so the *same* agent
  code can target this replica or the live ARC servers by config.
- A scripted/oracle agent per game (for regression + to bound the optimal action
  count) and a simple learning agent as the first research vehicle.
