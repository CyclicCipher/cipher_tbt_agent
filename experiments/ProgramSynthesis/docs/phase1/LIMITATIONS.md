# WorldModelAgent — Limitations & Domain-Specificity

> An honest accounting of what the v0–iter4 symbolic agent (`agent/wm/`) does NOT do, what it
> assumes, and where it is fit to LockPath rather than general. Written 2026-06-13 after the agent
> reached 6/12 full-game wins. The headline: **the *solution* is not baked in, but the agent's
> *capabilities* were shaped reactively by LockPath's mechanics, and generality to an unseen game
> is untested.**

## 1 · What is NOT in the code (the integrity check)

- **No hardcoded colors.** There is no `if color == 3`. Every "key/door/pad/block/hazard" reference
  in the source is a *comment*. The code keys only on *discovered* structures (`agent_color`,
  `goal_colors`, `contact_effect`, `pushable_colors`, `required_absent`, `blocker_colors`).
- **No game internals.** The agent imports only the `arc_agi_3` interface (`FrameData`,
  `GameAction`, `Agent`). It never touches `LockPath`'s state; the oracle (which does) is a teacher
  for the *transformer* experiments, not used by this agent.
- **Inputs are only what ARC-AGI-3 gives:** the frame grid + the score signal + available actions.
  The goal/win-condition is never read — it is learned ([[reference_arc_agi3_signals]]).

## 2 · The legitimate priors (Core Knowledge — ARC-permitted)

Seeded, domain-blind, and explicitly allowed by the benchmark: **objectness** (a coherent unit),
**agency** (the thing that moves with actions is "me"), **persistence** (objects don't change
without cause → unexplained change is a *surprise*), **contact** (cause is attributed to the color
just touched). These are assumptions, but they apply to any object world, not to LockPath.

## 3 · Simplifying assumptions (real limitations)

These happen to fit LockPath; a game violating them would break the agent:

- **Single-cell agent.** `detect_move` identifies the agent as a *single* cell that translates. A
  multi-cell agent would not be detected.
- **Single-cell blocks.** Push detection and `plan_push_to` track one block cell; multi-cell or
  multiple simultaneous blocks are not handled (and would explode the `(agent, block…)` state).
- **Background = the modal color.** Fails if no color dominates, or if "empty" varies.
- **Goal shape.** The goal is representable as "reach a color **and** have a set of colors absent."
  Goals needing ordering, counts, timing, or relational predicates are out of scope.

## 4 · The engineered rule-*type* vocabulary (the key caveat)

The inducer can only discover rules of the **types we built**, added *reactively* as LockPath's
levels demanded them:

| rule type | added for | discovers (from data) |
|---|---|---|
| movement (`action → Δ`) | navigation | which action moves the agent how |
| blocker (`color blocks`) | walls | which colors stop movement |
| contact-opens (`C → opens D`) | key→door | which contact opens which barrier |
| push (`color translates`) | block→pad | which colors are pushable |
| context-absence goal (`F_τ(C)`) | conjunctive goal | which colors must be absent to win |
| death-avoid (`color is deadly`) | hazard | which colors kill on contact |

**The specific *instances* (which colors) are genuinely discovered; the *types* are ours.** Those
types are general (push = Sokoban, key→door = Zelda, etc.), so this isn't LockPath-specific *per
se* — but a mechanic whose *type* is not in this list (rotation, gravity, conveyor, timing,
multi-step recipes, tool use) would simply not be discovered. **The only true test of "not
test-fit" is a different game we did not build for, and that has not been run.** That is the
honest, unfalsified generality claim.

## 5 · Failure modes

**Handled — Sokoban deadlocks (iter5).** Undirected exploration could shove the block against a wall
where it can no longer reach the pad, stranding the level (was 4/12 stuck at L2). Now handled by
*experience*, not avoidance: when the agent's own model says the cover is unreachable **even
optimistically** (treating openable doors + unknown colors as passable — so a closed door is *not*
mistaken for a wall), it declares a true deadlock, records the block's current cell as a dead end,
and RESETs the level; thereafter it refuses to *wander* the block into a learned dead cell (deliberate
goal-directed pushes are unaffected). It learns by touching the stove once. **Result: 12/12 seeds win
all four levels** (1–3 resets on the six that used to strand). See `agent/wm/agent.py` (`_would_strand`,
the cover/deadlock branch) and `world_model.py` (`dead_block_cells`).

**Latent — hazard over-constraint (L3).** From hazard-free levels the agent can't know whether the
hazard's *presence* blocks a win, so it can over-constrain `required_absent` with the hazard, and the
self-correcting experiment that would refute it is gated by a loose "coverable" test (a block can't
actually be pushed onto a deadly cell). It does **not** manifest on the current 12 seeds (all win L3),
but the under-determination is real and would surface on a layout where the goal is reachable only
*before* covering the pad. Tie "coverable" to an actual `plan_push_to` path to close it.

## 6 · Scoring caveat (RHAE-proxy)

`agent/wm/score.py` grades against the **oracle's optimal** action count — *harsher* than the real
RHAE's **human** baseline, because **this is a learning agent that must explore** (it does not know
the rules), and the oracle never explores. So **~100% vs the oracle is not the target**: a large
part of the agent's action count is *irreducible exploration*, which a human (untrained on the game)
also pays. Our **37.9% vs oracle** (12/12 seeds completing all 4 levels) likely understates
performance vs a human baseline (unknown). The right use of this number is as an *internal* yardstick
to drive efficiency up, not as a comparison to the real benchmark's %.

## 7 · One-line generality boundary

> The agent **discovers the specific rules of a single-cell-object gridworld whose mechanic *types*
> are already in its inducer's vocabulary**, learning everything else (which colors, which controls,
> which goal, which conditions) from the frame and the score. It is validated on LockPath only.
