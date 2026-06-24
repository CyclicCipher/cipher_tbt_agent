# Replica test — the prior-minimal Phase-2 agent on ARC-AGI-3

> **Goal.** Solve the `LockPath` replica (all 4 levels) with **only the bare floor of domain-general
> priors**, discovering everything the Phase-1 agent had *seeded*. Then compare to the Phase-1
> baseline. New code lives in `agent/wm2/` (the proven `agent/wm/` stays untouched as the baseline).
>
> **Baseline to beat (Phase 1, `agent/wm/`):** 12/12 seeds win all 4 levels, RHAE-proxy 37.9%
> (`docs/phase1/FINDINGS.md`).

## Prior accounting (the whole point)

**KEPT — the floor (domain-general, unavoidable; ref `VOLUME_CONCEPTS.md §0`):**
- **Sensory interface** — a frame is a grid of `(x, y, color)` cells; the action is an *opaque* key
  (no assumed semantics); the score and game-state are observed. This is the ARC-AGI-3 I/O contract.
- **Metric / temporal continuity** — position is a metric; a colour that leaves one cell and appears
  at a nearby cell is the same unit moving (nearest-position matching). Universal in spatial domains.
- **Compression / prediction (MDL)** — a regularity counts only if it holds consistently across
  observations and shortens the description; this drives every discovery via the `volume/` machinery.

**REMOVED — the Phase-1 seeded priors (now to be discovered):**
- ~~Agency = "the single cell that translates is me"~~ → **discovered**: whichever colour the actions
  control (the action-consistent, localized colour).
- ~~Objectness = "cells of a colour are objects"~~ → presence/locality + the `volume/` concepts.
- ~~Contact = "attribute change to the colour just touched"~~ → **relation discovery** on transitions.
- ~~The rule-type vocabulary (move / block / open / push / context / death)~~ → `relation.fit`
  (any conditionable transition manifold) + region concepts; no hand-authored rule types.

## Increments (checklist)

**Approach (corrected 2026-06-13): ENHANCE, don't rebuild.** Phase 2 *modifies* the Phase-1 agent —
it must not re-implement the acting loop, exploration, BFS planner, goal logic or deadlock learning,
which already exist and are proven. So `agent/wm2` **subclasses** `agent/wm`: `VolumeAgent(WorldModelAgent)`
inherits the whole machinery, and a `DiscoveredWorldModel(WorldModel)` swaps **only the seeded priors**
in the induction, **one at a time, verifying the win holds after each**. (Discovery is therefore
*online inside the reused loop* — random-walk data alone never reaches the key/block, so it could not
support offline edge discovery anyway.) The Phase-1 baseline stays untouched for comparison.

Prior removals (each a small, verified diff):

- **A. Agency — [DONE 2026-06-13, 12/12 held].** Replaced Phase-1's single-cell prior
  (`detect_move` requires exactly one gained + one lost cell) with discovered agency
  (`discover_dynamics`: the action-controlled, localized colour — no single-cell assumption), then ran
  the reused induction on that discovered agent. **All 12 seeds still win all 4 levels** (`agent_color`
  discovered == 2 every time). Tests: `tests/test_wm2_agent.py`, `tests/test_wm2_perceive.py`.
- **B. Contact rule-types — [DONE 2026-06-13, 12/12 held].** Replaced Phase-1's two hand-coded
  detectors — a `push` template tied to the cell `q+Δ`, and an `open` scan for colour→bg — with ONE
  general **residual-signature** learner (`DiscoveredWorldModel._learn_contact`): the contacted colour
  *reappearing* ⇒ displaced (pushable); another colour *vanishing* ⇒ removed (an opening). No effect
  type is templated to the agent's geometry. **All 12 seeds still win.** It discovers `pushable={6}`
  and `opens={4→5, 6→7}` — and the `6→7` (pushing the block onto the pad makes the pad vanish) is the
  *same* residual mechanism finding block→pad that finds key→door, with no separate template. So the
  push/open geometry priors were inessential too. (Note: the effect *categories* push/open still exist
  at the planner interface — fully removing them needs a forward-simulating planner; deferred.)
- **C. Objectness / goal-condition.** If still seeded after B: discover the goal context-condition as a
  learned `volume` region (generalizing `required_absent`); revisit any remaining objectness assumption.
- **Checkpoint after A+B (2026-06-13).** `VolumeAgent` = 12/12 wins, **RHAE-proxy 37.5%** vs the
  baseline's 37.9% — *equal performance on a strictly smaller prior floor*. The two seeded priors were
  inessential. **This remains the working prior-minimal agent.**
- **C. Typed planner → forward-simulation [attempted 2026-06-13; partial, parked].** `ForwardPlanAgent`
  + `plan.py` replace Phase-1's three typed planners and the typed `_decide` (exploit/cover/experiment)
  with ONE BFS that forward-simulates the discovered transitions to a winning state — cover/open/
  experiment *emerge from search*. **It generalizes exploitation** (L0/L1 solve with no special-casing;
  the open-door and cover-pad plans emerge), but caps at **~2–4/12** on the full game. The gap is
  diagnostic, not a bug: the flat forward-BFS over primitive moves is the **predicted scaling wall**,
  and reaching 12/12 means re-deriving every Phase-1 reactive fix (deadlock dead-cells, hazard self-
  correction, directed epistemic discovery) inside a slower planner — negative architectural value.
  The win-count bounced (0→4→2) with each reactive patch — the thrashing signal. **Finding:** the gap
  between flat forward-sim and the baseline *is* the reactive machinery that doesn't reduce to flat
  planning. Kept as the foundation for the right answer, not wired as the production agent.
- **Hierarchical planner — BUILT, the architecture works (2026-06-13).** `hplan.py` +
  `HierarchicalPlanAgent`: plan over the **discovered edges as macros** (reach-trigger / push-onto-
  target / reach-goal), composed in prerequisite order — *Merge applied to action* — each refined by a
  focused BFS. **Both predictions confirmed:** (1) the scaling wall is gone — seed 2 solves the full
  game in **120 steps** (vs the flat planner's slow/timeout wins); test
  `test_hierarchical_planner_can_win_by_composing_macros` locks it in. (2) deadlock is a clean
  **high-level infeasibility** — when the block is stranded, the PUSH macro has no refinement and
  `hplan` returns `None` (verified: `plan_push_to → NONE`, `hplan → None`), exactly as designed.
- **The residual gap is the control policy, NOT the planner — and hand-tuning it is the wrong move.**
  Win-count sits at ~2–4/12 because the AGENT'S orchestration of *plan ↔ explore ↔ epistemic-probe ↔
  reset* is a pile of hand-tuned branches, and tuning them thrashed (2→4→2→0 across edits; a premature
  deadlock-reset that fired on L2 entry took it to 0/12 before being reverted). This is the same
  thrash the owner twice warned about. **The principled fix is to stop hand-coding the control policy
  and derive it** — active inference / expected-free-energy, which unifies explore (epistemic value),
  exploit (pragmatic value) and experiment/recover under ONE objective (the project's active-inference
  spine, `DISCOVERY_PROGRAM.md`). The hierarchical *planner* is the exploitation half of that EFE; the
  missing half is principled action-selection, not more reactive branches.
- **EFE control policy — attempted (`EFEAgent`), did not reach parity (2026-06-13).** Replaced the
  reactive branches with value-ordered selection (pragmatic = the hierarchical planner / exploit a win;
  epistemic = reduce model uncertainty with honest accounting via `_block_landings`; recover = reset
  only when uncertainty is exhausted). It **exposed a real latent planner bug** — when the block covers
  the pad, rebuilding the static `base` grid erases the pad, so the win-test never recognizes the pad
  as already-satisfied → reset loop (the earlier hierarchical agent had been masking this with lucky
  coverage-wandering). But the obvious fix (treat already-absent required colours as satisfied)
  **regressed the hierarchical agent** (seed 2 stopped winning), and EFE still scored 0/12. **Conclusion
  (honest):** the planner ✕ discovery ✕ recovery interactions are a genuine thicket — not resolvable by
  incremental fixes without breaking something else. Stopped to avoid thrashing (the score oscillated
  4→2→0 across control edits).
- **Net replica-test result.** **`VolumeAgent` (A+B) is the deliverable: 12/12, 37.5% RHAE, on a
  strictly smaller prior floor.** The two seeded priors removed (single-cell agency; push/open
  geometry) were inessential — equal performance proves it. The typed planner *generalized* in form
  (flat → hierarchical, fast, clean deadlock-as-infeasibility — `HierarchicalPlanAgent` wins seed 2 in
  120 steps), but turning the general planner into a *fully prior-minimal winning agent* needs a
  principled control policy whose planner ✕ discovery ✕ recovery coupling is harder than the planner
  itself — the real open problem, and one for deliberate design, not more tweaks. `ForwardPlanAgent` /
  `HierarchicalPlanAgent` / `EFEAgent` are kept as documented WIP, not production.

- **Inventory + disciplined integration (2026-06-13, `AGENT_STACK.md`).** Mapped the whole stack;
  L1/L2 (perception, world model) and L3 (planners) are solid/separable — *all* trouble is in L4
  (control). **Step 1 — port the deadlock triad** (learn `dead_block_cells` + avoid `_would_strand`,
  which the wm2 agents had dropped): done, suite green, **win-count unchanged (2/12)** → a clean
  *isolated negative*: the deadlock triad was missing but is **not** the bottleneck. **The bottleneck
  is the covered-pad bug** — *confirmed* (seed 0: 228 steps with the pad covered, still required, yet
  `hplan=None`): when the block covers the pad, the rebuilt `base` loses the pad and `opened` starts
  empty, so the planner can't represent "already covered." **But every local fix regresses the one
  working seed (seed 2)** — a permissive win-test, a static-layout idea, and seeding `opened` from the
  live grid each broke seed 2. **Finding:** the planner's *context* logic is **tangled** —
  `required_absent` (the F_τ(C) condition) ✕ `opened` (removed in-plan) ✕ `removable` ✕ already-absent
  ✕ the experiment are scattered and interacting, so the working seed balances on the current (buggy)
  logic. **This is not patchable; it needs a clean redesign of "is the goal's context satisfied in a
  simulated state" as ONE coherent notion** — the identified next piece of work.

## Honest expectations

**Matching the baseline is not required** — the deliverable is the *prior-minimization*. Increment A
shows the agency prior was inessential (the win survived its removal). Later removals (rule-types, goal
condition) are real research risk and may cost seeds; that loss is itself the finding — it tells us
exactly which priors were load-bearing and where the floor is genuinely too thin. Each removal is a
small diff against a passing baseline, so a regression localizes the cause immediately.
