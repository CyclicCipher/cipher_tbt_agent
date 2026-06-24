# Agent Design — the self-revising symbolic world-model

> The concrete instantiation of Phase 3 of [DISCOVERY_PROGRAM.md](DISCOVERY_PROGRAM.md): an agent
> for the ARC-AGI-3 replica that *discovers the rules of each unknown game and plans in them*. Its
> learning mechanism is Phase 2 (MDL-as-flux relaxation) over an object-relative representation
> (Phase 1), and it is honestly downstream of the Phase 0 gate (is discovery downhill?).
>
> **Status:** design. `theoretical` unless noted. 4 GB constraint; the symbolic core is CPU.

---

## 1 · The spine: this *is* (discrete) active inference

The architecture is active inference with a symbolic generative model. The mapping is exact, and it
matters because it hands us a candidate **Phase-0 relaxation potential**: variational free energy is
one quantity that perception, action, and learning all descend.

| active inference | here |
|---|---|
| generative model | the **schema** (objects, relations, rules) |
| perceptual inference (min. free energy) | **Perceptor** — infer the object graph from the frame |
| prior preferences | the **inferred goal** predicate |
| expected free energy = epistemic + pragmatic | the **Arbiter** — explore (Explorer) vs exploit (Planner) |
| structure learning: Bayesian model reduction / expansion | the **Inducer** — starve redundant structure (reduction = MDL), grow structure at the residual (expansion) |

So **"MDL as flux" has a precise instantiation: free energy is the flux.** Structure that reduces
free energy (improves the model's account of observations) thickens; redundant structure is pruned
by model reduction; residual free energy the current model can't reduce drives expansion.

**Honest caveat (do not oversell).** Active inference reframes the hard questions; it does not
dissolve them. (a) Structure learning in active inference is itself unsolved. (b) Expected-free-
energy planning is an expectation *over policies* — a search that can re-hide the combinatorial
explosion inside the Planner. So this is the right vocabulary and a strong Phase-0 lead, not a free
lunch. Phase 0 is still the gate.

**Relaxation + directed search (MuZero).** Discovery crosses regime boundaries, and you cannot relax
into a representation you don't have — so the Inducer's *growth* and the Explorer are genuine
(directed) search, while relaxation does the within-regime bulk. MuZero is the precedent: it learns
its model and plans in it by *guided* search (policy prunes, value truncates), and its **learned
value** is what makes long-horizon, sparse-reward planning tractable — so we add a distance-to-goal
value to the Planner (the pragmatic-value term above). MuZero's caution: a *value-equivalent* model
(one that only models what the current goal needs) is efficient but may not transport across levels;
we keep the **explanatory** model and use precision-weighting for efficiency. If the relaxation
Planner fails, MuZero-style guided search is the proven fallback.

---

## 2 · The seed: Core Knowledge priors *are* the rule-language

The rule-language is not invented per task; it is seeded with Chollet's Core Knowledge priors
(Spelke & Kinzler; the priors ARC-AGI is built on) — which double as the schema's type/relation
vocabulary AND as the metric/invariants the binding lesson demands:

| Core Knowledge prior | schema types | schema relations |
|---|---|---|
| **objectness** | object | cohesion, persistence, contact/influence |
| **geometry / topology** | region, shape | adjacency, containment, symmetry, alignment, relative-position, connectivity |
| **number** | count | count, compare, equal |
| **agentness** | agent, action | action → effect, goal-directedness |

Two reasons this is the right seed:
- **It is the same thing as the metric lesson.** Objectness = the object-relative metric; geometry =
  the spatial metric; agentness = the causal metric. Seeding Core Knowledge = giving the model the
  right metric — exactly what made PoPE win in Phase 1.
- **It is bitter-lesson clean and it narrows the landscape.** These primitives are domain-*blind*
  (objectness applies to any game; they are not "arithmetic tokens"). Restricting rules to "typed
  relations over Core-Knowledge primitives" makes the hypothesis space far smaller and more
  structured than arbitrary programs — which is precisely what could make the discovery landscape
  *benign* (downhill). **The seed is a direct candidate answer to Phase 0's "what rule-language
  makes discovery relax."** The Kaleidoscope "atoms of meaning" = these innate seeds + atoms the
  Inducer mines from experience.

---

## 3 · Components (inputs → outputs)

Boundary — **in:** `FrameData` (64×64 grid, score, state, available actions); **out:**
`(GameAction, optional (x,y))`.

| component | input | output | carries |
|---|---|---|---|
| **Perceptor** | frame grid + previous object graph | object graph: objects (color, shape, position), relations, the inferred agent-object — in **object-relative** coords | the binding/metric lesson (§5 discusses how) |
| **Schema / world-model** *(store)* | written by Inducer + Goal-inferrer | typed objects + relations + rules (`context → effect`) + goal predicate, each with a strength (the flux) | the regime |
| **Inducer** | `(graph_{t-1}, action, graph_t)` + schema | updated schema: reinforce predicting rules, **grow** a rule at the residual, starve unused (Bayesian reduction/expansion) | Phase 2 — relaxation (reinforce/starve) + directed structure-search (grow) |
| **Goal inferrer** | score history + object graphs | goal predicate — the target configuration the score rewards | goal-from-score (no instructions given) |
| **Planner** | current graph + schema rules + goal + learned value | next action / short plan, by **relaxing toward the goal, guided by a distance-to-goal value** | relaxation as default; guided search (MuZero) is the fallback |
| **Value** *(learned)* | current graph + goal | distance-to-goal estimate (how close to reward) | long-horizon planning tractable (MuZero); the EFE pragmatic term |
| **Explorer** | schema residual / uncertainty | action maximizing expected residual reduction (epistemic value) | directed data-gathering search (epistemic value) |
| **Arbiter** | model confidence, goal-known?, the two proposals | the action sent to the env; explore/exploit switch; **carries the schema forward at level transitions** | transport across levels (EFE) |

## 4 · The loop

`observe FrameData → Perceptor builds graph_t → Inducer updates schema from the last transition →
Goal-inferrer refines the goal from the score → Arbiter picks explore vs exploit → Planner or
Explorer emits an action → env.step → repeat.` On a new level the schema is **carried forward** and
only the residual (the new mechanic) drives fresh rule-growth — the composition/transport measured
in Phase 1.

## 5 · The perceptor — the neuro-symbolic seam

The Perceptor's hard job is objectness: turn a grid into a discrete, object-relative object graph.
Three routes, in increasing reliance on deep learning:

1. **Symbolic segmenter** (default, 0 VRAM): group cells by color-connectivity + persistence across
   frames. Cheap and bitter-lesson-clean, but brittle on ambiguous grids (what is "one object").
2. **Small object-centric DL** (slot attention): the real DL answer to objectness — unsupervised
   decomposition into object slots. Needs some training/VRAM; the slots are continuous, so a
   discretization step feeds the symbolic core. The only place the *Parked* transformer budget might
   be justified.
3. **Encoder-free ingest + emergent objects** (speculative, the interesting one): borrow Gemma 4's
   encoder-free move — patchify the grid, project each patch with a single matrix multiply into a
   token field (we already do this) — and **let objects emerge from the symbolic relaxation itself**:
   objects are the stable units the rules keep binding to, condensing out of the patch field like
   slime-mold tubes condense out of flow. This is where DL's cheap ingestion and the symbolic core
   genuinely fuse, and it dissolves the perception/reasoning boundary instead of bolting two systems
   together. Highest risk, highest payoff.

**Resource note:** Gemma 4 itself needs 16 GB — we can't run it; only the encoder-free *principle*
(linear patch projection, no heavy encoder) ports, and it's cheap.

## v0 status (built + tested, 2026-06-12)

Built in `agent/wm/` (perceptor / world_model / planner / agent), symbolic + CPU, runs locally.
With **zero LockPath knowledge** it discovers, from the frame and score alone: the agent-object
(`agent_color=2`, by watching what moves), its own controls (the four action→delta deltas — not
assuming ACTION1="up"), the background, the goal color (`3`, from the score signal), and blocker
colors (`1` walls, `5` doors). It then plans (BFS over the learned model) and transports the schema
across levels. **Result: 1.7 levels/game over 10 seeds (random 1.0); always solves the
navigation level by genuine discovery, often the next by luck.**

The ceiling is exactly the predicted boundary: it handles anything reducible to *discover controls +
goal + navigate*, and **breaks where genuine causal structure must be discovered** (key→door,
block→pad). It only passes those when exploration *accidentally* collects the key, etc. — it does
not yet model the causal rule. That break is the **residual** the Inducer must grow structure for
(Phase 2's structure-proposing search) — the concrete next problem, and the first real test of the
Phase 0 gate. The Planner (BFS) is the bounded-search side; a learned value (§1 MuZero) is not yet
needed at this grid scale but is the next addition for longer horizons.

**Iteration 1 — surprise-driven causal induction (2026-06-12).** Added two Core Knowledge priors —
*persistence* (predict nothing changes but the agent → flags surprises) and *contact* (attribute a
surprise to the color just contacted). The Inducer now learns causal rules `contact C -> opens D`
(verified: learns key→door `{4:{5}}` from surprise in 7/10 seeds), and the Planner reasons over
`(position, opened-set)` so it can route "reach the key, which opens the door, then the goal" (probe:
a closed-door level is unplannable without the rule, 12 actions with it). **But the metric didn't
move (still 1.7 levels).** Diagnosis (the procedure): the structure is *correct* but its benefit is
*masked* — within L1 the rule is learned only after the key is already touched (door open → rule
redundant), and its transport value (deliberate use in L3) is gated by the still-unmodeled *push*
mechanic. Two lessons: (a) the Planner's `(pos, opened-set)` state is the live seed of the
`2^switches` BFS blow-up (the "ill-conceived-when-scaled" component); (b) failures are
multiply-gated — one correct addition need not move the metric. **Iteration 2:** model *push*
(contact → object translation) to unlock L2/L3, and add *epistemic* exploration (go contact
unexplored colors to learn rules before needing them — the active-inference drive that fixes the
"learned too late" masking).

**Iteration 2 — epistemic exploration + push detection (2026-06-13).** Added epistemic exploration
(deliberately plan to *contact* objects whose effect is unknown), death-avoidance, and push
*recognition* (contact → object translation → `pushable_colors`). First run: **no change (still
1.7)** — traced to a *wrong* component, not a missing one: the `tries≥3 → no-op` resolution
heuristic gave up learning an action's delta after 3 blocked attempts at one wall, so some seeds
never learned "up" and couldn't reach the door. Fix: an action is a no-op only if it fails to move
from ≥4 *distinct* positions. Result: **reliable 2.0 levels — every seed solves L0+L1 by
understanding** (iteration 1's causal rule finally pays off, exactly as the diagnosis predicted).
Push detection is verified correct (unit probe) but **masked at L2**: the agent beelines to its
believed goal (reach color-3) and fixates there, never pushing the block, because L2's true win is
*conjunctive* (goal reached AND pad covered) — a too-simple goal model plus exploit-fixation.
**Iteration 3:** learn structured/conjunctive goals (no reward when the goal-color was reached →
the goal hypothesis is incomplete → keep exploring), and fix the arbiter so a believed goal that
yields no reward is distrusted.

**Iteration 3 — context-dependent goal, the QKG lift (2026-06-13).** Lifted the goal from a
globally-valid color to a context-dependent triplet `F_τ(C)` ([[reference_quantum_knowledge_graph]]),
learned from the *sparse win signal* (MuZero: the reward function is *learned*, not given;
[[reference_arc_agi3_signals]]) by contrasting win-contexts vs reached-goal-but-no-win-contexts.
**Verified:** from L0/L1 wins (`{1,3}` present) vs the L2 reach-no-win (`{1,3,6,7}`), the agent
induces `required_absent={6,7}` — the goal now reads "reaching goal-3 wins *iff* block-6 and pad-7
are absent." The arbiter **un-fixates** (won't beeline to a goal whose context isn't met), and
behaviour changed correctly: it now investigates the block/pad (`contacted` gained 6,7; learned
`pushable={6}`) instead of oscillating on the goal (iter-2 had `contacted=[3,4]`, `pushable=[]`).
**Metric flat at 2.0, masked two ways:** (a) the condition is *over-constrained* (`{6,7}` not `{7}`)
— genuine sparse-signal **under-determination**: the agent can't know 6 may be present at a real win
without ever seeing one; (b) achieving the condition needs *push-to-target* planning. **Iteration 4
(entangled):** deliberate push-to-target planning (shove block-6 onto pad-7 → 7 absent) would cover
the pad, reach the goal, and win — and that win, recorded with 6 present, would **self-correct** the
over-constraint to `{7}`. So one mechanism both solves L2 and disambiguates `F_τ(C)`.

**Iteration 4 — push-to-target planning; the loop closes (2026-06-13).** Added a Sokoban-style
`plan_push_to` (BFS over `(agent, block)` — the `pos × world-state` blow-up made explicit), an
arbiter that *makes the goal sufficient* by pushing a block onto a coverable required-absent color
(cover the pad), and an experiment-of-last-resort (reach the goal anyway to test an over-constrained
condition). Two bugs found by tracing, not assuming: (1) the experiment fired before the agent had
explored the block — reordered so push-to-target/exploration precede it; (2) goal-seeking BFS shoved
the covering block *off* the pad to reach a goal behind it — fixed by routing goal-plans *around*
pushable blocks (`avoid` in `plan_to`). Also fixed a `move_model[RESET]=(-3,-3)` corruption (the
post-death RESET teleport was induced as a move). **Result: mean 3.17/4 levels, 6/12 seeds win the
ENTIRE replica** (navigate → key→door → push-block-onto-pad-as-the-goal's-context → avoid the hazard
→ compose all of it on L3), e.g. seed 6 wins in 95 actions. **The entangled loop closed exactly as
predicted:** covering the pad + experimenting wins L2, and that win — recorded with the block
present — self-corrects `required_absent {6,7}→{7}`, which is what makes L3 solvable. Remaining
failure (4/12 stuck at L2): **Sokoban deadlocks** — undirected pushing during exploration shoves the
block against a wall, after which it can't be repositioned onto the pad.

**Iteration 5 — deadlock by experience, not avoidance (2026-06-13).** Rather than fence off risky
pushes (a hand-coded "don't push toward walls" would be domain-specific *avoidance by default*), the
agent learns from the mistake: when its *own model* says the cover is unreachable, it RESETs and
remembers. The subtlety — found by tracing, not assuming — is telling a *true* deadlock from "not
reachable yet": the first cut reset-looped 5934× on L3 because the block's path to the pad runs
through a *closed but openable* door, which `plan_push_to` reports as unreachable. Fixed with an
**optimistic reachability probe** (`plan_push_to(..., passthrough=openable ∪ unknown)`): only if the
block can't reach a target *even treating doors/unknowns as passable* is it walled in by permanent
blockers — a genuine deadlock. Then it records the block's cell in `dead_block_cells`, RESETs, and
afterward refuses to *wander* the block into a learned dead cell (`_would_strand`; deliberate
goal-directed pushes are untouched — it only learns the specific cells it was burned by, never a
blanket rule). **Result: 12/12 seeds win the entire replica** (up from 6/12); the six that used to
strand recover in 1–3 resets. RHAE-proxy **31.5%→37.9%**. The L3 hazard over-constraint
(`required_absent` gaining `8`) did not manifest on these 12 seeds but remains latent — see
`LIMITATIONS.md §5`.

## 6 · Dependencies & open questions

- **Downstream of Phase 0.** The Inducer's *growth* and the Planner/Explorer are exactly what Phase 0
  must keep *directed*: relaxation for the within-regime bulk **plus** directed (residual / value /
  epistemic-value-guided) search for crossing boundaries — neither blind enumeration nor pure
  relaxation. Keeping that search directed is the whole bet.
- **Value-equivalence vs transport** (MuZero): the explanatory model transports across levels but
  costs efficiency; precision-weighting is the hoped resolution.
- **Perceptor is a hard prerequisite.** General objectness without per-game tuning is unsolved in
  ambiguous cases; wrong objects → every downstream rule is learned over garbage.
- **EFE planning can re-hide search** (§1 caveat) — watch the Planner.
- **The bitter-lesson knife-edge** applies to every operator (segmenter, inducer, goal-inferrer,
  reinforce/grow rule): domain-general or it's CTKG-v1.
