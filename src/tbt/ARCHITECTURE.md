# ARCHITECTURE.md — the brain we are building (design + rationale)

**This doc is the stable DESIGN and its neuroscience grounding. It contains NO status or progress claims** — those rotted
the previous ARCHITECTURE.md ("P3 done / 137 passed" while the branch was at 80). Current build state lives ONLY in
`STATUS.md` (derived from the code). The build discipline lives in `RULES.md`. This doc says *what* the brain is and *why*.

---

## 1. The core claim

**The neocortex is a value-free sensorimotor WORLD-MODEL.** It predicts the consequences of movement and proposes actions
(via L5), but it has no goals and no value of its own — it models *structure*, not *worth* (Hawkins's "old brain / new
brain": the neocortex learns the model; the old/subcortical brain holds the goals, drives, and value).

Therefore **planning is not something the cortex does. Planning is what a LOOP does *with* the cortical model.** We do not
build "a planner module." We build the cortical model and wire it into the cortico-basal-ganglia-thalamo-hippocampal loop
that plans. This is why the previous codebase — which built the cortex and left the loop's other arms orphaned (basal
ganglia un-wired in a "collapse", thalamus and forward model never wired) — could perceive but never plan.

## 2. ONE reusable column, used TWICE (TBT's uniform cortex)

Every cortical region runs the *same* column algorithm — learn a model in a reference frame by sensorimotor prediction;
regions differ only in what they connect to and what they model. So the same column is instantiated on two kinds of space:

- **SENSORY columns** — model **physical / object space** (feature-at-location, forward prediction, recognition).
- **PFC columns** — model **abstract TASK / GOAL space** (the goal-state, the plan hierarchy). *Same algorithm*, an
  abstract reference frame instead of a sensory one. The PFC is **not** a separate module — it is the column on a
  task-space. (Grounded: grid-like reference-frame codes for abstract/conceptual spaces are found in entorhinal cortex and
  medial PFC, the same machinery as for physical space — Constantinescu/O'Reilly/Behrens 2016; Bellmund et al. 2018.)

The goal-state and the plan hierarchy are therefore just "stable objects in L2/3" — the same recognition/pooling mechanism
as a sensory object, but the object is a goal/plan and the frame is abstract.

## 3. The planning loop (each piece, its role, and where it comes from)

```
   env frame                                                            action
      │                                                                    ▲
      ▼   (agent.py = pure plumbing: NO decisions)                         │
 ┌────────────────── NEOCORTEX — the value-free model ──────────────────┐  │
 │  SENSORY columns  ──►  world-model (predict, propose moves)          │  │
 │  PFC/TASK columns ──►  goal-state + plan hierarchy                    │  │
 └───────┬───────────────────────────────────────┬─────────────────────┘  │
         │ forward model                          │ the held goal          │
         ▼                                        ▼                        │
 ┌──────────────────────── HIPPOCAMPUS — ROLLOUT ───────────────────────┐  │
 │  sweeps candidate future trajectories (preplay/VTE) by querying the   │  │
 │  cortical forward model step by step, toward the goal                 │  │
 └───────────────────────────────┬──────────────────────────────────────┘  │
                                  │ candidate trajectories                   │
                                  ▼                                          │
 ┌──────────────────── BASAL GANGLIA — VALUE + SELECT ──────────────────┐   │
 │  score by reinforcement (dopamine RPE) → SELECT by disinhibition      │   │
 └───────────────────────────────┬──────────────────────────────────────┘   │
                                  │ the winner                               │
                                  ▼                                          │
 ┌──────────────────────── THALAMUS — GATE / RELAY ─────────────────────┐   │
 │  gate the selection back to cortex; drive the motor output ───────────┼───┘
 └───────────────────────────────────────────────────────────────────────┘
```

| loop element | role | legacy source (reuse via RULES.md #5, re-wire, re-test end-to-end) |
|---|---|---|
| sensory column | physical world-model + recognition | `l4*`, `l5_displacement`+`operator`, `l23_object`, `sequence` (forward model), `encoders`, `retina` |
| PFC / task column | goal-state + plan hierarchy (same column, abstract frame) | the SAME column, instantiated on a task/goal space (`reference_hierarchy_substrate`) |
| hippocampus | ROLLOUT: prospective sweep over the model + episodic binding | `hippocampus` (repurpose from localization-only) |
| basal ganglia | VALUE (RPE) + action SELECTION (disinhibition) | `basal_ganglia` (`BasalGanglia`, `OpponentActor`) — orphaned, resurrect |
| thalamus | GATE / route the selection; L5's driver target | `thalamus` — underdeveloped, needs real work |
| value-at-leaf | the SR as a rollout leaf bootstrap (NOT the whole value) | `l6_sr.SuccessorFeatures` |

The computational spec of the loop is EfficientZero-V2 (learned model = cortex, value = BG/dopamine, policy/select = BG,
search/rollout = hippocampal preplay) — the SAME loop from the algorithm side. We take EZ-V2's *algorithm*, not its
gradient-trained nets.

## 4. One decision, end to end
Perceive → sensory columns update the world-model; the PFC column holds/updates the goal-state → the hippocampus rolls out
candidate trajectories over the world-model toward the goal → the basal ganglia score by value and SELECT one by
disinhibition → the thalamus gates the selected action back and drives the motor output → `agent.py` enacts it in the
environment → repeat.

## 5. Hard rules (founding constraints — beyond `RULES.md`)

1. **MULTI-COLUMN ALWAYS.** No experiment, and no slice, runs without BOTH a sensory column AND a PFC/task column. Never
   single-column, no exceptions. (A single column can only ever be perception — the exact rabbit-hole that consumed the
   last codebase. The planning loop is only meaningful with both a world-model and a goal-model.)
2. **NO DECISION LOGIC OUTSIDE THE LOOP.** `agent.py` and every script NEVER select an action, compute a value, or pick a
   goal. Selection is the basal ganglia; `agent.py` is environment plumbing only (frame in → enact the loop's output → step).
3. **THE MODEL STAYS IN THE COLUMN.** The hippocampus *orchestrates* the rollout by querying the column's forward model; it
   never holds a copy of the model. (A migrated model = the parallel-copy mistake.)
4. Everything in `RULES.md` still holds: unwired is a bug (reachability test), integration is done, build vertical,
   check-before-building, tests monotonic.

## 6. Honest caveats (do not overclaim)
- PFC being neocortex does not mean it plans alone — it sits in its own cortico-BG-thalamic loop; "same column algorithm" is
  a claim about the *column*, not that PFC does planning solo.
- The abstract task/goal map is *shared* between PFC and the entorhinal-hippocampal system (grid codes appear in entorhinal
  too), which is consistent with the hippocampus rolling out *over* it.
- That hippocampal preplay implements a *specific* search algorithm (MCTS-like) is an active modelling area, not settled
  mechanism — treat the EZ-V2 ↔ brain-loop map as a strong, useful correspondence, not a proven identity.

## 7. Why the column factors the world — the place-invariance lesson (plain English)

Everything above says the column keeps *content*, *location*, and *state* as separate pieces and reuses **one** model
across locations. This section says, in plain terms, *why that is not optional* — the concrete failure that forces it.
(This is design rationale; the experiments and numbers that motivate it live in `STATUS.md` and the memory
`project_place_invariance_needs_factored_state`, not here.)

**The problem.** Take something a person finds trivial: add 1 to a number, one digit at a time, carrying when a digit
rolls over from 9. Build the obvious thing — show the whole number at once and learn "this pattern → that pattern" — and
the model learns the units column, the tens column, and so on **as unrelated facts**. It never notices they are the *same*
operation in different places. The moment it meets a digit position it was never trained on (say the hundreds), it is
helpless, even though the rule there is identical: it scores **zero** on the new position while being perfect on the old
ones.

**Why it happens.** A plain sequence memory (and a plain HTM) has **no weight-sharing across positions** — knowledge
learned at one spot simply does not exist at another spot. This is a documented limitation, not a bug we introduced.
Convolutional nets paper over it by copying one filter everywhere, but that is a spatial hack the brain doesn't use.

**The brain's fix, and ours.** Don't look at the whole thing at once. **Move a sensor over it, one location at a time, and
apply the *same* small model at every location.** The digit becomes *content* (a "5" looks the same wherever it sits); the
position becomes a *location* signal (a separate "where am I" code, used only for addressing, kept out of the content);
anything that has to travel between positions — the carry — becomes a small piece of *state* handed from one fixation to
the next. Because it is literally the same model reused at each location, a brand-new position just works: it inherits the
rule for free. In our tests the monolithic way scores 0% on the unseen position and the sensorimotor way scores 100%.

**The consequences for this design:**
- **This is why a column has more than one layer.** Content, location, and state are different jobs and must be kept
  apart so each can be *reused*; one blob that mixes them cannot generalize across positions.
- **This is why we are always multi-column (§5.1).** The state that travels between locations naturally lives in a
  *second* column (a task/PFC column) whose output becomes the *context* for the sensory column. A single layer's activity
  is pinned to its own content, so it cannot hold that travelling state by itself.
- **The state must be *supplied*, not wished into existence.** This substrate learns transitions online; it does not, on
  its own, invent a hidden variable and discover that remembering it pays off later. So the architecture has to *give* the
  state its own slot — which is exactly what reference frames and a task column are for. We keep that slot **general** (it
  carries whatever a task needs), never a hand-wired "carry bit" — that would be the bitter-lesson trap. **This is TBT's
  actual stance on learning, not a workaround:** credit assignment is solved *structurally, not algorithmically* — a
  reference frame anchors every prediction to a (feature, location), so a mismatch is attributable *locally and immediately*
  and no error is propagated back through time (no backprop; learning is local, associative, unsupervised). The one thing
  that *does* need long-range credit — reward — is the **basal ganglia's** job (dopamine RPE), never the cortex's. So "the
  architecture supplies the factored state" and "TBT provides structure via reference frames instead of discovering it" are
  the same sentence.

## 8. What makes a layer a layer — role = (context-in, target-out), both wired by hand

§2's claim ("one column algorithm; regions differ only in what they connect to") is right, but coding it needs one
refinement that biology makes explicit: a layer's role has **two** determinants from **two different sources**, and only
one of them is "input."

- **What a layer computes** is set by its **input wiring** — which signal drives its basal context (a thalamic location, its
  own recurrence, a top-down goal, another column). The microcircuit itself is *uniform* across all of cortex (Mountcastle);
  thalamic input is only ~10% of the synapses in its target layer, so a region's specialization is almost entirely a matter
  of *what it is wired to* (Douglas & Martin). This is the knob our `HTMLayer` already exposes as `context=`.
- **Where a layer's output goes** is set by **cell-type identity**, fixed in development by birth-order (inside-out) + a
  transcription-factor code (Fezf2/Ctip2 → L5b subcortical **motor**; Tbr1 → L6 cortico**thalamic**; Satb2 → cortico-cortical).
  "L5 is the motor output" and "L6 drives the thalamus" are **not** derivable from what feeds a layer — they are intrinsic
  projection identities.

**So in code a layer is not a subclass — it is one `HTMLayer` instance plus a declared `(context-in, target-out)` pair,
both asserted when we compose the column.** The mechanism stays uniform; the differentiation is entirely the wiring, and the
output half we assert by hand exactly as development asserts it genetically. This is also *why* we never expect a layer's job
to *emerge*: like the reference frame in §7, it is **given, not learned**.

## Sources
- Hawkins, *A Thousand Brains* — cortex = model, old brain = goals/value; uniform cortical algorithm.
- Redgrave/Gurney/Prescott — basal ganglia as the vertebrate solution to the selection problem; cortico-BG-thalamo-cortical
  loop, selection by disinhibition.
- Miller/Botvinick/Brody 2017, *Nat. Neurosci.* — dorsal hippocampus is causally necessary for model-based planning.
- Hippocampal preplay / vicarious trial-and-error / SWR — prospective simulation of candidate futures.
- Constantinescu, O'Reilly & Behrens 2016, *Science*; Bellmund et al. 2018, *Science* — grid-like reference-frame codes for
  abstract/conceptual spaces in entorhinal + medial PFC.
- Mountcastle 1978 — the uniform cortical column: one repeated circuit across all neocortex (§2, §8 premise).
- Douglas & Martin 2004, *Annu. Rev. Neurosci.* — the canonical cortical microcircuit; thalamic input is ~10% of synapses,
  so a region's function comes from its CONNECTIVITY, not a different circuit (§8, the context-in half).
- Molyneaux/Arlotta/Macklis (projection-neuron fate); Fezf2/Ctip2 → L5b subcerebral, Tbr1 → L6 corticothalamic, Satb2 →
  callosal — birth-order (inside-out) + a transcription-factor code fixes laminar/projection IDENTITY (§8, the target-out half).
- Thousand Brains Project 2024 (arXiv 2412.18354) — learning is local, associative, unsupervised; credit assignment is solved
  STRUCTURALLY by reference frames (error localized to a feature-at-location), not by backprop (§7).
