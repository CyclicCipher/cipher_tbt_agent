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

## 3. The planning loop — the regions, their substrate, and their contracts

```
   env frame                                                        action
      │                                                                ▲
      ▼   (agent.py = pure plumbing: NO decisions / value / senses)    │
 ┌────────────── NEOCORTEX — the value-free world-model ───────────┐   │
 │  SENSORY columns  → predict feature-at-location, propose L5 move│   │
 │  PFC / TASK columns → the goal-state + plan hierarchy           │   │
 └──────┬───────────────────────────────────┬─────────────────────┘   │
        │ forward model                     │ the held goal           │
        ▼                                   ▼                         │
 ┌────────────── HIPPOCAMPUS — ROLLOUT ────────────────────────────┐   │
 │  sweep candidate futures (preplay/VTE) over the cortical model, │   │
 │  toward the goal  →  candidate trajectories                     │   │
 └───────────────────────────┬─────────────────────────────────────┘   │
                             ▼                                         │
 ┌────────── VALUE / DOPAMINE — the CRITIC (old brain) ────────────┐   │
 │  expected reward, learned by TD as the SR read-off V = w·(M·R); │   │
 │  scores the candidates, emits the scalar RPE δ + tonic ρ        │   │
 └───────────────────────────┬─────────────────────────────────────┘   │
                             │ values + δ                              │
                             ▼                                         │
 ┌────────────── BASAL GANGLIA — SELECT ───────────────────────────┐   │
 │  SELECT the best candidate by disinhibition; δ trains Go/NoGo   │   │
 └───────────────────────────┬─────────────────────────────────────┘   │
                             │ the winner                              │
                             ▼                                         │
 ┌────────────── THALAMUS — GATE / ROUTE ──────────────────────────┐   │
 │  default-off; DISINHIBIT the selected channel to the motor;     │   │
 │  route + bind content⊗location across columns for voting ───────┼───┘
 └──────────────────────────────────────────────────────────────────┘
```

**The regions — role · substrate · reuse-source.** *How each region learns* is a design commitment (from the 2026-07-10
study, `notes/bg_thalamus_value_research.md`): **no region uses backprop / an ANN** — the substrate is the whole point.

| region | role | substrate (how it learns) | reuse-from (RULES #5) |
|---|---|---|---|
| sensory column | physical world-model + recognition | modified-HTM/SDR — LOCAL, self-supervised prediction error | `l4*`, `l5_displacement`+`operator`, `l23_object`, `encoders`, `retina` |
| PFC / task column | goal-state + plan hierarchy (same column, abstract frame) | same as the column | the SAME column on a task frame (`reference_hierarchy_substrate`) |
| **value / dopamine critic** | expected reward → the scalar RPE δ + tonic ρ | SDR-linear TD = the **successor-representation** read-off | `l6_sr.SuccessorFeatures` |
| basal ganglia | SELECT by value (disinhibition) — the Go/NoGo ACTOR | modified-HTM/SDR — LOCAL **dopamine-gated three-factor Hebbian**, D1/D2 opponent (OpAL) | `basal_ganglia` (`BasalGanglia`, `OpponentActor`) |
| thalamus | route + gate the selection (disinhibition) + bind content⊗location for voting | **DETERMINISTIC** — no learner inside; only optional slow gain | `thalamus` |
| hippocampus | ROLLOUT — prospective sweep over the model — + episodic binding | orchestration over the column's model (holds NO model of its own) | `hippocampus` |

**The value critic, precisely.** Value is not a separate network: it is a **read-off of L6's predictive map** (the
successor representation), `V = w·(M·R)` — routine value is one cheap dot product, rollout is the sparing fallback. The
critic emits **two distinct errors from one update** (Gardner/Gershman 2018): a *vector* SF-Bellman error that trains the SR
MAP — this runs even at **zero reward**, it is the epistemic / structure-learning signal the cortex owns — and a *scalar*
`δ = r + γV(s′) − V(s)`, the **dopamine** the basal ganglia consumes. A ceiling we have PROVEN
(`project_linear_value_cannot_hold_sokoban`): a linear value cannot represent relational V*; for those, `value` shrinks to
**scoring rollout leaves** and the plan comes from the hippocampal sweep, not the read-off.

**The contracts (who speaks what — the interface each region exposes).**
- **peripheral (retina/effectors):** `transduce(field) → [(feature, location)]` · `decode(motor_sdr) → command`
- **cortical column:** `observe(feature, location)` · `predict()` · `recognise() → (object, pose)` · `motor() → SDR` · `goal`
- **value critic:** `value(φ) → v` · `dopamine(φ, r, φ′) → δ` · `rho() → ρ`
- **basal ganglia:** `select(candidates) → winner` · `learn(δ)`
- **thalamus:** `bind(cols) → R` · `read(R, query)` · `gate(winner) → motor`
- **hippocampus:** `rollout(goal, cortex) → trajectories`
- **agent:** `step(obs) → action` — plumbing ONLY: `transduce → cortex.observe → «loop» → thalamus.gate → decode`; and `reward → critic → δ → bg.learn`.

The computational spec of the loop is EfficientZero-V2 (learned model = cortex; **value = the critic (SR/dopamine)**;
**policy/select = the basal ganglia**; search/rollout = hippocampal preplay) — the SAME loop from the algorithm side. We
take EZ-V2's *algorithm*, not its gradient-trained nets.

## 4. One decision, end to end
Perceive → sensory columns update the world-model; the PFC column holds/updates the goal-state → the hippocampus rolls out
candidate trajectories over the world-model toward the goal → the **value critic** scores them (the SR read-off, plus the
dopamine δ that trains the selector) → the **basal ganglia** SELECT one by disinhibition → the thalamus gates the selected
action to the motor output → `agent.py` enacts it in the environment → repeat.

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

## 8. The two primitives — ASSOCIATE and TRANSFORM (and why they are not one)

§7 says the column factors content, location, and state and reuses **one** model across locations. Implementing that reuse
needs **two different primitives**, and the load-bearing design decision here is to keep them **separate** — trying to make
one do both jobs is exactly the §7 failure. (This does *not* contradict §2's "one column algorithm" or `htm.py`'s "one
sequence-memory mechanism": those are about the ASSOCIATE machinery being uniform across layers and regions. TRANSFORM is
the *second* primitive that composes with it.)

- **ASSOCIATE — the `HTMLayer`.** Bind content to a context and recall it: "at *this* location/context, *that* feature."
  Learned by growing dendrite segments; per-instance, tabular. This is L4 (feature-at-location), L2/3 (object), and the
  sequence memories. **It does not generalize a transition across positions** — a fact learned at one place is a different
  segment from the same fact at another (§7's 0%-on-a-novel-place). That is not a defect to fix in the HTMLayer; it is the
  wrong tool for that job.
- **TRANSFORM — the operator.** Apply an action's effect as a **learned, group-structured permutation of a code**: "action
  A shifts the location code *this* way." It generalizes across every position **by construction** — a shift is the same
  operation everywhere, including places never visited. This is L6a (path integration) and L5 (displacement).

**They compose — that composition *is* the TBT forward model.** Predict the next feature from the next *movement*, not from
the previous feature (`htm.py`): the operator moves the location (TRANSFORM, generalizes), then the HTMLayer reads the
feature at the new location (ASSOCIATE, binds): `loc' = M(action)·loc ; feat' = recall_at(loc')`. That ordering is what
makes the model order-invariant.

**The operator, on our substrate — no matrix, no gradient, no ANN.** The location is a `GridEncoder` SDR: a bump per
`(scale, axis)` module, each module a ring of `scale` cells. `M(action)` is a **cyclic shift per module** (a
block-structured permutation; `GridEncoder.modules()` are the blocks), so the whole operator for one action is a vector of
per-module integer shifts. It is *discovered* by reading the per-module phase delta of an observed `(loc, action, loc')` and
voting — a genuine translation gives a constant shift across all positions. This is the entorhinal path-integration
mechanism (`φ ← φ + Δ mod scale`) with `Δ` **learned per action** instead of hard-coded (we do not get to assume
"ACTION1 = north" — that would be the bitter-lesson trap).

**One mechanism, generality dialed by the code it acts on.** A heading-dependent action ("FORWARD") shifts location by an
amount that *depends on* heading — not expressible as a constant per-module shift. So the operator instead acts on the
**conjunctive** code `location × heading` (`ConjunctiveEncoder`, `module_grids()`): a base-phase shift that is a *function
of* the heading phase (the non-abelian SE(2) case). Same mechanism — a permutation discovered by phase-delta voting — with
the abelian (heading-independent) case as its degenerate special case. The *kind* of an action (move vs turn) emerges from
which modules its learned shift touches; nothing per-action is coded.

**Two apparent tensions, both resolved by the composition:**
- *Operator (group representation) vs. discrete graph / SR.* The operator is the **regular, free kernel** — self-motion in
  empty space, true everywhere. **Irregularity (a wall, a push, a toggle) is not in the operator** — it is a *context-gated
  override*: the operator predicts the shift; a *local relational context* predicts the exception (blocked/pushed), and that
  override, being local, still generalizes across position (via the HTMLayer context) and warps the reachability graph the
  SR reads (a wall = reshaped reachability). One operator applies to every object incl. self; *whether* it applies is read
  from context — emergent, never coded.
- *Operator vs. the successor representation.* Same object, two timescales: `M(action)` is the **one-step transition
  generator**; the SR is `Σₖ γᵏ (policy-averaged M)ᵏ`, its discounted **resolvent** (grid cells diagonalize both). So the
  operator is **logically prior to the SR** — learn the one-step transform first, then accumulate it into the SR for value.
  (This is why `ROADMAP.md` Phase 3a builds the operator and Phase 3b's SR reads off it.)

**Where it lives.** A new `operator.py` (one file, one concept), owned by L6a (path integration) and L5 (the displacement
content it stores) per the §9 wiring table. It is **not** a parallel system — it is the missing owner of "the grid
path-integration operator" the column already names but never built; the `HTMLayer` stays the ASSOCIATE primitive, unchanged.

## 9. What makes a layer a layer — role = (context-in, target-out), both wired by hand

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
- Mountcastle 1978 — the uniform cortical column: one repeated circuit across all neocortex (§2, §9 premise).
- Douglas & Martin 2004, *Annu. Rev. Neurosci.* — the canonical cortical microcircuit; thalamic input is ~10% of synapses,
  so a region's function comes from its CONNECTIVITY, not a different circuit (§9, the context-in half).
- Molyneaux/Arlotta/Macklis (projection-neuron fate); Fezf2/Ctip2 → L5b subcerebral, Tbr1 → L6 corticothalamic, Satb2 →
  callosal — birth-order (inside-out) + a transcription-factor code fixes laminar/projection IDENTITY (§9, the target-out half).
- Thousand Brains Project 2024 (arXiv 2412.18354) — learning is local, associative, unsupervised; credit assignment is solved
  STRUCTURALLY by reference frames (error localized to a feature-at-location), not by backprop (§7).
- §3 loop grounding (dopamine = RPE — Schultz; SR value read-off — Dayan/Stachenfeld/Gershman; OpAL Go/NoGo — Collins & Frank;
  driver/modulator + core/matrix — Sherman & Guillery / Jones; generalized PE — Gardner/Gershman 2018): full cited review in
  `notes/bg_thalamus_value_research.md`; the region decomposition + build plan in `ROADMAP.md`.
