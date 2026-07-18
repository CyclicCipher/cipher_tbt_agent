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
- **TRANSFORM — the operator.** Apply an action's effect as a **learned group action on the location state**: "action A
  displaces me *this* way." It generalizes across every position **by construction** — the same displacement everywhere,
  including places never visited. This is L6a (path integration) and L5 (displacement).

**They compose — that composition *is* the TBT forward model.** Predict the next feature from the next *movement*, not from
the previous feature (`htm.py`): the operator moves the location (TRANSFORM, generalizes), then the HTMLayer reads the
feature at the new location (ASSOCIATE, binds): `loc' = M(action)·loc ; feat' = recall_at(loc')`. That ordering is what
makes the model order-invariant.

**The operator, on our substrate — no gradient, no ANN.** L6a's state is a **continuous pose** `(position, R)` — an n-vector
and an n×n **rotation matrix** (`column.py` `_pose`) — and the `GridEncoder` SDR is a **read-out** of it (`_code`), re-encoded
fresh at every fixation. The operator (`operator.MotionOperator`) learns, per action, the **body-frame displacement + body-frame
rotation**, as a running mean over observed `(pose, action, pose')` triples; `apply` maps that body-frame delta through the
*current* orientation: `p' = p + R·d`, `R' = R·ΔR`. `Δ` is **learned per action**, never hard-coded — we do not get to assume
"ACTION1 = north" (the bitter-lesson trap). This is the entorhinal mechanism `φ ← (φ + M·d) mod 1` with the phases
**continuous**, as Lewis 2019 states them.

**Why orientation is a matrix, not an angle** (cut-over 2026-07-15 — the same lesson as the continuous state, one level up). A
scalar `heading°` is an **SO(2)-only encoding**: SO(3) is 3-DOF and non-abelian, so no scalar can name it. Both primary
sources say matrix outright — Monty's pose is "location + orientation (**three orthonormal vectors**), continuous, never
discretized" (`reference_tbt_pose_invariant_recognition`), and Gao 2021 has motion as "a learned **group-representation
matrix** acting on the location code" (`reference_operator_as_group_representation`). So 2-D was the special case, and
**degrees are now a read-out** (`to_angle`/`from_angle`), exactly as the grid SDR is a read-out of the pose. One code path
serves n=2 (ARC frames) and n=3 (3-D environments); `wrap` is gone, because angle wrap-around was an artifact of the scalar.

**Why the state is continuous and the code is a read-out (the 2026-07-15 CUT-OVER — this replaced a discrete design).** The
operator used to be a **permutation** of the grid SDR (a cyclic shift per module). That is the natural move on a discrete
code, and it is a real constraint, not an implementation detail: *a permutation can only represent group elements that map the
code's lattice onto itself.* Translation is then exact only axis-aligned, and rotation only at the lattice's point group — the
**crystallographic restriction** (2/3/4/6-fold). Two measurements settled it:
- **The drift falsifier.** Quantised phases under an off-lattice rotation drift **linearly** with path length (overlap fell to
  0.50 by step 2; the decode was off by 13 cells at step 30). The drift is *systematic*, so multi-scale error correction — which
  only fixes *independent* errors — cannot remove it. Exact sub-cases (N=4) passed, which is the signature of a lattice
  constraint rather than a bug.
- **The bake-off.** Against a continuous-phase variant, exact off-grid rotation needed **N > 2π·radius** modules (confirmed
  exactly: r=1→8, r=2→16, r=4→32, r=8→64, r=16→128) — cost unbounded in object radius, and *quadratic* for SO(3). The
  continuous state matched at 1.00 with 0.00 error at every radius.

So: **continuous state, discrete read-out.** Translation and rotation are exact at any vector and any angle; the read-out's
quantisation is bounded and **never accumulates**, because each fixation re-encodes from the state rather than composing
rounded shifts. The binary SDR keeps the job it is right for — **identity**, where overlap means similarity — and stops doing
the job it is wrong for: carrying a **metric**. (`GridEncoder(orientations=)` and `RotationOperator` were deleted in the same
move, per the no-parallel-systems rule.)

**Non-abelian motion falls out for free.** An orientation-dependent action ("FORWARD") displaces by an amount that *depends
on* which way the body faces. SE(n) is the **semidirect product Rⁿ⋊SO(n)**: the translation transforms under a representation
that depends on the rotation (biologically, conjunctive grid × head-direction cells, Sargolini 2006). Because the operator
stores the delta in the **body frame** and `apply` maps it through the current orientation, non-commutativity is
**structural** — FORWARD;TURN ≠ TURN;FORWARD — with *no keying, no ring, and no discretisation*. One observation generalises
to every position **and every orientation**, including orientations never observed (tested at 37° after learning only at
0°/90°). This is strictly better than the keyed design it replaced, which needed a sample per `(action, heading)` and could
only key on headings it had seen; the deferred `ConjunctiveEncoder` *tensor* is now **moot**. In **3-D the rotations
themselves stop commuting** (yaw∘pitch ≠ pitch∘yaw) — a property SE(2) structurally cannot exhibit, and `R' = R·ΔR` delivers
it with no new code. The *kind* of action (move vs turn) emerges from which components of the learned delta are non-zero;
nothing per-action is coded.

**Pose-invariant recognition (SO(2) *and* SO(3), BUILT — the pose is SOLVED, not scanned).** Recognising a known object at a
novel pose is *inference*, not a stored table — and Monty's mechanism is stronger than a search: *"you recognize an unseen
orientation because you **SOLVE** for the rotation; you don't recall it"* (`reference_tbt_pose_invariant_recognition`). Monty
solves it by **aligning frames**, because its features carry a local frame (surface normal + curvature). Ours are
colour-at-location — **non-morphological**, no intrinsic orientation — so we build the frame from the inter-fixation
**displacement geometry** instead: an object at `(R, t)` puts model point ℓ at `rotate(R, ℓ) + t`, so differences cancel `t`,
giving `R·(ℓᵢ − ℓ₀) = pᵢ − p₀`. **n−1 independent displacements determine R** (`operator.solve_rotation`, the TRIAD method:
orthonormalize each side into a right-handed frame, `R = Uᵀ·V`), then `t = p₀ − rotate(R, ℓ₀)`. So 2-D needs 2 fixations and
3-D needs 3 — and there is **no angular resolution to sample, so SO(3) costs no more than SO(2)**, which is exactly what
design A bought. What is *hypothesised* is the correspondence (which model point each fixation touched); the pose is *derived*
and then *verified* by the model's own prediction.

Three things fall out of the group structure rather than being coded: a rotation is an **isometry**, so every pairwise
distance must be preserved (one rule subsuming the old distance *and* angle checks); **degenerate** fixations (coincident in
2-D, collinear in 3-D) leave R genuinely undetermined, so we return nothing rather than invent one; and `R = Uᵀ·V` is always a
**proper** rotation, so a **mirrored** object is refuted by the evidence — reflections are not in SO(n), and a chiral object's
mirror is a different object.

The pieces, each in its owning layer: the **L4→L6a associative link** (`Column._link`) is Lewis 2019's *"the sensory input
activates the union of locations"* — a sensed feature recalls the union of `(identity, location)` where it occurs, which seeds
the hypotheses; **L2/3** grades identity support (`ColumnPooler.support`); the **operator** is untouched. A hypothesis is
`(object, R, t)` = Monty's **(object ID, pose)**, continuous; evidence per fixation is the model's own prediction fit — *match
adds, mismatch subtracts*. This **subsumes** the retired scan (a shared anchor is just `t = 0`).

**The population is the answer** (`reference_population_code_belief`). Tied hypotheses mean the evidence does not separate
them, and that is information: a 4-fold object returns its whole **symmetry orbit**, because it genuinely has no single pose —
reporting one angle would be a lie. A single fixation returns *nothing*, because one point fixes no rotation.

**Learning is an EPISODE, not a fixation** (BUILT — this is the same evidence machinery, turned on the model itself).
Recognition is only ever as good as the model it reads, and L2/3 used to corrupt that model: it committed to an identity at
the *first* fixation of a learning sweep and persisted unconditionally, so two objects sharing a feature-at-location **merged**
into one chimeric identity (measured: two objects → one). The subtlety worth keeping: at fixation 1 those objects are
genuinely **indistinguishable**, so recognising the first is *correct inference* — the defect was the absence of **revision**,
and a per-fixation loop structurally cannot revise, because by the time a later fixation contradicts the choice the earlier
ones are already bound to the wrong identity. Refutation needs the object's **extent**, which only the whole sweep carries.

So learning is `start_object` → `sense_sweep` × n → **`commit`**: buffer the episode, ask `recognize` whether a known object
explains it, then reinforce that object or mint a new identity and bind the sweep to it. This is Monty's structure exactly
(Buffer → update an existing graph, or build a new one), and it leaves each layer one job: **`pool` INFERS and never mints;
`mint`/`bind` LEARN, at the episode boundary.** Scoring and binding are the *same* traversal (`Column._replay`), which is what
lets learning inherit pose-invariance: meeting a known object at a **novel pose reinforces it instead of duplicating it**, and
binds in the object's *own* frame so rotated coordinates never enter the model.

**The bar is "nothing refutes it", not a score.** A known object explains the sweep iff **no fixation contradicts** it: a
*partial* view is all match and no contradiction (so it reinforces rather than fragmenting), while one genuine contradiction
means a different object however much else agrees. This is threshold-free *and* it is the safe rule — `_replay(learn=True)`
binds every predicted fixation, so tolerating a contradiction would **bind** it, i.e. tolerance silently corrupts the model,
which is how the merge happened in the first place. (A `evidence ≥ ½·fixations` bar merged a **chiral pair** by a single hair:
3 matches − 1 refutation = 2 = the bar. Its tolerance was arbitrary — it depended on how many *other* fixations agreed.)
Tolerating k contradictions is a question for a sensor-**noise** model, deferred with noise itself. One more prune keeps
minting honest: an all-**burst** sweep is *unlearned*, not *new*, so minting waits for L4 to predict something (a burst code is
location-agnostic and would teach "feature → object", the feature-only trap).

**The sweep splits itself at an object boundary** (BUILT — no boundary cue from the caller). `reference_tbt_segmentation_and_grouping`
is unambiguous that this is the same signal: *"it relies on feature and morphology mismatch to implicitly detect boundaries"*,
and *"'two proto-things are really one' / 'one is really two' is just what the EVIDENCE concludes — not a re-run of a
segmenter."* So detection is free — R4's refutation *is* the boundary. The hard part is **interpretation**: a prefix match plus
a contradiction reads either as *(a)* a boundary (the sweep covered A and entered B) or *(b)* one different object that merely
shares a prefix with A, and both fit the same evidence. This is the segmentation problem itself, and the two readings are in
direct tension — the merge fix above *depends* on (b).

**The model breaks the tie: did the prefix EXHAUST the object?** You leave an object when you reach its **edge**. If the prefix
visited every location in A's model (the extent is already in `_link`), A ended there and a new object begins — reading (a);
if it covered only part of A, the sweep never reached an edge and (b) is the better parse. The remainder is then just a fresh
episode — the same `commit`, recursively — and a new object anchors its frame at **its own first fixation**, which is what
makes it recognisable alone later (Lewis 2019: a fresh grid phase *is* the object's origin). Measured: a known object followed
by a **novel** one is split, and the novel one becomes independently recognisable **without ever being marked**.

*The honest limit, and it is one thread with dynamics:* a **partial** sweep of A that wanders into a novel B is absorbed as one
blob (the prefix never exhausted A, so nothing licenses a split), and a wholly novel scene mints **one blob** — correctly, since
*"the object is a recognition construct"*: with no model there is no object. Splitting a blob needs a second signal, and the
same source names it — *"the best grouping cue we already have: COMMON FATE (what moves together)"*. That is the **dynamics**
slice (the operator over object poses), so cold-start segmentation and "any unsupported object falls" are the same problem.

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
  (This is why `ROADMAP.md` Phase 3a builds the operator and Phase 3c's SR reads off it.)

**Where it lives.** A new `operator.py` (one file, one concept), owned by L6a (path integration) and L5 (the displacement
content it stores) per the §10 wiring table. It is **not** a parallel system — it is the missing owner of "the grid
path-integration operator" the column already names but never built; the `HTMLayer` stays the ASSOCIATE primitive, unchanged.

**A note on L2/3 pooling — the ASSOCIATE primitive in a POOLING regime, not a third primitive.** L2/3 recognition — pooling
the L4 feature-at-location stream into a **stable object IDENTITY** that persists as the sensor moves and is re-pooled only
on a prediction error — is still associative (Hebbian feedforward L4→identity), *not* a new kind of computation like
TRANSFORM. What it adds is a **stable output decoupled from the instantaneous input**, which the `HTMLayer` cannot express
(its active cells ARE its proximal input); so L2/3 gets its own small engine (`pooler.ColumnPooler`), exactly as L6a gets
`operator.MotionOperator`. The two primitives above are unchanged. Crucially, **L2/3 holds the IDENTITY only** — the
object's structure (which feature at which location, the displacements between them) stays DISTRIBUTED across L4 (features),
L6a (locations), L5 (displacements); the object model is not one layer's data structure (`reference_tbt_layers_4_23`).
Cross-column VOTING over these identities is the THALAMUS's job (§3), a multi-column slice, not the single column's.

**Object-centric frame + the emergent boundary — ONE event does both.** Recognition must be TRANSLATION-invariant: an object
is the same wherever it sits. Per Lewis et al. 2019, allocating a fresh grid-frame origin for an object is the *same act* as
individuating it (grid frames have no origin — the arbitrary origin gives translation invariance, and the fresh location
space IS the object's uniqueness). On our substrate that unification is: a single **recognition-failure** event both
(a) re-anchors the L6a frame (`start_object` re-origins the grid → sensing becomes OBJECT-RELATIVE) and (b) starts a fresh
L2/3 identity. The trigger is emergent, not a symbolic segmenter: **L4's burst is the signal.** The pooler pools only the
PREDICTED (non-burst) L4 stream — a burst means "this feature-at-location is not yet learned," so during LEARNING it persists
(let L4 train) and at either phase a contradiction no known object explains IS the boundary. Verified: same features in a
swapped arrangement get DIFFERENT identities (the frame is load-bearing); a continuous two-object sweep is segmented with no
explicit reset; and at learning the sweep splits itself, so the caller marks nothing (`column.py` `start_object`/`perceive`/
`commit`; research in `notes/tbt_object_frame_and_bootstrap_research.md`).

**The ONLINE path solves its place too** (BUILT — this closed a measured asymmetry). `perceive` used to bind and recall at
whatever coordinate the caller supplied, so it only recognised an object entered **at its learned origin**: measured, the same
object shifted to (7,3) read `[-1,-1,-1]` and entered mid-object `[-1,-1]`, while `sense_sweep`+`recognize` solved the
identical presentation. That asymmetry was doing two kinds of damage — the caller's coordinate frame was a **silent contract**
(a test violated it and passed for weeks, because nothing checked; the online path just trusted the numbers), and there was no
online `(object, pose)` stream for **dynamics** to track.

Now the online path is Monty's **evidence-based LM**: a live population of `(object, pose)` hypotheses that each fixation
**narrows**, never recomputed (`reference_tbt_layers_4_23`: "recognition is INCREMENTAL evidence accumulation, NEVER
recomputed"). One scoring rule serves both paths (`_fit` — `_replay` loops it over a buffer, `_narrow` calls it per live
hypothesis), so there is no second mechanism. Seeding needs `dims` fixations (n−1 displacements determine a rotation), and
before that the only evidence is the **L4→L6a union** — Lewis 2019's "the sensory input activates the union of locations": a
feature only one object carries names it outright; a shared feature is honestly ambiguous. Same object, after: shifted →
`[0,0,0]`, entered mid-object → `[0,0]`.

Two things this **removed**, both of which had been quietly hiding the assumption: L2/3's support-only `pool()` (it could
answer "which object does this code support?" but never "…and where am I on it?", so it *required* the origin — the column's
population subsumes it, and `support` remains as the population's weight), and the boundary's **re-anchoring** (`perceive` no
longer teleports the frame on recognition failure; the next object's pose is solved). A hypothesis population also reports
ambiguity honestly: agreeing on the object while differing on the pose is a symmetry orbit → the object; disagreeing on the
object → nothing, rather than a forced guess.

## 9. Relations between objects — L5 displacement, and where physical law lives

§8's operator is about ONE frame: self-motion and object pose. A *relation* — "the block is resting on the table", "the key
is in the lock" — is between **two object frames**, and TBT is explicit that this is a different cell type with a
complementary job (`reference_tbt_layers_4_23`, Hawkins 2019 "Framework"):

- **Grid cells (L6a):** `location + movement → location`. BUILT (§8).
- **Displacement cells (L5PT, thick-tufted):** `location + location → the relation`. **The inverse. BUILT** (`Column.relate`
  → the relative pose of one object in another's frame, position- AND orientation-invariant BY CONSTRUCTION). The same cells
  carry compositional objects (a thing made of sub-things at relative displacements) and the motor output — Hawkins: they
  "alternately represent movements sent subcortically, and compositional objects sent up". *NB L5 = IT + PT (§10, §15 D1):
  this is the PT displacement role; the L5IT associative integrator that gates into it is deferred.*

So "resting on" is a displacement between two object frames, and object COMPOSITION and object RELATIONS are the same
machinery. **A relation is a displacement that stays STABLE as the pair moves** (`observe_relation`/`relation_of`): assumed
fixed from the first view, dissolved the moment the two move independently (`feedback_prefer_generalize_then_correct`).
Measured: the relative pose of a rigid pair is invariant across every position and shared orientation, confirms as a relation,
and breaks when one object moves alone. This extends common fate — "moves together → one thing" — to "fixed relative pose →
a relation", the substrate the support-override below reads.

**Physical law = the operator's free kernel + a context-gated override.** §8 already states the shape: the operator is the
**regular, free kernel**, true everywhere; irregularity is a *context-gated override* where a **local relational context**
predicts the exception. Read that with gravity in place of a wall and it is the same sentence: the free kernel is *everything
falls*; "supported" is the override. **A table stopping a fall and a wall stopping a push are ONE mechanism** — which is why
gravity is not a physics module, and never will be one here.

**How one demonstration generalises to every object — the answer is architectural, not statistical.** This is §7's lesson
again. TRANSFORM is position- and orientation-invariant *by construction*: an action's effect learned in a 5×5 region
dead-reckons exactly at (45,50), and a body-frame delta learned at 0°/90° holds at 37°. You do not generalise gravity by
watching a thousand falls; you generalise because the operator is defined **over a reference frame** and therefore applies
everywhere. The frame generalises, not the data. Two thirds of the machinery is already here: `recognize` returns
`(object, pose)`, and `MotionOperator.learn(key, before_pose, after_pose)` already has exactly the right signature.

**The operator over OBJECT poses — BUILT** (`Column.dynamics`, an L5 engine: L6a's operator path-integrates where the
*sensor* is; this one learns what an action does to a *thing out there*). Its input is the model's own output — the poses
`recognize`/`perceive` **solve** — so perception feeds dynamics end to end. Measured from **one** demonstration of a shove,
at one place, on one object: correct at positions never demonstrated, on the same object **rotated** to 90° and 217°, and on
a **different object never once seen to move**. That is the claim above, tested.

**INTRINSIC vs EXTRINSIC — the two ways the group acts, and an empirical fact about the action.** A *body's* motion is
intrinsic (`p' = p + R·d`): "FORWARD" means forward *from where I face*, which is what makes SE(n) non-commutative. An
*object's* motion is **extrinsic** (`p' = p + d`): a shoved block goes where it was shoved and a rock falls down, whatever
way they happen to be turned. This is not a style choice — measured on the identical demonstration, an intrinsic operator
sends a 90°-turned block to (20,23), **90° off** the shove it was shown, where extrinsic gives (23,20). Which one an action
obeys is itself discoverable (a self-propelled object is intrinsic); for now it is declared per operator, and discovering it
is the same open problem as the KEY below. *NB the extrinsic frame must be **allocentric** for a law like gravity to be
invariant. With a static observer the observation frame already is one; a moving observer needs the ego→allo transform —
which is the **hippocampus's** job, not a column's (`reference_tbt_frames_and_hippocampus`: HPC = one global allocentric
map, cortex = many local object-centric frames), and is deferred until the sensor moves.*

**COMMON FATE — what moves together is one thing (BUILT, and it closes R7's cold-start blob).** Everywhere else the boundary
is a prediction mismatch *against a model*, which is exactly why a wholly novel scene can only mint one blob: with no model
there is no object. **Motion needs no model** — a feature at `p` last look and `p'` now moved by `d`, and fixations sharing `d`
moved together (`Column._common_fate_groups`, gated by `look_again`). Measured on a scene swept as ONE episode, never told
there are two things: static → `[[0,1,2,3]]` (one group — correct), one part moves → `[[0,1],[2,3]]`, **two things found by
motion alone**; a scene moved *rigidly* stays one group, so the cue groups by *shared* motion, not mere change.

*Its refusals are load-bearing, and both were forced by measurement.* Correspondence is exact-feature-match, so it **refuses**
when a feature repeats in a look (a 4-fold object senses one feature at four places; unguarded it shattered a symmetric object
into its cells) or has no counterpart. And it must be told the scene is the same one (`look_again`): "the previous **episode**"
is not "the previous **look**" — objects are studied back-to-back, and auto-rolling the buffer invented motion between two
unrelated objects, fragmenting a chiral pair. The general fix for both is the one `_key` needs anyway: motion should **narrow
a population of correspondences**, not be read off a dict.

**The grouping now PERSISTS — the ART orienting reset** (`Column._commit_split`). `commit` checks common fate *before*
recognition, because motion refutes a single-object hypothesis that recognition would otherwise accept: if the scene split
into >1 motion group, no single object may claim more than one group (one object cannot be in two places), so each group is
committed on its own and a known identity is admitted **only if its whole model fits within the group** — else a fresh
identity is **recruited** (Grossberg's mismatch reset). This is what *creates the rival* the size principle then needs.
Measured: a blob learned from two things always seen together, then one part set in motion, **tears into its two parts** —
each becomes its own object, recognisable alone, and the library stays bounded (blob shell + two parts) under continuous
motion rather than exploding.

**The answer to the un-binding question is: DON'T — recruit, and let ART CHOICE retire the blob.** Four literatures converge:
ART **recruits** a new category on mismatch rather than eroding the old (the stability–plasticity answer); latent-cause
inference **creates a new state** on prediction error; Xu & Carey show **spatiotemporal** individuation precedes featural by
~10 months (motion is the primary cue); and the size principle says a small model seen whole beats a big one seen half. So
the blob is never erased — once its parts exist as rivals, **CHOICE** (`Column._choice` = `T_j = |I∧w_j|/(α+|w_j|)`,
normalised by the CATEGORY) keeps them winning and the blob simply stops being chosen, dying of disuse. `ColumnPooler.perm_dec`
stays dead code, correctly: L2/3 needs no LTD.

*One subtlety the build exposed:* the ART **choice** and **vigilance** functions must be counted at the **feature-at-location**
granularity (matched fixations over model size — `Column._replay`'s `matched` count, which uses L2/3 `support`), not at raw
L4-cell level. Minting binds *burst* cells (~M per column) while recognition fires the *predicted* ~1, so an L4-cell choice
made a torn-off piece overlap only `1/M` of its own inflated receptive field and **tie its parent blob** — the pooler's
`support` is the burst-independent level, and choice/vigilance are counted from it. (The two ART normalisations, restated at
this level: **CHOICE** = matched/(α+|model|), the size principle that ranks rivals; **VIGILANCE** = the `refuted_at` bar at
ρ=1, "nothing may go unexplained", whose ρ<1 relaxation is the deferred sensor-noise knob.) The tell that we had the level
wrong: we independently re-derived choice by hand, badly, over three iterations — the cure for hand-coded dynamics is to *use*
the settled mechanism (`RULES` #5), not re-derive it.

*Honest scope:* the tear is demonstrated under **continuous** motion (each look adds a fresh displacement). Re-committing a
*fused, now-static* multi-object scene as one episode is not something a correct agent loop does, and is not handled — once
motion has individuated the parts, perception should track them separately, which is the game-loop's job. And the mint-time
label (a stable counter) makes a retired blob a harmless dead shell, not a label-shifting hazard.

**Mechanism check before building the override (2026-07-16, `reference_tbt_object_behaviors`).** The TBP *Object Behaviors*
doc reframes and confirms the plan: a behaviour is a **separate reference frame** from static morphology, **object-independent**
so it *transfers* across objects (direct confirmation of "one demo → every object" — their independent frame is our
geometry-keyed generalisation), and **state-conditioned** — `predict(action, STATE) → effect`, where STATE is temporal,
morphological (open/closed), or relational. So the override is the *null-state-default* special case of state-conditioning, and
"everything falls" being the default (vs learning both states) is our deliberate aggressive prior
(`feedback_prefer_generalize_then_correct`). Two consequences: **(i)** condition on a *general STATE SDR*, with relations as one
supplier, not an HTMLayer hard-wired to relations; **(ii)** TBP puts the relational/compositional context at a **HIGHER region**
(a behaviour bound as a feature of a *scene* object), so the faithful override is **multi-column** — a compositional column over
the sensory one, not a gate on the single column (see §5.1). Prediction-error-driven behaviour learning is TBP's stated *open
territory*; our burst-binds-the-present-state discovery (core HTM) fills it — grounded, but the part most worth validating.

**The context-gated OVERRIDE — gravity + support, BUILT, and MULTI-COLUMN.** The operator is the free kernel (everything falls),
and a recognised STATE predicts the exception (supported ⇒ stays). It lives in a **second, compositional column** (`Agent._scene`),
whose features-at-locations are recognised `(object-id, pose)` routed up from the sensory column by the **thalamus**
(`Thalamus.project` — the content⊗location binding deferred until "a multi-object scene"; this is that task). The dynamics
effect is keyed on `(action, STATE)` where STATE is the object's relational geometry (`state_of`, geometry- not identity-keyed);
`state=∅` is the free-kernel default, a supported object is a non-null state with its own keyed effect, and an unlearned state
**falls back** to the free kernel (specific-overrides-default, the HTM high-order-over-first-order structure — no `if supported`).
Measured end to end: a supported object stays, a free one falls, a **new object never demonstrated** stays when supported
(TBP's object-independent behaviour frame — the state is geometry-keyed, so the behaviour transfers), and **removing the support
returns the state to null so it falls again** (assume, then correct). This is the first compositional slice: "one column, used
thrice" (sensory ⊕ task ⊕ compositional), honouring §5.1, and it finally exercises the thalamus's binding role. *A wall stopping
a push is the same mechanism with a partial-effect state instead of a null one — a follow-up, not a new idea.*

**What is still missing, in dependency order:**
1. **The operator's KEY, discovered rather than given.** The override *learns* which state gates which effect, but it does not
   yet DISCOVER *which* relational feature is the true condition versus a spurious correlate — if every supported block seen
   was also blue, geometry-keyed state can't yet tell "supported" from "blue" without CONTRAST (a blue unsupported thing that
   falls). Prediction-error-driven disambiguation is that slice (`feedback_subgoal_types_from_dynamics`,
   `reference_l5_operator_kinds`); "every object falls alike" is a hypothesis the world refutes (feathers), and today the state
   also generalises only across the SAME quantised geometry (`_quantise`), not yet across geometry variants.

## 10. What makes a layer a layer — role = (context-in, target-out), both wired by hand

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
- Mountcastle 1978 — the uniform cortical column: one repeated circuit across all neocortex (§2, §10 premise).
- Douglas & Martin 2004, *Annu. Rev. Neurosci.* — the canonical cortical microcircuit; thalamic input is ~10% of synapses,
  so a region's function comes from its CONNECTIVITY, not a different circuit (§10, the context-in half).
- Molyneaux/Arlotta/Macklis (projection-neuron fate); Fezf2/Ctip2 → L5b subcerebral, Tbr1 → L6 corticothalamic, Satb2 →
  callosal — birth-order (inside-out) + a transcription-factor code fixes laminar/projection IDENTITY (§10, the target-out half).
- Thousand Brains Project 2024 (arXiv 2412.18354) — learning is local, associative, unsupervised; credit assignment is solved
  STRUCTURALLY by reference frames (error localized to a feature-at-location), not by backprop (§7).
- §3 loop grounding (dopamine = RPE — Schultz; SR value read-off — Dayan/Stachenfeld/Gershman; OpAL Go/NoGo — Collins & Frank;
  driver/modulator + core/matrix — Sherman & Guillery / Jones; generalized PE — Gardner/Gershman 2018): full cited review in
  `notes/bg_thalamus_value_research.md`; the region decomposition + build plan in `ROADMAP.md`.
