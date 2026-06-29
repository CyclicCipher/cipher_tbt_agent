# TARGET ARCHITECTURE — the cortical column as the engine of intelligence

*Rewritten from scratch 2026-06-28, after the reset to **columns + API only**. Supersedes the previous doc. The
agent is many copies of ONE column algorithm talking over a thin fabric; there is almost no harness.*

## Thesis

Intelligence is **many instances of one cortical-column algorithm**, each learning a reference frame and predicting
within it, communicating over a thin fabric. The column is the engine: **perception, recognition, the world-model,
dynamics, and goal-proposal are all column faculties.** Only four things are NOT the column:

1. **Sensorimotor organs** — sensors (raw frame → *features at a location*, via receptive fields) and motors (ACTION1–6).
2. **Thalamus** — the inter-column communication fabric (routing, binding; voting deferred) + the goal-state channel.
3. **Basal ganglia** — the action gate (Go/NoGo, dopamine-RPE; emergent allocation).
4. **Reward** — the pragmatic value signal from the score.

Everything else flows through **columns + their communication**.

We borrow from **Monty** (the Thousand Brains Project implementation) the faculties we lack — chiefly evidence-based
recognition with *inferred pose*. We are **not** a Monty clone: our reference frame is the **SR-eigenframe**
(topology-general, not 3-D Euclidean), we have **value** (Monty is a pure recogniser, and so cannot do a goal task
like ARC), and we have a **conditional-dynamics** faculty. The aim is a *more* biologically faithful column than
Monty's, keeping our advantages and adding its recognition/learning machinery.

## The loop (active inference — reuse from prior commits)

> sense → each column **recognises** (which object, what pose) + **path-integrates** its belief + **predicts dynamics**
> → each column's **GSG** proposes goal-states (epistemic value: resolve uncertainty) alongside **reward** (pragmatic
> value) → **basal ganglia** gate → **motor** acts. One continuous interaction; the model persists across levels.

Expected free energy = epistemic + pragmatic; exploration and exploitation are ONE value. The achiever / EFE
arbitration / goal-directed + novelty-directed planning we built before is reusable (see git history) and drives this
loop — it is not new work, it is re-grounding what we had on the columns.

**Predictive-state structure (HTM-style — the learning signal).** The loop is *predict-then-compare*: at the END of a
turn each column enters a PREDICTIVE STATE (it predicts the next state given the chosen action and the active goal); at
the START of the next turn the actual sensation is compared to that prediction. **The mismatch is the single learning
signal** — it sharpens the operators (a graph edge + its reliability, the SR, the recognition evidence) and flags
surprise (an unpredicted change = a boundary; HTM's burst). This unifies the loop: a GOAL is a particular prediction (a
desired future state), motor actions are chosen to *make the prediction come true* (active inference = acting to fulfil
predictions), and motor learning is just the prediction error on an action's outcome — short-term (was this move's
effect as expected) and long-term (the operator sharpens). No separate module: the column already predicts (`predict` /
recognition / SR); the loop holds the prediction and learns from the comparison. This subsumes the old `events.py`
reafference + `forward.prediction_error`.

## The column's faculties — HAVE / FOLD / BUILD

The column is "learn a structural map, predict from it," through four layers. Each faculty is **HAVE** (works today),
**FOLD** (a current standalone script's *function* moves into a layer), or **BUILD** (Monty has it, we don't yet).

- **L6 — reference frame (HAVE offline; online learning is step 2; absorbs `factorize`).** The SR frame of the
  transition graph (Stachenfeld 2017: grid cells *are* the SR eigenvectors) — topology-general (grid-like on metric
  space, correct on rings/trees/abstract spaces). Today via a batch `eigh`; step 2 makes it **online (TD-SR + an
  optional Hebbian grid layer)** — see "Incremental / online model building". **Factorization lives here:** independent factors are separable eigen-subspaces of the
  *action* graph, and where they don't separate, a second column is allocated (basal ganglia). So `factorize.py`'s job
  becomes a property of L6 + multi-column allocation, not a side script. *(Research check: eigen-subspaces ≈ disentangled
  factors because the frame is built from ACTIONS, not statics — validate, don't assume.)*
- **L5 — the displacement / MOTOR-OUTPUT / thalamus-driver layer (a column ENGINE, not a dead operator).** L5 holds the
  per-action **displacement** as a first-class, position-invariant object — and that ONE object has four uses: the
  *generalizing base operator* ("action a shifts by Δ", so it predicts a's effect at an UNVISITED state), the **motor
  command** (cortex's main output → the enacted action — L5's PT/ET cells drive subcortical motor centres), the
  **efference copy** (the predicted effect → the predictive state), and the **feed-forward DRIVER of the higher-order
  thalamus** (the inter-column message; Sherman & Guillery — L5 is the trans-thalamic driver, L6 is first-order
  modulatory feedback). `residual`'s conditional/structured part is the per-(s,a) **exceptions** to the base
  displacement (the wall/door), held by L5's edges. **Status:** the per-action OPERATOR is now SEATED IN L5
  (`L5.observe`/`predict`/`successors`; the column delegates, `predict` = the efference copy) — the *structural* reseat,
  done (so 3b and the dynamics column build ON L5, not a bare dict). Remaining (the **L5 reseat (finish)**, build-order
  step 5, with the sensor's poses): the position-invariant DISPLACEMENT (generalize to unvisited (s,a)) + the literal
  motor / thalamus feed-forward output. **NOT superseded** — see `reference_layer5_role`. (Do NOT retire L5; nearly
  doing so on 2026-06-28 was the error this records.)
- **L4 / L23 — content + memory (HAVE).** Bind entity ⊗ location (L4); pool in the one shared object memory (L23); read
  back. This is Monty's "features at locations."
- **Belief + recurrence/memory (HAVE function; absorbs `recurrence`).** The column carries a state across the
  sensorimotor sequence: **PREDICT** by the L5 operator (efference copy — survives partial observability), **CORRECT**
  toward what is sensed. We DO need this recurrence/memory. The current `recurrence.py` (a learned per-channel Mamba/SSD
  gate) is a leftover from the language SSM; its *function* — the predict↔correct gain, i.e. precision-weighting /
  Kalman gain — folds into the column's belief+evidence update. In the new design this recurrence becomes Monty-style
  **evidence accumulation** (a running evidence sum; the gain starts fixed and may later be a learned precision). The
  standalone SSM-gate does not earn its keep as a top-level concept.

### BUILD — Monty has, we lack (the spine)
- **Evidence-based recognition with inferred pose (BUILD — the priority).** Carry MANY `(object, pose)` hypotheses with
  graded evidence; each step, project every hypothesis by the (rotated) displacement and score the match at its new
  location; **infer pose** from sensed features; pick the most-likely hypothesis with a relative confidence/termination
  threshold. This is the core LM loop and the home of the deleted `recognize.py`. Single-hypothesis path integration
  (HAVE) is its degenerate case.
- **Incremental / online model building (BUILD).** Today's `consolidate()` runs a full `eigh` of the transition graph
  (O(n³), recomputed from scratch) — a batch shortcut, too slow per step on 64×64. **Eigendecomposition is the wrong tool
  online; replace it, in this order:**
  1. **Factor the state first.** The `eigh` is only costly because n = raw cells; over a few *factored* coordinates the
     graph is tiny and the algorithm choice nearly stops mattering. This is the `factorize → L6` job doing double duty,
     and it is largely the sensor's responsibility (hand the column a small factored state, not 4096 pixels).
  2. **Online TD successor representation.** Learn the SR by temporal difference — `M(s,·) ← M(s,·) + α[e_s + γM(s′,·) −
     M(s,·)]`, O(visited states)/step, no `eigh`; its ROWS are place-cell-like codes that already encode topology
     (Dayan 1993; STDP/theta-sweeps do exactly this in cortex — no organism runs a batch eigendecomposition). Add a node
     when the sensed location is novel.
  3. **Hebbian grid layer, only if needed.** Extract the top-k eigenvectors of the streaming SR by Oja/Sanger's GHA
     (O(nk)/step) — the online, biologically-faithful route to the grid frame (grid cells emerge from the place-cell SR
     by NON-NEGATIVE PCA — Dordek et al. 2016), for the grid's benefits (multi-scale, vector-navigation to UNVISITED
     goals). The batch `eigh` is kept only as an offline reference / occasional partial-eigh fallback.
- **Goal State Generator (BUILD).** From its own model, propose the next pose to visit that best disambiguates the
  leading hypotheses (epistemic value), emitted on the thalamic goal-state channel, arbitrated with reward (pragmatic)
  and gated by the basal ganglia.

## Dorsal / ventral — one algorithm, two specialisations (this saves us a whole behaviour model)

Same column, same algorithm, **specialised only by its input stream** (Mountcastle's canonical microcircuit; the
function of a cortical region is set by its connections, not a different algorithm). Parvocellular-like input (sustained,
static features) → ventral **"what" / morphology**; magnocellular-like input (transient, **change**) → dorsal
**"where/how" / dynamics**. Monty's object-behaviours doc states it directly: *"if parvocellular cells are input to L4,
the column learns the static morphology; if magnocellular, the dynamic behavior."*

So we do **not** write a behaviour/dynamics model. A "dynamics column" is the **same `CorticalColumn` fed the change
stream** (our reafference residual / salient cells) instead of the static-feature stream. One algorithm, two instances,
differing only by input — the pose⊕content thing we built-and-lost collapses to "point a column at the change stream."

## Attention is not a module

Attention is the **motor / path-integration system choosing where to sample**: OVERT = a saccade (move the sensor);
COVERT/mental = path-integrate the belief over the reference frame **without** moving the body (a withheld movement —
the premotor theory of attention). So attention = **GSG** (proposes the locus) + **path integration** (moves the locus)
+ **basal ganglia** (gates overt vs covert). No attention script. Salience still influences it, as the bottom-up
estimate of expected information gain (the same epistemic currency the GSG spends).

## Inter-column messaging — deferred (heterogeneous frames)

Monty's CMP voting works *because every LM shares ONE Euclidean frame* — a vote ("sense it *here*, by our relative
displacement") transforms by a common displacement. **Our columns each learn a DIFFERENT SR-eigenframe**, so voting
needs **learned cross-frame registration** (Hebbian / co-occurrence binding in the shared `d_mem` space). Feed-forward +
thalamic VSA binding work now; pose-aware *voting* across heterogeneous frames is a genuinely harder, deferred problem.

## What we keep that Monty lacks
- The **SR-eigenframe** — topology-general, so the same machinery navigates abstract / relational / conceptual spaces
  (the reasoning substrate; attention and "mental saccades" over concept space are the same operation).
- **Value** (reward + basal ganglia) → goal-seeking. Monty is a recogniser; this is why we can do ARC and it can't.
- The **conditional-dynamics** faculty (predict the world's responses, not just recognise form).
- **Path integration** as a built-in belief.

## Script dispositions (the cleanup this implies)
- `column_learner.py` — **DELETE** (orphaned demo driver; its drive-and-learn function is the agent loop, not a script).
- `factorize.py` — dissolve into **L6** (eigen-subspace factors + BG allocation); remove once folded.
- `residual.py` — dissolve into **L5** (state-dependent operators); remove once folded.
- `recurrence.py` — **SUPERSEDED online**: path integration is now discrete graph tracking (predict-by-edge + snap to a
  sighting), not a gated vector belief, so the column no longer imports it. Removable (kept only as reference).
- **Keep:** `column.py`, `l4/l5/l6/l23`, `thalamus.py`, `basal_ganglia.py`, `reward.py`, and the API
  (`arc_run.py`, `arc_sdk.py`, `tasks/core.py`).

## Build order (deadline-aware)
1. **Evidence-based recognition with inferred pose**, in the column — the spine; subsumes `recognize.py`.
2. **Incremental / online learning** in L6 — **DONE**. `OnlineSR` (TD, no `eigh`) carries value/topology; the column's
   `predict` / `loc_*` run over the exact learned transition **graph** (state-dependent by construction — it subsumes
   the L5 matrix-operator-AS-PREDICTOR and `residual`'s conditional structure; L5's role is NOT subsumed — the per-action
   operator is now SEATED IN L5 (`L5.observe`/`predict`; structural reseat done), and the displacement / motor / thalamus
   part finishes with the sensor, step 5); recognition carries continuous pose. Decided after neuroscience
   (reference_brain_reference_frames_orthogonalization): the brain orthogonalises by sparse pattern separation (not
   eigh) and path-integrates by a continuous-attractor bump / discrete snapping (not a matrix op over codes), so the
   matrix operator + the recurrence are superseded online. *ARCHIVED alternatives to the chosen graph+SR (kept on the
   shelf if it proves insufficient): (B) a **Hebbian grid layer** — Oja / Sanger's GHA / non-negative PCA on the
   streaming SR → orthonormal SR-eigen ("grid") codes that drive a matrix operator + continuous operator composition +
   vector-navigation to UNVISITED goals; (C) **`_sparsify_topk`** (random projection + top-k = DG / fly-mushroom-body
   pattern separation) → near-orthogonal sparse codes for an associative operator. Revisit only if the discrete graph
   can't generalise to unseen state-actions and the SR's reachability isn't enough.*
3. **GSG + reward + basal ganglia → the active-inference loop** (re-ground the prior-commit planner on the column).
4. **Dorsal/ventral dynamics column** (the change stream) — cheap once 1–3 exist.
5. **L5 reseat (finish)** — the per-action OPERATOR is already SEATED IN L5 (`L5.observe`/`predict`; structural reseat
   done before 3b, so 3b + the dynamics column build on L5). Finish it here, with poses: the position-invariant
   DISPLACEMENT (generalize to UNVISITED state-actions), the agent's action as L5's motor output, and the thalamus
   feed-forward driver = L5's output. Lands right before the sensor (the displacement needs poses). L5 is a column engine.
6. **The sensor (retina)** → column input; run the continuous loop on a real game.
7. Later: **cross-frame voting**; **compositional hierarchy** of columns.

## Honest risks
- The evidence loop + online learning over the SR-frame **within the action budget** is unproven — the make-or-break.
- "Eigen-subspaces = disentangled factors" and "fixed gain vs learned precision" are **research questions to validate**,
  not settled facts.
- Heterogeneous-frame voting is genuinely hard; deferring it limits multi-column consensus for now.
- We have **not completed a real ARC level** — that remains the milestone everything is judged against.
