# HIPPOCAMPUS — the allocentric cognitive map (the world-anchored frame)

*Design + build plan, 2026-06-29. Forced by the live-test blocker #1: the egocentric sensor cannot separate
self-motion from world-motion. Companion to `TARGET_ARCHITECTURE.md`; grounds `reference_tbt_frames_and_hippocampus`
("the moment a hippocampus becomes required") — we are at that moment.*

## 0. Why (the blocker)
The refactored agent is **egocentric** (a window around the fovea + the controllable object's path-integrated
position). In an egocentric frame **self-motion and world-motion are indistinguishable in the raw signal**: when
the agent moves by Δ, every static thing shifts by −Δ (optic flow). The live cn04 run showed exactly this —
**90–171 cells change per action**, a jumping centroid, noisy per-action deltas — the model "thinks the world is
moving around it." It has the efference (it knows it *acted*) but **never cancels the self-induced flow**, so the
action's consequence looks like the world rearranging. This also starves L5 (no clean displacement → blocker #2)
and the state (no stable world → no navigation).

## 1. How the brain solves it (researched — `reference_hippocampus`)
A three-stage pipeline:
1. **Estimate self-motion WITHOUT the visual scene** — the **efference copy** (motor-command copies "used to cancel
   sensory input arising from observer movement") + **vestibular** (head motion). Predict the *global* flow.
2. **Flow parsing** (Rushton & Warren) — globally **subtract** the predicted self-motion flow from the optic flow;
   the **residual is real object-motion**. Combined with vestibular/efference in **MSTd / VIP**.
3. **Egocentric→allocentric transform via GAIN FIELDS** — `image_position × self_position` (retinal × eye/head/body)
   = the coordinate transform; **retrosplenial cortex** is the translator (parietal egocentric → entorhinal/
   hippocampal allocentric). The destination is the **hippocampal–entorhinal cognitive map**: **grid cells** (the
   metric ruler, path-integrated), **place cells** (allocentric locations), **head-direction cells**, **boundary
   cells** (anchor to walls). The map is **anchored to landmarks/boundaries** (they set the axes' phase/spacing/
   orientation); re-seeing a landmark **corrects path-integration drift** (loop closure).

## 2. The key insight: the flow problem is partly SELF-INFLICTED — and the map resolves the 7b tension
The ARC frame is a **FIXED top-down board** — already world-anchored. Our **egocentric (fovea-centered) sensor**
(7b, adopted for state RECURRENCE) is what *introduced* the global-flow problem (everything shifts relative to the
moving fovea). The allocentric cognitive map dissolves the 7b tension (recurrence vs stability): an **OBJECT-LEVEL
allocentric map** — objects placed at WORLD positions by recognised IDENTITY — is **stable** (objects don't move
when the agent does) AND **recurs** (the configuration + the agent's place recur). So the map gives both, where the
raw board (stable, doesn't recur) and the raw egocentric patch (recurs, not stable) each gave only one.

## 3. Reuse (cortical columns are evolved-from-hippocampal — most pieces exist)
- **Grid metric** = `l6_grid.py`, the shelved multi-scale hex grid = **grid cells**. UN-SHELVE it as the spatial
  world metric (the SR `l6_sr` stays the *relational/topological* frame; the grid is the *spatial* one).
- **Self-motion estimate (efference)** = `sensor._delta` (per-action displacement) — but estimated as the **global**
  coherent shift, not a small fovea blob.
- **Landmarks / loop closure** = `l23_object` recognition (we built it) → identify a stable object, anchor/reset.
- **Path integration** = `column.loc_*` (predict-by-edge + snap) → the agent's place, updated by the efference.
- **Place codes** = the SR rows / grid codes. The cognitive map is these pieces *assembled* + the transform.

## 3b. How the hippocampus talks to the columns (entorhinal-in, thalamus-out)
The brain uses **two** routes, and they split cleanly onto what we have:
- **IN = the entorhinal gateway (cortico-cortical, NOT thalamic).** Cortex → entorhinal superficial → hippocampus →
  entorhinal deep → cortex is the *binding* loop. The EC is where grid cells live, and TBT says the cortex
  *replicated* the EC-grid into every column's L6 — so the columns are evolved-from-entorhinal; the hippocampus
  proper is the one structure that binds *across* columns into a single world. **In our terms:** the hippocampus
  module reads each column's **L2/3 recognition output** (which object) + its egocentric position + the **efference**
  (`sensor._delta`) DIRECTLY — no thalamus on the way in. The column's own L6 grid stays its *local* object-centric
  frame; the hippocampus is the *global* one.
- **OUT = the thalamic context route (this is the part that DOES go through the thalamus).** Hippocampal context and
  head-direction reach the cortex via the **nucleus reuniens** (HC↔mPFC) and the **anterior thalamic nuclei** (Papez:
  HC→mammillary→anterior thalamus→retrosplenial, head-direction). **In our terms:** the hippocampus broadcasts the
  allocentric frame back as a **top-down prior through `thalamus.py`** — and the channel already exists:
  **`read_location`** ("top-down goal-state SET... the task column setting a goal-state in the spatial column"). The
  hippocampus's "where am I / where is object X in the world" is the same shape — a location bound to content,
  broadcast to whichever column needs it. We REUSE the location channel; we do not invent a new fabric.

This is the same top-down-prior mechanism as the deferred **heterarchy** ("a higher level's goal sets the prior of
the level below") — the thalamus is the ONE routing fabric for both goal-states and allocentric context. **For cn04
(H1, a transformation game)** there is no movement and ~one column's worth of scene, so the thalamic *broadcast* is
barely exercised yet; the immediate work is the **entorhinal-gateway side** (build the object-level allocentric scene
from the columns' recognition). The thalamic broadcast earns its keep at multi-column / movement games (the
head-direction/position prior).

## 4. Build plan (each stage suite-green; offline reproductions)
- **H0 — frame-check. ✅ DONE 2026-06-29 — and it changes the plan.** Result: the best global shift that explains
  each movement-action transition is **(0,0)** (identity matches as well as any shift) → **cn04 is a
  TRANSFORMATION / state-change game** (cells change IN PLACE: `4`-bar, `0`-boxes, `e`-boxes on bg `a`), **not a
  movement game.** (The memory's "cn04 ACTION1→(0,−3)" was wrong.) **CONSEQUENCE:** the hippocampus's *flow
  cancellation* (H2 — separating self-motion from world-motion) **does NOT apply to cn04** (there is no
  self-motion). The frame-check prevented building the wrong thing. **BUT** the allocentric *object map* (H1) is
  still the right substrate — for cn04 it provides the **stable, recurring, object-level scene** on which the L5
  **`recolor`** operator (KIND-general, already built — in-place content change) models the transformation. The
  egocentric patch sensor is *mismatched* to cn04: it foveates in-place changes as if they were movement → the noisy
  deltas. So: **H1 first (the substrate, general); flow-cancellation (H2) DEFERRED to a real movement game; cn04's
  path = H1 + the `recolor` operator.** Open: most public games may be transformation games (ls20 too) — check the
  KIND before assuming movement.
- **H1 — the allocentric object map. ✅ BUILT 2026-06-30 (`hippocampus.py`, 5 tests green).** `Hippocampus.scene(frame)`
  binds proto-objects to allocentric world SLOTS (corner-anchored = growth-stable place cells) → a FACTORED
  `(slot, colour, size)` state. PROVEN (synthetic): it recurs where the per-pixel `content_sig` churns (a reshaping
  object of the same colour+size is ONE state) and a real transformation (toggle/growth) moves it. **HONEST cn04
  measurement — it does NOT crack cn04 (recurrence 0.08, worse than the egocentric patch's 0.35), and the reason is
  diagnostic:** cn04's recurrence-killer is the *spatial reshaping* of the tree's branches, and a FAITHFUL allocentric
  map correctly *preserves* that (so it can't recur). The only cn04 encoding that recurred — the position-free
  `(colour,size)` multiset (33 states, 0.57) — does so only by THROWING AWAY the reshaping that is cn04's whole
  mechanic, so it's too lossy to plan with. **Conclusion: no tabular state both recurs AND preserves cn04's dynamics
  → cn04 (and ls20) need a GENERATIVE FORWARD MODEL (predict the transformation, plan by rollout), not the tabular
  graph. H1 is the correct SUBSTRATE that model predicts in (per-object features), and the right representation for
  movement/multi-object scenes — but it is not, by itself, the unlock for structured-dynamics games.** Next: the
  generative dynamics on top of H1 (see the new strand below).
- **H2 — self-motion cancellation (flow parsing).** Estimate the per-action GLOBAL self-displacement (efference);
  subtract it; the residual is world-motion. Distinguishes "I moved" from "the world changed".
- **H3 — landmark anchoring (loop closure).** Re-seeing a recognised object resets the drifting agent place
  (recognition → correct), so the map stays world-anchored over a long run.
- **H4 — the allocentric STATE.** The map → the column's state (the agent's place + the object configuration),
  recurring AND stable → L5 sees clean displacements (fixes blocker #2), navigation works.

## 5. Honest risks / open
- **Single organ vs per-column.** TBT says each column replicates the grid machinery; the GLOBAL allocentric map
  binding all columns is the hippocampal organ. For our single/few-column agent the column's L6 grid + the
  transform IS the map. Cross-column allocentric binding (many frames → one world) is the deferred hard part.
- **cn04 might be a transformation game** (H0 decides). If so, the hippocampus is the right GENERAL capability but
  not cn04's specific fix — and we learn that before building the wrong thing.
- Don't reintroduce the per-pixel state that didn't recur (7b): the allocentric map is OBJECT-LEVEL (by identity).
