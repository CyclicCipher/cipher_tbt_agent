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
- **H1 — the allocentric object map.** A world-anchored map: objects placed at WORLD coords by recognised identity +
  the agent's place. For a fixed board the transform is near-identity (image≈world) + track the agent; for an
  ego-centred/scrolling view, apply the efference transform. The map is the new substrate the state reads from.
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
