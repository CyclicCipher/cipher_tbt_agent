# REWORK_PLAN — one online EZ-V2 loop on the column; drastically less code

**Status: DRAFT 2026-06-26.** Supersedes REORG_PLAN Part B3/B4 (the emergent-subgoal work) — EZ-V2 removes
explicit subgoals entirely, so that work folds into this. REORG Part A (the `src/` merge) and B1/B2 (the thin
agent + the `Environment` contract) STAND and feed directly into this. Detailed paper notes: `src/tbt/EZV2_NOTES.md`.

**ACCEPTANCE TEST (the completion bar, set by Cipher 2026-06-26):** the rework is DONE only when `tbt/` — the MODEL
ARCHITECTURE — contains **ZERO domain/environment-specific code** (no grids/colours/moves/doors/pads/keys/subgoal
types; the move geometry, dynamics, and goal are all LEARNED, or live in `perception/`+`tasks/`). The bitter lesson
made into a pass/fail check. See **Success criteria**.

## The thesis
EfficientZero-V2 is a **model-based, search-driven, online** learner: four nets (representation `H`, dynamics
`G`, policy `P`, value `V`), act by a tiny tree search, learn from the search's own targets. **Our column already
IS H+G** (SR-frame latent + L5 operators + the recurrence); **`reward.py` is V**; **the basal ganglia is the
policy prior `P`** that the search refines. So the rework is not new machinery — it is *deleting* machinery and
wiring what's left into one loop:

- The **offline `collect` / online `agent` split dies** → one online act-by-search-and-learn loop (EZ-V2's
  exploration only works online).
- **Explicit subgoals die** (no `fire/cover/goal` enumeration, no abstract subgoal-MDP, no `scaling_probe` 2^K
  wall). The **value over the SR-frame IS the plan**: the SR is literally the successor-representation = the value
  basis, so a sparse goal-reward propagates over it and the agent climbs V; a **shallow sampled search** refines
  the next move. Subgoals, if they reappear, are *eigenoptions read off the SR-frame*, not enumerated.
- The **three parallel perception learners die** (`DynamicsModel`, `GoalModel`, `ObjectPerceiver`): the column's
  `G` learns the dynamics, `V` learns the reward/goal from the score, and the body is the efference copy. This is
  the real de-siloing — today `perception/` is a stripped second world-model beside the column (the exact
  [[feedback_one_model]] anti-pattern).

Net effect, the point of this plan: **fewer files, far fewer lines, one model, general by construction** — the
transformer-backbone aesthetic. Reasonable performance + fluid intelligence come from the *same* change.

## Target architecture (the whole agent)
```
src/
  column.py     # THE MODEL. H: obs-latent (SR-eigenframe L6) · G: dynamics (L5 ⊕ learned effects, the recurrence)
                #   · readout (L4 content / L23 memory). L4/L5/L6/L23 folded in like a transformer block's sublayers.
  agent.py         # THE LOOP. online act-by-sampled-search (Gumbel + Sequential Halving, V-bootstrapped) over the
                   #   neocortex + flattened-prior exploration + learn-online (V: SVE+mixed). Absorbs reward.py +
                   #   planner.py + the old collect loop + the thin driver. USES (does not absorb) the BG + thalamus.
  perceive.py      # THE ONLY task-format code: obs -> column input (segment, efference-copy body, motion, the
                   #   grid/colour/action vocab). Absorbs objects/scene/dynamics_perceive/object_perceiver.
  recurrence.py    # the ONE canonical SelectiveRecurrence (shared by the column + the language SSM). [keep]
  thalamus.py      # cross-column binding — the model is task⊕space, not one column. [KEEP: the full neocortex]
  basal_ganglia.py # the learned POLICY PRIOR P the search refines (+ emergent allocation). [KEEP]
  env.py           # the Environment contract (reset/step/actions, + the click coords). [keep]
  tasks/           # the worlds: core · game · harness (contract view folded in) · layouts · games/. [del oracle, agents]
  demos/           # validations — a later consolidation pass (19 -> a handful).
  research/        # factorize (→ the future SENSORY column for raw pixels) · residual (its carry demo). Not the loop yet.
```
Core agent = **~6 files** (column, agent, perceive, recurrence, thalamus, basal_ganglia) + env + tasks/. ~27 core
scripts → ~14 — the reduction is from killing the parallel learners, the oracle, the planner, and the offline split.

## Code-reduction map (current -> fate)
| file | fate | why |
|---|---|---|
| `perception/perceive.py` | **KEEP** = the one perception file | absorbs the rest's perception parts |
| `perception/objects.py` | merge → perceive | segment / modal_background are perception primitives |
| `perception/dynamics_perceive.py` | split: perceiver→perceive, `collect`→agent loop | learning goes online |
| `perception/object_perceiver.py` | **delete** | body = efference copy (perceive); pushable → column `G` |
| `perception/goal_discover.py` | **delete** | the win/goal → `V` learns it from the score (EZ-V2 has no goal model) |
| `perception/scene.py` | merge → perceive | format adapter stays; the `WorldModel` role-decode dissolves into G/V |
| `tbt/column.py` | **KEEP** = the model | fold L4/L5/L6/L23 in |
| `tbt/l4…/l5…/l23_object.py` | fold → column | transformer-block style: sublayers in one file |
| `tbt/l6_grid.py` | **delete** | the innate hex prior is "off"; the SR-frame is the substrate |
| `tbt/dynamics.py` (DynamicsModel) | **delete** | conditional effects → the column's `G` (augmented recurrence) |
| `tbt/reward.py` | merge → agent | it IS `V` + the search; lives in the loop |
| `tbt/planner.py` | **delete** | subsumed by the sampled search |
| `tbt/agent.py` | **becomes** the online loop | absorbs reward/planner/collect; USES the BG (P) + thalamus |
| `tbt/recurrence.py`, `tbt/env.py` | **KEEP** | canonical recurrence; the contract |
| `tbt/basal_ganglia.py` | **KEEP (core)** | it IS the policy prior P the search refines (+ emergent allocation) |
| `tbt/thalamus.py` | **KEEP (core)** | cross-column binding — the model is task⊕space (the full neocortex) |
| `tbt/factorize.py` | → `research/`, then core | action-orbit disentanglement = the future SENSORY column (raw pixels) |
| `tbt/residual.py` | → `research/` | recursive-residual — its carry demo; effects now live in the column's `G` |
| `tasks/oracle.py` | **delete** | ditched (slow + a crutch) |
| `tasks/agents.py` | **delete** | RandomAgent/run_episode — inline in the demo that needs it |
| `tasks/contract.py` | merge → harness/env | one env file |

## The online loop (the heart — EZ-V2 on the column)
```
state: the column (H,G), V (over the SR-frame), a small replay buffer.
each env step:
  z   = column.perceive(obs)                 # H — the latent: body cell on the SR-frame + the scene
  cand = sample_K(prior(z) ∪ flattened_prior(z))     # K candidate moves; the flattened share = exploration
  for a in cand: q[a] = r̂(z,a) + γ·V(G(z,a))         # 1-step model lookahead, V-bootstrapped leaf
  a*  = sequential_halving(cand, q, sims≈16)          # tiny pure-exploration bandit picks the best
  obs = env.step(a*)
  buffer.add(z, a*, reward, obs)
  # learn online:
  column.update(buffer)      # H/G: INCREMENTAL TD-SR (local updates, no batch eigh); G learns effects online
  V.update(buffer)           # value from the sparse score; target = mixed( multi-step-TD , SVE )
```
- **V over the SR-frame** is what carries the long horizon: the search stays shallow (a few sims), `V` at the leaf
  supplies "how close to the goal" because the SR propagates the sparse reward. The agent climbs V to the goal.
- **SVE** value target = mean of N short imagined `G`-rollouts (the recurrence rolled forward), `γ^{2t}`-damped.
- **Mixed target**: trust real returns while `G`/`V` are young or data is very fresh; SVE for the stale middle.
- **Exploration** = the flattened-prior share of the sample (∪ the existing `reward.py` novelty bonus). NO oracle,
  NO separate curiosity module. This is the fix for the depth gap (LockPath-L2) the oracle was masking.
- **Effects, two kinds (the EZ-V2 re-read line).** A VISIBLE effect — a key opening a *rendered* door, a pad being
  covered — is just a frame-change `G` learns to predict; the latent encodes the visible frame, NO augmented
  structure (full-obs, S3). A HIDDEN effect — the toggle's door, invisible when open (aliased) — needs the
  recurrence to carry a belief + cloning to split the aliased cell (S5); that is the ONLY place a non-observable
  latent dimension is ever minted.

## Staged execution (incremental, each gated by the steps metric — solves online, in ≤ budget)
- **S1 — merge to one online loop + the free deletions.** Fold `collect` into `agent` (learn while playing); delete
  `oracle.py`, `agents.py`; merge `contract`→harness. Loop still uses today's value/subgoals, just online. *Gate:*
  the easy games still solve online (MultiKey, Sokoban) at comparable steps. _(Oracle removal: already done.)_
  _PROGRESS 2026-06-26: the merged loop `play_online` is built (act+learn in one pass, no collect/oracle).
  MultiKey **2/2 online**, LockPath **2/4** (L0/L1). Sokoban **0/3 online** — one continuous run explores far less
  than 150 reset-rich episodes, so it never finds the cover: the S4 exploration gap, surfaced early.
  Multi-episode + ε-explore (anneal): DONE — **MultiKey converges to OPTIMAL online** [9,13], 20/20 of the last
  20 episodes (vs [190,31] single-episode). **Sokoban still 0/3**, and DIAGNOSED: its cover-all win is **~1/150
  episodes by random** (offline landed exactly 1 lucky win → bootstrapped goal/req → 3/3; the online loop learns
  body+pushable fine but landed 0 — ~50% likely at that rate). So it's the EXPLORATION FRONTIER (S4: deliberate
  novelty over the augmented pad-covered state — cover on PURPOSE), not a merge defect. NEXT: free deletions +
  S2; S4 cracks Sokoban._
- **S2 — collapse the parallel learners into the column.** Body→efference copy in `perceive`; dynamics→column `G`;
  goal/reward→`V`. Delete `goal_discover`, `object_perceiver`, `dynamics`; collapse `perception/`→`perceive.py`.
  *Gate:* world-model quality (the learned effects/roles) ≥ today, online; steps hold.
  _PROGRESS 2026-06-26: WRINKLE — DELETING the learners (dynamics→G, goal→V) can't precede S3/S5, which are what
  give the column its G/V. So S2 splits: **(a) the file/code consolidation NOW** (perception 6→1, no capability
  change), **(b) the learner-deletion AFTER S3/S5**. **(a) DONE: perception 6→2** — `perceive.py` = primitives +
  segmentation + ObjectPerceiver(E) + GoalModel(F) + DynamicsPerceiver (all the "obs→learned-symbols"); `scene.py`
  = the format adapter; `collect` moved to the eval harness. objects/object_perceiver/goal_discover/dynamics_perceive
  deleted; online MultiKey unchanged (2/2 optimal) + offline collect still learns, both validated. REMAINING for
  S2: remove the offline-path duplication (`evaluate`/`full_obs`/`collect` — entangled with the partial-obs eval),
  and the learner-DELETION (→ column G/V) which waits on S3/S5._
- **S3 — the sampled search replaces subgoals.** Gumbel + Sequential-Halving move-search, V-bootstrapped; delete
  `planner`, fold `reward`→agent. *Gate:* no subgoal enumeration anywhere; steps ≤ today on the solvable games.
  _DESIGN (re-cut 2026-06-26, after re-reading EZ-V2 §representation): the augmented state is NOT hand-coded —
  that is a bitter-lesson violation AND unnecessary. EZ-V2 assumes FULL observability and never builds an augmented
  state: the relevant fact (door open/closed, pad covered) is OBSERVABLE (in the frame), so the latent just encodes
  the visible frame and a **TD-learned `V(latent)`** carries the multi-step value — `V(at the key, door-closed)` is
  high because firing the key reaches the goal; the function GENERALISES (no 2^K). So S3 = **a TD-learned `V` over
  the column's latent** (the SR-frame, which already encodes the visible scene/map) + a **shallow search** picking
  moves by `Q = r̂ + γ·V(G(s,a))` (Gumbel/Seq-Halving only earn their keep on the real-ARC click). This DELETES the
  whole subgoal layer (`_subgoals`/`_value_subgoals`/`_plan`/`_navigate`, the fire/cover/goal enumeration). Hard
  part: TD-learning V from sparse reward (credit assignment) — helped by the SR being the value basis + EZ-V2's
  SVE/mixed targets. HIDDEN-state games (the toggle) are NOT this; they need the recurrence + cloning (S5). First
  step: a TD-learned V over the place codes driving move-selection, A/B vs `_subgoals`, then cut over + delete it._
  _PROGRESS 2026-06-26 — mechanism BUILT + validated end-to-end (`tbt/value.py`: `Value` + `ValuePlanner`). Three
  findings, each load-bearing: **(1)** a plain TD-V over the place code is OPTIMAL on navigation (L0: 8 steps) but
  ALIASED on composition (L1 key+door: worse than random) — the door is observable in the frame's content but the
  door cell stays walkable, so the place code is identical open/shut. Fix = **bind** `z = place ⊗ state` (state =
  which learned effects fired), which de-aliases. **(2)** the Hadamard bind has norm ~1/√d → `V(z)≈0` stalls TD;
  **unit-normalise** `z`. **(3)** a 1-step value-greedy COLLAPSES on L1 (commits to bumping the shut door);
  depth sweep in the fast lab: depth 1/2/3 collapse, **depth-5 model-rollout search solves 40/40 at ~optimal** —
  EZ-V2's thesis confirmed (the value is necessary, the SEARCH does the credit assignment). Real stack, LEARNED
  forward model (column walkable = "where", learned effects = "what"): **L0 30/30 @ 8 steps, L1 30/30 @ ~13
  steps**, NO subgoal enumeration. Iteration-speed fix: a perception-free value-lab (the map as a graph) + a
  latent cache + a pickled world model — minutes → ~1s. REMAINING S3: the PUSH forward model (L2 block+pad — the
  state must include pads-covered + G must predict a block sliding); multi-level keying (r̂/term are (state,cell,
  move) — fine per level, conflate across levels); the CUTOVER (drive the agent with `ValuePlanner`, delete
  `_subgoals`/`_value_subgoals`/`_plan`/`_navigate`); then the Phase-2 acceptance-test cleanup (move `_state`/
  `_predict` colour logic into perception so `tbt/` is vector-only)._
  _G BUILD DESIGN (resolved 2026-06-26, after the objectperceiver/disentanglement discussion — the bitter lesson
  at the dynamics + perception layers):_
  - _A push-SPECIFIC forward model = 0% closer to the goal. The general G already exists: **`recursive_residual`
    over factored coordinates** (the same MDL search that found carry; ANY conditional mechanic = a conditional
    coordinate-delta). The fix = delete `_predict`, factor the frame into coordinates, roll `recursive_residual`
    forward as G. Push/door/pad/toggle all emerge as learned rules — no per-mechanic code._
  - _**THE CRUX = the coordinate frame** (lab `scratchpad/test_general_dynamics.py`): absolute coords learn every
    NON-relational effect (move 100%) and miss every relational one (**push 0%**); **EGOCENTRIC** coords (object
    pos relative to the body) make push a plain literal (**100%**), because the relation becomes local. So G =
    `recursive_residual` over egocentric object coordinates._
  - _**The coordinates are common-fate movers, not connected-components.** The object DOF the dynamics needs =
    cells that translate together under action (the disentanglement/action-orbit principle at the grouping level).
    Connected-component `segment()` stays ONLY as the perceptual bootstrap (a fair Core-Knowledge objectness prior;
    needed to perceive a still object before it moves). Evidence (`test_common_fate.py`): on a multi-colour rigid
    object + two touching independently-moving same-colour blocks, same-colour CC mis-groups 6 cells, multicolour
    CC 3, **common-fate 0** — uniquely right on touching-independent + multi-colour objects._
  - _**Column-specialization (Cipher's idea, folded in):** an **egocentric column** (G + the common-fate
    coordinates + interaction dynamics; seeded by the partial-obs `_frame_column`) **thalamus-bound** to the
    **absolute SR-frame map column** (navigation / goal location), **BG-gated**. Mirrors hippocampal-allocentric +
    parietal-egocentric + the retrosplenial transform. Navigation reads absolute; dynamics reads egocentric; the
    value binds across both — also the answer to "could a column take that role"._
  - _Build increments: **(1)** G = egocentric common-fate coords → `recursive_residual`, ONE model predicts push
    AND door (lab); **(2)** the egocentric/absolute column pair + thalamus binding; **(3)** the object-aware value
    latent (V binds object coords, not just the agent place); **(4)** cutover + delete the enumeration; validate
    L1/L2/Sokoban from one learned G. Sparse reward is answered by G itself: find the reward ONCE, then PLAN
    through G to re-reach it (EZ-V2 sample-efficiency), no dense signal needed._
- **S4 — exploration + value targets.** Flattened-prior + novelty; SVE + mixed target; V over the SR-frame. *Gate:*
  reaches a DEEP goal online (LockPath L2/L3) that random never reached — the thing the rework is *for*.
- **S5 — the HIDDEN-STATE frontier (recurrence + cloning) — beyond EZ-V2.** For state the frame does NOT reveal —
  the toggle's door, invisible when OPEN (an ALIASED observation: same cell, passable vs blocked) — a stateless
  latent cannot represent it, and EZ-V2 (full-obs, feed-forward) cannot either. The column's **recurrence carries a
  belief** (`A = L5 ⊕ effects`: "I pressed the switch") and **CSCG one-shot cloning** splits the aliased cell into
  context states on prediction error (exafference). The genuinely novel part + the honest open problem (one-shot
  *online* cloning; CSCG normally needs EM). *Gate:* the toggle solved online with NO hand-coded door-state.
  _(NB: the FULL-OBS effects — a key opening a *visible* door — are NOT here; G learns those visible frame-changes
  in S3. S5 is only for state the frame hides.)_
- **S6 — fold the layers + final reduction.** L4/L5/L6/L23→column; move `factorize/residual`→`research/` (BG +
  thalamus STAY — the neocortex); consolidate demos. *Gate:* file/line count down ~½; all prior gates still green.

## Decisions (resolved 2026-06-26) + the one genuine frontier
- **Multi-column stays in the core.** The model is task⊕space columns + thalamus binding; **`basal_ganglia` is
  EZ-V2's policy prior `P`** that the search refines, thalamus is the cross-column bind. The full neocortex driven
  by the EZ-V2 value/search — not narrowed to one column.
- **Online `H` = incremental TD-SR now** (Monty-style local updates), not a batch-eigh-on-a-timer bridge. The right
  online substrate from the start; the batch `eigh` survives only as a fallback / test oracle.
- **Consistency is NOT a bolted-on gradient head — it IS the column.** EZ-V2's SimSiam loss exists to force a
  *neural* encoder to become predictive without collapsing. Ours is predictive **by construction** (the SR-eigenframe
  IS the successor representation), the **L5 operators are least-squares-fit so `operator@place[s] ≈ place[s2]` —
  the consistency objective in closed form** — and an orthonormal eigenframe cannot collapse. So no head for
  STRUCTURE. The one role it doesn't cover is the **raw-pixel perceptual encoder** (real ARC frames → latent, which
  we hand-code today); the TBT answer there is a **SENSORY COLUMN** that learns objectness/features via the same
  mechanisms (action-orbits / SR-over-features, `factorize.py` the seed) — more columns, not more loss functions.
- **The frontier: credit assignment over sparse, multi-step reward.** EZ-V2 leans on V + SVE + replay; we lean on
  the SR being the value basis. S4 proves this or finds it wanting — the one place this could genuinely not work.

## Success criteria — THE ACCEPTANCE TEST (the bitter lesson, made testable)
**The plan is NOT complete until there is ZERO domain/environment-specific code in the MODEL ARCHITECTURE
(`tbt/`).** The column + recurrence + search + value must be GENERAL — they work on any structure (line / ring /
2-D / tree / language) and know nothing of grids, colours, moves, doors, pads, or keys. Every environment-specific
fact is either **LEARNED** (the column's L5 operators = the action geometry; its latent + `G` = the dynamics/
effects; `V` = the goal/value) or **quarantined in `perception/` + `tasks/`**. Concretely, by the end `tbt/` has
no hand-coded action deltas (move geometry is *learned* operators), no effect/role literals, no fire/cover/goal
subgoal-type enumeration, no hand-built augmented state. Heuristic check:
`grep -inE 'grid|colour|color|delta|door|pad|key|fire|cover|pushable|blocking|GameAction|subgoal' src/tbt/*.py`
returns nothing structural. (Today it does NOT pass — `tbt/planner.py` IS the domain-specific model code that S3
dissolves; that's the point.)

Plus: online, oracle-free, no per-mechanic code; solves the suite (incl. a DEEP-mechanic game random couldn't
reach) measured in **steps**; perception **6→2 (done)**, the subgoal planner dissolved into the general search +
value; **no parallel world-model beside the column**, **no enumerated subgoal types**, **no hand-coded augmented
state**.
