# REWORK_PLAN — one online EZ-V2 loop on the column; drastically less code

**Status: DRAFT 2026-06-26.** Supersedes REORG_PLAN Part B3/B4 (the emergent-subgoal work) — EZ-V2 removes
explicit subgoals entirely, so that work folds into this. REORG Part A (the `src/` merge) and B1/B2 (the thin
agent + the `Environment` contract) STAND and feed directly into this. Detailed paper notes: `src/tbt/EZV2_NOTES.md`.

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
- **Conditional mechanics** (cover/toggle): `G` predicts the *augmented* next latent (position ⊕ effect-state), so
  V can value "cover the pad → goal reachable". This is the EMERGENT_PLAN "A = L5 ⊕ residual effects", now the
  only place effects live.

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
- **S4 — exploration + value targets.** Flattened-prior + novelty; SVE + mixed target; V over the SR-frame. *Gate:*
  reaches a DEEP goal online (LockPath L2/L3) that random never reached — the thing the rework is *for*.
- **S5 — augmented `G`.** effects in the recurrence → cover/toggle solved online. *Gate:* Sokoban/Toggle online.
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

## Success criteria
Online, oracle-free, no per-mechanic code: solves the suite (incl. a DEEP-mechanic game random couldn't reach)
measured in **steps**; **core agent ≈ 6 files**, perception **6→1**, tbt **15→~6**, total core scripts **~½**, lines
well down; and the end state has **no parallel world-model beside the column** and **no enumerated subgoal types**.
