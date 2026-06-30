# L6_PLAN — revive Layer 6 (the location frame: grid cells), then wire it into planning

*Plan, 2026-06-30. Survives context compaction. The navigation gap (the agent is 5-100x over the oracle on the
revived games, Sokoban 0/3) traces to L6 being DISCONNECTED. Companion to `FORWARD_MODEL_PLAN.md`,
`reference_eigenoptions_subgoals`, `reference_brain_planning`, `reference_grid_sr_eigenbasis`, `project_offline_game_benchmark`.*

## 0. The diagnosis (measured) — what each layer is actually doing
- **L5 = the workhorse**: the tabular transition graph (`edges`), the operator (`disp`/`recolor`), the forward model
  (`field_rule`), the motor. The engine of every game.
- **L4 = half-used**: `L4.encode` (the content codebook) feeds the forward model's `feature_field`; `L4.bind`/`readout`
  (feature-at-location) is NEVER called in the loop (only in `refresh`).
- **L6 = DEAD**: the innate grid (`l6_grid`) is inert (never driven); the SR (`l6_sr`) is `observe`d every step
  (learning `M`) but `M` is **never read** in the loop -- `refresh()` (which builds the place codes) is called only in
  a unit test. So the SR is wasted O(states^2) compute; the grid never ran.
- **So the games run on L5.edges + `reward.py`'s value sweep** -- a bare tabular RL loop (+ L4.encode for the dynamics
  grain). NO location frame / metric / grid / spatial generalization. THAT is why navigation is 5-100x over oracle.
- **Oracle baseline** (`python src/arc_offline.py all 2500`): Sokoban 0/3, LockPath 2/4, MultiKey 1/2 (nav);
  Toggle/Tetris/CollectAll WIN (dynamics). RHAE proxy `(oracle/agent)^2` ~0.01 -- ~0 on the real 5x-human budget.
- **The EFE dead-zone** (the navigation failure mode, measured): once local exploration is exhausted (exploreFrac->0)
  and no reward is found yet (R_ext empty), BOTH EFE terms go flat -> random walk in a pocket DISJOINT from the
  solution (Sokoban oracle-path coverage `X.........`). The flat frontier optimism gives no DIRECTION -- the gap.

## 1. Homework (done 2026-06-30)
- **Grid = SR eigenvectors** (Stachenfeld); each grid cell = one eigenvector; mesh size ∝ eigenvalue (multi-scale).
- **Module cap**: discrete modules, ratio ~1.42 (≈√2), only ~**3-5** of them -> a small `k` (~5), geometric scales.
- **Online learning = Oja/Sanger** (Hebbian PCA, incremental, no batch `eigh`); Dordek/Fiete: grid cells emerge as
  the (non-negative) PCA of place-cell inputs -> a Hebbian layer extracting the top-k eigenvectors of the SR.
- **`l6_grid` is a COMPLETE innate multi-scale hex grid** (3 scales x 3 dirs, path-integrable, grid->place,
  error-correcting decode) -- just never driven. The 2D prior already exists.
- **Salience** (Cipher's flag) is a SEPARATE, validated exploration driver (bottom-up perceptual novelty, orient to a
  standout stimulus) -- complementary to the structural eigenoption; a deferred follow-up.

## 2. The design
L6 = grid cells = the **innate hex** (`l6_grid`, the 2D metric prior) **re-tuning toward the learned SR
eigenstructure** (Oja-extracted top-k eigenvectors = the layout's bottlenecks). The brain does exactly this (grid
re-tunes to barriers/rewards). It gives BOTH:
- the **METRIC** -> vector navigation (reach a KNOWN goal directly, not retrace edges = exploitation efficiency), and
- the **EIGENSTRUCTURE** -> eigenoptions (cross bottlenecks to FIND goals = the dead-zone / exploration fix).

## 3. Increments (each suite-green; validate by the oracle metric -- the 5-100x must collapse)
- **L6-I1 -- the LEARNED grid-cell representation.** Give the column a proper grid: the top-k SR eigenvectors (`k≈5`,
  geometric), extracted from `l6_sr.M`, initialized from the innate hex. Periodic `eigh`/SVD is fine for the small
  game graphs (26-159 states); Oja/Sanger is the scale path (note in code). Expose `col.grid`/`grid_code(s)`. PURE
  representation, no behavior change -> cannot regress the suite. Validate: matches the offline SVD; the top vector
  separates interior (high-visit) from boundary (visits~1), as the SV0 probe already showed.
- **L6-I2 -- WIRE it into planning.** (a) the SR NEED in `reward.py._need` (currently a flat 1.0 proxy) = place-code
  similarity, so prioritized sweeping focuses on the agent's actual future. (b) VECTOR NAVIGATION via the grid metric
  (a shortcut to a known goal, vs retracing the graph). This is the EXPLOIT-side efficiency.
- **L6-I3 -- the EXPLORE/EXPLOIT split (salvages the eigenpurpose).** The dead-zone fix, done right (NOT one conflated
  value -- that's what broke it): `V_exploit` = sweep over R_ext ONLY (clean -> `test_live_loop` convergence kept);
  `V_explore` = epistemic + the EIGENPURPOSE (the SR-eigenvector gradient toward the under-visited extreme, off the
  L6-I1 grid). ARBITRATE per state (same shape as the `_tab_spread` tabular/forward gate): if `V_exploit` has a
  gradient here (a known way to reward), exploit; else explore. This is the brain's explore/exploit arbitration (Daw).

## 4. Salvaged-prototype context (so it isn't relearned)
The reverted eigenpurpose prototype (`reference_eigenoptions_subgoals`) computed the SR SVD inline in the agent and
added the eigenpurpose as a PERSISTENT reward in `reward_total`. Result: at scale 1.0 it made things WORSE (overrode
exploitation -> LockPath 2->1, MultiKey 1->0); at scale 0.1 it HELPED (MultiKey 1/2 -> 2/2, L0 937->604) but still
broke `test_live_loop` convergence (the persistent term contaminates the exploit policy). LESSON: the eigenpurpose is
sound but must live in a SEPARATE explore value (L6-I3), not be summed into the one value. The mechanism + the MultiKey
validation are the evidence the split will work.

## 5. Files / where things are
`l6_sr.py` (OnlineSR -- `M`, `value`/`values` built+tested but the planning-value use was reverted; the SR is learned
but unread in the loop), `l6_grid.py` (the inert innate hex grid), `reward.py` (the value sweep; `_need` flat proxy;
`reward_total`), `agent.py` (`_choose` -> `_tab_value` -> `col.act`; `_tab_spread` arbitration; FM1-4 forward model),
`column.py` (coordinator; `refresh` builds place codes but is uncalled in the loop), `arc_offline.py` (the oracle
benchmark -- the validation harness), `arc_sdk.py` (TbtPolicy; local=False = config_state for the nav games).
