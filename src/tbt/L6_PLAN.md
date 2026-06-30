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
  - **I3-attempt-1 (2026-06-30, reverted): the eigenpurpose as a GATED BONUS is too weak -- it MUST be PROPAGATED.**
    Tried the cheap version first: in `_choose`'s flat case (`_tab_spread<=eps`), add `_explore_bonus(state)` = the
    eigenpurpose of each action's predicted next-state (`col.predict`, so untried actions get a direction too), scaled
    by `beta`, alongside the forward-model bonus. RESULT: `test_live_loop` stayed green (the GATING works -- proof the
    flat-case gate preserves convergence, unlike the prototype's persistent term), but nav did NOT improve (MultiKey
    1/2 @937, LockPath 2/4, Sokoban 0/3 -- identical to baseline). WHY: a one-step bonus (~`beta`*1 = 0.3) is dwarfed by
    the FRONTIER OPTIMISM (`explore`=3.0) that every untried action already carries -> the agent still picks untried-
    blindly, the eigenpurpose can't bend the trajectory. The prototype worked precisely because it was SWEPT into the
    value (propagated deep across the graph), not added at the leaf. CONCLUSION: I3 needs a real second SWEEP -- a
    separate `V_explore` dict that PROPAGATES the eigenpurpose (`reward.plan` over `reward_total + intrinsic`), arbitrated
    against `V_exploit` by gradient. The `_explore_bonus` eigenpurpose computation (grid-cells oriented toward under-
    visited, normalized [0,1]) is correct + reusable; only the WIRING (leaf-bonus -> swept-value) was wrong. Build I3 as
    the two-value `reward.py` refactor: parameterize the sweep by `(V_dict, reward_fn)`, run it twice, arbitrate.
  - **I3 BUILT (2026-06-30, `fa0453f`), 89 green, no regressions.** `reward.py` sweeps TWO values (`_sweep(T,preds,cur,
    V,reward_fn,q)` x2): **`V_exploit`** = `_reward_base` (reward + epistemic, optimistic) -- the NORMAL-operation value
    -- and **`V`** = `_reward_base + intrinsic` (the eigenpurpose). The agent runs on `V_exploit` EVERYWHERE except the
    measured **EFE dead-zone**, detected by a clean BOOLEAN (not a flatness threshold -- the key correction): `dead_zone
    = not R_ext and all((state,a) in tried)` (reward-less anywhere yet AND every action HERE already tried). There it
    switches to `V` for a propagated, DIRECTED escape. So the eigenpurpose fires throughout an explored pocket (-> its
    frontier) but NOT at the boundary (untried -> frontier optimism), and NEVER after the first reward (-> pure exploit/
    transfer). `_eigenpurpose()` = grid cells oriented anti-to-visits, normalized, `beta`-scaled; its SVD is THROTTLED
    (`_ep_every=16`; the direction is slow-changing). **Result: MultiKey first goal 937 -> 250 (3.7x); LockPath 2/4 and
    Sokoban 0/3 unchanged** (Sokoban needs the multi-step PUSH maneuver, not directed exploration -- a separate gap).
  - **Two traps hit + fixed during the build (so they aren't repeated):** (1) `V_exploit` must KEEP the optimism +
    epistemic (the OLD value); making it reward-ONLY lost the optimism-driven goal-seeking and regressed 3 tests
    (live-loop convergence 9 vs 5, CollectAll exploit-over-forward, learning-progress contrast). (2) The forward-model /
    dead-zone gate must read the `V_exploit` action-SPREAD, NOT an all-zero clean value -- gating on an always-zero
    spread turned the dense field ON every step (1.2M `_field_all_bg` calls/60 steps = catastrophic). Gate on the right
    spread; throttle the SVD.

## 4. Salvaged-prototype context (so it isn't relearned)
The reverted eigenpurpose prototype (`reference_eigenoptions_subgoals`) computed the SR SVD inline in the agent and
added the eigenpurpose as a PERSISTENT reward in `reward_total`. Result: at scale 1.0 it made things WORSE (overrode
exploitation -> LockPath 2->1, MultiKey 1->0); at scale 0.1 it HELPED (MultiKey 1/2 -> 2/2, L0 937->604) but still
broke `test_live_loop` convergence (the persistent term contaminates the exploit policy). LESSON: the eigenpurpose is
sound but must live in a SEPARATE explore value (L6-I3), not be summed into the one value. The mechanism + the MultiKey
validation are the evidence the split will work.

## 6. AFTER L6 -- reconnect L4 / L5 / L2-3 + their L6 interactions (the predict-sense-update loop)
Once L6 is a live location frame (I1-I3), wire the OTHER layers into the proper TBT loop -- today's opaque-config_state
shortcut bypasses them. The TBT cycle (Hawkins): **L6 location -> L4 predicts the feature-at-location -> sense + compare
-> L2/3 settles the object (lateral voting) -> L5 emits the next movement -> L6 path-integrates** to the next location.
Reviving L6 makes the location frame the substrate the whole column reads; these steps reconnect to it.

- **L7-A: L4 <-> L6 (feature-at-location, in the loop).** Run `L4.bind`/`readout`/`predict_feature` EVERY step (today
  only `refresh`, which is uncalled, does this). The agent's state becomes the sensed FEATURE bound to the L6 grid
  LOCATION, not the opaque `config_state`. L6 (location) modulates which feature L4 predicts; the mismatch is the
  learning signal. This is the TEM canvas (content x location, `reference_brain_generative_model`) that the forward
  model already half-uses -- now over the REAL grid frame, so it also lifts the DYNAMICS games (the forward model
  predicts feature-at-grid-location, not raw cells).
- **L7-B: L5 -> L6 (path integration closes the motor loop).** L5's chosen displacement (the efference copy)
  PATH-INTEGRATES L6's grid (`l6_grid.path_integrate` / `loc_move`), so the location updates with every move. The
  operator (`disp`/`recolor`/`edges`) stays, but now reads/writes the LOCATION-ANCHORED state instead of opaque symbols
  -- the operator generalises over the grid metric (the same displacement anywhere).
- **L7-C: L2/3 <-> L4(x)L6 (the object).** The object = the graph of (feature, grid-location) pooled over a sensing
  sequence; recognition + lateral CMP voting run on THIS and predict (top-down) the feature L4 should expect. Folds the
  standalone recognizer (`recognize_object`) into the loop; gives object permanence under the grid frame.
- **L7-D: the STATE shift.** From opaque `config_state` -> LOCATION-ANCHORED (grid place + features-at-locations). This
  is what finally gives the metric (vector nav), generalisation across positions, and the hippocampal binding the
  cross-column / heterarchy work needs.

Order: A (L4-over-L6) -> B (L5->L6 path integration) -> C (L2/3) -> the unified predict-sense-update loop. Validate each
by the oracle metric (`arc_offline.py`) + the suite; expect navigation AND dynamics efficiency to lift together once
the column predicts over a real location frame rather than memorising an opaque graph. (Then: cross-column heterarchy.)

## 5. Files / where things are
`l6_sr.py` (OnlineSR -- `M`, `value`/`values` built+tested but the planning-value use was reverted; the SR is learned
but unread in the loop), `l6_grid.py` (the inert innate hex grid), `reward.py` (the value sweep; `_need` flat proxy;
`reward_total`), `agent.py` (`_choose` -> `_tab_value` -> `col.act`; `_tab_spread` arbitration; FM1-4 forward model),
`column.py` (coordinator; `refresh` builds place codes but is uncalled in the loop), `arc_offline.py` (the oracle
benchmark -- the validation harness), `arc_sdk.py` (TbtPolicy; local=False = config_state for the nav games).
