# BASAL_GANGLIA_PLAN — the faithful basal ganglia (grounding P2: BG-arbitrated action selection)

*2026-06-30 (session 2). `GROUNDING_PLAN.md` P2 ("replace `agent._choose`'s hand-coded arbitration with `BG.gate`")
became **"make the basal ganglia FAITHFUL, then wire it"** once we actually studied the organ — the user's call: we'd
built on an immature `basal_ganglia.py`, the same immaturity we found in the GSG. This doc stores the research + the
build proposal. **Benchmark = anatomical correctness** (each part does its BG job, connected as the anatomy specifies),
NOT a game score — test by MECHANISM. Full mechanism + citations: memory **`reference_basal_ganglia`**. Companions:
`GROUNDING_PLAN.md` (P2), `MOTOR_REFACTOR_PLAN.md §8.6` (the GSG's commitment = the STN), `reference_gsg_goal_generation`
(Cisek affordance competition = the cortical side of this selection), `reference_brain_planning` (the value the saliences carry).*

## 1. What the basal ganglia ARE (the research, condensed)
The BG is a **centralized SELECTION device** (Redgrave/Gurney–Prescott–Redgrave): many subsystems bid for the shared
motor/cognitive resource; one wins. Five mechanisms, all decision-relevant:
1. **Selection by SALIENCE, default-closed.** Each competing channel presents a scalar **salience** (value + urgency).
   Output nuclei (GPi/SNr) TONICALLY INHIBIT everything; selection = focal **DISINHIBITION** of the winner (**direct/D1**)
   while competitors are SUPPRESSED (**indirect/D2 + STN**) — off-centre, on-surround winner-take-all. Not a spread.
2. **Actor–critic learning by dopamine RPE (DELAYED, outcome-based).** Dopamine ≈ the TD reward-prediction-error. A
   **critic** (striosomes / ventral striatum) learns reward prediction → emits `δ`; that `δ` trains the **actor** (matrix
   → GPi/SNr selection weights) by DA-gated plasticity. What to select is learned by whether selecting PAID OFF, not by
   the bid itself.
3. **OpAL opponency (Collins & Frank).** TWO opponent actors: **D1 "Go"** learns action BENEFITS, **D2 "NoGo"** learns
   COSTS. `Act(a) = β_g·G(a) − β_n·N(a)`. Three-factor Hebbian updates (the actor weight scales its own update) make the
   two NON-redundant — they specialize by reward-probability range.
4. **Tonic dopamine = the explore/exploit + vigor GAIN.** `β_g = β·max(0,1+ρ)`, `β_n = β·max(0,1−ρ)`; **ρ = tonic-DA
   state** shifts the Go/NoGo balance smoothly (rich → Go/exploit; lean → NoGo/avoid). Explore↔exploit is a GAIN, not a
   switch; tonic DA also sets response VIGOR (opportunity cost of time).
5. **STN / hyperdirect = "hold your horses" (Frank 2006).** STN excitation ∝ decision **CONFLICT** → a global brake that
   RAISES the decision threshold when options are near-tied → buys settling time, prevents premature/impulsive commits,
   changes which action wins. The anatomical seat of **commitment / anti-thrash**.

**Equations (build-ready, OpAL\*):** `δ = R − V(a)` (bandit) / `R + γ·V(s') − V(s)` (sequential); `V ← V + α_c·δ`;
`G ← G + α_G·G·δ`; `N ← N − α_N·N·δ`; `Act(a) = β_g·G(a) − β_n·N(a)`; softmax select. `ρ>0` weights Go (discriminate
GAINS/exploit), `ρ<0` weights NoGo (discriminate LOSSES/avoid).

## 2. Where our current `basal_ganglia.py` is IMMATURE
`gate(options, values)` does `argmax(value + affinity)` and nudges `affinity += lr·(value − affinity)` WITHIN the call.
Against §1 it is missing: the **critic/RPE separation** (it reinforces the BID, not the delayed OUTCOME); **surround
suppression** (no active competitor inhibition); the **opponent Go/NoGo** (one value, no benefit/cost split → no
aversion); the **tonic-DA gain** (our explore/exploit is the hand-coded `dead_zone` boolean in `agent._choose`); the
**STN conflict/commitment**. `select`/`reinforce` (the column ALLOCATOR — MoE over the column pool, load-balanced) is a
DIFFERENT faculty and is closer to faithful (`reinforce` by value ≈ a crude RPE, `load` ≈ NoGo surround); it is untouched
by this plan and matters when the heterarchy adds columns. This plan upgrades the ACTION selector.

## 3. The PROPOSAL — the faithful BG (recommended shape: DISSOLVE the channels)
The GROUNDING_PLAN's "BG arbitrates 3 value-CHANNELS (exploit / eigenpurpose / field)" is a legacy of the hand-coded
switch. The faithful, simpler shape **dissolves the channels**: the BG selects the **ACTION** directly via opponent
Go/NoGo over ONE **combined salience**, with explore/exploit as the tonic-DA gain and commitment as the STN term. The
"grains" become TERMS in the salience, not competing meta-channels. Mapping to our code (reuse — invent nothing):

| BG part | our seat | what it does |
|---|---|---|
| **critic + RPE `δ`** | **`reward.py`** | already learns value + learning-progress; expose the per-step `δ` (score / `lp`) that trains the actor. This is also grounding **M1** (the value the salience carries). |
| **salience(a)** | the column's value read over `col.predict(s,a)` | `benefit ⊕ epistemic ⊕ eigenpurpose(L6) ⊕ field(L5)` — the M3/M4 "channels" fold in HERE as salience TERMS, no meta-selection. |
| **Go / NoGo actors** | **`basal_ganglia.py`** (new) | `G(a)` benefits, `N(a)` costs (three-factor OpAL), trained by `δ`; `N` gives principled AVERSION (GAME_OVER as a cost) the single value lacks. |
| **tonic-DA `ρ`** | driven by the value landscape (reward reachability / uncertainty from `reward.py`) | the graded explore/exploit gain — **replaces the `dead_zone` boolean**. |
| **STN conflict term** | **`basal_ganglia.py`** (new) | near-tied saliences → raise the threshold / hold the current choice → commitment + anti-thrash; ALSO serves the GSG's hysteresis (`MOTOR_REFACTOR §7.3/§8.6`). |
| **selection (disinhibit winner)** | `col.act` (the inverse-model motor) | enacts the max-salience action; the surround (rivals) suppressed. |

`agent._choose`'s entire 2-case `if tab_spread>0 & not dead_zone … else …` DISSOLVES: no `tab_spread`/`dead_zone`
booleans — one salience, one DA gain, one STN threshold. `_choose` returns to a thin "build salience → BG selects".

## 4. Staging (each MECHANISM-tested + suite-green; judge WHOLE per [[feedback_dont_salvage_between_critical_steps]])
- **B1 ✅ — the critic's `δ`.** `reward.RewardModel.critic_delta(s, s2)` = δ = `reward_exploit(s) + γ·V_exploit(s2) −
  V_exploit(s)` — the TD reward-prediction-error over the CRITIC value (clean reward + `lp`, NO eigenpurpose). Exposed
  for the actor (unwired). *Tests (mechanism, `test_basal_ganglia.py`):* δ equals the explicit TD formula; δ→0 along the
  optimal path once converged + mastered (lp=0); δ<0 for a step AWAY from reward; δ winds down as the reward is learned.
  Suite 101→103 (additive).
- **B2 ✅ — STRUCTURAL dissolution (behaviour-neutral).** `_choose`'s two-branch `if/else` + the `dead_zone`-forced
  branch collapse into ONE salience `V_exploit + g·(V − V_exploit)` selected by a single `col.act` (field gated by `g`).
  Suite 103 green, behaviour-IDENTICAL. **KEY FINDING (corrects the plan):** the eigenpurpose's explore/exploit gate is
  inherently a SHARP step ("any reward gradient → exploit"), NOT a graded tonic-DA gain — a graded/reachability gain
  *poisons* (the eigenpurpose fights the reward gradient; SR reachability also LAGS the reward sweep). So the sharp gate
  `g = (tab_spread≈0 or dead_zone)` is KEPT as a principled value-landscape signal, and **the graded tonic-DA gain
  DECOUPLES from here and moves to B3's Go/NoGo** (a different axis: reward exploit-gains vs avoid-losses). What B2 did
  NOT do: introduce a learned actor (that is B3) — B2 is the one-salience BASE B3 rides on. [The eigenpurpose stays
  (load-bearing for long-horizon planning, user); its SVD → online Hebbian PCA is a LATER swap, not now — §6 below.]
- **B3a ✅ — the OpAL Go/NoGo actor, in isolation.** `basal_ganglia.OpponentActor`: opponent `G`/`N` per (context,action),
  three-factor OpAL updates trained by the B1 critic `δ`, `Act=β_g·G−β_n·N`, tonic-DA `ρ` sets `β_g/β_n`. Mechanism-tested
  (`test_basal_ganglia.py`): Go/NoGo specialize (benefit vs cost), a cost earns NEGATIVE actor value (aversion), `ρ`
  shifts the gain. Additive, not wired. Suite 103→105.
- **B3b — live integration: BLOCKED, needs redesign (finding 2026-06-30).** Naively ADDING `Act(a)` to the salience +
  training on every-step TD `δ` REGRESSED navigation (0/8). TWO causes: (i) the model-free actor's per-(state,action)
  values FIGHT the strong model-based planner (`reward.py`) and, worse, its NoGo suppresses the SUBOPTIMAL exploratory
  steps nav needs; (ii) the actor's raison d'être is AVERSION, but `reward.py` is reward-ONLY — so TD `δ<0` means merely
  "worse-than-expected" (normal in exploration), NOT "bad." Training NoGo on that penalises exploration. ⇒ the actor
  earns its keep only with a GENUINE cost signal (GAME_OVER / punishment) it can learn NoGo from, on scenes that HAVE
  aversion (the current replicas are reward-only → nothing for it to add). **Redesigned B3b:** add GAME_OVER (+ blocked/
  no-progress) as a COST; the actor learns NoGo from THAT (aversion = NEW info, not duplicating the reward value);
  demonstrate the agent avoiding an aversive outcome. The generic-TD-every-step wiring is the wrong signal — DROPPED.
  (Deeper alt if that's insufficient: arbitrate the model-free actor vs the model-based planner as two controllers — a
  bigger design, deferred.)
- **B5 — STN commitment.** The conflict term raises the threshold on near-tied saliences (hold the current choice). *Test:*
  anti-thrash on a two-attractor scene; and it supplies the GSG's commitment (cross-check `MOTOR_REFACTOR §8.6`).

## 4b. The BG↔THALAMUS connection — what to account for NOW (the thalamus rework itself is DEFERRED)
The BG does not emit actions — it **selects by DISINHIBITING a thalamocortical LOOP** (Alexander/DeLong/Strick parallel
loops; Chevalier & Deniau disinhibition): GPi/SNr tonically inhibit the motor/associative thalamus (VA/VL/MD); the direct
pathway REMOVES that inhibition on the winner → RELEASES its cortical program; the surround stays inhibited. (The STN
"hold your horses" acts through this too — it re-excites GPi/SNr → more thalamic inhibition → holds all loops closed
under conflict.) The BG's INTERNAL computation (B1–B5: critic-RPE, Go/NoGo, tonic-DA, STN) is thalamus-INDEPENDENT, so we
build it now; only the OUTPUT framing must be right so it survives the heterarchy (three cheap constraints):
1. **Keep the two BG faculties DISTINCT:** `gate` (select among ACTIONS/options) vs `select` (allocate among COLUMNS/loops
   — the MoE allocator). The loop-selector is exactly what the thalamus will RELAY later; do not merge them.
2. **Selection = "RELEASE the selected program," not "emit action index k."** `col.act` is the degenerate SINGLE-LOOP
   enactment; keep the BG selecting among OPTIONS (actions now; column-programs in the heterarchy) so it generalises free.
3. **The BG-gated RELAY thalamus is a DIFFERENT thalamic function than our `thalamus.py`** (which models the higher-order
   Sherman–Guillery DRIVER — trans-cortical inter-column `bind`/`read`). Two thalamic roles (relay vs driver); the
   eventual thalamus rework covers BOTH. The BG plan must only NOT CONTRADICT the relay role — it does not build it.
**Deferred (do NOT plan yet, user's call):** the thalamus rework itself (the BG-gated relay + unifying it with the
higher-order driver). It is a heterarchy-era prerequisite, tackled AFTER the BG — one big bite at a time.

## 5. The decision (RESOLVED) + guardrails
- **DECISION (user, 2026-06-30 s2): the DISSOLVED-SALIENCE shape** — the BG selects the ACTION via opponent Go/NoGo over
  ONE combined salience (§3), retiring the hand-coded switch most faithfully. The explicit-channel alternative is NOT
  taken. B2 builds the dissolved form.
- **Context, never game features** ([[feedback_bitter_lesson]]): the salience/`ρ` read the VALUE LANDSCAPE (reward
  reachability, uncertainty, `lp`) — the cortical state — NOT colours/objects/scene heuristics.
- **One model** ([[feedback_one_model]]): every change is in the `tbt/` system; the critic is `reward.py` (reuse), the
  actor is `basal_ganglia.py` (revise), the selection is `col.act` (reuse). No parallel selector.
- **Reuse** ([[feedback_reuse_canonical_components]]): do not reimplement value (reward.py) or the motor (col.act);
  the BG adds the OPPONENCY + `ρ` + STN, nothing else.
- **Rides on:** M1 (the critic value — B1 advances it), M3/M4 (eigenpurpose/field become salience terms — subsumed
  here). Precedes P3/P4 only loosely; B1's critic work overlaps M1.

## 6. Reviewed 2026-06-30: the eigenpurpose SVD (KEEP; online Hebbian PCA is a LATER swap)
The ONLY SVD in the system is `l6_sr.grid()` (`np.linalg.svd(M)`, top-k=5 singular vectors = grid cells), consumed by
`eigenpurpose()` (directed-exploration intrinsic reward; capped n≤400, throttled every 16 steps). O(n³). **Decision
(user):** the eigenpurpose is load-bearing (it is how long-horizon planning/action is achieved — eigenoptions/bottleneck
sub-goals), so KEEP it; switch the batch SVD to **online Hebbian PCA (Oja/Sanger/GHA)** — the intended scale path already
noted in `l6_sr` — WHEN APPROPRIATE (better perf: streaming O(nk); fresher: no 16-step staleness; more faithful: grid
cells emerge online, not by batch eigh). NOT now (it does not block B3). Free interim option if the SVD ever bites:
truncated/randomized top-k SVD (`svds`/randomized), same result, O(n²k). Deferred rethink: directed exploration could
become a GSG EXPLORATION GOAL (SR-frontier, no SVD) at the exploration/§3 step — recorded, not scheduled.

## Sources
Redgrave, Prescott & Gurney 1999 (selection problem, PubMed 10362291); Gurney–Prescott–Redgrave 2001 (GPR model,
PubMed 11417052/3); Joel, Niv & Ruppin 2002 (actor–critic, PubMed 12371510); Collins & Frank OpAL / "normative
advantages of opponency" (eLife 85107); Frank 2006 "Hold your horses" (STN, PubMed 16945502); Cavanagh et al. 2011
(STN decision threshold, Nat Neurosci nn.2925); tonic-DA explore/exploit (eLife 51260; Niv 2007 vigor). Full notes:
memory `reference_basal_ganglia`.
