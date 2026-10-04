# Curriculum-less Coconut and search — planning

Status: PLANNING, opened 2026-10-03. Nothing here is built or decided; every "proposed" is for discussion.
Context: `BRAINSTORM.md` §3 (the first pass at this topic), `REFERENCES.md`, `refs/efficientzero_v2_2403.00564.md`.

**The question.** How does a looped transformer learn to think in continuous thoughts (Coconut's sequence-axis
recurrence) when there is no language-substitution curriculum and no human reasoning trace — only a verifier that
says, at the end, whether the answer is right? And what SEARCH supplies the learning signal that the curriculum used
to supply?

**Why search is the crux.** Coconut's curriculum gave every thought a target: at stage k, the loss on the remaining
language steps tells each thought what it must enable. Without it, Coconut reports "no better than no-CoT". An
outcome-only signal has to be turned into per-step targets somehow, and that is what a policy-improvement operator
does: at every state it visits, search produces an improved action distribution and a value — targets for the
policy and value heads at every step, from one bit at the end. The search is the curriculum's replacement, so most
of this document is about the search.

---

## 1. Ground rules (what the design must satisfy)

- **R1 — no traces.** No human or natural-language reasoning traces, no substitution curriculum. (A curriculum over
  PROBLEM difficulty is a different thing and is allowed unless decided otherwise — §9 Q5.)
- **R2 — outcome-only reward,** from a verifier, minus a compute cost. Whether partial credit (§5.2) and
  machine-state auxiliaries are allowed is open (§9 Q4, Q6).
- **R3 — thoughts are vectors fed back into the model,** never decoded to tokens between steps.
- **R4 — the model decides how many thoughts.** No padding to a fixed count.
- **R5 — compute is accounted in forward passes.** Every comparison between arms is at an equal pass budget.
- **R6 — thoughts must be necessary.** On the task family, a no-thought model of the same depth must provably or
  measurably fail; otherwise any result is uninterpretable (a model that can answer without thinking will learn to
  ignore its thoughts).

## 2. Thinking as a decision process — the exact definition

- **State** `s_t` — the problem x and the thoughts `z_1 … z_t`, as the model has absorbed them: concretely the
  model's internal state after position t (the last residual, plus its attention cache or Spotlight memory).
- **Actions** — `continue(z)` with `z` in the thought space Z (§3), or `answer`.
- **Transition** — `continue(z)`: feed z as the input at position t + 1 and run the looped core (K passes) → `s_{t+1}`.
  Deterministic, and EXACT: the dynamics are the network itself.
- **Terminal** — `answer`: decode y from `s_t`; reward `R = 1[y correct] − λ_T·t − λ_K·(total passes)`.

Five properties that make this unlike the games MuZero and EfficientZero were built for:

1. **Known, exact dynamics.** No world model to learn; the simulator is free and perfect (AlphaZero's situation,
   not MuZero's). EfficientZero's representation/dynamics learning and its consistency loss drop out.
2. **An exact value for stopping, at every node.** `answer` can be taken anywhere, and its value is the verifier's
   verdict on one decode — no learned estimate needed. Only the value of CONTINUING has to be estimated.
3. **Short episodes, terminal reward.** T is tens of thoughts at most, so Monte Carlo rollouts to the verdict are
   affordable while the value head is still useless.
4. **The environment trains with the agent.** The simulator shares weights with the policy, so the "dynamics" drift
   as training proceeds — reanalyse with a target network (EZ-V2: self-play net refreshed every 100 updates, target
   every 400) is the standard answer.
5. **Branching needs the state forked.** Attention: a prefix-shared, tree-structured k/v cache. Spotlight: a
   copy-on-write overlay — one thought step writes at most 9 · H · L cells (`refs/spotlight_memory_percepta_2026.md`).

## 3. The thought space — the decision that shapes every search algorithm

| | thought space | what the search sees | improvement guarantee | notes |
|---|---|---|---|---|
| **A1** | raw `z ∈ ℝ^d` (Coconut) | a d-dimensional continuous action | asymptotic only (EZ-V2 eq 7), and K = 16 samples in hundreds of dimensions are near-orthogonal noise | the literature baseline (Soft Tokens, Hard Truths: noise + RLOO) |
| **A2** | a default plus a low-dimensional steer: `z = f(h) + B(h)·u`, `u ∈ ℝ^m`, m ≈ 4–16 | an m-dimensional continuous action | asymptotic (EZ-V2) | `u = 0` is pure Coconut; search only nudges it; keeps Coconut's richness |
| **A3a** (rejected, Q1) | hard latent tokens: `z = e_a`, a from a learned codebook of N (64–256) | a categorical action | **exact** (Gumbel MuZero: at any simulation budget, provided the visited actions' values are evaluated correctly) | thoughts are decodable symbols of a private vocabulary |
| **A3b** (rejected, Q1) | soft latent tokens: `z = Σ_i π′_i e_i`, the SEARCH-IMPROVED distribution as mixture weights | a categorical action; the executed thought is the improved distribution itself | exact for the search; the thought superposes the branches | Soft Thinking's "concept token", produced by search; Coconut's "BFS in superposition", made explicit |
| **A4** (rejected, Q1) | codebook entry plus a continuous residual | categorical + continuous | mixed | later, if A3 is too coarse |

**Decided 2026-10-03 (the user, Q1): a codebook does not count as a continuous thought.** A3a, A3b and A4 are out.
The thought stays a free vector in ℝ^d (A1, A2), and the search has to work there — which makes the dimension of
the thought the central problem of the search. That problem is §10.

~~Proposed: build A3 first …~~ (withdrawn with Q1; the reasoning was that A3 gets Gumbel MuZero's exact
improvement guarantee at tiny simulation budgets — §10 is the attempt to get comparable leverage in continuous ℝ^d.)

## 4. The search

### 4.1 What the search must produce

At every state it visits: (i) an action to execute when collecting data; (ii) a **policy target** π′ (the per-step
target that replaces the curriculum); (iii) a **value target** v′; (iv) through the `answer` action's share of π′,
the **halting** signal. At test time it may also be run as a solver (§4.7).

### 4.2 Candidate algorithms

| | algorithm | per-step targets? | cost per problem (§4.8) | role |
|---|---|---|---|---|
| **S0** | no search: REINFORCE / RLOO / group-relative baseline, exploration by noise (A1) or sampling (A3) | no — one return for the whole chain | lowest | the baseline from the literature |
| **S1** | best-of-N + imitation (expert iteration / STaR-style): sample N chains, keep the verified ones, train on them | yes, but only along successful chains | N × T | the strongest simple baseline; self-generated, so allowed under R1 |
| **S2** | **root-only Gumbel** at every thought: Gumbel-top-k children, each evaluated by a rollout to the verdict (later: by V), Sequential Halving, π′ from completed Q | yes, at every visited state | n × T²/2 thought-steps (rollouts) | **the proposed first search** |
| **S3** | full sampled Gumbel tree (EZ-V2): halving at the root, policy-sampled children below, search-based value estimation | yes | n × T (once V is usable) | the scaled version once V is trustworthy |
| **S4** | S2/S3 with extra PROPOSALS: an inverse-model step toward a goal (GCML), a value-gradient step `z + η∇_z V` (A1/A2 only) | yes | as S2/S3 | proposal mixing, corrected per Sampled MuZero |
| **S5** | value-gradient refinement only (DDPG-like, A1/A2) | partial | low | dangerous alone — the policy exploits the critic's errors |

### 4.3 Proposed v0 — Gumbel thought search with exact stop values (S2 on A3)

*Superseded by §10.6 after Q1 (no codebooks). The skeleton — exact stop value, rollouts until V is calibrated,
Sequential Halving, π′ from completed Q, per-step targets — carries over unchanged; what changes is how the
candidate thoughts are generated (step 2) and what is executed (step 6).*

At a state s:

1. **Stop value, exact.** `Q(s, answer) = verifier(decode(s))`, one decode, cached. With partial credit (§5.2) it is
   graded; without, 0/1.
2. **Candidates.** Gumbel-top-k over `logits π(· | s)` for the continue-actions: k = 8–16 distinct codebook entries
   (sampling without replacement), plus `answer`.
3. **Evaluate continuing.** Each simulation runs one child one step (`s' = step(s, e_a)`) and then estimates its value:
   - *early training:* ONE rollout of the current policy from s' to its own `answer`, scored by the verifier — an
     unbiased Monte Carlo sample;
   - *once V is calibrated* (measured by V-vs-outcome on held-out problems): `V(s')`, or a short rollout plus V —
     EZ-V2's switch from multi-step to search-based targets, keyed here to a measured calibration rather than a
     step count.
4. **Sequential Halving** across the candidates with n = 8–32 simulations in total; unvisited candidates get the
   completed Q (V(s), or the policy-weighted mean of the visited Q's, as in Gumbel MuZero / EZ-V2 App. B).
5. **Outputs.** `π′ = softmax(logits + σ(completedQ))` over continue-actions AND `answer`; `v′` = the mean of the
   simulation returns (search-based value estimation); the executed action = the halving winner.
6. **Execute.** A3a feeds `e_{a*}`; A3b feeds `Σ_i π′_i e_i` (the improved distribution as the thought).

**Why this shape.** Root-only search with rollouts needs no trusted value function, which we will not have at the
start; the exact stop value removes half of what a value function would have to know; Sequential Halving with
Gumbel noise is what makes 8 simulations a valid policy improvement (Gumbel MuZero's guarantee, discrete case —
conditional on the visited actions' values being right, so with single noisy rollouts it holds in expectation, not
per search); and the per-step π′ is the dense target Coconut's curriculum used to provide.

### 4.4 The losses — what trains what

- **policy head:** `CE(π′, π)` at every searched state (plus EZ-V2's "simple loss" `−log π(a*)` as an option);
- **value head:** `(v′ − V(s))²`, or the categorical form;
- **answer head:** CE toward the known answer at stop positions (generated tasks know the answer — a differentiable
  outcome signal, allowed under R1);
- **the core (trunk):** through all three heads, AlphaZero-style; in A3b also through the thought feedback path
  (BPTT over a truncated window of thoughts);
- **exit gate for the depth axis** (if learned): Ouro's two stages, later (§4.6).

### 4.5 The continuous arm (A2) — now the main line, extended in §10

EZ-V2's sampling-based Gumbel search on the m-dimensional steer u: a squashed-Gaussian policy (mean through tanh,
std through softplus), root candidates from the policy AND a flattened prior (EZ-V2's `AS2`, for exploration),
policy-only samples below the root, `π′ = softmax(σ(completedQ))` over the sampled candidates, and the simple loss
for the high-dimensional case. Same stop values, same rollouts, same budget.

### 4.6 Halting on two axes

- **Sequence (how many thoughts):** `answer` is an action inside the search. A per-thought cost λ_T makes it win when
  `Q(continue) − λ_T < Q(answer)`. Hazard: λ_T from step 0 collapses to "answer immediately" before thinking has ever
  paid — start λ_T at 0, keep an entropy floor on `answer` vs continue (Ouro's stage-1 idea), raise λ_T later.
- **Depth (passes per thought):** fixed K while the search is being made to work (predictable cost per simulation);
  a learned exit (Ouro's two-stage gate) or a convergence test afterwards.

### 4.7 Training-time only, or test-time too?

AlphaZero's lesson is that the policy network alone eventually plays strongly: search can be AMORTISED into the
policy. Report both: the amortised policy (no search at test time) and the policy plus test-time search at a
stated budget. If the amortised policy matches the searched one, the model has internalised the search — which is
Coconut's "BFS in superposition" claim, now measurable. (For ARC on Kaggle, test-time search is allowed and the
budget is a GPU-hours cap, so the searched number is the one that would be submitted.)

### 4.8 Cost model — honest numbers

Unit: one thought-step = one position through the looped core (K passes, incremental with a cache or Spotlight).
- S0 / plain sampling: T steps per problem.
- S1 (best-of-N): N·T.
- S2 with rollouts, searching at every thought: about n·T²/2 — e.g. n = 16, T = 8: ~576 thought-steps, ~70× plain
  sampling. With V bootstrapping instead of rollouts: n·T = 128, ~16×.
- Reanalyse adds a re-search of stored problems with the target network.
Implications: get V calibrated early (it turns T² into T); search only at a subset of thoughts (e.g. uncertain ones —
where π is high-entropy); keep models small enough that 70× is affordable on a 3050 Ti; and compare every arm at an
equal number of thought-steps (R5).

## 5. Exploration — getting the first success

With no traces, the first chains are random. Search amplifies a weak policy (it finds successes sampling would
miss), but if the success probability is ~0, eight simulations find nothing either. Four levers, all outcome-based:

### 5.1 A problem-difficulty frontier (not a trace curriculum)
A generator with a difficulty knob (hops in pointer chasing; program steps per output byte in Brainfuck), sampled
where the current success rate is near 50% — prioritised level replay / a self-paced frontier, or the self-play
generator of 2609.30063 scored by the learner's progress. Start just above the no-thought ceiling (R6), where ONE
thought suffices.

### 5.2 Partial credit
Score answers by degree: per-digit or per-byte correctness, edit distance to the answer, fraction of output bytes
right. Densifies the signal without saying anything about HOW to think. Risk: rewards plausible-looking wrong
answers; keep the exact-match term dominant.

### 5.3 Hindsight relabelling (speculative)
A failed chain still computed SOMETHING. In a parameterised task family the generator can often name the problem
whose answer it computed — "you returned the 3-hop answer to a 5-hop question" — and the chain becomes a SUCCESS for
that relabelled problem (HER for reasoning). Outcome-based and allowed; produces successes long before the first
real one. Only works where the family is closed under such relabelling.

### 5.4 GCML as a cheap default policy for rollouts
Rollouts dominate S2's cost. A fast default — GCML-style "move the state toward the goal" via a learned inverse
model, where a goal state exists (games) — can stand in for the full policy during rollouts, as AlphaGo's fast
rollout policy did. Noise in it makes it a sampler of diverse rollouts.

## 6. Stability and collapse, specific to search

The general table is `BRAINSTORM.md` §3.6. What search adds:
- **Proposal collapse** — the policy's covariance shrinks (or its low-rank subspace locks onto a few directions)
  and every candidate is the same thought. Guard: a σ floor and EZ-V2's widened-prior candidates (§10.6);
  detect by the spread of the candidates.
- **Stop collapse** — `answer` always wins before thinking has paid (§4.6).
- **Value exploitation** — once V replaces rollouts, the search maximises V's errors. Guard: keep a fraction of
  rollouts always (MC grounding), and track V-vs-outcome calibration.
- **Moving dynamics** — the core changes under the search's feet. Guard: search with a target network; reanalyse.
- **Thought ignoring** — the model solves easy instances without using its thoughts and never learns to. Guard: R6,
  and a thought-ablation check on every evaluation (zero or shuffle the thoughts; accuracy must fall).

## 7. Experiments, in order (proposals; each gets a pass and a refute before it runs)

- **P0 — the task family and its no-thought ceiling.** Pointer chasing / multi-hop with h hops (and a Brainfuck
  family, reusing the E46 generator in `experiments/ziplearn/e46.py` as code); train a no-thought looped model of
  the intended size and find h* where it fails. Establishes R6.
- **P1 — the baselines, at equal thought-steps:** (a) no thoughts; (b) **BPTT-only** — Coconut thoughts trained by
  the answer's cross-entropy alone, no curriculum: Coconut's own negative result, which must be reproduced here
  before anything is claimed against it; (c) S0 with noise (Soft Tokens, Hard Truths); (d) S1 (best-of-N + distil).
- **P2 — the §10.6 search** (and its ablations, §10.6). Pass: solves hop counts the P1 arms cannot at the same budget, and the thought
  ablation destroys it. Refute: no better than S1.
- **P3 — S3 with search-based values; population search (§10.4 D5) as the alternative to the tree.**
- **P4 — halting learned** (both axes); λ_T schedule.
- **P5 — amortised vs test-time search** (§4.7).
Metrics everywhere: success vs difficulty at equal thought-steps; thought ablation; candidate spread;
V calibration; thoughts per problem against difficulty (does thinking grow with need?).

## 8. Interplay with memory (Spotlight) and depth routing

- Search state forking: §2 item 5. Spotlight makes it cheap; attention needs a tree-structured cache.
- With Spotlight, a thought need not be an appended position at all: it can be a WRITE into memory that later
  thoughts read — the decision process's state then lives in memory, and the action could include where to write.
  Out of scope for v0; recorded because it changes §2's state definition.
- Depth routing (Block AttnRes vs the boundary operator) is `BRAINSTORM.md` §1; the thought fed back along the
  sequence axis needs its own normaliser regardless.
- With Spotlight, the read/write ADDRESSES are low-dimensional continuous controls with large consequences — a
  natural thing for the search to vary (§10.4, D4).

## 9. Open questions for discussion

- **Q1 — Discrete codebook or truly continuous?** **Answered 2026-10-03: a codebook is not a continuous thought.**
  The search must work in ℝ^d — §10.
- **Q2 — Is search needed at test time, or only for training?** (§4.7.)
- **Q3 — Does the answer's cross-entropy alone suffice in our setting?** Coconut says no without the curriculum.
  Unknown (2026-10-03) — P1(b) answers it. If it DOES suffice on these tasks, the search is a speed-up, not a
  necessity.
- **Q4 — Partial credit: allowed?** (§5.2.)
- **Q5 — A problem-difficulty curriculum: allowed?** (§5.1.) It is not a substitution curriculum, but it is a
  curriculum.
- **Q6 — Machine-state auxiliaries** (e.g. "the thought should let a probe read the tape"): allowed as training
  signals, or probes only?
- **Q7 — Who proposes candidates?** The policy alone, or also GCML-style goal-directed proposals and value-gradient
  steps (S4)? Proposal mixing needs Sampled MuZero's correction to stay a valid improvement.
- **Q8 — Where does search run?** At every thought (expensive, dense targets) or only where the policy is uncertain
  (cheap, sparse targets)?

---

## 10. The dimensionality problem — search in a continuous thought space (brainstorm, 2026-10-03)

With codebooks ruled out (Q1), the search acts on a thought z ∈ ℝ^d with d = 64 … 1024. This section asks whether
that breaks EZ-V2-style search, why, and what could replace it.

### 10.1 Yes — dimensionality was EZ-V2's problem too

- The paper names it: EfficientZero "is unable to handle high-dimensional action spaces, especially in continuous
  control"; one of EZ-V2's two "pivotal questions" is "how to perform efficient planning using tree search in
  high-dimensional and continuous action spaces" (§4.1).
- Both of its search remedies are dimension remedies: the widened prior `AS2` "given the challenges posed by
  high-dimensional and large continuous action spaces" (§4.2), and the simple loss −log p(a*) "in high-dimensional
  action spaces" / "tasks with a large action dimension, such as Quadruped walk" (eq 9, App. C).
- **The largest action space it demonstrates is 12-dimensional.** Actuator counts from DeepMind Control's own model
  files: cartpole, pendulum, acrobot 1; reacher, finger, ball-in-cup 2; hopper 4; cheetah, walker 6; quadruped 12.
  Humanoid (21) and dog (38) are not in its benchmark.
- Its ablation shows search quality matters most exactly there: swapping its search for Sample MuZero's drops
  Quadruped Walk 925.8 → 254.7 (proprio) and 433.3 → 141.4 (vision), Walker Run 657.2 → 381.1, while 1-D Cartpole
  Swingup Sparse barely moves (795.4 → 789.3) (App. J.2, Tables 4–5). Quadruped is also where more simulations still
  help (Fig. 3) and where it trails TD-MPC2 (Quadruped Run 510.6 vs 742.1).
- A thought is 5–85× wider than anything EZ-V2 searched.

### 10.2 Why — one line of maths, and a toy that confirms it

For a smooth value, locally `Q(μ + δ) ≈ Q(μ) + gᵀδ`. Let a step move at most a distance r (a trust region — §10.5
says why one is needed). The gradient step gains `r‖g‖`. A random direction gains `r‖g‖·N(0,1)/√d`, so the best of
K random candidates gains about `r‖g‖·E[max of K normals]/√d` ≈ `1.77·r‖g‖/√d` at K = 16: **the fraction of the
available improvement that sampling recovers falls as 1/√d, and more samples help only as √(ln K).** Over many
iterations, zeroth-order (random-direction) methods need up to d times more iterations than gradient methods
(Nesterov & Spokoiny 2017).

`research/dim_search_toy.py` (numpy, seconds; output `research/dim_search_toy.json`): fraction of the best step
recovered, trust radius fixed, 4,000 trials per cell; the value depends on an m = 8-dimensional subspace of z.

| d | random, best of 16 | random, best of 256 | random within the right 8-D subspace, best of 16 | one step along a gradient with cosine 0.3 to the truth | needle hit rate, best of 16 |
|---|---|---|---|---|---|
| 2 | 0.97 | 1.00 | 0.97 | 0.30 | 0.94 |
| 12 (EZ-V2's largest) | 0.50 | 0.73 | 0.60 | 0.30 | 0.00075 |
| 64 | 0.22 | 0.35 | 0.60 | 0.30 | 0 |
| 256 | 0.11 | 0.18 | 0.60 | 0.30 | — |
| 1024 | 0.055 | 0.088 | 0.60 | 0.30 | — |

Readings: (1) at 12-D random search still recovers half a step — which is why EZ-V2 works where it was tested;
(2) at thought dimensions it recovers 5–11%, and 16× the samples buys almost nothing; (3) searching the RIGHT
subspace makes d irrelevant — only m matters; (4) a gradient that is mostly wrong (cosine 0.3) beats 256 random
samples once d ≳ 256; (5) when success is a needle (sparse reward, a narrow region of good thoughts) random search
fails exponentially — at 12-D, 16 samples hit a target of radius r/2 0.075% of the time.

### 10.3 What thoughts have that EZ-V2's actions do not

(a) **Differentiable, exact dynamics** — the next state is the network's own forward pass, so `∂(anything downstream)
/ ∂z` costs one backward pass; only V is learned. (b) **An exact stop value** at every node. (c) **A strong default** —
the deterministic Coconut state f(h). (d) **Computable sensitivity** — the Jacobian of the future with respect to z
says which directions matter. (e) **Exact rollouts.** The curse of §10.2 is a ZERO-ORDER curse; a thought search can
use first-order information, which a black-box environment never offers.

### 10.4 Ways out — every one keeps the thought a free vector in ℝ^d

| | idea | how it beats d | cost | main risk | refs |
|---|---|---|---|---|---|
| **D0** | EZ-V2 as published: isotropic Gaussian candidates + widened prior | it does not (§10.2) | K evaluations | recovers 5–11% of a step | 2403.00564 |
| **D1** | **propose by gradient, dispose by verifier**: candidates along ∇_z V(step(s, z)) at a few step sizes (a line search), each verified by an exact rollout | the direction costs one backward pass in any d | +1 backward per node | V's gradient is adversarial in high dimension (adversarial examples are a high-dimensional phenomenon); verification keeps bad candidates out but cannot make a wrong gradient right | SVG 1510.09142; GradCEM 2004.08763; Guided ES 1806.10230 |
| **D2** | **sample the improved policy with gradients**: π′ ∝ π·exp(Q/τ) — the object Gumbel's completed-Q transform and MPO's E-step target — sampled by a few Langevin steps `z ← z + η(∇log π + ∇Q/τ) + √(2η)·ξ` from the policy mean; Sequential Halving over the samples | gradient-driven moves | a few backward passes per candidate | step-size tuning; same V-gradient risk as D1 | MPO 1806.06920; AWR 1910.00177; Diffusion-QL 2208.06193 |
| **D3** | **search a learned low-dimensional subspace**: a Gaussian policy with covariance `B·Bᵀ + σ₀²I`, `B ∈ ℝ^{d×m}`; exploration is effectively m-dimensional (toy: 0.60 at every d, m = 8) | d → m | as D0 | the subspace is wrong; keep a few full-d candidates to correct it | intrinsic dimension 1804.08838; CMA-ES 1604.00772; Guided ES 1806.10230 |
| **D4** | **search a few CONTROLS, not the thought**: the thought stays the deterministic Coconut state; the search varies a small set of continuous controls that steer the computation — with Spotlight, the 2-D read/write ADDRESSES of a designated head ("search where to look, not what to think"); the depth K of this thought; a low-dimensional intent vector u fed into the core | d → (number of controls) | as D0 | the model learns to ignore the controls — measure ∂outcome/∂u | NTM 1410.5401; Spotlight (refs/) |
| **D5** | **population search over whole chains (SMC)**: N particles, each extended by the policy (or D2's proposals), reweighted by a learned twist ≈ exp(V/τ), resampled; the verifier weights the end | no per-dimension search at all; d shows up only as weight degeneracy when proposals are poor | N × T thought-steps, GPU-parallel | weight collapse; needs good proposals | Twisted SMC 2404.17546 |
| **D6** | **iterated population per thought (CEM / iCEM / MPPI)**, TD-MPC2's planner | adapts its sampling distribution over rounds | high — TD-MPC2 predicts 9,216 latent states per decision (EZ-V2 App. J.3) | cost | iCEM 2008.06389; TD-MPC2 2310.16828 |

Where the ideas sit: D1/D2 use the GRADIENT the environment gives us for free; D3/D4 shrink the SPACE searched;
D5 changes the UNIT of search from a step to a whole chain (and is the closest to Coconut's "a frontier of
hypotheses in superposition", made explicit as a population); D6 is the expensive baseline.

D5 is also the natural partner of Spotlight: every particle carries a copy-on-write memory overlay, and resampling
copies overlays, not caches.

### 10.5 The trap a gradient opens: answer-encoding thoughts

At training time the answer y* is known, so `∇_z log p(y* | …)` is an exact, unlearned gradient — tempting as a
proposal. But a free vector can simply ENCODE the answer: the thought that best satisfies that gradient is "write y*
into z", which no policy can produce without already knowing y*. As a target it is unreachable, and chasing it
teaches nothing about how to get there. Guards: (a) bound every improvement in KL around the current policy (MPO's ε;
Gumbel's σ-transform of completed Q does this implicitly), so targets stay reachable from where the policy is;
(b) at search time prefer gradients of V, which never sees y*; use answer-gradients, if at all, only as a
training-time proposal inside the trust region; (c) the thought ablation and R6 catch a policy that answers by
pattern-matching instead of thinking.

### 10.6 A revised proposal (v1) — continuous candidates on the v0 skeleton

The skeleton of §4.3 stays: exact stop value; rollouts until V is calibrated, V afterwards; Sequential Halving over
the candidates; `π′` from completed Q; execute the winner. Candidate generation changes — K ≈ 8–16 per node:
- the **default** thought (the deterministic Coconut state; u = 0);
- 2–3 **gradient** candidates along `∇_z V(step(s, z))` at different step sizes (D1), or Langevin samples (D2);
- samples from the **low-rank Gaussian** (D3, m ≈ 8);
- 1–2 from a **widened prior in all d** (EZ-V2's `AS2`), so a wrong subspace can still be corrected.

Policy update: weighted maximum likelihood of the Gaussian (mean and low-rank covariance) on the candidates, with
weights `softmax(σ(completedQ))`, importance-corrected for the mixture of proposals (Sampled MuZero), KL-bounded
(MPO) — plus EZ-V2's simple loss toward the winner. The covariance update is what LEARNS the subspace: directions
that keep paying get variance, the rest shrink (CMA-ES's idea, in policy form).

Arms, at equal thought-steps: D0; v1; v1 without gradients; v1 without the subspace; D5; S1 (best-of-N + distil);
S0 (RLOO with noise — Soft Tokens, Hard Truths).

### 10.7 Cheap checks before building (CPU)

- **E-dim1 — the toy.** Done (§10.2).
- **E-dim2 — the effective dimension of a real thought.** On a looped model (trained, and at initialisation), the
  singular-value spectrum of `∂(answer logits k steps later) / ∂z`. Energy in m ≪ d directions justifies D3/D4; a
  flat spectrum leaves gradients (D1/D2) as the only lever.
- **E-dim3 — how good is ∇_z V̂?** Cosine between the learned value's gradient and a finite-difference gradient of
  the true rollout value, tracked over training. The toy's bar: cosine ≈ 0.3 already beats 256 random samples at
  d ≳ 256.

### 10.8 Questions for the user — and where they stand (2026-10-04)

- Is D4 — the thought deterministic and continuous, the search varying only a few controls (e.g. where Spotlight
  reads and writes) — "continuous thought" in your sense? *The user: it could limit the model.* §11.4 point 7 agrees:
  a search confined to a subspace it did not learn fails from d = 64; D4 survives only as a learned subspace that the
  search can still leave — which is what `mcts_hybrid` is.
- Population (D5) or tree (v1)? *The user: tree seems more likely.* §11: a real tree (with backtracking) is the best
  family from budget ≈ 512 up — provided its candidates come from gradients; with EZ-V2's sampled candidates it fails.
- Answer-gradients at training time? *The user: experiment.* Planned in tier 2 (§11.6).


---

## 11. Which search works? — a CPU benchmark on a mock task (run 2026-10-04)

**What it answers:** at an equal compute budget, which search over continuous thoughts finds a solution — TEST-TIME
search on fixed dynamics. It does not yet answer which search gives the best TRAINING signal (§11.6).
Code: `search_bench/` — `mock_task.py` (the task), `searchers.py` (the algorithms), `run_bench.py` (the grid),
`summarize.py` (these tables); results in `search_bench/runs/*.json`. CPU only (PyTorch for gradients), about
15 minutes for every grid below on 4 cores.

### 11.1 The mock task

A **thought-space graph walk** with exactly the properties of §2 (details in `mock_task.py`'s docstring). The state is
a probability distribution over the 64 nodes of a random graph (4 out-edges each) — continuous, able to hold several
branches at once. A thought z ∈ ℝ^d is normalised and acts only through a hidden 8-dimensional projection; at each
node, an out-edge is taken in proportion to softmax(β·key·u) against a STAY option, so a thought must POINT at the
right edge's key to move. The target is 4 edges away, the horizon 8 thoughts; the verifier (argmax node = target, with
more than half the mass) is checked at every simulated state, which is the exact stop value. A learned value is mocked
as a per-node table γ^distance plus noise. Knobs:
- **regime** — *loose* (β 16, gate 0.3): finding the right DIRECTION is the problem; *tight* (β 8, gate 0.5): a thought
  must be PRECISE, or mass leaks and the verifier fails four steps later;
- **d** — 16 … 1024, effective dimension fixed at 8;
- **value noise** 0.1 / 0.3; **value exploitable** — a spurious value term driven by thought directions the verifier
  ignores (the risk of following a learned value's gradient); **prior** — untrained (isotropic) or partly trained
  (cosine 0.3 to the right thought).
Budget = thought-steps (a gradient through a step costs 3). Instances are shared across methods (paired comparison).
A sanity check confirms every instance is solved by the right thoughts in exactly 4 steps.

### 11.2 The methods (15 + 3 references)

- *Sampling only:* `random_shooting` (best-of-N chains), `cem` (cross-entropy method, TD-MPC2's family), `smc`
  (particles twisted by V), `look_iso` (EZ-V2's candidates, one-step lookahead), `mcts_iso` (EZ-V2's candidates in a
  real tree), `look_randsub` (sampling confined to a FIXED random 8-dim subspace — §10.4 D4's failure mode).
- *Gradient-using:* `grad_greedy` (one value-gradient step per thought, restarted), `smc_grad`, `grad_trajopt`
  (Adam on the whole chain through the exact dynamics), `look_grad` / `look_langevin` / `look_guided` (one-step
  lookahead with gradient, Langevin-chain or learned-subspace candidates), and the real tree (best-first with
  progressive widening and backtracking) with four expansion rules: `mcts_grad`, `mcts_langevin`, `mcts_guided`
  (samples in the subspace spanned by the tree's own recent value gradients), `mcts_hybrid` (learned-subspace samples,
  each refined by two noisy gradient steps).
- *Reference:* `look_oracle` — sampling in the TRUE effective subspace.

### 11.3 Results — success rate (fraction of instances solved)

**Dimension** (value noise 0.1, untrained prior; budget 512 / 4096 thought-steps; 60 instances per cell):

| method | loose d=16 | loose d=64 | loose d=256 | loose d=1024 | tight d=16 | tight d=64 | tight d=256 | tight d=1024 |
|---|---|---|---|---|---|---|---|---|
| `random_shooting` | 0.05 / 0.10 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `cem` | 0.05 / 0.78 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `smc` | 0.35 / 0.95 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `look_iso` | 0.30 / 0.77 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `mcts_iso` | 0.28 / 0.97 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `look_randsub` | 0.35 / 0.70 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `look_oracle` | 0.68 / 0.85 | 0.53 / 0.72 | 0.52 / 0.58 | 0.32 / 0.57 | 0.03 / 0.32 | 0.00 / 0.28 | 0.00 / 0.30 | 0.00 / 0.18 |
| `grad_greedy` | 0.92 / 1.00 | 0.87 / 0.97 | 0.73 / 0.90 | 0.62 / 0.82 | 0.35 / 0.72 | 0.32 / 0.63 | 0.13 / 0.33 | 0.02 / 0.03 |
| `smc_grad` | 0.97 / 1.00 | 0.88 / 0.95 | 0.73 / 0.88 | 0.47 / 0.70 | 0.73 / 0.88 | 0.33 / 0.73 | 0.13 / 0.33 | 0.00 / 0.03 |
| `grad_trajopt` | 0.80 / 0.98 | 0.70 / 0.97 | 0.37 / 0.80 | 0.32 / 0.68 | 0.37 / 0.93 | 0.17 / 0.85 | 0.10 / 0.60 | 0.03 / 0.58 |
| `look_langevin` | 0.62 / 0.68 | 0.55 / 0.58 | 0.47 / 0.48 | 0.40 / 0.43 | 0.60 / 0.63 | 0.45 / 0.47 | 0.42 / 0.45 | 0.27 / 0.32 |
| `mcts_grad` | 0.87 / 0.98 | 0.67 / 0.93 | 0.47 / 0.87 | 0.32 / 0.63 | 0.25 / 0.78 | 0.20 / 0.62 | 0.08 / 0.18 | 0.00 / 0.02 |
| `mcts_langevin` | 0.75 / 0.95 | 0.72 / 0.93 | 0.62 / 0.82 | 0.43 / 0.58 | 0.68 / 0.80 | 0.60 / 0.83 | 0.45 / 0.65 | 0.30 / 0.42 |
| `mcts_guided` | 0.93 / 1.00 | 0.85 / 1.00 | 0.90 / 1.00 | 0.80 / 1.00 | 0.18 / 0.58 | 0.28 / 0.60 | 0.17 / 0.52 | 0.12 / 0.38 |
| `mcts_hybrid` | 0.82 / 1.00 | 0.80 / 0.98 | 0.73 / 0.98 | 0.68 / 0.93 | 0.58 / 0.92 | 0.63 / 0.93 | 0.55 / 0.88 | 0.45 / 0.82 |

**Robustness at d = 256** (budget 512 / 4096; 40 instances per cell):

| method | loose: baseline | loose: exploitable V | loose: prior cos 0.3 | loose: V noise 0.3 | loose: V noise 0.3 + prior | tight: baseline | tight: exploitable V | tight: prior cos 0.3 | tight: V noise 0.3 | tight: V noise 0.3 + prior |
|---|---|---|---|---|---|---|---|---|---|---|
| `look_iso` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `mcts_iso` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| `grad_greedy` | 0.75 / 0.93 | 0.53 / 0.85 | 0.75 / 0.75 | 0.30 / 0.53 | 0.17 / 0.17 | 0.15 / 0.35 | 0.03 / 0.07 | 0.40 / 0.40 | 0.00 / 0.00 | 0.03 / 0.03 |
| `smc_grad` | 0.72 / 0.88 | 0.60 / 0.82 | 0.82 / 0.93 | 0.07 / 0.10 | 0.15 / 0.17 | 0.07 / 0.28 | 0.03 / 0.05 | 0.50 / 0.55 | 0.00 / 0.00 | 0.03 / 0.03 |
| `grad_trajopt` | 0.33 / 0.80 | 0.03 / 0.57 | 0.33 / 0.80 | 0.03 / 0.12 | 0.03 / 0.12 | 0.15 / 0.62 | 0.00 / 0.15 | 0.15 / 0.62 | 0.00 / 0.03 | 0.00 / 0.03 |
| `mcts_langevin` | 0.57 / 0.82 | 0.55 / 0.88 | 0.57 / 0.72 | 0.03 / 0.07 | 0.07 / 0.15 | 0.42 / 0.65 | 0.15 / 0.38 | 0.45 / 0.57 | 0.00 / 0.00 | 0.00 / 0.03 |
| `mcts_guided` | 0.85 / 1.00 | 0.55 / 0.88 | 1.00 / 1.00 | 0.15 / 0.60 | 0.42 / 0.75 | 0.15 / 0.53 | 0.00 / 0.00 | 0.62 / 0.88 | 0.05 / 0.05 | 0.07 / 0.30 |
| `mcts_hybrid` | 0.70 / 1.00 | 0.40 / 0.95 | 0.75 / 0.97 | 0.07 / 0.35 | 0.10 / 0.33 | 0.53 / 0.90 | 0.07 / 0.53 | 0.55 / 0.85 | 0.00 / 0.10 | 0.07 / 0.12 |

**Budget at d = 256** (value noise 0.1, untrained prior; 40 instances per cell):

| method | loose B=128 | loose B=256 | loose B=512 | loose B=1024 | loose B=2048 | tight B=128 | tight B=256 | tight B=512 | tight B=1024 | tight B=2048 |
|---|---|---|---|---|---|---|---|---|---|---|
| `grad_greedy` | 0.45 | 0.70 | 0.75 | 0.82 | 0.90 | 0.10 | 0.12 | 0.15 | 0.23 | 0.30 |
| `smc_grad` | 0.42 | 0.62 | 0.72 | 0.80 | 0.80 | 0.00 | 0.03 | 0.07 | 0.20 | 0.25 |
| `grad_trajopt` | 0.00 | 0.12 | 0.33 | 0.60 | 0.65 | 0.00 | 0.00 | 0.15 | 0.33 | 0.53 |
| `mcts_langevin` | 0.28 | 0.35 | 0.57 | 0.70 | 0.78 | 0.17 | 0.25 | 0.42 | 0.53 | 0.57 |
| `mcts_guided` | 0.35 | 0.50 | 0.85 | 0.97 | 1.00 | 0.05 | 0.07 | 0.15 | 0.35 | 0.38 |
| `mcts_hybrid` | 0.17 | 0.57 | 0.70 | 0.95 | 1.00 | 0.15 | 0.38 | 0.53 | 0.78 | 0.88 |

### 11.4 What it says

1. **Sampling-only search fails from d = 64 up — in every condition, tree or no tree, trained prior or not.**
   EZ-V2's candidate generation does not transfer to thought-sized vectors: `look_iso` and `mcts_iso` score 0.00 in
   every cell from d = 64, as do random shooting, CEM and plain SMC. §10.2's prediction, now on a task with sparse
   reward, multi-step structure and a noisy value.
2. **Gradient information is the necessary ingredient.** Every method that works takes ∇_z V through the exact
   dynamics. Even sampling in the TRUE effective subspace without gradients (`look_oracle`) stays below the
   gradient-using trees (d = 256: loose 0.52 / 0.58 against 0.62–0.90 / 0.82–1.00; tight 0.00 / 0.30).
3. **A real tree beats one-step lookahead with the same candidates** — backtracking matters once the value is noisy:
   loose d = 256, `mcts_guided` 0.90 / 1.00 vs `look_guided` 0.50 / 0.65; tight d = 256, `mcts_langevin` 0.45 / 0.65 vs
   `look_langevin` 0.42 / 0.45.
4. **The best candidate generator depends on what the hard part is.** Direction-finding (loose): the LEARNED
   SUBSPACE (`mcts_guided`) — 0.80–0.93 at budget 512 and 1.00 at 4096 at every d up to 1024. Precision (tight):
   gradient REFINEMENT (`mcts_langevin`, `mcts_hybrid`). **`mcts_hybrid` is the best all-rounder**: the best method in
   the tight regime from d = 64 to 1024 (0.55 / 0.88 at d = 256, 0.45 / 0.82 at d = 1024), and in the loose one
   0.68–0.82 at budget 512 and 0.93–1.00 at 4096.
5. **At very small budgets, simple wins.** At 128 thought-steps (loose), restarted gradient-greedy chains (0.45) beat
   every tree; trees need roughly 512 to pay for their bookkeeping. Training-time budgets will sit in this range.
6. **The value's quality dominates everything.** At value noise 0.3 the tight regime collapses for every method
   (≤ 0.12 except `mcts_guided` with a trained prior, 0.30). An exploitable value hurts whole-chain gradient
   optimisation most (tight, budget 4096: 0.62 → 0.15); trees, which verify every simulated state exactly, recover
   more (`mcts_hybrid` 0.90 → 0.53). No search rescues a bad value — calibrating V is as important as the search.
7. **Confining the search to a subspace it did not learn fails** (`look_randsub`: 0.00 from d = 64). This is the risk in
   "keep the thought deterministic and search only a few controls" (§10.4 D4): the restriction is safe only when the
   subspace is LEARNED from the problem and the search can still leave it.

### 11.5 Caveats — what this mock does not show

The dynamics are fixed and known (nothing is trained); the low-dimensional structure is planted, and real thoughts
may have none; the value is a noisy table with one planted exploit, not a network that off-distribution states can
fool in unplanned ways; one task family; every method's constants (step sizes, UCT bonus, widening) were set once and
not tuned per method, so close cells could reorder under tuning; 40–60 instances per cell (standard error about
0.06–0.08), so differences under ~0.15 are not reliable. The budgets (512, 4096) are test-time scale; training will
have less.

### 11.6 Next — the learning loop (tier 2)

The question that matters for training is not "which search finds a solution" but "which search gives the policy
the most useful targets". Tier 2: train a small policy (and a LEARNED value, replacing the mocked one) on many
instances of the same mock, with each search as the improvement operator (`π′` targets as in §4.4), and measure,
at equal compute: the policy's success WITHOUT search (amortisation), with search, and the learning speed. Candidates:
`mcts_hybrid`, `mcts_guided`, `grad_greedy` (the cheap-budget winner), `smc_grad`, and S1 (best-of-N + distil) and
S0 (RLOO with noise) as the literature baselines. It also gives Q3 an arena: an answer-gradient arm (§10.5) inside a
trust region against value-gradients only.

---

## 12. Testing the GCML-inspired component — plan (2026-10-04; RAN the same day, results in §13)

**What the component is.** From GCML (`src/tbt/notes/gcml_neural_sampling_cognitive_maps.md`): an INVERSE model W
that maps a wanted state change to the action that produces it, `z = W(s* − s) + ε`, learned by a local Hebbian rule
from observed transitions (`ΔW = η·z·(s' − s)ᵀ`). In this document it has three proposed roles: (a) a goal-directed
PROPOSAL inside the tree (S4); (b) a cheap ROLLOUT policy — "imagination" without search (§5.4); (c) a search-free
PLANNER, GCML's own use (iterate the forward model, W choosing each step, noise for diversity).

**The test has to be built around its assumptions**, because each can fail in thought space:
1. *State-invariant action effects.* GCML's world is (nearly) linear: the action's effect `V·a` is the same everywhere
   (eq 11). A thought's effect on a network's state depends on the state. The mock's current graph is the worst case
   (every node's edges have their own keys), so a single linear W cannot work there BY CONSTRUCTION.
2. *A state code in which subtraction means something.* `s* − s` must point along the way to the goal (grid cells
   in GCML's spatial case). Our raw state (a distribution over nodes) has no such geometry.
3. *A known goal state s*.* Games may supply one; in reasoning the goal is the answer, which is unknown.
4. *Data to learn W from.* GCML learns from random exploration with a few discrete actions. In high d a random thought
   mostly does nothing (§10.2, §11), so `s' − s ≈ 0` and Hebbian learning gets no signal — the dimensional curse
   again, this time on learning the inverse model rather than on searching.

**What the component would buy if its assumptions hold: cost.** A proposal `W(s* − s)` costs a matrix-vector product;
a value-gradient proposal costs a backward pass through the core (in the mock 3 thought-steps; in a looped
transformer, a backward pass through K passes × L layers). §11.4 point 5 says small budgets matter, so cheap
proposals could matter.

### 12.1 Three directions to a better thought — how they relate

With J = ∂s'/∂z the local action Jacobian of the dynamics:
- the **value gradient** (what §11's winners use): `∇_z V = Jᵀ ∇_s V` — needs a backward pass through the dynamics;
- the **exact local inverse** (Gauss-Newton toward a goal): `z ∝ J⁺ (s* − s)` — needs the Jacobian;
- **GCML**: `z ∝ W (s* − s)`, W a LEARNED, state-independent stand-in for J⁺ — needs nothing at use time;
- **GCML + value** (no goal state needed): `z ∝ W ∇_s V` — the value's gradient in STATE space (cheap: no backprop
  through the core), mapped to a thought by W. It answers assumption 3;
- a **state-conditioned inverse** `W(s)·(s* − s)` (a small network) — GCML generalised past assumption 1.

### 12.2 The mock, with a structure dial (assumptions 1–2)

- **V0** — today's random graph, keys per node: no shared structure (control: GCML should fail).
- **V1** — random graph, GLOBAL edge keys (k action types shared by every node: "the same thought does the same
  operation everywhere"); destinations still arbitrary. State code: raw (a distribution over nodes) or an
  **SR / Laplacian-eigenmap embedding** of the graph — the successor-representation frame of the old TBT column —
  under which `s* − s` points along paths.
- **V2** — a torus grid, global N/E/S/W keys, coordinate (grid-cell-like Fourier) state code: GCML's own world, the
  positive control.
- A dial κ between them: global keys plus per-node perturbations of size κ.

### 12.3 The tests

- **G0 — probe the precondition (no learning).** "Action-effect invariance": the spread of J = ∂s'/∂z across states,
  `mean ‖J_i − J̄‖ / ‖J̄‖`, per variant. The same probe can later be run on a real looped model (beside E-dim2, §10.7)
  — it would say, before anything is built, whether GCML can work there.
- **G1 — can W be learned, and from what data?** Hebbian (GCML eq 14) and ridge regression, from N transitions whose
  thoughts come from (a) random exploration, (b) the thoughts the tree search executed on OTHER instances, (c) oracle
  demonstrations (upper bound). Measured against N: cosine between `W(s* − s)` and the right thought; one-step success.
  *Expected:* (a) fails in high d (no signal); (b) works where the structure exists — the search bootstrapping its
  own inverse model.
- **G2 — direction quality at equal cost (W fixed, no search).** Value gradient, exact `J⁺`, `W(s* − s)`,
  `W ∇_s V`, `W(s)`: cosine with the right thought, greedy multi-step success, cost per proposal. Across V0/V1/V2 and
  d ∈ {64, 256}.
- **G3 — inside the tree.** New expansion rules `mcts_gcml` (`W(s* − s) + ε` samples) and `mcts_gcml_v`
  (`W ∇_s V + ε`), GCML rollouts in place of prior rollouts, and a mixture with the hybrid. Budget sweep 64–2048 —
  the regime where cheap proposals should show.
- **G4 — GCML's own claim: the search-free planner.** Iterate `z_t = W(s* − s_t) + ε` with restarts, against
  `grad_greedy` and `mcts_hybrid` at equal budget.

**Pre-registered expectations.** V2: W learns from search-generated or even random data (d permitting), its proposals
come close to the value gradient's quality at a fraction of the cost, and GCML-in-the-tree wins at budgets ≤ 512. V1:
works only with the SR embedding. V0: fails (control). **Refuted if** GCML does not beat the value gradient at EQUAL
COST in V2 — then its cost advantage does not survive even in its own world. **Decisive for the real model:** G0's
invariance on a trained looped network. If a thought's effect varies as much as V0's, a state-independent W is out,
and only `W(s)` (or gradients) remain.


---

## 13. GCML tests — results (run 2026-10-04)

**Setup.** §12's plan on the loose regime (value noise 0.1, untrained prior), d ∈ {64, 256}, 8 worlds per (variant,
d) = 80 worlds. Per world: 16 TRAINING problems give the search experience (the `mcts_hybrid` search at budget 512,
every simulated transition recorded); 4,096 random-thought and 4,096 demonstration transitions; nine inverse models
(random / demonstration / search experience × Hebbian / ridge / state-conditioned network); 5 HELD-OUT problems for
G2 and G3/G4 (40 per cell). Code: `search_bench/structured_task.py` (the structure dial), `gcml.py` (inverse models,
deltas, measurements), `searchers_gcml.py` (the GCML search algorithms), `run_gcml.py`; follow-up probes
`probe_gcml.py`, `probe_kstep.py`; tables `summarize_gcml.py`; data `search_bench/runs/gcml_*.json`. About 20 CPU
minutes in all.

### 13.1 G2 — how good is the direction? (one-step progress from every node of the held-out problems)

| proposal | V0 d=64 | V0 d=256 | V1raw d=64 | V1raw d=256 | V1sr d=64 | V1sr d=256 | V2coord d=64 | V2coord d=256 | V2sr d=64 | V2sr d=256 |
|---|---|---|---|---|---|---|---|---|---|---|
| random thought | 0.01 | 0.00 | 0.01 | 0.00 | 0.01 | 0.00 | 0.01 | 0.00 | 0.01 | 0.00 |
| value gradient ∇_z V — cost 3 | 0.54 | 0.59 | 0.53 | 0.59 | 0.53 | 0.59 | 0.54 | 0.59 | 0.54 | 0.59 |
| exact local inverse J⁺·(goal − state) — reference | 0.15 | 0.18 | 0.15 | 0.18 | 0.08 | 0.09 | 0.82 | 0.86 | 0.08 | 0.07 |
| W from random thoughts (ridge) · (goal − state) | 0.20 | 0.01 | 0.20 | 0.03 | 0.23 | 0.05 | 0.97 | 0.96 | 0.96 | 0.94 |
| W from random thoughts (Hebbian, GCML's rule) · (goal − state) | 0.21 | 0.01 | 0.19 | 0.01 | 0.23 | 0.06 | 0.97 | 0.97 | 0.84 | 0.65 |
| W from search experience (Hebbian) · (goal − state) | 0.29 | 0.29 | 0.33 | 0.36 | 0.35 | 0.38 | 0.95 | 0.95 | 0.71 | 0.71 |
| W from search experience (ridge) · (goal − state) | 0.29 | 0.29 | 0.34 | 0.34 | 0.36 | 0.38 | 0.94 | 0.94 | 0.88 | 0.89 |
| W from search experience (ridge) · ∇_c V — no goal | 0.26 | 0.26 | 0.34 | 0.34 | 0.38 | 0.36 | 0.88 | 0.88 | 0.72 | 0.75 |
| W(c) network from search experience · (goal − state) | 0.29 | 0.32 | 0.34 | 0.35 | 0.41 | 0.44 | 0.99 | 1.00 | 0.96 | 0.94 |
| W(c) network from search experience · ∇_c V — no goal | 0.08 | 0.11 | 0.11 | 0.11 | 0.19 | 0.21 | 0.92 | 0.91 | 0.55 | 0.48 |
| W from demonstrations (ridge) · (goal − state) | 0.39 | 0.39 | 0.37 | 0.36 | 0.36 | 0.36 | 0.82 | 0.82 | 0.84 | 0.83 |
| W(c) network from demonstrations · (goal − state) | 0.43 | 0.43 | 0.45 | 0.46 | 0.53 | 0.52 | 1.00 | 1.00 | 1.00 | 1.00 |

### 13.2 G1 — what can the inverse model be learned from?

Signal strength — how much a transition actually changes the state code, relative to a full move:

| d | random thoughts | search transitions |
|---|---|---|
| 64 | 0.08–0.13 | 0.32–0.59 |
| 256 | 0.02–0.04 | 0.33–0.58 |

Ridge W against the amount of experience (one-step progress, goal-directed):

| variant d | random N=256 | random N=1024 | random all (4096) | search N=256 | search N=1024 | search all |
|---|---|---|---|---|---|---|
| V0 d=64 | 0.02 | 0.08 | 0.20 | 0.24 | 0.29 | 0.29 |
| V0 d=256 | 0.00 | 0.00 | 0.01 | 0.24 | 0.29 | 0.29 |
| V1raw d=64 | 0.04 | 0.10 | 0.20 | 0.30 | 0.34 | 0.34 |
| V1raw d=256 | 0.00 | 0.00 | 0.03 | 0.30 | 0.34 | 0.34 |
| V1sr d=64 | 0.05 | 0.12 | 0.23 | 0.34 | 0.35 | 0.36 |
| V1sr d=256 | 0.00 | 0.00 | 0.05 | 0.34 | 0.38 | 0.38 |
| V2coord d=64 | 0.85 | 0.96 | 0.97 | 0.92 | 0.95 | 0.94 |
| V2coord d=256 | 0.64 | 0.93 | 0.96 | 0.91 | 0.95 | 0.94 |
| V2sr d=64 | 0.62 | 0.89 | 0.96 | 0.82 | 0.87 | 0.88 |
| V2sr d=256 | 0.25 | 0.79 | 0.94 | 0.83 | 0.89 | 0.89 |

### 13.3 G3/G4 — search success (budget 64 / 256 / 1024; 40 held-out problems per cell)

d = 64:

| algorithm | V0 | V1raw | V1sr | V2coord | V2sr |
|---|---|---|---|---|---|
| `grad_greedy` | 0.42 / 0.78 / 0.95 | 0.35 / 0.68 / 0.93 | 0.35 / 0.68 / 0.93 | 0.33 / 0.65 / 0.88 | 0.33 / 0.65 / 0.88 |
| `mcts_hybrid` | 0.00 / 0.68 / 0.97 | 0.00 / 0.62 / 0.97 | 0.00 / 0.62 / 0.97 | 0.00 / 0.75 / 0.95 | 0.00 / 0.75 / 0.95 |
| `mcts_guided` | 0.20 / 0.60 / 0.93 | 0.28 / 0.72 / 0.97 | 0.28 / 0.72 / 0.97 | 0.28 / 0.78 / 0.97 | 0.28 / 0.78 / 0.97 |
| `gcml_greedy` | 0.03 / 0.03 / 0.03 | 0.05 / 0.07 / 0.07 | 0.07 / 0.10 / 0.10 | 0.57 / 0.70 / 0.68 | 0.35 / 0.45 / 0.47 |
| `gcml_greedy_v` | 0.00 / 0.03 / 0.03 | 0.10 / 0.12 / 0.12 | 0.15 / 0.15 / 0.15 | 0.17 / 0.23 / 0.23 | 0.10 / 0.10 / 0.10 |
| `invnet_greedy` | 0.00 / 0.00 / 0.03 | 0.05 / 0.10 / 0.07 | 0.07 / 0.12 / 0.20 | 0.78 / 0.88 / 0.93 | 0.80 / 0.82 / 0.82 |
| `invnet_greedy_v` | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.33 / 0.35 / 0.47 | 0.03 / 0.03 / 0.05 |
| `mcts_gcml` | 0.03 / 0.03 / 0.03 | 0.03 / 0.03 / 0.07 | 0.07 / 0.07 / 0.15 | 0.45 / 0.60 / 0.68 | 0.30 / 0.47 / 0.50 |
| `mcts_gcml_v` | 0.00 / 0.00 / 0.03 | 0.05 / 0.05 / 0.05 | 0.10 / 0.15 / 0.17 | 0.12 / 0.23 / 0.25 | 0.07 / 0.07 / 0.12 |
| `mcts_invnet` | 0.00 / 0.00 / 0.05 | 0.03 / 0.03 / 0.07 | 0.07 / 0.17 / 0.20 | 0.60 / 0.85 / 0.95 | 0.70 / 0.82 / 0.82 |
| `mcts_invnet_v` | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.30 / 0.40 / 0.45 | 0.00 / 0.03 / 0.05 |
| `mcts_hybrid_gcml` | 0.00 / 0.68 / 0.80 | 0.00 / 0.70 / 0.97 | 0.00 / 0.60 / 0.97 | 0.00 / 0.75 / 0.95 | 0.00 / 0.75 / 0.97 |

d = 256:

| algorithm | V0 | V1raw | V1sr | V2coord | V2sr |
|---|---|---|---|---|---|
| `grad_greedy` | 0.53 / 0.80 / 0.95 | 0.33 / 0.78 / 0.88 | 0.33 / 0.78 / 0.88 | 0.35 / 0.65 / 0.78 | 0.35 / 0.65 / 0.78 |
| `mcts_hybrid` | 0.00 / 0.62 / 0.97 | 0.00 / 0.78 / 0.97 | 0.00 / 0.78 / 0.97 | 0.00 / 0.72 / 0.95 | 0.00 / 0.72 / 0.95 |
| `mcts_guided` | 0.15 / 0.70 / 0.95 | 0.28 / 0.70 / 0.97 | 0.28 / 0.70 / 0.97 | 0.38 / 0.75 / 0.95 | 0.38 / 0.75 / 0.95 |
| `gcml_greedy` | 0.00 / 0.00 / 0.00 | 0.07 / 0.10 / 0.10 | 0.07 / 0.07 / 0.07 | 0.57 / 0.60 / 0.62 | 0.40 / 0.45 / 0.50 |
| `gcml_greedy_v` | 0.00 / 0.00 / 0.00 | 0.07 / 0.07 / 0.07 | 0.10 / 0.10 / 0.10 | 0.17 / 0.23 / 0.25 | 0.12 / 0.12 / 0.12 |
| `invnet_greedy` | 0.05 / 0.05 / 0.05 | 0.07 / 0.07 / 0.10 | 0.12 / 0.12 / 0.12 | 0.72 / 0.75 / 0.85 | 0.55 / 0.62 / 0.68 |
| `invnet_greedy_v` | 0.03 / 0.03 / 0.03 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.28 / 0.30 / 0.30 | 0.07 / 0.07 / 0.07 |
| `mcts_gcml` | 0.00 / 0.00 / 0.00 | 0.03 / 0.05 / 0.07 | 0.07 / 0.07 / 0.10 | 0.47 / 0.57 / 0.60 | 0.23 / 0.38 / 0.45 |
| `mcts_gcml_v` | 0.00 / 0.00 / 0.03 | 0.05 / 0.05 / 0.05 | 0.07 / 0.07 / 0.07 | 0.10 / 0.17 / 0.23 | 0.10 / 0.12 / 0.12 |
| `mcts_invnet` | 0.00 / 0.05 / 0.07 | 0.05 / 0.07 / 0.10 | 0.05 / 0.07 / 0.10 | 0.53 / 0.75 / 0.80 | 0.45 / 0.57 / 0.65 |
| `mcts_invnet_v` | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.17 / 0.30 / 0.33 | 0.05 / 0.05 / 0.05 |
| `mcts_hybrid_gcml` | 0.00 / 0.50 / 0.90 | 0.00 / 0.50 / 0.90 | 0.00 / 0.57 / 0.90 | 0.00 / 0.72 / 0.95 | 0.00 / 0.65 / 0.95 |

### 13.4 Preconditions — which probe predicts whether GCML will work?

| variant | its code | invariance of its code | raw | sr | coord |
|---|---|---|---|---|---|
| V0 | raw | **0.01** | 0.01 | 0.02 | — |
| V1raw | raw | **0.01** | 0.01 | 0.01 | — |
| V1sr | sr | **0.01** | 0.01 | 0.01 | — |
| V2coord | coord | **0.84** | 0.00 | 0.05 | 0.84 |
| V2sr | sr | **0.05** | 0.00 | 0.05 | 0.84 |

| variant | d | G0 invariance (its code) | 1-step predictability on search transitions | k-step probe k=1 | k=2 | k=3 |
|---|---|---|---|---|---|---|
| V0 | 64 | 0.01 | 0.38 | 0.16 | 0.08 | 0.03 |
| V0 | 256 | 0.01 | 0.40 | 0.13 | 0.06 | 0.05 |
| V1raw | 64 | 0.01 | 0.39 | 0.19 | 0.07 | 0.06 |
| V1raw | 256 | 0.01 | 0.44 | 0.20 | 0.08 | 0.08 |
| V1sr | 64 | 0.01 | 0.25 | 0.15 | 0.07 | 0.06 |
| V1sr | 256 | 0.01 | 0.30 | 0.10 | 0.05 | 0.03 |
| V2coord | 64 | 0.84 | 0.49 | 0.58 | 0.44 | 0.42 |
| V2coord | 256 | 0.84 | 0.52 | 0.54 | 0.44 | 0.43 |
| V2sr | 64 | 0.05 | 0.50 | 0.54 | 0.42 | 0.40 |
| V2sr | 256 | 0.05 | 0.54 | 0.49 | 0.41 | 0.40 |

(The k-step probe: fit ridge W on the ONE-step pairs of value-gradient chains on training problems; on held-out chains,
cosine between W·(c_{t+k} − c_t) and the first thought z_t. It needs no goal and no knowledge of the right thoughts.)

### 13.5 What it says

1. **In its own world, GCML is excellent and free.** On the grid with consistent action semantics (V2), W·(goal −
   state) — ridge or network, fitted on random or search experience — makes progress from 0.88–1.00 of all nodes,
   above the value gradient's 0.54–0.59, which costs 3
   thought-steps, while W costs a matrix-vector product. A LEARNED GLOBAL inverse even beats the EXACT LOCAL one
   (J⁺: 0.82–0.86 with coordinates, 0.07–0.08 with the SR code): W is amortised over many transitions, J⁺ is
   linearised at a single point.
2. **Outside it, it fails** — and consistent action semantics are not enough. With a random graph (V0, V1) every
   linear W is at ≤ 0.39 and the best network (trained on demonstrations) at ≤ 0.53 — at or below the value gradient
   (0.53–0.59). **The
   pre-registered V1 expectation is refuted:** the SR eigenmap did not rescue a random graph with global keys
   (V1sr ≈ V1raw). What GCML needs is GEOMETRY: a goal-minus-state difference that is linearly related to a good first
   step.
3. **Search: GCML wins where the budget is smallest, then plateaus.** In V2 at budget 64, the inverse-network planner
   (`invnet_greedy`) solves 0.72–0.78 of problems with coordinates and 0.55–0.80 with SR, against ≤ 0.38 for every
   baseline (the hybrid tree scores 0.00 at 64, the subspace tree 0.28–0.38). By budget 1024 it plateaus (0.68–0.93) where the gradient trees reach 0.95–0.97.
   The linear-W planner plateaus lower (0.47–0.68). Adding GCML proposals to the hybrid tree (`mcts_hybrid_gcml`)
   did not help. In V0/V1 every GCML method stays ≤ 0.20.
4. **The goal-free form is not ready.** W·∇_c V (no goal) is weaker than W·(goal − state) everywhere (V2coord
   0.88 vs 0.94; the network version much worse), and its search variants stay ≤ 0.47. Our kernel estimate of the
   value's state gradient may be the weak link; open.
5. **Learn W from search experience, with ridge or a network — not from random thoughts with Hebb's rule.** A search
   transition carries 0.32–0.59 of a full move; a random thought 0.08–0.13 at d = 64 and 0.02–0.04 at d = 256. 256
   search transitions give 0.82–0.92 progress in V2; random ones need ≥ 1,024 at d = 256 (V2sr: 0.25 → 0.79 → 0.94).
   (In this mock tiny moves are noise-free, so random data still works given enough of it; in a real network they would
   sit in noise.) GCML's Hebbian rule matches ridge with an uncorrelated code (coordinates: 0.95–0.97) and falls behind
   with a correlated one (SR: 0.65–0.84 vs 0.88–0.96) — it lacks ridge's decorrelation.
6. **The right precondition probe is the k-step one.** G0's invariance does not predict success (V2sr: 0.05, yet W
   works); one-step predictability on search transitions separates the worlds only weakly (0.25–0.44 vs 0.49–0.54).
   The k-step probe separates them cleanly: at k = 3, 0.40–0.43 where GCML works and 0.03–0.08 where it does not —
   and a real looped model can produce the value-gradient chains it needs.

**Against the pre-registration (§12.3).** V2: held — W beats the value gradient at zero cost, and GCML wins at the
smallest budgets (through the search-free planner rather than inside the tree). V1: refuted (the SR embedding does
not rescue it). V0: held (fails). The refutation criterion (GCML no better than the value gradient at equal cost in
V2) was not met. G0, named "decisive for the real model", is replaced by the k-step probe.

### 13.6 Recommendation

GCML is a CONDITIONAL component. Run the k-step probe on the real looped model's own value-gradient chains first.
If its thought space shows GCML structure (k = 3 predictability far above zero), use a learned inverse — a
state-conditioned network, trained on search experience — as a cheap first try: the search-free planner at small
budgets, before handing over to the gradient tree (`mcts_hybrid`) as budget grows. It needs a goal (games can supply
one); the goal-free form is not ready. If the probe reads near zero, leave GCML out.

**Caveats.** A mock, with planted structure; the dynamics are fixed and soft, so tiny effects are noise-free (which
flatters random exploration); W is learned per world from 16 problems' experience while the baselines learn nothing
across problems, so the comparison favours GCML; 40 problems per search cell (standard error up to ~0.08); constants
set once.

---

## 14. The pipeline these results suggest (discussed 2026-10-04)

**The user's framing:** (1) learn a domain at the start with regular search; (2) then GCML as a large speedup for
domains whose map is known; (3) over time, compositionality or general structure makes GCML useful on new domains.
Kept, with these refinements:

**Stage 1 — search learns the domain, and teaches everything else.** The gradient-proposing tree (§11) is the general
method. Its simulated transitions are also the best data for the inverse model (§13.2: 0.32–0.59 of a full move per
transition against 0.02–0.04 for random thoughts at d = 256; 256 of them suffice in a geometric world), and its
outcomes train the value that every method depends on (§11.4 point 6).

**Stage 2 — two speedups, complementary, not one:**

| | general | cost | needs geometry | new goals in a known domain |
|---|---|---|---|---|
| search (gradient tree) | yes | high | no | yes, at full cost |
| distilled policy (search → policy, §11.6) | within its training distribution | low | no | only goals like those it was trained on |
| GCML inverse model | where geometry exists | lowest | yes | **yes, zero-shot**: `W(s* − s)` for any goal |

GCML's distinctive gain is goal-generality inside a mapped domain; distillation is the speedup available where there
is no geometry. The switch is decided by MEASUREMENT, per domain: the k-step probe (§13.4) says whether GCML can work;
a running record of how often its first step makes progress says whether it does. (The brain is thought to arbitrate
between planning and habit by their reliability — Daw, Niv & Dayan, Nature Neuroscience 2005.) And GCML is a first
try, not a replacement: it wins at the smallest budgets and plateaus (§13.5 point 3), so search takes over when it
stalls.

**Stage 3 — general structure. Plausible, with a mechanism and a hard limit.**
- *Mechanism:* GCML's own forward-model objective, `Q·o_{t+1} ≈ Q·o_t + V·a_t` (its eq. 11), is an objective for
  LEARNING a state code in which actions add — the property `W(s* − s)` needs. §13 hints it is learnable: the SR code,
  computed from transition statistics alone, worked with the state-conditioned inverse (0.94–0.96).
- *Limit — commutativity:* a code in which every action is a fixed translation exists only if the actions (nearly)
  commute. ZipLearn's E12 found exactly this: commuting action sets compress to coordinates (8.9 bits), non-commuting
  ones need a word table (1,190 bits). No code makes a random graph or a permutation puzzle GCML-friendly as a whole.
- *Compositionality as the way past it:* domains built from known parts (objects × operations), each part commutative
  in its own slice of the code, let a goal decompose into per-part differences, each mapped by a known inverse
  (GCML's building-block result: trained on 5 blocks, solved 8). The price is a code that factors by part — the
  binding problem of `experiments/NOTES.md`.

**Pre-registered — the geometry test (§15).** Three worlds: the grid; a PRODUCT of two grids (a compositional world);
the permutation group S₅ with four generators (non-commuting — the negative control). Codes: raw (one-hot), SR, the
true coordinates where they exist, and a code LEARNED from search experience with GCML's eq-11 objective (solved by
alternating least squares, whitened against collapse). In the product world the inverse model learns only from
problems whose goal changes ONE part, and is tested on goals that change BOTH. Measured: one-step progress of
`W(goal − state)`, the k-step probe, and the GCML planner's success at budgets 64 and 256.
*Expected:* the learned code recovers geometry on the grid (progress well above raw); in the product world a
geometric code (coordinates or learned) generalises from one-part to two-part goals; on S₅ every code fails (progress
at the level of a random graph, probe near zero). *Refuted if* the learned code does no better than raw on the grid,
or if S₅ succeeds (then commutativity is not the limit).
