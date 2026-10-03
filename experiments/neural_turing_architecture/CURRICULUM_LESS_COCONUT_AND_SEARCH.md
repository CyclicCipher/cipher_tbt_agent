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
| **A3a** | hard latent tokens: `z = e_a`, a from a learned codebook of N (64–256) | a categorical action | **exact** (Gumbel MuZero: at any simulation budget, provided the visited actions' values are evaluated correctly) | thoughts are decodable symbols of a private vocabulary |
| **A3b** | soft latent tokens: `z = Σ_i π′_i e_i`, the SEARCH-IMPROVED distribution as mixture weights | a categorical action; the executed thought is the improved distribution itself | exact for the search; the thought superposes the branches | Soft Thinking's "concept token", produced by search; Coconut's "BFS in superposition", made explicit |
| **A4** | codebook entry plus a continuous residual | categorical + continuous | mixed | later, if A3 is too coarse |

**Proposed:** build **A3** first (A3a and A3b share all the machinery and differ by one line), **A2** as the
continuous arm, and **A1** as the baseline. Reasons: A3 gets the exact improvement guarantee at tiny simulation
budgets — and every simulation is a forward pass of the looped core, so tiny budgets are the regime we live in; its
thoughts are legible; and A3b is still a continuous thought. The cost: expressivity limited to the hull of the
codebook, and a codebook that can collapse (§6).

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
- **codebook:** through its use as thoughts; plus a usage-entropy term against collapse (§6);
- **exit gate for the depth axis** (if learned): Ouro's two stages, later (§4.6).

### 4.5 The continuous arm (A2), for comparison

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
- **Codebook collapse** — the policy settles on one or two entries; every thought is the same. Guard: a usage-entropy
  term on the codebook; Gumbel noise at the root; detect by codebook perplexity.
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
  before anything is claimed against it; (c) S0 with noise (Soft Tokens, Hard Truths); (d) S1 on A3a.
- **P2 — S2 on A3a, then A3b.** Pass: solves hop counts the P1 arms cannot at the same budget, and the thought
  ablation destroys it. Refute: no better than S1.
- **P3 — S3 with search-based values; A2 with EZ-V2's continuous search.**
- **P4 — halting learned** (both axes); λ_T schedule.
- **P5 — amortised vs test-time search** (§4.7).
Metrics everywhere: success vs difficulty at equal thought-steps; thought ablation; codebook perplexity;
V calibration; thoughts per problem against difficulty (does thinking grow with need?).

## 8. Interplay with memory (Spotlight) and depth routing

- Search state forking: §2 item 5. Spotlight makes it cheap; attention needs a tree-structured cache.
- With Spotlight, a thought need not be an appended position at all: it can be a WRITE into memory that later
  thoughts read — the decision process's state then lives in memory, and the action could include where to write.
  Out of scope for v0; recorded because it changes §2's state definition.
- Depth routing (Block AttnRes vs the boundary operator) is `BRAINSTORM.md` §1; the thought fed back along the
  sequence axis needs its own normaliser regardless (A3 bounds it automatically: a convex mix of codebook rows).

## 9. Open questions for discussion

- **Q1 — Discrete codebook or truly continuous?** A3b is continuous but lives in the codebook's hull; A2 is fully
  continuous. Is the codebook a betrayal of "continuous thought", or the right inductive bias?
- **Q2 — Is search needed at test time, or only for training?** (§4.7.)
- **Q3 — Does the answer's cross-entropy alone suffice in our setting?** Coconut says no without the curriculum; P1(b)
  tests it here. If it DOES suffice on these tasks, the search is a speed-up, not a necessity.
- **Q4 — Partial credit: allowed?** (§5.2.)
- **Q5 — A problem-difficulty curriculum: allowed?** (§5.1.) It is not a substitution curriculum, but it is a
  curriculum.
- **Q6 — Machine-state auxiliaries** (e.g. "the thought should let a probe read the tape"): allowed as training
  signals, or probes only?
- **Q7 — Who proposes candidates?** The policy alone, or also GCML-style goal-directed proposals and value-gradient
  steps (S4)? Proposal mixing needs Sampled MuZero's correction to stay a valid improvement.
- **Q8 — Where does search run?** At every thought (expensive, dense targets) or only where the policy is uncertain
  (cheap, sparse targets)?
