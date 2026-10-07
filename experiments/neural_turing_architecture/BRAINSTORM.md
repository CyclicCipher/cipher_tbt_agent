# Neural Turing Architecture — brainstorm

Status: BRAINSTORM, opened 2026-10-03. Nothing here is built or decided. Sources are in `REFERENCES.md`.
Working name only.

**The intent (the user's words, condensed).** Combine Ouro-style looping (depth recurrence learned during training)
with Coconut-style continuous chain of thought (sequence recurrence), using what Chen et al. showed about looping,
growth and boundary operators. Then three open threads: (1) do Block Attention Residuals make the boundary
operator unnecessary? (2) could a Spotlight-style memory replace the attention layers — linear cost, length
extrapolation, and memory that can be OVERWRITTEN rather than only appended? (3) how does the model DISCOVER
latent reasoning with no substitution curriculum and no human reasoning traces — "EfficientZero V2 for thoughts",
GCML on the search side, Gemini's suggestions, and protection against collapse?

**Standalone.** ZipLearn is not proceeding (the user, 2026-10-03); this experiment builds this architecture. Nothing
here depends on ZipLearn's machinery or rules (its DESIGN §18 forbade a "policy over thoughts", which thread (3)
proposes). What carries over is evidence — E26, E28, E30, E31, E41, E46 — and possibly the `h1_lid.py` substrate.

**Updated 2026-10-03 (later the same day):** the Spotlight post and Percepta's reference code were read in full
(§2, `refs/spotlight_memory_percepta_2026.md`); the EfficientZero V2 paper was read in full
(`refs/efficientzero_v2_2403.00564.md`); thread (3) now has its own planning document,
`CURRICULUM_LESS_COCONUT_AND_SEARCH.md`, which supersedes §3 below where they differ.

## 0. The architecture as currently imagined

```
tokens ─► prelude ─► e   (the anchor)
                     │
           ┌─────────▼──────────────────────────┐
           │ core block(s), applied k = 1..K     │   DEPTH axis (Ouro, Chen et al.): refine the same positions
           │   input_k = route(e, h_1 … h_{k-1})  │   ← boundary operator and/or Block AttnRes      (§1)
           │   reads/writes memory               │   ← attention today; Spotlight memory?          (§2)
           │   exit gate: stop refining?          │   ← Ouro two-stage gate
           └─────────┬──────────────────────────┘
                     ▼
                   coda ─► thought head: z_t ~ π(· | h)   or "answer now"       (§3)
                     │
                     └─► z_t becomes the input at position t+1           SEQUENCE axis (Coconut)
```

Two loops, nested: per thought, up to K passes (the INNER loop, ZipLearn DESIGN §18.2's "depth refines a fixed
state"); per problem, up to T thoughts (the OUTER loop, "sequence adds state"). Two halting decisions: stop
refining (depth) and stop thinking (sequence). The name fits the 2014 original: a controller (the looped core,
one pass = one clock tick) with read/write access to a memory, learning to compute without being shown how.


## 0.1 Standing principles and insights — read before designing anything (kept current)

Each line is one thing this experiment has decided or learned, with where its evidence lives. Add to this list when
something new is learned; strike through (do not delete) what is overturned. "Mock" = the CPU thought-space mock of
the planning doc §11–§18: one kind of world, so a mock result is PROVISIONAL until reproduced on the task suite with a
real model.

**Decisions (the user's)**
1. **One general method.** No set of specialised algorithms with a dispatcher deciding which one a piece of data
   needs; case structure (commuting or not, geometric or not) goes into what is LEARNED — policy, value,
   representation — under one search (2026-10-06; planning §19).
2. **Trained, not written,** until the architecture is settled (2026-10-06). *Insight (unmeasured):* written weights
   are nearly flat — an E46 row is driven to −M off its token class, so its gradient is ≈ 0 — so a written machine
   cannot seed gradient-based search unless its sharpness M is lowered; M acts as a temperature, the same dial as
   Spotlight's sharp VM kernel against its smooth trained one.
3. **Diverse training environments.** No conclusion from a single task family or a single mock world; every
   experiment reports across a suite of structurally different families (2026-10-06; the suite is built in P0,
   planning §20).
4. **Thoughts are free vectors in ℝ^d** — no codebook (Q1, 2026-10-03).
5. **Outcome-only;** machine execution states are probes, never targets; a problem-difficulty frontier is allowed —
   and load-bearing (item 7). From scratch, synthetic tasks, attention first, `h1_lid.py` (§5, 2026-10-06).
6. **GCML and the TBT-related questions are out of the main line** (2026-10-06).
6b. **No Coconut-faithful baselines** — no compute on reproducing or beating Meta's setup; arms test our architecture
    (2026-10-06).

**Insights (with evidence)**
7. **Learning-theory lens:** from the outcome alone a k-step composition is exponentially hard for gradient
   learners; intermediate targets or a LADDER of difficulty are the escapes. The frontier carries the weight; the
   search algorithm is secondary (planning §19.1).
8. **Sampling-only search and outcome-only RL die from d ≈ 64** in a thought space; gradients through the exact
   dynamics are necessary (§10.2 derivation + toy; §11, §16 — mock).
9. **The answer's gradient through the thoughts is a strong signal in principle** (`bptt`, §16 — mock). P1(b) is the
   real test.
10. **Coconut's no-curriculum failure has two candidate causes:** gradient pathology through a raw, unnormalised
   fed-back state — the user's hypothesis is that our loop's normalisation and input re-injection avoid it — and
   credit assignment, which no architecture removes. Not tested against a Coconut-faithful arm (item 6b); P1(b) tests
   the credit-assignment half on our architecture (mixed levels against hardest-only, planning §19.5).
11. **Training targets must lie on the policy's own path** (§16 point 8: the tree found the best solutions and taught
    worst). Search as one more step of the TRAINING update (Ataraxos's update equivalence) keeps them there by
    construction (§19.3).
12. **The zone of proximal development has measures:** the self-play paper's gradient-alignment reward sees the next
    rung before any success (where p(1 − p) reads zero); Ataraxos's |advantage| filter is its per-decision form (§19.2–3).
13. **Damped dynamics:** anneal the pull toward a fixed simple policy together with the step size — regularisation as
    an "energy reserve"; spent too fast, entropy collapses and learning stops (Ataraxos, §19.3).
14. **A value is trained on the search's own data by its own backup;** labelling explored-but-unexpanded states as
    failures collapsed it (§16.2).
15. **Answer-encoding trap:** a free thought can carry the answer itself; guard with KL-bounded targets and the
    thought ablation (§10.5, R6).
16. **Spotlight:** overwritable memory, O(1) work per token, a search fork is a copy-on-write overlay of ≤ 9·H·L
    cells (§2). Comes in after NTA-M validates it.
17. **Depth axis:** attention residuals across passes held 0.60–0.77 at 2× the trained passes where the fixed boundary
    operator collapsed to 0.08; a window over passes failed (E30; §1). The sequence axis still needs its own
    normaliser on the fed-back thought (§1 gap 4).
18. **The leap shows up inside a single hop.** A lookup with key and value on separate tokens needs two attention
    steps that pay off only together and never left chance (our model 12,000 steps; a plain 2-layer transformer
    10,000); with the pair in ONE token it is learned in ~2,000 steps (planning §20.1). Spotlight's write — a key
    addresses a cell that holds the value — is the one-token form: memory that binds key and value when it writes
    spares attention a leap when it reads.
19. **Uniform mixing of difficulty blocks even the easy levels** (`aff` level 1: 1.00 alone, 0.04 inside a uniform mix of
    48 levels). The frontier is load-bearing even with no thoughts (planning §20.1).
20. **A depth knob must be checked against shortcuts.** Short-circuiting Boolean siblings and counted Brainfuck loops
    both let a no-thought model score well at "depth 8" / "256 steps" (planning §20.1). A family's depth is the depth
    its answer actually REQUIRES, not the size of its generator's knob.

---

## 1. Do Block Attention Residuals remove the need for the boundary operator?

**Short answer: for the depth axis, mostly yes. For the sequence axis, no.** Keep a thin boundary operator, and let
attention residuals do the routing over a BOUNDED source set.

**What the boundary operator does** (Chen et al.: BO(h, e) = RMSNorm(h) + α·e between passes). Two jobs: (a) bound
the carried state's magnitude, so the stream does not grow with every pass; (b) re-inject the input every pass, the
stability condition Geiping et al. argue for.

**What Block AttnRes covers.** (a) A pass's input is a softmax-weighted, i.e. CONVEX, combination of its sources,
so it cannot exceed the largest source; the AttnRes paper reports bounded, periodic magnitudes, each block boundary
resetting the accumulation. (b) The anchor is source 0, so re-injection is available to every pass, weighted per
token by content. ZipLearn DESIGN §15 already calls the fixed `rms_norm(x) + α·anchor` "the one-source special
case" — the anchor plus only the last pass, at fixed weights.

**Where it falls short — five gaps.**

1. *Re-injection becomes optional.* E30's trained routes let the anchor fade from 1.00 to 0.01 by pass 4. Geiping
   et al.'s argument says the input must enter every pass. Fading did not hurt at 2× passes in E30 — run at 8
   passes after training at 4, full-history AttnRes kept 0.60–0.77 trained-task accuracy while the tied-core FIXED
   boundary operator collapsed to 0.08 (held-out at 8 passes: 0.12–0.13 against 0.00; one seed, so the sign is
   established, not the size). But loops run far beyond the trained count — the "loop until converged" regime — are
   untested.
2. *Unbounded loops make the source list unbounded.* Full history costs O(K·d) memory per token and O(K) per mix:
   fine at K = 4, a problem at K = 64. The obvious fix — a window — is REFUTED: E30's {anchor, last two passes}
   fell to 0.19–0.20 trained-task accuracy at 2× passes and lost the held-out gain.
3. *Convergence-halting wants a fixed state.* A pass that reads all earlier passes is not a map on a fixed-size
   state, so "stop when the state stops changing" and fixed-point arguments (Anderson mixing, DEQ) lose their footing.
4. *It does nothing for the sequence axis.* AttnRes mixes over DEPTH. A Coconut thought fed back as the next input
   embedding is untouched by it, and AttnRes passes VALUES raw (only the keys are normalised), so an over-large
   thought flows straight through. Coconut's c = 3 loss spikes and its unnormalised feedback are exactly this.
   That axis still needs a normaliser — RMSNorm on the fed-back thought, or the simplex projection of §3.5.
5. *The two were never measured together.* Chen et al. measured the boundary operator; Kimi measured AttnRes (in a
   non-looped model). They may be complementary — RMSNorm on the carried state plus content routing — or redundant.
   No one has said.

**What E30 already told us to write.** Its conclusion was: "the pass's input is the last pass plus a decaying memory
of all earlier passes, the anchor fading rather than re-injected at fixed weight, no window". A *decaying memory of
all earlier passes* is an exponential moving average — a FIXED-SIZE state. That resolves gaps 2 and 3 without a
window, and it was never tested as a source.

**Proposed: NTA-Q1** (CPU/3050-sized, reuses the E30 apparatus `h1_lid.py --res loop --loop_res attnres`, plus one
new source type). Arms on the h1_lid composition task, trained at K = 4:
(A) the fixed boundary operator; (B) AttnRes, full history (E30); (C) AttnRes, window {anchor, last 2} (E30's
failure, as a control); (D) AttnRes over {anchor, last pass, EMA of earlier passes}; (E) D plus RMSNorm on the
carried state and a floor on the anchor's weight. Measure held-out accuracy at K, 2K and 4K passes, the stream's
norm per pass, and the learned routes.
*Pass:* D or E ≥ B at K and at 4K, with memory constant in K. *Refute:* D below C, or E below A.

---

## 2. Spotlight memory in place of the attention layers

### 2.1 What it is — now read in full (`refs/spotlight_memory_percepta_2026.md`)

Per head, keys and queries are given learned 2-D ADDRESSES on an unbounded lattice. A key WRITES the 3 × 3 cells
around its address and a query READS the 3 × 3 around its own, weighted by a compact, normalised cos² bump, so that
addresses receive gradients. A cell is allocated on its first write and holds a small recurrent state: in the
trained models a d_k × d_v DeltaNet state per head (routing in 2-D; content inside the cell by the delta rule), and
in the released hand-built VM a 2-vector updated by `S ← S + w·(v − α·S)`, with a write gate β and an erase factor α.
Work per token is constant (9 cells per head); memory grows with what has been written; O(T) in total.

Reported: near-perfect multi-query associative recall, including 131K pairs and keys OVERWRITTEN mid-sequence
(stale-value rate zero) where attention returns stale values and Gated DeltaNet saturates; language-model loss at
140M–670M close to Gated DeltaNet and below attention; 8K-trained models recall a needle at 128K at 93–100%
(attention 0%, fixed-state ≤ 5.6%). Percepta also hand-built a MicroPython interpreter inside an 8-layer, d = 16
Spotlight transformer (`percepta-ai/spotlight-vm`, Apache-2.0), with constant work per step over 47M-token traces.

It has exactly the properties this brainstorm wanted — linear cost, extrapolation past the training length,
memory that can be overwritten — and it is small: the reference read/write is about twenty lines of C.

### 2.2 The earlier guesses, scored

Before the post could be read, this section listed four candidates. The answer combines two of them: learned
low-dimensional addressing that selects a constant number of cells (product-key-like; Percepta's own 2-D heads),
and a DeltaNet (delta-rule) state inside every cell. The earlier verdict "probably NOT a DeltaNet" was wrong — the
DeltaNet is inside each cell; what is NOT DeltaNet-like is the growing, sparsely addressed collection of them.
Sparse Access Memory (Rae et al. 2016) remains the closest ancestor in spirit.

### 2.3 We already had mutable memory — on the depth axis

E46's Brainfuck interpreter keeps its tape as tokens and rewrites them IN PLACE every pass (`block B: write
tape[dp] <- register`, a `Match` from the tape side). In a depth-looped model the positions are fixed and each pass
rewrites their residuals: **the positions are a register file.** The "KV cache you can expand but cannot fix"
problem exists only along the SEQUENCE axis. Spotlight is the same idea made sparse and growing, and moved to the
sequence axis: each pass reads and writes 9 cells per head instead of attending over all T positions, and a thought
can overwrite its own earlier notes. E41 is the thesis from the other side: byte-identical weights, a world model
entirely in the context (an empty context scores 0.000) — "separate intelligence from memory".

### 2.4 Where it sits in this architecture

- **As the sequence mixer** inside the prelude, core and coda blocks, in place of self-attention. The AttnRes depth
  mix (§1) stays: it attends over a handful of depth sources, not over the sequence, so it is not where the cost is.
- **On the depth axis** each pass of each token writes, then reads. Design choice D1: (a) one lattice per (layer,
  head) shared by every pass — natural with tied weights; passes communicate through memory; a later pass can
  overwrite what an earlier pass of the same token wrote; (b) a separate table per pass (K× the cells); (c) memory
  written only outside the core (prelude/coda), the core reading only.
- **On the sequence axis** a thought costs O(1) — there is no growing cache — and the state of the thought process
  lives partly in memory.
- **For search** a node that adds one thought changes at most 9 · H · L cells, so forking is a copy-on-write overlay
  (`CURRICULUM_LESS_COCONUT_AND_SEARCH.md` §2).
- **Sharp vs smooth.** The VM's kernel (`cos²(πt/2)`, |t| < 1) makes an integer address exact RAM; the trained
  kernel (`cos²(πt/3)`, |t| < 3/2) is smooth. One mechanism spans both regimes.

### 2.5 What the post leaves open — implementation questions

How addresses are parameterised and kept in range (raw linear projections in the VM); how training is parallelised
(the per-token write-then-read is a recurrence — DeltaNet has chunked parallel forms, the sparse scatter complicates
them; at our sizes a sequential scan may simply be acceptable); batched allocation on a GPU (a bounded grid per
sequence, or hashing with collisions); gating or decay inside a cell; whether addresses need a regulariser to keep
distinct keys apart (the post's routing figure shows matching queries and keys converging to shared cells while
distinct keys spread apart — learned, with no explicit term mentioned).

### 2.6 Tests (proposed: NTA-M)

- **M0 — understand the reference.** Build and run `spotlight-vm` (C, one CPU core) and check our reading of its
  read/write against it. Then validate our own trainable implementation on Percepta's published MQAR behaviour
  (extrapolation from ≤ 64 to 256 pairs; the overwrite variant) before using it for anything else.
- **M1 — binding load beyond training** (Zoology's MQAR protocol; `experiments/binding_mqar.py` is a harness, though
  it imports Mamba-3 from a gitignored local clone).
- **M2 — overwrite:** a key–value stream with updates; query the latest value.
- **M3 — a tape longer than trained:** Brainfuck programs (the E46 generator) whose tape grows past training length.
Arms: softmax attention; Gated DeltaNet; Spotlight.

---

## 3. Discovering latent reasoning with no curriculum and no traces

The first pass at this thread. It continues in `CURRICULUM_LESS_COCONUT_AND_SEARCH.md`, which works out the
thought space, the search algorithm, exploration, cost and the experiment order, and supersedes this section where
they differ.

### 3.1 What Coconut's curriculum was really supplying

Per-step TARGETS. At stage k the first k language steps are replaced by thoughts, and the loss on the remaining
language tokens tells each thought what it must enable — dense supervision. Without it, Coconut reports "no better
than no-CoT". So whatever replaces the curriculum must manufacture per-step targets from the outcome alone. That is
precisely what SEARCH does in MuZero-family methods: a policy-improvement operator turns one end-of-episode outcome
into an improved policy and a value at every node. (It is also why DiffusionBlocks gives each block a local target —
ZipLearn DESIGN §20.)

### 3.2 Thinking as a decision process

- **state** s_t = the problem plus the thoughts so far, as the model has computed them;
- **action** z_t ∈ ℝ^d (continuous), plus a discrete "answer now";
- **transition**: append z_t and run the model — deterministic, and EXACT;
- **reward**: at the end, the verifier (answer correct or not), minus λ per thought or per pass.

The difference from MuZero/EfficientZero is decisive: **the dynamics are known — they are the network itself.**
MuZero learns a model because the environment is unknown; here the simulator is free and perfect, the situation
AlphaZero was in. Two consequences: the learned-dynamics machinery is not needed in the base design, and each
simulation costs a full forward pass of the looped model (so a search that improves the policy with FEW simulations
matters more than anything else). A third, subtler one: the "environment" shares weights with the policy, so it
changes as the policy trains.

### 3.3 What transfers from EfficientZero V2

- **Gaussian policy over a continuous action + sampling-based Gumbel search** — root candidates sampled from π(z | s)
  and from a flattened copy of it (for exploration), Sequential Halving to pick among them. In the discrete case
  (Gumbel MuZero) policy improvement is guaranteed at any budget; in EZ-V2's continuous case the argument holds only
  as the number of policy samples grows (their eq. 7) — and a thought is far higher-dimensional than EZ-V2's
  actuator vectors. That is why the planning document prefers a codebook (discrete) thought space for search.
- **Search-based value estimation** — the root's empirical mean as the value target; makes early, stale data usable.
- **Reanalyse** — with exact dynamics, old problems can be re-searched with the current model at any time.
- **Not needed by default:** the learned representation/dynamics, the value prefix and the SimSiam consistency
  loss — they exist because the environment is unknown. They return if we ADD a cheap learned thought-dynamics model
  to avoid running the full core at every search node ("imagining the imagination"); then EfficientZero's
  consistency loss is what keeps that model from collapsing.

### 3.4 What GCML adds on the search side

- **A goal-directed proposal.** GCML's inverse model maps a wanted change to the action that causes it, u = W(s* − s).
  Candidates for the search can come from three sources: the Gaussian policy; an inverse-model step toward a goal
  embedding (where a goal state exists — games); and a value-gradient step z + η·∇_z V (DDPG-like — GCML's W is a
  linear, Hebbian stand-in for this direction). Sampled MuZero's correction keeps the policy-improvement guarantee
  valid when proposals come from a mixture.
- **A search-free default.** GCML imagines by iterating its forward model with W choosing each step: O(1) per step,
  self-correcting, and it beat tree search on nodes visited in its tasks. So: GCML-style rollout as the default way
  of thinking, Gumbel search only where the value is uncertain — routine as a read-off, search as the fallback.
- **Noise as a sampler.** GCML's ε turns one planner into a sampler of diverse plans — the same job σ does in the
  Gaussian policy.
- **Caveat.** GCML's W is linear and works because its embedding makes the map nearly linear; a reasoning task's
  thought space may not be.

### 3.5 Gemini's suggestions, one by one

| # | suggestion (Gemini, 2026-10-03) | verdict | why, and the change |
|---|---|---|---|
| 1 | **Stochastic latent emissions**: the head outputs μ_t, log σ_t; z_t = μ_t + ε ⊙ σ_t | **keep** | It is EZ-V2's continuous policy. Soft Tokens, Hard Truths shows noise on the fed-back embedding is enough exploration for RL (RLOO) at 8B with no reference CoT — the simplest baseline arm. |
| 2 | **Pathwise gradients via reparameterisation**: terminal rewards backpropagate into μ, σ "without REINFORCE's variance" | **keep only with a critic** | A terminal reward is a verifier's verdict — not differentiable in z. Reparameterisation needs a differentiable path: a learned V/Q (as in SAC) or a learned model (as in Dreamer). Then the policy will EXPLOIT the critic's errors, pushing z to where V is wrongly high — a second kind of drift. Ground it with real (exact) rollouts and the verifier. (The answer's log-likelihood IS differentiable through the thoughts — that is Coconut's loss, and alone it did not learn without the curriculum.) |
| 3 | **Information bottleneck**: D_KL(q(z_t \| h_t) ‖ N(0, I)) | **modify** | A per-sample KL to a fixed prior pushes every thought toward the prior — posterior collapse, thoughts that carry nothing — the very failure to avoid. Magnitude control is better done by normalisation (§1, gap 4). If a KL is wanted: free bits. Better: regularise the AGGREGATE distribution of thoughts (LeJEPA's SIGReg, or WAE-style), which prevents dimensional collapse without making each thought uninformative. |
| 4 | **Soft-embedding simplex projection**: z_t = softmax(W_p h_t / τ) · E, τ annealed from high to low | **keep, re-motivated** | "Drift from the pretrained semantic manifold" is a problem of adapting a pretrained LM; if this experiment trains from scratch, that motivation is weak. The real benefits: bounded magnitude for free (the convex hull of the codebook), legibility (a thought is a readable distribution), τ → 0 gives hard one-hot thoughts (discrete, decodable), and a codebook makes the SEARCH discrete, where its improvement guarantee is exact (planning doc §3). Gumbel-softmax gives it stochasticity and reparameterisation together. Soft Thinking (training-free) and Soft Tokens, Hard Truths (RL) are the evidence. Cost: thoughts limited to the hull of the codebook — test a learned codebook against the token embeddings. |
| 5 | **JEPA-style state prediction**: thoughts must predict representations of future task-relevant observations | **keep for environments; probe only otherwise** | In a game, "the thought predicts the embedding of the next observation" IS EfficientZero's temporal-consistency loss (stop-gradient target) and SPR. For pure reasoning tasks the only "future observations" are execution states, which is trace supervision by another name. Use them to PROBE (can the tape be decoded from the thoughts?), not as a training target, in the main arm. |
| 6 | **Curriculum by circuit depth**, a learned continue/halt token, reward 0 when too few steps, a λ·K penalty | **keep** — the same idea is recorded in ZipLearn DESIGN §18.2 | The two-dimensional frontier (length × depth) and "depth as a speed prior". As a curriculum over PROBLEM difficulty it does not conflict with "no substitution curriculum". Concrete generator: E46's Brainfuck programs (steps per output byte = the depth a problem needs); or a self-play generator (2609.30063) scored by the learner's progress. The penalty schedule matters: λ·K from step 0 collapses to K = 0 — use Ouro's two stages (entropy-regularised exit distribution first, then tune the gate on realised gains). "Reward 0 at K = 0" needs no engineering: on problems deeper than the network it happens by itself. |

### 3.6 Collapse modes, and what guards each

| mode | symptom | detect by | guard |
|---|---|---|---|
| magnitude drift along the sequence axis | loss spikes as thoughts per step grow (Coconut, c = 3) | ‖z_t‖ by step | normalise the fed-back thought; or the simplex projection |
| thoughts ignored (posterior collapse) | accuracy unchanged when thoughts are zeroed or shuffled | thought ablation | tasks a K = 0 model provably cannot solve (depth > layers: parity, pointer chasing, Brainfuck); no per-sample KL; free bits |
| dimensional collapse | thoughts live in a few dimensions | effective rank of a batch of z | SIGReg; VICReg's variance/covariance terms |
| policy entropy collapse | σ → 0 early; search candidates identical | σ over training | entropy bonus with an auto-tuned temperature (SAC); a σ floor |
| critic exploitation | V high where the verifier says wrong | V-vs-outcome calibration | search with exact rollouts; a trust region to the previous policy |
| halting collapse | always stops at the first pass / thought | the exit distribution | Ouro stage 1 (entropy-regularised exit); PonderNet's prior |
| learned-dynamics collapse (only if a learned thought model is added) | constant latent | — | EfficientZero's consistency loss |

### 3.7 What trains what — so nothing is left without a signal

- **policy head** π(z | s) ← the search-improved distribution (a KL target), as in Gumbel MuZero / EZ-V2;
- **value head** V(s) ← search-based value estimates and the verifier;
- **the core (trunk)** ← the policy and value losses (AlphaZero-style: the trunk is trained through its heads), the
  answer's cross-entropy given the thoughts, and optional auxiliaries;
- **exit gate (depth)** ← Ouro's two-stage objective; **answer-now (sequence)** ← an action inside the search.

---

## 4. First experiments, if we build this (proposals, not pre-registrations yet)

- **NTA-Q1** — attention residuals vs the boundary operator in a loop (§1).
- **The thinking experiments (P0–P5)** — the task family and its no-thought ceiling, the baselines (including
  Coconut's BPTT-only negative result), Gumbel thought search, the continuous arm, halting, amortised vs test-time
  search: `CURRICULUM_LESS_COCONUT_AND_SEARCH.md` §7.
- **NTA-M** — the memory tests of §2.6.

---

## 5. Decisions for the user

1. **Starting point.** From scratch on synthetic tasks, or from a small pretrained LM? This decides whether
   "semantic drift" (Gemini #4's motivation) is a real problem.
2. **Outcome-only, or are machine execution states allowed** as an auxiliary signal (Gemini #5)? "No human text
   traces" rules out human traces; it does not by itself rule out a Brainfuck tape.
3. **Task suite priority:** Brainfuck (E46 generator), pointer chasing / parity / graph connectivity, the ARC
   replica games, or a mix.
4. **Sequence mixer:** start with attention (simplest, known) and swap in Spotlight once NTA-M validates our
   implementation — or build on Spotlight from the start?
5. **Code:** a self-contained model file in this folder, or import `experiments/transformers/h1_lid.py`?
6. The planning document's own questions: `CURRICULUM_LESS_COCONUT_AND_SEARCH.md` §9.

**Answered 2026-10-06** (the user agreed to these recommendations):
1. **From scratch, on synthetic tasks.** "Semantic drift" (Gemini #4) is therefore not a problem this experiment has.
2. **Outcome-only.** Machine execution states are PROBES, never training targets: as a target, a tape is a trace, and
   the learning-theory lens (planning doc §19.1) says intermediate targets are exactly what makes the problem easy —
   using them would answer a different question.
3. **Pointer chasing first** — the hop count is the depth knob, and one wrong hop sends the endpoint to an effectively
   random node, so it is the hard, parity-like case. **Then Brainfuck** from the E46 generator.
4. **Attention first;** Spotlight once NTA-M validates our implementation. P1(b) tests the learning signal, and changing
   the sequence mixer at the same time would confound it.
5. **Import `experiments/transformers/h1_lid.py`** — it already has the loop, the boundary operator and AttnRes (E30).
6. The planning document's §9 — answered the same day there.

(Resolved 2026-10-03: the Spotlight post and code were read once the network allowed it; the EfficientZero V2
notes were re-made from the paper as `refs/efficientzero_v2_2403.00564.md`.)
