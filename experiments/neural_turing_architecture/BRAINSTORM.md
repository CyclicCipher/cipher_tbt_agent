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

**Standalone.** This is not ZipLearn. ZipLearn's DESIGN §18 rule 1 forbids exactly what thread (3) proposes — "no
policy over thoughts" — so those rules do not bind here. What carries over is evidence: E26, E28, E30, E31, E41, E46
and the `h1_lid.py` substrate.

## 0. The architecture as currently imagined

```
tokens ─► prelude ─► e   (the anchor)
                     │
           ┌─────────▼──────────────────────────┐
           │ core block(s), applied k = 1..K     │   DEPTH axis (Ouro, Chen et al.): refine the same positions
           │   input_k = route(e, h_1 … h_{k-1})  │   ← boundary operator and/or Block AttnRes      (§1)
           │   reads/writes memory               │   ← attention today; Spotlight-like slots?      (§2)
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

---

## 1. Do Block Attention Residuals remove the need for the boundary operator?

**Short answer: for the depth axis of a trained model, mostly yes. For the sequence axis, no. For a written brain,
no.** Keep a thin boundary operator, and let attention residuals do the routing over a BOUNDED source set.

**What the boundary operator does** (Chen et al.: BO(h, e) = RMSNorm(h) + α·e between passes). Two jobs: (a) bound
the carried state's magnitude, so the stream does not grow with every pass; (b) re-inject the input every pass, the
stability condition Geiping et al. argue for. In ZipLearn's written brains `BoundaryOp` also does (c): discrete error
correction — quantise to one-hot, commit pred → state, clear scratch, halt.

**What Block AttnRes covers.** (a) A pass's input is a softmax-weighted, i.e. CONVEX, combination of its sources,
so it cannot exceed the largest source; the AttnRes paper reports bounded, periodic magnitudes, each block boundary
resetting the accumulation. (b) The anchor is source 0, so re-injection is available to every pass, weighted per
token by content. ZipLearn DESIGN §15 already calls the fixed `rms_norm(x) + α·anchor` "the one-source special
case" — the anchor plus only the last pass, at fixed weights.

**Where it falls short — six gaps.**

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
5. *Written brains need discrete operations.* No softmax mix performs an argmax re-quantisation or a clear.
   `h1_lid.LoopedModel` currently refuses `res="attnres"` together with `core_layers != 1` or a `BoundaryOp`
   (`experiments/transformers/h1_lid.py:528`).
6. *The two were never measured together.* Chen et al. measured the boundary operator; Kimi measured AttnRes (in a
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

## 2. A Spotlight-style memory in place of the attention layers

### 2.1 What is actually known — very little

The post itself could not be read from this session (percepta.ai is blocked here). From coverage only:

- "separating intelligence from memory, allowing knowledge and skills to grow without changing the model's weights";
- it replaces attention with an unbounded memory;
- every token can read and write that memory by *learning to index the specific cells it needs*, and the number of
  cells a token touches is constant however large the memory gets — explicitly contrasted with Mixture-of-Experts.

The same team (Tzamos et al.) published *Can LLMs be computers?* in March 2026: a WebAssembly interpreter compiled
into a 7-layer, d = 36 transformer with 18 two-dimensional attention heads, decoded through "HullKVCache" in
O(k + log n) per token. That is the closest thing to a mechanism we have.

**Not known:** how an address is produced (a learned key matched against cell keys? a hash? a geometric query?),
what a write does (overwrite, add, gated erase-then-write?), how cells are allocated and freed, how the discrete
selection is trained, whether position enters at all, whether any attention remains (a hybrid?), and every number.

### 2.2 Four known mechanisms that fit the stated properties

| candidate | read / write | fits | does not fit |
|---|---|---|---|
| **Sparse Access Memory** (Rae et al. 2016) | top-k cells by approximate nearest-neighbour search; writes go to the least-recently-used or just-read cells | constant k cells per step; read AND write; memory size decoupled from compute; trained end-to-end | its controller was an LSTM, not a transformer |
| **Product-key addressing** (Lample 2019; Berges 2024) | split the query in two halves, top-k against two sets of √N sub-keys, combine: exact top-k of N in O(√N) | "learns to index specific cells"; the MoE comparison is the one memory-layer papers make | in memory layers the cells are PARAMETERS; Spotlight's cells must be per-sequence STATE written by tokens — a mutable variant nobody we found has published |
| **Hull lookup** (Percepta's own HullKVCache, read through its name) | with 2-D keys, argmax_i q·k_i is attained at a vertex of the keys' convex hull, found by binary search in O(log n); writes = insert/delete points in a dynamic hull | same team; exact hard lookup; logarithmic | speculative — how that becomes "learned indexing" of an unbounded memory is not public |
| **Delta-rule fast weights** (DeltaNet, Gated DeltaNet, Kimi Delta Attention; Titans) | a d×d matrix; each token erases the old value at its key and writes a new one | mutable; linear time; strong length extrapolation | fixed capacity, and every token touches the whole state — contradicts "constant cells however large the memory". Probably NOT Spotlight, but the strongest baseline for "mutable, linear-cost memory" |

### 2.3 We already have mutable memory — on the depth axis

E46's Brainfuck interpreter keeps its tape as tokens and rewrites them IN PLACE every pass (`block B: write
tape[dp] <- register`, a `Match` from the tape side). In a depth-looped model the positions are fixed and each pass
rewrites their residuals: **the positions are a register file.** The "KV cache you can expand but cannot fix" problem
exists only along the SEQUENCE axis, where thoughts and tokens append. Two costs remain: every pass attends over all
T positions — O(T²) per pass, O(K·T²) per thought — and the memory's size is the context's size.

So, for this architecture, a Spotlight-like memory means two concrete changes: make each pass's read/write SPARSE (k
cells, not T positions), and let the sequence axis WRITE INTO SLOTS instead of appending — a thought can then
overwrite its own working notes. E41 is the same thesis from the other side: byte-identical weights, a world model
that is entirely in the context (an empty context scores 0.000). "Separate intelligence from memory" is that
result, made mutable and cheap.

### 2.4 Where it would sit

Inside the core block, in place of self-attention: each pass reads k cells, computes, writes k cells. The AttnRes
depth-mix STAYS — it is attention over a handful of depth sources, not over the sequence, so it is not where the
cost is. Once thoughts write into slots, the action of §3 grows an address part: what to write AND where — the NTM
controller in full.

### 2.5 Tests any memory candidate must pass (proposed: NTA-M)

- **M1 — binding load beyond training.** Associative recall with m pairs, trained to m₀, tested to 8·m₀.
  `experiments/binding_mqar.py` is a harness (it imports Mamba-3 from a local clone not in the repo — the mamba
  clone is gitignored).
- **M2 — overwrite.** A key-value stream with UPDATES; query the latest value. Append-only attention must learn
  recency; a slot memory overwrites. The test Spotlight's "mutable" claim is about.
- **M3 — a tape longer than trained.** Brainfuck programs (E46's generator, the 2609.30063 distribution) whose tape
  grows past the training length.
Arms: softmax attention; Gated DeltaNet; a SAM-style top-k slot memory; a product-key slot memory written per token.

### 2.6 To learn what Spotlight really is

Either add `percepta.ai` (and `arxiv.org`, which also blocked this session) to the cloud environment's allowed
domains, or save the post's text into this folder (e.g. `sources/spotlight_memory.md`). Then §2.1–2.2 get
reconciled against it.

---

## 3. Discovering latent reasoning with no curriculum and no traces

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

- **Gaussian policy over a continuous action + sampling-based Gumbel search** — candidates sampled from π(z | s),
  Gumbel-top-k + sequential halving; policy improvement is guaranteed even with a handful of simulations, and EZ-V2
  shows the guarantee holds for continuous actions. This is the core import.
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
| 4 | **Soft-embedding simplex projection**: z_t = softmax(W_p h_t / τ) · E, τ annealed from high to low | **keep, re-motivated** | "Drift from the pretrained semantic manifold" is a problem of adapting a pretrained LM; this experiment trains from scratch (or from a written brain), so that motivation is weak. The real benefits: bounded magnitude for free (the convex hull of the codebook), legibility (a thought is a readable distribution), and **τ → 0 is exactly the written `BoundaryOp`'s Quantise** (argmax → one-hot) — the trained and the written forms of one operator. Gumbel-softmax gives it stochasticity and reparameterisation together. Soft Thinking (training-free) and Soft Tokens, Hard Truths (RL) are the evidence. Cost: thoughts limited to the hull of the codebook — test a learned codebook against the token embeddings. |
| 5 | **JEPA-style state prediction**: thoughts must predict representations of future task-relevant observations | **keep for environments; probe only otherwise** | In a game, "the thought predicts the embedding of the next observation" IS EfficientZero's temporal-consistency loss (stop-gradient target) and SPR. For pure reasoning tasks the only "future observations" are execution states, which is trace supervision by another name. Use them to PROBE (can the tape be decoded from the thoughts?), not as a training target, in the main arm. |
| 6 | **Curriculum by circuit depth**, a learned continue/halt token, reward 0 when too few steps, a λ·K penalty | **keep — already in ZipLearn DESIGN §18.2** | The two-dimensional frontier (length × depth) and "depth as a speed prior" are recorded there. Concrete generator: E46's Brainfuck programs (steps per output byte = the depth a problem needs); or a self-play generator (2609.30063) scored by the learner's progress. The penalty schedule matters: λ·K from step 0 collapses to K = 0 — use Ouro's two stages (entropy-regularised exit distribution first, then tune the gate on realised gains). "Reward 0 at K = 0" needs no engineering: on problems deeper than the network it happens by itself. |

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
- **NTA-0 — is a thought ever necessary?** A task family where a K = 0, fixed-depth model provably fails (pointer
  chasing with more hops than layers; Brainfuck outputs needing more steps than passes). Arms: no thoughts; Coconut
  feedback with noise trained by RLOO (the Soft Tokens, Hard Truths recipe — the literature's baseline). Measured:
  accuracy against hops; the thought ablation of §3.6. Pass: the thought arm solves hop counts the no-thought arm
  cannot, and the ablation destroys it.
- **NTA-1 — does search beat sampling?** Sampling-based Gumbel search over thoughts (EZ-V2-style) against RLOO,
  at an EQUAL number of forward passes. Pass: deeper problems solved at the same compute. Refute: no difference.
- **NTA-M** — the memory tests of §2.5.

---

## 5. Decisions for the user

1. **Starting point.** From scratch on synthetic tasks, from a BrainBuilder-written brain (the E46 interpreter as
   initial weights — ZipLearn §21.8's gradient-compatibility clause), or from a small pretrained LM? This decides
   whether "semantic drift" (Gemini #4's motivation) is a real problem.
2. **Outcome-only, or are machine execution states allowed** as an auxiliary signal (Gemini #5)? "No human text
   traces" rules out human traces; it does not by itself rule out a Brainfuck tape.
3. **Task suite priority:** Brainfuck (E46 generator), pointer chasing / parity / graph connectivity, the ARC
   replica games, or a mix.
4. **Spotlight:** allow `percepta.ai` in the cloud environment, or paste the post into this folder.
5. **EfficientZero V2 notes:** `src/tbt/EZV2_NOTES.md` and the PDF are not in the pushed repo; push them if they
   exist locally.
6. **Code:** a self-contained model file in this folder, or import `experiments/transformers/h1_lid.py`?
