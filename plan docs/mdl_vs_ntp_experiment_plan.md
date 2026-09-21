# Experiment plan: MDL compression-progress objective vs. next-token prediction

## 0. Operating instructions (for the executing agent)

Execute this plan end to end in this repository without pausing for confirmation.

- Work through the phases in §8 in order.
- Stop early only for a STOP condition (§11).
- Where the plan underspecifies something, pick the simplest option that keeps every arm identical except for the differences in §4. List each such choice under "Deviations" in your final message (§12).

Ground rules:

1. **No summary files.** Do not write implementation summaries, READMEs, changelogs, or any markdown/text file describing what you built or found.
   - Permitted outputs: code, configs, tests, run logs (jsonl/csv), checkpoints, and the tables/plots listed in §10.
   - The only write-up is your final chat message.
2. **Location.** Work in `experiments/mdl_vs_ntp/`. If the repo is under git, work on a new branch `exp/mdl-vs-ntp` and commit at phase boundaries.
3. **Core model code.** Do not modify it; wrap it (§3). If a core change is unavoidable (causality, embedding access, vocab or max-length config), keep it minimal and report it.
4. **Shared setup.** All models train from scratch. Every arm shares:
   - architecture and parameter count
   - tokenizer and data generator
   - evaluation code and decoding
   - init seed for a given seed index
5. **Hardware.** RTX 3050 Ti, 4 GB VRAM. The host may be Windows, so:
   - use `pathlib`
   - no fork-based multiprocessing; `num_workers=0`
   - Python entrypoints only (no shell scripts)
   - no `torch.compile` unless the repo already uses it successfully
   - bf16 autocast, with fp32 fallback if the model is unstable
6. **Environment.** Use the repo's existing Python environment. Install only small missing packages (numpy, pyyaml, matplotlib, pytest).
7. **Long jobs.** Run them as detached background processes that log to disk, then poll the logs. Never depend on a single blocking call for multi-hour work.
8. **Honesty.** Report null and negative results plainly. The goal is to test the objective, not to make it win.

## 1. Background (read once)

Next-token prediction (NTP) rewards tokens as *ends*: it compresses the sample as given. The hypothesis under test is that a two-part MDL objective, which compresses the *source* that generates the data, drives competence and discovery together:

```
J(θ, Λ) = L(Λ) + Σ_f  E_{y ∈ Y_f} [ L(y | Λ) ]
```

- Λ is a library of reusable program fragments (macros) proposed by the model itself.
- Y_f is the set of verified solutions for task family f.
- L is description length in bits.

A proposed macro m is rewarded by ΔJ(m), the amount it shortens verified solutions of *other* families. That is its counterfactual effect on the cost of solving future problems, which makes the model's tokens *means* rather than ends.

Four mechanisms make up the objective. Each is isolated by an arm in §4.

1. **Solution term.** The model trains on the *shortest* verified solution under the current library, not an arbitrary correct one.
2. **Library term with artifact-level credit.** Proposed macros receive a terminal REINFORCE reward ΔJ/L(m). ΔJ is scored on families the proposal never saw (anti-leakage).
3. **Epiplexity-gain curriculum.** Families are sampled by the learner's own prequential learning progress. Progress is zero both for structure a bounded learner cannot absorb (pseudorandom "noisy-TV" families) and for structure it already absorbed. References:
   - Finzi et al. 2026, *From Entropy to Epiplexity* (arXiv 2601.03220)
   - Zhang et al. 2026, *Intelligence from Learnable Novelty* (arXiv 2607.18433)
   - *Epiplexity Guided Data Selection and Generation* (arXiv 2608.11746)
4. **Replay scoring.** ΔJ is computed by re-segmenting the archive of verified solutions under a candidate library. This is a replay-simulator estimate that needs no model calls (cf. Dream-RSI, Zheng et al., Sep 2026). Lineage: DreamCoder / Stitch / LILO library learning, and Schmidhuber's compression progress.

**Why a small program DSL.** It makes "artifact", "reuse", and "verification" measurable at 1–10M parameters. The claim under test is about the objective, not about symbolic libraries.

**Where the bounded learner enters.** ΔJ itself uses a cheap symbolic coder. The bounded learner enters in two places:
- the curriculum, through its own measured progress;
- usage pruning: a macro survives only if the model actually uses it to find solutions.

## 2. Environment (`env/`)

### 2.1 DSL

Values are integers 0–9. Lists have length ≤ 8. Inputs have length 4–8 with uniform values.

| op | arity | semantics | generator arg range |
|---|---|---|---|
| `add k` | 1 | x → (x+k) mod 10 | 1–9 |
| `mul k` | 1 | x → (x·k) mod 10 | {2,3,4,6,7,8,9} |
| `neg` | 0 | x → (10−x) mod 10 | — |
| `rev` | 0 | reverse | — |
| `sort` | 0 | ascending | — |
| `rot k` | 1 | rotate left by k mod len (empty → empty) | 1–7 |
| `take k` | 1 | first k elements | 1–7 |
| `drop k` | 1 | drop first k elements | 1–3 |
| `fgt k` | 1 | keep x > k | 0–6 |
| `feven` | 0 | keep even values | — |
| `uniq` | 0 | drop repeats, keep first occurrence | — |
| `csum` | 0 | prefix sums mod 10 | — |
| `diff` | 0 | (x[i+1] − x[i]) mod 10, length n−1 | — |
| `swp` | 0 | swap pairs (0,1), (2,3), … | — |

Semantics must be total for any digit argument 0–9 (e.g. `take 0` → empty, `add 0` → identity), because sampled programs may use any digit.

**Planted latent macros** (`?` = hole). No model is ever told these names. Ground-truth programs are always written fully expanded into primitives.

- Train set (P0–P9): `sort rev` · `fgt ? sort` · `csum take ?` · `rev rot ?` · `swp rev` · `diff uniq` · `neg add ?` · `mul ? fgt ?` · `feven csum rev` · `uniq sort take ?`
- Held-out only (Q0–Q1): `add ? swp csum` · `drop ? neg sort`

Unit-test that all 12 are semantically distinct from each other and from every single primitive. Use a fingerprint over 64 fixed inputs with canonical hole arguments.

**Banned macro adjacencies** (never adjacent in training; required in H-comp): (P0,P2), (P1,P5), (P2,P7), (P3,P8), (P4,P9), (P5,P6), (P6,P0), (P7,P3).

### 2.2 Families and splits

A **family** is a template: a sequence of units.
- Each unit is a planted macro with p = 0.6 (uniform over the split's allowed set), or a primitive with p = 0.4 (uniform over the 14 ops).
- Each argument is either *fixed* in the template (p = 0.5, drawn from the generator range) or *free* (resampled per instance).

An **instance** is a template plus an argument assignment, 4 demo inputs, and 1 query input.

| split | families | units | constraints |
|---|---|---|---|
| train-structured | 420 (294 labeled, 126 unlabeled) | 1: 15%, 2: 40%, 3: 45% | P0–P9 only; no banned adjacency; no Q pattern |
| train-noisy | 90 | — | output = keyed hash of input |
| train-partial | 30 | — | output = keyed permutation of input |
| H-fresh | 100 | same as train | same grammar as train; disjoint from train |
| H-comp | 100 | 2–3 | contains ≥ 1 banned adjacency |
| H-depth | 100 | 4–5 | train grammar otherwise |
| H-novel | 60 | 2–3 | exactly one unit is Q0 or Q1 |

- Labeled families expose ground-truth programs for training.
- Unlabeled families expose only I/O, to every arm.

**Template validity** (checked on 64 random draws):
- non-empty output in ≥ 75% of draws;
- output ≠ input in ≥ 50%;
- ≥ 10 distinct outputs;
- no redundant primitive adjacency in the expanded program: `rev rev`, `neg neg`, `sort sort`, `uniq uniq`, `feven feven`, or a same-op pair of add / mul / rot / fgt / take / drop;
- semantic fingerprint distinct from every accepted template in *every* split. The fingerprint is the outputs on 64 fixed inputs with free args set to canonical values (add 3, mul 3, rot 2, take 3, drop 1, fgt 2), plus the free-arg count.

If a unit-count quota can't be met after 20k attempts, reallocate to the neighboring count and log it.

**Leakage rule.** Match patterns against the *expanded* template, where a fixed or free argument both match a hole. Train, H-fresh, and H-depth templates must not contain any banned pair's expanded pattern or any Q0/Q1 pattern. This includes occurrences assembled from primitive units.

**Instance validity.** At least 4 of the 5 outputs are non-empty, and there are at least 2 distinct demo outputs. Otherwise resample inputs (≤ 20 tries), then resample arguments.

**Noise.** Implement both functions yourself; never use Python's `hash()`.
- noisy: `h = splitmix64(key ^ polyhash(x))`, then `y_i = splitmix64(h + i) % 10` for `i < len(x)`.
- partial: a Fisher–Yates permutation of x driven by a splitmix64 stream seeded with `h`.

**Determinism.** Everything derives from `(env_seed=1234, split, family_id, instance_seed)`. Generate splits once into `env_cache/` and share them across all runs. Also pre-generate these fixed sets:
- probe set: 4 instances per train family;
- scoring archive: 8 instances per labeled family, with ground-truth programs;
- the eval sets in §9.

### 2.3 Tokens and formats

Vocabulary (73 tokens), in this order:
- `PAD BOS EOS`
- `<T> <I> <P>`
- `<ex> <in> <out> <q> <ans> <prog> </prog> <def> </def> ? _`
- digits 0–9
- the 14 ops
- 32 macro slots `M0–M31`

Every list is written **fixed-width**: 8 slots, right-padded with `_`. As a result, all prompts of a given format have the same length and no padding masks are needed.

```
T  (98 tok):  BOS <T> [<ex> <in> x×8 <out> y×8]×4 <q> <in> xq×8 <ans> yq×8 EOS
I  (≤108):    BOS <I> [<ex> <in> x×8 <out> y×8]×4 <prog> PROGRAM </prog> EOS     (prompt = 79 tok)
P  (≤176):    BOS <P> [<prog> PROGRAM </prog> _…]×5 <def> DEF </def> EOS          (each context slot exactly 30 tok)
```

**PROGRAM.** Units in order. Each op or macro token is followed by its argument digits. Budget: 28 tokens including `</prog>`.

**DEF.** 2–4 units. Argument positions take a digit or `?`, with at most 2 holes. The expanded pattern must be ≤ 8 ops.

**Loss masks.**
- T: all 40 output slots plus EOS.
- I: program tokens, `</prog>`, and EOS.
- No loss on inputs, structure tokens, or PAD.
- P is never trained with NTP.

**Probe NLL** (bits/token) covers the output slots of demos 2–4 plus the answer (32 tokens). **Noisy-digit NLL** is the same quantity restricted to digit targets.

### 2.4 Grammar-constrained decoding (`env/grammar.py`)

A per-sequence state machine returns an allowed-token mask. Apply it to the logits before temperature.

**Program states:**
- `unit` allows ops, active macro slots, and `</prog>`.
- An arity-k token is followed by k `arg` states (digits only).
- When the remaining budget only fits the pending args plus `</prog>`, force completion.
- After `</prog>`, emit EOS, then PAD.

**Def states:** the same, except:
- args allow digits ∪ `?`; mask `?` once 2 holes are used;
- `</def>` is allowed only after ≥ 2 units and is forced after 4.

**Other masks:**
- Inactive macro slots are always masked. For ntp, exit, and mdl_nolib, *all* slots are inactive.
- T-format answer decoding (E1) is masked to digits ∪ `_`.

**Test:** 10k random masked samples must all parse, expand, and execute.

## 3. Applying the objective to our architecture (`model_adapter.py`)

### Phase 0 recon

Find the model class and config that the repo currently treats as its baseline, with all current modifications enabled. Record:
- forward signature and outputs
- causality
- positional scheme and max length
- tied or untied embeddings
- auxiliary losses
- multi-exit, deep-supervision, halting, or loop-count knobs
- KV-cache support
- whether a prequential/epiplexity probe utility already exists (e.g. from the learnability-diagnostic work)

Choose width and depth to land at **~4M params** with vocab 73 and max_len ≥ 192. The sweep size is 4M; Tier 3 adds 1M and 10M.

### Adapter interface

```python
class ModelAdapter(nn.Module):
    def forward(self, ids) -> tuple[Tensor, Tensor | None]      # logits [B,T,V], aux_loss
    @torch.no_grad()
    def generate(self, prompts, max_new, temperature, grammar) -> Tensor   # equal-length prompts
    def seq_logprob(self, ids, gen_mask, grammar) -> Tensor     # Σ log π over generated tokens, with grad,
                                                                # from the SAME masked distribution used to sample
    def init_slot(self, slot_id: int, from_ids: list[int]) -> None
    def set_loop_depth(self, n: int) -> bool                    # False if the architecture has no such knob
    def num_params(self) -> int
```

### Integration rules

1. **Outer loop only.** The objective wraps the model. The model needs only causal LM forward, sampling, sequence log-probs, and embedding-row init. Core blocks stay untouched.
2. **Causality test** (required). Perturb tokens at positions > t; logits at positions ≤ t must be unchanged (max abs diff < 1e-5 in fp32).
3. **Existing losses and inference paths.**
   - If the architecture has deep supervision, multi-exit, halting, or an aux loss, the NTP anchor uses the repo's standard multi-exit loss plus aux loss, identically in every arm.
   - Sampling, probe NLL, and `seq_logprob` all use the distribution the repo uses at inference.
4. **Time bound.** The per-token compute (width × depth × loops) is the observer's time bound. Keep it fixed during training in every arm. If `set_loop_depth` works, E6 (§9) sweeps it at eval time.
5. **`init_slot`.**
   - On acceptance, set the slot's input embedding to the mean of its expansion's op embeddings. Do the same for the output row if the head is untied.
   - On pruning, re-initialize the row with the model's init distribution.
6. **Generation.** Use the KV cache if the model has one; otherwise recompute the prefix at each step. Chunk generation to ≤ 1024 sequences.
7. **Optimizer.**
   - Use the repo's baseline AdamW. Do not enable experimental optimizer variants, such as the item-52 SNR preconditioner, which showed no gain in fresh-sample regimes and would confound the comparison.
   - Settings: betas (0.9, 0.95), grad clip 1.0, batch 64.
   - LR schedule is a function of consumed compute fraction (§5): 2% warmup, then cosine decay to 10% of peak.
8. **Probe utility.** If one already computes per-token bits, reuse it for probe NLL.

## 4. Arms

| arm | family sampling | wake / self-training | induction target | library + proposer | ΔJ scoring set |
|---|---|---|---|---|---|
| `ntp` | uniform | no | ground truth | no | — |
| `exit` | uniform | yes | one random verified program | no | — |
| `mdl` | epiplexity gain | yes | MDL-shortest verified | yes | disjoint families, family-averaged |
| `mdl_nolib` | epiplexity gain | yes | MDL-shortest (primitives only) | no | — |
| `curio` | prediction error | yes | MDL-shortest verified | yes | disjoint, family-averaged |
| `mdl_insample` | epiplexity gain | yes | MDL-shortest verified | yes | the proposal's own context families, instance-summed |

What each comparison isolates:
- `ntp` vs `mdl`: the whole objective.
- `exit` vs `mdl`: compression vs. "reward correct answers" (ends-only self-training).
- `mdl_nolib` vs `mdl`: the library/invention term.
- `curio` vs `mdl`: the noisy-TV claim.
- `mdl_insample` vs `mdl`: anti-leakage scoring.

**Batch of 64:**
- `ntp`: 64 supervised sequences.
- Other arms: 48 supervised + 16 from the self-found buffer (all 64 supervised while the buffer is empty).

**Supervised sequences.** Sample the family from the arm's distribution and generate a fresh instance.
- Labeled families use format I with p = 0.5, otherwise T.
- Unlabeled, noisy, and partial families always use T.
- In library arms, ground-truth targets are rewritten under the current Λ at batch time.

## 5. Compute accounting and main loop

**TE (token-equivalents):**
- every training or REINFORCE step: 3 × its non-PAD tokens;
- every forward-only pass: 1 × tokens. This covers probes, plus wake and proposer sampling counted as prompt + generated tokens (cached-equivalent).

**Excluded from TE:**
- CPU work (execution, DP scoring). Log it as CPU-seconds instead.
- In-training logging and eval.

**Budget.** Every run stops at the same budget `B`, so ntp takes more optimizer steps than the self-training arms; comparisons are at matched compute.
- In Phase 0, measure ntp throughput and set `B` so one ntp run takes about 30 min of wall-clock.
- Record `B` in `config.yaml`.

```
probe_all() if arm uses a curriculum          # baseline at TE = 0
while TE < B:
    if arm != "ntp" and TE crossed the next 5%-of-B boundary:     # 20 rounds
        if curriculum arm: probe_all(); curriculum.update()
        wake()
        if library arm: propose_and_update_library()
    batch = make_batch(arm)
    loss = masked_ce(model(batch)) + aux
    optimizer_step(loss); TE += 3 * nonpad(batch); lr = schedule(TE / B)
    every 50 steps log: step, TE, lr, losses by format/source, grad_norm, tok/s,
                        batch composition by category (labeled by unit count, unlabeled, noisy, partial)
checkpoint at 25/50/75/100% of B: model, optimizer, RNG states, library, curriculum, buffer (runs must be resumable)
```

## 6. The objective, precisely (`mdl/`)

### 6.1 Description lengths (`library.py`)

**Macro representation.** A macro stores its fully expanded primitive pattern as a sequence of `(op, argspec)` pairs, where argspec is either `const v` or `hole i`.
- Every `?` is a distinct hole.
- Nested macro references are expanded at acceptance.
- Record acceptance order.

**`segment(prog, Λ)`.** Dynamic programming over the primitive sequence. At each position, emit either a primitive or any macro whose pattern matches: holes bind the arguments, consts must match exactly. Unit costs:
- each unit (op, macro, or terminator): c_u = log2(14 + |Λ| + 1);
- each emitted argument: c_a = log2(10).

**Description-length quantities.**

| quantity | definition |
|---|---|
| `L(y \| Λ)` | min over segmentations of units·c_u + args·c_a + c_u (terminator) |
| `L_def(m)` | see below |
| `J_S(Λ)` | Σ_{f∈S} mean_{y∈Y_f} L(y \| Λ) + Σ_{m∈Λ} L_def(m) |
| `ΔJ_S(m)` | J_S(Λ) − J_S(Λ ∪ {m}) |
| `loo(m)` | J_S(Λ \ {m}) − J_S(Λ) |
| `support(m)` | number of distinct families in S whose best segmentation uses m |

`L_def(m)`:
- Segment m's pattern with the macros accepted before m, using c_u' = log2(14 + |Λ| + 2).
- Each argument slot (hole or const) costs log2(11).
- Add one terminator unit.
- For candidates, segment with all of the current Λ.
- For library members, recompute every round.

**Scoring speed** (target < 10 ms per candidate):
- Encode programs as strings and prefilter each candidate with a regex built from its pattern.
- Re-segment only the programs that match.
- For non-matching programs, apply the c_u change analytically from cached (units, args) counts.
- Dedupe candidates and cache scores by canonical pattern.

### 6.2 Epiplexity-gain curriculum (`curriculum.py`)

This is a prequential learning-progress estimator of epiplexity gain. The update runs every round for all 540 train families:

```
ℓ_f(r)  = probe NLL of family f at round r;  ℓ_f^min = min over earlier rounds
progress = max(0, ℓ_f(r−1) − ℓ_f(r) − 0.01)          # hysteresis against parameter drift
relearn  = max(0, ℓ_f(r) − ℓ_f^min − 0.05)            # structure once absorbed, now forgotten
g_f      ← 0.7·g_f + 0.3·(progress + relearn)
q        = stable_softmax(g / (0.5·std(g) + 1e-8))
q        = water_fill_cap(q, 0.05)                   # iterative cap at 5% per family
p        = 0.9·q + 0.1·uniform
```

- `curio` uses the identical pipeline with g_f replaced by s_f = ℓ_f(r), the current prediction error.
- Diagnostic only: log a p(1−p) proxy from each family's wake solve rate (EMA) and its rank correlation with p.

### 6.3 Wake (all self-training arms)

**Sampling each round:**
- Draw W = 256 tasks: family from the arm's distribution, fresh instance.
- Sample N = 16 programs per task at T = 1.0, with grammar masking and the arm's active library.
- Expand and execute each program, then verify it on all 4 demos plus the query.

**Buffer.**
- Entries are keyed by `(family_id, arg assignment)`. Each entry holds verified expanded programs:
  - `exit`: 1 random program;
  - other arms: up to 4 distinct programs.
- Cap at 30k entries, FIFO.

**Using the buffer at batch time.**
1. Generate fresh inputs with the stored arg assignment and compute true outputs with the family generator.
2. Drop any stored program that disagrees on the fresh I/O (a spurious solution).
3. Pick the target:
   - library arms: argmin L(y | Λ), emitted in library form;
   - `mdl_nolib`: the shortest primitive program;
   - `exit`: its stored program.

**Logging.** Log verified rate by category and unit count, plus per-macro usage among verified programs (needed for usage pruning).

### 6.4 Proposer and library update (`mdl`, `curio`, `mdl_insample`)

1. **Archive.** For each structured train family with known programs, collect up to 8 programs:
   - labeled families: the 8 ground-truth programs from the scoring archive;
   - unlabeled families: up to 8 best buffer programs.

   Each round, randomly split these families 50/50 into a proposal set P_r and a scoring set S_r. Held-out families never appear here.
2. **REINFORCE** (8 steps × 64 contexts per round).
   - **Context:** 5 programs, one instance each, from 5 distinct P_r families, written in current library form.
   - **Sample:** one DEF per context at T = 1.0.
   - **Reward:**
     - invalid → −1;
     - semantically equal to a primitive or an existing macro (by fingerprint) → 0;
     - otherwise `clip(ΔJ(m) / L_def(m), −1, 5)`.
   - **ΔJ set:** S_r, family-averaged. In `mdl_insample`, use the 5 context families' archive programs, instance-summed, with no family averaging.
   - **Advantage:** RLOO (r_i − mean of the others), normalized by batch std.
   - **Loss:** `β · mean_i(−A_i · seq_logprob_i)` with β = 0.1, applied as a separate optimizer step on the shared weights.
3. **Accept.** Consider all candidates scored this round and greedily accept up to 2. A candidate qualifies if:
   - ΔJ > 0, and
   - support ≥ 3 families (≥ 1 in `mdl_insample`).

   After each acceptance, re-score the next candidate. On acceptance, call `init_slot`. When the library is full (32 macros), a candidate replaces the lowest-`loo` macro only if its ΔJ exceeds that `loo`.
4. **Prune.** Remove a macro if either:
   - `loo ≤ 0` on S_r (in `mdl_insample`: on P_r, instance-summed), or
   - it appears in no wake-verified program for 3 consecutive rounds after its acceptance round. This is the bounded-learner check.
5. **Diagnostics** (never used for acceptance).
   - Symbolic miner: take all contiguous 2–4-op windows in S_r programs, with up to 2 args turned into holes. Keep the top 200 by family support and score them.
   - Write to `rounds.jsonl`:
     - `miner_best`
     - best proposed ΔJ
     - `proposer_regret = miner_best − best proposed ΔJ`
     - proposal validity rate and mean reward
     - library contents with ΔJ / loo / support
     - J_S
     - accept and prune events

## 7. Tests (`tests/`, pytest; all must pass before Phase 4)

**Executor**
- Hand-written cases for every op.
- Totality for arguments 0–9.

**Segmentation**
- `expand(segment(p, Λ)) == p` for 5k random programs and random libraries.
- DP optimality vs. brute force on programs of ≤ 6 ops with ≤ 4 macros.

**Scoring sanity** (archive from 60 train families)
- ΔJ > 0 for ≥ 8 of the 10 planted train macros.
- ΔJ ≤ 0 for ≥ 90% of 200 random 3-op fragments that match no planted macro.
- Leakage check: a macro equal to one family's full template has in-sample ΔJ > 0 (its own family, instance-summed) and ΔJ ≤ 0 on disjoint families.

**Splits**
- No held-out template or fingerprint appears in train.
- The banned-pair and Q-pattern rules from §2.2 hold.
- Quotas are met, or the reallocation is logged.
- Regenerating the splits is deterministic.

**Formats and grammar**
- Formats have fixed lengths, and loss masks cover exactly the specified positions.
- The grammar passes the 10k-sample test from §2.4.

**Adapter**
- Causality test passes.
- `init_slot` changes only the intended row(s).

**Curriculum**
- On synthetic loss traces, a flat-high trace (noise) gets only floor mass under the gain rule and top mass under `curio`.

## 8. Phases

**Phase 0 — Recon and adapter.**
- Complete §3.
- Measure ntp throughput and set `B`.

**Phases 1–2 — Build the components.**
- Environment (§2), library, curriculum, wake, and proposer (§6).
- Tests (§7).

**Phase 3 — Smoke test.** Run each arm at 2% of `B`, with rounds every 0.5%. Check:
- no NaN or Inf;
- TE stops within ±2% of target;
- the round, buffer, and library code paths all execute;
- resuming from a checkpoint works.

**Phase 4 — Pilot** (budget B/3, seed 0).
1. LR sweep on `ntp` only: {3e-4, 1e-3, 3e-3}. Pick the best by ID transduction EM and use it for every arm. Tuning only on ntp is deliberately conservative toward mdl.
2. Run `exit`, `mdl`, and `mdl_nolib` at the chosen LR.
3. Check the gates:

| gate | condition | if it fails |
|---|---|---|
| G1 | ntp ID transduction EM ≥ 30%, or loss still clearly falling | If still falling: double `B`. If flat and < 30%: try the 10M config once, then STOP. |
| G1b | ntp ID-labeled verified@16 ≥ 20% (quick E3 on 100 tasks) | Same remedy as G1. |
| G2 | mdl wake verified rate on labeled families ≥ 5% by round 4 | Debug decoding and execution before continuing. |
| G3 | mdl accepts ≥ 1 macro by round 8 | See below. |
| G4 | noisy-digit NLL ≥ 3.0 bits in every arm | Lower means the hash leaks: fix the generator and restart the pilot. |
| G5 | mdl ID EM no more than 5 points below mdl_nolib | Set β = 0.03 and rerun the mdl pilot. |
| G6 | mdl_nolib ID EM no more than 5 points below ntp | Raise the uniform mix to 0.2 for all curriculum arms and rerun their pilots. |

   If G3 fails:
   - `miner_best > 0` but no proposal ever scores > 0: add a proposer warm start to *all* library arms (200 supervised P-format steps whose target is a random contiguous 2–3-op window from a context program), and report it as a deviation.
   - `miner_best ≤ 0`: there is a cost bug; fix it.
4. Project full-matrix wall-clock (training plus eval) from the pilot timings. If it exceeds 16 h, reduce in this order before touching `B`:
   1. Tier 2 → 2 seeds.
   2. Tier 3 → seed 0 only.

**Phase 5 — Full matrix.**
- `run_matrix.py` must be resumable and skip any run that already has a final checkpoint.
- Launch it detached and poll `runs/matrix.log`.

| tier | runs |
|---|---|
| 1 | `ntp`, `exit`, `mdl` × seeds {0, 1, 2} |
| 2 | `curio`, `mdl_nolib` × {0, 1, 2} |
| 3 | `mdl_insample` × {0, 1}; plus E6 size runs (`ntp` and `mdl` at ~1M and ~10M, seed 0), only if the architecture has no loop-depth knob |

**Phase 6 — Evaluation and analysis.** §9 and §10.

## 9. Evaluation (`evaluate.py`)

Evaluation code is identical for all arms and runs on final checkpoints unless noted.
- Eval instances and per-task sampling RNG seeds are fixed and shared across arms.
- Held-out families never touch training, probes, archives, buffers, or proposer contexts.

**E1 — Transduction exact match.** Greedy, masked decoding; the full 8-slot answer must match.
- Instances:
  - ID-structured: 2 fresh instances × 420 families;
  - each H split: 8 instances per family.
- Report by split and by unit count.
- Also run at the 25/50/75% checkpoints, for learning curves vs. TE.

**E2 — NLL.**
- Probe NLL per split.
- Noisy-digit NLL and partial-family NLL, each on 4 fresh instances per family.

**E3 — Induction search.**
- Tasks:
  - 2 instances per H family;
  - 1 instance for each of 150 labeled and 126 unlabeled train families.
- Sampling: waves of 4, 4, 8, 16, 32, 64 (128 total) at T = 1.0, using the arm's final library.
- The answer is the first demo-consistent program. Stop a task only when a sample is both demo-consistent *and* query-correct.
- Metrics:
  - verified@k and oracle pass@k for k ∈ {1, 4, 16, 64, 128};
  - samples-to-first-correct (median, plus the fraction censored at 128);
  - primitive length of the first correct program;
  - macro usage rate.
- If search eval exceeds ~10 min per checkpoint, drop to 1 instance per H family and note it.

**E4 — Library** (library arms only).
- Planted-macro recovery:
  - exact pattern match;
  - partial match: a const where the planted macro has a hole, or vice versa.
- `precision_aligned`: fraction of library macros that are contiguous sub-patterns or concatenations of planted macros.
- `family_specific_rate`: fraction of macros whose support across all train archives is ≤ 2 families.
- Trajectories of library size, J_S, reward, and regret.

**E5 — Curriculum.** Per round, report probability mass and realized batch share by category.

**E6 — Time bound.** Run E1 and E2 for ntp and mdl across compute levels:
- if `set_loop_depth` works: depths {1, train/2, train, 2×train};
- otherwise: the Tier 3 size runs.

Expected: noisy-digit NLL stays flat near log2(10) ≈ 3.32 at every compute level, while structured NLL falls as compute rises.

**E7 — Adaptation** (stretch; only after everything else is done). Use H-comp and H-depth.
- Each arm gets 3 extra rounds of its own wake/self-training on held-out I/O only, at equal TE. ntp uses the exit procedure.
- Report the gain in verified@16 per TE.
- The adapted models are separate and never feed into E1–E6.

## 10. Analysis (`analyze.py` → `results/`)

**Statistics.** Two-level bootstrap: resample seeds, then tasks within each seed; 10k reps. Arm-vs-arm deltas are paired by seed index. Report means with 95% CIs.

**Tables** (`results/tables/`):
- `main.csv`: arm × split × metric.
- `paired_mdl_vs_ntp.csv`, `paired_mdl_vs_exit.csv`, `paired_mdl_vs_nolib.csv`.
- `library.csv`, `curriculum.csv`.
- `compute.csv`: per run, TE, optimizer steps, wall-clock, and CPU-seconds.
- `predictions_check.csv`.

**Plots** (`results/plots/`):
- verified@k and pass@k curves per split, with CI bands;
- ID EM vs. TE;
- curriculum mass by category over rounds (mdl vs. curio);
- library size and J_S over rounds;
- proposer reward and regret over rounds;
- noisy-digit NLL by arm;
- the E6 compute-bound plot.

**Pre-registered predictions.** Verdict rules for each delta:
- PASS: the CI excludes 0 (or clears the stated margin) in the predicted direction.
- FAIL: it excludes it in the opposite direction.
- INCONCLUSIVE: otherwise.

Write every sub-comparison as its own row. A composite is PASS only if all of its parts pass.

| id | prediction |
|---|---|
| H1 competence | ID transduction EM: lower CI of (mdl − ntp) ≥ −2 pts |
| H2 search efficiency | verified@16 on H-comp *and* H-depth: mdl > ntp *and* mdl > exit |
| H3 discovery | oracle pass@128 on H-depth: mdl > ntp |
| H4a invention | mdl recovers ≥ 5/10 planted macros (exact or partial) in ≥ 2 of 3 seeds |
| H4b library matters | H-comp verified@16: mdl > mdl_nolib |
| H4c anti-leakage | family_specific_rate(mdl_insample) > family_specific_rate(mdl), and H-comp verified@16: mdl > mdl_insample |
| H5 noisy TV | Noisy-family share of supervised sampling over the whole run: curio ≥ 30% and mdl ≤ 8% (uniform = 16.7%) |
| H6 no harm | H-novel verified@16: lower CI of (mdl − ntp) ≥ −3 pts |

**Interpretation rules** (apply these in the final message):
- H2 passing without H3: the objective improves search efficiency (sharpening), not the support of what can be found.
- mdl ≈ exit: the gain comes from self-training, not from compression.
- H2 passing while H4b fails: the curriculum is doing the work, not invention.

## 11. STOP conditions

Stop and report if:
- causality cannot be guaranteed even with a minimal core change;
- OOM persists at batch 8;
- NaNs persist after halving the LR twice;
- G1 still fails after the 10M attempt.

Bugs in code this plan asks you to write are not STOP conditions. Fix them.

## 12. Final chat message (the only write-up)

1. **Setup actually used:** param count, model config, `B`, LR, seeds completed, total wall-clock.
2. **Main table:** one row per arm, with columns:
   - ID EM;
   - verified@16 on H-fresh, H-comp, H-depth, and H-novel;
   - H-depth pass@128;
   - median samples-to-solve;
   - noisy-digit NLL.
3. **Predictions:** the §10 table with PASS / FAIL / INCONCLUSIVE and the numbers behind each verdict.
4. **Library and curriculum:** mdl seed 0's final library next to the planted macros, and the curriculum mass curve for mdl vs. curio, summarized in numbers.
5. **Deviations:** every departure from this plan, every core-code change, and any STOP.
6. **Interpretation:** two or three sentences, following the rules in §10.
