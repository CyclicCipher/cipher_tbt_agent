# Experiment 1 — the dagger memory against the reversal curse (constraint C1)

**Pre-registered 2026-10-18, before any result run.**

Before writing this I ran three smoke tests (300–500 steps, not saved) to check speed and the code path:
- a 6000-step run takes about 47 s;
- after 500 steps, both MLP kinds were at about 0.008 hits@1 on one-way reverse questions (chance 0.0012), so no early
  sign either way.

## Question

The dagger memory reads its MLPs backwards through the transpose of the same weights. Does that let a small transformer
answer reversed questions about facts it saw only forwards? It first has to learn, from other facts seen both ways, which
token names each converse.

## Why it might work

A plain MLP is a key–value memory: y = V σ(Kx).
- Training on "a r → b" shapes a unit's key (its row of K) to match the cue for (a, r), and its value (its column of V) to
  write b.
- Nothing in that update can be read from b back to a. This is the reversal curse's one-way update (Zhu et al. 2024).

The dagger MLP adds y += Kᵀ σ(Vᵀx), the same parameters read backwards.
- A cue containing b now activates the unit through its value, and the unit writes back its key, which contains a.
- What remains to learn is shared across facts:
  - the attention must carry b's identity unchanged, so the cue matches the value;
  - the converse token must route the question into this path.
- Both can be learned from the paired facts and should transfer to the one-way facts.

## Data (`data.py`, facts fixed by data seed 0)

- **Entities:** 1000 single-token entities.
- **Relations:** 6, each with its own converse token. Two of each kind: one-to-one, many-to-one, one-to-many. That gives
  2,999 facts, and a vocabulary of 1,012 tokens.
- **Prompt format:** two tokens. The forward prompt is [a, r] with answer b; the reverse prompt is [b, r°] with answer a.
- **Split.** Reverse questions are grouped by their key (r, b). Half the groups are paired: both directions trained. The
  rest are one-way: forward only. No reverse prompt of a one-way group is ever trained.
- **Training prompts:** 4,523 (every forward fact, plus the reverse of every paired fact).
- **Evaluation sets:**
  - `fwd` (2,503 forward questions);
  - `rev_paired` (1,264);
  - `rev_oneway` (1,232);
  - **`rev_oneway_clean` (1,227), the PRIMARY set:** one-way questions none of whose answers is linked to b by any other
    fact, so "name something linked to b" cannot score by coincidence.
- **Chance:** hits@1 = 0.0012.

## Model and training (`model.py`, `train.py`)

- **Model:** causal transformer, 2 layers, d = 128, 4 heads, pre-LayerNorm, GELU MLPs, learned positions, tied
  input/output embeddings (except arm D). Initialisation as nanoGPT.
- **Optimiser:** AdamW, lr 1e-3, betas (0.9, 0.98), weight decay 0.01; 200-step linear warm-up, then constant.
- **Training:** batch 512, 6,000 steps. Loss: cross-entropy on the answer token.
- **Precision:** bf16 autocast.
- **Logging:** evaluated every 250 steps, with full curves logged.
- **Seeds:** the seed varies the initialisation and batch order; the facts stay fixed.

## Arms (`run_e1.py`)

| arm | MLP | purpose |
|---|---|---|
| A | plain, width m | baseline |
| B | dagger, width m | the hypothesis; same parameters as A, 2× the MLP FLOPs |
| C | plain, width 2m | compute control: same MLP FLOPs as B, 2× the MLP parameters |
| D | dagger, width m, untied embeddings | does the cue/value code match have to be built in (C3)? |

**Capacity (the pilot).** Arm A runs at seed 0 with m = 8, 16, 32, … (doubling, up to 1024) and stops at the first m
whose final training accuracy is ≥ 0.99. That m is **tight**, and 8× it is **ample**.

Why vary capacity: with spare units, the model can memorise each paired reverse fact separately and never learn to use
the transpose. With scarce units, reusing the transpose is the cheaper solution.

**Grid:** 4 arms × {tight, ample} × seeds {0, 1}. A configuration that two arms share (C at m equals A at 2m) is trained
once and reused. Pilot runs are reused too.

## Measures (per evaluation set)

- **hits@1:** the top entity is a correct answer.
- **mass:** probability on the correct set, with the softmax taken over entities only. This is also the set-valued read
  (C5).
- **leak:** probability on entities linked to the cue by other facts.
- **train_acc:** hits@1 over all training questions. **reach99** is the first step at which it is ≥ 0.99.
- **Dagger arms (B, D):** every measure again with the transpose read switched off at test time.

## Predictions, and what counts as support

- **P1 (baseline).** A's one-way clean hits@1 ≤ 0.02 at both capacities and both seeds. This is the reversal curse
  reproduced, allowing for a small above-chance answer prior.
- **P2 (the hypothesis).** At one capacity or more, in both seeds, B's one-way clean hits@1 is ≥ 0.10 and at least 0.05
  above C's.
  - **Supported** if met.
  - **Strongly supported** if B is ≥ 0.5.
  - **Refuted** if B ≤ 0.02 at both capacities.
- **P3 (mechanism; applies if P2 holds).** Switching the transpose read off at test time drops B's one-way clean hits@1
  to ≤ 0.02.
- **P4 (capacity).** B's one-way clean hits@1, averaged over seeds, is higher at tight than at ample.
- **P5 (code match).** D is below B, averaged over seeds. No prior on the size of the gap.
- **P6 (cost).** B reaches train_acc ≥ 0.99 at both capacities, so the constraint does not block memorisation. reach99
  is reported against A.

## Failure modes and their diagnostics

1. **The transpose path goes unused.** The model memorises paired reverse facts as new forward-path facts.
   Diagnostic: B's `rev_paired` hits@1 with the transpose off. If it doesn't drop, the path is unused.
2. **Code mismatch.** "b" arriving as a cue (after attention) does not match "b" written as a value (aligned to the
   output embedding). Diagnostic: B against D, and the size of the leak.
3. **Relation leakage.** The transpose read of b returns the subjects of every relation into b. Diagnostic: leak mass on
   one-way clean questions; it should exceed A's if this happens.

## Cost

- **Pilot:** 8 runs at most, about 30–50 s each.
- **Grid:** 16 configurations at most, minus reuse; each run ≤ 2 min.
- **Total:** under 20 minutes on the laptop GPU.

---

## Results

*(to be filled in after the runs)*
