# CONTINUATION: Compositional Arithmetic Curriculum on Mamba3

**Date:** 2026-02-19 (updated)
**Priority:** IMMEDIATE — Tests the core generalization hypothesis.
**Context:** See Mistakes #42 (memorization), #38 (ePC archived), #34 (next-step prediction).

**Mamba3 backbone status:** Paper audit complete (Mistakes #45, #46, #47). Intentional deviations: PoPE (replaces RoPE), StableSSM (replaces standard A-log). **MIMO is broken** — see Mistake #47 and "IMMEDIATE: MIMO Fix" section below. Fix MIMO before running curriculum experiments with MIMO enabled.

---

## IMMEDIATE: MIMO Fix (Mistake #47)

**Priority:** Fix before any MIMO-enabled training runs.

### The Bug

The MIMO implementation in `mamba3_block.py` (lines 645-700) folds R ranks into the head dimension, creating `nheads*r` independent heads each with their own state. This **multiplies** state size and compute by R — the exact opposite of the paper's intent.

### What the Paper Says (Appendix D)

MIMO is designed to increase hardware efficiency at inference by providing more expressive I/O without growing the state:

```
State:  H_t ∈ R^(N×P) per head      ← same size regardless of R
Write:  H_t = α * H_{t-1} + B_t @ X_t^T    where B(N,R) @ X(P,R)^T = (N,P)  ← rank-R update
Read:   Y_t = H_t^T @ C_t           where H(N,P)^T @ C(N,R) = (P,R)          ← R readout vectors
Output: down-project P×R → P → D
```

Key: all R ranks share the same N×P state. The rank-R outer product `B @ X^T` is a sum of R rank-1 updates — more expressive than rank-1, but state stays N×P.

### What Our Code Does (Wrong)

- Folds R into heads: `nheads*r` effective heads, each with independent state
- State size: N × P × nheads × R (R× too large)
- Also wrong: `mimo_x_proj = Linear(d, d*r)` — paper says two-stage D→P→P×R via W_X' and W_X
- Also wrong: `mimo_out_proj = Linear(d*r, d)` — paper says P×R→P→D

**Note:** The sequential recurrence `mamba3_mimo_recurrence()` (lines 385-439) already has correct shared-state math. The bug is only in the SSD path and the projection layers.

### Fix Plan

1. **MIMO X projection** — Replace `Linear(d, d*r)` with the paper's two-stage: W_X' (d→d, or equivalently implicit in head reshape to P) then W_X (P→P×R per head, i.e., `Linear(headdim, headdim*r)`)
2. **MIMO output projection** — Replace `Linear(d*r, d)` with W_O' (P×R→P per head) then W_O (d→d_model, already exists as `out_proj`)
3. **SSD MIMO forward path** — Two options:
   - **(a) Dedicated MIMO SSD:** Modify the SSD kernel to handle rank-R writes and reads on a shared state. The intra-chunk quadratic term becomes `Y_j = Σ_i (C_j^T @ L @ B_i) @ X_i` — O(R²) cross-rank terms but on small Q×Q matrices.
   - **(b) Sequential recurrence for MIMO:** For small models (our d=128), just use `mamba3_mimo_recurrence()` when r>1. Simple, correct, adequate for our scale.
   - Recommendation: start with (b) for correctness, implement (a) later if speed matters.
4. **Verify** — Check that r=1 produces identical results before and after the change (MIMO projections should be no-ops or absent for r=1).

### Files to Modify

- `experiments/Mamba3/mamba3_block.py` — MIMO projections in `__init__`, MIMO forward path in `forward()`
- Sequential recurrence is already correct — can be used directly for option (b)

---

## The Problem

Every model we've trained memorizes instead of generalizing on multi-step algorithmic tasks:

- **Stage 2** (5-rule pattern induction): 99% train, ~25% test.
- **Associative recall** (Naja ablation): 100% train, 8-22% test.
- **Permutation_4**: 55-87% train, 25% test (chance).
- **Multi-scale memory**: 39-100% train, 7% test (chance).

The model treats composite tasks as monolithic lookup tables. With 1.26M parameters and only 5K training examples, memorization is always the path of least resistance.

## The Hypothesis: Compositional Curriculum

**Can the model learn to compose sub-skills?**

### Attempt 1: 12-Stage Curriculum (FAILED)

Root causes:

1. **Early stages trivially memorizable** — Stages 1-5 had only 9-10 problems each, all below the split threshold. The model "passed" by memorizing lookup tables, never learning algorithms.
2. **Interleaved counting gave the answer away** — `DOT 1 DOT 2 DOT 3 = 3` taught copying, not counting. The answer was in the input.
3. **Composition test too small** — Stage 6 had only 100 problems (80 train / 20 test). Heavy memorization pressure.
4. **No stage forced algorithmic learning** — Every early stage had a small enough problem space that memorization sufficed.

### Attempt 2: 4-Stage Curriculum (FAILED)

Collapsed 12 stages into 4, skipping sub-skill stages entirely. Stage 1 jumped straight to combined DOT+TEN counting with shuffled order.

Root causes:

1. **Skipped sub-skill scaffolding** — New Stage 1 ≈ old Stage 6 but without Stages 2-5 teaching individual counting first. The model never learned to count DOTs or TENs independently.
2. **Removed process supervision** — Old interleaved counting trained the model HOW to count step by step. New format only asked for the final answer, which the model memorized.
3. **Autoregressive output asymmetry** — In `= tens ones`, the tens digit must be predicted from input alone (→ learns counting), but the ones digit can condition on the ground-truth tens digit via teacher forcing (→ memorizes the combination instead of counting). Result: test per-token [1.00|0.00].
4. **No composition cues** — Zero signal telling the model to reuse prior skills for new tasks.
5. **No partial credit** — Exact-match metric hid the fact that one counting skill was mastered while the other wasn't.

## Revised Curriculum (8 Stages — Prerequisite-Complete)

### Why the 5-stage curriculum was wrong

The original 5-stage curriculum jumped from counting (Stages 1-2) to single-digit arithmetic (Stage 3) with four unverified assumptions about digit semantics: ordinality, place value, comparison, and the meaning of addition. See Mistake #44 and `DESIGN_GUIDE.md` for the category theory formulation.

The fix is to add intermediate stages that ground digits in quantities and teach addition as a composition of counting + successor, not as a lookup table.

### Design Principles

- **Prerequisite completeness** — Before teaching any composition, verify ALL prerequisite skills are learned. No missing edges in the knowledge graph. (Category theory constraint: morphism well-definedness requires codomain/domain matching.)
- **Sub-skills first, composition second** — Learn DOT and TEN counting independently (Stage 1) before combining (Stage 2)
- **Operations reduce to known skills** — Addition reduces to counting-up. Subtraction reduces to counting-down. The scratchpad shows the reduction explicitly.
- **Process supervision via scratchpad** — Every intermediate computation is visible in the work area. No hidden-state-only reasoning.
- **No false fact stages** — If a skill has >20 entries and can be decomposed into simpler operations, it's a composition stage, not a fact stage. The only genuine facts are symbol-to-meaning mappings.
- **Partial credit** — Per-token accuracy tracks individual skill mastery; display shows both train and test breakdowns
- **Enough problems for real splits** — Confounders (TENs in DOT-counting, DOTs in TEN-counting) expand problem space to 100 even for individual counting

### Architecture

**Mamba3LM** from `experiments/Mamba3/mamba3_block.py`:
- `model(input_ids) -> logits` (simple LM interface)
- Weight-tied embedding
- Config: `d_model=128, d_state=64, n_layer=4, headdim=64`

### Vocabulary (Scratchpad Framework)

```
Token 0:  PAD    Token 1:  WORK   Token 2:  NOTE   Token 3:  SEP
Token 4:  0  ...  Token 13: 9
Token 14: +    Token 15: -    Token 16: =
Token 17: DOT    Token 18: TEN
Token 19: SUCC   Token 20: PRED
Token 21: GT     Token 22: LT     Token 23: EQ
Token 24: :      (scratchpad column separator in counting-based stages)
vocab_size = 25
```

Built deterministically by `_build_vocab()` in `train_arithmetic.py`. Structural tokens (PAD, WORK, NOTE, SEP) come from the `Vocab` class. WORK replaces the old `=` separator. NOTE marks queries in the input. SEP separates column steps in the scratchpad. SUCC/PRED are successor/predecessor operations. GT/LT/EQ are comparison results. `:` separates the column setup from the counting-up/down sequence in Stages 5-8.

### Stage 1: Query Counting (sub-skill)

**Format:** `[PAD..., <shuffled DOTs and TENs>, NOTE, QUERY, WORK, count]`
- Each sample randomly asks: "how many DOTs?" or "how many TENs?"
- NOTE marker in the input tells the model what to count (Mistake #43: query must be in input, not output)
- n_result = 1 (single count digit)
- Example: `DOT TEN DOT TEN DOT NOTE DOT WORK 3` (query=DOT, answer=3)
- Example: `DOT TEN DOT TEN DOT NOTE TEN WORK 2` (query=TEN, answer=2)
- Example: `TEN TEN TEN TEN NOTE TEN WORK 4` (query=TEN, answer=4)
- 100 problems (10×10 count combinations), split 80/20
- **Status:** PASSING — reaches ≥95% test in ~8 epochs (seed=123).

### Stage 2: Combined Counting with Scratchpad (composition)

**Format:** `[PAD..., <shuffled DOTs and TENs>, WORK, DOT, d, TEN, t]`
- Reuses Stage 1's query tokens as composition cues
- The model counts DOTs first (DOT → d), then TENs (TEN → t)
- n_result = 4 (DOT, d, TEN, t — cue tokens are ungraded)
- Example: `DOT TEN DOT TEN DOT WORK DOT 3 TEN 2`
- Same 100 problems, same train/test split as Stage 1
- **Status:** PASSING — reaches ≥95% test in ~26 epochs (seed=123).

### Stage 3: Successor / Predecessor (ordinality)

**Format:** `[PAD..., a, NOTE, SUCC/PRED, WORK, result]`
- Teaches the number line: what digit comes next? what digit comes before?
- NOTE marker specifies the operation (successor or predecessor)
- n_result = 1 (the next/previous digit)
- Wraps at boundaries: SUCC(9) = 0, PRED(0) = 9 (mod-10 successor, no carry — carry comes later)
- Example: `4 NOTE SUCC WORK 5` (successor of 4 is 5)
- Example: `7 NOTE PRED WORK 6` (predecessor of 7 is 6)
- Example: `9 NOTE SUCC WORK 0` (wraps around)
- 20 problems (10 digits × 2 operations), composition stage with held-out specs
- **Prerequisite verification:** Does the model understand digit ordering?
- **New vocab tokens:** SUCC, PRED

### Stage 4: Comparison (digit ordering)

**Format:** `[PAD..., a, NOTE, b, WORK, GT/LT/EQ]`
- NOTE separates the two operands
- Output is a single token: GT (a > b), LT (a < b), or EQ (a = b)
- n_result = 1
- Example: `7 NOTE 3 WORK GT` (7 > 3)
- Example: `2 NOTE 8 WORK LT` (2 < 8)
- Example: `5 NOTE 5 WORK EQ` (5 = 5)
- 100 problems (10×10 digit pairs), composition stage with held-out specs
- **Prerequisite:** Stage 3 (successor teaches ordering)
- **New vocab tokens:** GT, LT, EQ

### Stage 5: Counting-Based Addition (addition = counting-up)

**Format:** `[PAD..., a, +, b, WORK, <count-up sequence>, =, carry, ones]`
- The scratchpad explicitly shows counting up b steps from a
- Reduces addition to the successor operation learned in Stage 3
- n_result = variable (b intermediate digits + 3 for `= carry ones`)
- Example: `3 + 4 WORK 4 5 6 7 = 0 7` (start at 3, count up 4 steps: 4,5,6,7)
- Example: `8 + 5 WORK 9 0 1 2 3 = 1 3` (crosses tens boundary: 9,0,1,2,3 → carry=1, ones=3)
- Example: `6 + 0 WORK = 0 6` (zero steps, result is just 6)
- Max sequence length: 9 intermediate digits (for +9), fits easily in seq_len=48
- 100 problems (10×10 addition pairs), composition stage with held-out specs
- **Prerequisite:** Stage 3 (successor function)
- **Key test:** Does the model use counting, or does it memorize? Held-out spec test reveals this.

### Stage 6: Counting-Based Subtraction (subtraction = counting-down)

**Format:** `[PAD..., a, -, b, WORK, <count-down sequence>, =, borrow, ones]`
- The scratchpad shows counting down b steps from a (using predecessor from Stage 3)
- a ≥ b only (non-negative results)
- n_result = variable (b intermediate digits + 3 for `= borrow ones`)
- Example: `7 - 3 WORK 6 5 4 = 0 4` (start at 7, count down 3 steps: 6,5,4)
- Example: `5 - 0 WORK = 0 5` (zero steps)
- 55 problems (a ≥ b pairs), composition stage with held-out specs
- **Prerequisite:** Stage 3 (predecessor function)

### Stage 7: Two-Digit ± Single-Digit (bridge, column scratchpad)

**Format:** `[PAD..., a1, a0, OP, 0, b0, WORK, <column scratchpad>]`
- a ∈ 10-99, b ∈ 0-9 (zero-padded to match Stage 8 format)
- Column scratchpad: ones column → tens column → hundreds column → final answer
- Each column step reuses Stage 5/6's counting-based format
- n_result = variable (depends on column count lengths)
- Example: `2 3 + 0 4 WORK 3 + 4 : 4 5 6 7 = 0 7 SEP 2 + 0 : = 0 2 SEP 0 2 7`
- Total: 1,800 problems (90×10×2), split ~1,440/360
- **Prerequisite:** Stages 5-6 (counting-based single-digit ops)

### Stage 8: Two-Digit ± Two-Digit (composition test, column scratchpad)

**Format:** `[PAD..., a1, a0, OP, b1, b0, WORK, <column scratchpad>]`
- a, b ∈ 10-99
- Same column scratchpad format as Stage 7
- n_result = variable
- Example: `5 1 + 4 2 WORK 1 + 2 : 2 3 = 0 3 SEP 5 + 4 : 5 6 7 8 9 = 0 9 SEP 0 9 3`
- Total: ~12,195 problems, split ~9,756/2,439
- **This is the critical composition test:**
  - The model has learned counting (Stages 1-2), ordinality (Stage 3), comparison (Stage 4), counting-based addition/subtraction (Stages 5-6), multi-digit format (Stage 7)
  - It must compose: column-wise counting + carry propagation
  - Problem space far too large to memorize
- **Two experimental arms:**
  1. **Curriculum:** Train Stages 1→2→3→4→5→6→7→8 (advance at ≥95% test)
  2. **Direct:** Train on Stage 8 data only (same total training budget)
- **Hypothesis:** Curriculum arm generalizes; direct arm memorizes.

---

## Evaluation Protocol

For each stage:
1. **Train set:** Sampled from held-in problem combinations
2. **Test set:** Held-out operand combinations (never seen during training)
3. **Metric:** Exact-match accuracy on the result tokens
4. **Per-token accuracy** for multi-token results — shows BOTH train and test breakdowns
5. **Partial credit:** Per-token display tracks individual skill mastery
6. **Advancement:** Move to next stage when test accuracy ≥ 95%

---

## Implementation

### File Structure

```
experiments/Mamba3/
├── mamba3_block.py           # Mamba3 model (DO NOT MODIFY)
├── arithmetic_tasks.py       # Old task generators (SUPERSEDED by scratchpad)
├── train_arithmetic.py       # Curriculum training script (uses scratchpad framework)
├── continual.py              # EWC, DER++, differential LR
└── archived_epc/             # Archived, ignore

experiments/scratchpad/
├── __init__.py               # Exports Vocab, split_problems
├── framework.py              # Vocab, Problem, Step, Grader, ProblemGenerator, Curriculum
├── DESIGN_GUIDE.md           # Curriculum design principles (category theory, prerequisites)
└── generators/
    ├── __init__.py            # Exports all generators
    ├── counting.py            # QueryCountingGenerator (S1), CombinedCountingGenerator (S2)
    ├── ordinality.py          # SuccessorGenerator (S3), ComparisonGenerator (S4)
    ├── arithmetic.py          # CountingAdditionGenerator (S5), CountingSubtractionGenerator (S6)
    └── multi_digit.py         # TwoDigitSingle (S7), TwoDigit (S8)
```

### Running Experiments (ON GPU, not Claude's machine)

```bash
# Curriculum arm: stages 1→2→3→4→5→6→7→8
python train_arithmetic.py --curriculum --target_stage 8 --results_file curriculum.jsonl

# Direct arm: stage 8 only
python train_arithmetic.py --stage 8 --epochs 200 --results_file direct.jsonl

# Quick test: individual stages
python train_arithmetic.py --stage 1 --epochs 50
python train_arithmetic.py --stage 2 --epochs 50
python train_arithmetic.py --stage 3 --epochs 50  # successor/predecessor
python train_arithmetic.py --stage 4 --epochs 50  # comparison
python train_arithmetic.py --stage 5 --epochs 50  # counting-based addition
python train_arithmetic.py --stage 6 --epochs 50  # counting-based subtraction
```

---

## Key Constraints

- **Do NOT run training on Claude's machine** (Mistake #36). Implement, commit, push.
- **Use Mamba3LM, not NajaLM.** This tests the curriculum hypothesis, not Naja.
- **Per Mistake #41:** Answer prediction from `logits[:, p-1]` for position p.
- **Per Mistake #42:** Verify test accuracy is above chance before drawing conclusions.
- **Per Mistake #44:** Every composition stage must have all prerequisites taught and verified. No missing edges in the knowledge graph. Single-digit addition is a composition stage (counting-up), not a fact stage.

## Success Criteria

1. **Stage 1 reaches ≥95% test accuracy** (individual counting generalizes)
2. **Stage 2 reaches ≥95% test accuracy** (composition works for counting)
3. **Stage 3 reaches ≥95% test accuracy** (successor/predecessor — ordinality)
4. **Stage 4 reaches ≥95% test accuracy** (comparison — digit ordering)
5. **Stage 5 reaches ≥95% test accuracy** (addition as counting-up — the key grounding test)
6. **Stage 6 reaches ≥95% test accuracy** (subtraction as counting-down)
7. **Stages 7-8 each reach ≥95% test accuracy** (multi-digit column arithmetic)
8. **Curriculum arm on Stage 8 achieves significantly higher test accuracy than direct arm** (full composition works)
9. **No catastrophic forgetting** — accuracy on earlier stages stays above 90%
10. If Stage 8 succeeds, can extend with multiplication, PEMDAS later

## Key Files to Read First

| File | What's in it |
|------|-------------|
| `MISTAKES.md` | 46 documented mistakes — **always read first** |
| `CLAUDE.md` | Architecture overview, priorities |
| `experiments/scratchpad/framework.py` | Scratchpad framework (Vocab, Problem, Grader) |
| `experiments/scratchpad/generators/` | Stage 1-5 problem generators |
| `experiments/Mamba3/train_arithmetic.py` | Training script (uses scratchpad) |
