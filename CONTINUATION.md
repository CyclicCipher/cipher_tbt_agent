# CONTINUATION: Compositional Arithmetic Curriculum on Mamba3

**Date:** 2026-02-16
**Priority:** IMMEDIATE — Tests the core generalization hypothesis.
**Context:** See Mistakes #42 (memorization), #38 (ePC archived), #34 (next-step prediction).

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

## Revised Curriculum (5 Stages)

### Design Principles

- **Sub-skills first, composition second** — Learn DOT and TEN counting independently (Stage 1) before combining (Stage 2)
- **Process supervision via scratchpad** — Stage 2 reuses Stage 1's query tokens as composition cues, teaching the model to chain two counting operations
- **Explicit composition cues** — DOT/TEN tokens after `=` signal which counting skill to apply, bridging between stages
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
vocab_size = 19
```

Built deterministically by `_build_vocab()` in `train_arithmetic.py`. Structural tokens (PAD, WORK, NOTE, SEP) come from the `Vocab` class. WORK replaces the old `=` separator. NOTE marks queries in the input. SEP separates column steps in the scratchpad.

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

### Stage 3: Single-Digit +/- (arithmetic facts)

**Format:** `[PAD..., a, OP, b, WORK, carry, ones]` (2-digit zero-padded result)
- Addition: all (a,b) pairs, 100 problems
- Subtraction: a ≥ b only, 55 problems
- Total: 155 problems, split ~124/31
- Example: `3 + 4 WORK 0 7`, `8 + 5 WORK 1 3`, `7 - 3 WORK 0 4`
- n_result = 2 (carry digit + ones digit)

### Stage 4: Two-Digit ± Single-Digit (bridge, column scratchpad)

**Format:** `[PAD..., a1, a2, OP, 0, b, WORK, <column scratchpad>]`
- a ∈ 10-99, b ∈ 0-9 (zero-padded to match Stage 5 format)
- Column scratchpad: ones column → tens column → hundreds column → final answer
- n_result = 21 (full column-by-column scratchpad)
- Example: `2 3 + 0 4 WORK 3 + 4 + 0 = 0 7 SEP 2 + 0 + 0 = 0 2 SEP 0 0 2 7`
- Total: 1,800 problems (90×10×2), split ~1,440/360

### Stage 5: Two-Digit ± Two-Digit (composition test, column scratchpad)

**Format:** `[PAD..., a1, a2, OP, b1, b2, WORK, <column scratchpad>]`
- a, b ∈ 10-99
- Same column scratchpad format as Stage 4
- n_result = 21 (full column-by-column scratchpad)
- Example: `5 1 + 4 2 WORK 1 + 2 + 0 = 0 3 SEP 5 + 4 + 0 = 0 9 SEP 0 0 9 3`
- Total: ~12,195 problems, split ~9,756/2,439
- **This is the critical composition test:**
  - The model has learned counting (Stages 1-2), single-digit ops (Stage 3), multi-digit format (Stage 4)
  - It must compose: column-wise addition + carry propagation
  - Problem space far too large to memorize
- **Two experimental arms:**
  1. **Curriculum:** Train Stages 1→2→3→4→5 (advance at ≥95% test)
  2. **Direct:** Train on Stage 5 data only (same total training budget)
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
└── generators/
    ├── __init__.py            # Exports all generators
    ├── counting.py            # QueryCountingGenerator (S1), CombinedCountingGenerator (S2)
    └── arithmetic.py          # SingleDigit (S3), TwoDigitSingle (S4), TwoDigit (S5)
```

### Running Experiments (ON GPU, not Claude's machine)

```bash
# Curriculum arm: stages 1→2→3→4→5
python train_arithmetic.py --curriculum --target_stage 5 --results_file curriculum.jsonl

# Direct arm: stage 5 only
python train_arithmetic.py --stage 5 --epochs 200 --results_file direct.jsonl

# Quick test: individual stages
python train_arithmetic.py --stage 1 --epochs 50
python train_arithmetic.py --stage 2 --epochs 50
python train_arithmetic.py --stage 3 --epochs 50
```

---

## Key Constraints

- **Do NOT run training on Claude's machine** (Mistake #36). Implement, commit, push.
- **Use Mamba3LM, not NajaLM.** This tests the curriculum hypothesis, not Naja.
- **Per Mistake #41:** Answer prediction from `logits[:, p-1]` for position p.
- **Per Mistake #42:** Verify test accuracy is above chance before drawing conclusions.

## Success Criteria

1. **Stage 1 reaches ≥95% test accuracy** (individual counting generalizes)
2. **Stage 2 reaches ≥95% test accuracy** (composition works for counting)
3. **Stages 3-4 each reach ≥95% test accuracy** (arithmetic sub-skills)
4. **Curriculum arm on Stage 5 achieves significantly higher test accuracy than direct arm** (full composition works)
5. **No catastrophic forgetting** — accuracy on earlier stages stays above 90%
6. If Stage 5 succeeds, can extend with multiplication, PEMDAS later

## Key Files to Read First

| File | What's in it |
|------|-------------|
| `MISTAKES.md` | 43 documented mistakes — **always read first** |
| `CLAUDE.md` | Architecture overview, priorities |
| `experiments/scratchpad/framework.py` | Scratchpad framework (Vocab, Problem, Grader) |
| `experiments/scratchpad/generators/` | Stage 1-5 problem generators |
| `experiments/Mamba3/train_arithmetic.py` | Training script (uses scratchpad) |
