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

The model treats composite tasks as monolithic lookup tables. With 1.26M parameters and only 5K training examples, memorization is always the path of least resistance. More data or longer training (grokking) might help, but neither addresses the fundamental question: **can the model learn to compose sub-skills?**

## The Hypothesis: Compositional Curriculum

How many examples of addition does it take to teach a child? About 50-100, covering single-digit pairs (10×10 = 100 total). But then the child generalizes to multi-digit addition *without* seeing all 1000×1000 two-digit cases.

The child can do this because they learned **composable sub-skills first**:

1. **Counting** — numbers are ordered, "next" means +1
2. **Place value** — "23" means 2×10 + 3 (positional decomposition)
3. **Single-digit addition** — the 100 base facts
4. **The carry rule** — when a column exceeds 9, send 1 to the next column

Each stage has near-100% coverage at the *algorithmic level*. By the time they see "23 + 48", the only new thing is "apply the same column procedure and carry."

**Our models skip all of this.** They see associative recall as a monolithic function from 32-token sequence → answer token. They never decompose the task into "scan for matching key" then "read adjacent value."

**The experiment:** Build a staged arithmetic curriculum on the Mamba3 backbone. Train each sub-skill to generalization before introducing the next. Test whether compositional training enables generalization on composite tasks that fail with direct training.

---

## Experimental Design

### Architecture

**Mamba3LM** from `experiments/Mamba3/mamba3_block.py`:
- `model(input_ids) -> logits` (simple LM interface)
- Weight-tied embedding
- Config: `d_model=128, d_state=64, n_layer=4, headdim=64`
- ~500K params (smaller than Naja's 1.26M — important for generalization)

### Vocabulary

```
Token 0:  PAD
Token 1:  digit 0
Token 2:  digit 1
...
Token 10: digit 9
Token 11: +  (PLUS)
Token 12: -  (MINUS)
Token 13: *  (TIMES)
Token 14: /  (DIVIDE)
Token 15: =  (EQUALS)
Token 16: >  (GT)
Token 17: <  (LT)
Token 18: TRUE
Token 19: FALSE
Token 20: (  (LPAREN)
Token 21: )  (RPAREN)
Token 22: DOT (● unit object, represents 1)
Token 23: TEN (■ ten-bundle, represents 10)
Token 24: NEXT (→ successor arrow)

vocab_size = 25
```

Digits are encoded as `digit + 1` (so token 1 = digit 0, token 10 = digit 9). Multi-digit numbers use multiple digit tokens in sequence (e.g., "23" = tokens [3, 4]). DOT/TEN tokens represent quantities for cardinality stages.

### Curriculum Stages

#### Stage 1: Digit Successor (learn digit ordering)

**Format:** `[PAD..., a, NEXT, b]` where b = a + 1
- Example: `3 NEXT 4`, `0 NEXT 1`, `8 NEXT 9`
- 9 problems total (0→1 through 8→9)
- **Tests:** Does the model learn the natural order of digit tokens?
- **Foundational:** This is the most basic number sense — what comes next?

#### Stage 2: Single-Digit Counting (learn cardinality — digits = quantities)

**Format:** `[PAD..., DOT, ..., DOT, =, d]` where count(DOT) = d
- Example: `DOT DOT DOT = 3`, `DOT DOT DOT DOT DOT = 5`, `= 0` (no dots)
- 10 problems total (0-9 dots → digit)
- **Tests:** Does the model learn that digit symbols represent quantities of objects?
- **Foundational:** Grounds digit tokens in concrete quantity (cardinality principle)

#### Stage 3: Count TENs (same counting skill, new token)

**Format:** `[PAD..., TEN, ..., TEN, =, d]` where count(TEN) = d
- Example: `TEN TEN TEN = 3`
- 10 problems total (0-9 TENs → digit)
- **Tests:** Does counting transfer across token types?
- **Foundational:** Prerequisite for place value (TEN = tens column)

#### Stage 4: Count DOTs — two-digit output (bridge to multi-token results)

**Format:** `[PAD..., DOT, ..., DOT, =, 0, d]` where count(DOT) = d
- Example: `DOT DOT DOT = 0 3` (same count as Stage 2, but in two-digit format)
- 10 problems total (0-9 DOTs)
- **Tests:** Can the model learn two-digit output format?
- **Key insight:** Stages 1-3 all produce n_result=1. This stage isolates "learn to produce two result tokens" from "learn to segment input". The tens digit is trivially 0; the ones digit is the already-known DOT count.

#### Stage 5: Count TENs — two-digit output (bridge to multi-token results)

**Format:** `[PAD..., TEN, ..., TEN, =, d, 0]` where count(TEN) = d
- Example: `TEN TEN TEN = 3 0` (TEN count maps to tens position)
- 10 problems total (0-9 TENs)
- **Tests:** Does the model learn that TEN count goes in the tens position?
- **Key insight:** Combined with Stage 4, the model now knows: DOT count → ones position (Stage 4), TEN count → tens position (Stage 5). Stage 6 composes these two skills.

#### Stage 6: Two-Digit Counting (learn place value via ten-bundles)

**Format:** `[PAD..., TEN, ..., TEN, DOT, ..., DOT, =, d1, d2]`
- TEN tokens = bundles of 10, DOT tokens = ones
- Example: `TEN TEN DOT DOT DOT = 2 3` (23 = 2 tens + 3 ones)
- Example: `TEN = 1 0` (10 = 1 ten + 0 ones)
- 90 problems total (10-99)
- Max sequence: 9 TENs + 9 DOTs + = + 2 digits = 21 tokens
- **Tests:** Does the model compose "count TENs → tens digit" + "count DOTs → ones digit"?
- **Builds on:** Stages 4-5 (two-digit output format with position-specific counting)

#### Stage 7: Magnitude Comparison (learn that digits have ordinal meaning)

**Format:** `[PAD..., a, CMP, b, RESULT]`
- Example: `3 > 1 → TRUE`, `2 > 7 → FALSE`, `5 < 8 → TRUE`
- CMP ∈ {`>`, `<`}, RESULT ∈ {`TRUE`, `FALSE`}
- Single-digit only
- 200 problems (10×10×2)
- **Tests:** Does the model learn magnitude ordering of digit tokens?
- **Builds on:** Stage 1 (digit ordering) should make comparison easier

#### Stage 8: Digit Distance (magnitude awareness — how far apart?)

**Format:** `[PAD..., a, -, b, =, d]` where d = a - b, a >= b
- Example: `8 - 4 = 4`, `5 - 5 = 0`
- 55 problems total
- **Tests:** Does the model learn not just "which is bigger?" but "by how much?"
- **Builds on:** Stage 7 (comparison)

#### Stage 9: Successor / Predecessor (learn +1/-1 as arithmetic operations)

**Format:** `[PAD..., a, +, 1, =, result]` and `[PAD..., a, -, 1, =, result]`
- Example: `3 + 1 = 4`, `7 - 1 = 6`
- 18 problems total (9 successors + 9 predecessors)
- **Tests:** Does the model learn that + and - modify quantity?
- **Builds on:** Stage 1 (successor) + Stage 2 (cardinality)

#### Stage 10: Single-Digit Arithmetic (learn the four operations)

**Format:** `[PAD..., a, OP, b, =, d1, d2]` (result always 2 digits, zero-padded)
- Addition: `3 + 4 = 0 7`, `8 + 5 = 1 3`
- Subtraction: `7 - 3 = 0 4` (only a ≥ b)
- Multiplication: `3 * 4 = 1 2`, `2 * 3 = 0 6`
- Division: `8 / 2 = 0 4` (only exact divisions, b > 0)
- **Tests:** Does the model learn each operation?
- **Builds on:** Stages 1-9 (number sense + simple operations)

#### Stage 11: Two-Digit Arithmetic (composition of place value + operation + carry)

**Format:** `[PAD..., d1, d2, OP, d3, d4, =, r1, r2, r3]` (result always 3 digits)
- Example: `2 3 + 1 4 = 0 3 7`, `4 5 - 1 8 = 0 2 7`
- **This is the critical generalization test:**
  - The model has learned place value (Stage 6) and single-digit ops (Stage 10)
  - It must compose: place value + single-digit operation + carry
- **Two experimental arms:**
  1. **Curriculum:** Train Stages 1→...→11 sequentially (advance at ≥95% test)
  2. **Direct:** Train on Stage 11 data only (same total training budget)
- **Hypothesis:** Curriculum arm generalizes; direct arm memorizes.

#### Stage 12: PEMDAS (composition of operations with precedence)

**Format:** `[PAD..., a, OP1, b, OP2, c, =, r1, r2, r3]` (result always 3 digits)
- Example: `2 + 3 * 4 = 0 1 4` (not 020)
- Example: `8 - 2 + 3 = 0 0 9`
- Tests whether the model applies operator precedence
- Only attempted if Stage 11 succeeds

### Evaluation Protocol

For each stage:
1. **Train set:** Sample problems with random operands
2. **Test set:** Held-out operand combinations (never seen during training)
3. **Metric:** Exact-match accuracy on the result tokens
4. **Advancement:** Move to next stage when test accuracy ≥ 95%

For the composition test (Stage 11):
1. **Curriculum arm:** Sequential training through Stages 1→...→11
2. **Direct arm:** Same model, same total epochs, trained only on Stage 11 data
3. **Control:** Random curriculum order (stages shuffled, not sequential)

### Loss Function

Next-step prediction cross-entropy on the full sequence (same as Naja/JEPA training). The model learns to predict every token including operators and equals signs, but accuracy is measured only on the result tokens.

For tasks with fixed-position answers (comparison, successor), also use `logits[:, -2]` answer prediction (per Mistake #41) as a secondary metric.

---

## Implementation Plan

### File Structure

```
experiments/Mamba3/
├── mamba3_block.py           # Existing Mamba3 model (DO NOT MODIFY)
├── train_arithmetic.py       # NEW: Curriculum training script
├── arithmetic_tasks.py       # NEW: Task generators for all stages
└── archived_epc/             # Existing, ignore
```

### Phase 1: Task Generators (`arithmetic_tasks.py`)

Create generators for each stage. Each returns `(sequences, targets)` tensors following the existing convention (PAD=0, left-padded).

Functions implemented:
- `generate_digit_successor(n_samples, ...)` → Stage 1 (digit ordering)
- `generate_counting(n_samples, ...)` → Stage 2 (DOT cardinality, 1-digit output)
- `generate_count_tens(n_samples, ...)` → Stage 3 (TEN cardinality, 1-digit output)
- `generate_counting_2d(n_samples, ...)` → Stage 4 (DOT cardinality, 2-digit output)
- `generate_count_tens_2d(n_samples, ...)` → Stage 5 (TEN cardinality, 2-digit output)
- `generate_two_digit_counting(n_samples, ...)` → Stage 6 (place value composition)
- `generate_comparison(n_samples, ...)` → Stage 7 (magnitude comparison)
- `generate_digit_distance(n_samples, ...)` → Stage 8 (how far apart?)
- `generate_successor(n_samples, ...)` → Stage 9 (±1 arithmetic)
- `generate_single_digit(n_samples, ...)` → Stage 10 (four operations)
- `generate_two_digit(n_samples, ...)` → Stage 11 (two-digit arithmetic)
- `generate_pemdas(n_samples, ...)` → Stage 12 (precedence)
- `VOCAB` dict mapping symbols to token IDs (25 tokens incl. DOT, TEN, NEXT)
- `decode_tokens(tensor)` → human-readable string (for debugging)

### Phase 2: Training Script (`train_arithmetic.py`)

Build on patterns from `train_naja.py` but with curriculum logic:

1. **Single-stage mode:** `python train_arithmetic.py --stage 6 --epochs 50`
2. **Curriculum mode:** `python train_arithmetic.py --curriculum --target_stage 11`
   - Trains stages 1→11 sequentially
   - Advances when test_acc ≥ 95% (configurable via `--advance_threshold`)
   - Reports per-stage epoch counts
3. **Direct mode:** `python train_arithmetic.py --stage 11 --epochs 200`
   - Same total budget as curriculum, but only Stage 11 data
4. **Comparison output:** JSON results file with per-stage learning curves

Key features:
- Uses `Mamba3LM` (not NajaLM)
- Same training infrastructure: AdamW, cosine LR, AMP, gradient clipping
- `--results_file` for JSON output (compatible with ablation runner pattern)
- No diagnostics charts (keep it simple)
- Per-epoch accuracy on BOTH the current stage AND all previous stages (to detect catastrophic forgetting)

### Phase 3: Run Experiments (ON GPU, not Claude's machine)

```bash
# Curriculum arm: stages 1→...→11
python train_arithmetic.py --curriculum --target_stage 11 --results_file curriculum.jsonl

# Direct arm: stage 11 only, same total epochs
python train_arithmetic.py --stage 11 --epochs 200 --results_file direct.jsonl

# Quick test: curriculum through Stage 6 (place value composition)
python train_arithmetic.py --curriculum --target_stage 6 --results_file curriculum_s6.jsonl
```

---

## Key Constraints

- **Do NOT run training on Claude's machine** (Mistake #36). Implement, commit, push.
- **Use Mamba3LM, not NajaLM.** This experiment tests the curriculum hypothesis, not the Naja architecture. Mamba3 is simpler and has fewer confounds.
- **Measure generalization, not memorization.** Test sets must contain operand combinations never seen in training. Per Mistake #42, verify that test accuracy is above chance before drawing conclusions.
- **Per Mistake #41:** Answer prediction from `logits[:, -2]`, not `logits[:, -1]`.
- **Keep it simple.** No feature flags, no presets, no ablation grid. Two arms: curriculum vs direct. One model architecture. One clear question.

## Success Criteria

1. **Stages 1-10 each reach ≥95% test accuracy** (proves sub-skills are learnable)
2. **Curriculum arm on Stage 11 achieves significantly higher test accuracy than direct arm** (proves composition works)
3. **No catastrophic forgetting** — accuracy on earlier stages stays above 90% while training later stages
4. If Stage 11 succeeds, Stage 12 (PEMDAS) is a bonus
5. **Immediate test:** Stage 6 (two-digit counting) should now pass with bridge Stages 4-5 in place

## Key Files to Read First

| File | What's in it |
|------|-------------|
| `MISTAKES.md` | 42 documented mistakes — **always read first** |
| `CLAUDE.md` | Architecture overview, priorities |
| `experiments/Mamba3/mamba3_block.py` | Mamba3 model (the backbone for this experiment) |
| `experiments/Naja/tasks.py` | Task generator pattern to follow |
| `experiments/Naja/train_naja.py` | Training loop pattern to follow |
