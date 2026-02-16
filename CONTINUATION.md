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

vocab_size = 22
```

Digits are encoded as `digit + 1` (so token 1 = digit 0, token 10 = digit 9). Multi-digit numbers use multiple digit tokens in sequence (e.g., "23" = tokens [3, 4]).

### Curriculum Stages

#### Stage 1: Magnitude Comparison (learn that digits have ordinal meaning)

**Format:** `[PAD..., a, CMP, b, RESULT]`
- Example: `3 > 1 → TRUE`, `2 > 7 → FALSE`, `5 < 8 → TRUE`
- CMP ∈ {`>`, `<`}, RESULT ∈ {`TRUE`, `FALSE`}
- Single-digit only
- 90 possible comparisons (10×10 minus 10 equal pairs)
- **Tests:** Does the model learn magnitude ordering of digit tokens?
- **Coverage:** Near-complete (can see most/all of the 90 pairs)
- **Generalization test:** Held-out digit pairs

#### Stage 2: Successor / Predecessor (learn +1/-1 as operations on quantity)

**Format:** `[PAD..., a, +, 1, =, result]` and `[PAD..., a, -, 1, =, result]`
- Example: `3 + 1 = 4`, `7 - 1 = 6`
- Result is always a single digit (restrict to 0-9 range)
- 18 examples total (9 successors + 9 predecessors)
- **Tests:** Does the model learn that + and - modify quantity?
- **Advancement criterion:** ≥95% test accuracy

#### Stage 3: Single-Digit Arithmetic (learn the four operations)

**Format:** `[PAD..., a, OP, b, =, r1, (r2)]`
- Addition: `3 + 4 = 7`, `8 + 5 = 1 3` (result can be 1-2 digits)
- Subtraction: `7 - 3 = 4` (only a ≥ b to avoid negatives)
- Multiplication: `3 * 4 = 1 2`, `2 * 3 = 6`
- Division: `8 / 2 = 4` (only exact divisions, b > 0)
- **Tests:** Does the model learn each operation?
- **Coverage:** All valid single-digit pairs for each operation
- **Key question:** Does prior training on Stage 1-2 help learn Stage 3 faster?

#### Stage 4: Two-Digit Arithmetic (composition of place value + operation + carry)

**Format:** `[PAD..., d1, d2, OP, d3, d4, =, r1, r2, (r3)]`
- Example: `2 3 + 1 4 = 3 7`, `4 5 - 1 8 = 2 7`
- **This is the critical generalization test:**
  - The model has NEVER seen two-digit addition directly
  - It must compose: place value (Stage 1-2) + single-digit operation (Stage 3) + carry
- **Two experimental arms:**
  1. **Curriculum:** Train Stages 1→2→3→4 sequentially (advance at ≥95% test)
  2. **Direct:** Train on Stage 4 data only (same total training budget)
- **Hypothesis:** Curriculum arm generalizes; direct arm memorizes.

#### Stage 5: PEMDAS (composition of operations with precedence)

**Format:** `[PAD..., a, OP1, b, OP2, c, =, result]`
- Example: `2 + 3 * 4 = 1 4` (not 20)
- Example: `8 - 2 + 3 = 9`
- Tests whether the model applies operator precedence
- Only attempted if Stage 4 succeeds
- **Bonus:** Add parentheses: `( 2 + 3 ) * 4 = 2 0`

### Evaluation Protocol

For each stage:
1. **Train set:** Sample problems with random operands
2. **Test set:** Held-out operand combinations (never seen during training)
3. **Metric:** Exact-match accuracy on the result tokens
4. **Advancement:** Move to next stage when test accuracy ≥ 95%

For the composition test (Stage 4):
1. **Curriculum arm:** Sequential training through Stages 1→2→3→4
2. **Direct arm:** Same model, same total epochs, trained only on Stage 4 data
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

Functions to implement:
- `generate_comparison(n_samples, held_out_pairs)` → Stage 1
- `generate_successor(n_samples)` → Stage 2
- `generate_single_digit_arithmetic(n_samples, ops, held_out_pairs)` → Stage 3
- `generate_two_digit_arithmetic(n_samples, ops, held_out_pairs)` → Stage 4
- `generate_pemdas(n_samples, held_out_exprs)` → Stage 5
- `VOCAB` dict mapping symbols to token IDs
- `decode_tokens(tensor)` → human-readable string (for debugging)

### Phase 2: Training Script (`train_arithmetic.py`)

Build on patterns from `train_naja.py` but with curriculum logic:

1. **Single-stage mode:** `python train_arithmetic.py --stage 3 --epochs 50`
2. **Curriculum mode:** `python train_arithmetic.py --curriculum --target_stage 4`
   - Trains stages sequentially
   - Advances when test_acc ≥ 95% (configurable via `--advance_threshold`)
   - Reports per-stage epoch counts
3. **Direct mode:** `python train_arithmetic.py --stage 4 --epochs 200`
   - Same total budget as curriculum, but only Stage 4 data
4. **Comparison output:** JSON results file with per-stage learning curves

Key features:
- Uses `Mamba3LM` (not NajaLM)
- Same training infrastructure: AdamW, cosine LR, AMP, gradient clipping
- `--results_file` for JSON output (compatible with ablation runner pattern)
- No diagnostics charts (keep it simple)
- Per-epoch accuracy on BOTH the current stage AND all previous stages (to detect catastrophic forgetting)

### Phase 3: Run Experiments (ON GPU, not Claude's machine)

```bash
# Curriculum arm: stages 1→2→3→4
python train_arithmetic.py --curriculum --target_stage 4 --results_file curriculum.jsonl

# Direct arm: stage 4 only, same total epochs
python train_arithmetic.py --stage 4 --epochs 200 --results_file direct.jsonl

# Control: each stage independently
python train_arithmetic.py --stage 1 --epochs 50 --results_file stage1.jsonl
python train_arithmetic.py --stage 2 --epochs 50 --results_file stage2.jsonl
python train_arithmetic.py --stage 3 --epochs 50 --results_file stage3.jsonl
```

---

## Key Constraints

- **Do NOT run training on Claude's machine** (Mistake #36). Implement, commit, push.
- **Use Mamba3LM, not NajaLM.** This experiment tests the curriculum hypothesis, not the Naja architecture. Mamba3 is simpler and has fewer confounds.
- **Measure generalization, not memorization.** Test sets must contain operand combinations never seen in training. Per Mistake #42, verify that test accuracy is above chance before drawing conclusions.
- **Per Mistake #41:** Answer prediction from `logits[:, -2]`, not `logits[:, -1]`.
- **Keep it simple.** No feature flags, no presets, no ablation grid. Two arms: curriculum vs direct. One model architecture. One clear question.

## Success Criteria

1. **Stage 1-3 each reach ≥95% test accuracy** (proves sub-skills are learnable)
2. **Curriculum arm on Stage 4 achieves significantly higher test accuracy than direct arm** (proves composition works)
3. **No catastrophic forgetting** — accuracy on stages 1-3 stays above 90% while training Stage 4
4. If Stage 4 succeeds, Stage 5 (PEMDAS) is a bonus

## Key Files to Read First

| File | What's in it |
|------|-------------|
| `MISTAKES.md` | 42 documented mistakes — **always read first** |
| `CLAUDE.md` | Architecture overview, priorities |
| `experiments/Mamba3/mamba3_block.py` | Mamba3 model (the backbone for this experiment) |
| `experiments/Naja/tasks.py` | Task generator pattern to follow |
| `experiments/Naja/train_naja.py` | Training loop pattern to follow |
