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

The original 12-stage curriculum failed — it provided no advantage over direct training. Root causes:

1. **Early stages trivially memorizable** — Stages 1-5 had only 9-10 problems each, all below the split threshold. The model "passed" by memorizing lookup tables, never learning algorithms.
2. **Interleaved counting gave the answer away** — `DOT 1 DOT 2 DOT 3 = 3` taught copying, not counting. The answer was in the input.
3. **Composition test too small** — Stage 6 had only 100 problems (80 train / 20 test). Heavy memorization pressure.
4. **Composition wasn't real** — Segmentation (finding where TENs end and DOTs begin) was a hidden prerequisite never explicitly taught.
5. **No stage forced algorithmic learning** — Every early stage had a small enough problem space that memorization sufficed.

## Revised Curriculum (4 Stages)

### Design Principles

- Every stage has enough problems for meaningful train/test splits (≥100)
- No free answers in the input (no interleaved counting)
- Counting uses randomized DOT/TEN order (no run-length shortcut)
- Each stage teaches a skill that directly composes into later stages
- Consistent output format within result-size groups

### Architecture

**Mamba3LM** from `experiments/Mamba3/mamba3_block.py`:
- `model(input_ids) -> logits` (simple LM interface)
- Weight-tied embedding
- Config: `d_model=128, d_state=64, n_layer=4, headdim=64`

### Vocabulary

```
Token 0:  PAD
Token 1:  digit 0  ...  Token 10: digit 9
Token 11: +    Token 12: -    Token 13: *    Token 14: /
Token 15: =    Token 16: >    Token 17: <
Token 18: TRUE   Token 19: FALSE
Token 20: (    Token 21: )
Token 22: DOT (● unit object)    Token 23: TEN (■ ten-bundle)    Token 24: NEXT
vocab_size = 25
```

### Stage 1: Mixed Counting (grounding digits in quantities)

**Format:** `[PAD..., <shuffled DOTs and TENs>, =, tens_digit, ones_digit]`
- DOT and TEN tokens in random order (no fixed TENs-first ordering)
- No interleaved running counts — model must actually count each type
- Example: `DOT TEN DOT TEN DOT = 2 3` (2 TENs, 3 DOTs)
- Example: `TEN TEN TEN TEN = 4 0` (4 TENs, 0 DOTs)
- Example: `= 0 0` (nothing to count)
- 100 problems (10×10 count combinations), split 80/20
- Each combination generates many shuffled arrangements → diverse training
- **Teaches:** Digits represent quantities. TEN count → tens place, DOT count → ones place.

### Stage 2: Single-Digit +/- (arithmetic facts)

**Format:** `[PAD..., a, OP, b, =, d1, d2]` (2-digit zero-padded result)
- Addition: all (a,b) pairs, 100 problems
- Subtraction: a ≥ b only, 55 problems
- Total: 155 problems, split ~124/31
- Example: `3 + 4 = 0 7`, `8 + 5 = 1 3`, `7 - 3 = 0 4`
- **Teaches:** Addition and subtraction operations, operator dispatch, 2-digit output format.

### Stage 3: Two-Digit ± Single-Digit (bridge)

**Format:** `[PAD..., a1, a2, OP, 0, b, =, r1, r2, r3]` (3-digit result)
- a ∈ 10-99, b ∈ 0-9 (zero-padded to match Stage 4 format)
- Subtraction always valid (a ≥ 10 > 9 ≥ b)
- Total: 1,800 problems (90×10×2), split ~1,440/360
- Example: `2 3 + 0 4 = 0 2 7`, `5 0 - 0 3 = 0 4 7`
- **Teaches:** Multi-digit I/O format, carry propagation — reusing single-digit skill from Stage 2.

### Stage 4: Two-Digit ± Two-Digit (composition test)

**Format:** `[PAD..., a1, a2, OP, b1, b2, =, r1, r2, r3]` (3-digit result)
- a, b ∈ 10-99
- Addition: 8,100 problems. Subtraction (a ≥ b): ~4,095 problems.
- Total: ~12,195 problems, split ~9,756/2,439
- Example: `2 3 + 4 8 = 0 7 1`, `9 5 - 1 8 = 0 7 7`
- **This is the critical composition test:**
  - The model has learned counting (Stage 1), single-digit ops (Stage 2), multi-digit format (Stage 3)
  - It must compose: column-wise addition + carry propagation
  - Problem space far too large to memorize
- **Two experimental arms:**
  1. **Curriculum:** Train Stages 1→2→3→4 (advance at ≥95% test)
  2. **Direct:** Train on Stage 4 data only (same total training budget)
- **Hypothesis:** Curriculum arm generalizes; direct arm memorizes.

---

## Evaluation Protocol

For each stage:
1. **Train set:** Sampled from held-in problem combinations
2. **Test set:** Held-out operand combinations (never seen during training)
3. **Metric:** Exact-match accuracy on the result tokens
4. **Per-token accuracy** for multi-token results (identifies bottleneck positions)
5. **Advancement:** Move to next stage when test accuracy ≥ 95%

---

## Implementation

### File Structure

```
experiments/Mamba3/
├── mamba3_block.py           # Mamba3 model (DO NOT MODIFY)
├── arithmetic_tasks.py       # Task generators (4 stages)
├── train_arithmetic.py       # Curriculum training script
├── continual.py              # EWC, DER++, differential LR
└── archived_epc/             # Archived, ignore
```

### Running Experiments (ON GPU, not Claude's machine)

```bash
# Curriculum arm: stages 1→2→3→4
python train_arithmetic.py --curriculum --target_stage 4 --results_file curriculum.jsonl

# Direct arm: stage 4 only
python train_arithmetic.py --stage 4 --epochs 200 --results_file direct.jsonl

# Quick test: single stage
python train_arithmetic.py --stage 1 --epochs 50
python train_arithmetic.py --stage 2 --epochs 50
```

---

## Key Constraints

- **Do NOT run training on Claude's machine** (Mistake #36). Implement, commit, push.
- **Use Mamba3LM, not NajaLM.** This tests the curriculum hypothesis, not Naja.
- **Per Mistake #41:** Answer prediction from `logits[:, p-1]` for position p.
- **Per Mistake #42:** Verify test accuracy is above chance before drawing conclusions.

## Success Criteria

1. **Stages 1-3 each reach ≥95% test accuracy** (sub-skills are learnable)
2. **Curriculum arm on Stage 4 achieves significantly higher test accuracy than direct arm** (composition works)
3. **No catastrophic forgetting** — accuracy on earlier stages stays above 90%
4. If Stage 4 succeeds, can extend with multiplication, PEMDAS later

## Key Files to Read First

| File | What's in it |
|------|-------------|
| `MISTAKES.md` | 42 documented mistakes — **always read first** |
| `CLAUDE.md` | Architecture overview, priorities |
| `experiments/Mamba3/mamba3_block.py` | Mamba3 model (backbone) |
| `experiments/Mamba3/arithmetic_tasks.py` | Task generators (4 stages) |
| `experiments/Mamba3/train_arithmetic.py` | Training script |
