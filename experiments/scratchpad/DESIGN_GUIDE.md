# Scratchpad Curriculum Design Guide

**Purpose:** Document hard-won lessons from designing compositional curricula. Every principle here was learned from a failure. This guide will inform the Category Theory Knowledge Graph structure — if the graph is structured wrong, curriculum training on it won't generalize.

---

## Core Principle: The Unpredictable Token

**Every token in the work area must be deterministically derivable from the input + all previous work tokens.**

If a token is randomly chosen and placed in the output, the model has no basis to predict it. The loss on that token is irreducible noise that destabilizes training.

**Diagnostic:** If per-token accuracy is at chance (50% for binary, 10% for digits), the model likely cannot predict that token from available context. Check: is the information needed to produce this token actually present in the input?

**Mistake #43:** QueryCountingGenerator placed the query type (DOT/TEN) randomly in the work area. The model hit 51% on it (binary chance). Fix: moved query to input via NOTE marker.

---

## Core Principle: Fact Stages vs Composition Stages

Not all stages test the same thing. A curriculum has two kinds of stages:

### Fact Stages (memorization is the goal)
- Teach atomic knowledge: arithmetic facts (3+4=7), vocabulary, lookup tables
- The model MUST memorize these — there's no sub-algorithm to derive 3+4=7
- **Do NOT hold out specs.** Train and test on the same spec set.
- Test accuracy measures: "has the model memorized all facts?"
- Example: Stage 3 (single-digit +/-) has 155 facts. All 155 should appear in training.

### Composition Stages (generalization is the goal)
- Test whether memorized facts compose into new capabilities
- **Hold out specs.** Train/test on disjoint operand combinations.
- Test accuracy measures: "can the model apply learned sub-skills to unseen combinations?"
- Example: Stage 5 (two-digit +/-) has ~12K problems. Held-out specs test whether column arithmetic generalizes.

### The Distinction Matters
Stage 3 failed because we held out 31 of 155 arithmetic facts. The model memorized the 124 training facts perfectly (100% train) but couldn't generalize to unseen facts (10% test = chance). This isn't a model failure — it's a curriculum design error. Single-digit addition IS a lookup table. Holding out entries tests whether the model learned modular arithmetic, which requires grokking (500-1000+ epochs past memorization).

**Rule of thumb:** If the stage teaches atomic, non-decomposable knowledge, it's a fact stage. If it requires combining previously learned sub-skills, it's a composition stage.

| Stage | Type | Held-out? | What test measures |
|-------|------|-----------|-------------------|
| 1 (query counting) | Composition | Yes | Can the model count unseen (d,t) combos? |
| 2 (combined counting) | Composition | Yes | Can two counts compose? |
| 3 (single-digit +/-) | **Fact** | **No** | Has the model memorized all 155 facts? |
| 4 (two-digit ± one) | Composition | Yes | Can column scratchpad generalize? |
| 5 (two-digit ± two) | Composition | Yes | Full compositional generalization? |

---

## Core Principle: Per-Token Diagnostics Are Non-Negotiable

Always display per-token accuracy, even for n_result=1. The failing token pinpoints which sub-computation breaks.

**Pattern observed repeatedly:** When a stage first fails, exactly one token position is at chance while others are high. This tells you:
1. Which computation fails (ones digit, carry, query type, etc.)
2. Whether it's a predictability problem (token can't be derived from input) or a generalization problem (model memorizes instead of learning)

### Diagnostic decision tree

```
Per-token accuracy at chance?
  YES → Is the information to predict it in the input?
    NO  → Design bug: move info to input (Mistake #43)
    YES → Is this a fact stage with held-out specs?
      YES → Don't hold out specs for fact stages
      NO  → Memorization problem: need more data, more epochs, or better scratchpad structure
  NO (but exact-match is low) → Multiple tokens partially correct
    → Check if error in early token cascades to later tokens (teacher forcing hides this)
```

---

## Core Principle: Scratchpad Structure Must Mirror Computation

The scratchpad format should decompose the target computation into steps that:
1. Each step is individually learnable (atomic or previously taught)
2. Steps flow left-to-right matching autoregressive generation
3. Later steps can condition on earlier steps (carry propagation)
4. Format is consistent across stages (column format reused in S3, S4, S5)

### Good: Column scratchpad for multi-digit arithmetic
```
5 1 + 4 2 WORK 1 + 2 + 0 = 0 3 SEP 5 + 4 + 0 = 0 9 SEP 0 9 3
```
Each column step is a Stage 3 fact. Carries flow explicitly between columns.

### Bad: Direct answer for multi-digit arithmetic
```
5 1 + 4 2 WORK 0 9 3
```
No intermediate steps. Model must compute everything in hidden state.

### Design test
For each work token, ask: "What previously-seen tokens does the model need to produce this?" If the answer includes tokens that haven't appeared yet in the sequence, the scratchpad order is wrong.

---

## Structural Tokens and Their Roles

| Token | Role | Where it appears |
|-------|------|-----------------|
| PAD | Left-padding | Before question |
| WORK | Separates input from output | Between question and work area |
| NOTE | Marks a query/hint in the input | In question, before WORK |
| SEP | Separates steps within work area | Between column operations |

**WORK** is the boundary between what the model reads (question) and what it must produce (work area). Everything before WORK is teacher-forced input; everything after is evaluated.

**NOTE** signals "the next token is a query/instruction, not data." Used when the model needs to know WHAT to compute (e.g., count DOTs vs TENs).

**SEP** groups work tokens into logical steps. Each SEP-separated group should correspond to one sub-computation (e.g., one column of arithmetic).

---

## Failure Patterns (Ranked by Frequency)

### 1. Unpredictable output token (~50% of first-attempt failures)
**Symptom:** One token at chance, others high.
**Cause:** Information needed to produce the token is not in the input.
**Fix:** Move the information to the question area (before WORK).

### 2. Held-out split on fact stage (~25% of first-attempt failures)
**Symptom:** 100% train, chance test. All tokens fail equally.
**Cause:** Atomic knowledge held out — model can't generalize lookup tables.
**Fix:** Use same specs for train and test (fact stage, not composition stage).

### 3. Insufficient scratchpad decomposition
**Symptom:** 100% train, low test. Model memorizes input→output mapping.
**Cause:** The scratchpad doesn't break the computation into learnable sub-steps.
**Fix:** Add intermediate steps that reuse previously-learned operations.

### 4. Catastrophic forgetting
**Symptom:** Previous stage accuracy drops during new stage training.
**Cause:** New training overwrites circuits from earlier stages.
**Fix:** Replay (resample from previous stages each epoch), EWC, or differential LR.

### 5. Format inconsistency across stages
**Symptom:** Composition stage fails despite sub-skill stages passing.
**Cause:** The format changed between stages — the model can't reuse learned patterns.
**Fix:** Ensure later stages literally contain earlier stage formats as sub-sequences.

---

## Implications for Category Theory Knowledge Graph

When building a curriculum over a knowledge graph:

1. **Leaf nodes are fact stages** — atomic definitions, axioms, base cases. Don't hold out.
2. **Composition nodes are composition stages** — theorems derived from lemmas, multi-step proofs. Hold out.
3. **Edge direction matters** — if A → B means "A is prerequisite for B", then B's scratchpad should contain A's format as a sub-step.
4. **Per-token diagnostics per graph node** — when a node fails, the failing token tells you which prerequisite edge is broken.
5. **Graph structure = scratchpad structure** — the work area for a composition node should mirror its incoming edges. If a theorem uses Lemma A and Lemma B, the scratchpad should have a step for each.

---

## Checklist for Designing a New Stage

- [ ] Every work token is deterministically derivable from input + previous work tokens
- [ ] The stage is correctly classified as fact or composition
- [ ] Fact stages: `is_fact_stage = True`, no held-out split
- [ ] Composition stages: held-out specs, test measures generalization
- [ ] Scratchpad format reuses sub-sequences from prerequisite stages
- [ ] Per-token accuracy will be displayed (always, even n_result=1)
- [ ] Structural tokens (NOTE, SEP) are used consistently
- [ ] Run a few samples through `decode_tokens()` and verify by hand
