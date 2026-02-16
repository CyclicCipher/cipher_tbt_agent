# Scratchpad Curriculum Design Guide

**Purpose:** Document hard-won lessons from designing compositional curricula. Every principle here was learned from a failure. This guide will inform the Category Theory Knowledge Graph structure — if the graph is structured wrong, curriculum training on it won't generalize.

---

## Core Principle: The Unpredictable Token

**Every token in the work area must be deterministically derivable from the input + all previous work tokens.**

If a token is randomly chosen and placed in the output, the model has no basis to predict it. The loss on that token is irreducible noise that destabilizes training.

**Diagnostic:** If per-token accuracy is at chance (50% for binary, 10% for digits), the model likely cannot predict that token from available context. Check: is the information needed to produce this token actually present in the input?

**Mistake #43:** QueryCountingGenerator placed the query type (DOT/TEN) randomly in the work area. The model hit 51% on it (binary chance). Fix: moved query to input via NOTE marker.

---

## Core Principle: Prerequisite Completeness (Category Theory Constraint)

**Every composition (morphism) requires its constituent objects to be well-defined in the model's learned representation.**

In category theory, composition g ∘ f is only defined when the codomain of f equals the domain of g. In curriculum terms: the OUTPUT representation of skill A must match the INPUT representation needed by skill B. If the model learns counting (objects → digits) but addition treats digits as arbitrary symbols (not quantities), the composition "counting → addition" is undefined. The codomain of counting (digits-as-quantity-labels) doesn't match the domain of addition (digits-as-arbitrary-tokens-in-a-fact-table).

### The prerequisite verification rule

Before teaching a composed skill, verify that:
1. **All prerequisite skills are learned** (not assumed, not skipped)
2. **The model's learned representation of prerequisites includes the properties needed for composition** — counting produces digits, but does the model know that digit 5 means "more than" digit 3? That the digits are ordered? That one TEN equals ten DOTs?
3. **Each composition is explicitly grounded** in an operation the model already performs — addition should REDUCE to counting in the scratchpad, not be taught as an independent lookup table

### Missing prerequisite diagnostic

When a composition stage fails (100% train, chance test), don't just add more data or epochs. Ask:

```
For each input token in this stage:
  Does the model demonstrably understand what this token MEANS?
  Was that meaning explicitly taught and verified in a prior stage?
  Does the scratchpad show HOW prior skills produce the answer?
```

If any answer is "no", there's a missing prerequisite — a gap in the knowledge graph.

### Example: The counting → arithmetic gap

The original curriculum jumped from counting (Stages 1-2) to single-digit arithmetic (Stage 3) with these unverified assumptions:

| Assumption | Verified? | What's missing |
|-----------|-----------|----------------|
| Digits have ordinal structure (5 comes after 4) | **No** | Successor/predecessor stage |
| One TEN = 10 DOTs (place value) | **No** | Place value grounding stage |
| Digits represent comparable quantities (7 > 3) | **No** | Comparison stage |
| Addition means "combine quantities and recount" | **No** | Addition-as-counting stage |

Without these, the model sees `3 + 4 WORK 0 7` with no reason to believe 7 follows from 3 and 4 except as a memorized fact. Counting taught it to associate quantities with digits, but never taught it that digits represent quantities that can be combined.

---

## Core Principle: Fact Stages Are a Design Smell

The original design guide classified stages as either "fact stages" (memorization is the goal) or "composition stages" (generalization is the goal). **This distinction was premature.**

### The old view (partially wrong)
- "Single-digit addition IS a lookup table. There's no sub-algorithm to derive 3+4=7."
- Therefore: don't hold out specs, let the model memorize all 155 facts.

### The corrected view
If the model can count (and it can — Stage 1 generalizes), then there IS a sub-algorithm for 3+4:

```
3 + 4 WORK 3 4 5 6 7 = 0 7
```

Start at 3, count up 4 steps (using the successor function), arrive at 7. This is addition DERIVED from counting, not memorized as an independent fact.

### Why this matters for multi-digit generalization

A model that memorizes 155 single-digit facts has learned a lookup table. When it encounters column arithmetic in Stage 4-5, it has 155 entries to retrieve, but no understanding of WHY 3+4=7. The column scratchpad decomposes multi-digit arithmetic into single-digit operations, but if single-digit operations are opaque lookup tables, the model has no way to verify its own work, detect errors, or generalize the carry mechanism to new situations.

A model that learns addition-as-counting has a PROCEDURE it can apply to any digit pair. The carry mechanism is a natural consequence: when you count past 9, you wrap to 0 and increment the tens digit. The model doesn't need to memorize "8+5=13" — it counts 8→9→10→11→12→13 and observes the tens digit changed.

### When is a fact stage genuinely needed?

A stage should be classified as a fact stage ONLY when:
1. The knowledge is genuinely atomic — no decomposition into simpler skills exists
2. The model has no prior skill that could derive the knowledge
3. The number of facts is small enough that memorization is practical

Examples of genuinely atomic facts:
- Symbol-to-meaning mappings (digit "3" means three objects)
- Axioms that can't be derived (commutativity of addition, if you want to teach it)

Examples of FALSE fact stages (contain hidden compositional structure):
- Single-digit addition (decomposable into counting)
- Multiplication tables (decomposable into repeated addition)
- Subtraction facts (decomposable into counting down)

**Rule of thumb:** If a "fact stage" has more than ~20 entries, look for hidden compositional structure. Large fact tables are a sign that the knowledge graph is missing intermediate nodes.

### Stage classification (revised)

| Stage | Type | Held-out? | What test measures |
|-------|------|-----------|-------------------|
| 1 (query counting) | Composition | Yes | Can the model count unseen (d,t) combos? |
| 2 (combined counting) | Composition | Yes | Can two counts compose? |
| 2.5a (successor/predecessor) | Composition | Yes | Does the model know digit ordering? |
| 2.5b (comparison) | Composition | Yes | Can the model compare quantities? |
| 3 (addition as counting-up) | **Composition** | **Yes** | Does addition reduce to counting? |
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

### Categorical structure of curricula

A curriculum forms a category where:
- **Objects** = skills/concepts the model can learn (counting, ordinality, addition, etc.)
- **Morphisms** = "reduces to" or "builds on" relationships (addition reduces to counting)
- **Composition** = transitive skill dependencies (addition builds on counting, which builds on object recognition)
- **Identity** = every skill trivially reduces to itself

The **composition law** is the key constraint: if f: counting → ordinality and g: ordinality → addition, then g ∘ f: counting → addition must be well-defined. In curriculum terms: the model must be able to trace any composed skill back through all its prerequisites.

### Morphism well-definedness = prerequisite completeness

A morphism f: A → B is well-defined only if:
1. Object A is established (the prerequisite skill is verified learned)
2. Object B's domain matches A's codomain (the output format of A is the input format of B)
3. The morphism itself is taught (the scratchpad shows how A's skill produces B's output)

**If any of these three conditions fail, the composition is undefined and the model will memorize instead of compose.**

### Revised graph design rules

1. **Leaf nodes are genuinely atomic** — symbol-to-meaning mappings, single axioms, irreducible definitions. NOT lookup tables that contain hidden structure.
2. **Every internal node must have explicit incoming edges** — if a skill has prerequisites, ALL prerequisites must be leaf or internal nodes with their own stages. Missing edges = missing stages.
3. **Edge direction encodes reduction** — if A → B, then B's scratchpad must SHOW the reduction to A. The work area literally contains A's format as a sub-computation.
4. **Per-token diagnostics per graph node** — when a node fails, the failing token tells you which incoming edge is broken.
5. **Graph completeness check** — for every internal node, ask: "Does the model provably understand every input symbol's meaning?" If not, there's a missing incoming edge from a node that teaches that meaning.
6. **Codomain/domain matching** — the tokens produced by stage A must be consumed by stage B in the same semantic role. If A produces digits-as-quantity-labels, B must consume them as quantities, not as arbitrary symbols.
7. **No large fact tables** — if a leaf node has >20 entries, it likely contains compositional structure. Decompose it into a sub-graph with genuine leaves and internal composition nodes.

---

## Checklist for Designing a New Stage

### Prerequisite completeness (category theory check)
- [ ] List ALL prerequisite skills this stage assumes
- [ ] For each prerequisite: is it taught and verified in a prior stage?
- [ ] For each input token: does the model demonstrably understand its MEANING (not just its token ID)?
- [ ] If this stage composes two skills A and B: does A's output format match B's input format?
- [ ] If this is marked as a "fact stage": verify there is NO compositional decomposition into simpler skills the model already has. If there is, decompose it.

### Standard checks
- [ ] Every work token is deterministically derivable from input + previous work tokens
- [ ] The scratchpad explicitly shows HOW prior skills produce the answer (reduction, not lookup)
- [ ] Scratchpad format reuses sub-sequences from prerequisite stages
- [ ] Per-token accuracy will be displayed (always, even n_result=1)
- [ ] Structural tokens (NOTE, SEP) are used consistently
- [ ] Run a few samples through `decode_tokens()` and verify by hand
- [ ] Composition stages: held-out specs, test measures generalization on unseen combinations
