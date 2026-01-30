# Abstract Reasoning for Danganronpa Agent: Categorical Analysis

## The Challenge

Danganronpa requires:
1. **Reading comprehension** - Parse dialogue, understand meaning
2. **Evidence collection** - Remember testimony, physical evidence
3. **Logical deduction** - Find contradictions between evidence and testimony
4. **Theory formation** - Generate hypotheses about culprit identity
5. **Hypothesis testing** - Evaluate which theory fits all evidence
6. **Action selection** - Present correct evidence at correct time

This is **compositional reasoning** - building complex conclusions from simple premises.

## Categorical Structure of Abstract Reasoning

### 1. **Logical Inference is Composition**

Categorical insight: **Inference rules are morphisms**.

```
Evidence₁ → Conclusion₁ (morphism f)
Conclusion₁ → Conclusion₂ (morphism g)
----------------------------------
Evidence₁ → Conclusion₂ (composition g ∘ f)
```

**Implication:** Abstract reasoning subnet must support **compositional structure**.

```python
# Abstract reasoning area needs:
reasoning = config.add_subnet(
    "reasoning", "simple",
    input_size=evidence_size,
    output_size=conclusion_size,
    # CRITICAL: Must compose with itself (recurrent/iterative)
    is_recurrent=True
)

# Composition: Apply reasoning multiple times
# evidence → conclusion₁ → conclusion₂ → ... → final_answer
```

### 2. **Working Memory is a State Monad**

Categorical insight: **Memory is state threaded through computations**.

In category theory, stateful computation is a **state monad**: `S → (A, S)`
- Input: State S
- Output: Result A and updated state S'

For Danganronpa:
```
State = {collected_evidence, current_hypotheses, dialogue_history}

Reasoning: State → (Action, State')
  Input: Current state
  Output: Next action + updated state
```

**Implication:** Need explicit **working memory subnet** that persists across timesteps.

```python
# Working memory (state)
working_memory = config.add_subnet(
    "working_memory", "simple",
    input_size=memory_capacity,  # e.g., 2048 dims
    output_size=memory_capacity,
    is_recurrent=True  # State persists and updates
)

# Reasoning reads from and writes to memory
reasoning_read = config.add_subnet(
    "reasoning_read", "exponential",
    input_size=memory_capacity,
    output_size=512,  # Abstract concepts
    domain=working_memory,
    codomain=reasoning_concepts
)

reasoning_write = config.add_subnet(
    "reasoning_write", "exponential",
    input_size=512,
    output_size=memory_capacity,
    domain=reasoning_concepts,
    codomain=working_memory
)
```

### 3. **Hypotheses are Coproducts**

Categorical insight: **Alternative theories form coproduct**.

```
Hypotheses = Suspect₁ + Suspect₂ + ... + Suspectₙ

At any time, agent believes ONE hypothesis (coproduct = "or")
```

**Implication:** Hypothesis space is a **coproduct over possible culprits**.

```python
# Each hypothesis is a component
hypothesis_A = config.add_subnet("hypothesis_A", "simple", ...)
hypothesis_B = config.add_subnet("hypothesis_B", "simple", ...)
# ... one per suspect

# Hypothesis space is coproduct
hypotheses = config.add_subnet(
    "hypotheses", "coproduct",
    input_size=evidence_size,
    output_size=max(hypothesis_sizes),
    components=[hypothesis_A, hypothesis_B, ...]
)
```

Agent selects ONE hypothesis based on evidence, switches when contradictions found.

### 4. **Evidence Combination is Product**

Categorical insight: **Multiple pieces of evidence combine via product**.

```
Evidence = Testimony × PhysicalEvidence × TimelineData

Reasoning needs ALL pieces jointly (product = "and")
```

**Implication:** Evidence representation is **product of modalities**.

```python
# Evidence components
testimony = config.add_subnet("testimony", "simple", ...)
physical_evidence = config.add_subnet("physical_evidence", "simple", ...)
timeline = config.add_subnet("timeline", "simple", ...)

# Combined evidence is product
evidence = config.add_subnet(
    "evidence", "product",
    input_size=combined_input,
    output_size=sum([testimony.output_size, physical.output_size, timeline.output_size]),
    components=[testimony, physical_evidence, timeline]
)
```

### 5. **Limits: Most Constrained Consistent Theory**

Categorical insight: **Truth is the limit (universal object satisfying all constraints)**.

In category theory, **limit** = most refined object satisfying all conditions.

For Danganronpa: The correct culprit is the **limit** of evidence - the unique theory that:
- Explains all physical evidence
- Doesn't contradict any testimony
- Fits the timeline
- Accounts for all clues

**Implication:** Theory selection subnet should find **limit** of evidence.

```python
# Theory selection finds limit: most constrained consistent hypothesis
def select_theory(hypotheses, evidence):
    """
    Find limit: hypothesis that satisfies ALL evidence constraints.

    Categorically: Limit is terminal cone over evidence diagram.
    """
    # Filter hypotheses by constraints
    consistent = [h for h in hypotheses if satisfies_all_evidence(h, evidence)]

    # Limit is most refined (strongest constraints)
    return most_constrained(consistent)
```

This can be implemented as a subnet that:
1. Takes evidence (product)
2. Evaluates each hypothesis (coproduct)
3. Outputs most constrained consistent one (limit)

### 6. **Adjunctions: Concrete ↔ Abstract**

Categorical insight: **Adjunctions relate concrete evidence to abstract theories**.

Adjunction: Functor pair (F, G) with F ⊣ G

For reasoning:
```
F: Concrete → Abstract (abstraction functor)
   "This knife was found at scene" → "Suspect had access to weapon"

G: Abstract → Concrete (concretization functor)
   "Suspect A is guilty" → "Predict: knife has suspect A's fingerprints"
```

**Implication:** Reasoning has **bidirectional** structure.

```python
# Abstraction: Evidence → Theory
abstraction = config.add_subnet(
    "abstraction", "exponential",
    input_size=evidence_size,
    output_size=theory_size,
    domain=evidence,
    codomain=theory
)

# Concretization: Theory → Predictions
concretization = config.add_subnet(
    "concretization", "exponential",
    input_size=theory_size,
    output_size=prediction_size,
    domain=theory,
    codomain=predictions
)

# Adjunction condition: Evidence → Predictions via Theory
# Should be consistent: (concretization ∘ abstraction)(evidence) ≈ predictions
```

This is essentially **predictive coding for abstract reasoning** - theories predict evidence, errors update theories.

## Proposed Architecture for Danganronpa Agent

```
┌─────────────────────────────────────────────────────────────┐
│ PERCEPTION (from vision/language)                           │
├─────────────────────────────────────────────────────────────┤
│ Text input (dialogue) → Language encoding                   │
│ Visual input (scene)  → Vision encoding                     │
│                                                             │
│ Combined: Language × Vision (product)                       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ EVIDENCE COLLECTION                                         │
├─────────────────────────────────────────────────────────────┤
│ Evidence = Testimony × Physical × Timeline (product)        │
│   ├─ Testimony: What characters said                       │
│   ├─ Physical: Objects, locations, conditions              │
│   └─ Timeline: Sequence of events                          │
│                                                             │
│ Stored in: Working Memory (state monad)                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ ABSTRACT REASONING                                          │
├─────────────────────────────────────────────────────────────┤
│ Theory Formation:                                           │
│   Evidence → Hypotheses (abstraction functor)              │
│                                                             │
│ Hypotheses = H₁ + H₂ + ... + Hₙ (coproduct)                │
│   Each hypothesis: "Suspect X is guilty because..."        │
│                                                             │
│ Theory Testing:                                             │
│   Hypotheses → Predictions (concretization functor)        │
│   Compare predictions to evidence (limit finding)          │
│                                                             │
│ Iterative Refinement (composition):                        │
│   Evidence → Theory₁ → Theory₂ → ... → Final Theory       │
│   (recurrent reasoning subnet)                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ ACTION SELECTION                                            │
├─────────────────────────────────────────────────────────────┤
│ Action = PresentEvidence + AdvanceDialogue + ... (coproduct)│
│                                                             │
│ Evidence Selection: Theory → BestEvidence (exponential)    │
│   "Given current hypothesis, which evidence contradicts    │
│    opponent's claim?"                                       │
└─────────────────────────────────────────────────────────────┘
```

## Categorical Subnet Specification

```python
config = NetworkConfig()

# Working memory (state monad)
memory = config.add_subnet(
    "working_memory", "simple",
    input_size=2048,
    output_size=2048,
    is_recurrent=True  # State persists
)

# Evidence is product
testimony = config.add_subnet("testimony", "simple", input_size=512, output_size=512)
physical = config.add_subnet("physical", "simple", input_size=512, output_size=512)
timeline = config.add_subnet("timeline", "simple", input_size=512, output_size=512)

evidence = config.add_subnet(
    "evidence", "product",
    input_size=combined,
    output_size=1536,  # 512 + 512 + 512
    components=[testimony, physical, timeline]
)

# Abstraction: Evidence → Theory
abstraction = config.add_subnet(
    "abstraction", "exponential",
    input_size=1536,  # Evidence
    output_size=256,  # Abstract theory
    domain=evidence,
    codomain=theory_space,
    is_recurrent=True  # Iterative reasoning
)

# Hypotheses are coproduct
hypotheses = config.add_subnet(
    "hypotheses", "coproduct",
    input_size=256,
    output_size=256,
    components=[hyp_1, hyp_2, ..., hyp_n]
)

# Concretization: Theory → Predictions
concretization = config.add_subnet(
    "concretization", "exponential",
    input_size=256,  # Theory
    output_size=1536,  # Predicted evidence
    domain=theory_space,
    codomain=evidence
)

# Validate
report = config.validate()
report.print_report()
```

## Key Categorical Insights

1. **Composition enables chaining inferences** - Reasoning subnet must compose with itself

2. **State monad for memory** - Working memory persists and updates across timesteps

3. **Products for evidence combination** - Multiple pieces of evidence combined jointly

4. **Coproducts for hypotheses** - Alternative theories, one selected at a time

5. **Limits for truth-finding** - Correct theory is most constrained consistent explanation

6. **Adjunctions for abstraction/concretization** - Bidirectional concrete ↔ abstract mapping

7. **Exponentials for inference** - Reasoning as function space Evidence → Conclusion

## Critical Difference from Vision/Motor

**Vision/Motor:**
- Feedforward with limited feedback
- Relatively fixed structure
- No long-term state

**Abstract Reasoning:**
- **Deeply recurrent** (iterative refinement)
- **Stateful** (working memory persists)
- **Compositional** (chain inferences)
- **Structured by category theory** (products, coproducts, limits, adjunctions)

The categorical structure is **essential**, not optional - it defines what reasoning IS.

## Validation Test

Use the validator to check abstract reasoning architecture satisfies:
- ✓ Evidence is proper product
- ✓ Hypotheses is proper coproduct
- ✓ Abstraction/concretization are exponentials
- ✓ Adjunction condition (can compose both directions)
- ✓ Reasoning is recurrent (composition with self)

If validated, architecture is **categorically sound** for abstract reasoning.
