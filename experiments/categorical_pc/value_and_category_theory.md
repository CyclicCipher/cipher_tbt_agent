# Value-Based Decisions and Category Theory as Design Tool

## 1. Why We Need Value-Based Decisions for Danganronpa

You're absolutely right - I dismissed vmPFC too quickly.

### The Problem: Motivation

**Without values/goals, the agent has no reason to do anything.**

Current predictive coding formulation:
- Minimize prediction error
- That's it

**But:** Minimizing prediction error could mean:
- Stare at blank wall (very predictable!)
- Do nothing (no errors if you don't predict anything)
- Give up when stuck (avoids prediction errors from hard problems)

**We need:** Something that makes the agent WANT to win Danganronpa.

---

## Value System Requirements

### What the agent needs to value:

1. **Task completion** - Finishing the game, not getting stuck
2. **Information seeking** - Exploring to find evidence (even when uncertain)
3. **Strategic behavior** - Making moves that advance goals, not random actions
4. **Persistence** - Continuing after failures

### vmPFC Computational Properties (Revisited)

From earlier research:
- Exploits **higher-order structure** in decision problems
- Infers interdependencies between actions
- NOT just reward maximization - **structural inference**

**Key insight:** vmPFC doesn't just assign values to actions. It builds a **model of how actions relate** and uses that for planning.

---

## Two Approaches to Values

### Approach 1: Reward Shaping (Traditional RL)

```python
# Reward function
reward = 0
if solved_case:
    reward += 100
if found_evidence:
    reward += 10
if advanced_dialogue:
    reward += 1
if stuck_in_loop:
    reward -= 5

# Update: Minimize prediction error + maximize reward
loss = prediction_error + λ * (-reward)
```

**Problem:** Have to hand-craft rewards for everything. Brittle.

### Approach 2: Structural Value Model (vmPFC-like)

```python
# Learn structure of decision problem
action_dependencies = learn_action_structure(experiences)

# Value based on: "Does this action advance my goals?"
# Not: "Does this action get immediate reward?"

# Example:
# "Talk to suspicious character" has low immediate value
# But high structural value (opens investigation paths)
```

**Advantage:** Learns compositional value structure, not just reward associations.

---

## Proposed: Value Subnet (vmPFC-like)

```python
class ValueSubnet:
    """
    vmPFC-like value estimation.

    NOT just reward prediction.
    Models compositional structure of decision problem.
    """

    def __init__(self):
        # Action-outcome model (compositional)
        self.action_model = StructuredModel(
            input_size=512,   # Current state
            output_size=256   # Predicted outcome structure
        )

        # Value estimation based on structure
        self.value_estimator = ValueEstimator(
            input_size=256,   # Outcome structure
            output_size=1     # Scalar value
        )

    def forward(self, state, candidate_action):
        # Predict: What structure does this action create?
        outcome_structure = self.action_model(state, candidate_action)

        # Evaluate: Does this structure advance my goals?
        value = self.value_estimator(outcome_structure)

        return value
```

**Inductive bias:** Actions have compositional structure (not independent).

**Connection to goals:** Value estimator learns to prefer structures that lead to task completion.

---

## Where Does Value Subnet Connect?

```
Vision + Language → Association → Working Memory
                                         ↓
                                   Value Subnet
                                         ↓
                              (scores candidate actions)
                                         ↓
                                   Motor Selection
                                         ↓
                                      Action
```

**Input:** Current state (from working memory)
**Output:** Value estimates for candidate actions
**Used by:** Motor system for action selection

**Size:** ~2-3M params (similar to other reasoning components)

---

## 2. Category Theory as Design Constraint Tool

You're right - I over-corrected. Category theory IS useful, even if the brain doesn't explicitly implement it.

### What Category Theory Provides

**NOT:** A description of what the brain does
**IS:** A constraint system for ensuring compositional structure

Analogy: Type systems in programming
- Languages don't "use type theory"
- But type theory CONSTRAINS programs to be well-formed
- Catches errors at compile time, not runtime

Similarly:
- Brain doesn't "use category theory"
- But categorical constraints ensure **compositional coherence**
- Prevents architectural nonsense

---

## Which Categorical Constraints Are Actually Useful?

### 1. Composition (Definitely Useful)

**Constraint:** If you have f: A→B and g: B→C, you must be able to compose g∘f: A→C

**Why it matters:**
- Ensures hierarchical processing actually works
- Predictions at different levels stay consistent
- Information flows through network without dimension mismatches

**Validator catches:** "Vision outputs 768 dims but Association expects 512"

**Verdict:** **Use this.** Catches real bugs.

---

### 2. Products (Probably Useful)

**Constraint:** If you need both X and Y jointly, create X×Y with proper projections

**Why it matters:**
- Ensures multimodal information is properly combined
- Prevents accidental information loss
- Makes dependencies explicit

**Example:** Evidence = Testimony × Physical × Timeline

**Validator catches:** "Product output should be 1536 but is 1024"

**Verdict:** **Use for multimodal integration.** Makes structure explicit.

---

### 3. Coproducts (Maybe Useful)

**Constraint:** If you have alternatives A | B | C, create A+B+C with selection mechanism

**Why it matters:**
- Makes mutually exclusive choices explicit
- Prevents "doing multiple things at once" bugs

**Example:** Motor = Keyboard + Mouse + Gaze (one at a time)

**Validator catches:** "Coproduct needs selection mechanism"

**Verdict:** **Use for action selection.** Makes exclusivity explicit.

---

### 4. Exponentials/Adjunctions (Probably Over-Engineering)

**Constraint:** Function spaces B^A must have proper currying/uncurrying

**Why it matters:** ???

I proposed this for abstraction/concretization, but:
- Not clear it helps in practice
- Adds complexity without obvious benefit
- Might be theoretical elegance without practical value

**Verdict:** **Skip for now.** Too speculative.

---

### 5. Universal Properties (Too Abstract)

**Constraint:** Products/coproducts must satisfy universal properties

**Why it matters:** Ensures canonical structure

**But:** Validator checks this automatically. We don't need to think about it explicitly during design.

**Verdict:** **Let validator handle it.** Don't over-think.

---

## Revised Position on Category Theory

### USE Category Theory For:

1. **Type checking** (composition, dimensions)
   - Prevents dimension mismatches
   - Ensures information flow works
   - **Validator tool is essential**

2. **Multimodal integration** (products)
   - Makes dependencies explicit
   - Vision × Language × Memory
   - Clear what's needed jointly

3. **Action selection** (coproducts)
   - Makes alternatives explicit
   - Keyboard | Mouse | Gaze
   - Prevents doing multiple things

### DON'T Use Category Theory For:

1. **Over-engineering subnets**
   - 18 different subnet types
   - Exponentials for everything
   - Adjunctions everywhere

2. **Replacing neuroscience**
   - Canonical microcircuit tells us actual structure
   - Temporal hierarchy tells us actual dynamics
   - Use CT to constrain, not replace

3. **Theoretical elegance**
   - If it doesn't catch bugs or clarify design, skip it

---

## Proposed Minimal Categorical Architecture

Using category theory **as constraint tool**, not design driver:

```python
config = NetworkConfig()  # Use validator

# Position 0: Vision (with feedback - from canonical microcircuit)
vision_superficial = config.add_subnet(...)  # Error computation
vision_deep = config.add_subnet(...)        # Output
vision = config.add_subnet("vision", "product", components=[...])

# Position 0: Motor (coproduct - CT ensures selection)
motor = config.add_subnet("motor", "coproduct",
                         components=[keyboard, mouse, gaze])

# Position 1: Association (product - CT ensures multimodal integration)
assoc = config.add_subnet("association", "product",
                         components=[vision_features, language_features])

# Position 2: Working Memory (simple - no CT structure needed)
memory = config.add_subnet("working_memory", "simple",
                          is_recurrent=True)

# Position 2: Value (simple - just function approximation)
value = config.add_subnet("value", "simple", ...)

# VALIDATE (catches composition errors, dimension mismatches)
report = config.validate()
if not report.is_valid():
    fix_architecture()  # Before implementing!
```

**Total subnets:** ~8-10 (not 18)

**Category theory role:**
- Validator catches bugs
- Products make multimodal integration explicit
- Coproducts make action selection explicit
- But NOT driving the entire design

---

## Summary

### 1. Value-Based Decisions: YES, Need Them

- Can't have agent without motivation
- vmPFC structural model is the right approach
- ~2-3M params for value subnet
- Learns compositional action structure

### 2. Category Theory: Useful as Constraint Tool

**Use for:**
- Dimension checking (composition)
- Multimodal integration (products)
- Action selection (coproducts)

**Don't use for:**
- Over-engineering (18 subnets)
- Replacing neuroscience (canonical microcircuit tells us structure)
- Theoretical elegance without practical benefit

**The validator tool is valuable.** It catches bugs. Keep using it.

But don't let category theory drive the entire architecture. Let neuroscience drive it, use CT to validate.

---

## Revised Minimal Architecture (With Both Insights)

```
Position 0: SENSORY/MOTOR
├─ Vision (layered: superficial + deep, feedback)
├─ Language (text encoding)
└─ Motor (coproduct: keyboard + mouse + gaze)  ← CT constraint

Position 1: ASSOCIATION
└─ Multimodal (product: vision × language)  ← CT constraint

Position 2: ABSTRACT
├─ Working Memory (recurrent, agranular PFC-like)
└─ Value (compositional action model, vmPFC-like)  ← NEW

FEEDBACK:
├─ Attention → Vision (from working memory)
└─ Motor prep → Motor (from value)
```

**Total:** ~8 subnets
**CT validation:** Ensures composition works, multimodal integration is correct, action selection is exclusive
**Neuroscience:** Provides laminar structure, temporal hierarchy, feedback pathways
**Value system:** Provides motivation to actually do things

Does this synthesis make sense?
