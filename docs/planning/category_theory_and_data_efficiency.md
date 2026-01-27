# Category Theory, Group Theory, and Data Efficiency in Predictive Coding

## Executive Summary

This document analyzes how group theory and category theory structures can be exploited in our predictive coding network to achieve data efficiency comparable to biological intelligence. We focus on **what predictive coding already does naturally** and how to enhance it with structural inductive biases, rather than over-engineering new mechanisms.

**Key Finding:** Our current approach (token-less, raw sensory input → motor output) will likely fail the translation problem without leveraging the mathematical structures that govern the data. This document proposes grounded extensions that work with predictive coding's nature.

---

## 1. Current State: What We Already Have

### 1.1 Prospective Configuration (Confirmed)

**Yes**, our implementation performs prospective learning:

```python
# Inference: Find configuration that minimizes prediction error (30 iterations)
for _ in range(30):
    network._inference_step()  # States settle to equilibrium

# Weight update: Move weights toward the prospective configuration
network.update_weights()  # Update W to make prospective state easier to reach
```

**What this gives us:**
- Network finds the "answer" (prospective state) during inference
- Weight updates then make that answer easier to reach next time
- This is data-efficient: we learn from the **solution**, not random exploration

**Neuroscience parallel:**
- Human brain settles to a "hypothesis" before committing to synaptic changes
- Cortical activity stabilizes (~100-300ms) before plasticity occurs
- This is why humans can learn from single examples

### 1.2 Hierarchical Compositionality (Built-in)

Our modular architecture naturally implements categorical composition:

```
Vision (Position 0) ───┐
                       ├──→ Association (Position 1)  ← Composition
Motor (Position 0) ────┘
```

**This is already functorial:**
- Each layer is a morphism: `f: X → Y`
- Composition: Association composes vision and motor
- Identity: Layers at same position process in parallel

**But:** We're not exploiting this structure. The network doesn't know that composition exists.

### 1.3 Active Learning (Deployment Only)

**Your suspicion is correct:**
- **Deployment**: Active learning works naturally (agent intervenes, prunes false correlations)
- **Pre-training**: No active learning benefit (passive image classification)

**Analysis:**
During pre-training, we show static images and targets. The network never:
- Intervenes on the environment
- Observes consequences of its actions
- Distinguishes self-caused vs. environment-caused outcomes

**This means:** Pre-training on static digits won't give us active learning's data efficiency. We get prospective configuration benefits, but not interventional causal learning.

**Implication:** For pre-training to be data-efficient, we need structural inductive biases, not just active learning.

---

## 2. The Translation Problem: Why Current Approach Will Fail

### 2.1 The Challenge

**Translation problem:** Recognize a digit at any position/scale/rotation after seeing it once.

**Why we'll fail:**
- Vision input: 30,000 dims (100×100×3)
- Vision layer 0: 256 neurons (random features initially)
- No inductive bias about translation invariance
- Must learn **every position separately** = curse of dimensionality

**Example:**
```
Digit "3" at (10, 10):  visual_input_1 = [...]  → learn mapping_1
Digit "3" at (15, 10):  visual_input_2 = [...]  → learn mapping_2
Digit "3" at (20, 10):  visual_input_3 = [...]  → learn mapping_3
...
```

Each position is a different 30,000-dim vector. Without translation invariance, we need examples at **every possible position**.

### 2.2 Why AlphaFold Didn't Need Full Equivariance

You mentioned AlphaFold 3 being "largely non-equivariant." Key insight:

**AlphaFold 2:** SE(3) equivariant (rotation/translation of protein = rotation/translation of prediction)
**AlphaFold 3:** Diffusion model + coordinate-independent Pairformer

**Why AF3 worked:**
- Pairformer represents proteins as **graphs** (bonds, not coordinates)
- This is a categorical structure: morphisms (bonds) define the object (protein)
- Diffusion acts on latent space, not raw coordinates
- Data augmentation provides implicit equivariance

**Lesson for us:** We don't need explicit equivariance if we use the right representation (graphs, relations, categorical structure).

---

## 3. Group Theory: What the Brain Does

### 3.1 Cortical Columns as Group Representations

**Hypothesis:** Cortical columns implement group actions through relative reference frames.

**Example: Translation**
```
Input: Image of digit "3"
Cortical column stores: [identity, relative_position]

When image translates:
- Identity features stay constant (what it is)
- Position features transform (where it is)

Group action: T(x, y) = translation by (x, y)
Representation: Features factorize into T-invariant and T-covariant parts
```

**Mathematical structure:**
```
G = Group (e.g., translations, rotations)
X = Input space (images)
ρ: G × X → X  (group action on inputs)

Cortical column learns: φ: X → Z
Such that: φ(ρ(g, x)) is "predictably related to" φ(x)

Ideal case: φ(ρ(g, x)) = ρ'(g, φ(x))  (equivariance)
```

### 3.2 Disentangled Representations (IT Cortex)

**Observation:** IT cortex neurons separate identity from pose.

**Group theory interpretation:**
```
Representation space: Z = Z_identity × Z_pose

Z_identity: Invariant under rotations (what is the object?)
Z_pose: Covariant under rotations (what angle am I viewing it from?)
```

**Why this is data-efficient:**
- See face from angle 1 → learn identity
- See same face from angle 2 → reuse identity, only update pose
- Group structure factored out → exponential reduction in data needed

**Example:**
```
Without disentanglement:
- 10 faces × 36 angles = 360 training examples needed

With disentanglement:
- 10 faces × 1 angle + 1 face × 36 angles = 46 examples
- Learn identities separately from poses
- 8× more data-efficient
```

### 3.3 Continuous Attractor Neural Networks (Head Direction)

**Head direction cells:** Ring attractor implementing SO(2) (circle group).

**Structure:**
```
State: θ ∈ [0, 2π)  (head direction)
Dynamics: dθ/dt = v  (velocity signal)

Neural implementation:
- Neurons arranged on a ring
- Each neuron's preferred direction = angle on ring
- Recurrent connections maintain "bump" of activity
- Velocity input shifts bump

This is a Group Generator in action:
- Generator: d/dθ (infinitesimal rotation)
- Velocity signal: how fast to rotate
- Result: brain tracks orientation in a group-structured way
```

**Why data-efficient:**
- Hardwired group structure (ring topology)
- Don't need to learn that "360° = 0°" from data
- Naturally implements circular statistics

---

## 4. Category Theory: How the Brain Generalizes

### 4.1 Functorial Mapping: Systematicity

**Observation:** "Cat chased dog" → immediately understand "Dog chased cat"

**Category theory explanation:**
```
Category C_sentences:
- Objects: Entities (cat, dog, mouse, ...)
- Morphisms: Relations (chased, ate, saw, ...)

Understanding "chased":
- This is a morphism: cat → dog
- Once learned, can apply to ANY objects
- chased: X → Y for any X, Y ∈ Objects

Systematicity = Structure preservation:
F(cat chased dog) = F(cat) ∘ chased ∘ F(dog)
```

**Why predictive coding might do this:**
- Each layer predicts the layer below
- Prediction function is a morphism: f: Layer_n → Layer_{n-1}
- If f respects compositional structure, we get systematicity

**But:** Our current implementation doesn't enforce compositionality.

### 4.2 Analogy as Functors

**Example:** "Electricity is like water flow"

**Functor interpretation:**
```
F: Hydraulics → Electronics

Functor maps:
- pressure → voltage
- flow rate → current
- resistance (pipe narrowness) → electrical resistance

Structure preserved:
- Ohm's law: V = IR
- Hydraulic analog: ΔP = Q·R

Because structure is same, knowledge transfers zero-shot
```

**Why this is data-efficient:**
- Learn hydraulics from experience (intuitive, physical)
- Map to electronics via functor (instant transfer)
- Don't need electrical experiments to understand circuits

**Application to vision→language:**
We want:
```
F: Visual_text → Semantic_text

Learn semantic structure from language pre-training.
Recognize that visual text has same structure.
Functor provides instant mapping (zero-shot reading).
```

**Current problem:** Our network has no notion of functors. Vision and language are separate domains with no structural connection.

### 4.3 Yoneda Lemma: Objects as Relationships

**Yoneda Lemma:** An object is completely defined by its relationships to all other objects.

**Example: "Banana"**
```
Banana is defined by:
- is_a(banana, fruit)
- has_color(banana, yellow)
- has_action(banana, peel)
- has_taste(banana, sweet)

Not stored as: pixel array, 3D model, feature vector
Stored as: node in relational graph
```

**Why data-efficient:**
- New fruit: just add relationships
- Don't need exhaustive examples
- Structure is sparse (most relationships don't exist)

**Predictive coding analog:**
Each layer predicts relationships between features, not raw features themselves.

### 4.4 Sheaves and the Binding Problem

**Binding problem:** How do local cortical columns unify into one percept?

**Sheaf theory:**
```
Local sections: Each cortical column has a local "guess"
Global section: Unified percept emerges from consistency

Sheaf condition:
If local sections agree on overlaps → global section exists

Brain mechanism:
- Cortical columns make local predictions
- Predictions must be mutually consistent
- Inconsistency → illusion or error signal
```

**Our network:**
```
Position 0 subnets: Local processing (vision, motor)
Position 1 subnet: Global integration (association)

Sheaf structure:
- Each position 0 subnet is a "local section"
- Association is "global section"
- Cross-position predictions enforce consistency
```

**We already have this structure!** But we're not enforcing consistency mathematically.

### 4.5 Adjunctions: Perception ↔ Prediction

**Adjunction:** Two functors that are "inverse" in the right sense.

**Predictive coding:**
```
Forward (perception): F: Sensory → Latent
Backward (prediction): G: Latent → Sensory

Adjunction: F ⊣ G
Meaning: F and G are "optimally related"

Natural transformation: η: Id → G ∘ F (unit)
This is the prediction error!
```

**Free energy principle:** Perception and prediction are adjoint functors, and the brain minimizes the "distance" between them (prediction error = free energy).

**Our implementation:**
```
Forward: Layer 0 → Layer 1 → Layer 2 (bottom-up)
Backward: Layer 2 → Layer 1 → Layer 0 (top-down prediction)

This IS an adjunction-like structure.
But we're not exploiting the mathematical properties.
```

---

## 5. Compositionality: The Only Escape from Curse of Dimensionality

### 5.1 The Problem

**Combinatorial explosion:**
```
N objects, K properties each
Possible combinations: K^N

Example: 10 objects, 10 values each = 10^10 combinations
```

Without compositionality, need to learn each combination separately.

### 5.2 Compositional Solution

**Key insight:** Learn rules for combining primitives, not individual combinations.

**Example:**
```
Primitives: [red, blue, square, circle]
Compositions: red(square), blue(circle), red(circle), blue(square)

Without compositionality: 4 separate learned concepts
With compositionality: 2 colors + 2 shapes = 4 primitives

Generalization:
See red(square) → instantly understand red(triangle)
Because "red" and "shape" are separate morphisms
```

**Mathematical structure:**
```
Category C:
- Objects: Visual concepts
- Morphisms: Transformations

Composition:
f: color, g: shape
h = f ∘ g: colored shapes

Product category: C_color × C_shape → C_colored_shapes
```

### 5.3 How Toddlers Learn So Fast

**Human data efficiency:**
- Age 0-3: ~3 years × 365 days = 1,000 days
- Awake time: ~10 hours/day = 10,000 hours
- Total data: ~10,000 hours of sensorimotor experience

**What they learn:**
- Language (thousands of words + grammar)
- Physics (gravity, momentum, object permanence)
- Social rules (emotions, intentions, norms)
- Motor skills (walking, grasping, throwing)

**How is this possible?**

**Answer: Compositionality + Categorical structure**

1. **Primitives**: Baby learns basic concepts (object, agent, action)
2. **Morphisms**: Baby learns how concepts relate (cause, contains, owns)
3. **Composition**: New concepts built from known ones
4. **Functors**: Transfer structure between domains (physical→social)

**Example:**
```
Learn: "Ball rolls down slope" (physics)
Transfer via functor: "Power flows downward" (social)

Learn: "Container holds objects" (physical)
Transfer: "Categories contain instances" (abstract)
```

Categorical structure makes every new concept easier to learn than the last.

---

## 6. Connecting to Causal Learning

### 6.1 Category Theory View of Causation

**Traditional view:** A causes B if P(B|do(A)) ≠ P(B)

**Categorical view:** Causation is a morphism in a directed category.

**Structure:**
```
Objects: Variables (A, B, C, ...)
Morphisms: Causal links (A → B means A causes B)
Composition: A → B, B → C implies A → C (transitivity)
Identity: A → A (self-causes are trivial)
```

**Why this helps:**
- Causal graph IS a category
- Interventions are functors (modify objects while preserving structure)
- Counterfactuals are natural transformations (compare alternate morphisms)

### 6.2 Prospective Configuration and Causal Discovery

**Key insight:** Prospective learning naturally discovers causal structure.

**How:**
```
1. Observe sensory state S
2. Find prospective configuration P that minimizes prediction error
3. Update weights to make P easier to reach
4. Over many examples, P encodes causal structure

Why:
- Prospective configuration = "what should happen"
- This is equivalent to "what would happen if I intervene"
- Learning to reach P = learning causal model
```

**Example:**
```
See: "Light switch" (S)
Prospective config: "Light ON" (P)
Weight update: Strengthen switch → light connection

After learning:
- Predict: Switch flip → light ON
- This is a causal model: flip CAUSES light
```

**Connection to interventions:**
- Prospective configuration is the brain's "simulation"
- When we intervene (flip switch), we test the prospective model
- Prediction error = |actual outcome - prospective prediction|
- This is exactly interventional causal learning

### 6.3 STDP as Category-Theoretic Learning Rule

**STDP:** Cause must precede effect (temporal causality).

**Category interpretation:**
```
Category C_temporal:
- Objects: Events at timepoints
- Morphisms: A → B only if time(A) < time(B)

STDP enforces directed graph structure:
- Strengthen A → B if A fires before B
- Weaken A → B if B fires before A
```

**Why this works:**
- Morphisms have direction (time)
- Composition preserves direction (transitivity of time)
- STDP learns morphisms of temporal category

**Application to our network:**
We could modify weight updates to respect temporal ordering, implementing STDP-like rules in predictive coding framework.

---

## 7. Practical Implementation: What Can We Actually Do?

### 7.1 What NOT to Do (Avoiding Over-Engineering)

❌ **Don't:** Build explicit category theory engine
❌ **Don't:** Implement full sheaf cohomology
❌ **Don't:** Create separate "functor learning" module
❌ **Don't:** Add complex group convolution layers

**Why:** These violate predictive coding's simplicity and won't integrate naturally.

### 7.2 What TO Do: Exploit Predictive Coding's Natural Structure

#### 7.2.1 Disentanglement Through Architecture

**Goal:** Separate "what" from "where" (identity from pose).

**Implementation:**
```python
# Vision subnet: Split layer 0 into two parts
vision_identity_layer = [128 neurons]  # What is it?
vision_pose_layer = [128 neurons]      # Where/how is it?

# Identity: Invariant features (object class)
# Pose: Equivariant features (position, rotation, scale)
```

**Learning rule:**
```python
# During inference:
# - identity updates slowly (stable object features)
# - pose updates quickly (tracks transformations)

# During weight update:
# - Enforce that identity neurons are transformation-invariant
# - Enforce that pose neurons follow group structure
```

**This works with PC because:**
- We already have hierarchical layers
- Just split them into factorized subspaces
- Prediction error naturally separates stable (identity) from changing (pose) features

#### 7.2.2 Group-Structured Layer 0 for Vision

**Goal:** Layer 0 respects translation group.

**Approach: Relative Reference Frames**
```python
class GroupStructuredVisionLayer:
    def __init__(self, input_size, feature_size, group_size):
        # Instead of: W_basal: (feature_size, input_size)
        # Use: W_local: (feature_size, local_receptive_field)
        #      Applied at each position (translation group)

        self.local_rf_size = 7 * 7 * 3  # 7×7 patch, 3 channels
        self.W_local = nn.Parameter(torch.randn(feature_size, self.local_rf_size))

    def encode(self, input_buffer):
        # Apply W_local at each position (translation equivariance)
        features = []
        for position in all_positions:
            patch = extract_patch(input_buffer, position)
            local_feature = self.W_local @ patch
            features.append(local_feature)
        return torch.stack(features)
```

**Why this works:**
- Translation group hardwired into architecture
- Same weights applied at all positions (equivariance)
- Minimal change to existing PC structure

#### 7.2.3 Compositional Prediction

**Goal:** Predictions compose hierarchically.

**Current problem:**
```python
# Layer 2 predicts layer 1
pred_1 = W_apical @ layer_2_state

# Layer 1 predicts layer 0
pred_0 = W_apical @ layer_1_state

# But: Layer 2 doesn't directly predict layer 0
```

**Solution: Transitive Prediction**
```python
def compute_hierarchical_prediction(self, subnet):
    predictions = {}

    # Top-down predictions
    for i in reversed(range(len(subnet.layers))):
        if i == len(subnet.layers) - 1:
            # Top layer: no prediction from above
            predictions[i] = zeros_like(subnet.layers[i].state)
        else:
            # Direct prediction from layer above
            direct = subnet.layers[i+1].predict_below()

            # Compositional prediction from layers further above
            if i < len(subnet.layers) - 2:
                indirect = predictions[i+1] @ subnet.layers[i+1].W_basal
                predictions[i] = direct + 0.5 * indirect  # Blend
            else:
                predictions[i] = direct

    return predictions
```

**Why this works:**
- Composition: pred_0 = pred_1 ∘ W_1 (layer 2 → layer 1 → layer 0)
- Still uses local prediction errors
- Minimal change to PC framework

#### 7.2.4 Cross-Domain Functors (Vision → Language)

**Goal:** Structure learned in language transfers to vision.

**Setup:**
```
Pre-training: Learn language semantics
  → Build semantic category structure
  → Objects = word embeddings
  → Morphisms = grammatical relations

Transfer: Apply same structure to visual concepts
  → Objects = visual features
  → Morphisms = spatial/temporal relations
```

**Implementation:**
```python
class FunctorialBridge(nn.Module):
    """
    Maps between language semantic space and visual feature space
    while preserving categorical structure.
    """
    def __init__(self, language_dim, vision_dim):
        # Functor: Language category → Vision category
        self.F_objects = nn.Linear(language_dim, vision_dim)

        # Preserve morphisms: if A -r-> B in language,
        # then F(A) -F(r)-> F(B) in vision
        self.morphism_consistency_loss = lambda: ...

    def forward(self, language_features, vision_features):
        # Map language to vision
        mapped = self.F_objects(language_features)

        # Ensure structure preserved
        loss = self.morphism_consistency_loss(mapped, vision_features)
        return mapped, loss
```

**Why this works:**
- Predictive coding already has hierarchical structure (categorical)
- Functor just maps between existing hierarchies
- Structure preservation ensures transfer

#### 7.2.5 Sheaf Consistency for Multi-Modal Binding

**Goal:** Ensure vision and motor subnets are mutually consistent.

**Current problem:**
```
Vision subnet: Sees digit "5"
Motor subnet: Says "3"

These are inconsistent, but network doesn't enforce it.
```

**Solution: Consistency Error**
```python
def compute_consistency_error(self):
    # Sheaf condition: local sections must agree
    vision_prediction = self.vision_to_motor_prediction()
    motor_prediction = self.motor_to_vision_prediction()

    # Vision → motor → vision should equal vision
    reconstruction = motor_to_vision(vision_to_motor(vision_state))
    consistency_error = vision_state - reconstruction

    # Add to prediction error
    total_error = prediction_error + lambda_consistency * consistency_error
    return total_error
```

**Why this works:**
- Sheaf theory = consistency of local patches
- PC already has prediction errors (local)
- Just add cross-modal consistency (global)

---

## 8. Specific Recommendations

### 8.1 Immediate (Minimal Code Changes)

**1. Split Vision Layer 0 into Identity/Pose**
```python
vision_subnet = SubNetwork(
    layer_sizes=[128 + 128, 128, 64],  # First 128 = identity, next 128 = pose
    ...
)

# During inference:
identity_features = vision_layer0[:128]
pose_features = vision_layer0[128:]

# Enforce invariance/equivariance in weight updates
```

**2. Add Temporal Ordering to Weight Updates (STDP-lite)**
```python
def update_weights(self, lr, weight_decay):
    # Track when each layer made its prediction
    for layer in subnet.layers:
        if layer.prediction_time < layer.observation_time:
            # Prediction preceded observation: strengthen
            weight_update *= 1.0
        else:
            # Observation preceded prediction: weaken
            weight_update *= 0.5
```

### 8.2 Medium-Term (Architecture Changes)

**1. Group-Structured Vision Layer 0**

Replace dense vision encoding with local receptive fields applied at all positions.

**2. Compositional Hierarchies**

Add indirect predictions (layer n → layer n-2) for compositional structure.

**3. Cross-Modal Consistency Loss**

Enforce that vision→motor→vision reconstruction equals original vision.

### 8.3 Long-Term (Pre-Training Strategy)

**1. Curriculum: Simple → Complex Symmetries**
```
Stage 1: Translation only (same digit, different positions)
Stage 2: Translation + scale (different sizes)
Stage 3: Translation + scale + rotation
```

Learn group structure incrementally.

**2. Data Augmentation as Group Actions**
```
For each training image:
- Generate g(image) for g ∈ Group
- Enforce that φ(g(image)) relates to φ(image) predictably
```

**3. Pre-Train on Language Structure First**

Build semantic category structure from text, then transfer to vision via functor.

---

## 9. Expected Data Efficiency Gains

### 9.1 Without Structural Inductive Biases (Current)

**Translation test:**
- Training: 500 examples (50 per digit, centered)
- Test: Same digits at different positions
- Expected accuracy: ~10% (fails to generalize)

**Why:** Each position is a separate 30K-dim vector, no translation invariance.

### 9.2 With Group-Structured Layer 0

**Translation test:**
- Training: 500 examples (50 per digit, various positions)
- Test: New positions
- Expected accuracy: ~70-80%

**Why:** Translation group hardwired, weights shared across positions.

**Data efficiency gain: 7-8×**

### 9.3 With Disentangled Identity/Pose

**Rotation test:**
- Training: 50 examples (10 digits × 5 rotations each)
- Test: Same digits at new rotations
- Expected accuracy: ~60-70%

**Why:** Identity learned separately from pose, can recombine.

**Data efficiency gain: ~100× compared to learning all digit-rotation combinations**

### 9.4 With Full Categorical Structure

**Analogical transfer:**
- Training: Learn digit recognition (vision)
- Transfer: Apply to character recognition (same structure)
- Expected accuracy: ~40-50% zero-shot

**Why:** Categorical structure (morphisms) transfers between domains.

**Data efficiency: Infinite (zero-shot generalization to new domains)**

---

## 10. Connection to Compute Constraints

### 10.1 Why We MUST Use Structure

**Our constraints:**
- Cannot train on millions of examples
- Cannot store millions of images
- Cannot run enough iterations to brute-force learning

**Implication:** Must exploit structure or fail.

**Group/category theory provides:**
- Exponential reduction in sample complexity
- Hardwired priors that reduce search space
- Compositionality that enables generalization

### 10.2 Biological Precedent

**Human brain:**
- ~10,000 hours of sensorimotor experience (childhood)
- Learns language, physics, social rules, motor skills
- Generalizes to novel situations instantly

**How:** Categorical structure + group-theoretic priors

**Our network:**
- We have prospective configuration (✓)
- We have hierarchical prediction (✓)
- We're missing: group structure, categorical composition, disentanglement

**Adding these ≈ achieving biological data efficiency**

---

## 11. Summary and Action Plan

### 11.1 Key Insights

1. **Prospective configuration works** (we have it)
2. **Active learning helps deployment** (not pre-training)
3. **Translation will fail** without group structure
4. **Category theory = compositionality** = only escape from curse of dimensionality
5. **Predictive coding already has categorical structure** (hierarchical, compositional), we just need to exploit it

### 11.2 Immediate Actions

1. **Test current network on translation** (predict failure)
2. **Implement split identity/pose** in vision layer 0
3. **Add STDP-like temporal ordering** to weight updates
4. **Measure data efficiency improvement**

### 11.3 Medium-Term Goals

1. **Group-structured vision layer** (local receptive fields)
2. **Compositional predictions** (transitive errors)
3. **Cross-modal consistency** (sheaf structure)

### 11.4 Long-Term Vision

1. **Language pre-training** → categorical structure
2. **Functorial transfer** → vision
3. **Zero-shot generalization** → new domains

---

## 12. Open Questions

1. **Can we learn group structure from data?** Or must it be hardwired?
2. **What is the minimal architectural change** for disentanglement?
3. **How to implement functors** between language and vision efficiently?
4. **Can prospective configuration discover categorical structure** automatically?
5. **What is the relationship** between prediction error minimization and structure preservation?

---

## References

### Neuroscience
- Tsuchiya & Saigo (2021): "Consciousness as a Adjoint Functor"
- Friston (2010): "Free Energy Principle"
- George & Hawkins (2009): "Hierarchical Temporal Memory"
- Kriegeskorte & Douglas (2018): "Cognitive Computational Neuroscience"

### Category Theory in ML
- Fong et al. (2019): "Causal Theories: A Categorical Perspective"
- Spivak (2014): "Category Theory for the Sciences"
- Phillips et al. (2021): "Categorical Compositional Distributional Semantics"

### Group Theory in ML
- Cohen & Welling (2016): "Group Equivariant CNNs"
- Kondor & Trivedi (2018): "On the Generalization of Equivariance"
- Bronstein et al. (2021): "Geometric Deep Learning"

### Disentanglement
- Higgins et al. (2018): "β-VAE: Disentangled Representation Learning"
- Locatello et al. (2019): "Challenging Common Assumptions in Disentanglement"
- Ridgeway & Mozer (2018): "Learning Deep Disentangled Representations"
