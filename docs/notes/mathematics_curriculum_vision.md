# Training on Mathematics Curriculum: Vision and Challenges

**Date:** 2026-01-24

## Vision

Train the predictive coding network on the **entire mathematics curriculum** from:
- **Basic arithmetic** (2 + 2 = 4)
- **Algebra** (solve for x: 2x + 3 = 7)
- **Calculus** (derivative of x² = 2x)
- **Differential equations** (dy/dx = y, solve for y(t))

**Purpose:**
1. **Prove stable learning** on structured, sequential tasks
2. **Test continual learning** (can it learn algebra without forgetting arithmetic?)
3. **Multi-task benchmark** (different problem types in one model)
4. **Catastrophic forgetting research** (the key challenge)

## Why Mathematics is Perfect for This

**Advantages over typical AI benchmarks:**

1. **Clear ground truth:** 2+2 = 4 is objectively verifiable
2. **Hierarchical structure:** Calculus builds on algebra builds on arithmetic
3. **Multiple modalities:** Symbolic (equations), numerical (calculations), procedural (solving)
4. **Difficulty gradient:** Can test from trivial to PhD-level
5. **Testable transfer:** Does learning calculus help with algebra?

**Similar to how humans learn math:**
- Start simple (counting)
- Build abstractions (variables)
- Apply to new domains (physics, economics)
- **Don't forget basics** when learning advanced topics!

## Data Format

**Sequence-to-sequence tasks:**

```
Input:  "What is 2 + 3?"
Output: "5"

Input:  "Solve for x: 2x + 3 = 7"
Output: "x = 2"

Input:  "Derivative of x^2 with respect to x"
Output: "2x"

Input:  "Integral of 2x dx"
Output: "x^2 + C"
```

**Or token-level autoregressive:**
```
Input:  [2, +, 3, =]
Output: [5]

Input:  [solve, for, x, :, 2, x, +, 3, =, 7]
Output: [x, =, 2]
```

**Representation options:**
1. **Text tokens:** Like GPT (most flexible)
2. **Symbol embeddings:** Each math symbol gets embedding
3. **Graph structure:** Parse tree of equations
4. **Hybrid:** Structured for math, text for explanations

## Catastrophic Forgetting: The Core Challenge

**What is catastrophic forgetting?**

When neural networks learn task B, they often **completely forget task A**.

**Example:**
1. Train on addition (100% accuracy)
2. Train on multiplication
3. Test addition → **30% accuracy!** (catastrophic forgetting)

**Why it happens:**
- Weights optimized for task A get overwritten for task B
- No mechanism to protect important weights
- Gradient descent is "greedy" - only cares about current task

**Catastrophic forgetting in math curriculum:**
```
Phase 1: Train on arithmetic (2+2, 5*3, etc.)
  → 95% accuracy

Phase 2: Train on algebra (2x+3=7, x^2-4=0)
  → Algebra: 90% accuracy
  → Arithmetic: 40% accuracy!!! (forgot how to add!)

Phase 3: Train on calculus (d/dx x^2, ∫ 2x dx)
  → Calculus: 85% accuracy
  → Algebra: 30% accuracy (forgot!)
  → Arithmetic: 10% accuracy (barely remembers!)
```

## Why Predictive Coding Might Help

**Standard backprop has no memory of old tasks.**

**Predictive coding has built-in mechanisms that might prevent forgetting:**

1. **Hierarchical representations:**
   - Lower layers: basic patterns (digits, operators)
   - Higher layers: abstract concepts (equations, derivatives)
   - Old knowledge stays in lower layers while higher layers learn new tasks

2. **Top-down predictions:**
   - Network maintains "beliefs" about what it expects
   - When learning new task, can check if it contradicts old knowledge
   - If contradiction → gentle update, not overwrite

3. **Local learning rules:**
   - Each layer updates based on local error
   - No global backprop that can destroy remote knowledge
   - More modular learning

4. **Inference phase:**
   - Network "settles" into state before learning
   - Can retrieve old knowledge during inference
   - Then update weights without destroying retrieval ability

**But this is speculative - needs testing!**

## Continual Learning Strategies

If predictive coding alone doesn't prevent forgetting, combine with:

### 1. **Elastic Weight Consolidation (EWC)**

**Idea:** Protect important weights when learning new task

```python
# After learning arithmetic:
fisher_information = compute_fisher_information(weights, arithmetic_data)

# When learning algebra:
loss = algebra_loss + λ * sum(fisher * (W - W_old)²)
# Penalizes changing weights that were important for arithmetic
```

**Analogy:** "Freeze" weights that are critical for old tasks

### 2. **Progressive Neural Networks**

**Idea:** Add new neurons for new tasks, keep old neurons frozen

```python
# Architecture:
Layer 0 (arithmetic): [100 neurons] → frozen
Layer 1 (algebra):    [100 neurons] → connections from Layer 0
Layer 2 (calculus):   [100 neurons] → connections from Layers 0 & 1
```

**Analogy:** Build on top of old knowledge without modifying it

### 3. **Memory Replay**

**Idea:** Interleave old examples when learning new task

```python
# When training on algebra:
batch = {
    algebra_problems:    80%,  # New task
    arithmetic_problems: 20%   # Old task (replay)
}
```

**Analogy:** "Review" old material while learning new

### 4. **Sparse Coding / Dropout**

**Idea:** Force network to use different neurons for different tasks

```python
# Dropout during training:
active_neurons = random_subset(all_neurons, 50%)
# Forces redundancy - multiple pathways for each task
```

### 5. **Meta-Learning (MAML)**

**Idea:** Learn to learn in a way that doesn't forget

```python
# Train on sequence of tasks
# Optimize for: fast learning + no forgetting
# "Inner loop": adapt to new task
# "Outer loop": update meta-parameters to minimize forgetting
```

### 6. **Task-Specific Routing**

**Idea:** Different tasks use different pathways through network

```python
# Gate neurons based on task:
if task == "arithmetic":
    active_pathway = pathway_A
elif task == "algebra":
    active_pathway = pathway_B

# Pathways share some neurons, have some dedicated
```

## Proposed Experiment Design

**Phase 1: Single-Task Baselines** (week 1)
- Train on arithmetic only → measure accuracy
- Train on algebra only → measure accuracy
- Train on calculus only → measure accuracy
- **Goal:** Establish upper bound for each task

**Phase 2: Sequential Training (test forgetting)** (week 2)
- Train arithmetic (to 95% accuracy)
- Then train algebra → **measure arithmetic accuracy** (expect drop!)
- Then train calculus → **measure all previous tasks**
- **Goal:** Quantify catastrophic forgetting

**Phase 3: Continual Learning Interventions** (weeks 3-4)
- Test EWC: Does it preserve arithmetic when learning algebra?
- Test memory replay: How much replay is needed?
- Test progressive nets: Can we scale to 10 task types?
- **Goal:** Find best strategy to prevent forgetting

**Phase 4: Curriculum Design** (week 5)
- Does order matter? (arithmetic→algebra→calculus vs random order)
- Does gradual difficulty help? (easy→medium→hard within each topic)
- Does interleaving help? (arithmetic, algebra, arithmetic, calculus, ...)
- **Goal:** Optimize learning schedule

## Success Metrics

**For stable learning:**
- [ ] Each task reaches >90% accuracy when trained in isolation
- [ ] Training is stable (no divergence) over 1000+ iterations
- [ ] Convergence is smooth (no wild oscillations)

**For continual learning:**
- [ ] After learning N tasks, accuracy on task 1 is still >80%
- [ ] Forward transfer: Learning task N+1 is faster if learned tasks 1..N first
- [ ] Backward transfer: Learning task N+1 improves accuracy on tasks 1..N

**For catastrophic forgetting:**
- [ ] Vanilla sequential training shows forgetting (baseline)
- [ ] EWC reduces forgetting by >50%
- [ ] Memory replay reduces forgetting by >70%
- [ ] Combined strategies prevent forgetting (>95% retention)

## Dataset Sources

**Existing datasets:**
1. **Mathematics Dataset** (Saxton et al., 2019)
   - 2 million questions across 56 categories
   - Arithmetic, algebra, calculus, probability
   - Free download from DeepMind

2. **MATH Dataset** (Hendrycks et al., 2021)
   - 12,500 competition math problems
   - High school to college level
   - Includes step-by-step solutions

3. **Khan Academy** (via scraping, with permission)
   - Full curriculum from K-12 through college
   - Structured by topic and difficulty

**Synthetic generation:**
```python
# Generate infinite examples:
def generate_arithmetic():
    a, b = random.randint(1, 100), random.randint(1, 100)
    op = random.choice(['+', '-', '*', '/'])
    return f"{a} {op} {b} =", eval(f"{a} {op} {b}")

def generate_algebra():
    a, b, c = random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)
    # 2x + 3 = 7 → x = 2
    return f"{a}x + {b} = {c}", (c - b) / a
```

## Integration with Current Network

**Modifications needed:**

1. **Add embedding layer:**
   ```python
   # Convert tokens (text/symbols) to vectors
   self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
   ```

2. **Add output layer:**
   ```python
   # Convert network state to token predictions
   self.output_projection = nn.Linear(hidden_dim, vocab_size)
   ```

3. **Sequence handling:**
   ```python
   # Process sequences of tokens
   # Option A: Treat as batch (all at once)
   # Option B: Recurrent (one token at a time)
   # Option C: Transformer-style (with positional encoding)
   ```

4. **Loss function:**
   ```python
   # Cross-entropy over predicted tokens
   loss = F.cross_entropy(predicted_tokens, target_tokens)
   ```

## Timeline

**Before math curriculum (now):**
- Fix optimizer (Adam/Muon) → 1 week
- Prove stable 400-iteration training → 1 week
- Add activity regularization (prevent saturation) → 3 days

**Math curriculum MVP (4-6 weeks):**
- Week 1-2: Data pipeline + single-task baselines
- Week 3: Sequential training (measure forgetting)
- Week 4-5: Continual learning interventions
- Week 6: Write up results, compare to baselines

**Long-term (3-6 months):**
- Scale to full curriculum (56 categories)
- Test on novel problem types
- Compare to GPT-4, Claude on math reasoning
- Publish findings on predictive coding for continual learning

## Why This Matters

**Scientific contribution:**
- Test if predictive coding naturally prevents forgetting
- Compare to standard continual learning methods
- Understand biological learning vs artificial

**Practical application:**
- If it works: architecture for lifelong learning
- Math reasoning is useful for many domains
- Techniques transfer to other sequential learning tasks

**Alignment with your interests:**
- Structured, verifiable task (not fuzzy vision/language)
- Tests core hypothesis about predictive coding
- Builds toward AGI-like continual learning

## Next Steps

1. **Get network stable** (Adam optimizer, activity regularization)
2. **Start simple:** Single task (arithmetic only)
3. **Measure forgetting:** Sequential training (arithmetic → algebra)
4. **Iterate on solutions:** Try EWC, replay, progressive nets
5. **Scale up:** Full curriculum once proven on subset

The math curriculum is an excellent proving ground for predictive coding's potential as a continual learning architecture!
