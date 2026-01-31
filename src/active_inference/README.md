# Active Inference & Curiosity-Driven Learning

This module implements **Active Inference** and **Curiosity-Driven Learning** for data-efficient training of Predictive Coding networks.

## Overview

Instead of randomly sampling training data, this implementation uses **epistemic value** to guide sample selection, prioritizing samples that maximize learning progress. This is based on three foundational theories:

1. **Friston's Active Inference** - Minimize Expected Free Energy (EFE) to balance exploration and exploitation
2. **Oudeyer's Learning Progress** - Focus on the "Zone of Proximal Development" where learning rate is highest
3. **Schmidhuber's Compression Progress** - Seek patterns that maximize compression of the internal model

## Key Concepts

### Expected Free Energy (EFE)

```
EFE = Ambiguity + Risk
    = -Epistemic_Value - Pragmatic_Value
    = -H[p(y|x)] - log p(y*|x)
```

- **Epistemic Value** (Exploration): How much will observing this sample reduce uncertainty?
- **Pragmatic Value** (Exploitation): How much will this sample help achieve goals?

Lower EFE → Higher priority for sampling.

### Learning Progress

```
Learning_Progress = -d(Error)/dt ≈ -(Error_t - Error_{t-1})
```

Positive learning progress indicates the model is actively learning from a sample.

### Sample Categories

1. **Mastered** - Low error, minimal learning progress (already learned)
2. **Learnable** - Medium error, positive learning progress (Zone of Proximal Development)
3. **Noise** - High error, no progress (unlearnable)
4. **Unvisited** - Not yet explored

## Components

### 1. LearningProgressTracker

Tracks per-sample learning dynamics.

```python
from src.active_inference import LearningProgressTracker

tracker = LearningProgressTracker(
    num_samples=1000,
    window_size=20,          # Keep last 20 observations per sample
    mastery_threshold=0.1,   # Error < 0.1 = mastered
    noise_threshold=2.0,     # Error > 2.0 with no progress = noise
)

# Update after processing a sample
stats = tracker.update(sample_id=42, error=0.5)

print(f"Learning progress: {stats['learning_progress']}")
print(f"Category: {stats['category']}")  # 'mastered', 'learnable', or 'noise'
```

### 2. ExpectedFreeEnergyCalculator

Computes EFE for sample selection using Friston's formulation.

```python
from src.active_inference import ExpectedFreeEnergyCalculator
import torch

efe_calc = ExpectedFreeEnergyCalculator(
    num_classes=10,
    epistemic_weight=1.0,   # Weight for exploration
    pragmatic_weight=1.0,   # Weight for exploitation
    temperature=1.0,
)

# Compute EFE for a prediction
logits = torch.randn(10)
preferred_class = torch.tensor(5)

result = efe_calc.compute_expected_free_energy(
    logits,
    preferred_outcome=preferred_class
)

print(f"EFE: {result['efe']}")
print(f"Epistemic value: {result['epistemic']}")  # H[p(y|x)]
print(f"Pragmatic value: {result['pragmatic']}")  # log p(y*|x)
print(f"Priority: {result['priority']}")          # Higher = more important
```

### 3. ActiveCurriculumManager

Manages intelligent sample selection for training.

```python
from src.active_inference import ActiveCurriculumManager

manager = ActiveCurriculumManager(
    num_samples=6000,
    num_classes=10,
    sampling_strategy='learning_progress',  # 'random', 'learning_progress', 'pure_epistemic', 'balanced'
    temperature=1.0,
    exploration_rate=0.1,
)

# Training loop
for epoch in range(num_epochs):
    manager.start_epoch()

    # Get samples ordered by learning priority
    epoch_indices = manager.get_epoch_indices(prioritize_learnable=True)

    for sample_idx in epoch_indices:
        data, target = dataset[sample_idx]

        # Train on sample
        output = model(data, target)
        error = compute_error(output, target)

        # Update curriculum manager
        manager.update(
            sample_idx=sample_idx,
            error=error,
            logits=output,
            target=target,
        )

    # Print status
    manager.print_status()
```

### 4. ActiveInferenceDiagnostics

Visualizes and analyzes training dynamics.

```python
from src.active_inference import ActiveInferenceDiagnostics

diagnostics = ActiveInferenceDiagnostics(num_samples=6000, num_classes=10)

# During training
for step in range(training_steps):
    # ... train ...

    diagnostics.update(
        sample_idx=idx,
        error=error,
        category=stats['category'],
        efe=stats.get('efe'),
        epistemic=stats.get('epistemic_value'),
        pragmatic=stats.get('pragmatic_value'),
    )

# After training
diagnostics.print_summary()
diagnostics.save_plots('./plots')

# Individual plots
fig = diagnostics.plot_learning_curves()
fig = diagnostics.plot_category_evolution()
fig = diagnostics.plot_visit_distribution()
fig = diagnostics.plot_efe_components()
```

## Sampling Strategies

### Random (Baseline)
```python
sampling_strategy='random'
```
Standard random shuffling. No active inference.

### Learning Progress
```python
sampling_strategy='learning_progress'
epistemic_weight=1.0
pragmatic_weight=1.0
```
Prioritizes samples with highest learning progress (Oudeyer et al.).

### Pure Epistemic (Maximum Exploration)
```python
sampling_strategy='pure_epistemic'
epistemic_weight=1.0
pragmatic_weight=0.0
```
Maximizes uncertainty reduction. Pure exploration.

### Balanced (Exploration + Exploitation)
```python
sampling_strategy='balanced'
epistemic_weight=1.0
pragmatic_weight=1.0
```
Balances epistemic and pragmatic value using full EFE.

## Expected Results

Compared to random sampling, active inference should provide:

1. **Faster Convergence** - Reach target accuracy in fewer samples
2. **Better Data Efficiency** - Learn from fewer total observations
3. **Developmental Stages** - Easy samples mastered first, then harder ones
4. **Adaptive Curriculum** - Automatically focus on learnable content

## Example: MNIST Training

See `experiments/categorical_pc/train_vision_mnist_active.py` for a complete example.

```bash
cd experiments/categorical_pc
python train_vision_mnist_active.py
```

Expected behavior:
- **Epoch 1**: High exploration, visiting many samples
- **Epoch 2-3**: Focus on learnable samples (digits 4, 7, 8, 9)
- **Epoch 4-5**: Most samples mastered, fine-tuning on difficult cases

## Theoretical Background

### Free Energy Principle (Friston)

The brain minimizes **variational free energy**, which upper-bounds surprise:

```
F = D_KL[Q(s) || P(s|o)] - ln P(o)
  ≥ -ln P(o)  (Surprise)
```

Where:
- Q(s) = Approximate posterior (brain's belief about hidden states)
- P(s|o) = True posterior (given observations)
- P(o) = Model evidence (surprise if observation occurred)

### Active Inference

To minimize free energy, agents can:
1. **Update beliefs** (Perception) - Change Q(s) to fit observations
2. **Change observations** (Action) - Sample data to minimize expected free energy

Expected Free Energy guides action selection:

```
G(π) = E_Q(o,s|π)[ln Q(s|π) - ln P(o,s)]
     = Ambiguity + Risk
```

Policies (sequences of samples) are selected to minimize G(π).

### Learning Progress (Oudeyer)

Agents are intrinsically motivated to maximize the **rate** of competence improvement:

```
LP = -dC/dt
```

Where C is a competence measure (e.g., prediction error).

This creates a self-organizing curriculum:
- Avoid mastered content (LP ≈ 0, low error)
- Avoid unlearnable noise (LP ≈ 0, high error)
- Focus on learnable content (LP > 0, moderate error)

### Compression Progress (Schmidhuber)

Interestingness is the first derivative of compressibility:

```
I(t) = C(t-1) - C(t)
```

Where C is the compression ratio of the world model.

Data that enables better compression is intrinsically rewarding.

## Mathematical Details

### EFE Decomposition

For classification tasks:

```
G(x) = -H[p(y|x)] - log p(y*|x)
```

Where:
- H[p(y|x)] = Entropy of prediction (epistemic value)
- y* = Preferred outcome (goal)
- p(y*|x) = Probability of achieving goal

### Priority Score

```
Priority(x) = -G(x) = H[p(y|x)] + log p(y*|x)
```

Higher priority → Sample this next.

### Softmax Sampling

```
p(sample i) ∝ exp(Priority(i) / temperature)
```

Temperature controls exploration:
- High T → More uniform (exploration)
- Low T → More peaked (exploitation)

## Performance Considerations

### Memory Efficiency

- Learning progress tracker: O(N * W) where N = samples, W = window size
- EFE calculator: O(1) per sample
- Cached logits: O(N * C) where C = num_classes

### Computational Cost

**Learning Progress**: Very cheap
- Simple moving average and derivative estimation
- No additional forward passes needed

**EFE (with cache)**: Cheap
- Uses cached logits from training forward pass
- Only entropy and log-prob computations

**EFE (without cache)**: Expensive
- Requires forward pass for all samples each epoch
- Use cached version or learning progress instead

## References

1. **Friston, K. (2010).** The free-energy principle: a rough guide to the brain? *Trends in Cognitive Sciences*.

2. **Friston, K. et al. (2017).** Active Inference, Curiosity and Insight. *Neural Computation*.

3. **Friston, K. et al. (2020).** Generalized Free Energy and Active Inference. *Biological Cybernetics*.

4. **Oudeyer, P.Y. et al. (2007).** Intrinsic Motivation Systems for Autonomous Mental Development. *IEEE Transactions on Evolutionary Computation*.

5. **Schmidhuber, J. (2010).** Formal Theory of Creativity, Fun, and Intrinsic Motivation. *IEEE Transactions on Autonomous Mental Development*.

6. **Millidge, B. et al. (2020).** Deep Active Inference as Variational Policy Gradient. *arXiv:2009.04820*.

7. **Pathak, D. et al. (2017).** Curiosity-driven Exploration by Self-supervised Prediction. *ICML*.

## Contributing

To add new sampling strategies:

1. Add strategy name to `ActiveCurriculumManager.__init__`
2. Implement in `ActiveCurriculumManager.get_next_batch`
3. Add tests in `tests/test_active_inference.py`
4. Document expected behavior

## License

MIT License - See LICENSE file for details.
