# Bayesian Predictive Coding Experiment

## Objective

Test whether **Bayesian inference with uncertainty quantification** improves predictive coding performance on MNIST compared to the baseline (point estimate) implementation.

## Research Questions

1. **Does KL divergence regularization improve generalization?**
   - Hypothesis: KL term prevents overfitting, better test accuracy
   - Measure: Test accuracy, train-test gap

2. **Does uncertainty tracking help with data efficiency?**
   - Hypothesis: Network knows when it's uncertain, can request more data on hard examples
   - Measure: Accuracy vs number of training samples

3. **How does uncertainty evolve during training?**
   - Hypothesis: Uncertainty decreases as network learns
   - Measure: Average variance per layer over epochs

4. **Can we detect when the model is uncertain?**
   - Hypothesis: High variance on misclassified examples
   - Measure: Correlation between variance and prediction accuracy

## Theory: Free Energy with KL Divergence

### Standard PC (Baseline)

```
Energy = 0.5 * (mu - x)^2
```

Just prediction error. No regularization.

### Bayesian PC (This Experiment)

**Variational Free Energy:**
```
F = Accuracy + Complexity
  = E_q[-log p(obs | x)] + KL[q(x) || p(x)]
```

**Accuracy term** (precision-weighted prediction error):
```
E_accuracy = 0.5 * precision * (mu - x_mean)^2 + 0.5 * log(variance)
```

**Complexity term** (KL divergence from prior):
```
KL = 0.5 * (log(prior_var / post_var) + post_var / prior_var +
            (post_mean - prior_mean)^2 / prior_var - 1)
```

### Why This Matters

**KL divergence:**
- Prevents deviating too far from prior beliefs
- Acts as Bayesian regularization
- Balances fitting data vs staying simple
- This is how Bayesian inference differs from maximum likelihood

**Uncertainty quantification:**
- Network maintains variance (how confident it is)
- Can say "I don't know" (high variance)
- Enables active learning (query high-uncertainty examples)
- Tracks epistemic uncertainty (model uncertainty)

## Implementation

### Value Nodes as Distributions

**Standard PC:**
```python
x = point estimate (single value)
```

**Bayesian PC:**
```python
x_mean = mean of posterior distribution
x_log_var = log variance of posterior (for numerical stability)
x_var = exp(x_log_var)  # Actual variance
```

### Energy Computation

```python
# Precision-weighted error
precision = 1.0 / (x_var + min_var)
accuracy = 0.5 * precision * (mu - x_mean)^2 + 0.5 * log(x_var)

# KL divergence from prior
kl = 0.5 * (log(prior_var / x_var) + x_var / prior_var +
            (x_mean - prior_mean)^2 / prior_var - 1)

# Total free energy
energy = accuracy + kl
```

### Hyperparameters

```python
prior_variance = 1.0  # Prior belief about variance
min_variance = 1e-6   # Numerical stability
initial_variance = 0.1  # Starting uncertainty
```

## Experimental Design

### Baseline (Standard PC)

- Use `src/network/pc_layer.py`
- Point estimates only
- Already tested: 95.63% test accuracy in epoch 1

### Treatment (Bayesian PC)

- Use `experiments/BayesianPC/bayesian_pc_layer.py`
- Distributional value nodes (mean + variance)
- KL divergence regularization

### Controlled Variables

- Same architecture: [784, 256, 256, 10] (3-layer)
- Same optimizer: SGD for inference, Adam for weights
- Same learning rates: 0.1 (inference), 0.001 (weights)
- Same dataset: MNIST (60k train, 10k test)
- Same number of inference iterations: T=20 (smaller network)

### Measured Variables

**Performance:**
- Test accuracy
- Train accuracy
- Train-test gap (overfitting measure)
- Convergence speed

**Uncertainty:**
- Per-layer variance (mean)
- Variance on correct vs incorrect predictions
- Variance evolution over epochs
- Correlation between variance and error

**Regularization:**
- KL divergence magnitude
- Comparison to baseline (is KL term active?)

## Predictions

**If Bayesian approach works:**

1. ✓ Test accuracy ≥ baseline (KL regularization helps)
2. ✓ Lower train-test gap (less overfitting)
3. ✓ Variance decreases during training (network becomes more certain)
4. ✓ High variance on misclassified examples (network knows when it's wrong)
5. ✓ Better data efficiency (fewer samples needed for same accuracy)

**If it doesn't work:**

1. ✗ Test accuracy < baseline (KL hurts performance)
2. ✗ Variance doesn't correlate with errors (not meaningful)
3. ✗ Variance stays high (network doesn't learn to be confident)

## Running the Experiment

```bash
# Baseline (already done)
python train_mnist_pc.py

# Bayesian PC
python experiments/BayesianPC/train_mnist_bayesian.py

# Compare results
python experiments/BayesianPC/compare_results.py
```

## Expected Outcomes

Based on Bayesian deep learning literature:

- **Small improvement** in test accuracy (~0.5-1%)
- **Clear uncertainty signal** - variance correlates with errors
- **Regularization effect** - KL prevents overfitting
- **Interpretability** - can say "I don't know" via high variance

## Next Steps If Successful

1. **Active learning**: Use uncertainty to select informative samples
2. **Catastrophic forgetting**: Track variance to detect when forgetting occurs
3. **Out-of-distribution detection**: High variance on novel examples
4. **Hierarchical priors**: Learn priors from data instead of fixing them

## Files

- `bayesian_pc_layer.py` - Core implementation
- `bayesian_pc_trainer.py` - Training loop (same as baseline)
- `train_mnist_bayesian.py` - MNIST experiment
- `README.md` - This file

## References

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Graves, A. (2011). Practical variational inference for neural networks
- Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation
- Kingma & Welling (2014). Auto-encoding variational Bayes
