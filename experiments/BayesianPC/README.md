# Bayesian Predictive Coding (CORRECT Implementation)

Implementation of **Bayesian Predictive Coding** from:
> "Bayesian Predictive Coding" by Alexander Tschantz, Magnus Koudahl, Hampus Linander, Lancelot Da Costa, Conor Heins, Jeff Beck, Christopher Buckley (2025)
> arXiv:2503.24016
> VERSES AI Research Lab

## What is Bayesian Predictive Coding?

BPC extends standard Predictive Coding by placing **posterior distributions over network weights** (not hidden states). This enables:
- **Epistemic uncertainty quantification** - know what the network doesn't know
- **Faster convergence** - closed-form Bayesian updates (no gradient descent on weights)
- **Biologically plausible** - Hebbian weight updates (pre/post-synaptic activity)
- **Better calibration** - uncertainty decreases with more data

## Key Differences from Standard PC

| Aspect | Standard PC | Bayesian PC |
|--------|------------|-------------|
| **Value nodes** | MAP estimates (scalars) | MAP estimates (scalars) |
| **Weights** | Point estimates | Matrix Normal Wishart posterior |
| **Learning** | Gradient descent | Closed-form Bayesian update (Eq 7) |
| **Weight updates** | Backprop-style | Hebbian (function of activity) |
| **Architecture** | Flexible | Weights OUTSIDE activation (conjugacy) |
| **Convergence** | Slow (many epochs) | Fast (few epochs in full-batch) |
| **Uncertainty** | None | Epistemic + aleatoric |

## CRITICAL: What is Bayesian?

**Bayesian treatment is on WEIGHTS, not value nodes!**

- ❌ **WRONG** (mistake #12): Make value nodes Bayesian (distributions over hidden states)
- ✅ **CORRECT**: Make weights Bayesian (distributions over parameters)

Value nodes are ephemeral (reset each forward pass). Weights accumulate knowledge across training.

## Algorithm Overview

Following Algorithm 1 from the paper:

### E-step (Inference)
Optimize value nodes z via gradient descent on free energy:
```python
for t in range(T):
    z ← z - α·∇_z E(z, λ)
```
where E(z, λ) = task_loss + Σ_l <(z_l - W_l f(z_{l-1}))^T Σ_l^{-1} (z_l - W_l f(z_{l-1}))>

### M-step (Learning)
Closed-form Bayesian update of natural parameters (Equation 7):
```python
η_l* = η_l^(0) + Σ_n [f(z*_{l-1})f(z*_{l-1})^T,
                     f(z*_{l-1})z*_l^T,
                     z*_l z*_l^T,
                     1]
```

This is a **Hebbian update** - function of pre-synaptic f(z_{l-1}) and post-synaptic z_l activity!

## Architecture Constraint

**CRITICAL:** Weights must be OUTSIDE activation function:

```python
# CORRECT (enables conjugacy):
z_l = W_l · f(z_{l-1}) + noise

# WRONG (breaks conjugate priors):
z_l = f(W_l · z_{l-1}) + noise
```

This architectural constraint is essential for closed-form updates.

## Files

- `bayesian_pc_layer.py` - BayesianPCLayer with Matrix Normal Wishart weight posterior
- `bayesian_pc_trainer.py` - Two-phase EM algorithm (E-step + M-step)
- `train_mnist_bayesian.py` - MNIST training script
- `README.md` - This file

## Usage

```python
from experiments.BayesianPC import BayesianPCNetwork, BayesianPCTrainer

# Create model
model = BayesianPCNetwork(
    layer_sizes=[784, 256, 256, 256, 256, 256, 128, 10],
    activation='relu',
)

# Create trainer
trainer = BayesianPCTrainer(
    model=model,
    T=35,  # Inference iterations
    inference_lr=0.01,  # Value node learning rate
    kappa=0.25,  # Natural param learning rate decay
    device='cuda',
)

# Train on batch
results = trainer.train_on_batch(
    inputs=x,
    loss_fn=F.cross_entropy,
    targets=y,
)

# Test (uses expected weights)
results = trainer.test_on_batch(
    inputs=x_test,
    loss_fn=F.cross_entropy,
    targets=y_test,
)
```

## Training on MNIST

```bash
python experiments/BayesianPC/train_mnist_bayesian.py
```

**Expected results:**
- Comparable to baseline PC (95.63% test accuracy)
- Faster convergence in full-batch setting
- Uncertainty quantification included in diagnostics

## Matrix Normal Wishart Distribution

Weight posterior: q(W_l, Σ_l) = MatrixNormalWishart(W_l, Σ_l | M_l, V_l, Ψ_l, ν_l)

**Parameters:**
- M_l: Mean weight matrix (out_features × in_features)
- V_l: Column covariance (in_features × in_features)
- Ψ_l: Wishart scale (out_features × out_features)
- ν_l: Degrees of freedom (scalar)

**Natural parameters:** η_l = [V_l^{-1}, M_l V_l^{-1}, Φ_l + M_l V_l^{-1} M_l^T, ν_l - d_y + d_x - 1]

**Why Matrix Normal Wishart?**
- Conjugate prior for Gaussian likelihood with unknown mean and covariance
- Enables closed-form Bayesian updates (no gradient descent needed)
- Natural parameters accumulate sufficient statistics

## Comparison to Baseline

**Baseline PC:**
- 7 layers: [784, 256, 256, 256, 256, 256, 128, 10]
- T=35 inference iterations
- 95.63% test accuracy (epoch 1), 96.31% (epoch 2)

**BPC Goal:**
- Match or exceed baseline accuracy
- Demonstrate faster convergence (fewer epochs in full-batch)
- Provide meaningful uncertainty estimates

## References

1. Tschantz et al. (2025). "Bayesian Predictive Coding". arXiv:2503.24016
2. Whittington & Bogacz (2017). "An approximation of the error backpropagation algorithm in a predictive coding network with local hebbian synaptic plasticity". Neural Computation.
3. Millidge et al. (2022). "Predictive coding approximates backprop along arbitrary computation graphs". Neural Computation.

## Implementation Notes

- Value nodes optimized with Adam (not SGD like baseline PC)
- Natural parameters updated with learning rate schedule κ_t = t^{-κ}
- Minibatch updates use stochastic natural gradient descent
- Full-batch updates are exact Bayesian inference

## Mistake #12

This implementation supersedes the INCORRECT version in `experiments/BayesianPC_INCORRECT_ARCHIVED/` which mistakenly put posterior distributions over value nodes instead of weights. See MISTAKES.md for details.
