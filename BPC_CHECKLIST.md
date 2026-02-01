# BPC Implementation Checklist (from Appendix F.1)

This checklist contains EVERY detail from Appendix F.1 that must be implemented.
Every code change MUST be checked against this list.

## Architecture (energy & MNIST datasets)

- [ ] **4 layers total** (not counting input layer)
  - Layer sizes: [784, 128, 128, 128, 10]
  - Paper: "four-layer neural network with 128 hidden units per layer"

- [ ] **ReLU activations**
  - Paper: "ReLU activations"

- [ ] **Batch size = 128**
  - Paper: "Training is performed using mini-batches of size 128"

## Weight Initialization

- [ ] **W initialization**: Uniform U(-√k, √k) where k = 1/in_features
  - Paper: "We initialized the linear weights W of shape (out_features, in_features) from a uniform distribution U(−√k, √k), where k = 1/in_features"

- [ ] **Bias initialization**: Uniform U(-√k, √k) where k = 1/in_features
  - Paper: "Similarly, the bias b of shape (out_features) is initialized from U(−√k, √k)"

## BPC Optimizer Settings

- [ ] **Adam optimizer for hidden states** (value nodes during inference)
  - Paper: "For BPC, we used the Adam [24] optimizer for hidden states"
  - NOT SGD, NOT AdamW - specifically Adam

- [ ] **Learning rate = 0.01 for hidden states**
  - Paper: "with a learning rate of 0.01"

- [ ] **10 iterations per batch** (T = 10)
  - Paper: "and 10 iterations per batch"

- [ ] **Parameter learning rate κ_t = t^(-ε)**
  - Paper: "For the parameter learning rate, we used κ_t = t^(-ε), where t is the total number of updates"

- [ ] **ε = 0.25**
  - Paper: "and −ϵ was set to 0.25"

## Prior Hyperparameters

- [ ] **M^(0) = matrix of zeros**
  - Paper: "We set the prior over the weights M^(0) to be a matrix of zeros of the appropriate size"

- [ ] **V^(0) = 10 * I**
  - Paper: "We set the prior over V^(0) to be 10 · I where I is an identity matrix"

- [ ] **Ψ^(0) = 1000 * I**
  - Paper: "set Ψ^(0) to be 1000 · I"

- [ ] **ν^(0) = d_y + 2**
  - Paper: "Finally, we set ν^(0) to be d_y + 2"

- [ ] **Initial posterior η = prior (except M uses W initialization)**
  - Paper: "All initial estimates of the posterior natural parameters η were set to the same as the prior, besides M which uses the initialisation described for W"

## Algorithm Details

- [ ] **Discriminative training**: Fix z_0 = x and z_L = y
  - Paper (page 3): "trains models in a discriminative manner by fixing the input nodes to z_0 = x^(i) and the output nodes to z_L = y^(i)"

- [ ] **E-step**: Optimize value nodes Z via gradient descent
  - Paper (Algorithm 1, lines 3-8): Iterate until convergence or T iterations

- [ ] **M-step**: Closed-form Bayesian update (Equation 7)
  - Paper (Algorithm 1, line 10): η_l ← η_l^(0) + Σ_n [sufficient statistics]

- [ ] **Minibatch update**: η*_l = (1 - κ)η_l + κη*_l
  - Paper (page 3): "we introduce a learning rate κ and update the natural parameters as η*_l = (1 − κ)η_l + κη*_l"

## Comparison Baselines (for reference)

**BP:**
- Adam optimizer, LR = 0.001

**PC:**
- AdamW for parameters: LR = 0.0002, weight decay = 0.65
- SGD for hidden states: LR = 0.01, momentum = 0.65
- 10 iterations of hidden state updates per batch

## Two Moons Dataset

- [ ] **Same settings as above, but:**
  - Single hidden layer with 100 hidden units
  - Paper: "smaller network architecture consisting of a single hidden layer with 100 hidden units"

## Important Notes

**What the paper does NOT say:**
- ❌ NO "adaptive learning rate" based on 1/λ_max(A_l) - that's Appendix B theoretical analysis
- ❌ NO scaling LR by precision - precision is already in gradient (Equations 15-16)
- ❌ Page 6: "the current estimate of Σ acts as an adaptive learning rate during inference" - meaning it's IN the gradient, not scaling the LR

**What gets optimized:**
- E-step: Value nodes Z (using Adam with LR=0.01)
- M-step: Natural parameters η (using closed-form update with learning rate κ_t = t^(-0.25))
