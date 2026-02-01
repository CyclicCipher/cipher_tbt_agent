# Bayesian Predictive Coding (BPC) - Complete Implementation Checklist

This checklist contains EVERY implementation detail from the BPC paper. Each item must be verified in the implementation.

---

## 1. ALGORITHM 1 - Main BPC Algorithm (Page 3)

### Algorithm Structure
- [ ] **Line 1**: Randomly initialize η (natural parameters)
- [ ] **Line 1**: Set prior parameters η^(0) = (M^(0), V^(0), Ψ^(0), ν^(0))
- [ ] **Line 2**: Loop over each (x, y) batch
- [ ] **Line 3**: Begin repeat loop for inference
- [ ] **Line 4**: Initialize Z (latent variables)
- [ ] **Line 5**: Loop for l = 1 to L (all layers)
- [ ] **Line 6**: Update z_l ← z_l - α ∂E/∂z_l
- [ ] **Line 7**: End layer loop
- [ ] **Line 8**: Check convergence of z OR maximum T iterations
- [ ] **Line 9**: Loop for l = 1 to L (parameter update)
- [ ] **Line 10**: Update η_l ← η_l^(0) + Σ_n (f(z_{l-1}^{*n})f(z_{l-1}^{*n})^T, f(z_{l-1}^{*n})z_l^{*n⊤}, z_l^{n*}z_l^{n*⊤}, 1)
- [ ] **Line 11**: End parameter loop
- [ ] **Line 12**: End batch loop

### Key Implementation Notes
- [ ] Convergence check: Either z converges OR T iterations reached
- [ ] Maximum iterations T is a hyperparameter
- [ ] Learning rate α is a hyperparameter
- [ ] Natural parameters η accumulate statistics

---

## 2. EQUATIONS 1-11 - Core Mathematical Formulations (Pages 2-4)

### Equation 1: Generative Model (Page 2)

#### Model Structure
- [ ] **Hierarchical model** with L layers
- [ ] **Variables**: Z = {z_l: l = 0...L}
- [ ] **Parameters**: Θ = {W_l, Σ_l: l = 1...L}

#### Joint Distribution
- [ ] **p(Z, Θ)** = p(z_0)p(W_0, Σ_0) ∏_{l=1}^L p(z_l | z_{l-1}, W_l, Σ_l)p(W_l, Σ_l)

#### Likelihood
- [ ] **p(z_l | z_{l-1}, W_l, Σ_l)** = N(z_l | W_l f(z_{l-1}), Σ_l)
- [ ] **CRITICAL**: W_l is OUTSIDE the non-linear activation function f(·)
- [ ] f(·) is a point-wise non-linear activation function

#### Prior Distribution
- [ ] **p(W_l, Σ_l)** = MN(W_l | M_l^(0), Σ_l^{-1}, V_l^(0)) W(Σ_l^{-1} | Ψ_l^(0), ν_l^(0))
- [ ] **Matrix Normal** prior for W_l with parameters: M_l^(0), Σ_l^{-1}, V_l^(0)
- [ ] **Wishart** prior for Σ_l^{-1} with parameters: Ψ_l^(0), ν_l^(0)

#### Base Distribution
- [ ] **p(z_0)** = N(z_0 | μ_0, Σ_0)

#### Implementation Notes
- [ ] **Bias term**: Used in practice but ignored in equations for simplicity
- [ ] **Conjugacy**: Matrix Normal Wishart is conditionally conjugate given network activity Z

### Equation 2: Variational Posterior (Page 2)

#### Posterior Factorization
- [ ] **q_λ(Θ)** = ∏_{l=1}^L q(W_l, Σ_l)

#### Posterior Form
- [ ] **q_λ(W_l, Σ_l)** = MN(W_l | M_l, Σ_l^{-1}, V_l) W(Σ_l^{-1} | Ψ_l, ν_l)

#### Variational Parameters
- [ ] **λ** = {M_l, V_l, Ψ_l, ν_l : l = 1...L}
- [ ] **M_l**: Mean matrix for W_l
- [ ] **V_l**: Column covariance matrix for W_l
- [ ] **Ψ_l**: Scale matrix for Σ_l^{-1}
- [ ] **ν_l**: Degrees of freedom for Σ_l^{-1}

### Equation 3: Objective Function (Page 2)

- [ ] **E(Z, λ)** = ⟨log q_λ(Θ) - log p(Z, Θ)⟩_{q_λ(Θ)}
- [ ] Expectation taken under q_λ(Θ)
- [ ] Equivalent to variational free energy
- [ ] **Assumption**: q_λ(Z) is a Dirac delta distribution

### Equation 4: EM Algorithm (Page 2)

#### E-Step (Inference)
- [ ] **Z^*** = arg min_Z E(Z, λ)
- [ ] Minimize over latent variables Z

#### M-Step (Learning)
- [ ] **λ^*** = arg min_λ E(Z^*, λ)
- [ ] Minimize over variational parameters λ
- [ ] Uses converged latent variables Z^*

### Equation 5: Energy Function Decomposition (Page 3)

#### Energy Expression
- [ ] **E(Z, λ)** = (1/2) Σ_{l=1}^L ⟨(z_l - W_l f(z_{l-1}))^T Σ_l^{-1} (z_l - W_l f(z_{l-1}))⟩_{q_λ(W_l, Σ_l)} + C

#### Components
- [ ] **E_l**: Precision-weighted prediction error for layer l
- [ ] E_l = ⟨(z_l - W_l f(z_{l-1}))^T Σ_l^{-1} (z_l - W_l f(z_{l-1}))⟩_{q_λ(W_l, Σ_l)}
- [ ] **C**: Terms independent of Z
- [ ] **1/2** factor in front of sum

### Equation 6: Gradient Descent Update (Page 3)

#### Update Rule
- [ ] **z_l ← z_l - α ∂E/∂z_l**
- [ ] α is the learning rate (hyperparameter)

#### Gradient Computation
- [ ] **∂E/∂z_l** = (1/2)(∂E_l/∂z_l + ∂E_{l+1}/∂z_l)
- [ ] Combine gradients from layer l and layer l+1
- [ ] **1/2** factor in gradient

#### Iteration
- [ ] Repeat until convergence OR maximum T iterations
- [ ] Converged latent variables denoted Z^⋆ (or Z^*)

### Equation 7: Closed-Form Parameter Update (Page 3)

#### Full-Batch Update
- [ ] **η_l^⋆** = η_l^(0) + Σ_n (f(z_{l-1}^{*n})f(z_{l-1}^{*n})^T, f(z_{l-1}^{*n})z_l^{*n⊤}, z_l^{n*}z_l^{n*⊤}, 1)

#### Components
- [ ] **η_l^(0)**: Natural parameters of the prior
- [ ] **f(z_{l-1}^{*n})f(z_{l-1}^{*n})^T**: Outer product of pre-synaptic activity
- [ ] **f(z_{l-1}^{*n})z_l^{*n⊤}**: Cross-product of pre- and post-synaptic activity
- [ ] **z_l^{n*}z_l^{n*⊤}**: Outer product of post-synaptic activity
- [ ] **1**: Vector of ones (dimension = number of data points n)
- [ ] Sum over all data points n in batch

#### Mini-Batch Update
- [ ] Introduce learning rate **κ**
- [ ] **η_l^⋆** = (1 - κ)η_l + κη_l^⋆
- [ ] Stochastic natural gradient descent on variational parameters

#### Interpretation
- [ ] Hebbian function of pre- and post-synaptic activity
- [ ] Exact solution when Z^⋆ estimated for entire dataset

### Equation 8: Natural Parameters Relation (Page 3)

- [ ] **η_l** = (V_l^{-1}, M_l V_l^{-1}, Ψ_l^{-1} + M_l V_l^{-1} M_l^T, ν_l - d_{y_l} + d_{x_l} - 1)

#### Components
- [ ] **First component**: V_l^{-1}
- [ ] **Second component**: M_l V_l^{-1}
- [ ] **Third component**: Ψ_l^{-1} + M_l V_l^{-1} M_l^T
- [ ] **Fourth component**: ν_l - d_{y_l} + d_{x_l} - 1
- [ ] **d_{y_l}**: Output dimension of layer l
- [ ] **d_{x_l}**: Input dimension of layer l

### Equation 9: Deterministic Forward Pass (Page 4)

- [ ] Use expected parameter values: **⟨W_l, Σ_l⟩_{q_λ(W_l, Σ_l)}**
- [ ] Ignores uncertainty in parameters
- [ ] Used for deterministic predictions

### Equation 10: Monte Carlo Sampling (Page 4)

#### Sampling Procedure
- [ ] Sample **W_l ~ MN(W_l | M_l, Σ_l^{-1}, V_l)**
- [ ] Sample **Σ_l^{-1} ~ W(Σ_l^{-1} | Ψ_l, ν_l)**
- [ ] Average network outputs across multiple samples
- [ ] Used to estimate predictive uncertainty

### Equation 11: Analytical Uncertainty Propagation (Page 4)

#### Propagation Formula
- [ ] **q(z_l)** ∝ ⟨N(z_l | W_l z_{l-1}, Σ_l)⟩_{q(z_{l-1})q_λ(W_l, Σ_l)}

#### Approximations
- [ ] **q(z_{l-1})**: Approximated as Gaussian
- [ ] **Input layer**: q(z_0) is Dirac delta
- [ ] **Non-linearity**: Uncertainty propagated through f(·) using deterministic approximation
- [ ] Use method from [20] (Wu et al., 2018)
- [ ] Suitable for ReLU activations

---

## 3. TRAINING AND TESTING (Page 3)

### Training Modes

#### Discriminative Training
- [ ] Fix input nodes: **z_0 = x^(i)**
- [ ] Fix output nodes: **z_L = y^(i)**
- [ ] Used for supervised learning

#### Unsupervised Training
- [ ] Only fix top layer: **z_L**
- [ ] Leave other layers free

### Dataset Format
- [ ] **D** = {(x^(i), y^(i))}_{i=1}^N
- [ ] N training examples

### Testing Modes
- [ ] **Mode 1**: Deterministic forward pass (Equation 9)
- [ ] **Mode 2**: Monte Carlo sampling (Equation 10)
- [ ] **Mode 3**: Analytical uncertainty propagation (Equation 11)

---

## 4. APPENDIX A - Distribution Definitions (Page 9)

### Equation 12: Matrix Normal Distribution

#### Parameters
- [ ] **W** ∈ R^{d_y × d_x}: Random matrix
- [ ] **M** ∈ R^{d_y × d_x}: Mean matrix
- [ ] **Σ^{-1}** ∈ R^{d_y × d_y}: Row precision matrix
- [ ] **V** ∈ R^{d_x × d_x}: Column covariance matrix

#### Density Function
- [ ] **p(W | M, Σ^{-1}, V)** = exp(-1/2 Tr(Σ^{-1}(W - M)V^{-1}(W - M)^T)) / ((2π)^{d_y d_x/2} |Σ|^{d_x/2} |V|^{d_y/2})

#### Implementation Components
- [ ] Trace operation: **Tr(Σ^{-1}(W - M)V^{-1}(W - M)^T)**
- [ ] Normalization constant: **(2π)^{d_y d_x/2}**
- [ ] Determinant term 1: **|Σ|^{d_x/2}**
- [ ] Determinant term 2: **|V|^{d_y/2}**

### Equation 13: Wishart Distribution

#### Parameters
- [ ] **Σ^{-1}**: Random symmetric positive definite matrix
- [ ] **Ψ**: Scale matrix
- [ ] **ν**: Degrees of freedom

#### Density Function
- [ ] **p(Σ^{-1} | Ψ, ν)** = |Σ^{-1}|^{(ν - d_y - 1)/2} exp(-1/2 Tr(Σ^{-1}Ψ^{-1})) / (2^{νd_y/2} |Ψ|^{ν/2} Γ_{d_y}(ν/2))

#### Implementation Components
- [ ] Determinant power: **|Σ^{-1}|^{(ν - d_y - 1)/2}**
- [ ] Exponential term: **exp(-1/2 Tr(Σ^{-1}Ψ^{-1}))**
- [ ] Normalization: **2^{νd_y/2}**
- [ ] Scale determinant: **|Ψ|^{ν/2}**
- [ ] Multivariate gamma function: **Γ_{d_y}(ν/2)**

### Equation 14: Matrix Normal Wishart Distribution

#### Log Joint Density
- [ ] **log p(W, Σ | M, V, Ψ, ν)** =
  - [ ] **-1/2 Tr(Σ^{-1}(W - M)V^{-1}(W - M)^T)**
  - [ ] **+ d_x/2 log|Σ^{-1}|**
  - [ ] **- 1/2 Tr(Σ^{-1}Ψ^{-1})**
  - [ ] **+ (ν - d_y - 1)/2 log|Σ^{-1}|**
  - [ ] **- d_y d_x/2 log(2π)**
  - [ ] **+ d_y/2 log|V^{-1}|**
  - [ ] **- νd_y/2 log(2)**
  - [ ] **- log Γ_{d_y}(ν/2)**

#### Parameters
- [ ] **M**: Mean matrix
- [ ] **V**: Column covariance matrix
- [ ] **Ψ**: Scale matrix
- [ ] **ν**: Degrees of freedom

---

## 5. APPENDIX B - Derivative Formulations (Pages 9-10)

### Equation 15: Gradient w.r.t. z_l from E_l

- [ ] **∇_{z_l} E_l** = ⟨Σ_l^{-1}(z_l - W_l f(z_{l-1}))⟩_{q_λ(W_l, Σ_l)}

#### Implementation
- [ ] Compute prediction error: z_l - W_l f(z_{l-1})
- [ ] Multiply by precision: Σ_l^{-1}
- [ ] Take expectation under q_λ(W_l, Σ_l)

### Equation 16: Gradient w.r.t. z_l from E_{l+1}

- [ ] **∇_{z_l} E_{l+1}** = -D(z_l) ⟨W_{l+1}^T Σ_{l+1}^{-1}(z_{l+1} - W_{l+1}f(z_l))⟩_{q_λ(W_{l+1}, Σ_{l+1})}

#### Components
- [ ] **D(z)** = diag(f'(z)): Diagonal Jacobian of pointwise non-linear transfer function
- [ ] Negative sign in front
- [ ] Backpropagated prediction error from layer l+1

### Equation 17: Expectation of Σ_l^{-1} W_l

- [ ] **⟨Σ_l^{-1} W_l⟩_{q_λ(W_l, Σ_l)}** = ν_l Ψ_l M_l

#### Implementation
- [ ] Multiply degrees of freedom ν_l
- [ ] Multiply scale matrix Ψ_l
- [ ] Multiply mean matrix M_l
- [ ] Result: ν_l Ψ_l M_l

### Equation 18: Expectation of W_l^T Σ_l^{-1} W_l

- [ ] **⟨W_l^T Σ_l^{-1} W_l⟩_{q_λ(W_l, Σ_l)}** = M_l ν_l Ψ_l M_l + d_l V_l

#### Components
- [ ] **First term**: M_l ν_l Ψ_l M_l (quadratic in mean)
- [ ] **Second term**: d_l V_l (scaled covariance)
- [ ] **d_l**: Dimension of z_l

### Equations 19-20: Alternative Form of ∇_{z_l} E_{l+1}

#### Equation 19
- [ ] **∇_{z_l} E_{l+1}** = -D(z_l) M_{l+1}^T ⟨Σ_{l+1}^{-1}⟩ (z_{l+1} - M_{l+1}f(z_l))

#### Equation 20
- [ ] **+ d_{l+1} D(z_l) V_{l+1} f(z_l)**

#### Combined Expression
- [ ] Combines mean-based term (Eq. 19)
- [ ] Plus variance correction term (Eq. 20)
- [ ] d_{l+1} is dimension of z_{l+1}

### Optimal Learning Rate (Appendix B)

- [ ] **Upper bound on α**: Inverse of maximum eigenvalue of A_l
- [ ] **A_l** = Σ_l^{-1} + W_{l+1}^T Σ_{l+1}^{-1} W_{l+1}
- [ ] Can be dynamically updated with posterior distribution updates
- [ ] Note: Derivative of f'(z) bounded by 1 for ReLU
- [ ] Block triangular structure of dynamics
- [ ] Spectrum dominated by A_l

---

## 6. APPENDIX C - Predictive Coding Comparison (Pages 10-11)

### Equation 21: PC Generative Model

- [ ] **p(Z | Θ)** = p(z_0) ∏_{l=1}^L p(z_l | z_{l-1}, W_l, Σ_l)
- [ ] **p(z_l | z_{l-1}, W_l, Σ_l)** = N(z_l | W_l f(z_{l-1}), Σ_l)
- [ ] **KEY DIFFERENCE**: No priors over parameters Θ (unlike BPC Eq. 1)

### Equation 22: PC Energy Function

- [ ] **E(Z, Θ)** = -log(p(z_0) ∏_{l=1}^L p(z_l | z_{l-1}, W_l, Σ_l))
- [ ] **E(Z, Θ)** = 1/2 Σ_{l=1}^L (z_l - W_l f(z_{l-1})) · Σ_l^{-1} (z_l - W_l f(z_{l-1})) + C
- [ ] **E_l**: Precision-weighted prediction error
- [ ] **KEY DIFFERENCE**: No expectation under q_λ(Θ) (unlike BPC Eq. 5)
- [ ] **KEY DIFFERENCE**: Function of Θ, not λ

### Equation 23: PC EM Algorithm

- [ ] **Z^*** = arg min_Z E(Z, Θ)
- [ ] **Θ^*** = arg min_Θ E(Z^*, Θ)
- [ ] **KEY DIFFERENCE**: Optimizes Θ directly, not λ

### Equation 24: PC Gradients

- [ ] **∇_{z_l}** = ∂E/∂z_l = 1/2(∂E_l/∂z_l + ∂E_{l+1}/∂z_l)
- [ ] **∇_{θ_l}** = ∂E/∂θ_l = 1/2 ∂E_l/∂θ_l
- [ ] Gradient descent for both Z and Θ

### Equation 25: PC Update Rule

- [ ] **z_l → z_l - α(Σ_l^{-1} z_l - D(z_l) W_{l+1}^T Σ_{l+1}^{-1}(z_{l+1} - W_{l+1}f(z_l)))**
- [ ] Direct update without expectation

### Key Differences BPC vs PC
- [ ] BPC has priors p(W_l, Σ_l); PC does not
- [ ] BPC optimizes variational parameters λ; PC optimizes parameters Θ directly
- [ ] BPC has expectation under q_λ(Θ); PC does not
- [ ] BPC uses closed-form updates for parameters; PC uses gradient descent
- [ ] BPC represents posterior distributions; PC uses point estimates (MAP/ML)

---

## 7. APPENDIX D - Conjugate Update Proof (Page 11)

### Equation 26: Layer-wise Energy

- [ ] **E(z_l, λ_l, z_{l-1})** = ⟨log q_λ(W_l, Σ_l) - log p(z_l, W_l, Σ_l | z_{l-1})⟩_{q_λ(W_l, Σ_l)}

### Equation 27: Natural Parameters Equivalence

- [ ] **λ_l ≡ η_l**
- [ ] **η_l** = (V_l^{-1}, M_l V_l^{-1}, Φ_l + M_l V_l^{-1} M_l^T, ν_l - d_y + d_x - 1)
- [ ] Note: Uses Φ_l instead of Ψ_l^{-1} in this formulation

### Equation 28: Optimal Natural Parameters

- [ ] **η_l^*** = (
  - [ ] V_{l,0}^{-1} + f(z_{l-1})f(z_{l-1})^T,
  - [ ] M_{l,0} V_{l,0}^{-1} + f(z_{l-1})z_l^T,
  - [ ] Φ_{l,0} + M_{l,0} V_{l,0}^{-1} M_{l,0}^T + z_l z_l^T,
  - [ ] ν_{l,0} - d_y + d_x - 1 + 1
  - [ ] )

#### Prior Parameters
- [ ] **X_{l,0}** notation denotes prior parameters (X = V, M, Φ, ν)
- [ ] Prior contributes: V_{l,0}^{-1}, M_{l,0} V_{l,0}^{-1}, Φ_{l,0} + M_{l,0} V_{l,0}^{-1} M_{l,0}^T, ν_{l,0}

#### Sufficient Statistics
- [ ] **f(z_{l-1})f(z_{l-1})^T**: Pre-synaptic activity
- [ ] **f(z_{l-1})z_l^T**: Cross-activity
- [ ] **z_l z_l^T**: Post-synaptic activity
- [ ] **1**: Count (added to degrees of freedom)

#### Properties
- [ ] **Exact Bayesian update**: Due to full conjugacy
- [ ] **Optimal variational posterior = true posterior**

---

## 8. APPENDIX F - Experiment Details (Pages 12-13)

### F.1 Accuracy Experiments

#### General Settings
- [ ] Performance averaged over **5 random seeds**
- [ ] **Regression**: Mean Squared Error (MSE)
- [ ] **Classification**: One-hot encoded labels during training
- [ ] **Test time**: argmax over output nodes

#### Architecture: Energy and MNIST

##### Network Structure
- [ ] **4-layer neural network**
- [ ] **128 hidden units per layer**
- [ ] **ReLU activations**

##### Training
- [ ] **Mini-batch size**: 128

##### Weight Initialization
- [ ] **W**: Shape (out_features, in_features)
- [ ] Sample from **U(-√k, √k)** where k = 1/in_features
- [ ] **b**: Shape (out_features)
- [ ] Sample from **U(-√k, √k)** where k = 1/in_features

#### BP Hyperparameters
- [ ] **Optimizer**: Adam
- [ ] **Learning rate**: 0.001

#### PC Hyperparameters

##### Parameter Updates
- [ ] **Optimizer**: AdamW
- [ ] **Learning rate**: 0.0002
- [ ] **Weight decay**: 0.65

##### Hidden State Updates
- [ ] **Optimizer**: SGD
- [ ] **Learning rate**: 0.01
- [ ] **Momentum**: 0.65

##### Iteration Schedule
- [ ] **Hidden state iterations**: 10 per batch
- [ ] **Weight gradient steps**: 1 per batch

##### Source
- [ ] Parameters follow [34] (Pinchetti et al., 2024)

#### BPC Hyperparameters

##### Hidden State Updates
- [ ] **Optimizer**: Adam
- [ ] **Learning rate**: 0.01
- [ ] **Iterations per batch**: 10

##### Parameter Learning Rate
- [ ] **Schedule**: κ_t = t^{-ϵ}
- [ ] **t**: Total number of updates
- [ ] **ϵ**: 0.25
- [ ] Formula: **κ_t = t^{-0.25}**

##### Prior Parameters
- [ ] **M^(0)**: Matrix of zeros (appropriate size)
- [ ] **V^(0)**: 10 · I (I = identity matrix, appropriate size)
- [ ] **Ψ^(0)**: 1000 · I
- [ ] **ν^(0)**: d_y + 2

##### Posterior Initialization
- [ ] All natural parameters **η** initialized to prior values
- [ ] **Exception**: M uses same initialization as W (uniform distribution)

#### Architecture: Two Moons

##### Network Structure
- [ ] **Single hidden layer**
- [ ] **100 hidden units**

##### Other Settings
- [ ] Same hyperparameters as Energy/MNIST

### F.2 Synthetic Regression

#### General Settings
- [ ] Same parameter settings as Two Moons

#### Aleatoric Uncertainty Task

##### Data Generation
- [ ] **y = -(x + 0.5) · sin(3πx) + N(0, (0.45 · (x + 0.5))^2)**
- [ ] **x**: Normally distributed around zero
- [ ] Heteroscedastic noise: variance depends on x

##### Implementation
- [ ] Parameterize output layer with variance node
- [ ] Follow method from [20] (Wu et al., 2018)
- [ ] Propagate uncertainty through network
- [ ] Estimate first and second-order moments of output

#### Epistemic Uncertainty Task

##### Data Generation
- [ ] **y = x^3 + N(0, 9)**
- [ ] **x**: Uniformly sampled
  - [ ] Half from interval [3, 5]
  - [ ] Half from interval [-5, -3]
- [ ] Homoscedastic noise: variance = 9

##### Implementation
- [ ] Draw multiple samples from parameter posterior q_λ(Θ)
- [ ] Visualize predicted functions
- [ ] Show spread of predictions

### F.3 UCI Datasets

#### Architecture
- [ ] **2 hidden layers**
- [ ] **50 hidden nodes per layer**
- [ ] Other parameters same as previous experiments

#### BPC Settings
- [ ] Same as described in F.1

#### BBB (Bayes by Backprop) Hyperparameters
- [ ] **Optimizer**: Adam
- [ ] **Learning rate**: 0.001
- [ ] **Prior mean**: 0.0
- [ ] **σ (sigma)**: 1.0
- [ ] **Batch size**: 100

#### Log Predictive Density (LPD) Computation
- [ ] **Both BPC and BBB**: Draw 20 posterior samples per data batch
- [ ] Average log-likelihood across samples

#### Datasets Tested
- [ ] yacht
- [ ] concrete
- [ ] wine
- [ ] housing
- [ ] power
- [ ] energy

#### Metrics
- [ ] **LPD**: Log Predictive Density
- [ ] **RMSE**: Root Mean Squared Error

---

## 9. ADDITIONAL IMPLEMENTATION DETAILS

### Matrix Operations Required

#### Basic Operations
- [ ] Matrix multiplication
- [ ] Matrix transpose
- [ ] Matrix inverse
- [ ] Outer product
- [ ] Trace operation
- [ ] Determinant

#### Advanced Operations
- [ ] Eigenvalue computation (for optimal learning rate)
- [ ] Matrix square root (for sampling)
- [ ] Cholesky decomposition (for efficient sampling)
- [ ] Kronecker product (implicit in Matrix Normal)

### Distribution Sampling

#### Matrix Normal Sampling
- [ ] Input: M (mean), Σ^{-1} (row precision), V (column covariance)
- [ ] Generate standard normal matrix
- [ ] Apply Cholesky factors of Σ and V
- [ ] Add mean M

#### Wishart Sampling
- [ ] Input: Ψ (scale matrix), ν (degrees of freedom)
- [ ] Generate ν samples from multivariate normal
- [ ] Form sample covariance matrix
- [ ] Scale by Ψ

### Expectation Computations

#### Under Matrix Normal Wishart
- [ ] ⟨W_l⟩ = M_l
- [ ] ⟨Σ_l^{-1}⟩ = ν_l Ψ_l
- [ ] ⟨Σ_l^{-1} W_l⟩ = ν_l Ψ_l M_l (Eq. 17)
- [ ] ⟨W_l^T Σ_l^{-1} W_l⟩ = M_l ν_l Ψ_l M_l + d_l V_l (Eq. 18)

### Convergence Criteria

#### Latent Variable Convergence
- [ ] Check change in z_l between iterations
- [ ] Use tolerance threshold (not specified in paper)
- [ ] OR reach maximum T iterations

#### Training Convergence
- [ ] Monitor loss/accuracy on validation set
- [ ] Early stopping (not specified in paper)
- [ ] Fixed number of epochs

### Activation Functions

#### Supported
- [ ] **ReLU**: f(x) = max(0, x)
- [ ] f'(x) = 1 if x > 0, else 0
- [ ] Bounded derivative (≤ 1)

#### Requirements
- [ ] Point-wise operation
- [ ] Differentiable (almost everywhere)
- [ ] Compatible with deterministic approximation [20]

### Memory Requirements

#### Per Layer
- [ ] **z_l**: Latent variables (batch_size × d_l)
- [ ] **M_l**: Mean weights (d_{l+1} × d_l)
- [ ] **V_l**: Column covariance (d_l × d_l)
- [ ] **Ψ_l**: Scale matrix (d_{l+1} × d_{l+1})
- [ ] **ν_l**: Degrees of freedom (scalar)
- [ ] **Gradients**: ∂E/∂z_l (batch_size × d_l)

#### Temporary Storage
- [ ] Prediction errors
- [ ] Precision-weighted errors
- [ ] Jacobians D(z_l)
- [ ] Sufficient statistics for updates

### Computational Complexity

#### Per Inference Iteration
- [ ] Forward pass through network: O(L · batch_size · d^2)
- [ ] Gradient computation: O(L · batch_size · d^2)
- [ ] Update z: O(L · batch_size · d)

#### Per Parameter Update
- [ ] Compute sufficient statistics: O(L · batch_size · d^2)
- [ ] Update natural parameters: O(L · d^2)
- [ ] Convert natural to standard parameters: O(L · d^3) (matrix inversions)

---

## 10. CRITICAL IMPLEMENTATION NOTES

### Must-Have Features
- [ ] **W_l outside f(·)**: Essential for closed-form updates
- [ ] **Conjugate priors**: Matrix Normal Wishart required
- [ ] **Natural parameters**: Use η for efficient updates
- [ ] **Hebbian updates**: Pre/post-synaptic activity products

### Common Pitfalls
- [ ] Forgetting 1/2 factors in energy and gradients
- [ ] Placing W_l inside f(·) breaks conjugacy
- [ ] Not using expectations for BPC gradients
- [ ] Mixing PC and BPC update rules
- [ ] Incorrect natural parameter conversions

### Numerical Stability
- [ ] Check positive definiteness of Σ_l, V_l, Ψ_l
- [ ] Regularize matrix inversions (add small diagonal)
- [ ] Use log-space for determinants
- [ ] Clip gradients if necessary
- [ ] Monitor condition numbers

### Validation Checks
- [ ] Verify energy decreases during inference
- [ ] Check parameter updates are well-formed
- [ ] Validate distribution parameters (ν > d_y + 1)
- [ ] Test on synthetic data first
- [ ] Compare BPC to PC on same task

---

## 11. OPTIONAL EXTENSIONS (Mentioned in Paper)

### Future Directions
- [ ] Low-rank approximations for V_l, Ψ_l
- [ ] Structured approximations for large networks
- [ ] Pre-training with BP, then BPC for uncertainty
- [ ] Identity-like priors for Σ to encourage disentanglement
- [ ] Monte Carlo sampling for latent variables Z
- [ ] Langevin dynamics for Z (combine with [31, 32])

### Not Required for Basic Implementation
- [ ] Variational Laplace comparison
- [ ] Sparse priors
- [ ] Auto-associative memory
- [ ] Continual learning
- [ ] Network pruning

---

## CHECKLIST SUMMARY

### Total Equations to Implement
- [x] **Main paper**: Equations 1-11 (11 equations)
- [x] **Appendix A**: Equations 12-14 (3 equations)
- [x] **Appendix B**: Equations 15-20 (6 equations)
- [x] **Appendix C**: Equations 21-25 (5 equations, for comparison)
- [x] **Appendix D**: Equations 26-28 (3 equations, proof)
- [x] **Total**: 28 equations

### Total Algorithm Steps
- [x] **Algorithm 1**: 12 steps

### Total Hyperparameters (BPC)
- [x] α (inference learning rate)
- [x] T (max inference iterations)
- [x] κ or κ_t (parameter learning rate schedule)
- [x] ϵ (learning rate decay exponent)
- [x] M^(0) (prior mean)
- [x] V^(0) (prior column covariance)
- [x] Ψ^(0) (prior scale)
- [x] ν^(0) (prior degrees of freedom)
- [x] Network architecture (L, d_l)
- [x] Activation function (e.g., ReLU)
- [x] Batch size
- [x] Optimizer for hidden states (e.g., Adam)

### Experiment-Specific Hyperparameters
- [x] **Weight initialization**: U(-√k, √k)
- [x] **Energy/MNIST**: 4 layers, 128 units, batch 128
- [x] **Two Moons**: 1 layer, 100 units
- [x] **UCI**: 2 layers, 50 units, batch 100
- [x] **Number of seeds**: 5
- [x] **MC samples for LPD**: 20

---

## VERIFICATION PROCEDURE

For each implementation:
1. [ ] Verify all 28 equations are correctly coded
2. [ ] Verify Algorithm 1 steps 1-12 are implemented
3. [ ] Test each distribution (MN, W, MNW) sampling and density
4. [ ] Test gradient computations against finite differences
5. [ ] Verify natural parameter conversions (Eq. 8, 27, 28)
6. [ ] Test on synthetic data with known solution
7. [ ] Reproduce Two Moons results
8. [ ] Reproduce Energy results
9. [ ] Reproduce MNIST results
10. [ ] Compare uncertainty estimates on synthetic regression
11. [ ] Verify all hyperparameters from Appendix F are used
12. [ ] Check numerical stability on all tests

---

**END OF COMPLETE CHECKLIST**

*This checklist contains every equation, every algorithm step, every hyperparameter, and every implementation detail from the Bayesian Predictive Coding paper. No assumptions. No omissions.*
