# Mistakes File

## Purpose & Usage Instructions

This file catalogues all mistakes made during development - both code bugs and architectural/theoretical errors. **ALWAYS** consult this file before:
- Writing new code
- Designing architectures
- Solving problems
- Making design decisions

**When adding mistakes:**
- Check for duplicates first
- Include: what was wrong, why it failed, what the correct approach is
- Link to specific commits/files if applicable
- Update attempted fixes as we learn more

---

## Architectural & Theoretical Mistakes

### 1. Convolutional Neural Networks with Predictive Coding
**What we tried:** Building a CNN trained with predictive coding
**Why it failed:** Repeatedly failed implementation despite multiple correction attempts. CNN architectures may have fundamental incompatibilities with our PC implementation approach.
**Correct approach:** Use standard predictive coding network architectures as documented in literature, not custom CNN hybrids. Reference VERSES (Karl Friston's company) for scaling solutions.
**Status:** Abandoned - use standard PC approaches

### 2. Custom Two-Compartment Neuron Design
**What we tried:** Custom neuron architecture with separate error and representation compartments
**Why it failed:** Overcomplicated and non-standard. Not aligned with proven predictive coding implementations.
**Correct approach:** Use standard PC architecture from Bogacz Group - PCLayer holds value nodes (_x) as nn.Parameters, computes energy E = 0.5*(mu - x)^2, returns x during training
**Status:** FIXED - implemented standard PCLayer in src/network/pc_layer.py (2026-02-01)

### 3. Output Clamping for Pretraining
**What we tried:** Clamping network outputs to target labels during pretraining (e.g., forcing output to "5" for MNIST digit 5)
**Why it failed:** Forces convergence without building the pathways that would allow the network to converge on the right answer naturally. Prevents proper feature learning and representation building.
**Correct approach:** Allow network to build proper internal representations through prediction error minimization. Don't force outputs - let the network learn the mappings.
**Status:** Critical - never use output clamping

### 4. Error Signal Propagation with Exponential Precision Scaling
**What we tried:** Increasing precision by 10x at each layer to combat error signal fade in deep networks (5-7+ layers)
**Why it failed:** Fundamentally wrong approach. Creates numerical instability and doesn't address root cause of vanishing gradients in predictive coding.
**Correct approach:** Research VERSES solutions for scaling predictive coding beyond 5-7 layers. Likely involves architectural changes, not parameter hacks.
**Status:** Never attempt exponential precision scaling again

---

## Process & Workflow Mistakes

### 5. Not Learning from Repeated Mistakes
**What happened:** Made the same errors multiple times without maintaining a record
**Why it failed:** No systematic way to track and prevent repeat errors
**Solution:** This mistakes file. Always consult before making decisions.
**Status:** Implemented (this file)

### 6. Context Compression Loss
**What happened:** Lost critical research context mid-task due to conversation compression
**Why it failed:** No long-term memory system for preserving important design decisions, constraints, and research findings
**Solution:** Created persistent documentation:
  - MISTAKES.md (this file)
  - NETWORK_PROPOSAL.md (architectural decisions)
  - RESEARCH_NOTES.md (literature findings)
  - RUNNING_MNIST.md (usage instructions)
**Status:** IMPLEMENTED - documentation system in place (2026-02-01)

---

## Code Implementation Mistakes

### 7. Import Errors in Non-Root Files (RECURRING ISSUE)

**Problem:** `ModuleNotFoundError: No module named 'src'` when running files in subdirectories

**Why it happens:**
- Python imports are relative to the script's location
- Files in `tests/` or `experiments/` can't find `src/` module
- Happens every time we create files outside the root directory

**Solution:** Add path manipulation at the top of every non-root script
```python
import sys
import os
# Add parent directory (or root) to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now imports work
from src.network import PCNetwork
```

**Fixed in:**
- `tests/test_pc_basic.py`
- `experiments/BayesianPC/train_mnist_bayesian.py`

**Status:** RECURRING - must remember for every new subdirectory file

---

### 8. Optimizer Conflating Value Nodes and Network Parameters (CRITICAL BUG)

**What we did wrong:** Created weight optimizer with `model.parameters()` which includes BOTH network weights AND value nodes
```python
# WRONG - includes value nodes!
self.optimizer_p = optimizer_p_fn(self.model.parameters(), lr=weight_lr)
```

**Why it failed catastrophically:**
- Value nodes (`_x`) were optimized by TWO conflicting optimizers:
  - `optimizer_x` during inference (correct)
  - `optimizer_p` during learning (WRONG)
- Contradictory gradient updates destroyed the energy landscape
- Network could not learn - stuck at random guessing (8% accuracy on MNIST)
- Loss and accuracy completely flat across all epochs

**How we found it:**
- User reported training stuck at 8.35% accuracy (random baseline for 10 classes)
- Zero improvement over 5 epochs
- Train and test metrics identical and unchanging
- Traced through optimizer creation in pc_trainer.py line 55-57

**Correct approach:**
```python
# Separate parameters: exclude value nodes from weight optimizer
self.optimizer_p = optimizer_p_fn(
    self.model.get_network_parameters(),  # Only Linear weights/biases
    lr=weight_lr
)
```

**Implementation:**
- Added `get_network_parameters()` method to PCNetwork
- Filters out value nodes from parameter list
- Only yields Linear layer weights and biases
- Weight optimizer now correctly optimizes ONLY network parameters

**Root principle violated:** In PC, value nodes and weights are optimized in SEPARATE phases, never simultaneously. Mixing them breaks the algorithm.

**Files fixed:**
- `src/network/pc_layer.py`: Added `get_network_parameters()` method
- `src/network/pc_trainer.py`: Changed optimizer_p to use filtered parameters
- `tests/test_pc_basic.py`: Fixed gradient test to check network parameters only

**Status:** Attempted fix had no effect - see mistake #9

---

### 9. Detaching Computational Graph (THE ACTUAL BUG - CRITICAL)

**What we did wrong:** Used `mu.detach()` when initializing value nodes and computing energy
```python
# WRONG - breaks gradient flow!
self._x = nn.Parameter(mu.detach().clone(), requires_grad=True)
error = mu.detach() - self._x
```

**Why it completely prevented learning:**
- Value nodes (x) were detached from predictions (mu)
- mu is detached from network weights
- Computational graph: input → weights → mu [BREAK] x → loss
- NO gradients reach network weights from ANY source:
  - Loss gradients: loss → x (but x detached from mu) → STOP
  - Energy gradients: energy → x and mu (but mu detached) → STOP
- Weights receive ZERO gradients
- Cannot learn at all

**Symptoms:**
- Debug showed: "Weights changed: False, Max weight change: 0.000000"
- Inference works (free energy decreases)
- But weights never update
- Accuracy stays at random guessing forever

**Root cause:** Misunderstanding of how PC separation works
- I thought: detach graph to separate inference/learning phases
- Actually: optimizer separation handles which params update when
- Graph must stay connected for gradients to flow

**Correct approach:**
```python
# Keep in computational graph!
self._x = nn.Parameter(mu.clone(), requires_grad=True)  # No detach
error = mu - self._x  # No detach
```

**Why this works:**
- Gradients computed for ALL parameters (x and weights)
- optimizer_x.step() only updates x (its param group)
- optimizer_p.step() only updates weights (its param group)
- Each optimizer updates its own params, graph stays connected

**Files fixed:**
- `src/network/pc_layer.py`: Removed mu.detach() calls (lines 76, 81)

**Status:** FIXED (2026-02-01) - the REAL fix this time

---

### 10. Bayesian PC Variance Collapse and NaN Catastrophe (CRITICAL)

**What we did wrong:** Set min_variance too low (1e-6) causing numerical instability
```python
# WRONG - variance collapses to 1e-6, precision explodes to 1e6
min_variance = 1e-6
precision = 1.0 / (x_var + min_variance)  # → 1e6 → NaN gradients
```

**Symptoms:**
- loss=nan, acc=9.38% (random guessing)
- All diagnostics show NaN
- Variance collapses to minimum, precision explodes
- Network cannot learn at all

**Why it failed:**
- Variance clamped at [1e-6, ∞) collapses to 1e-6
- Precision = 1/variance → 1e6 (massive number)
- Precision * error^2 → overflow → NaN
- Gradients become NaN, weights get NaN, everything breaks
- No upper bound on variance means no protection from collapse

**Correct approach:**
```python
# Proper numerical bounds
min_variance = 0.01  # Not 1e-6!
max_variance = 10.0  # Prevents precision collapse
x_var = torch.exp(x_log_var).clamp(min=min_variance, max=max_variance)
precision = 1.0 / x_var  # Safe, bounded to [0.1, 100]
```

**Files fixed:**
- `experiments/BayesianPC/bayesian_pc_layer.py`: Changed min_variance to 0.01, added max_variance=10.0
- Updated get_x_variance() to clamp with both min and max
- Removed redundant min_variance additions in energy computation

**Status:** FIXED (2026-02-01) - proper variance bounds prevent NaN

---

### 11. Breaking Experimental Control Variables

**What we did wrong:** Changed architecture from 7 to 3 layers without justification
```python
# WRONG - can't compare to baseline!
layer_sizes = [784, 256, 256, 10]  # 3 layers
T_inference = 20  # 5 * 3 layers

# Baseline was:
# layer_sizes = [784, 256, 256, 256, 256, 256, 128, 10]  # 7 layers
# T_inference = 35  # 5 * 7 layers
```

**Why it's wrong:**
- Experiment compares Bayesian PC to baseline PC
- Changed network architecture between treatments
- Can't isolate effect of Bayesian inference from effect of network size
- Violates basic experimental design: control all variables except the one being tested

**User feedback:** "why are there only 3 layers when the vanilla MNIST experiment had 7? You clearly have not managed your control variables well."

**Correct approach:**
- Keep ALL hyperparameters identical except the treatment variable
- Treatment variable: Bayesian inference (mean+variance) vs point estimates
- Control variables: architecture, T, learning rates, batch size, etc.

**Files fixed:**
- `experiments/BayesianPC/train_mnist_bayesian.py`: Restored 7-layer architecture and T=35

**Root principle violated:** Scientific experiments require controlled variables. Can't test effect of X while simultaneously changing Y.

**Status:** FIXED (2026-02-01) - experimental control restored

---

### 12. Fundamental Bayesian PC Conceptual Error - Wrong Thing Made Bayesian (CRITICAL)

**What we did wrong:** Put posterior distributions over **value nodes** (hidden states) instead of **weights** (parameters)

```python
# WRONG IMPLEMENTATION (experiments/BayesianPC/ - INCORRECT)
class BayesianPCLayer:
    def __init__(self):
        # Made value nodes Bayesian (mean + log_variance)
        self._x_mean = nn.Parameter(...)
        self._x_log_var = nn.Parameter(...)
        # Weights stayed as point estimates
        self.linear = nn.Linear(...)

    def energy(self):
        # KL divergence on value nodes
        kl = KL[q(x) || p(x)]  # WRONG!
        return accuracy + kl
```

**Why it's completely wrong:**
1. **Value nodes are ephemeral** - optimized fresh every forward pass for current input
   - They represent posterior beliefs about the CURRENT observation
   - Making them Bayesian doesn't capture epistemic uncertainty
   - They reset every batch - no knowledge accumulation

2. **Weights are what accumulate knowledge** - updated across all training data
   - Epistemic uncertainty is "what do we know about the true parameters?"
   - Weight posteriors capture uncertainty that decreases with more data
   - This is what Bayesian deep learning means

3. **Breaks conjugacy** - my approach had no closed-form updates
   - Used gradient descent on variance parameters (slow, unstable)
   - Real BPC uses Matrix Normal Wishart conjugate priors
   - Enables closed-form Hebbian weight updates (Equation 7 in paper)

4. **Architecture was wrong** - I had weights INSIDE activation
   - My implementation: `mu = f(linear(x))` → weights inside f()
   - Required for conjugacy: `mu = linear(f(x))` → weights outside f()
   - This architectural constraint is essential for closed-form updates

**What BPC actually does (from Tschantz et al. 2025, arXiv:2503.24016):**

```python
# CORRECT IMPLEMENTATION (from Algorithm 1)
class BayesianPCLayer:
    def __init__(self):
        # Value nodes as MAP estimates (point values, optimized during inference)
        self._x = None  # Created during forward, scalar per node

        # Weights as Matrix Normal Wishart posterior
        self.weight_posterior = {
            'M': ...,    # Mean matrix
            'V': ...,    # Column covariance
            'Ψ': ...,    # Scale matrix (Wishart)
            'ν': ...,    # Degrees of freedom
        }

    def inference_step(self, mu):
        # E-step: Optimize value nodes (same as standard PC)
        error = <Σ^(-1)(z - Wf(z_{l-1}))>_q(W,Σ)
        self._x -= α * error  # Gradient descent on x

    def learning_step(self, z_star):
        # M-step: Closed-form Bayesian update on weight posterior
        # Equation 7: Hebbian function of pre/post-synaptic activity
        η* = η_prior + Σ_n [f(z*_{l-1})f(z*_{l-1})^T,
                             f(z*_{l-1})z*_l^T,
                             z*_l z*_l^T, 1]
        # Update natural parameters (closed form!)
```

**Key differences:**
| Aspect | My Wrong Implementation | Correct BPC |
|--------|------------------------|-------------|
| **Bayesian treatment** | Value nodes (hidden states) | Weights (parameters) |
| **Value nodes** | Distributions N(μ, σ²) | MAP estimates (scalars) |
| **Weights** | Point estimates | Matrix Normal Wishart posterior |
| **Learning** | Gradient descent on variance | Closed-form Bayesian update (Eq 7) |
| **Weight updates** | Backprop-style | Hebbian (pre/post activity) |
| **Architecture** | Weights inside f() | Weights outside f() (conjugacy) |
| **Convergence** | Slow (gradient descent) | Fast (closed-form) |
| **Uncertainty** | Wrong (on ephemeral states) | Correct (on parameters) |

**What the paper says explicitly:**
- Page 1: "estimates a posterior distribution over **network parameters**" (not hidden states!)
- Page 2: "latent variables Z are represented via maximum a posteriori (MAP) estimates" (point values!)
- Page 2: "parameters Θ are represented...posterior distribution" (this is what's Bayesian!)
- Page 3, Equation 7: "Hebbian function of pre- and post-synaptic activity" (closed-form weight update)
- Page 2: "placed the parameters W_l **outside** of the non-linear activation function f(·), which is essential for enabling the closed-form updates"

**User feedback:**
- "I don't entirely buy the explanation that what we have is already close but I'll humor the idea"
- Provided paper: "Bayesian Predictive Coding" (Tschantz et al., 2025)
- After NaN results: "There are still some glaring problems in your code. In any case, I don't think that your assumptions going into the code were correct."

**Correct approach:**
1. Keep value nodes as MAP estimates (point values, optimized via gradient descent)
2. Represent weights as Matrix Normal Wishart distributions q(W_l, Σ_l | M_l, V_l, Ψ_l, ν_l)
3. Use conjugate priors for closed-form updates
4. Move weights outside activation: z_l = W_l · f(z_{l-1})
5. Implement Equation 7 for Hebbian weight updates
6. Natural parameters accumulate sufficient statistics across batches

**Files to archive (INCORRECT IMPLEMENTATION):**
- `experiments/BayesianPC/bayesian_pc_layer.py` - wrong Bayesian treatment
- `experiments/BayesianPC/bayesian_pc_trainer.py` - gradient descent on wrong things
- `experiments/BayesianPC/train_mnist_bayesian.py` - uses wrong implementation
- All fixes to mistakes #10 and #11 were fixing bugs in fundamentally wrong code

**Root cause:** Misunderstood what "Bayesian" means in "Bayesian Predictive Coding"
- I assumed: make the inference Bayesian (distributions over hidden states)
- Actually: make the learning Bayesian (distributions over weights)
- This is the difference between Bayesian inference and Bayesian learning

**Status:** IDENTIFIED (2026-02-01) - current BayesianPC implementation is architecturally wrong and must be completely rewritten

---

### 13. Skimming Research Papers Instead of Reading Thoroughly (CRITICAL PROCESS ERROR)

**What I did wrong:** Made lazy speculations about implementation details instead of reading the paper carefully

**The situation:**
- Diagnostic showed energy = 640,000 (should be ~10), precision = 258,000x baseline
- Inference was diverging (ΔF = -12)
- I speculated: "Energy normalization: Paper may normalize by precision somehow"
- User called me out: "Isn't this speech strongly implying you lazily skimmed the paper, if you can't even answer a question like that?"

**What the paper ACTUALLY says (Appendix B, page 10-11):**
- "the dynamics dominated by the spectrum of **A_l = Σ_l^{-1} + W_{l+1}^T Σ_{l+1}^{-1} W_{l+1}**"
- "upper bound on the maximum learning rate parameter as **approximately given by the inverse of the maximum eigenvalue of the A_l**"
- "can be dynamically updated with updates to the posterior distribution over the parameters"

**The answer was RIGHT THERE in Appendix B:**
- NO energy normalization
- YES adaptive learning rate selection: α ≈ 1 / λ_max(A_l)
- With E[Σ^{-1}] = 258,000 × I, the optimal α ≈ 1/258,000 ≈ 3.87e-6
- My code used fixed α = 0.01 (2,580x too large!)

**Why this is inexcusable:**
1. The paper explicitly addresses this in the appendix
2. I claimed ignorance: "Why don't you know this?" - because I didn't read it
3. Made unfounded speculations instead of checking the source
4. Wasted time on wrong hypotheses (energy normalization, architecture differences)
5. User had to point me to the exact section I should have read

**The actual problem:**
```python
# What I had (WRONG):
def inference_step(self, mu, lr=0.01):  # Fixed LR regardless of precision!
    error = self.compute_error(mu)
    self._x -= lr * error

# What the paper says (CORRECT):
def inference_step(self, mu):
    # Adaptive LR based on precision spectrum
    A_l = Sigma_inv + W_next^T @ Sigma_inv_next @ W_next
    alpha_optimal = 1.0 / max_eigenvalue(A_l)
    error = self.compute_error(mu)
    self._x -= alpha_optimal * error
```

**Root principle violated:**
When implementing a research paper:
1. **READ THE ENTIRE PAPER** - especially appendices with implementation details
2. **NEVER speculate** when the answer is in the paper
3. **CHECK appendices** for implementation details (Appendix B had the exact formula)
4. **Don't be lazy** - "may normalize somehow" is intellectual laziness
5. **Trust the authors** - they solved these problems, the solution is documented

**User's exact feedback:**
- "Stop. Skimming. The. Paper."
- "Stop hook feedback: It was unbelievably arrogant and stupid of you to think you were smarter than the paper's authors"
- Referenced MISTAKES.md: "not parameter hacks" (like changing Ψ from 1000 to 1)

**Correct approach when stuck:**
1. Reread the paper thoroughly, especially methods and appendices
2. Look for implementation details, hyperparameter selection, optimization procedures
3. Check if authors provide code or supplementary materials
4. Only speculate AFTER confirming the answer isn't in the paper

**The lesson:**
**NEVER, EVER SKIM A RESEARCH PAPER YOU ARE IMPLEMENTING.**
- Read the full paper
- Read all appendices
- Read supplementary materials
- Then implement
- No speculation before thorough reading

**Files that need fixing:**
- `experiments/BayesianPC/bayesian_pc_layer.py` - Add adaptive learning rate
- `experiments/BayesianPC/bayesian_pc_trainer.py` - Compute optimal α per layer

**Status:** IDENTIFIED (2026-02-01) - must implement adaptive learning rate from Appendix B

---

#### SECOND INSTANCE: Misreading Appendix B (Theoretical) vs Appendix F (Practical Implementation)

**What I did wrong (again!):** Implemented the "adaptive learning rate" from Appendix B as α ≈ 1/λ_max(A_l) = 3.88e-6, but this was THEORETICAL dynamics analysis, not the ACTUAL implementation

**The situation:**
- After implementing adaptive LR, diagnostic showed: "Using adaptive inference LR: 3.88e-06"
- Inference STILL didn't converge: ΔF = 0.0000 (no convergence)
- Training showed no learning: loss=2.3026, acc=9.38% (random guessing)
- Vanishing errors: massive activity in layer 1, zero elsewhere

**User's feedback:**
- "I think you almost certainly interpreted 'don't skim over the paper' as 'don't skim over the paper on this particular issue'"
- "when you were actually supposed to interpret it as 'carefully read the entire paper and don't miss a detail'"
- Told me to read **Appendix F thoroughly**

**What Appendix F.1 (page 12) ACTUALLY says:**
> "For the energy and MNIST datasets, we employ the same neural network architecture. Specifically, we use a **four-layer neural network with 128 hidden units per layer** and ReLU activations. Training is performed using **mini-batches of size 128**."

> "For BPC, we used the **Adam optimizer for hidden states, with a learning rate of 0.01 and 10 iterations per batch**."

**What I had implemented (WRONG):**
```python
# Architecture: 7 layers, 256 units, batch 64, T=35
layer_sizes = [784, 256, 256, 256, 256, 256, 128, 10]
T = 35
batch_size = 64
inference_lr = 0.01  # But then overridden by adaptive 3.88e-6!
```

**What the paper actually uses (CORRECT):**
```python
# Architecture: 4 layers, 128 units, batch 128, T=10
layer_sizes = [784, 128, 128, 128, 10]
T = 10
batch_size = 128
inference_lr = 0.01  # FIXED, not adaptive!
```

**The critical insight I missed:**

From **Page 6 Discussion**:
> "the current estimate of Σ acts as an adaptive learning rate during inference of latent variables Z"

**The precision Σ^{-1} is ALREADY IN THE GRADIENT** (Equations 15-16):
```
∇E = ⟨Σ^{-1}(z - Wf)⟩
```

When E[Σ^{-1}] = 258,000, the errors are already weighted 258,000x more heavily. The gradient is ALREADY scaled by precision!

**By using α = 1/258,000 = 3.88e-6, I was DOUBLE-PENALIZING:**
- Gradient: 258,000 × error
- Step: 3.88e-6 × (258,000 × error) = 1 × error ✗ WRONG!

**Should be:**
- Gradient: 258,000 × error
- Step: 0.01 × (258,000 × error) ✓ CORRECT!

**Appendix B vs Appendix F confusion:**
- **Appendix B**: Theoretical dynamics analysis (α ≈ 1/λ_max is upper bound on stable LR)
- **Appendix F**: ACTUAL implementation details (LR = 0.01, T = 10, 4 layers, 128 units)

I implemented the theoretical analysis instead of the practical implementation!

**Root principle violated (AGAIN!):**
1. I read Appendix B but NOT Appendix F
2. I interpreted "don't skim" narrowly (just that section) instead of broadly (entire paper)
3. I assumed theoretical dynamics = implementation details
4. I didn't check experimental setup in Appendix F

**The lesson (reinforced):**
**WHEN IMPLEMENTING A RESEARCH PAPER:**
1. Read METHODS section for algorithm
2. Read APPENDICES for theoretical analysis (Appendix B - dynamics)
3. Read EXPERIMENTAL DETAILS for actual hyperparameters (Appendix F - implementation)
4. Don't confuse theoretical bounds with practical settings
5. Match the paper's experimental setup EXACTLY before making changes

**Files fixed:**
- `experiments/BayesianPC/bayesian_pc_layer.py` - REMOVED incorrect get_optimal_inference_lr()
- `experiments/BayesianPC/bayesian_pc_trainer.py` - Use fixed LR=0.01 from Appendix F
- `experiments/BayesianPC/train_mnist_bayesian.py` - Match paper: 4 layers, 128 units, batch 128, T=10

**Status:** FIXED (2026-02-01) - using paper's actual implementation from Appendix F.1

---

### Successful Implementation (2026-02-01)

**Approach:** Minimal custom implementation based on standard PC algorithm
**Files created:**
- `src/network/pc_layer.py` - PCLayer and PCNetwork classes
- `src/network/pc_trainer.py` - PCTrainer with two-phase algorithm
- `train_mnist_pc.py` - Full MNIST training with diagnostics
- `test_pc_basic.py` - Basic functionality tests

**Key decisions:**
1. Used standard PCLayer architecture from Bogacz Group
2. Value nodes (_x) as nn.Parameters optimized during inference
3. Energy function: E = 0.5 * (mu - x)^2
4. Two-phase training: inference (35 iterations) + learning (weight update)
5. Proper weight initialization (He for ReLU)
6. Comprehensive diagnostics for vanishing error detection

**What worked:**
- Direct implementation of published algorithm
- Simple, understandable code structure
- Followed MISTAKES.md guidelines throughout
- No custom neuron designs
- No output clamping
- No exponential precision scaling

**Pending:**
- Full training run (requires environment with PyTorch installed)
- μPC residual scaling (add if vanishing errors detected)
- Integration with active inference wrapper

---

## Research References to Consult

- **VERSES AI** (Karl Friston): Solutions for scaling predictive coding
- **Millidge et al.**: Predictive coding networks implementations
- **Whittington & Bogacz**: Standard predictive coding architectures
- **Friston's Active Inference papers**: Theoretical foundations

---

### 14. Adding Cross-Entropy Task Loss That Paper Never Specified (CRITICAL)

**What I did wrong:** Added `F.cross_entropy(outputs, targets)` as a separate task loss in the free energy, combined as `free_energy = loss + total_energy`. The paper never uses a separate task loss.

**What the paper actually says (page 3):**
> "trains models in a discriminative manner by **fixing the input nodes to z₀ = x⁽ⁱ⁾ and the output nodes to z_L = y⁽ⁱ⁾**"

The output is **clamped** to the target (one-hot encoded). The prediction errors at all layers (including the output) are the ONLY learning signal. There is NO separate cross-entropy loss.

**Why this was catastrophic:**
- Cross-entropy (~2.3) drowned out prediction error energy (~10⁻⁶)
- The BPC machinery (Bayesian posteriors, Hebbian updates, prediction errors) contributed NOTHING
- The model was effectively doing backprop through value nodes via cross-entropy
- Ψ had no effect because the energy terms were negligible
- 88% accuracy came entirely from output-layer backprop, not PC

**The correct approach:**
1. Clamp z₀ = input, z_L = one_hot(target) (both observed, not optimized)
2. Optimize ONLY hidden value nodes z₁,...,z_{L-1} via prediction error energy
3. Free energy = Σ_l E_l (sum of prediction errors, NO task loss)
4. The large prediction error at the output layer (target vs prediction) drives learning

**Root principle violated:** Implement what the paper says, not what seems convenient. PC doesn't need a separate task loss — the clamped output IS the supervisory signal.

**Files fixed:**
- `experiments/BayesianPC/bayesian_pc_trainer.py` — clamp output, remove task loss, exclude output from optimizer
- `experiments/BayesianPC/bayesian_pc_layer.py` — save last layer input for energy recomputation
- `experiments/BayesianPC/train_mnist_bayesian.py` — remove loss_fn from training call

**Status:** FIXED (2026-02-06) - output clamping per paper's Algorithm 1

---

### 15. Using SGD for ePC Error Optimization with Precision-Weighted Gradients (CRITICAL)

**What we did wrong:** Used SGD with lr=0.001 for error optimization in eBPC, matching the ePC paper's hyperparameters.

**Why it failed catastrophically:**
- BPC precision E[Σ^{-1}] = νΨ ≈ 0.001 (from posterior updates) scales ALL error gradients down ~1000x
- SGD lr=0.001 gives effective step size ~1e-6 (lr × gradient_scale)
- Errors stay at ~0 throughout inference → hidden states un-inferred
- Un-inferred states → garbage sufficient statistics → Hebbian updates corrupt weights
- Accuracy DEGRADED from ~90% (epoch start) to 38% (epoch end)
- Per-layer energies at 10^{-16} (effectively zero)

**Symptoms:**
- Accuracy degrading within first epoch (not stagnant — actively getting worse)
- All per-layer energies near zero
- Inference convergence at ~1e-8 (nothing moving)
- Error optimization doing nothing visible

**Why ePC paper used SGD but we can't:**
- Standard ePC uses identity precision (Σ^{-1} = I) — gradients are unit scale
- BPC's learned precision can be orders of magnitude different from 1
- The gradient scale mismatch makes fixed-LR SGD useless

**Correct approach:**
```python
# Use Adam — its adaptive normalization compensates for gradient scale
error_optim = optim.Adam(errors, lr=0.01)  # Not SGD!
```

**Why Adam works here:**
- Adam normalizes by running variance of gradients
- Even if gradients are scaled down 1000x, Adam adapts its effective step
- lr=0.01 matches BPC Appendix F.1 (which also uses Adam for inference)

**Result:** 95.74% test accuracy with Adam (up from 38% with SGD), exceeding both BPC (93.5%) and standard PC (95.14%)

**Root principle:** When combining two methods (ePC + BPC), don't blindly copy hyperparameters from one — understand how they interact.

**Status:** FIXED (2026-02-07) — use Adam lr=0.01 for error optimization in eBPC

---

### 16. Diagonal MNW Approximation Breaks Positive Definiteness (ACTIVE)

**What we did wrong:** Replaced full V (in×in) and Ψ (out×out) matrices with diagonal vectors, assuming the math would "just work."

**Why it broke:**
- Full MNW: Φ = η₃ - η₂ η₁⁻¹ η₂ᵀ is guaranteed PD (Schur complement of PD matrix)
- Diagonal approximation: Φ_diag = η₃ - Σᵢ(η₂ᵢ²/η₁ᵢ) can go negative
- When R² > 1 (sum of squared correlations exceeds 1), Φ_diag becomes negative
- Negative Φ → negative Ψ → negative precision → energy explosion → NaN

**Symptoms:**
- 9.8% test accuracy (random chance)
- NaN losses
- Layer 4 (output) energy at 10^{11}
- Explosion within first 4-5 batches

**First fix attempt:** Clamp Φ_diag at prior value (psi_inv_prior). **DID NOT WORK** — identical broken results.

**Diagnostic script created:** `experiments/eBPC_ResNet/diagnose_diagonal.py` — traces all natural parameters, standard parameters, raw Phi, precision at each batch. Not yet run.

**Hypotheses still to test:**
1. Is the Φ clamp code path actually executing?
2. Is M (weight mean) exploding? (η₂/η₁ ratio)
3. Are output predictions exploding?
4. Is NaN originating from somewhere other than precision?
5. Are the sufficient statistics (ss1, ss2, ss3) computed correctly for diagonal case?

**Root principle:** Mathematical guarantees of full-matrix formulations don't automatically transfer to diagonal approximations. Verify PD guarantees analytically before implementing.

**Status:** ACTIVE — diagnostic script created, awaiting results

---

### 17. Python Cannot Import from Hyphenated Directory Names

**What we did wrong:** Named a directory `eBPC-ResNet` (with hyphen).

**Why it broke:** Python interprets hyphens as minus operators in import statements. `from experiments.eBPC-ResNet import ...` is parsed as `experiments.eBPC` minus `ResNet`.

**Correct approach:** Use underscores: `eBPC_ResNet`

**Status:** FIXED (2026-02-07) — renamed to eBPC_ResNet

---

### 18. Low-Rank η1 Violates MNW Quadratic Constraint (CRITICAL)

**What we did wrong:** When truncating η1 = diag(d) + U·U^T to rank-k, absorbed only `diag(R)` (diagonal of residual matrix R = AA^T - U_new·U_new^T) into d.

**Why it broke:**
- The MNW block matrix `[[η1, η2^T], [η2, η3]]` must be PSD (the "quadratic constraint")
- Equivalent to Schur complement Φ = η3 - η2·η1⁻¹·η2^T > 0
- `diag(diag(R))` is NOT ≥ R in PSD ordering (off-diagonal elements of R are lost)
- So η1_approx < η1_true → η1_approx⁻¹ > η1_true⁻¹ → Schur complement goes negative
- Negative Φ → negative precision → energy explosion → NaN

**Two wrong approaches tried first:**
1. **Residual-based Ψ** — replaced η3 with residual-based psi_inv. Broke MNW conjugacy, created positive feedback (M↑ → r↑ → psi_inv↑ → precision↑ → M↑)
2. **Schur complement with diag(R) absorption** — correct formula but η1_approx < η1_true, so Phi went to -2.31e+31

**Correct approach: Spectral norm inflation**
- λ_max(R) = largest dropped eigenvalue from eigendecomposition
- Set d_inflation = λ_max(R) for ALL diagonal entries (not per-element diag(R))
- Then η1_approx - η1_true = λ_max(R)·I - R ≥ 0 (since all eigenvalues of R ≤ λ_max(R))
- This guarantees η1_approx ≥ η1_true → η1_approx⁻¹ ≤ η1_true⁻¹ → Φ stays positive

**Root principle:** When approximating a component of a joint distribution, the approximation must preserve the **validity constraints** of the full distribution. For MNW, this is the PSD constraint on the block natural parameter matrix — a quadratic matrix inequality.

**Status:** FIX IMPLEMENTED (2026-02-07) — spectral norm inflation in `_update_eta1_lowrank`

---

### 19. FITC Diagonal Correction Fails When k << data_rank (CRITICAL)

**What we did wrong:** Replaced conservative Gershgorin inflation (d_i = Σ_j |R_ij|) with FITC diagonal correction (d_i = R_ii) from GP literature, assuming the residual was "prior-dominated" for all layers.

**Why it broke catastrophically:**
- FITC absorbs only diag(R) of the dropped residual — exact diagonal, NO PSD upper bound
- This does NOT guarantee η1_approx ≥ η1_true (missing off-diagonal of R)
- When off-diagonal R is large, V = η1_approx⁻¹ is too large → M = η2·V explodes
- Runaway feedback: larger V → larger M → larger ss2 → larger η2 → even larger M
- Result: M reaches 1.75e17 within 15 batches, Phi goes to -9.76e+35, NaN

**The critical condition: FITC only works when k > data_rank for that layer.**

Layer analysis with proportional k (ratio=8), batch_size=128:
- Layer 1 (in=785, k=98): data rank ≤ 128, k=98 captures most directions. Residual is prior-dominated (diagonal prior has no off-diag). diag(R) ≈ R. FITC is safe.
- Layers 2-4 (in=129, k=20): data rank ≤ 128 ≈ 129! Data can fill the ENTIRE space. k=20 drops 109 eigenvalues that are heavily data-driven with strong off-diagonal coupling. diag(R) << R by orders of magnitude. FITC fails.

**Phi mins by layer:** L1: -3.43e+13, L2: -2.67e+26, L3: -9.76e+35, L4: +5.55
Layer 4 (out=10, smallest) survived. Layers 2-3 exploded worst.

**Why the Phi clamp at 1.0 didn't save us:**
The clamp limits precision, but the explosion happens in M (via V), not in the precision pathway. By the time we compute Phi, M is already at 1e17. Clamping Phi can't un-explode M.

**What Gershgorin got right (despite being too conservative):**
- Gershgorin guarantees diag(d) ≥ R in PSD sense
- This makes η1_approx ≥ η1_true → V_approx ≤ V_true → M is bounded
- Too conservative for Layer 1 (over-regularized, 82.59%) but STABLE

**Correct approach (not yet implemented):**
Need a hybrid:
1. FITC (diag(R)) for layers where residual is prior-dominated (k ≈ data_rank)
2. Conservative bound (Gershgorin or spectral) for layers where data fills the space
3. Or: increase k so ALL layers have prior-dominated residuals (k > batch_size)
4. Or: use full η1 for small layers (in ≤ ~256) and low-rank only for large ones

**Root principle:** An approximation's validity depends on the REGIME, not just the formula. FITC is a principled approximation that works beautifully when the low-rank component captures most of the data's structure. It fails when significant data-driven correlation is dumped into the residual. Always verify the preconditions of an approximation match the actual operating regime.

**Status:** ACTIVE (2026-02-08) — FITC reverted to NaN. Need hybrid approach or increased k.

---

## Update Log

- 2026-02-01 (initial): Initial file created with 6 major mistakes catalogued
- 2026-02-01 (implementation): Fixed mistake #2 (custom neurons) with standard PCLayer implementation
- 2026-02-01 (documentation): Fixed mistake #6 (context loss) with persistent documentation system
- 2026-02-01 (code complete): Added successful implementation section with all PC network code
- 2026-02-01 (CRITICAL FIX): Found and fixed mistake #8 - optimizer was conflating value nodes with network parameters, causing complete learning failure. Training was stuck at 8% (random guessing). Fixed by separating parameter lists.
- 2026-02-01 (THE REAL BUG): Mistake #8 fix had ZERO effect. Debug script showed weights never changed. Found actual bug (mistake #9): mu.detach() broke computational graph so NO gradients reached weights. Removed detach calls. This is the real fix.
- 2026-02-01 (BAYESIAN PC BUGS): Fixed mistakes #10 and #11 - Bayesian PC had NaN catastrophe from variance collapse (min_variance=1e-6 too low, no max bound) and broke experimental control by changing from 7 to 3 layers. Fixed by setting min_variance=0.01, max_variance=10.0, and restoring 7-layer architecture.
- 2026-02-01 (FUNDAMENTAL ERROR): Identified mistake #12 - entire BayesianPC implementation is conceptually wrong. Put Bayesian posteriors over VALUE NODES (hidden states) instead of WEIGHTS (parameters). User provided paper (Tschantz et al. 2025) showing BPC uses Matrix Normal Wishart weight posteriors with closed-form Hebbian updates, not distributions over hidden states. All code in experiments/BayesianPC/ must be archived and rewritten from scratch following Algorithm 1 from the paper.
- 2026-02-01 (CRITICAL PROCESS ERROR): Mistake #13 - Skimmed paper instead of reading thoroughly. Made lazy speculations about "energy normalization" when paper explicitly covered adaptive learning rate in Appendix B/C. NEVER SKIM PAPERS.
- 2026-02-07 (eBPC SGD FAILURE): Mistake #15 - Used SGD for error optimization when BPC precision scaled gradients 1000x down. Switched to Adam lr=0.01 → 95.74% test accuracy.
- 2026-02-07 (DIAGONAL BREAKAGE): Mistake #16 - Diagonal MNW approximation breaks PD guarantee. Phi_diag can go negative → NaN. Active debugging.
- 2026-02-07 (NAMING): Mistake #17 - Python can't import from hyphenated directories. Use underscores.
- 2026-02-08 (QUADRATIC CONSTRAINT): Mistake #18 - Low-rank η1 truncation violates MNW quadratic constraint. Fixed via spectral norm inflation. Stable at 82.59%.
- 2026-02-08 (FITC FAILURE): Mistake #19 - FITC diag(R) correction fails for Layers 2-4 where k=20 << data_rank≈128. V explodes → M reaches 1e17 → NaN at batch 15. FITC only works when residual is prior-dominated (k > data_rank).
