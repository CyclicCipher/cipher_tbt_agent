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

### 20. KFAC (KRONOS) Is Structurally Incompatible with ePC's E_local (CRITICAL)

**What we tried:** Built KRONOS, a KFAC-family second-order weight optimizer using LRPD decomposition, to replace Adam for ePC weight updates. Spent 6+ versions (v1-v5.2) debugging.

**Why it failed fundamentally:**

ePC's E_local violates all three assumptions KFAC relies on:

1. **G factor is degenerate**: After Newton inference (T=2), prediction errors are tiny → output gradients ≈ 0 → G = E[gg^T] has trace(G) << trace(A) by 40,000:1. Half the Kronecker factorization carries zero curvature information.

2. **Raw gradients are tiny**: E_local divides by batch_size * energy_scale on small errors. Raw gradient magnitudes are ~1000x smaller than standard networks. Any approach that normalizes to or preserves raw magnitude (v3, v5.2) produces microscopic steps → 64-72% accuracy.

3. **A eigenvalues span 750:1 across layers**: L1 median(d_a) = 0.15, L4 = 0.0003. Any uniform treatment (flat damping, uniform LR) creates cross-layer mismatch. Adaptive damping overcorrects (λ_L4 = 0.000002 → A^{-1} amplification explodes to 972x).

**The catch-22 we couldn't escape:**

| Fix | Problem it creates |
|---|---|
| Keep raw magnitude | Steps too small (64-72%) |
| Amplify uniformly (high LR) | 100x cross-layer mismatch |
| Clip to fixed norm | Destroys magnitude info → expensive direction-only modifier |
| Drop momentum | Loses gradient smoothing, noisier optimization |
| Adaptive damping | Overcorrects for deep layers → explosion |
| Norm-preserving | Same as "keep raw magnitude" → tiny steps |

Best result: v4 (full A+G, flat damping, clipping) at 95.4% — only 1.5% below Adam's 96.8%, but 30% slower and far more complex.

**Root cause: ePC needs per-element MAGNITUDE adaptation (what Adam provides), not matrix DIRECTION rotation (what KFAC provides).** KFAC's value is rotating gradients via A^{-1} and G^{-1}. Adam's value is per-element scale normalization via running variance. ePC's tiny, scale-varying gradients need the latter.

**What would have worked:**
- Just use Adam (which is already a diagonal Fisher approximation)
- Or diagonal Fisher/Hessian: F_diag = diag(A) ⊙ diag(G) (cheaper, no matrix rotation)
- KFAC may work for non-ePC architectures where G carries real curvature

**LRPD library value:** The LRPD library (woodbury, schur, log_det, online_update, alt_decompose) remains valuable for future work (JEPA sparse GP, Mamba state covariance) where full-rank covariance matrices appear naturally. The library is sound — the application was wrong.

**Status:** ARCHIVED (2026-02-10) — KRONOS moved to experiments/archived_kronos/, Newton+Adam adopted for all ePC experiments

---

### 21. INT8 QAT Destroys ePC Accuracy (CRITICAL)

**What we tried:** Fake INT8 weight quantization (QAT) during training using `torch.nn.utils.parametrize` with Straight-Through Estimator.

**Why it failed:**
- ePC's error optimization is highly sensitive to weight precision
- Fake quantization adds quantization noise to weights every forward pass
- The Newton error step converges based on the EXACT current weights
- INT8 noise (127 discrete levels per tensor) corrupts the energy landscape
- Result: 57.09% at epoch 10 vs 80.49% without QAT — 23% accuracy loss

**Symptoms:**
- Learning proceeds but much slower than without QAT
- All other metrics look normal (energies, error magnitudes, convergence)
- The accuracy simply plateaus much lower

**Why this is specific to ePC:**
- Standard backprop networks tolerate QAT because gradients adapt to the quantized weights
- ePC runs T iterations of error optimization on FIXED (quantized) weights per batch
- The quantization noise creates a noisy energy landscape that errors can't optimize well
- Each batch sees a DIFFERENT quantization (weights change → new quantization grid)

**Root principle:** ePC's inference loop amplifies weight noise because it optimizes errors against fixed weights for multiple iterations. Any noise in weights gets amplified by T iterations of optimization against that noise. Standard networks don't have this problem because they do a single forward pass.

**Status:** DISABLED (2026-02-10) — quantize_bits=0 in both training scripts. QAT code retained for reference.

---

### 22. AdaWoodbury Rank-1 Correction Provides No Benefit Over Adam

**What we tried:** AdaWoodbury — Adam's diagonal second moment + a rank-1 Woodbury curvature correction via online PCA of the gradient stream. Theory: capture cross-parameter curvature that diagonal Adam misses, using H ≈ diag(√v+ε) + α·uu^T and Woodbury inversion.

**Algorithm:**
- Online PCA: u tracks top eigenvector of E[gg^T] via streaming power iteration
- Adaptive α: measures excess curvature along u vs diagonal prediction
- After warmup (100 steps): applies rank-1 Woodbury correction to Adam step
- Memory: 50% over Adam (3n vs 2n per param). Compute: 3 extra dot products/step.

**Results:**
- MNIST: 96.74% test (3 epochs) — matches Adam's 97.07% within noise. Not remarkable.
- CIFAR-10: 41.75% test epoch 1, 42.41% epoch 2 — same or slightly worse than Adam baseline (~42% epoch 1, 80.49% epoch 10). No convergence speedup visible.

**Why it failed:**
- Rank-1 correction is too weak: one direction of curvature correction across millions of parameters is a drop in the ocean
- The dominant eigenvector of E[gg^T] captures gradient VARIANCE, not loss curvature — these are related but not the same thing
- ePC gradients come from E_local (sum of local layer energies), which already decomposes curvature across layers. The global gradient's top eigenvector doesn't align with any meaningful per-layer curvature direction
- The adaptive α mechanism correctly detects when diagonal is sufficient (ratio ≈ 1) — it self-disables, making AdaWoodbury equivalent to Adam in practice
- Fundamentally: if the diagonal (Adam) already explains 99%+ of the curvature structure, a rank-1 off-diagonal correction explains ~0.001% of what's left

**Lesson:** Second-order methods need to match the problem's curvature structure. ePC's curvature is:
1. **Per-element scale variation** (handled well by Adam's v)
2. **Cross-layer scale mismatch** (handled by Adam's per-parameter normalization)
3. **Block-diagonal** (each layer's E_local is independent)
The actual curvature structure is block-diagonal per layer, not low-rank globally. A global rank-1 correction addresses none of these.

**What might actually work for faster convergence:**
- Layer-wise learning rate adaptation (each layer's E_local has different scale)
- Larger batch size + linear scaling rule (more signal per step)
- Better LR schedules (warmup already helps significantly)
- Architectural changes (error-gated inference for speed, not optimizer changes for convergence)
- Accept that ePC's convergence rate IS the convergence rate — the error optimization loop is the bottleneck, not the weight optimizer

**Status:** IMPLEMENTED but INEFFECTIVE (2026-02-10) — Code retained in ada_woodbury.py. train_cifar10.py defaults to adawoodbury but can switch back to 'adam'. Recommend reverting to Adam.

---

### 23. Inference-Mode Forward Pass Overwrites ePC Errors Before Diagnostics (BUG)

**What we did wrong:** In the ePC-Mamba training loop, called `model(inputs)` (without targets) for accuracy evaluation BEFORE collecting `get_diagnostics()`. The `forward(targets=None)` path sets `self.pce.errors = [0.0] * (n_layer - 1)`, replacing tensor errors with scalar floats. Then `get_diagnostics()` checks `isinstance(e, Tensor)` → `False`, and collects nothing.

**Symptoms:**
- Per-Layer Energies plot completely empty (x-axis 0.0 to 1.0 = no data)
- Error Magnitudes plot completely empty
- H2 causal asymmetry = 0.000 (falls to else branch returning 0.0)
- Other diagnostics (convergence, weight magnitudes) worked because they read from saved attributes, not from error tensors

**Why it was hard to spot:**
- Convergence metric still worked (saved as `_E_initial` and `_E_final` during inference)
- The error overwrite happens silently — no error or warning
- The `isinstance(e, Tensor)` check was a correct guard but masked the data loss

**Root principle:** When a model has ephemeral state (like ePC errors), any function that resets that state must not be called between state population and state consumption. Order of operations matters.

**Fix:** Move `get_diagnostics()` and `get_hypothesis_diagnostics()` BEFORE `model(inputs)` for accuracy.

**Status:** FIXED (2026-02-10) — diagnostics collected before accuracy eval

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
- 2026-02-10 (KRONOS ARCHIVED): Mistake #20 - KFAC structurally incompatible with ePC. G factor degenerate (40,000:1 trace ratio), raw gradients tiny, cross-layer A eigenvalue spread 750:1. ePC needs Adam, not KFAC.
- 2026-02-10 (QAT FAILURE): Mistake #21 - INT8 fake quantization destroys ePC accuracy (57.09% vs 80.49%). ePC error optimization amplifies weight noise over T iterations.
- 2026-02-10 (ADAWOODBURY INEFFECTIVE): Mistake #22 - Rank-1 Woodbury correction over Adam provides no convergence benefit. Global rank-1 curvature doesn't match ePC's block-diagonal per-layer structure. MNIST 96.74% (≈Adam), CIFAR-10 no improvement.
- 2026-02-10 (BPC LRPD INTRACTABLE): MNW conjugacy rules are fundamentally hard to capture in any LRPD approximation. Multiple approaches attempted (diagonal, low-rank η1, FITC, spectral inflation) — all either break PD guarantees or over-regularize. The quadratic constraint (block PSD of [[η1,η2^T],[η2,η3]]) creates a tight coupling between the approximation of η1 and the validity of the Schur complement Φ. No known LRPD form preserves conjugacy + PD simultaneously at scale. Revisit only with fundamentally new approach.
- 2026-02-10 (DIAGNOSTIC ORDERING): Mistake #23 - forward(targets=None) resets pce.errors to scalar [0.0], destroying tensor errors before get_diagnostics() could read them. Fix: collect all error-dependent diagnostics before accuracy eval.
- 2026-02-11 (NEWTON STEP DEBUGGING SPIRAL): Mistakes #24-26 - Three failed attempts to "fix" Newton step for ePC-Mamba. Original rank-1 was working (86%) but we declared it broken and spent hours making it worse. See individual entries below.
- 2026-02-12 (ADAPTIVE DAMPING REGRESSION): Mistake #30 - Adaptive Newton damping from prospective configuration paper caused complete regression (99.2% → 7.75%). Reverted.
- 2026-02-12 (IPC NEVER WORKED): Mistake #31 - The "99.2% iPC" result was standard ePC. `--ipc` flag was added to parser/print at f417cb8 but NOT wired into training loop until 7fcb4de. Proof: 97ms speed = standard ePC, loss 290 = E_local/32. Real iPC = 126ms, loss 6834. 6 hours wasted on "non-reproducibility crisis." Standard ePC confirmed working: 99.03% at epoch 27 (seed=0).

---

### 24. Replacing a Working Newton Step with Broken Alternatives (DEBUGGING SPIRAL)

**What we tried:** The rank-1 LRPD Newton step gave convergence ≈ 1.0 (barely moved errors). We diagnosed this as "too conservative" and tried two replacements:
1. **Pure diagonal step** (H≈dI, step=1/d=0.91): Energy INCREASED 70x, accuracy dropped to 10%
2. **Dimension-normalized step** (α=1/(d+||g||²/n)): Still too aggressive, loss increased, 10% accuracy

**Why both broke:** The rank-1 step from zero errors degenerates to `α ≈ 1/(d+||g||²) ≈ 1/dim`, which IS tiny. But the replacement step sizes (0.91 and 0.48) were 100,000x larger — way past the stability boundary. The energy landscape through Mamba's SSD computation is highly non-quadratic; large steps overshoot catastrophically.

**What we should have done:** The rank-1 Newton WAS working — the model reached 86% accuracy with it. Convergence ≈ 1.0 looked bad but the small errors still provided useful (if weak) E_local gradient signal. The real problems were instability (weight dynamics) and slow learning (E_local gradient imbalance), not the Newton step itself.

**Root principle:** When a component works but looks suboptimal, diagnose whether it's actually the bottleneck before replacing it. A working solution replaced by a "better" one that breaks everything is a net loss.

**Status:** REVERTED (2026-02-11) — original rank-1 restored

---

### 25. Adam for Error Optimization Over-Optimizes, Killing E_local Signal (CRITICAL)

**What we tried:** Used `torch.optim.Adam(errors, lr=0.01)` with T=5 for error optimization, based on diagnose_newton.py showing Adam T=5 dropped energy by 334 (vs Newton's 2.0).

**Results:** 7.5% accuracy (random chance) despite excellent energy convergence (E_init=16722 → E_final=544). Errors were beautifully optimized but the model couldn't learn.

**Why it failed:** Adam-optimized errors perfectly compensate for the current (bad) weights. E_local computes `0.5 * ||f(x) - (f(x) + e)||² = 0.5 * ||e||²` per layer. When errors are large and precisely tuned to fix outputs, the local prediction error LOOKS small from each layer's perspective — the error was "placed" exactly where needed. So E_local gives each layer a gradient that says "everything is fine locally" even though the weights are wrong. No useful weight updates happen.

**The paradox:** Better error optimization → worse weight learning. ePC NEEDS imperfect error optimization so that E_local's local terms provide meaningful gradient signal.

**Root principle:** In a two-phase algorithm (inference + learning), optimizing Phase 1 too well can destroy Phase 2's signal. The phases are coupled — don't optimize them independently.

**Status:** ABANDONED (2026-02-11) — Adam for errors doesn't work with ePC's E_local

---

### 26. ePC Needs Many Layers — 2-Layer Networks Are Pathological

**What we observed:** On the copy task with 2 Mamba layers:
- Backprop: 93.5% in 50 epochs (12ms/batch)
- ePC (Newton T=2): 85.6% in 50 epochs (44ms/batch)
- ePC (Newton T=10): 52% in 10 epochs (103ms/batch)

Layer 1 received 490x less gradient than Layer 2 (from E_local's detach). ||e|| ≈ 0.002, so Layer 1's gradient ∝ ||e|| ≈ 0.002 while Layer 2 gets full CE gradient.

**Why 2 layers is pathological for ePC:**
- E_local detaches between layers, so each layer only gets gradient from its LOCAL prediction error
- With only 1 error between 2 layers, Layer 1's only gradient source is `||e_1||²`
- When Newton gives tiny errors (||e|| ≈ 0.002), Layer 1 is effectively frozen
- Layer 2 learns from CE gradient; Layer 1 drifts on noise
- With 8+ layers (like ResNet), gradient is distributed across 7+ error terms — no single layer starves

**Comparison:** MNIST MLP (4 layers, 3 errors) got 95.74%. CIFAR-10 ResNet (10 layers, 8 errors) got ~82%. The more layers/errors, the more gradient signal E_local distributes.

**Root principle:** ePC's biologically-plausible local learning comes at a cost: gradient flows only through local error terms. With few layers, this creates severe imbalance. The architecture must have enough layers for E_local to provide useful gradients to all layers.

---

### 27. Init Scale: Naive Output Projection Scaling Destroys Training

**What we tried:** Scaling `mixer.out_proj.weight` and `mlp.down_proj.weight` by 2x at initialization (`--init_scale 2.0`) to increase the Jacobian dy/de from epoch 1, hoping to bypass the 28-epoch deadlock phase.

**Why it failed:** The 2x scaling blew up activations (initial loss 4846 vs 290 for standard), and Newton went catastrophically unstable (convergence went to -60000). The model got stuck at 7% for 30 epochs — worse than standard ePC. Scaling output projections couples Jacobian magnitude with activation magnitude. You can't increase one without the other.

**Root principle:** Forward dynamics and Jacobian magnitude are coupled through the weights. Scaling weights to get larger Jacobians also scales activations, putting the model in a high-loss region where E_local gradients are unhelpful and Newton's rank-1 approximation breaks down.

**Status:** ABANDONED (2026-02-11) — init_scale doesn't work for ePC

---

### 28. mHC (Manifold-Constrained Hyperconnections) Too Slow and Unstable

**What we tried:** Proper implementation of DeepSeek's mHC (arXiv:2512.24880) with 2 parallel residual streams, Sinkhorn-Knopp doubly stochastic mixing (H_res), softmax aggregation (H_pre), and softmax distribution (H_post). The manifold constraint prevents signal blowup while giving Newton 2x more error degrees of freedom.

**Results:** ~3x slower (267 ms/batch vs 98 ms), reached 98.2% best accuracy but wildly unstable — test accuracy swung between 94% and 9% across epochs, final test acc was 32%. Convergence went negative (-32 mean). Newton's rank-1 Hessian approximation couldn't handle the richer stream-space landscape.

**Why it partially worked:** No deadlock plateau (30% by epoch 13 vs 7% for standard ePC). The stream mixing gave errors more paths to influence the output.

**Why it ultimately failed:** (1) Sinkhorn overhead: 80 calls per forward pass on tiny 2x2 matrices, but PyTorch dispatch adds up. (2) Stream summing at output dilutes each stream's error contribution. (3) Newton's rank-1 approximation breaks in the 2x larger error space.

**Root principle:** More degrees of freedom for Newton is not better when the Hessian approximation is fixed at rank-1. The approximation quality degrades as the problem dimension grows.

**Status:** ABANDONED (2026-02-11) — mHC overhead and instability outweigh benefits

**Status:** DOCUMENTED (2026-02-11)

---

### 29. muPC (Depth-muP) Crushes the Jacobian — Opposite of Init Scale

**What we tried:** Applied Depth-muP scaling (Innocenti et al. 2025, arXiv:2505.13124) to each Mamba3Block's non-residual contributions. Each mixer and MLP output was scaled by alpha = 1/sqrt(d_model * 2*n_layer) ≈ 0.031 for our 4-layer d=128 model.

**Results:** Stuck at 7-8% accuracy (random chance) for 14+ epochs. Loss plateaued at ~2780 (same as init_scale failure). Initial loss 175565 (catastrophically high). Newton convergence only 1.29 (vs ~2700 normally). Error norms were tiny (Layer 1: 0.004, Layer 4: 0.12).

**Why it failed:** Alpha = 0.031 means each sub-layer contributes only ~3% to the residual stream. This crushed the Jacobian dy/de to near-zero, making Newton corrections negligible. The muPC paper explicitly admits it doesn't fix PC inference landscape conditioning — it only stabilizes the forward pass, which doesn't help when the bottleneck is the Jacobian being too small for Newton to work.

**Root principle:** Init_scale (#27) and muPC (#29) are two sides of the same coin. Init_scale made the Jacobian too large (blew up activations). muPC made it too small (crushed non-residual contributions). Both break Newton. The Jacobian needs to be in a specific range for Newton's rank-1 approximation to cross the critical threshold that triggers the phase transition.

**Status:** ABANDONED (2026-02-11) — muPC designed for SGD/Adam error optimization, not Newton

---

### 30. Adaptive Newton Damping Broke Default Training — Complete Regression

**What we tried:** Implemented adaptive Newton damping inspired by Song et al. 2024 (prospective configuration, Nature Neuroscience). If a Newton step increased energy, the damping was doubled for the next step within the same batch. Also added convergence-based early stopping for error optimization.

**Results:** COMPLETE REGRESSION. Default `python experiments/Mamba3/train_epc.py` went from 99.2% at epoch 36 to 7.75% over 50 epochs. Loss stuck at ~2780 for all 50 epochs. The phase transition at epoch ~28 NEVER occurred. Speed also regressed (134ms vs ~98ms per batch). With `--iters 8 --conv_threshold 0.01`, speed was 482-502 ms/batch (5x slower), cancelled after 2 epochs.

**Why it failed:** The adaptive damping check (`if E_val > E_prev: effective_damp *= 2.0`) combined with the code restructuring introduced a subtle regression. The inference convergence chart showed massive negative spikes (-45000) early on, indicating Newton was massively overshooting. Despite damping resetting per batch, the doubled damping (0.2 vs 0.1) during those critical early batches appears to have prevented the system from building up the Jacobian magnitude needed for the phase transition at epoch ~28. The convergence values (mean=1.15) matched the muPC failure pattern (1.29), suggesting the Newton step was effectively neutered.

**Key insight:** The prospective configuration paper used γ=0.1 with T=128 iterations and SGD/gradient-descent-style error optimization. Their adaptive step reduction (halving γ on overshoot) works in that regime because SGD takes many small steps. Newton's rank-1 step in T=2 iterations is a fundamentally different regime — there's no room for the damping to recover within a 2-step loop. The doubled damping on step 1 makes step 2 too conservative, and that's all you get.

**Root principle:** Don't "improve" a working system based on insights from a paper that uses a fundamentally different optimization method. The prospective configuration paper uses SGD with T=128; we use Newton with T=2. The adaptive damping strategy doesn't transfer across these regimes.

**Status:** REVERTED (2026-02-12) — all adaptive damping code removed, defaults restored

**CORRECTION (2026-02-12):** The "99.2% at epoch 36" baseline this was compared against was NEVER iPC — it was standard ePC (see Mistake #31). The adaptive damping regression may have been partly real but was confounded by the iPC flag being newly wired up at the same time. The true baseline for standard ePC is 99.3% at epoch 44 (or 99.03% at epoch 27 with seed=0).

---

### 31. iPC Flag Was Not Wired Up — "99.2% iPC" Was Actually Standard ePC (CRITICAL)

**What happened:** At commit f417cb8 ("Add iPC method and --ipc CLI flag"):
- `--ipc` was added to argparse
- `print("Mode: iPC ...")` was added
- `ipc_train_step()` was added to epc_model.py
- **BUT the training loop was NOT updated** — it always ran standard ePC

The training loop was only updated at 7fcb4de ("Wire up iPC mode in training loop"), which added `if args.ipc:` to the batch processing code.

**The experiment that produced 99.2% was run between these commits.** It printed "Mode: IPC" but executed standard ePC. We spent an entire debugging session (~6 hours) trying to reproduce "iPC's 99.2%" — running the exact commit, testing 50 seeds, checking TF32, checking driver versions — when the result was never iPC to begin with.

**Evidence that it was standard ePC:**
1. **Speed**: 97ms/batch = standard ePC speed. Real iPC runs at 126ms (adds weight update per Newton step)
2. **Loss scale**: 290 at epoch 1 = E_local/32 (standard ePC reports `weight_loss.item()`). Real iPC reports raw energy (~6834)
3. **Convergence pattern**: Matches standard ePC's known plateau → phase transition behavior

**Impact of misdiagnosis:**
- Wasted ~6 hours debugging a "non-reproducibility crisis"
- Incorrectly blamed adaptive damping (#30), TF32, NVIDIA drivers, environment
- Added unnecessary TF32 disable code (removed)
- Created seed sweep and environment diagnostic scripts (useful but misdirected)
- Multiple incorrect entries in MEMORY.md and MISTAKES.md

**Root principle:** When a CLI flag is added to the parser but not to the execution path, it creates a silent failure — the user sees correct output messages but the feature isn't active. Always verify that a flag actually reaches the code that implements it. Add integration tests or at minimum run a smoke test with observable behavioral difference (e.g., timing or loss scale change).

**Verified working results (2026-02-12):**
- Standard ePC (seed=0): 99.03% at epoch 27 (phase transition at epoch 19)
- Standard ePC (no seed, historical): 99.3% at epoch 44
- Standard ePC (no seed, historical, mislabeled "iPC"): 99.2% at epoch 36
- iPC has NEVER been validated — it may or may not work

**Status:** FIXED (2026-02-12) — iPC disabled by default, defaults corrected

### 32. Autograd HVP Through CE + Multiple Error Nodes Produces NaN (CRITICAL)

**What happened:** CG optimizer's Hessian-vector product (HVP) produced all-NaN results, making CG unusable.

**Root cause:** PyTorch's autograd double-backward through `F.cross_entropy` (internally `log_softmax`) is numerically unstable when **multiple error leaf nodes** simultaneously participate in the computation graph. The NaN arises from the interaction of:
1. Second derivatives through the CE backward (∂²CE/∂logits² involves softmax Jacobian)
2. Multiple error paths converging at the same CE loss, creating cross-Hessian terms
3. Large intermediate values from second derivatives through Mamba3's exp(segsum(...)) chain

The Hessian itself is **mathematically finite** — finite-difference HVP confirms this (diag_hvp.py Part 8h). The NaN is purely an autograd numerical artifact.

**Diagnostic evidence (diag_hvp.py):**
- Parts 1-6: Every individual component (SSD, Mixer, Block) has valid HVP — OK
- Part 7: Per-layer error HVP in full ePC model — NaN (all errors in graph via `pce.E`)
- Part 8e: Single error through all blocks + CE — OK (only one leaf node)
- Part 8f: All errors in graph + CE — NaN (multiple leaf nodes)
- Part 8g: All errors in graph + `.sum()` loss — OK (no CE double backward)
- Part 8h: Numerical (finite-difference) HVP — OK (Hessian is finite)

**Fix:** Replaced autograd HVP (`create_graph=True` + double backward) with finite-difference HVP in `_cg_loop`: `Hd ≈ (g(e+εd) - g(e)) / ε` where ε=1e-4. This:
- Avoids `create_graph=True` entirely (also saves memory)
- Has comparable cost (1 extra fwd+bwd vs 1 expensive create_graph bwd + 1 bwd-through-bwd)
- Is provably correct for the quadratic error terms (exact for any ε)
- Matches numerical HVP ground truth for the CE terms

**Never do:** Use autograd double-backward (`create_graph=True` → `grad(grad(...))`) through cross-entropy loss when multiple leaf nodes contribute to the same loss. This is a known PyTorch numerical instability.

**Status:** FIXED (2026-02-12) — `_fd_hvp()` method + updated `_cg_loop()` in epc_model.py
