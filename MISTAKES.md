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

## Update Log

- 2026-02-01 (initial): Initial file created with 6 major mistakes catalogued
- 2026-02-01 (implementation): Fixed mistake #2 (custom neurons) with standard PCLayer implementation
- 2026-02-01 (documentation): Fixed mistake #6 (context loss) with persistent documentation system
- 2026-02-01 (code complete): Added successful implementation section with all PC network code
- 2026-02-01 (CRITICAL FIX): Found and fixed mistake #8 - optimizer was conflating value nodes with network parameters, causing complete learning failure. Training was stuck at 8% (random guessing). Fixed by separating parameter lists.
- 2026-02-01 (THE REAL BUG): Mistake #8 fix had ZERO effect. Debug script showed weights never changed. Found actual bug (mistake #9): mu.detach() broke computational graph so NO gradients reached weights. Removed detach calls. This is the real fix.
- 2026-02-01 (BAYESIAN PC BUGS): Fixed mistakes #10 and #11 - Bayesian PC had NaN catastrophe from variance collapse (min_variance=1e-6 too low, no max bound) and broke experimental control by changing from 7 to 3 layers. Fixed by setting min_variance=0.01, max_variance=10.0, and restoring 7-layer architecture.
- 2026-02-01 (FUNDAMENTAL ERROR): Identified mistake #12 - entire BayesianPC implementation is conceptually wrong. Put Bayesian posteriors over VALUE NODES (hidden states) instead of WEIGHTS (parameters). User provided paper (Tschantz et al. 2025) showing BPC uses Matrix Normal Wishart weight posteriors with closed-form Hebbian updates, not distributions over hidden states. All code in experiments/BayesianPC/ must be archived and rewritten from scratch following Algorithm 1 from the paper.
