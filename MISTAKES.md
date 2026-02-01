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
