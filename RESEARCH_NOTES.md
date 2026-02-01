# Predictive Coding Research Notes

## Core Concepts

### Standard PC Architecture

**Value Nodes + Error Nodes:**
- Each traditional neuron is split into two components
- **Value nodes (v)**: Hold predictions/representations
- **Error nodes (e)**: Hold prediction errors
- Predictions flow forward (bottom-up)
- Errors flow backward (top-down)

### Two-Phase Algorithm

**Phase 1: Inference (Iterative Error Minimization)**
- Run for T_inference iterations (typically ~5L steps, where L = number of layers)
- For 7 layers: ~35 iterations
- Update value nodes based on prediction errors
- Iterate until convergence (or max iterations)

**Phase 2: Learning (Weight Updates)**
- After inference converges, update weights once
- Uses local Hebbian-like updates
- No backpropagation required

### Energy Function
- PC minimizes a free energy functional
- Energy is guaranteed to decrease (or stay constant) at each step
- Converges to local minimum or saddle point

## Scaling Challenges

### The 5-7 Layer Problem

PC networks traditionally suffer performance degradation beyond 5-7 layers due to:

1. **Exponentially imbalanced errors** between layers during weight updates
2. **Vanishing error signals** - similar to vanishing gradients in backprop
3. **Poor energy propagation** from deep to shallow layers
4. **Prediction ineffectiveness** - predictions from previous layers don't effectively guide deeper layer updates

### Solutions from Recent Research

#### μPC (Micro-PC) Method (2025)
- Enables training of 100+ layer PC networks
- Uses Depth-μP parameterization
- **Key innovation:** Scale residual connections by 1/√L
- Precision-weighted optimization of latent variables
- Zero-shot transfer of learning rates across widths and depths

#### Residual Connections in PC
- Without proper scaling, residuals can harm PC network performance
- Energy propagated by residuals reaches higher layers faster than main pathway
- Solution: Scale by 1/√L to balance energy flow

#### Better Initialization (2025)
- Depth-aware initialization schemes
- Similar to Xavier/He initialization but adapted for PC
- Critical for deep networks

## Performance Benchmarks

### MNIST Results from Literature

**3-layer networks:**
- Hidden layers: 128 or 256 neurons
- Performance comparable to backpropagation
- Typical accuracy: >95%

**Deeper networks (5-7 layers):**
- VGG5 > VGG7 > VGG9 performance trend
- Without μPC: degradation after 7 layers
- With μPC: stable up to 128 layers tested

**Network sizes used:**
- MNIST/FashionMNIST: 2-3 hidden layers, 128-256 neurons each
- CIFAR: VGG-style architectures

### Computational Cost

**Inference overhead:**
- Requires T_inference iterations per sample (typically 20-35 for 5-7 layers)
- More computationally expensive than backprop forward pass
- Can be parallelized across batch dimension
- Recent work focuses on "faster PC" via better initialization to reduce iterations

## Key Research Groups & Resources

### VERSES AI
- Led by Karl Friston (Chief Scientist)
- Focus: Scalable predictive coding and active inference
- Approach: Hyper-efficient learning (less data, compute than LLMs)
- Beyond brute-force scaling paradigm

### Academic Groups
- **Bogacz Group** (Oxford): Foundational PC theory and implementations
- **Beren Millidge**: PC approximates backprop along arbitrary computation graphs
- **Whittington & Bogacz**: Local Hebbian synaptic plasticity approximates backprop

### Key Papers by Year

**2017:**
- Bogacz: "A tutorial on the free-energy framework for modelling perception and learning"
- Whittington & Bogacz: "An approximation of the error backpropagation algorithm in a predictive coding network"

**2022:**
- Millidge et al.: "Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?"

**2025:**
- "μPC: Scaling Predictive Coding to 100+ Layer Networks"
- "Faster Predictive Coding Networks via Better Initialization"
- "Towards Scaling Deep Neural Networks with Predictive Coding: Theory and Practice"

**2026:**
- Salvatori et al.: "A survey on neuro-mimetic deep learning via predictive coding" (comprehensive review)

## Implementation Repositories

### Recommended Libraries

1. **Bogacz-Group/PredictiveCoding** (PyTorch)
   - Official Bogacz group implementation
   - Four variants: Supervised, Monte Carlo, Recurrent, Temporal
   - MNIST tutorials included
   - Most authoritative source

2. **infer-actively/pypc** (PyTorch)
   - Contributors: Alexander Tschantz, Beren Millidge
   - GPU acceleration
   - Clean interface

3. **bjornvz/PRECO** (PyTorch)
   - Based on 2024 tutorial & survey paper
   - PCNs and PCGs (Predictive Coding Graphs)
   - PyTorch-style API

4. **BerenMillidge/PredictiveCodingBackprop**
   - Demonstrates PC approximates backprop
   - Includes CNNs, LSTMs, RNNs variants
   - Research-focused

### Implementation Notes

**Standard practices:**
- Use PyTorch for GPU acceleration
- T_inference = 5L is common heuristic
- Batch processing fully supported
- Weight updates once per inference convergence

**Activation functions:**
- ReLU works well (better gradient flow)
- tanh is biologically inspired choice
- Avoid sigmoid (vanishing gradient issues)

## Connection to Our Project

### Active Inference Integration
- PC provides sensory processing and learning
- Active inference guides action selection
- Free energy principle unifies both

### Curriculum Learning Synergy
- Learning progress signals can guide PC sample selection
- Epistemic value aligns with prediction error reduction
- "Learnable" samples are in optimal prediction error range

### Prospective Learning
- PC is retrospective (minimizes past errors)
- Prospective configuration looks forward
- Combination may improve data efficiency

## Open Questions

1. **How to integrate PC with active inference motor control?**
   - PC provides prediction errors
   - Active inference converts to motor commands
   - Need smooth gradient flow from perception to action

2. **Optimal inference iterations for real-time performance?**
   - 35 iterations may be too slow for 20 FPS sensorimotor loop
   - Can we reduce to 10-15 without accuracy loss?
   - Parallel processing possibilities?

3. **Best architecture for multimodal inputs?**
   - Foveal + peripheral vision streams
   - Audio spectrograms
   - How to merge in PC framework?

4. **Memory integration?**
   - PC is feedforward/recurrent
   - Need hippocampal-like episodic memory
   - How to combine with predictive coding?

## Lessons Learned (Cross-reference with MISTAKES.md)

### What Works
- Standard PC algorithms from literature
- Proven implementations (Bogacz Group, pypc)
- μPC scaling for deep networks
- ReLU activations
- Proper initialization

### What Doesn't Work
- Custom two-compartment neuron designs
- Output clamping for pretraining
- Exponential precision scaling (10x per layer)
- CNN-PC hybrids without proper architecture
- Ignoring convergence requirements

### Best Practices
1. Start with existing proven implementations
2. Use standard architectures before customizing
3. Monitor error signal magnitudes across layers
4. Track inference convergence
5. Compare against backprop baselines
6. Document everything in MISTAKES.md

## Next Research Directions

1. **Immediate:** Test 7-layer MNIST network with μPC
2. **Short-term:** Integrate with active inference wrapper
3. **Medium-term:** Multimodal architecture (vision + audio)
4. **Long-term:** Full sensorimotor learning in game environment

---

*Last updated: 2026-02-01*
*Consult this document before making architectural decisions*
