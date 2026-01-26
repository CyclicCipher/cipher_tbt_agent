# Causal Learning in Predictive Coding Networks

## Overview

This document explores how causal learning mechanisms from neuroscience can inform the design of predictive coding networks. The goal is to move beyond correlation-based learning toward genuine causal understanding.

---

## 1. Temporal Causality: STDP-Inspired Learning

### Biological Inspiration
**Spike-Timing Dependent Plasticity (STDP)** enforces a fundamental principle: *cause must precede effect*.

- If neuron A fires before neuron B, strengthen A→B connection
- If neuron B fires before neuron A, weaken A→B connection
- This creates asymmetric learning based on temporal ordering

### Application to Predictive Coding

Our network already has temporal convolutions and temporal state. We can extend this:

**Current State:**
- Temporal convolutions capture local temporal patterns
- Temporal state (`state_t`, `state_t-1`) tracks recent history
- But no explicit causal ordering constraint

**Proposed Extension:**
```python
# Weight update with temporal causality
if prediction_precedes_observation(t_pred, t_obs):
    # Strengthen: prediction correctly preceded observation
    weight_update = +lr * error * activation
elif observation_precedes_prediction(t_obs, t_pred):
    # Weaken: prediction came after observation (non-causal)
    weight_update = -lr * error * activation
```

**Implementation Strategy:**
1. Track *when* each layer makes a prediction (timestep `t_pred`)
2. Track *when* sensory evidence arrives (timestep `t_obs`)
3. Only strengthen weights if `t_pred < t_obs` (prediction precedes observation)
4. Weaken weights if `t_pred > t_obs` (prediction follows observation)

This enforces causal direction even in non-spiking networks.

---

## 2. Self vs. Environment Causation

### The Agency Problem

The brain distinguishes:
- **Self-caused outcomes**: "I moved my hand, the object moved"
- **Environment-caused outcomes**: "The object moved on its own"

This is critical for:
- Credit assignment (what did *I* cause?)
- Surprise detection (unexpected external events)
- Learning control policies (what can I influence?)

### Neuroscience Evidence

**Efference copy / corollary discharge:**
- Motor commands generate predictions of sensory consequences
- If prediction matches observation → self-caused
- If observation differs from prediction → environment-caused

**Brain regions:**
- Cerebellum: forward models of motor consequences
- Parietal cortex: distinguishes self vs. other motion
- Comparator circuits: prediction error when self-prediction violated

### Application to Predictive Coding

**Current Architecture:**
- Motor subnet at position 0 (generates actions)
- Vision subnet at position 0 (observes consequences)
- Association subnet at position 1 (integrates)

**Proposed Extension:**
Add **agency detection** through motor prediction error:

```python
# During action execution
motor_prediction = predict_sensory_outcome(motor_command)
actual_observation = vision_subnet.get_state()
agency_error = actual_observation - motor_prediction

if agency_error.norm() < threshold:
    # Outcome matches motor prediction → self-caused
    self_attribution = 1.0
else:
    # Outcome doesn't match motor prediction → environment-caused
    self_attribution = 0.0

# Use for credit assignment
weight_update_motor = self_attribution * lr * error
weight_update_environment = (1 - self_attribution) * lr * error
```

**Key Insight:**
- When `self_attribution = 1.0`: update motor-to-outcome weights (I caused this)
- When `self_attribution = 0.0`: update environment-to-outcome weights (world caused this)

This prevents the network from learning spurious motor→outcome mappings for events it doesn't actually control.

---

## 3. Bayesian Causal Inference: Learning Hidden Causal Graphs

### The Problem

Correlation ≠ Causation. The brain doesn't just learn P(B|A), it learns:
- Does A cause B?
- Does B cause A?
- Does hidden variable C cause both A and B?

### Neuroscience Evidence

**Brain regions for causal inference:**
- Prefrontal cortex: activated during causal reasoning tasks
- Hippocampus: structural learning, relational memory
- Basal ganglia: model-based reinforcement learning (causal world models)

**Behavioral evidence:**
- Humans spontaneously infer hidden causes
- Children as young as 3 years distinguish causation from correlation
- Causal reasoning emerges before formal logic

### Bayesian Causal Inference

The brain appears to compute:

```
P(A causes B | observations) ∝ P(observations | A causes B) × P(A causes B)
```

This requires:
1. **Generative model**: simulate what would happen if A causes B
2. **Likelihood**: compare simulation to observations
3. **Prior**: structural assumptions about causal graphs

**Hidden cause detection:**
If A and B correlate, but intervention on A doesn't change B:
→ Infer hidden cause C influencing both

### Application to Predictive Coding

**Challenge:** Predictive coding naturally learns correlations through prediction errors. How do we learn *causal structure*?

**Proposed Approach: Interventional Learning**

```python
# Standard predictive coding (observational)
prediction = f(A)
error = B - prediction
# This learns P(B|A) but not whether A causes B

# Interventional learning (causal)
# Simulate: "What if I change A?"
do(A = new_value)  # Intervention operator
predicted_B = f(new_value)
actual_B = observe()

if predicted_B ≈ actual_B:
    # A causally influences B
    causal_weight[A→B] += lr
else:
    # A doesn't cause B (correlation was spurious)
    causal_weight[A→B] -= lr
```

**Implementation Strategy:**
1. During exploration, randomly intervene on variables
2. Compare predicted vs. actual outcomes under intervention
3. Build causal adjacency matrix: `causal_weight[i][j]` = strength of i→j causal link
4. Use causal graph for planning and reasoning

**Hidden cause detection:**
If A and B correlate but interventions reveal no direct causal link:
→ Create latent variable Z
→ Learn Z→A and Z→B connections

This is similar to hierarchical predictive coding, where higher layers represent hidden causes.

---

## 4. Temporal Windows for Causal Relationships

### The Problem

Not all temporal relationships are causal:
- If A precedes B by 1 second → likely causal
- If A precedes B by 1 hour → probably coincidence
- If A precedes B by 1 month → almost certainly spurious

### Neuroscience Evidence

**Temporal credit assignment problem:**
- Dopamine reward prediction errors decay over ~1 second
- Hippocampal replay extends credit assignment window to minutes/hours
- Cortical learning operates at multiple timescales (fast/slow synapses)

**Biological solution: Eligibility traces**
- Synapses maintain "eligibility" for plasticity
- Eligibility decays exponentially with time
- If reward arrives before decay, synapse updates

### Application to Predictive Coding

**Current limitation:**
Our network treats all temporal relationships equally. A prediction 1 second ago and 1 minute ago get the same weight update.

**Proposed: Temporal Discounting of Causal Credit**

```python
def temporal_discount(delta_t, tau=1.0):
    """
    Discount causal credit based on time delay.

    Args:
        delta_t: Time between cause and effect
        tau: Characteristic timescale (decay constant)

    Returns:
        Discount factor in [0, 1]
    """
    return np.exp(-delta_t / tau)

# Weight update with temporal discounting
for event_t in history:
    delta_t = current_time - event_t.time
    discount = temporal_discount(delta_t, tau=causal_window)

    weight_update = discount * lr * error * activation
```

**Multiple timescale learning:**
```python
# Fast timescale: immediate causation (reflexes, tracking)
fast_update = temporal_discount(delta_t, tau=0.1)  # 100ms window

# Medium timescale: action-outcome learning
medium_update = temporal_discount(delta_t, tau=1.0)  # 1s window

# Slow timescale: contextual learning
slow_update = temporal_discount(delta_t, tau=10.0)  # 10s window

total_update = fast_update + medium_update + slow_update
```

**Adaptive causal window:**
Learn the appropriate temporal window from data:
- If interventions reveal causal relationship at delay Δt, expand window
- If delays beyond Δt show no causal relationship, shrink window

---

## 5. Structural Causal Learning: How the Neocortex Might Build Causal Graphs

### The Mystery

The brain can learn:
- "Flipping the light switch causes the light to turn on"
- "Pressing the brake causes the car to slow down"
- "Eating causes satiation"

This requires learning **graph structure**, not just weights.

### Hypothesized Mechanisms

**1. Prediction Error Gradients**
When A changes:
- Compute prediction errors in all downstream variables
- Variables with large errors are likely caused by A
- Build directed edges A→B for high-error variables B

**2. Conditional Independence Testing**
If A causes B:
- P(B|A) ≠ P(B)  (B depends on A)
- P(B|A, other) = P(B|A)  (A d-separates B from others)

Network could implement this through:
- Clamping A, observing change in B (dependence)
- Clamping intermediate C, checking if A→B link weakens (d-separation)

**3. Hierarchical Abstraction**
Lower layers learn concrete causal links:
- "Hand moves → object moves"

Higher layers learn abstract causal principles:
- "Agent actions → environment changes"
- "Physical contact → force transfer"

**4. Counterfactual Simulation**
To test if A causes B:
- Imagine: "What if A hadn't happened?"
- Simulate B under counterfactual A'
- If B changes → A causes B

### Implementation in Predictive Coding

**Proposed: Graph Discovery Layer**

```python
class CausalGraphLayer:
    """
    Learns causal adjacency matrix from observations and interventions.
    """
    def __init__(self, num_variables):
        # Adjacency matrix: causal_graph[i][j] = strength of i→j causal link
        self.causal_graph = torch.zeros(num_variables, num_variables)

    def update_graph(self, observations, interventions=None):
        """
        Update causal graph based on:
        - Observational data (correlations)
        - Interventional data (causation)
        """
        # Observational: learn correlations
        correlations = compute_correlation_matrix(observations)

        if interventions is not None:
            # Interventional: identify true causal links
            for intervention in interventions:
                var_i = intervention.variable
                predicted_effects = self.causal_graph[var_i] @ observations
                actual_effects = intervention.outcomes

                # Increase weight for variables actually affected
                effect_error = (actual_effects - predicted_effects) ** 2
                self.causal_graph[var_i] += -grad(effect_error)
        else:
            # No interventions: use observational heuristics
            # (can only learn correlations, not causation)
            self.causal_graph += lr * correlations

    def predict_intervention(self, var_i, new_value):
        """
        Predict outcome of intervening on variable i.
        """
        # Causal descendants of var_i
        effects = self.causal_graph[var_i]
        predicted_outcomes = new_value * effects
        return predicted_outcomes
```

**Integration with Predictive Coding:**
- Each layer maintains local causal graph
- Bottom layers: concrete object-level causation
- Top layers: abstract principle-level causation
- Causal graphs guide top-down predictions

---

## 6. Open Research Questions

### Theoretical
1. **How to unify predictive coding with causal inference?**
   - Predictive coding minimizes prediction errors
   - Causal inference maximizes explanatory power
   - Are these compatible objectives?

2. **What is the neural representation of causal graphs?**
   - Adjacency matrices in weight space?
   - Specialized "causal" neurons?
   - Temporal binding of cause-effect pairs?

3. **How does the brain discover hidden causes?**
   - Latent variable models?
   - Hierarchical Bayesian inference?
   - Structural EM algorithm?

### Practical
1. **Can we learn causal graphs from observation alone?**
   - Or do we need interventions/exploration?
   - How much data is required?

2. **How to handle partial observability?**
   - Real environments have hidden variables
   - Can the network infer their existence and causal role?

3. **How to scale causal learning to high-dimensional spaces?**
   - Full causal graphs are O(N²)
   - Need sparse structure learning

### Implementation
1. **Temporal credit assignment in recurrent networks**
   - How far back to propagate causal credit?
   - Eligibility traces? BPTT? e-prop?

2. **Agency detection in complex tasks**
   - Multiple agents, delayed consequences
   - Partial control (stochastic outcomes)

3. **Causal graph pruning**
   - Start with fully connected → prune weak links?
   - Or start sparse → add links as needed?

---

## 7. Immediate Next Steps

### Short-term (Current System)
1. **Add temporal discounting to weight updates**
   - Implement exponential decay based on prediction-observation delay
   - Test multiple timescales (fast/medium/slow)

2. **Implement agency detection**
   - Motor subnet predicts sensory outcomes
   - Compare prediction to observation
   - Separate self-caused vs. environment-caused learning

3. **Track temporal ordering**
   - Log when predictions are made
   - Log when observations arrive
   - Enforce STDP-like asymmetry

### Medium-term (Causal Extensions)
1. **Interventional learning mode**
   - Exploration: randomly perturb motor outputs
   - Observe causal consequences
   - Build causal graph of controllable variables

2. **Hidden cause detection**
   - Identify correlated variables with no direct causal link
   - Create latent variables to explain correlation
   - Similar to hierarchical PC, but explicit causal semantics

3. **Counterfactual simulation**
   - "What if I had acted differently?"
   - Use causal graph to simulate alternate timelines
   - Improve planning and credit assignment

### Long-term (Structural Learning)
1. **Dynamic causal graph**
   - Graph structure changes with context
   - "Rain causes wetness" vs. "Sprinkler causes wetness"
   - Context-dependent causal models

2. **Transfer learning via causal abstraction**
   - Learn high-level causal principles
   - "Contact → force transfer"
   - Apply to new objects/situations

3. **Multi-agent causal reasoning**
   - Distinguish own actions, other agents, environment
   - Cooperative/competitive causal models
   - Theory of mind through causal graphs

---

## 8. Relevant Literature

### Neuroscience
- **STDP**: Bi & Poo (1998), "Synaptic modifications in cultured hippocampal neurons"
- **Efference copy**: Wolpert & Flanagan (2001), "Motor prediction"
- **Causal reasoning**: Saxe et al. (2005), "Brain regions for causal reasoning"
- **Temporal credit assignment**: Gershman et al. (2014), "Dopamine and temporal credit assignment"

### Machine Learning
- **Causal inference**: Pearl (2009), "Causality"
- **Causal discovery**: Spirtes et al. (2000), "Causation, Prediction, and Search"
- **Interventional learning**: Peters et al. (2017), "Elements of Causal Inference"
- **Counterfactual reasoning**: Buesing et al. (2018), "Causal reasoning from meta-reinforcement learning"

### Predictive Coding
- **Friston's free energy**: Friston (2010), "The free-energy principle"
- **Active inference**: Friston et al. (2017), "Active inference and agency"
- **Temporal predictive coding**: Rao & Ballard (1999), "Predictive coding in the visual cortex"

---

## 9. Summary

Causal learning is essential for:
- **Credit assignment**: What caused this outcome?
- **Planning**: What will happen if I act?
- **Transfer**: Apply learned knowledge to new situations
- **Explanation**: Why did this happen?

Our predictive coding network can be extended with:
1. **Temporal causality** (STDP-inspired)
2. **Agency detection** (self vs. environment)
3. **Causal graph learning** (Bayesian structure learning)
4. **Temporal windows** (discounting distant causes)

These extensions move beyond correlation-based learning toward genuine causal understanding, bringing the network closer to biological intelligence.
