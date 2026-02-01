# Why This Implementation Is INCORRECT

## DO NOT USE THIS CODE

This folder contains an **architecturally wrong** implementation of Bayesian Predictive Coding that was archived on 2026-02-01.

## The Fundamental Error (Mistake #12)

**What this implementation did:**
- Put posterior distributions (mean + variance) over **value nodes** (hidden states z)
- Kept weights as point estimates
- Used gradient descent to optimize variance parameters
- Added KL divergence term on value node distributions

**Why it's completely wrong:**
1. **Value nodes are ephemeral** - they're optimized fresh every forward pass for the current input
   - They represent beliefs about the CURRENT observation, not accumulated knowledge
   - Making them Bayesian doesn't capture epistemic uncertainty
   - They reset every batch

2. **Weights accumulate knowledge** - they're what should be Bayesian
   - Epistemic uncertainty is about what we know about the parameters
   - Weight posteriors decrease in uncertainty as we see more data
   - This is what Bayesian deep learning means

## What Bayesian PC Actually Does

From **"Bayesian Predictive Coding"** by Tschantz et al. (2025), arXiv:2503.24016:

- **Value nodes:** MAP estimates (point values), optimized during inference via gradient descent
- **Weights:** Matrix Normal Wishart posterior distributions q(W, Σ | M, V, Ψ, ν)
- **Learning:** Closed-form Bayesian updates using conjugate priors (Equation 7)
- **Weight updates:** Hebbian function of pre/post-synaptic activity
- **Architecture:** Weights OUTSIDE activation function: z = W·f(z_{l-1})

## Results From This Wrong Implementation

- loss=nan, acc=9.80% (random guessing, worse than random)
- Complete training failure
- Free energy diverging instead of converging
- Even after "fixing" NaN bugs (#10) and architecture (#11), still fundamentally wrong

## User Feedback

> "There are still some glaring problems in your code. In any case, I don't think that your assumptions going into the code were correct. Remember how I said 'I don't buy this, but I'll humor it?'"

Then provided the BPC paper showing the correct approach.

## Files in This Archive

- `bayesian_pc_layer.py` - WRONG: Bayesian value nodes, point estimate weights
- `bayesian_pc_trainer.py` - Uses gradient descent (not closed-form updates)
- `train_mnist_bayesian.py` - Training script for wrong implementation
- `README.md` - Original description (now known to be wrong)

## What To Do Instead

See the correct implementation in `experiments/BayesianPC/` which implements Algorithm 1 from Tschantz et al. (2025) with:
- Matrix Normal Wishart weight posteriors
- Closed-form Hebbian weight updates (Equation 7)
- MAP estimates for value nodes
- Weights outside activation function

## Lesson Learned

**Don't assume** what "Bayesian X" means without reading the literature.
- I assumed: make the **inference** Bayesian (distributions over hidden states)
- Actually: make the **learning** Bayesian (distributions over weights)

This is the difference between Bayesian inference and Bayesian learning.

---

**Archived:** 2026-02-01
**Reason:** Fundamental conceptual error (Mistake #12)
**See:** MISTAKES.md for full details
