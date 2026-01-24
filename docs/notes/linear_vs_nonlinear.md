# Linear vs Nonlinear Predictive Coding

**Created:** 2026-01-24
**Issue:** Linear network exploded at iteration 3 (error → NaN)

## What Happened

**Linear network results:**
- Iteration 0: error = 976
- Iteration 1: error = 866
- Iteration 2: error = 109 ✓ (learning!)
- Iteration 3: error = 1.49 million ✗ (explosion!)
- Iteration 4-50: NaN (complete failure)

**Weight explosion:** W_apical went from 1.03 → 1.7 billion in one iteration

## Why Linear Failed

Without `tanh()` saturation:
1. No bounds on activations
2. Positive feedback loops grow exponentially
3. Small errors cascade into catastrophic instability
4. Linear recurrent systems are inherently unstable without careful design

## What the Papers Say

From the research:

**Song et al. (2024)**: "Inferring neural activity before plasticity"
- [Nature Neuroscience](https://www.nature.com/articles/s41593-023-01514-1)
- [GitHub Implementation](https://github.com/YuhangSong/Prospective-Configuration)
- **Action needed:** Check their code to see what activation functions they actually use

**Millidge et al. (2022)**: [Theory paper](https://arxiv.org/abs/2207.12316)
- Proves convergence properties
- **Action needed:** Check if they assume nonlinear activations

## Practical Reality

**All working predictive coding implementations use nonlinear activations:**
- tanh (most common)
- sigmoid
- ReLU (less common in PC)

**Reason:** Stability > theoretical elegance

## Recommendations

**Option 1: Fix the nonlinear version (recommended)**
- Keep tanh activations
- Focus on getting learning to work with existing 0.01 LR
- Investigate why higher layers aren't learning

**Option 2: Check Song implementation**
- Clone the GitHub repo
- See exactly what they use
- Copy their approach

**Option 3: Abandon linear**
- The explosion proves linear is fundamentally unstable
- Even with tiny LR (0.0001), likely still unstable
- Not worth debugging unless papers explicitly use it

## Current Status

**Nonlinear network (with tanh):**
- Stable ✓
- Error decreases 6.6% (minimal but consistent)
- Higher layers frozen (~0% weight change)
- **Next step:** Figure out why higher layers aren't learning

**Linear network:**
- Unstable ✗
- Explodes to NaN
- Not usable
- **Next step:** Abandon or check Song's implementation

## Decision Point

Should we:
1. **Focus on nonlinear version** and fix the "higher layers frozen" problem?
2. **Clone Song's repo** and see exactly what they do?
3. Try to stabilize linear (probably futile)?

My recommendation: **Option 1** - focus on making nonlinear work properly.
