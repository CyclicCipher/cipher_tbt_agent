# Mistakes Tracking System

## HOW TO USE THIS FILE

### Before Writing Code:
1. **READ THIS FILE** - Check if similar mistakes have been made before
2. **APPLY THE FIXES** - Use documented solutions to avoid repeating errors
3. **CHECK ARCHITECTURAL PATTERNS** - Ensure your approach aligns with known working patterns

### After Testing/Analysis:
1. **REVIEW RESULTS** - Compare failures against known mistakes
2. **UPDATE THIS FILE** - Add new mistakes with clear descriptions and fixes
3. **DO NOT DUPLICATE** - Search before adding; update existing entries if needed

### When Adding a Mistake:
- **TITLE**: Clear, searchable description
- **CONTEXT**: When/where it occurred
- **MISTAKE**: What went wrong
- **FIX**: How it was resolved
- **LESSON**: Why it happened and how to prevent it

---

## CODING MISTAKES

### 1. Incorrect Gradient Sign in Precision-Weighted Updates
**Context**: `precision_weighted_pc.py` - prediction error computation
**Mistake**: Used `delta_prob = logits - target_probs` which gives positive gradients when predictions are too high
**Fix**: Use `delta_prob = target_probs - logits` to get correct gradient direction (positive when we need to increase)
**Lesson**: Always verify gradient signs match mathematical definitions. Error = target - prediction for proper gradient descent.

### 2. Log-Space Precision Leading to Extreme Weighting
**Context**: `precision_weighted_pc.py` - precision computation
**Mistake**: Applied `exp(log_precision)` which creates extreme multiplicative weights (e.g., 1e-43 to 1e0)
**Fix**: Work directly with variances, use `1/variance` for precision weighting with proper normalization
**Lesson**: Exponential transformations of precision create numerical instability. Use linear variance space with normalization.

### 3. Broadcasting Errors with Precision Shapes
**Context**: Precision weighting in update rules
**Mistake**: Shape mismatches between `precision` [batch, seq, vocab] and `delta_prob` cause silent broadcasting errors
**Fix**: Always verify shapes match exactly before element-wise operations. Add assertions: `assert precision.shape == delta_prob.shape`
**Lesson**: PyTorch's broadcasting is helpful but can mask critical shape errors. Add explicit shape checks.

### 4. Confusion Between Loss and Update Gradients
**Context**: Various PC implementations
**Mistake**: Mixing up the sign conventions between loss gradients (for backprop) and PC update gradients
**Fix**: Clearly separate: Loss gradients minimize loss; PC updates minimize prediction error directly
**Lesson**: Document whether each gradient is for loss minimization or direct error minimization.

### 5. Not Checking for NaN/Inf Before They Propagate
**Context**: Multiple training runs
**Mistake**: Allowing NaN values to propagate through multiple layers before detection
**Fix**: Add `assert not torch.isnan(x).any()` after critical operations (precision computation, divisions, log operations)
**Lesson**: Detect numerical issues immediately at their source, not downstream.

---

## ARCHITECTURAL MISTAKES

### 1. Using Arbitrary Precision Constants Without Theoretical Justification
**Context**: Multiple attempts with `[1, 10, 100]` type constants
**Mistake**: Using hand-picked precision weights without grounding in actual prediction uncertainty
**Fix**: Compute precision from actual variance of predictions: `variance = (probs * (1 - probs)).sum(dim=-1, keepdim=True)` or similar
**Lesson**: Precision must reflect actual uncertainty in the model. Arbitrary constants are not theoretically grounded.

### 2. Ignoring Computational Cost of Variance Calculations
**Context**: Precision-weighted PC implementation
**Mistake**: Assuming variance/precision computation is expensive without measuring
**Fix**: **RESOLVED** - EMA variance computation is very cheap (O(n) variance + O(1) EMA update). Not a bottleneck.
**Lesson**: Don't assume computational costs - measure first. EMA is an efficient approach for running statistics.
**Reference**: categorical_network.py:184-198 uses efficient EMA approach

### 3. Misunderstanding VERSES Precision Weighting Approach
**Context**: Precision-weighted PC implementation
**Mistake**: Thinking we need to compute precision from variance when VERSES uses fixed schedules
**Fix**: **RESOLVED** - VERSES uses FIXED precision values (1.0, 10.0, 100.0) for layers 0, 1, 2 to prevent error decay in deep networks. Can also use computed precision, but fixed schedules work well.
**Lesson**: Read the actual papers carefully. "Towards the Training of Deeper PC Networks" (arXiv:2506.23800) shows fixed precision schedules are effective.
**Reference**: VERSES "spiking precision" applies large precisions to boost errors forward in deep networks

### 4. Assuming KL Divergence is Required for Precision Weighting
**Context**: Theoretical understanding of PC
**Mistake**: Thinking precision weighting requires computing KL divergence and full free energy
**Fix**: **RESOLVED** - Simple precision-weighted prediction error IS the correct implementation. Free energy provides theoretical justification, but practical implementations minimize precision-weighted squared errors.
**Lesson**: Theoretical formulation (free energy) ≠ implementation details. Precision-weighted errors are sufficient.
**Reference**: Friston's canonical neural networks paper confirms this approach

### 4. Making Multiple Changes Simultaneously
**Context**: Throughout development
**Mistake**: Changing gradient signs, precision computation, and weighting schemes all at once
**Fix**: Make ONE change at a time, test, verify, then proceed
**Lesson**: Impossible to debug when multiple changes interact. Isolate variables.

### 5. Not Maintaining a Baseline for Comparison
**Context**: Testing various PC implementations
**Mistake**: Not keeping a simple, working baseline to compare against
**Fix**: Always maintain a minimal working version (even if imperfect) to validate improvements against
**Lesson**: You can't measure progress without a reference point.

### 6. Not Testing Simple Cases First
**Context**: Throughout development
**Mistake**: Testing complex scenarios before verifying basic functionality works
**Fix**: Always test with simple, known-working cases first (e.g., single layer, single sample, known gradient)
**Lesson**: Build confidence incrementally. If simple cases fail, complex ones definitely will.

### 7. Implementing Theory Without Reading the Actual Papers
**Context**: Precision weighting implementation
**Mistake**: Assuming how precision weighting works instead of reading VERSES/Friston papers
**Fix**: **ALWAYS** read the original papers before implementing. VERSES uses fixed schedules, not computed variance!
**Lesson**: Academic papers contain crucial implementation details. Theory ≠ assumptions.

---

## RESOLVED INVESTIGATIONS

### 1. Precision Computation Efficiency ✓
**Answer**: EMA variance computation is very efficient and NOT a bottleneck
- EMA variance: O(n) variance + O(1) EMA update per layer
- Current implementation (categorical_network.py:184-198) is optimal
- Memory: One scalar per layer for running variance

### 2. VERSES Approach to Precision ✓
**Answer**: VERSES uses FIXED precision schedules, not computed variance
- "Spiking Precision": Large fixed precisions boost errors in deep layers
- Prevents exponential error decay in deep PC networks
- Can use [1.0, 10.0, 100.0] for layers 0, 1, 2
- Reference: "Towards the Training of Deeper PC Networks" (arXiv:2506.23800)

### 3. KL Divergence and Free Energy ✓
**Answer**: NO KL divergence needed for basic PC learning
- Precision-weighted prediction errors are sufficient
- Free energy provides theoretical justification only
- KL divergence only needed for: active inference, model selection, explicit probabilistic inference
- Current implementation is theoretically correct

### 4. Categorical Distribution Precision ✓
**Answer**: Multiple valid approaches
- **Entropy-based** (theoretically cleanest): `precision = 1 / (entropy + ε)`
- **Variance-based** (current): `precision = 1 / (variance + ε)`
- **Fixed schedules** (VERSES): [1.0, 10.0, 100.0] for increasing depth
- All are valid; fixed schedules may be most efficient and effective

---

## INVESTIGATION QUEUE

### Current Questions to Resolve:

1. **Why isn't the current implementation working?**
   - Is the gradient sign correct? (target - prediction vs prediction - target)
   - Are the shapes broadcasting correctly?
   - Are precision values reasonable or creating numerical issues?

---

## NEXT STEPS PROTOCOL

Before implementing any changes:
1. ✅ Check this file for related mistakes
2. ✅ Read relevant sections of codebase
3. ✅ Make ONE change at a time
4. ✅ Test immediately
5. ✅ Update this file with results
