# Architectural Lessons from Experimentation

## Lesson 1: Pairwise associations don't capture joint rules

The manifold of addition is a THREE-way relationship: `c = (a + b) mod k`.
This cannot be decomposed into pairwise relationships between a↔c and b↔c.
The pairwise matrix between a and c depends on what b is — it's not a fixed
pattern.

**Fix:** Store the full triple field at each module level. With period=10,
this is only 10³ = 1000 entries per module. Tractable.

## Lesson 2: Diffusion hurts more than it helps (so far)

Spreading activation from example points washes out the signal. After
diffusion, every phase has nonzero activation, making the field nearly
uniform. More examples + diffusion = worse performance.

**What works:** Raw Hebbian counts with NO diffusion. The pattern
emerges from accumulation of exact co-activations, not from spreading.

Diffusion might help for very sparse data (few examples, need to
interpolate). But our current diffusion is too aggressive — it spreads
uniformly rather than along the manifold. A better approach would be
anisotropic diffusion that spreads ALONG the manifold surface but not
perpendicular to it. This requires knowing the manifold shape first,
which is circular.

**Future direction:** Diffusion should happen AFTER the manifold is
partially known, to fill gaps — not during initial learning.

## Lesson 3: Cross-module scoring by product doesn't work

Multiplying per-module scores gives too much weight to noisy modules.
A high-activation phase at the tens level (from unrelated examples)
times a moderate score at the ones level beats the correct answer.

**Fix:** Sequential resolution with carry propagation. Resolve ones
first, compute carry, constrain tens, etc. This is column arithmetic —
and it works because modules at different scales are NOT independent.

## Lesson 4: The ones-module triple field learns modular addition correctly

With 100 examples and no diffusion, the ones-module triple field
shows the exact pattern: when b=4, a=3 maps to c=7; a=6 maps to c=0
(with carry); a=9 maps to c=3 (with carry). This IS modular addition,
discovered purely from Hebbian counting.

## Lesson 5: The product torus space is too large to diffuse in

The product space of 3 axes × 3 modules × 10 phases has 10^9 points
(if fully enumerated). Diffusing in this space is intractable.

**Fix:** Operate at each module level independently (10³ = 1000 points
per module). This is the key insight: the torus structure factorizes
the problem by SCALE, not by axis. Each module level has its own
compact triple field. Cross-module communication happens through carries.

## Lesson 6: More examples doesn't always mean better performance

With aggressive diffusion, 50 examples performed worse than 3.
Without diffusion, 100 examples (6/10) outperformed 50 (3/10).
The transition point depends on coverage: you need enough examples
to cover the 10×10 grid of ones-digit pairs (100 pairs for a given
operator). With fewer examples, gaps in the triple field cause wrong
answers for unseen digit combinations.

Theoretical minimum for perfect ones-module coverage: 100 examples
that cover all (a_ones, b_ones) pairs. For the tens module with
carries: more are needed because the carry depends on the ones digits.

## Lesson 7: The manifold IS the triple field pattern

We don't need to "fit" a manifold as a separate step. The triple field
itself IS the manifold — a discrete probability distribution over the
triple space at each module level. Querying the manifold = looking up
the field. The manifold shape (the diagonal stripe for addition) emerges
automatically from accumulating examples.

## Lesson 8: Mean-field factorization loses the joint structure

Treating axes independently (p(a,b,c) ≈ p(a)×p(b)×p(c)) loses the
essential joint structure. Addition is precisely the CORRELATION between
axes — the thing the factorization throws away.

The triple field preserves the joint structure within each module level.
The cross-module carries preserve the joint structure across scales.

## Lesson 9: Iterative computation reuses the same latent space

For expressions like "3 + 7 + 1 + 4", the system should NOT add more
dimensions. Instead, it should iterate: 3+7=10, 10+1=11, 11+4=15.
The same 3D latent space (a, b, c) is reused at each step, feeding
the result back as the first operand of the next step.

This is how humans do it. This is compositional reuse of the manifold.
The CatPlan planner handles the decomposition into steps; the manifold
handles each individual step.
