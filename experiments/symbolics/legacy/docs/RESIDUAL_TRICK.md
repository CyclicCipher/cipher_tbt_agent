# Why the residual trick stops at the outermost sum

A theory note (2026-10-26). The user asked for this to be worked out, because understanding it might matter for
machine learning beyond this folder. The established parts are marked as such; hypotheses are labelled.

## 1. The trick

Suppose the target is a sum of parts: y = f₁(x) + f₂(x) + … + f_K(x). Hold every part but one fixed. The best fk is
then the best fit to the partial residual

    r_k = y − Σ_{j≠k} f_j(x),

because ‖y − Σ_j f_j‖² = ‖r_k − f_k‖². The subproblem for f_k is an ordinary regression on its own target. It doesn't
need to know how the other parts were built, only their sum.

So the parts can be found one at a time:
- **Cost:** with B candidate forms per part, a joint search costs B^K, and the residual trick about K·B per sweep.
- **The same idea under four names** (all established): backfitting for additive models (Breiman & Friedman 1985;
  Hastie & Tibshirani 1990); boosting as greedy fitting of residuals (Friedman 2001); matching pursuit (Mallat & Zhang
  1993); and the symbolic ResNet in this folder, whose lesson was "O(B^d) becomes O(d·B)".
- **Its known weakness:** a greedy choice that only works together with a later one is missed. The subset-sum trap
  and diagnostic D2 showed this. Refitting and beams soften it without removing it.

## 2. Why it works: the algebra of +

Two properties of addition carry the trick:
1. **It can be undone.** Subtracting the other parts recovers exactly the target for the remaining one.
2. **Order doesn't matter.** The residual for f_k is the same however the other parts are arranged, so each part has
   one well-defined target.

Together, (ℝ, +) is an abelian (commutative) group, and the residual is just the inverse of the other parts'
contribution.

The trick transfers to any operation that becomes + in some coordinates:
- **Products of positive values:** log turns × into +, which is why fitting log y finds power laws.
- **Convolution:** the Fourier transform turns it into a product, and log then into a sum.
- **Linear dynamics in suitable observables:** the Koopman view.

The general statement (established mathematics): a connected commutative Lie group is, up to isomorphism, a product
of lines and circles, so its operation becomes addition in suitable coordinates. **Commutative combination is
exactly the case where a global "residual" exists.**

## 3. Why it fails: composition

Now suppose y = f(g(x)). Two subproblems arise, and they are not alike:
- **The outer function, given the inner:** fitting f on the pairs (g(xᵢ), yᵢ) is an ordinary 1-D regression. Easy.
- **The inner function, given the outer:** fitting g needs a target for g(x), and that target is f⁻¹(y).
  - If f is invertible on the data's range, f⁻¹(y) is a plain target, and g is again a regression. This is target
    propagation (Bengio 2014; Lee et al. 2015).
  - If f is many-to-one (x², sin, a saturation), f⁻¹(y) is a set: each point has several candidate targets, and
    choosing among them is combinatorial. With k such layers, up to (branches)^k combinations. Here exponential search
    genuinely enters.
  - Most of all, f is unknown while g is being searched, and vice versa. The two searches are coupled.

The underlying reason: composition is not commutative, f∘g ≠ g∘f, even among invertible maps, so there is no
coordinate system in which composition becomes addition globally. A sum has a residual for every part. A composition
has one only for its outermost layer, and only if that layer is invertible.

## 3b. Where exactly composition blocks the trick (added 2026-10-26)

"Many-to-one" here means many inputs give the same output: J₀ hits each value at many arguments, |x| at both x and
−x. Its inverse is then one-to-many, so the target for the inner function is a set, not a value.

**Not every composition blocks the trick.** z = 2x(y + 7) looks like a product of two parts, but multiplication
distributes over addition: z = 2xy + 14x, a sum of two terms. With x·y available as a term, the residual trick finds
it in two steps. With only single-input terms, f(x) + g(y), it can't: no sum of such terms makes xy. So "the trick
can't do products" is really "the vocabulary has no product terms". The trick decomposes the outer sum; what it
reaches depends on the terms it is allowed.

**Which outer functions let the trick through.** A composition f(g + h) stays decomposable when f(a + b) can be
written with finitely many functions of a and of b. Weierstrass's theorem on algebraic addition theorems (established)
says the functions with such a formula are exactly the rational functions, the rational functions of e^(kz) (which
covers sin, cos and tanh), and the elliptic functions. Outside them, f(g + h) has no finite split. The trick still gets
through if f is invertible over the data's range (the inversion route of §5).

**J₀, the Bessel function of the first kind of order zero, as the worked example.** Physically it is the radial
shape of a circular drum's fundamental vibration; it also appears in heat flow in cylinders, waves in round
waveguides, and diffraction through round holes. Three equivalent definitions:
- the solution of x²y″ + xy′ + x²y = 0 that is finite at 0;
- the series Σₖ (−1)ᵏ (x/2)²ᵏ / (k!)²;
- the integral (1/π) ∫₀^π cos(x sin θ) dθ.

It behaves like a decaying cosine: J₀(0) = 1, and J₀(x) ≈ √(2/(πx)) cos(x − π/4) for large x. Its zeros (2.405,
5.520, 8.654, …) are the drum's nodal circles. It blocks the trick in two ways:
- **It is many-to-one.** It oscillates, so its inverse gives one candidate argument per oscillation.
- **Its addition theorem is infinite.** Neumann's theorem, J₀(a + b) = J₀(a)J₀(b) + 2 Σ_{n≥1} (−1)ⁿ Jₙ(a)Jₙ(b), brings in
  every order Jₙ. It truncates well only for small arguments, where Jₙ(x) ≈ (x/2)ⁿ/n!.

Hardness also depends on the data's range. Up to its first minimum (argument 0 to 3.83), J₀ is monotone, so it is
invertible there, and inversion works.

**The hard cases, by what blocks the trick:**

| kind | examples | what blocks the residual trick | cost of the way out |
|---|---|---|---|
| many-to-one, no finite addition formula | J₀(x + y) or the Airy function Ai(x + y) over several oscillations; Γ(x + y) around its minimum near 1.46 | inverting gives a set of targets per point; no finite expansion | branch choice per point; grows with the number of oscillations |
| piecewise, with finitely many branches | \|x − y\|; max(x, y); (x + y) mod 1; compositions of ReLUs | the correct sum differs from region to region | region choice; doubles with each nested layer, so 2ᵏ for depth k: the "exponential with low N" case of `OPEN_PROBLEMS.md` |
| a finite expansion, but exponentially long | (x + y + 1)²⁰ (231 terms); the logistic map iterated k times, xₖ₊₁ = r·xₖ(1 − xₖ), a polynomial of degree 2ᵏ | decomposable in principle, but the number of terms explodes with depth | linear in terms, exponential in depth; a compact composite atom like (a·x + b)²⁰, or inversion, avoids it |

The mod-1 row is modular arithmetic: (a + b) mod 1 = (a mod 1 + b mod 1) mod 1 is an addition rule with a carry,
so two branches. The ReLU row is the source of a deep network's exponentially many linear regions.

**These look hard but are not:**
- **A resonance curve**, A = 1/√((ω₀² − ω²)² + γ²ω²): inversion gives A⁻² = (ω₀² − ω²)² + γ²ω², a sum of terms.
- **A Gaussian of a sum**, e^(−(x + y)²): its log is −x² − 2xy − y².
- **sinc(x + y)**: multiplied through by (x + y) it is a finite sum of products, by angle addition.
- **erf of anything**: erf is invertible.

**The frontier, in three parts:**
- **Decomposable:** the outer function distributes, has a finite addition formula, or is invertible over the
  data's range.
- **Exponential in depth:** finitely many branches or terms, multiplying with each nested layer.
- **Genuinely coupled:** many-to-one with no finite expansion, over a range spanning many branches.

## 4. Composition is additive to first order, and gradient descent uses exactly that

Take maps close to the identity: f = id + εF, g = id + εG. Then

    f∘g = id + ε(F + G) + O(ε²).

To first order, composition is addition. For maps generated by flows, e^A and e^B, the Baker–Campbell–Hausdorff
formula (established) gives the full correction:

    log(e^A e^B) = A + B + ½[A, B] + (1/12)([A, [A, B]] − [B, [A, B]]) + …

The commutator [A, B] = AB − BA is precisely the obstruction to treating a composition as a sum. It vanishes when the
two parts commute, and it is second order in the step size.

That explains two pillars of deep learning (the mechanisms are established; this unified reading is ours):
- **Gradient descent** changes parameters by small steps. By the chain rule, the change in a deep composition's output
  is, to first order, a sum of each parameter's contribution. Gradient descent applies the residual trick to
  infinitesimal steps. Its price is that it can only make small continuous moves, never discrete structural ones, such
  as "wrap this in a square root".
- **Residual architectures** (He et al. 2016) write each layer as x + F(x), with F small, so the network is a
  composition of near-identity maps. By the formula above, the whole stack is then approximately the sum of its
  blocks, plus commutator corrections. This matches the empirical finding that residual networks behave like
  ensembles of shallow paths (Veit et al. 2016), and the view of them as discretized flows (Chen et al. 2018). The
  transformer's residual stream is the same design: every layer adds to one sum.

**So deep learning handles composition by never leaving the regime where it is nearly a sum:** small steps in
parameters, small updates per layer. Discrete symbolic search can't stay in that regime: each structural move is
large.

## 5. Three ways around, and what each costs

| route | idea | where it works | its cost |
|---|---|---|---|
| inversion (target propagation) | choose the outer function, transform the data by its inverse, search the inner part as a sum | invertible outer functions | one search per outer candidate per level: polynomial for bounded depth. Many-to-one layers multiply the branches |
| change of coordinates | find a transform under which the target becomes a sum (log for products, Fourier for convolutions) | combinations that are commutative in disguise | finding the coordinates is itself a search. It is cheap when the candidate transforms are few and can be scored by how additive the transformed data are |
| stay near the identity | build the composition from small increments, so commutators stay second order | gradient-trained networks | only continuous, small moves; no discrete structure |

Experiment B3 tests the first two together. The guide scores a small family of outer functions f by how additive
f⁻¹(y) is (a first-order sensitivity fit), and the finder searches the inner sum under the best ones.

## 6. Why this might matter for machine learning (hypotheses, labelled as such)

- **H1. A measurable obstruction.** How much the order of two candidate components matters on the data, a
  data-level commutator, should predict how much joint search a composition needs. Near zero means the parts
  decompose, and residual fitting (polynomial, data-efficient) suffices. Large means they are coupled, and only joint
  or gradient search will do. This is cheap to measure and could tell a learner where to spend its search.
- **H2. Data efficiency follows decomposability.** Residual fitting learns each part from all the data at once, with
  no interference between parts. A learner that first finds coordinates in which its target is a sum should learn
  compositional structure with that efficiency. Deep networks do this implicitly: the last layer is a sum over
  features, and the earlier layers learn the coordinates. A symbolic learner has to do it explicitly, by inversion or
  coordinate search.
- **H3. One fact, seen three times in this project.** It connects to two earlier findings:
  - "an operation is a basis": a transformer computes × as log, then a Fourier-domain add, then exp. That is, it
    finds coordinates in which × becomes +;
  - "operator as a group representation: commutative works, non-commutative is the frontier".

  The residual trick's limit is the same boundary. Additive or commutative structure decomposes. Non-commutative
  composition decomposes only after a change of coordinates, or locally.
- **H4. Where exponential search is irreducible.** Many-to-one layers make targets set-valued, and branch choices
  multiply. That is a candidate for an "exponential line with low N" (`OPEN_PROBLEMS.md`), with N the number of such
  layers.

None of H1–H4 is tested yet. B3 gives the first data on H2's symbolic side: does choosing the most additive
coordinates make nested formulas reachable in polynomial time?
