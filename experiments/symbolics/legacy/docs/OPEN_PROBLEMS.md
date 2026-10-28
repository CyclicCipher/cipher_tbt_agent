# Open problems of the function finder

The barriers between the function finder (`regression/claude/symreg2.py`, after experiment B2) and expressing most
mathematical functions, from easy to fundamental (2026-10-26). Each one says what fails, where we saw it, and what
would address it. Status is updated as experiments run.

## The rule on cost: polynomial and exponential lines

Everything so far runs in polynomial time: greedy residual search with a fixed beam, depth-bounded constructions,
fixed candidate sets. The user (2026-10-26): exponential time with low N may be allowed. If so, the polynomial and
exponential lines are kept separate, in separate code paths, arms and records, so each result says which line it
belongs to. An "exponential with low N" method states its N and its growth law, e.g. 2^k branches for k non-invertible
layers.

## The problems

| # | problem | what fails | where we saw it | what would address it | status |
|---|---|---|---|---|---|
| 1 | **Missing atoms** | no trig or hyperbolic functions, no Gaussian e^(−x²), no root of a sum, no abs, max or hinge (nothing non-smooth), no special functions (erf, Bessel, elliptic) | not yet tested: the simulator's tasks are smooth power-law-like | one atom family each, with a fitted scale and shift; each with growth control outside the box (problem 6) | open |
| 2 | **Atoms see one input** | nothing like (a·x + b·y + c)^p or √(x² + y²): a gap between a hole and an edge, or a distance, can't be written. The constraint margins were a hand-made patch | experiment A (the plate's extrapolation), B (margins as a special case) | atoms over a fitted combination of inputs, fitted by variable projection | **B3 tests this** |
| 3 | **No composition** | terms are flat products of atoms: no f(g(x)), e.g. 1/(1 + (x/a)²), log(1 + x²), e^(−r/λ) cos(kr). The residual trick works only on the outermost sum (`RESIDUAL_TRICK.md`) | structural: nothing nested has been reachable | composition by inversion: choose an outer function f, transform the data by f⁻¹, search the inner sum; recursive, depth-bounded. Library templates grow the effective depth | **B3 tests depth 2 (one outer function)** |
| 4 | **The outer structure is a linear sum** | a product of sums (a + b)(c + d) only by multiplying out; denominators limited to integer monomials of degree ≤ 2 | B (bloated formulas: products expanded into many terms) | composition (problem 3); the residual for a product is a ratio, as the residual for a sum is a difference | open (partly by 3) |
| 5 | **Groups range over single inputs, never relations** | "the same term for each adjacent pair" can't be said; with 6 bending terms and 5 interaction terms the 8-term cap is exceeded | B2, the 6-segment beam: real interactions at the steps, written as many awkward product terms | relational templates (one term per pair in a set of input pairs), with the pairs found from data by a second-order sensitivity decomposition | **B3 tests this** |
| 6 | **More vocabulary means more overfitting** | a richer vocabulary fits noise and makes equally good formulas that disagree outside the data; terms tame in the box explode outside (h^4.55, r^6) | B (v2 extrapolated worse than v1), B2 (blow-ups up to 10¹⁹) | stronger priors (the library prior, description length); growth control outside the box (an enlarged check region, bounded exponents, abstaining where the frontier disagrees) | open (the next experiment after B3) |
| 7 | **"All functions" is out of reach in closed form** | most functions aren't elementary: antiderivatives (erf), solutions of differential equations, eigenvalues, series solutions, roots of transcendental equations. Many engineering quantities are exactly these | not yet confronted | programs (sums, integrals, iteration, equation solving). Searching programs is exponential in program length in general: polynomial time covers a bounded depth plus what the library has compressed into atoms. This is where an exponential line with low N would start | open, fundamental |
