# Experiment A: symbolic surrogates of a simulator

**Why (the user, 2026-10-26):** to get data on whether the function-finding method from this folder is good for
anything engineering-shaped. Program synthesis won't design parts and won't replace a simulator. The question is
narrower: from a handful of simulations, can the method find a compact formula for a quantity of interest that is
accurate, extrapolates, and is good enough to optimize against?

## The pieces (`regression/claude/`)

- **`fem2d.py`, the simulator:** 2D plane-stress finite elements in numpy/scipy (bilinear quads, sparse solve,
  10–50 ms a run). A hole is modelled by each element's solid fraction on a 16 × 16 sub-grid, so outputs vary
  smoothly with hole size and position. Checks:
  - a cantilever's tip deflection is within 1.6% of Timoshenko beam theory (4L³ + 3.12L) and converges as the mesh
    is refined;
  - a small hole raises a plate's compliance by 0.0225, against the dilute-limit theory value of 3πr² = 0.0236.
- **`symreg.py`, the function finder:** Gemini's symbolic ResNet done properly. It grows a formula one term at a time
  on the residual, refitting every coefficient after each addition. Each step adds the term that most reduces the
  error after the refit (orthogonal least squares), and a beam keeps the 4 best formulas of each size: the lookahead
  lesson from the subset-sum experiment.
  - Terms are products of input powers (exponents −2 … 4, total ≤ 4) and logs, plus denominator terms, since
    t = P/(1 + D) is fitted linearly as t = P − t·D.
  - Cross-validation picks y or log y and the size, by the one-standard-error rule.
  - A formula with a pole in the sampled range is rejected.
  - On test functions it recovers 4x³ + 3.12x from 10 points, (1 + 2x)/(1 + 0.5x) from 20, and 1 + 3r²/(1 − r) from 30,
    exactly. It finds a constant in pure noise.
  - Relatives: FFX, SINDy, orthogonal least squares.
- **`surrogates.py`, the experiment:** simulate, fit, report.

## Design

| problem | inputs | quantity | known theory |
|---|---|---|---|
| beam | L ∈ [2, 10] (height 1) | tip deflection of a clamped cantilever under a unit tip load | Timoshenko: ≈ 4L³ + 3.12L |
| plate | hole radius r ∈ [0.05, 0.3], hole height yc ∈ [0.5, 0.7] (yc + r ≤ 0.9) | compliance of a unit square plate in tension, relative to no hole | only the dilute limit (small r, central) |
| bracket | hole position xc ∈ [0.6, 3.4], yc ∈ [0.25, 0.65], radius r ∈ [0.05, 0.2] | tip deflection of a clamped 4 × 1 cantilever, relative to no hole | none |

- **Budgets:** 10, 30 and 100 training simulations, uniformly random in the box, 3 seeds each.
- **Test sets:** 500 random in-range designs.
- **Extrapolation sets** (200 designs): beam L ∈ (10, 14]; plate r ∈ (0.3, 0.38] with yc ∈ [0.5, 0.52]; bracket
  r ∈ (0.2, 0.26].
- **Optimization task (bracket):** a hole of r = 0.15 must go somewhere; choose (xc, yc) on a 15 × 9 grid to minimize
  tip deflection. Regret = (the surrogate's choice − the best) / (the worst − the best), over the simulator's values
  on the grid.
- **Surrogates:**
  - symreg;
  - a Gaussian process (Matern 5/2, one length scale per input, a noise term), the standard surrogate, with y or log y
    chosen by 5-fold cross-validation, as symreg does;
  - a full quadratic response surface (the classical one).
- **Measures:** NRMSE = RMSE / standard deviation of y over the set (0 is perfect; 1 is as bad as predicting the
  mean); the maximum error, in the same units; the formula's size. The simulator's own roughness (the residual of a
  degree-8 fit along a 1-D sweep through the box's centre, in the same units) is reported as a floor.
- **Inputs are given as dimensionless ratios** (lengths in units of the plate or beam height, loads and moduli 1).
  Engineers do this anyway, but it is structure supplied by hand.

## Predictions (2026-10-26, before any simulation was run)

| # | prediction |
|---|---|
| A1 | beam: symreg's in-range NRMSE ≤ 0.01 at n = 10 and its extrapolation NRMSE ≤ 0.02; the GP's extrapolation NRMSE is ≥ 5× symreg's at every n |
| A2 | plate: symreg's in-range NRMSE ≤ 0.05 at n = 100, and its extrapolation NRMSE < the GP's at n = 100 |
| A3 | bracket: the GP beats symreg in range at n = 100 (no compact formula exists in these coordinates) |
| A4 | bracket optimization: symreg's mean regret ≤ the GP's at n = 30 |
| A5 | the quadratic response surface is the worst in range at n = 100 on plate and bracket |
| A6 | symreg's formulas have ≤ 6 terms on beam and plate at n = 100 |

    python -m regression.claude.surrogates simulate
    python -m regression.claude.surrogates fit
    python -m regression.claude.surrogates report

## Results (2026-10-26; `runs/surrogates_*.txt`, `runs/surrogates/results.jsonl`; CPU: 3,741 simulations in 22 s, all fits in ~2 min)

Mean over 3 seeds. NRMSE: 0 is perfect, 1 is as bad as predicting the mean.

| problem | n | symreg in / ex | GP in / ex | quadratic in / ex | symreg terms |
|---|---|---|---|---|---|
| beam | 10 | **0.0001 / 0.0006** | 0.0026 / 0.17 | 0.057 / 0.38 | 3, 3, 4 |
| beam | 100 | **0.0000 / 0.0003** | 0.0001 / 0.13 | 0.035 / 0.45 | 4, 4, 4 |
| plate | 10 | **0.052** / **0.57** | 0.069 / 1.03 | 0.17 / 1.08 | 3, 2, 3 |
| plate | 30 | 0.030 / **0.48** | **0.022** / 0.54 | 0.13 / 1.12 | 5, 6, 7 |
| plate | 100 | 0.011 / 0.69 | **0.006 / 0.36** | 0.10 / 1.06 | 6, 9, 6 |
| bracket | 10 | **0.80** / **1.06** | 0.92 / 1.22 | 1.39 / 1.42 | 1, 2, 2 |
| bracket | 30 | 0.50 / **0.73** | **0.45** / 0.82 | 0.59 / 0.82 | 3, 8, 2 |
| bracket | 100 | 0.35 / 0.64 | **0.078 / 0.33** | 0.49 / 0.72 | 3, 6, 5 |

Bracket optimization regret (0 = the best hole position, 1 = the worst): n = 10: symreg 0.45, GP 0.12, quadratic 0.001;
n = 30: 0.001 / 0.002 / 0.002; n = 100: 0.058 / 0.005 / 0.001.

Simulator roughness floor (in the same units): beam 4 × 10⁻⁵; plate up to 9 × 10⁻³; bracket up to 8 × 10⁻³. The plate's
figure is an overestimate, because its sweep runs into the edge, where a degree-8 fit is poor. So the plate's
in-range errors at n = 100 (0.006–0.011) sit at or near the simulator's own roughness.

The formulas symreg found at n = 100 (seed 0):
- beam: y = 0.063 + 3.003 L − 0.072 L² + 3.98 L³. This is Timoshenko beam theory (4L³ + 3.12L) recovered from the
  simulator. The small L² term absorbs the discretization error (the elements are slightly too stiff in bending).
- plate: log y = (−0.0027 + 9.28 r² + 20.3 r³/yc − 93.6 r³ + 54.6 r³ √yc) / (1 − 3.77 r·yc). The leading 9.28 r² is
  the dilute limit: log(1 + 3πr²) ≈ 9.42 r².
- bracket: y = (0.969 + 0.047 r / (√xc · yc²)) / (1 − 0.080 yc²): crude.

**Verdicts.**
- **A1 confirmed:** beam, NRMSE 0.0001 in range at n = 10, 0.0006 in extrapolation. The GP's extrapolation error is
  280× larger.
- **A2 refuted:** plate, symreg is accurate in range (0.011 at n = 100, as predicted), but it extrapolates worse
  than the GP (0.69 against 0.36).
- **A3 confirmed:** bracket, the GP wins in range at n = 100 (0.078 against 0.35).
- **A4 confirmed, but uninformative:** at n = 30 every surrogate finds the best hole position (regret ≤ 0.002). The
  task was too easy: put the hole near the tip, near the neutral axis.
- **A5 confirmed:** the quadratic response surface is the worst in range at n = 100 on plate and bracket.
- **A6 refuted:** beam formulas have 4 terms, but one plate formula has 9 (the others 6).

## Reading

- **Where the quantity follows a compact law in the given coordinates, the function finder is far better than the
  standard surrogate.** On the beam it recovers beam theory from 10 simulations and extrapolates 40% past the
  training range almost exactly. A Gaussian process can't, whatever its budget. This is the case where a program,
  rather than a fit, is the point.
- **On a smooth 2-input response it is competitive in range,** and better at the smallest budget (0.052 vs 0.069 at
  n = 10). Its formula contains the known small-hole law. On the mean, it extrapolates worse than the GP.
  (Correction from part 2: this paragraph first blamed a missing variable, the gap between hole and edge. Per seed, two
  of the three formulas at n = 100 extrapolate 3–11× better than the GP (0.039 and 0.061 against the GP's 0.18–0.69).
  The third blows up (1.96) and drives the mean. The limitation is reliability, not a missing variable; see part 2.)
- **Where the effect is local and interaction-heavy (the bracket), it loses clearly.** A hole matters a lot near the
  root and near the edges, and hardly at all elsewhere. No short sum of power terms captures that, and the GP's local
  smoothing does better (0.078 vs 0.35).
- **So the method is a law-finder, not a general surrogate.** It is useful exactly where engineers want a law: sizing
  rules, scaling laws, design correlations. Where the honest answer is "it's complicated", a GP is better. The
  failure modes point at what's missing: the right combined variables (gaps, ratios, distances to constraints), which
  are again supplied structure unless something finds them.

**Not run (next candidates):**
- Give the library the design's own constraint slacks as candidate variables (the designer knows them: 0.9 − yc − r
  for the plate) and test plate extrapolation again.
- A harder optimization task: several holes, or mass against stiffness.

# Part 2: constraint margins, and what limits the function finder

**Why (the user, 2026-10-26):** try the constraint-margins follow-up. Above all, understand the current algorithm's
limitations, so it can be redesigned into something more general. The ideal is to beat the Gaussian process by a
wide margin everywhere.

## Additions (`surrogates.py`, `diagnostics.py`, `fem2d.py`, `symreg.py`)

- **A fourth problem, the stepped beam (6 inputs):** a clamped 4 × 1 cantilever made of 6 equal segments, segment i
  of height hᵢ ∈ [0.4, 1.0], centred. The quantity is tip deflection.
  - Euler–Bernoulli theory gives a compact law, Σᵢ 4wᵢ/hᵢ³. The simulator matches it to 4% for a uniform beam (the
    shear correction) and 5% for a tapered one.
  - Alternating thick and thin segments come out 19% more flexible than the law: the steps add local interactions.
  - Extrapolation set: heights in [0.3, 1.0], at least one below 0.4. Same budgets, seeds and test sizes.
  - **This is where structure should beat a GP most:** 6 inputs, a compact leading law.
- **Library scaling:** the full term set grows as 9^d (8 terms at d = 1, 328 at d = 3, 19,188 at d = 6). With 4 or
  more inputs, terms may combine at most 2 inputs (708 terms at d = 6). The 1–3-input problems are unchanged.
- **New surrogates:**
  - `symreg_margins` and `gp_margins`: given the design's constraint margins as extra inputs. Plate: s = 0.9 − yc − r.
    Bracket: s1 = yc − r − 0.05 and s2 = 0.95 − yc − r. The designer wrote these constraints, so the margins are fair
    inputs.
  - `hybrid`: symreg's formula as the trend, plus a GP fitted to its residuals (in symreg's y or log y).
- **Diagnostics** (`diagnostics.py`), each isolating one candidate limitation:
  - **D1 representation:** the library's best with every term at once (ridge regression, penalty by 5-fold CV), and
    the best k-term formula for k up to 30, both from all 300 training simulations.
  - **D2 search:** symreg's chosen formula at n = 100, refound with a beam 8× wider.
  - **D3 term cap:** how often cross-validation picks the largest size allowed.
  - **D4 locality (bracket):** the share of squared test error near the root (xc < 1.2) or within 0.1 of an edge,
    against that region's share of designs.

**Seen before writing these predictions** (a smoke test of the new code on plate, n = 30, seed 0, in range):
symreg_margins 0.019, gp_margins 0.012, hybrid 0.067. The hybrid equals symreg alone on that seed (0.067): the
residual GP learned almost nothing, because a misspecified formula leaves rougher residuals than the function
itself.

## Predictions (2026-10-26, before the part-2 runs)

| # | prediction |
|---|---|
| M1 | plate: with margins, symreg's extrapolation NRMSE at n = 100 is ≤ 0.35 (half of 0.69, and below the GP's 0.36) |
| M2 | bracket: the plain GP still beats symreg_margins in range at n = 100 |
| S1 | stepped: symreg's in-range NRMSE ≤ 0.05 at n = 100 and ≤ 0.5× the GP's at every n; its extrapolation NRMSE ≤ 0.3× the GP's at n = 100 |
| D1 | the library's every-term floor (300 simulations): bracket ≥ 0.15 in range, whichever transform; plate ≤ 0.01; beam ≤ 0.001 |
| D2 | a 32-wide beam lowers the chosen formula's training error by < 10% in every fit: search is not the bottleneck |
| D3 | at n = 100 the 8-term cap is hit in ≥ 2 of 3 fits on the bracket and on the stepped beam |
| D4 | bracket: the region's share of symreg's squared error is ≥ 2× its share of designs |
| D5 | the hybrid is within 10% of the better of symreg and the GP, in range, at n = 100, on every problem (after the smoke test I expect this to fail on the plate) |
| — | gp_margins against gp: no prediction |

    python -m regression.claude.surrogates simulate      (adds the stepped beam; existing simulations are kept)
    python -m regression.claude.surrogates fit           (fits only what is missing)
    python -m regression.claude.surrogates report
    python -m regression.claude.diagnostics

## Part 2 results (2026-10-26; `runs/surrogates_report2.txt`, `runs/surrogates_diagnostics.txt`, `runs/surrogates_explore.txt`)

### The new surrogates, at n = 100 (per seed: in range / extrapolation)

| problem | symreg | symreg_margins | GP | gp_margins | hybrid |
|---|---|---|---|---|---|
| plate | 0.005, 0.024, 0.005 / **0.039, 1.96, 0.061** | 0.23, 0.010, 0.018 / 0.29, 0.44, 0.12 | **0.005, 0.005, 0.008** / 0.18, 0.21, 0.69 | 0.005, 0.005, 0.007 / 0.17, 0.22, 0.66 | 0.005, 0.015, 0.005 / 0.039, 1.96, 0.061 |
| bracket | 0.40, 0.35, 0.30 / 0.79, 0.66, 0.48 | 0.77, 0.81, 0.65 / 12, 12, 168 | **0.056, 0.107, 0.071 / 0.26, 0.42, 0.31** | 0.068, 0.106, 0.062 / 0.28, 0.35, 0.27 | 0.064, 0.156, 0.110 / 0.36, 0.54, 0.33 |
| stepped | 0.33, 0.43, 0.32 / 0.47, 0.97, 0.60 | — | **0.099, 0.099, 0.101 / 0.38, 0.37, 0.29** | — | 0.33, 0.13, 0.09 / 0.47, 0.53, 0.32 |

Stepped beam at smaller budgets (mean, in range): n = 10 symreg 0.58, GP 0.59; n = 30 symreg 0.41, GP 0.33.

### The diagnostics

| | beam | plate | bracket | stepped |
|---|---|---|---|---|
| D1: library, every term (ridge, 300 simulations), in range | 0.0014 | 0.0056 | **0.053** | 0.057 |
| D1: best k-term formula (300 simulations), in range at k = 4 / 8 / 16 / 30 | 0 / 0 / 0 / 0 | 0.015 / 0.009 / 0.004 / 0.004 | 0.34 / 0.18 / 0.11 / **0.047** | 0.20 / 0.13 / 0.056 / 0.048 |
| D2: training error lower with a 32-wide beam (3 seeds) | 0, 0, 0% | **89, 28, 90%** | 0, **72**, 21% | **40**, 9, **57**% |
| D3: terms chosen at n = 100 (cap 9 including the constant) | 4, 4, 4 | 6, 9, 6 | 3, 6, 5 | 7, 5, 4 |
| D4: the near-root/near-edge region, 23% of bracket designs, holds | | | 56% of symreg's squared error, 72% of the GP's | |

### Exploratory, after the diagnostics (not pre-registered; `regression/claude/explore_fixes.py`; n = 100, mean of 3 seeds)

| change | beam in / ex | plate in / ex | bracket in / ex | stepped in / ex |
|---|---|---|---|---|
| none (symreg as above) | 0.0000 / 0.0003 | 0.011 / 0.69 | 0.35 / 0.64 | 0.36 / 0.68 |
| E1: add the exponent −3 | 0.0000 / 0.0003 | 0.005 / 0.056 | 0.35 / 0.59 | 0.35 / 0.69 |
| E2: beam 32 instead of 4 | 0.0000 / 0.0003 | 0.007 / 0.20 | 0.24 / **7,214** | 0.34 / 0.93 |
| E1 + E2 | 0.0000 / 0.0003 | 0.007 / 0.21 | 0.20 / **332** | 0.19 / 0.36 |

E1 changed the plate only through seed 1. Its seed-0 formula is identical to the original, and seed 1 now takes a
different path that happens to extrapolate. That is instability, not an effect of x⁻³.

The textbook law fitted directly to the stepped beam's 300 training simulations (least squares on the given terms):

| terms | in range | extrapolation |
|---|---|---|
| bending, c₀ + Σ cᵢ/hᵢ³ (7 terms) | 0.102 | **0.094** |
| + shear, Σ bᵢ/hᵢ (13 terms) | 0.094 | 0.125 |
| + neighbour interaction, Σ dᵢ/(hᵢhᵢ₊₁)^1.5 (18 terms) | **0.043** | 0.100 |

For comparison, the GP at n = 100 scores 0.100 / 0.34.

### Verdicts (3 confirmed, 5 refuted)

- **M1 confirmed on the mean, misleading per seed:** the margins cut plate extrapolation to 0.28 on average. But they
  made the two good seeds worse (0.039 → 0.29, 0.061 → 0.12) and the bad one better (1.96 → 0.44). In range they
  hurt (0.23 on seed 0), through a singular s^-½ term.
- **M2 confirmed:** on the bracket, symreg with margins is far worse than the GP. It is catastrophic in
  extrapolation (12–168) through terms like 1/(s1·s2), which blow up at the constraint boundary.
- **S1 refuted:** on the stepped beam symreg scores 0.36 in range, against the GP's 0.10.
- **D1 refuted where it mattered:** the library is not what limits the bracket. Every term at once reaches 0.053, and
  a 30-term formula 0.047, below the GP's 0.078 (though from 300 simulations, not 100).
- **D2 refuted:** search is a bottleneck. A wider beam finds formulas with up to 90% lower training error.
- **D3 refuted:** the 8-term cap is rarely reached. Cross-validation stops at 3–6 terms.
- **D4 confirmed:** the bracket's errors concentrate where a hole matters most (56% of symreg's squared error from 23%
  of designs). The GP's errors concentrate even more (72%).
- **D5 refuted:** the hybrid is never better than the GP in range except on the beam (plate 0.008 vs 0.006, bracket
  0.110 vs 0.078, stepped 0.18 vs 0.10).
- gp_margins against gp: the margins barely change the GP (plate 0.0058 → 0.0058; bracket 0.078 → 0.079).

## The limitations of the current function finder (the evidence above, combined)

1. **Its exponents are a fixed grid.** The stepped beam's law needs h⁻³, which was missing (−2 was the floor). Any
   law with an exponent off the grid (1.5, −2.7, a fitted rate) is unreachable, or approximated by several wrong terms.
2. **Its search is too greedy.** A beam of 4 leaves 20–90% of the achievable training error on the table (D2). Terms
   that only work together are never tried together. The stepped beam's 7-term textbook law matches the GP in range
   and extrapolates 3.6× better, but the search never assembles it.
3. **A stronger search exploits singular terms.** Widening the beam improves the bracket in range (0.35 → 0.24) and
   wrecks its extrapolation (7,214). Margins do the same (12–168). The pole guard checks denominators only, but a
   negative power of any variable that can approach zero is just as singular. Safety has to grow with search power;
   it is the maximizer problem from the beauty experiments in a new place.
4. **It cannot tell equally good formulas apart.** On the plate, three formulas fit the training data alike. Two
   extrapolate 3–11× better than the GP, one blows up. Cross-validation inside the box can't see the difference. A
   one-exponent change to the library flips which one is chosen (E1).
5. **It selects too small.** Cross-validation with the one-standard-error rule stops at 3–6 terms (D3), where the
   library could reach 0.05 on the bracket with 16–30 terms (D1). With noisy search, larger formulas don't earn
   their place in cross-validation.
6. **It has no notion of inputs of the same kind.** The stepped beam has six heights that play the same role, and the
   law is one term per height. Nothing tells the finder to try the same term on every height (a template).
7. **Its terms are global.** Power terms extend over the whole domain. The bracket's effect is local (near the root,
   near the edges), and that is where its error concentrates (D4). The GP has the same trouble, more so.
8. **Its library grows as 9^d,** so terms are capped at 2 inputs from 4 inputs up, and 3-way interactions are cut off
   in higher dimensions.
9. **Its hybrid is naive.** A GP fitted to the residuals of a misspecified formula learns little: those residuals are
   rougher than the function. The two have to be fitted together.

## What a more general version needs (a design, not built)

- **Terms with continuous parameters:** blocks like c·(a·x + b)^p, c·exp(k·(a·x + b)) and c·log(a·x + b). The linear
  coefficients are solved exactly and the nonlinear ones (exponent, shift, direction) by Gauss–Newton (variable
  projection). This answers 1. It also lets the finder build its own combined variables (a·x + b), which the margins
  showed can matter. And shifted forms like (s + b)^-1 place singularities outside the domain instead of on its
  boundary.
- **A real search:** wider beams plus swap moves (replace one term, keep the rest), restarts, or exact best-subset
  selection for small libraries (2).
- **Safety that scales with the search:** every term checked for singularities and runaway growth over the actual
  constrained design domain, not just denominators (3).
- **An ensemble instead of one formula:** keep every formula within the noise of the best and report their spread.
  Where they agree, trust the extrapolation; where they diverge, say so. Plate seed 1 would have shown up as
  disagreement (4). It also gives the uncertainty estimate a GP provides, which choosing the next simulation needs.
- **Selection by evidence:** Bayesian evidence or MDL instead of the one-standard-error rule, so that formulas grow
  with the data (5).
- **Templates:** declare exchangeable inputs, or detect them, and propose one term applied to each (6).
- **Gated formulas for local effects:** soft, input-dependent gates that select between formula blocks. This is the
  attention-residual idea from Gemini's regressor, used where it belongs (7).
- **The GP as the remainder, fitted jointly:** the formula's terms become the GP's mean functions (universal kriging),
  so the GP fits only what the formula can't explain. In range it should then never do worse than a GP, and far from
  the data it falls back to the formula (9).

**On beating the GP everywhere.** In range, on a smooth response with no compact structure, a well-fitted GP is hard
to beat at all; no method wins there by default. Here is what these four problems show. Wherever a law exists, the
function finder's formula extrapolates far better than the GP when it finds the right one:
- beam: 280×;
- plate: 3–11× in 2 of 3 seeds;
- stepped beam: 3.6×, if the textbook law had been found.

In range, a formula matches the GP only once it is complete (the stepped beam needs the neighbour terms to reach
0.043). So the realistic target is a composite that is never worse than the GP in range and much better outside it.
That design is above.
