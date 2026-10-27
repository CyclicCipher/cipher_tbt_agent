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
  n = 10). Its formula contains the known small-hole law. But it extrapolates no better than the GP. As the hole nears
  the edge, the response is governed by the gap between hole and edge, 0.9 − yc − r. That combination is not in the
  library, so the formula cannot follow it.
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
