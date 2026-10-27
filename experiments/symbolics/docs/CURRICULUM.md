# Experiment B: a learned library and a non-neural guide for the function finder

**Status:** pre-registered 2026-10-26, nothing built or run yet.

**Why (the user, 2026-10-26):** experiment A found that the function finder is an excellent law-finder where it
finds the law, and fails or is unreliable elsewhere (`SURROGATES.md`). Its limitations, from the diagnostics: a fixed
exponent grid, singular terms, unreliable selection among equally good formulas, no templates for inputs of the same
kind, and global terms only. The user's constraints for the redesign: stay in polynomial time, and no Gaussian
process. DreamCoder's library learning and a guide without a neural network fit those constraints
(`DREAMCODER_NOTES.md`).

**The question:** does a library learned from a curriculum of simpler simulated tasks, together with a guide built
from data statistics, let a polynomial-time finder solve held-out tasks that it cannot solve from primitives?

## The tasks

Every task comes from `regression/claude/fem2d.py`: its inputs vary within a box, the rest of the configuration is
fixed, and the output is one quantity. Outputs are dimensionless (E = 1, loads 1, thickness 1). Beam outputs are
deflections; hole outputs are ratios to the same structure without the hole. Two simulator extensions are needed: a
uniform load along the top edge, and a beam clamped at both ends with a mid-span load. Each will be checked against
beam theory before use (cantilever under uniform load, qL⁴/8EI; clamped–clamped beam with a central load, PL³/192EI).

**Training curriculum (48 tasks, 50 simulations each; only these data are seen while learning):**

| family | tasks | inputs varied (others fixed) |
|---|---|---|
| cantilever, tip load | 8 | {L}, {h}, {L, h}, {L, ν}, {h, ν}, {L, h, ν}, plus {L} and {L, h} on a second range |
| cantilever, uniform load | 8 | the same eight input sets |
| clamped–clamped beam | 8 | mid-span load: {L}, {h}, {L, h}, {L, ν}; uniform load: the same four |
| stepped cantilever, tip load | 6 | 2 segments {h1, h2}; 3 segments {h1, h2, h3}; 2 segments + length {h1, h2, L}; 3 segments of total length 3; 2 and 3 segments on a second height range |
| plate with a hole, tension | 8 | {r} central; {r} at yc = 0.65; {yc} at r = 0.15; {r, xc} (horizontal offset); {r} in a 2 × 1 plate; {r} with ν; {xc} at r = 0.2; {r, ν} |
| cantilever with a hole, tip load | 10 | {r} at three fixed positions; {xc} at r = 0.15; {yc} at r = 0.15; {xc, r} on the axis; {yc, r} at xc = 2; {xc, yc} at r = 0.1; {r} near the root; {xc} near an edge |

**Held-out tasks (never seen while learning):** n = 30 and n = 100 training simulations, 3 seeds each, 500 in-range
test designs, 200 extrapolation designs, as in experiment A.

| task | what is new | baseline from experiment A (n = 100, in / ex) |
|---|---|---|
| H1 stepped cantilever, 6 segments, tip load | 6 inputs; more segments than any training task | v1 0.36 / 0.68; GP 0.10 / 0.34; textbook law 0.10 / 0.094 |
| H2 cantilever with a hole {xc, yc, r} | all three hole inputs at once | v1 0.35 / 0.64; GP 0.078 / 0.33 |
| H3 plate with a hole {r, yc} | the vertical offset and radius together | v1 0.011 / 0.69 (2 of 3 seeds ≈ 0.05); GP 0.006 / 0.36 |
| H4 stepped cantilever, 4 segments, **uniform load** | uniform load and segments never appear together in training | — |
| H5 clamped–clamped beam, 3 segments, mid-span load | clamped–clamped and segments never appear together | — |
| H6 cantilever with a hole {xc, r}, **uniform load** | holes and uniform load never appear together | — |

## The system (all polynomial; no Gaussian process; no neural network)

### Wake: the finder, version 2 (`regression/claude/symreg2.py`, to be built)

- **Formula:** t = Σₖ cₖ φₖ(x; θₖ), optionally over a denominator 1 + D(x) as in version 1, with t = y or log y.
- **Primitive terms, each with continuous parameters θ:**
  - xᵢ^p;
  - xᵢ^p xⱼ^q;
  - (xᵢ + b)^p, with b kept so the base stays positive on the design domain;
  - log xᵢ and exp(k xᵢ);
  - (sⱼ + b)^p for the task's constraint margins sⱼ, with b > 0 (a margin's singularity stays outside the domain).
- **Fitting by variable projection:** the linear coefficients c are solved exactly by least squares. The nonlinear θ
  are refined by Levenberg–Marquardt, at most 50 function evaluations per candidate.
- **Search, polynomial:** as in version 1, the formula grows one term at a time on the residual, with a beam of 4 and
  at most 8 terms. Each step screens every candidate with θ on a grid (exponents −3 … 4 in halves) by the exact
  drop in squared error after a linear refit, as version 1 does. It then refines the 8 best by Levenberg–Marquardt.
  Cost per task: O(steps × beam × candidates × n), candidates O(d² + library size × bindings).
- **Safety over the design domain:** every candidate term is evaluated on 2,000 feasible designs, 20% of them on the
  box and constraint boundaries. It is rejected if any value is non-finite or its largest magnitude exceeds 10⁴×
  its median. Denominators keep the version-1 pole guard.
- **Selection by BIC:** BIC = n log(RSS/n) + (linear + nonlinear parameters) log n − 2 log P(terms | library). A
  log-y formula is compared on the y scale, with the Jacobian term 2 Σ log y. The size is the BIC minimum along the
  search path.
- **The frontier (an ensemble):** every formula met during the search whose BIC is within 6 of the best (posterior
  ratio ≥ e⁻³), at most 8. Each is weighted by exp(−BIC/2). The prediction is the weighted mean; the spread is the
  weighted standard deviation of the members' predictions, in units of the training y's standard deviation.

### Abstraction sleep: library learning (`regression/claude/library.py`, to be built)

- **Corpus:** the best formula of every training task whose in-sample NRMSE (training data only) is ≤ 0.02.
- **A term is a small tree:** a product of factors (power of an input, of a pair, of a shifted input or margin; log;
  exp) with their parameters.
- **Candidate templates come from two operators, both polynomial:**
  - **Anti-unification of term pairs** across tasks (their least general generalization). Inputs become slots. A
    parameter becomes a fixed constant when the paired values agree within 0.1 (this is how an exponent like −3 is
    learned), and a free parameter otherwise. Templates are capped at 5 factors.
  - **Group folding within a formula:** two or more terms identical up to which input they use, with equal fixed
    parameters, fold into Σ_{v∈G} c_v T(v) over a group G of inputs. Exponents are shared, coefficients per input.
    This is the template the stepped beam needs.
- **Description length** (in nats) of the corpus with library L: each template costs its size × log(number of
  primitives); each term instance costs −log P(template) plus ½ log n per free parameter. Templates are added
  greedily, one at a time, while the total drops (Stitch-style: one abstraction, rewrite, repeat).
- **The library's prior:** P(template) ∝ uses + 1.

### The guide (`regression/claude/guide.py`, to be built): which terms to try first

- **G1, deduction:** invert operators to split targets. The residual for sums (already the finder's core). log y
  for products, when G2 finds multiplicative separability.
- **G2, structure statistics, from the training data alone:**
  - **Additivity:** a first-order functional-ANOVA fit (a degree-4 polynomial in each input, summed, by least squares),
    on y and on log y. If R² ≥ 0.98 for either, the task splits: backfitting gives each input's component, each
    component gets a 1-D fit (version 2, at most 3 terms), and the components are summed. Then a joint refit, and
    the greedy search continues on the residual for interactions.
  - **Exponent estimates:** for each input, p̂ = 1 + the median slope of log|fᵢ′| against log xᵢ, from the fitted
    component. Each estimate becomes an extra initial value for that input's power terms.
  - **Exchangeable inputs:** inputs whose 1-D component fits share a template, with exponents within 0.3, form a
    group. The group is offered to group templates. Input names are never used.
  - **Margins:** the shifted-margin terms (sⱼ + b)^p are offered for every constraint of the task.
- **G3, a guide learned by the system (case-based):**
  - **Signature:** number of inputs, the two additivity R², the sorted exponent estimates, and a separability score
    (the mean absolute mixed second difference of log y, normalized).
  - **Lookup:** for a new task, retrieve the 3 nearest solved tasks by standardized signature distance, from training
    tasks and fantasies. Multiply the prior of every template they used by 3.
- **Fantasies:** each cycle, 100 synthetic tasks. Each sums 1–4 library templates with random bindings and
  parameters, on random input boxes in [0.2, 2], evaluated exactly (no simulation). They enter the case memory with
  their known templates.

### The loop

3 wake–sleep cycles on the 48 training tasks: wake (fit every task with the current library and guide), then
abstraction sleep, then fantasies. The library and case memory after cycle 3 are frozen for the held-out tasks.

## Arms (on the held-out tasks)

| arm | finder | guide | library |
|---|---|---|---|
| A0 | version 1 (`symreg.py`, as in experiment A) | — | — |
| A1 | version 2 | — | primitives only |
| A2 | version 2 | G1 + G2 | primitives only |
| A3 | version 2 | — | learned |
| A4 | version 2 | G1 + G2 + G3 | learned (the full system) |

**References, not arms:** the GP results from experiment A (H1–H3), and a GP fitted once on H4–H6 for the yardstick.
The textbook law fitted directly where one exists (H1, H4, H5).

## Measures

- NRMSE in range and in extrapolation (RMSE / standard deviation of y over the set).
- Formula size (terms, parameters) and the formula itself.
- The frontier's spread on the extrapolation set.
- Wall time per fit, and the time of the whole curriculum.
- The learned library, listed in full.

## Predictions (2026-10-26, before anything is built)

| # | prediction |
|---|---|
| P1 | **The DreamCoder claim:** A4 beats A1 in range at n = 100 by ≥ 2× on at least 4 of the 6 held-out tasks |
| P2 | **Stepped beam (H1):** A4 reaches NRMSE ≤ 0.10 in range and ≤ 0.15 in extrapolation at n = 100 |
| P3 | **The guide carries H1:** A2 (guide, no library) is within 1.5× of A4 in range on H1 |
| P4 | **New combinations (H4–H6):** A4 beats A1 by ≥ 2× in range on at least 2 of the 3 |
| P5 | **Bracket (H2):** no arm gets below 0.15 in range at n = 100 (local effects are not addressed here) |
| P6 | **Safety:** no version-2 arm has an extrapolation NRMSE above 10 on any held-out task, budget or seed (version 1 reached 7,214 with a wide beam) |
| P7 | **The frontier flags its failures:** of version-2 fits with extrapolation NRMSE > 1, ≥ 80% have a spread > 0.1; of the others, ≤ 30% do. Untestable if fewer than 3 such failures occur |
| P8 | **Cost:** the whole curriculum (3 cycles) takes ≤ 30 CPU-minutes; a held-out fit ≤ 60 s; going from n = 30 to n = 100 multiplies fit time by less than (100/30)² ≈ 11 |
| P9 | **The yardstick (reference only):** at n = 100, A4 beats the GP in range on at least 3 of the 6 held-out tasks, and in extrapolation on at least 5 of 6 |
| P10 | **What gets learned:** the library contains a power template with a fixed exponent −3 ± 0.1 and at least one group template |

## Rules

- Each run ≤ 10 minutes, in resumable stages.
- The held-out tasks are not simulated until the library is frozen.
- Any change to a specified item before the held-out evaluation is recorded here as a dated amendment, with its
  reason. Implementation details not specified here are decided during the build and listed in the results.
- If a component can't be built as specified (for example, if anti-unification as stated yields no group templates),
  that is a result, and it is recorded rather than patched around silently.

## Build order

1. The simulator extensions (uniform load, clamped–clamped), checked against beam theory.
2. The finder, version 2: tested on the experiment-A problems, and on test functions with known formulas
   (h^-3, shifted powers, a singular margin).
3. The guide's statistics, tested on functions of known structure (additive, separable, neither).
4. Library learning, tested on a hand-made corpus with a known best template.
5. The curriculum: simulation, then 3 wake–sleep cycles.
6. Freeze. Simulate the held-out tasks, fit every arm, report.

## Amendments (before any held-out task was simulated)

- **2026-10-26, exchangeable inputs.** Inputs now form a group when the guide's exponent estimates agree within 0.3.
  The specified test (1-D fits that share a template) compared term syntax, and equivalent forms differ in syntax:
  h⁻³ and (h + 0.0014)⁻³ landed in different templates on a test function, so only 2 of 4 equal inputs were grouped.
- **2026-10-26, implementation decisions in finder version 2** (within the specification, listed for the record):
  - The "8 best" refined each step are the 8 best distinct candidates; the same candidate from different grid
    starts used to fill every slot.
  - Each candidate's refinement fits only its own parameters. The kept formulas then get the joint refit.
  - The 50-evaluation budget includes finite-difference Jacobian evaluations.
  - Exponents and exp rates keep their sign and stay ≥ 0.25 from zero, because c·x^ε fakes log x and c·exp(εz) fakes
    z.
  - A shifted input's singularity sits at least 10% of the input's range outside the box.
  - The error used in BIC has a numerical floor (10⁻¹² of the total sum of squares, on the y scale for both
    transforms), so exact data cannot overfit.
  - Formulas with a denominator are pole-checked whenever a term is added (a bug, found in testing: pole-ridden
    formulas filled the beam).
  - With the guide, the search runs both from an empty formula (both transforms) and from the assembled 1-D fits (in
    the guide's transform, at most 4 more blocks), and BIC picks from all the formulas met.
- **2026-10-26, discovery of unknown formulas (the user: what we're doing isn't very useful unless it can uncover
  unknown formulas, which are much harder to find and especially to verify).**
  There is no answer key for an unknown law, but the simulator stands in for nature: it can be queried anywhere,
  cheaply. So a formula is verified the way a law is, by predicting far outside its data and by agreeing with known
  limits it was never shown. Added measures, all for the held-out tasks:
  - **Far extrapolation:** 200 designs with the inputs extended 50% of their range beyond the training box, where the
    design stays feasible (e.g. hole radius up to 1.5× its training maximum, while the hole fits).
  - **Limit checks, where a limit is known and absent from the training data:** for plates, the small-hole limit
    (compliance ratio − 1 → 3πr² as r → 0); for beams, the slender limit (Euler–Bernoulli as L/h grows). Each is
    evaluated from the formula and compared with theory.
  - **A candidate law:** a formula with ≤ 6 terms, in-range NRMSE ≤ 0.02 and far-extrapolation NRMSE ≤ 0.1.
  - Three held-out tasks have no textbook formula (H2, H3, H6). They are the real test. On H1, H4 and H5 the
    textbook law is only the leading part: the simulator's steps and shear add effects no textbook gives.

  | # | added prediction (before any held-out run) |
  |---|---|
  | P11 | A4 yields a candidate law on at least one of H3 and H6; none on H2 |
  | P12 | where A4 yields a candidate law, its known limit (if any) is within 10% of theory |
- **2026-10-26, the stepped beam's solid fraction is now exact** (analytic overlap of each element with each segment's
  band, instead of 16 × 16 sampling). Sampling made outputs change in small steps as a height varied: roughness 0.015 of
  the response's spread, at the "solved" threshold. Exact fractions bring it to 0.0086; the rest is mesh-scale. Other
  outputs are unchanged (no band edges inside elements). The six stepped curriculum tasks were re-simulated and
  refitted in cycle 1.
- **2026-10-26, "solved" respects the simulator's noise.** A task is solved when its training NRMSE is
  ≤ max(0.02, 2 × its roughness). Roughness: 61-point sweeps through the box centre along each input, the residual
  of a degree-8 polynomial fit, over the task's training standard deviation; the largest over the inputs. Reason:
  the beam-with-hole tasks have mesh-scale noise of about 0.04 of their (tiny) response spread, measured
  independently of any fit. A fixed 0.02 would exclude them for the simulator's sake, not the finder's. **Seen when
  this was decided:** cycle 1's training errors, in which several stepped and holed-beam tasks were just above 0.02.
- **2026-10-26, library learning's description length (after sleep 1, before any held-out run).** As specified (each
  parameter costs ½ log n, fixed or free; a template costs its size × log 7), **sleep 1 learned nothing**: of 280
  candidate templates from 46 solved tasks, every one lengthened the description, even one matching 21 terms. A
  template needed about 8 identical uses to pay. That is a mis-specification of the intent stated above
  ("Stitch-style"): Stitch and DreamCoder count every program's whole description, so a template saves its body,
  constants included, at every use. Amended: a constant costs log(120) nats (precision 0.1 over a range of 12)
  wherever it is coded, in a template body if fixed, in each instance if free. The original result is kept
  (`runs/curriculum/sleep1_original_objective.pkl`). **Seen when this was decided:** the cycle-1 formulas and the
  empty sleep-1 library.
- **Implementation decision:** templates never contain constraint-margin factors; margins index a task's own
  constraints, not inputs, so they stay primitives.
- **A limitation found on the way, not fixed:** when the finder fits log y, a power law becomes a linear term
  (log y = −2.999·log h + …). Its exponent is then a linear coefficient, which templates cannot see or learn.
- **2026-10-26, a bug found in the held-out evaluation, fixed to the specification.** The margin term was specified as
  (sⱼ + b)^p with b > 0, its singularity behind the constraint boundary s = 0. The code measured b from the
  smallest margin in the training domain instead. Extrapolated designs that are feasible but closer to a constraint
  than any training design then got a negative base: non-finite predictions in 15 of 252 fits, all on the three
  tasks with margins (H2, H3, H6). Fixed to the specification; every finder-v2 fit on those tasks (A1–A4, 72 fits) was
  rerun; the first results are kept (`runs/curriculum/heldout_results_margin_bug.jsonl`). The curriculum was not
  rerun: the library is frozen, and templates never contain margins.

## Results (2026-10-26; `runs/curriculum_*.txt`, `runs/curriculum/heldout_results.jsonl`, `runs/curriculum/heldout_summary.json`)

**What ran:**
- The curriculum: 48 tasks, 3 wake–sleep cycles, about 20 CPU-minutes.
- A frozen library of 22 templates. They include the bending law L³·h⁻³ (exponents 3.02, −2.98; twice, slots
  swapped), L³, a log-space product law Σ cᵥ·log v, and a group template Σ cᵥ·v⁻³·²⁵. Most others are artifacts of
  the finder's habits (exp(v, 1) and exp(v, 2) are untouched grid starts).
- Training errors did not improve across cycles. Mean training NRMSE by cycle: stepped 0.015 / 0.018 / 0.015,
  plate 0.027 / 0.033 / 0.035, holed beams 0.022 / 0.024 / 0.023.
- The held-out evaluation: 6 tasks × (5 arms + GP) × 2 budgets × 3 seeds, plus the textbook laws on the 3 stepped
  tasks. 7,200 simulations; finder-v2 fits took 5–66 s each.

**n = 100, geometric mean over 3 seeds, NRMSE in range / extrapolation / far extrapolation:**

| task | A0 (v1) | A1 (v2) | A2 (+ guide) | A4 (full) | GP (reference) | textbook law (7 terms) |
|---|---|---|---|---|---|---|
| H1 stepped, 6 segments | 0.26 / 0.75 / 3·10⁵ | 0.24 / 0.47 / 0.51 | 0.13 / 0.30 / 0.34 | **0.087** / 0.22 / 0.29 | **0.087** / 0.33 / 0.31 | 0.096 / **0.088** / **0.075** |
| H2 beam, hole (xc, yc, r) | 0.25 / 0.38 / 0.64 | 0.20 / 8.1 / 4.9 | 0.20 / 8.1 / 4.9 | 0.34 / 3.4 / 6.0 | **0.079 / 0.33 / 0.60** | — |
| H3 plate, hole (r, yc) | 0.0062 / **0.072 / 0.096** | 0.0080 / 0.16 / 0.23 | 0.0070 / 0.079 / 0.12 | 0.0077 / 0.21 / 0.29 | **0.0050** / 0.27 / 0.38 | — |
| H4 stepped, uniform load | 0.077 / 0.24 / 0.18 | 0.059 / 0.30 / 0.68 | 0.046 / 0.18 / 0.78 | 0.058 / 0.26 / 0.60 | **0.032** / 0.19 / 0.18 | 0.048 / **0.070 / 0.058** |
| H5 clamped, 3 segments | 0.094 / 0.21 / 0.21 | 0.11 / 0.59 / 2·10⁶ | 0.11 / 0.59 / 2·10⁶ | 0.087 / 0.46 / 0.64 | **0.019 / 0.19 / 0.10** | 0.38 / 0.77 / 0.78 |
| H6 beam, hole, uniform load | 0.021 / **0.17** / 0.25 | 0.014 / 0.23 / 0.42 | 0.014 / 0.23 / 0.42 | 0.020 / 0.32 / 0.48 | **0.016** / 0.18 / **0.22** | — |

A3 (library, no guide) ≈ A4 everywhere at n = 100. At n = 30 everything is worse, and several finder-v2 fits blow up
in extrapolation (up to 10²¹).

**Verdicts: 2 confirmed, 9 refuted, 1 untestable.**

| # | verdict | evidence |
|---|---|---|
| P1 | refuted | A4 beats A1 by ≥ 2× in range on 1 of 6 (H1, 2.8×); elsewhere 0.58–1.27× |
| P2 | refuted | H1: in range 0.087 (≤ 0.10, as predicted), but extrapolation 0.22 (predicted ≤ 0.15) |
| P3 | refuted, narrowly | A2 / A4 on H1 = 1.54 (predicted ≤ 1.5) |
| P4 | refuted | H4–H6: 1.02, 1.27, 0.70 |
| P5 | confirmed | H2: every arm ≥ 0.20 in range |
| P6 | refuted | the largest finder-v2 extrapolation or far NRMSE is 9.9·10²¹ |
| P7 | refuted, partly works | 61% of failed extrapolations (NRMSE > 1) have spread > 0.1, against 34% of the others (predicted ≥ 80% / ≤ 30%) |
| P8 | refuted, narrowly | curriculum ≈ 20 CPU-min (≤ 30, as predicted); n = 30 → 100 time ratio 1.1 (as predicted); but the slowest held-out fit took 66 s (predicted ≤ 60) |
| P9 | refuted | A4 beats the GP in range on 1 of 6 (H1, by 0.0004) and in extrapolation on 2 of 6 (H1, H3) |
| P10 | confirmed | exponent −2.98 in L³·h⁻³; three group templates |
| P11 | refuted | no candidate law (≤ 6 terms, in ≤ 0.02, far ≤ 0.1) from A4 or A1 on any task. The only one in 90 symbolic fits is v1 on the plate (seed 2) |
| P12 | untestable | no A4 candidate law. The v1 plate law's leading term 9.295·r² matches the dilute limit 3π r² = 9.42 r² (1.3%), but its constant puts y(0) at 0.997, so the limit check at r = 0.02 reads 0.10 instead of 1 |

## Reading

1. **The library worked exactly where the curriculum held the structure, and only there.** On the 6-segment beam, the
   learned group template Σ cᵥ·v⁻³·²⁵ took the error from 0.24 to 0.087 (2.8×), level with the GP in range. That is
   DreamCoder's mechanism doing what it should. But the exponent learned from messy curriculum formulas, −3.25, is
   wrong. Extra terms ride along, so the formula extrapolates worse than the 7-term textbook law (0.22 against
   0.088), and at the slender limit it is 83× off where the law is 4% off.
2. **The central failure is overfitting below the simulator's noise.** The simulator is nearly noise-free, so BIC keeps
   rewarding terms that fit its discretization. With version 2's rich vocabulary (continuous exponents, exp, shifts,
   rationals), formulas grow to 8–16 terms, extrapolate badly, and sometimes blow up. Version 1's smaller vocabulary
   extrapolates better on the plate and both holed beams. The library then learns from those bloated formulas, so
   most of its templates are artifacts: garbage in, garbage out.
3. **Compact laws exist and extrapolate; the finder doesn't find them.** On the stepped beams the textbook bending law,
   with only its coefficients fitted, has the best extrapolation of anything (far 0.075 and 0.058) and passes the
   slender limit (1.04, 1.03). Selection that chases accuracy below the noise floor walks past it.
4. **Discovery of unknown formulas: not achieved.** No candidate law on the three tasks without a textbook formula,
   except one from version 1 whose leading coefficient is right and whose constant is not.
5. **The ensemble's spread is a weak but real failure signal:** 61% of failed extrapolations are flagged, against 34%
   of good ones.

**What it says to change** (not built):
- **Fit to the noise, not below it.** Each task's simulator roughness is measured. Use it as the noise level in the
  likelihood (BIC with known σ), or stop adding terms once the residual reaches it. This is the main lever: it should
  cut the bloat, the artifact templates and the blow-ups together.
- **Learn the library from parsimonious formulas only,** e.g. the smallest frontier formula within the noise.
- **Pin known limits where the task states them.** A hole of radius 0 changes nothing, so the intercept is fixed by
  the physics; v1's plate law failed only on that.
- **Expose exponents that log-y fits hide in linear coefficients** to template learning.
- **Check safety over an enlarged box,** or abstain where the ensemble disagrees.

# Experiment B2: fit to the noise floor (pre-registered 2026-10-26, before the change was built)

**Why (the user, 2026-10-26):** "build the noise-floor fix and rerun". Experiment B's central failure: BIC, with the
noise level estimated from the residual, kept adding terms that fit the nearly noise-free simulator's mesh artifacts,
so formulas bloated, extrapolated badly and sometimes blew up, and the library learned the bloat. **This design is
informed by B's held-out results.** The held-out data are reused (they already exist; the curriculum and library code
never read them).

**The one change:** selection uses BIC with the noise level known instead of estimated:
- BIC = Σ (ŷ − y)² / σ² + (parameters) · log n − 2 log P(terms | library), with residuals on the y scale for both
  transforms, so no Jacobian term is needed.
- σ = max(the task's measured roughness, 0.002) × the standard deviation of its training y. The floor 0.002 sits
  just above the uniform beams' roughness (≤ 0.0013). The textbook laws fit those tasks' data to ≤ 10⁻⁴ (checked on
  5 curriculum tasks), so the floor asks for the law and no more.
- Roughness is measured as in B: 61-point sweeps through the box centre along each input. It is now also measured for
  the held-out tasks, as part of each task's description, like its constraints.
- A term is now kept only if it lowers the squared error by more than σ² log n, about 4.6 σ² at n = 100.

Everything else is unchanged:
- the search, the guide, library learning (with B's amended objective), and the "solved" rule;
- the curriculum is rerun from scratch (3 cycles) and the library re-frozen;
- arms A1–A4 are refitted on the held-out tasks;
- A0, the GP and the textbook laws are unchanged and reused from B, not rerun.

| # | prediction (before the change was built) |
|---|---|
| Q1 | **Parsimony:** the median size (terms, constant included) of A4's held-out formulas at n = 100 is ≤ 6 (B: 8–16) |
| Q2 | **Safety:** no finder-v2 fit at n = 100 has extrapolation or far NRMSE > 10 (B: up to 2·10⁶) |
| Q3 | **Extrapolation:** A4's far NRMSE at n = 100, geometric mean over the 6 tasks, is at least 1.5× lower than B's (0.67 → ≤ 0.45) |
| Q4 | **The cost in range is small:** A4's in-range NRMSE at n = 100 is within 1.5× of B's on at least 5 of 6 tasks |
| Q5 | **Discovery:** at least one candidate law (≤ 6 terms, in ≤ 0.02, far ≤ 0.1) on H3 or H6 from any of A1–A4 at n = 100 (B: none) |
| Q6 | **The stepped beam:** A4's far NRMSE on H1 at n = 100 is ≤ 0.15 (B: 0.29; the textbook law: 0.075) |
| Q7 | **A cleaner library:** fewer than B's 22 templates, still including a bending template L³·h⁻³ (exponents within 0.1) |
| Q8 | **The yardstick (reference only):** A4 beats the GP in far extrapolation on at least 3 of 6 tasks at n = 100 (B: 2) |

    python -m regression.claude.curriculum roughness_heldout
    python -m regression.claude.curriculum --variant nf wake 1   (then sleep 1, wake 2, sleep 2, wake 3, sleep 3)
    python -m regression.claude.curriculum --variant nf evaluate
    python -m regression.claude.curriculum --variant nf report

## B2 results (2026-10-26; `runs/curriculum_nf_*.txt`, `runs/curriculum_nf/heldout_results.jsonl`)

**What ran:**
- The curriculum was rerun with the noise level known: 3 cycles, about 40 s each, all 48 tasks solved.
- The library has 13 templates (B: 22). They are mostly physical: pow(v, −2.97) (h⁻³), pow(v, 2.98) (L³),
  pow(v, 3.97) (L⁴), and products carrying h^−2.98 or h^−3.01 alongside a fitted power of another input. A few
  artifacts remain (three exp rates, two shifts).
- On the held-out tasks, A1–A4 were refitted; A0, the GP and the textbook laws were reused from B.

**n = 100, geometric mean over seeds, in range / extrapolation / far (B → B2):**

| task | A4 in B | A4 in B2 | GP (B) | best B2 arm far |
|---|---|---|---|---|
| H1 stepped, 6 segments | 0.087 / 0.22 / 0.29 | **0.069** / 0.21 / 0.30 | 0.087 / 0.33 / 0.31 | A2 0.19 (textbook law 0.075) |
| H2 beam, hole (xc, yc, r) | 0.34 / 3.4 / 6.0 | 0.63 / 1.0 / 24 | **0.079** / 0.33 / 0.60 | A4 24 (GP 0.60) |
| H3 plate, hole (r, yc) | 0.0077 / 0.21 / 0.29 | 0.0051 / 0.16 / 0.29 | **0.0050** / 0.27 / 0.38 | A4 0.29 (v1 0.096) |
| H4 stepped, uniform load | 0.058 / 0.26 / 0.60 | **0.029** / 0.11 / 0.21 | 0.032 / 0.19 / 0.18 | A1 0.14 (textbook law 0.058) |
| H5 clamped, 3 segments | 0.087 / 0.46 / 0.64 | 0.050 / 0.53 / 0.91 | **0.019** / 0.19 / 0.10 | A2 0.50 |
| H6 beam, hole, uniform load | 0.020 / 0.32 / 0.48 | **0.015** / 0.12 / **0.17** | 0.016 / 0.18 / 0.22 | A3/A4 0.17 |

Formula sizes at n = 100 (median terms): H3 9 → 5. Every other task stays at 8–9 (B: 8–14).

**Verdicts: 2 confirmed, 6 refuted.**

| # | verdict | evidence |
|---|---|---|
| Q1 | refuted | median A4 size 9 at n = 100 (predicted ≤ 6); only the plate shrank (9 → 5) |
| Q2 | refuted | the largest finder-v2 extrapolation NRMSE at n = 100 is 9·10¹⁹ (A1 on H5); at n = 30, some predictions are non-finite |
| Q3 | refuted | A4's far NRMSE, geometric mean over tasks, 0.67 → 0.64 (1.05×; predicted ≥ 1.5×) |
| Q4 | **confirmed, and better than predicted** | A4's in-range error is lower than B's on 5 of 6 tasks (×0.50–0.79); H2 is worse (×1.86) |
| Q5 | refuted | no candidate law |
| Q6 | refuted | H1 far 0.30 (predicted ≤ 0.15) |
| Q7 | refuted as worded | 13 templates (< 22), but no template fixes both L³ and h⁻³; each bending template keeps one exponent free |
| Q8 | **confirmed** | A4 beats the GP in far extrapolation on 3 of 6 (H1, H3, H6), and in range on 3 of 6 (H1, H4, H6). B: 2 and 1 |

**Reading:**
1. **Fitting to the noise helped accuracy, not parsimony.** In-range errors fell on 5 of 6 tasks, by 1.3–2×, and
   the full system now beats the GP on half the tasks, in range and in far extrapolation, which B never did.
2. **Held-out formulas stayed large because the structure is real.** The held-out responses carry effects well above
   their noise (the stepped beams' interactions at the steps). The vocabulary can express them only as many awkward
   product terms, so formula size is now limited by representation, not by fitting noise.
3. **Catastrophic extrapolation remains.** It comes from terms like h^4.55 or r^6 inside products: tame in the box,
   explosive outside it. The safety check covers the design domain only, so it cannot see this. Growth outside the
   box has to be controlled directly: an enlarged check region, bounded exponent magnitudes, or abstaining where the
   frontier disagrees.
4. **The cleaner library is used little on the held-out tasks** (A3 ≈ A4 ≈ A1 on several).
5. **Still no candidate law.** The textbook bending law is still the best extrapolator on the stepped beams.

## Exploratory: estimating the noise from the dataset alone (the user, 2026-10-26: "Do we have a good way to discover the noise in a dataset? Maybe something like fluctuations where no progress can be made on learning their structure, related to epiplexity?")

The noise floors above came from sweeping the simulator along lines (active probing). Two estimators that use only the
training data were tested against those sweeps, on 54 tasks (`regression/claude/gamma_test.py`):
- **The Gamma test** (nearest-neighbour differences, extrapolated to zero distance; model-free) fails at these sizes.
  It overestimates by a median 8.6× on the noisy tasks, and reads 0.1 on noise-free ones. With 50–100 points in 1–6
  inputs, neighbours are far apart and the responses are strongly curved, so the extrapolation to zero distance is
  not linear.
- **A GP's fitted white-noise level** (model-based) works about half the time. It reads ~0 on the noise-free tasks;
  on noisy tasks the median ratio to the sweep value is 0.66, within a factor 2 on 50%. It matches closely on most
  stepped beams, and collapses to 0 on a few tasks by absorbing the noise into a wiggly fit.

**The conceptual point:** from a fixed, finite dataset, noise is defined only relative to a learner. Whatever no
learner within reach can compress looks like noise. That is the epiplexity split: structure is what a bounded learner
can extract; time-bounded entropy is the rest, so learning progress goes to 0 on noise and on structure out of reach
alike. A learning curve's plateau measures noise plus out-of-reach structure, and only access to the data source can
separate the two: replicates, or probes at small scales, where smooth structure vanishes but noise doesn't. A
simulator allows that; a passive dataset does not.
