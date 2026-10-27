# DreamCoder: how it works, and what it offers the function finder

Reading notes (2026-10-26) on Ellis et al., "DreamCoder: Growing generalizable, interpretable knowledge with
wake-sleep Bayesian program learning" (arXiv 2006.08381; PLDI 2021), and on its successor for library learning,
Stitch (Bowers et al., POPL 2023, arXiv 2211.16605). The constraints for using it here (the user): stay in
polynomial time, and no Gaussian process.

## How DreamCoder works

A **program** is a term in a typed λ-calculus built from a **library** L, which starts as a few primitives. A
probabilistic grammar over L gives every program a prior P[ρ | L]; short programs in the current library are
probable. A **task** x is a small data set (5–10 examples in most domains). DreamCoder alternates three phases.

**Wake: solve tasks.** For each task, enumerate programs in decreasing probability under a recognition model
Q(ρ | x) until a timeout. Keep those that solve the task, scored by the posterior P[x | ρ] · P[ρ | L]. This is
best-first enumeration, so it is exponential in program size. The paper's own footnote says search difficulty is
roughly breadth^depth: depth is the program's size, and breadth is the number of likely library functions at each
choice.

**Abstraction sleep: grow the library.** Add routines to L that compress the programs found while waking. The
objective is the description length of the library plus, for each task, the description length of the best
refactoring of its programs in the new library, Σ_x min −log P[x | ρ] P[ρ | L]. "Refactoring" means rewriting a
program into a semantically equal one that exposes a reusable piece; it is bounded to 3 λ-calculus evaluation
steps. The set of refactorings is astronomically large (about 10^14 for the paper's map example). A version-space /
e-graph data structure represents it in about 10^6 nodes, built by a dynamic program in minutes: polynomial in
program size, exponential in the step bound. Routines are added greedily while the objective improves.

**Dreaming sleep: train the recognition model.** Train a neural network Q(ρ | x) on (program, task) pairs. Half are
replays (tasks solved while waking, with their programs). Half are fantasies (programs sampled from the library's
grammar, run on sampled inputs to make synthetic tasks). Q outputs the grammar's weights for this task, i.e. which
library functions to try first.

**Why it works**, in the paper's terms: the library reduces the depth of search (higher-level routines mean fewer
calls), and the recognition model reduces its breadth (fewer likely choices per call). The two bootstrap each other:
a richer library makes richer fantasies, and a better recognition model solves more tasks, which feed the next
library.

**Results relevant here:**
- **Symbolic regression:** primitives +, ×, ÷ and real-valued parameters, optimized by inner-loop gradient descent
  and penalized by BIC. Given data from 1.7x² − 2.1x + 1.8, it infers a program with three parameters.
- **Physics:** 60 laws and identities from AP/MCAT cheat sheets, given as numerical data, starting from map, fold and
  arithmetic. After 8 wake-sleep cycles it learns 93% of them. It first learns vector algebra (inner products,
  sums, norms), then an inverse-square schema that gives both Newton's gravitation and Coulomb's law: "a change of
  basis" from sequence processing to physics.
- **Cost:** 20–100 CPUs per domain. One experiment took about five days on 64 CPUs; another, about a year of CPU
  time in total. Stitch finds abstractions 3–4 orders of magnitude faster than DreamCoder's compression, with 2
  orders less memory and comparable or better compression, by synthesizing abstractions top-down and pruning.

## What each piece offers here

The function finder is already the polynomial form of DreamCoder's wake phase, for formulas whose terms add. It
builds a k-term formula one term at a time on the residual: about k · M evaluations for a library of M terms,
instead of M^k for joint enumeration. That is the user's original lesson from Gemini's scripts, residual fitting
turns O(B^d) into O(d · B). The rest of DreamCoder sorts as follows.

| piece | keep? | why, and its polynomial form |
|---|---|---|
| wake: best-first enumeration | **no** | exponential in program size. Keep the residual-greedy finder |
| abstraction sleep: library learning | **yes, the main gain** | it attacks the measured failures. A pattern found once on an easy task becomes one library term, so a greedy finder takes it in one step. The stepped beam's law, one term per height, would be a single routine learned from simpler beams (limitation 6, templates). A combined variable like a hole-to-edge gap, once found, is reusable (limitation 1). The search gets deeper without becoming exponential. Polynomial form: our formulas are sums of parametric terms, so abstraction is anti-unification of terms across solved tasks (the least general generalization of two terms, linear time per pair), plus greedy description-length selection. No version space needed |
| the library's prior and Bayesian scoring | **yes** | a posterior with a BIC penalty per continuous parameter replaces the one-standard-error rule (limitation 5). The prior learned from earlier tasks says which terms are plausible |
| the frontier: several good programs per task | **yes** | DreamCoder keeps the top programs per task, weighted by posterior. That is the ensemble the plate needed (limitation 4): where good formulas disagree outside the data, say so |
| inner-loop fitting of continuous parameters | **yes** | the same idea as terms with continuous parameters (limitation 1). Variable projection is cheaper than gradient descent: the linear coefficients are solved exactly, the few nonlinear ones by Gauss–Newton with a fixed iteration count |
| dreaming: recognition model + fantasies | **yes, non-neural first** | the job is to rank library terms for this data before fitting. Cheap statistics can do it: log–log slopes estimate power-law exponents; a separability test says whether two inputs interact; equal slope estimates across inputs flag exchangeable inputs (the stepped beam's heights); steepening toward an edge flags a singularity just outside the domain. Fantasies (formulas sampled from the library and evaluated, no simulation needed) calibrate these statistics. A neural network can come later; inference is polynomial either way |
| the wake-sleep loop | **yes** | each phase above is polynomial, so each cycle is |

**What DreamCoder doesn't provide** (from our list of limitations): singularity safety over the design domain (3),
gated formulas for local effects (7), and noise handling. The paper names messy data as an open problem.

**What it needs that we lack:** many related tasks. DreamCoder trains on 100–200 per domain. A library learned from
4 problems is not a library. The simulator can make families cheaply: beams, plates and brackets under different
supports, loads, materials and hole layouts.

## A test that would show whether it works (not designed in detail yet)

Train the library on a curriculum of simulated tasks: uniform beams, two- and three-segment beams, plates with one
hole, and so on. Hold out harder relatives, such as the six-segment beam and the bracket. Measure on the held-out
tasks: accuracy in range, extrapolation, formula size, and run time against the current finder with primitives only.
DreamCoder's claim, transferred: the learned library lets a polynomial-time finder solve the held-out tasks that it
cannot solve from primitives.
