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

## Stitch: library learning by top-down synthesis (checked against the paper)

**The problem:** given a corpus of programs, find the abstraction (a function body whose holes become arguments)
that most compresses the corpus once every program is rewritten to use it.

**The method, corpus-guided top-down search:**
- Start from the most general partial abstraction, a single hole `??`. Refine it step by step, replacing a hole with a
  primitive applied to new holes, or with an argument variable.
- Each partial abstraction has match locations: the subtrees in the corpus it fits. Refining a hole can only shrink
  this set (the subset property).
- So the matched subtrees' total size is an upper bound on the compression any refinement can achieve.
- Branch and bound: discard any partial abstraction whose bound is below the best complete abstraction found so far,
  and expand the most promising bounds first. Two more pruning rules keep the result optimal, and the exact utility
  accounts for an argument used several times and for overlapping matches.
- One abstraction per iteration: rewrite the corpus with it and repeat. Later abstractions build on earlier ones,
  giving a hierarchical library.

**Against DreamCoder:** Stitch matches syntactically. It doesn't explore DreamCoder's semantic refactorings (up to
3 steps of inverse β-reduction, in a version space), and that is the trade: 3–4 orders of magnitude faster, 2 orders
less memory, compression as good or better. It is exact, and anytime (it can be stopped early with a useful result).

**Polynomial?** Branch and bound is exponential in the worst case, in the size of the abstraction. With a cap on
abstraction size the count of candidate bodies is a constant, so it is polynomial in the corpus, as DreamCoder's
bound on refactoring steps is. For our formulas (sums of parametric terms), a capped Stitch or plain
anti-unification of terms is enough.

## A guide without a neural network

What the guide has to do is rank which library terms a task needs before fitting. Three non-neural ways, all
polynomial:

1. **Deduction (witness functions).** Push the target down through an operator by inverting it. FlashMeta/PROSE
   (Polozov & Gulwani, OOPSLA 2015) builds its synthesizers this way.
   - For a sum the inverse is the residual, y − A, which is the symbolic ResNet.
   - For a product it is the ratio y / A; for a power, the root; for exp or log, the other one.
   - Gemini's `regression/gemini/test_target_prop.py` is this idea: invert layer 2 to get layer 1's targets.
   Each inversion turns one hard problem into smaller ones in O(n).
2. **Structure statistics, computed from the data directly:**
   - log–log slopes along each input estimate power-law exponents;
   - a functional ANOVA decomposition (Sobol indices, or HDMR fitted from random samples) shows which inputs matter
     and whether the response, or its log, is additive. If it is, the problem splits into 1-D fits, which keeps the
     search polynomial in high dimension;
   - mixed second differences of log y test separability, y = g(x)·h(z);
   - equal component shapes across inputs flag exchangeable inputs (a template);
   - steepening toward a boundary locates a singularity just outside the domain (a shifted term).
   These are the engineer's heuristics, written down.
3. **A guide learned by the system itself (the meta step).** Treat the guide as a program from the statistics in 2
   to term weights, and learn it with the same machinery: description-length compression over (statistics →
   solution) pairs, from replays and fantasies. A rule like "additive, slopes near −3 → try Σ cᵢxᵢ⁻³" is kept if it
   shortens the descriptions of solutions, i.e. if it saves search. The simplest version is case-based: retrieve the
   solved tasks with the most similar statistics and up-weight the terms they used. The result is interpretable,
   and it is the abstraction phase applied one level up: the library compresses solutions; the guide compresses how
   solutions are found.

On the stepped beam, 1 and 2 alone would likely have been enough. The response is close to additive in the six
heights, so it splits into six 1-D fits. Each 1-D slope is near −3, so each fit proposes h^-3. All six shapes match,
so the template follows.
