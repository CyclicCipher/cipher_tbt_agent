# Neural Turing Architecture — status and handoff

The entry point for a new session. Last updated 2026-10-06. All of this work is on branch **`exp/mdl-vs-ntp`**
(not merged to `main`); the user pulls it to their machine.

## What this experiment is

A from-scratch architecture: a looped transformer (Ouro-style depth recurrence; Chen et al.'s boundary operator vs
Block Attention Residuals across passes) that thinks in continuous thoughts along the sequence (Coconut) learned
WITHOUT a substitution curriculum or human reasoning traces — the learning signal supplied by SEARCH over thoughts
(EfficientZero-V2-for-thoughts, GCML-inspired components) — with Percepta's Spotlight memory as a candidate sequence
mixer and guards against collapse.

**Nothing of the architecture is built yet.** The work so far is design (the documents below) and CPU experiments on
a mock thought space, about the search and learning signal. ZipLearn (the experiment this branch began with) is not
being continued (user, 2026-10-03); the Thousand Brains agent in the rest of the repository is no longer the main focus.

## Read in this order

1. **This file.**
1b. **`BRAINSTORM.md` §0.1 — the standing principles and insights.** Every decision and lesson so far in one list;
   read it before designing anything, and add to it when something new is learned.
2. **`CURRICULUM_LESS_COCONUT_AND_SEARCH.md`** — the planning document and lab book. §1–§10 the design of search over
   continuous thoughts; **§11–§18 every experiment**, each with its setup, pre-registered predictions, results tables,
   verdicts and caveats; **§19 the current plan (2026-10-06): one general method** — the learning-theory lens, the
   zone of proximal development, Ataraxos, and the 2 × 2 P1(b).
3. `BRAINSTORM.md` — the architecture questions: §1 AttnRes vs the boundary operator (from ZipLearn's E30), §2
   Spotlight, §3 learning thoughts without a curriculum, §5 decisions still open for the user.
4. `REFERENCES.md` — every source with a verification status; `refs/` — notes from full reads (EfficientZero V2,
   Spotlight Memory).

## Decisions the user has made (do not reopen)

- **Q1 — a codebook is not a continuous thought.** The thought stays a free vector in ℝ^d; discrete/soft codebook
  thought spaces (A3, A4) are out. Learned prototype thoughts are fine as search CANDIDATES, not as a constraint.
- Keeping the thought deterministic and letting search vary only a few controls (D4) "could limit the model" — dropped.
- **Q2 — tree search** is the more likely family (the benchmarks agree).
- **Q3 — answer-gradients at training time: "we should experiment"** (done in the mock, §16).
- The pipeline of §14 (search learns a domain → GCML as a speedup where a map exists → general structure over time)
  and the escalating tiers of §17.5 as a division of labour by structural difficulty — both agreed with the user.
- Working rules: **no background agents (subagents)**; long CPU runs were run as background shell jobs. Commit and
  push to `exp/mdl-vs-ntp`; do not open PRs unless asked.
- **2026-10-06 — less focus on GCML and the TBT-related questions** (cognitive maps, the tiers, operator codes) in
  future work. §12–§18 stand as recorded; they are not the main line.
- **2026-10-06 — no written weights for this architecture until the architecture is settled.** Everything is
  trained; E46-style weight-writing waits until the design is final.
- **2026-10-06 — one general search method, not a set of specialists.** No dispatcher that decides which algorithm a
  piece of data needs (commuting or not, geometric or not): that cannot be classified at scale and overcomplicates
  the architecture. Structure belongs in what is LEARNED (policy, value, representation), under one search.
- **2026-10-06 — BRAINSTORM §5 and planning §9 answered** (the user agreed to the recommendations): from scratch on
  synthetic tasks; outcome-only, machine execution states as probes only; pointer chasing first, then Brainfuck;
  attention first, Spotlight after NTA-M; import `h1_lid.py`; amortise (test-time search optional, the same update as
  training); partial credit allowed with exact match dominant; a problem-difficulty frontier — yes; candidates = the
  policy refined by gradient, no GCML; search only on frontier problems.
- **2026-10-06 — no Coconut-faithful baselines.** Compute goes to our architecture only (P0's `plain6` arm and
  P1(b)'s architecture arm dropped before running).

## Results so far (one line each — the tables are in the planning document)

| § | experiment | finding |
|---|---|---|
| 10.2 | dimensionality toy (`research/dim_search_toy.py`) | best-of-K random thoughts recover ≈ E[max_K]/√d of a gradient step: sampling search dies with d |
| 11 | search benchmark on a mock thought space (fixed dynamics, mocked value) | sampling-only search (EZ-V2 candidates, CEM, SMC) scores 0.00 from d = 64; gradients through the dynamics are necessary; a real tree with backtracking beats one-step lookahead; `mcts_hybrid` is the best all-rounder, `grad_greedy` best at budgets ≤ 128; the value's quality dominates |
| 12–13 | GCML's inverse model W(goal − state) on a structure dial | excellent and nearly free where geometry exists (grid: progress 0.88–1.00 vs the value gradient's 0.54–0.59); fails on random graphs, even with global action semantics; learn W from SEARCH experience (ridge or a network); the k-step probe is the precondition test (k = 3: 0.40–0.43 where GCML works, 0.03–0.08 where not) |
| 14–15 | geometry test | a code learned with GCML's own eq-11 objective recovers the coordinates (grid R² 0.98–1.00) if its dimension matches; generalises across the parts of a product world; planner success falls with the number of axes a goal needs; on S₅ every additive code fails |
| 16 | training loop: a policy and a value learned from scratch, six improvement operators at equal thought-steps | RLOO and best-of-N learn nothing from d = 64; **Q3: back-propagating the answer's log-likelihood through the thoughts gives the best search-free policy (0.72–0.73)** — in a mock with fixed dynamics and decoder, so the real test (P1(b)) is still due; the best test-time solver is a tree on a value trained by its own Bellman backup (0.86); labelling explored states as failures collapsed tree values (a confound found and fixed); any gradient through the dynamics finds the effective subspace (cold start); `pi_grad` (policy sample + value-gradient step) is the best all-rounder; a tree is a poor teacher (distribution shift, hypothesis) |
| 17 | past commutativity — the four tiers | an additive code sees only the group's abelianisation; tiers: 0 additive/GCML, 1 factor + state-dependent inverse, 2 matrix (operator) codes, 3 search + macros. Heading world (egocentric moves and turns): a global W fails (progress 0.42–0.49 vs 0.91–0.95 on the grid), a heading-gated W recovers (0.90), a state-conditioned network not told the factor does best (planner 0.69–0.83 at budget 64); with action labels a learned 4-dim matrix code IS the position + heading code (R² 1.00); the exact S₅ matrix code path-integrates perfectly but lookahead in it solves 0.00–0.07; a model, a metric and a planner are three different things |
| 18 | operator codes for continuous thoughts (a gated mixture of operators) | pre-registered test **refuted**: learned from search + random thoughts, the code is not learnable (the gate is); exploratory follow-up — explore with the model's own prototype thoughts, refit: S₅ path-integrates like the exact code (0.85–0.91), grid 0.62–0.72, heading world partly (0.25–0.33); a metric estimated from the same exploration matches the true SR (R² 0.82–1.00); lookahead in the learned code solves 0.83–1.00 of grid and S₅ problems at budget 64 (above `mcts_hybrid` at 256); tier 3 is a matter of scale |

**Recommendations standing for the real model** (from §11.4, §13.6, §16, §17.8, §18.6): a gradient-proposing tree
(`mcts_hybrid`-like) as the search; train the value on search data with a Bellman backup; run the policy's own chain
before searching; the answer's gradient through the thoughts as the base policy signal, guarded by a thought ablation;
GCML only where the k-step probe says geometry exists; a state-conditioned inverse network for tier 1; explore with a
learned action repertoire, not isotropic noise; learn an operator code and a metric from that exploration.

## Next steps (2026-10-06 — planning doc §19.5)

1. **The real model, now the main line:** P0 (the no-thought ceiling on the diverse suite, planning §20) → P1(b) on
   our architecture (mixed levels vs hardest only; pre-register numbers first) → E-dim2 (the effective dimension of a
   real thought).
2. Optional, minutes: §16's `bptt` arm on L = 4 only (does the mock leak partial progress?).
3. Implement §19.4's single method once P1(b) says which learning signal survives.

Deprioritised 2026-10-06 (GCML / tier / operator-code threads; kept for the record): the heading world's operator code;
state-dependent operator codes; the tree as a teacher by relabelling; choosing the code's dimension; `mcts_guided` and
`smc_grad` as training operators; a GCML arm in a geometric world; the k-step probe on the real model.

## Code and data — the real model (`thinking/`)

| file | what |
|---|---|
| `tasks.py` | the diverse task suite (§20): six families (`ptr`, `s5`, `bool`, `ca`, `bf`, `aff`) × 8 depth levels, one shared vocabulary, `make_batch` |
| `p0_ceiling.py` | P0: train a no-thought model (`--arm loop --loops K` or `--arm plain --layers L`) on the suite; per-(family, level) normalised accuracy at mid and end; `h_solved`, `h_fail` |
| `runs/p0/` | P0 results (`*.json`, `*.log`); Brainfuck pools and checkpoints are `*.pt` (gitignored, rebuilt on demand) |

## Code and data — the mock (`search_bench/`)

| file | what |
|---|---|
| `mock_task.py` | the mock thought space: `TaskConfig`, `GraphWalk` (thought z ∈ ℝ^d acts through a hidden 8-dim projection; soft edge choice; verifier), `Sim`, `Budget` (a thought-step costs 1, a gradient through one 3) |
| `searchers.py` | the §11 algorithms (`ALGOS` registry; `mcts` with expansion rules `exp_hybrid`, `exp_guided`, …) |
| `structured_task.py` | `StructuredWalk`: a fixed world, varying problems; graphs `random`, `grid`, `grid2`, `perm` (S₅), `heading`; codes `raw`, `sr`, `coord`, `allo`; goal modes |
| `gcml.py`, `searchers_gcml.py` | inverse models (ridge, Hebbian, network, `GatedInverse`), experience collection, the k-step probe, `learn_code_als`; GCML planners |
| `train_loop.py` | §16 training loop (`--vlabel path\|bellman`) |
| `matrix_code.py` | §17.7 Part B: additive vs matrix codes with discrete actions |
| `operator_code.py` | §18: `OperatorModel`, path integration, prototype report, `op_look_*` / `sim_look_sr` planners |

| § | runner | data (`runs/`) | tables |
|---|---|---|---|
| 11 | `run_bench.py` | `grid*.json` | `summarize.py` |
| 13 | `run_gcml.py`, `probe_gcml.py`, `probe_kstep.py` | `gcml_*.json` | `summarize_gcml.py` |
| 15 | `run_geometry.py`, `probe_axes.py` | `geometry_*.json` | `summarize_geometry.py` |
| 16 | `train_loop.py` | `train_d*_s*.json` (path labels), `train_bellman_*.json` | `summarize_train.py` |
| 17 | `run_tiers.py` (Part A), `matrix_code.py` (Part B) | `tiers_*.json` | `summarize_tiers.py` |
| 18 | `run_operator.py` (`--explore`, `--emp_sr`) | `operator_{grid,heading,perm}.json`, `empsr_d64.json` | `summarize_operator.py` |

Every run uses fixed seeds; reruns were checked bit-identical where it mattered. Each file's docstring gives its
command line.

## Environment

- CPU only; the only dependency is PyTorch. In a fresh cloud container:
  `python3 -m venv <dir> && <dir>/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu`
  (the runs used torch 2.14 CPU, 4 cores). Run scripts from `search_bench/` or by path.
- Runs take from seconds to ~40 minutes; a foreground command is capped at 10 minutes, so split runs by world or
  seed (`--worlds`, `--seed0`, `--seeds`) or run them as background shell jobs.

## How the work has been done

- Each test is pre-registered in the planning document (expected results and refutation criteria) and committed
  BEFORE its code runs; results are reported against it, and anything designed after seeing results is labelled
  exploratory.
- Every comparison is at equal thought-steps. Claims in the planning document were checked against the tables before
  committing.
- Commit messages end with the `Co-Authored-By:` and `Claude-Session:` trailers (see `git log`).
