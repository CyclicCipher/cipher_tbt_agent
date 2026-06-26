# CLAUDE.md — Cipher's TBT Agent

## What this project is

A from-scratch AI agent built on the **Thousand Brains Theory** (cortical columns). One reusable column learns
the structure of any domain as a navigable geometry (the successor-representation eigenframe = grid cells);
composed into a small neocortex (task ⊕ space columns + thalamus + basal-ganglia gate), it plays an
ARC-AGI-3-style game, does exact arithmetic, and models language — **no hand-coded rules, no gradient training
on a big dataset.** Target: the **ARC Prize 2026 / ARC-AGI-3** interactive-reasoning benchmark. `README.md` is
the public-facing description; this file is the working memory.

The repo was pruned (2026-06) to just the TBT agent + its deps + the language corpora. The old experiment lines
(CTKG/symbolic-AI, Mamba/JEPA, the DEQ "settling core") were removed.

## Architecture

**The column — `experiments/RecurrentWorldModel/tbt/`.** ONE mechanism: learn a structural map, predict from it.
- `column.py` — L6 = SR-eigenvector location frame (`_sr_frame`), L5 = displacement operators, L4 = content
  codebook, L23 = object memory. `observe`/`consolidate` discover structure online; `learn_domain` takes it
  given; `predict`/`add`/`infer`/`anchor` read from it. L6 can also be driven as a **dynamic recurrent state**
  (`loc_reset`/`loc_move`/`loc_sense`/`loc_where`) — path integration.
- `recurrence.py` — `SelectiveRecurrence`: the ONE canonical selective-gated recurrence
  (`h = gate⊙A(h) + (1−gate)⊙drive`, per-channel learned gate). Used by BOTH L6 (A = the L5 operator) and the
  language SSM (A = identity, drive = content). **Never reimplement it per experiment.**
- `thalamus.py` — cross-column routing (`bind`/`read`/`read_location` — the goal-state channel).
- `basal_ganglia.py` — emergent allocation (Go/NoGo + dopamine-RPE; roles EMERGE, not assigned).
- `reward.py` — domain-agnostic critic/planner. `factorize.py` / `residual.py` / `dynamics.py` —
  disentanglement / recursive-residual structure learning / conditional-effect (dynamics) learning.

**The demos — `tbt/precursor/`** (runnable validations): number line, arithmetic (incl. carry/place value),
2-D/tree, `multicolumn` (emergent allocation), disentangle, passive, residual, dynamics,
`language`/`language_active`/`language_recurrent` (the language probes), `control_loop` (task ⊕ space additive
vs 2^K + emergent allocation), `recurrent_location` (path integration). `scaling_probe.py` = the flat-planner
2^K wall; `unified_demo.py` = multi-domain non-interference.

**The ARC agent — `experiments/ProgramSynthesis/`.**
- `agent/column/` — the TBT agent on the LockPath replica: `control_agent` (flat joint-BFS baseline),
  `dynamics_perceive` (learn the rules from play), `multicolumn_agent` (task ⊕ space — 4/4, 96.5%),
  `recurrent_agent` (partial-observability path integration), `perceive`, `column_score`.
- `agent/wm/` — the **predecessor symbolic world-model agent** (perceive→induce→infer-goal→plan, the direct
  ancestor) + the scorer (`score.py`) the column agent uses. Kept as a reference.
- `arc_agi_3/` — the LockPath game replica (mirrors the real ARC-AGI-3 agent API). `agent/layouts.py` —
  procedural LockPath generator (harder boards).

The corpora live in `corpora/` (Latin, Middle/Old High German); `precursor/language.py` `CORPUS` points there.

## Key results (all learned, no hand-coded rules)
- ARC LockPath replica: **4/4 levels, 12/12 seeds, 96.5% RHAE-proxy** (`multicolumn_agent`).
- Language (pooled Latin + MHG + OHG): the column's SR-frame IS a word embedding; recurrent next-token
  **PPL 152** (Markov 181, passive 166, bigram 164); coherent geometry; the learned gate finds that
  prepositions reset context.
- Arithmetic: exact, by navigation; generalizes to unseen multi-digit numbers (place value + carry, learned).
- Partial observability: the recurrent agent solves L0/L1 efficiently where a memoryless ablation fails
  (10–15× the actions).

## Active priority: real ARC-AGI-3
The replica feeds clean symbolic input. Real ARC-AGI-3 gives raw frames + a sparse score, no goal. Two frontier
pieces sit in front of the (done) control loop + learned dynamics + recurrence:
- **E (perception):** raw frame → objects/roles (currently hand-fed in `perceive.py`). The gate to generality.
- **F (value):** learn the reward/win-condition from the sparse score (the MuZero parallel).
Testing on real ARC-AGI-3: `pip install arc-agi`, wrap the agent as `choose_action(frames, latest_frame)`, run
against the games (Community leaderboard / Kaggle competition).

## Workflow & conventions
- Work directly on `main`. **Always activate the venv** (`venv/Scripts/activate` on Windows) before running.
- Run demos as modules from their experiment folder, e.g.
  `cd experiments/ProgramSynthesis && python -m agent.column.multicolumn_agent`;
  `cd experiments/RecurrentWorldModel && python -m precursor.language_recurrent`.
- **Never run heavy training on this machine** — the CPU demos are fine; large training is a GPU job for the user.
- One file = one concept. **No hand-coded rules / domain priors** (the bitter lesson). **Never reimplement core
  machinery per experiment** — extract one canonical component and USE it (e.g. `tbt/recurrence.py`).
- The folder names `RecurrentWorldModel`/`ProgramSynthesis` are historical (kept to not break import paths).
- Hardware: RTX 3050 Ti (4 GB VRAM); everything fits in 4 GB. CPU is sufficient for the demos.
