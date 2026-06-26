# CLAUDE.md — Cipher's TBT Agent

## ⚠ ACTIVE WORK (read first) — merge to `src/` + reorient to the reusable architecture
The agent had drifted into a 575-line ARC-specific solver (the anti-pattern). We are **killing the silo**: merge
`experiments/ProgramSynthesis` + `experiments/RecurrentWorldModel` into one top-level `src/` package, then
reorient the agent to a **thin shell over the column** (the `tbt/agent.py` 40-line template), with task-format
code quarantined in `perception/`/`tasks/` and planning moved into the column. **The plan + live checklist is
[REORG_PLAN.md](REORG_PLAN.md) — follow it; update its checkboxes as steps land.** Do Part A (pure-move merge)
then Part B (reorient per `tbt/EMERGENT_PLAN.md`). Remove this note when REORG_PLAN.md is fully checked.

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
- `agent/column/` — the ONE unified TBT agent (`unified_agent.py`: `UnifiedAgent` full-obs, `PartialObsAgent`
  egocentric) + its learned pieces: `dynamics_perceive` (`collect` — one play loop driving the three learners),
  `objects`/`object_perceiver` (E: segment → multi-cell objects, body/pushable/blocking from motion),
  `goal_discover` (F: goal + required-absent from the score), `perceive` (frame reading). The agent reads the
  learned mechanics, values RL/MuZero sub-goals (`reward.py`) gated by the basal ganglia, routes them via the
  thalamus, navigates the spatial column, and path-integrates the body via the recurrence. No per-mechanic code.
- `agent/wm/` — the **predecessor symbolic world-model agent** (perceive→induce→infer-goal→plan, the direct
  ancestor) + the scorer (`score.py`) the agent is evaluated with. Kept as a reference.
- `arc_agi_3/` — the LockPath + MultiKey game replicas (mirror the real ARC-AGI-3 API; a GENERIC BFS oracle via
  each game's `snapshot`/`restore`). `agent/layouts.py` — procedural LockPath generator.

The corpora live in `corpora/` (Latin, Middle/Old High German); `precursor/language.py` `CORPUS` points there.

## Key results (all learned, no hand-coded rules)
- ONE unified agent, no per-mechanic code: ARC LockPath **4/4 levels** (RHAE-proxy 59.5%) AND a structurally
  different mechanic it was never tuned for, **MultiKey 2/2 (100%)** — the sub-goal order LEARNED by RL from the
  sparse score (`reward.py` critic + BG gate), not hand-coded. (The old hand-coded `multicolumn_agent` scored a
  higher 96.5% only because it was fed the mechanics — disqualified; removed.)
- Partial observability: the SAME agent (`PartialObsAgent`), egocentric window, solves L0/L1 **8/8** by
  path-integrating the body (the recurrence) + remembering the map, where a memoryless ablation fails (4–6/8,
  5–15× the actions).
- Language (pooled Latin + MHG + OHG): the column's SR-frame IS a word embedding; recurrent next-token
  **PPL 152** (Markov 181, passive 166, bigram 164); the learned gate finds that prepositions reset context.
- Arithmetic: exact, by navigation; generalizes to unseen multi-digit numbers (place value + carry, learned).

## Active priority: harder/diverse replica, then real ARC-AGI-3
E (perception) and F (value) are DONE and folded into the one agent (objects from connected components; the
roles + win-condition learned from the score; sub-goals by RL/MuZero). The agent is general (no per-mechanic
code) but tested on a small replica. Next:
- **Harder + more DIVERSE replica** ("homework harder than the exam"): a procedural multi-mechanic generator +
  a held-out split, slightly above real ARC on each axis (perception load, reward sparsity, mechanic depth, the
  click action). Breadth (unseen mechanics) is the axis a single game can't fake — the real test of generality
  before the SDK.
- **Efficiency** (parked): the general agent is ~59.5% on LockPath vs the hand-coded 96.5% — the cost of the
  online door-bump + the factored cover-navigation; improve later.
- **Real ARC-AGI-3**: 64×64×16-colour frames, multi-cell objects, a click action, level-completion-only signal,
  135 multi-mechanic games. The interactive SDK is the **ARC-AGI-3-Agents** repo + an API key (the pip
  `arc-agi` is only the static ARC-1/2 dataset lib). Wrap the agent's `choose_action`.

## Workflow & conventions
- Work directly on `main`. **Always activate the venv** (`venv/Scripts/activate` on Windows) before running.
- Run demos as modules from their experiment folder, e.g.
  `cd experiments/ProgramSynthesis && python -m agent.column.unified_agent`;
  `cd experiments/RecurrentWorldModel && python -m precursor.language_recurrent`. (cp1252 consoles: prefix
  `PYTHONIOENCODING=utf-8`.)
- **Never run heavy training on this machine** — the CPU demos are fine; large training is a GPU job for the user.
- One file = one concept. **No hand-coded rules / domain priors** (the bitter lesson). **Never reimplement core
  machinery per experiment** — extract one canonical component and USE it (e.g. `tbt/recurrence.py`).
- The folder names `RecurrentWorldModel`/`ProgramSynthesis` are historical (kept to not break import paths).
- Hardware: RTX 3050 Ti (4 GB VRAM); everything fits in 4 GB. CPU is sufficient for the demos.
