# CLAUDE.md — Cipher's TBT Agent

## ⚠ ACTIVE WORK (read first) — Phase 2 Step C: objects as column-models, recognized by voting
**START HERE:** the memory **`reference_tbt_pose_invariant_recognition.md`** (the mechanism + sources) and
**`src/tbt/RESEARCH.md` R11** (the two "BUILT 2026-06-28" notes). Run `PYTHONPATH=src python -m pytest src/tests` (73 green);
the agent is one thin shell (`tbt/agent.py`) over a planner; the merge to `src/` is long done.

Phase 2 dissolves the symbolic scaffolding into a learned model + SIGNED value. **Step A** (the column IS the forward
model, `82b9bcd`) and **Step B** (the value-driven ROLLOUT planner over the multi-column spine — thalamus/CMP + basal
ganglia + per-object columns STAY, never stripped) are DONE. **Step C** (objects as column-models, recognized by
voting) — the OBJECT-RECOGNITION pivot is DONE (2026-06-28): a memorized rotation table was both buggy AND non-TBT
(the 2019 grid-cell model cannot recognize a rotated pose — its stated limitation), replaced by **`tbt/recognize.py`**,
a Monty-style evidence-based recognizer — CONTINUOUS pose SOLVED not recalled, online label-free object learning, the
rotation OPERATOR (`cells_at`, correct by construction), and multi-column pose-aware VOTING (`1b871cb`, `fab849a`); the
agent plays Tetris via recognition (the table is deleted). **Real-ARC STARTED** (`8772b36`): `perception/perceive.py`
`ObjectRecognizer` bridges `segment()` → recognition for object permanence under rotation in 64×64 frames.
**Real-ARC step 1 DONE (2026-06-28):** the planner now CONSUMES recognized objects over MULTI-OBJECT scenes —
`scene.py` delivers `Scene.movers=[(object_id, cells)]` (segment→recognize per pushable colour), and
`NeocortexPlanner` was generalized from single-cell movers to RIGID multi-cell bodies (rollout state `(agent,
focus-anchor, removed)`, push the whole footprint; single-cell = degenerate, all 5 replicas stay green); the BG focus
gate keys on object IDENTITY (so the id is load-bearing + same-colour objects disambiguate by shape). Bench:
`Sokoban.MULTICELL_LEVELS` (domino + L), the SAME agent solves M0/M1/full 2/2; suite 73/73; no new planner/game/
role-branch. See `RESEARCH.md` R11's 2nd "BUILT 2026-06-28" note.
**SDK adapter DONE (2026-06-28):** `src/arc_sdk.py` bridges THE agent to the real ARC-AGI-3-Agents SDK —
`TbtPolicy` (SDK-free, duck-typed on `FrameData`: `state`/`frame`/`levels_completed`) wraps `tbt.agent.Agent` into
`choose_action`/`is_done` returning `(action_name, coords)`; `make_arc_agent(factory)` lazily subclasses the SDK
`Agent` and maps name→`arcengine.GameAction` (+`set_data({"x","y"})` for ACTION6). Verified against a clone of
arcprize/ARC-AGI-3-Agents (real `agent.py`/`random_agent.py`); offline-test-gated by driving multi-cell Sokoban to
WIN through the SDK contract (suite 78/78, no API key). **Real SDK needs Python ≥3.12 + `pip install arc-agi>=0.9.1`
(provides `arcengine`+`arc_agi`); our venv is 3.11.9 → a separate 3.12 env is needed to run LIVE.**
**LIVE CONNECTIVITY + LEAN RUNNER DONE (2026-06-28):** registered API key reaches the hosted API (25 public games)
from a separate **3.12 `venv312/`** (Norton MITMs TLS → `pip install pip-system-certs`; see memory
`project_arc_agi3_live_env`). `src/arc_run.py` = the LEAN runner (`arc_agi.Arcade` ONLINE + a TbtPolicy-contract
policy), validated on live `ls20` (real frames = 64×64 single grid; `ls20` actions = ACTION1-4 movement). Lean chosen
over the full framework (which drags the langchain/LLM stack); lean IS the Kaggle-notebook submission shape.
**RE-ORIENTATION (THE ACTIVE PLAN) — memory `project_continuous_online_loop`:** the private test scores by RHAE
`(human/agent_actions)²`, terminating at 5× the human median actions/level, with NO free practice on the held-out
private games. So `explore_and_learn` (hundreds of random episodes) scores ~0 — WRONG loop. NEXT SESSION: build a
sample-efficient **CONTINUOUS online learn-and-plan loop** (fast body-id via efference copy; prediction-error-directed
exploration not random ε; goal-from-first-score; plan-on-model immediately; cross-level transfer), reusing
perception + `WorldLearner`/`GoalModel` + `NeocortexPlanner` + `reward.py` novelty; validate + measure
action-efficiency on public `ls20`. Then: the click action (`ACTION6`, learned from a click game), colour-as-feature
+ occlusion + rotation-permanence (where `vote`/cross-frame id-tracking earn their keep).
Also open: the value/L2 multi-piece-clear now runs on a FAITHFUL model (EZ-V2 robustness reserved for real-ARC's
imperfect ONLINE models). The replica role-schema strand (CollectAll/Toggle) — see `project_reorient_and_reconnect.md`
+ R11. NB: the architecture/run-command sections BELOW may cite stale `experiments/…` paths.

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
- **Real ARC-AGI-3**: 64×64×16-colour frames, multi-cell objects, a click action, level-completion-only signal.
  The interactive SDK is the **ARC-AGI-3-Agents** repo (`agents.agent.Agent` ABC: `choose_action`/`is_done`) + an
  API key from three.arcprize.org; types come from **`pip install "arc-agi>=0.9.1"`** (the ARC-AGI-3 *toolkit*,
  providing `arcengine` FrameData/GameAction/GameState + `arc_agi.EnvironmentWrapper` — NOT the old static ARC-1/2
  lib; that earlier note was wrong). Needs Python ≥3.12. Our agent is wired in via `src/arc_sdk.py` (`make_arc_agent`).
  Competition: Kaggle "ARC Prize 2026 – ARC-AGI-3", 3 public + 3 private games, sandboxed (NO internet, ~RTX 5090/8h)
  → self-contained agents only (hosted-LLM agents disqualified); RHAE scoring; deadline 2026-11-02.

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
