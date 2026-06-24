# ProgramSynthesis — side-project

A side-project to `experiments/RecurrentWorldModel/`. It began as a controlled test of François
Chollet's **deep-learning-guided program-synthesis / binding** thesis, pivoted to a from-scratch
**symbolic world-model agent** that solves a full ARC-AGI-3 replica by discovery, and is now entering
a redesign around **volume concepts**.

## Phases

**Phase 1 — built & (partly) proven.** Everything coded to date. Two threads:

- **(a) Binding experiment** — a fixed transformer trunk trained by behaviour-cloning from a BFS
  oracle on the `LockPath` replica, with the **state encoder as the experimental knob** (token /
  2D-PoPE / 2D+1-PoPE / slots / field), to test the thesis that *a relationship is cheap iff it is a
  coordinate operation, expensive iff it must be discovered as a correlation among tokens*. Status:
  built end-to-end (29 tests), runnable on GPU; directional. Docs: `docs/phase1/` —
  `chollet_connection.md`, `LEARNING_AGENT.md`, `EXPERIMENT_GOALS.md`, `RESULTS.md`. Code: `agent/`
  (`encoders.py`,
  `trunk.py`, `dataset.py`, `train_bc.py`).
- **(b) Symbolic world-model agent** — the pivot to symbolic AI (4 GB VRAM rules out meaningful NN
  training). A perceive→induce→infer-goal→plan agent that discovers a game's rules from **frame +
  score alone** and **wins all four levels of the replica on 12/12 seeds** (RHAE-proxy 37.9% vs the
  optimal oracle), with **no catastrophic forgetting** and graceful per-level distribution shift.
  **This is the main result.** Docs: `docs/phase1/` — `AGENT_DESIGN.md`, `FINDINGS.md`,
  `LIMITATIONS.md`. Code: `agent/wm/` (`perceptor.py`, `world_model.py`, `planner.py`, `agent.py`,
  `score.py`).

**Phase 2 — the redesign (active).** Documentation lives in `docs/phase2/` and **starts with
[VOLUME_CONCEPTS.md](docs/phase2/VOLUME_CONCEPTS.md)**: store knowledge as *volumes / regions* in a
**graph of local spaces**, with dimensions a discovered, MDL-promoted, grounded **tower**, in order to
(i) get context-dependent meaning *and an algebra* by geometry, (ii) drive the prior floor down to
**{sensory interface, metric, compression}**, and (iii) be interpretable by construction. The plan
modifies the Phase-1(b) symbolic agent first; NN transfer only if it earns its keep. Fuses with the
discovery mechanism in [DISCOVERY_PROGRAM.md](docs/phase2/DISCOVERY_PROGRAM.md) (MDL /
representation-regime change). Prototype code: `volume/`.

## Documentation map

| phase | thread | docs |
|---|---|---|
| 1 | binding experiment | [chollet_connection.md](docs/phase1/chollet_connection.md), [LEARNING_AGENT.md](docs/phase1/LEARNING_AGENT.md), [EXPERIMENT_GOALS.md](docs/phase1/EXPERIMENT_GOALS.md), [RESULTS.md](docs/phase1/RESULTS.md) |
| 1 | symbolic agent | [AGENT_DESIGN.md](docs/phase1/AGENT_DESIGN.md) (loop + iteration log 1–5), [FINDINGS.md](docs/phase1/FINDINGS.md) (self-contained writeup; §5.1 = no-forgetting / distribution-shift evidence), [LIMITATIONS.md](docs/phase1/LIMITATIONS.md) (honest domain-specificity) |
| 2 | volume concepts | **[VOLUME_CONCEPTS.md](docs/phase2/VOLUME_CONCEPTS.md)** (charter: §0 resolved representation decisions, papers, plan), [REPLICA_TEST.md](docs/phase2/REPLICA_TEST.md) (the prior-minimal agent on the replica + prior accounting), [DISCOVERY_PROGRAM.md](docs/phase2/DISCOVERY_PROGRAM.md) (the discovery mechanism it fuses with) |

## Code

- **[arc_agi_3/](arc_agi_3/)** — an offline replica of the **ARC-AGI-3 interactive reasoning
  benchmark**: faithful harness/object-model (`FrameData`, `GameAction`, `GameState`, `Scorecard`,
  64×64 grids, action gating, the WIN/GAME_OVER lifecycle) plus an original game, `LockPath`, whose
  levels introduce mechanics progressively and whose final level *composes* two separately-taught
  mechanics — an in-environment instance of RWM's A∘B test. See [arc_agi_3/README.md](arc_agi_3/README.md).
  The BFS oracle is in `arc_agi_3/oracle.py` (a teacher/validator, never part of an agent).
- **[agent/](agent/)** — Phase-1(a) binding-experiment learning agent.
  - `encoders.py` — the binding-channel knob: `none` / `content` / `pope2d` / `pope2d1` schemes
    (multi-axis rotary + QK-norm), plus patchify and `(px,py,t)` coordinates.
  - `trunk.py` — the fixed transformer trunk + policy/value heads (`build_model(binding)`); by
    construction `none`/`pope2d`/`pope2d1` have **identical parameter counts** — the binding is the
    only variable. ~350K params, 129 tokens, `scaled_dot_product_attention`, 16×16 canvas at patch=2.
  - `dataset.py` — oracle rollouts → (frame-window, inter-frame actions) → action pairs.
  - `metrics.py` — action-match accuracy (raw + masked), the train-vs-held-out gap, time-to-threshold.
  - `train_bc.py` — the behaviour-cloning sweep (a GPU job; see below).
  - `layouts.py` — procedural per-mechanic LockPath distributions (nav / key_door / block_pad /
    compose), BFS-validated, with a train/held-out split for the shift-invisibility test.
- **[agent/wm/](agent/wm/)** — Phase-1(b) symbolic world-model agent (`perceptor`, `world_model`,
  `planner`, `agent`, `score`). The winning seed-6 trace is `agent/wm/seed6_trace.json`.
- **[volume/](volume/)** — Phase-2 concept representation: `box` (subspace), `halfspace` (shape),
  `union` (modes), `relation` (edges/rules), `algebra` (meet/entails), `benchmark` (shape bake-off).
- **[agent/wm2/](agent/wm2/)** — Phase-2 **prior-minimal** agent (the replica test; see
  `docs/phase2/REPLICA_TEST.md`). `perceive.py` discovers agency + the move model from scratch.

## Running

**Symbolic agent (CPU — fast).** Activate the venv first.
```powershell
$env:PYTHONPATH = "experiments\ProgramSynthesis"
python -m pytest experiments/ProgramSynthesis/tests/test_wm_agent.py -q   # 16 tests
python -m agent.wm.score                                                   # RHAE-proxy across 12 seeds
```

**Binding-experiment BC sweep (a GPU job; Mistake #36 — train on the user's GPU, not here).**
```powershell
$env:PYTHONPATH = "experiments\ProgramSynthesis"

# channel dominance + shift invisibility on held-out layouts of one mechanic:
python -m agent.train_bc --train-mechanics key_door --steps 4000          # (repeat for nav / block_pad / compose)

# composition transfer: train on the constituents, test zero-shot on compose:
python -m agent.train_bc --train-mechanics nav,key_door,block_pad --test-mechanic compose --steps 6000
```
Output JSON lands in `agent/runs/`; `python -m agent.collect_runs` flattens them for the comparison
plot. Each run sweeps all four bindings and prints held-out masked accuracy, the train-vs-held-out
gap, and time-to-90%. Predicted ordering (`docs/phase1/LEARNING_AGENT.md §4`): `pope2d1 ≫ pope2d > content > none`
on the change/causality mechanics, gap ~0 for the pope arms and opening for `content`/`none`.
Performance note: tiny model (~350K params) ⇒ **low GPU-utilization% but fast wall-clock** — judge by
wall-clock, not the utilization graph; levers are `--batch 1024` or `torch.compile`.

## Status & next

- **Phase 1(b) symbolic agent: complete and the headline result** — 12/12 seeds win the full replica,
  no catastrophic forgetting (`docs/phase1/FINDINGS.md §5.1`). 16 tests pass.
- **Phase 1(a) binding sweep: built, awaiting a GPU run** (then the DAgger relabeller and the slot/
  field arms).
- **Phase 2 (active): design resolved** (`docs/phase2/VOLUME_CONCEPTS.md §0`) — graph of local spaces +
  discovered dimension tower. First prototype (`volume/`, in progress): let MDL *discover* a concept's
  dimensions and a graded region from data instead of being handed them — validating the structure in
  miniature before integrating with the agent.
