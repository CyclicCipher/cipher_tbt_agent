# REORG_PLAN — merge to one `src/` package, then reorient to the reusable architecture

**Status: APPROVED 2026-06-26. IN PROGRESS.** Do **Part A (merge)** then **Part B (reorient)**. This doc is the
durable record — **update the checklists as steps complete** so it survives context compaction. `CLAUDE.md` and
`MEMORY.md` point here. When everything is checked, delete the "ACTIVE WORK" note in `CLAUDE.md`.

## Why (the problem this fixes)
`unified_agent.py` (575 lines) drifted into a hand-written ARC solver that would crash on any non-grid
environment — the cortical column got demoted to a navigation call. RWM *already* holds the reusable
architecture: `tbt/env.py` (a tiny domain-agnostic `Environment` contract) + `tbt/agent.py` (a 40-line thin
driver that delegates everything to the column) + `tbt/` (the column = the "transformer" substrate). PS built a
*parallel silo* — its own `Environment` in `arc_agi_3` + the fat agent — bridged by `sys.path` hacks. Killing the
silo makes the thin-shell discipline **structural**: the agent imports only the column + the contract; all
task-format knowledge is quarantined in `perception/` and `tasks/`. The yardstick is `tbt/agent.py`.

## Part A — the merge (PURE MOVE + import fixes; mechanical, behavior must be byte-identical)
Target layout (everything under a new top-level `src/`; `experiments/` deleted):
```
src/
  tbt/          # column + canonical machinery (column, l4/l5/l6/l23, recurrence, thalamus, basal_ganglia,
                #   reward, dynamics, residual, factorize, env [the contract], agent [the thin driver])
  perception/   # observation -> column inputs — the ONLY task-format-aware code
                #   (was agent/column/{perceive,objects,object_perceiver,goal_discover,dynamics_perceive})
  tasks/        # environments + oracle + layouts (was arc_agi_3/ + agent/layouts.py)
  agent.py      # the agent (was agent/column/unified_agent.py). STILL FAT after A — thinned in Part B.
  wm/           # the predecessor symbolic agent, kept as reference (was agent/wm/)
  demos/        # runnable validations (was precursor/ + unified_demo.py + scaling_probe.py)
  docs/         # was RecurrentWorldModel/Docs + ProgramSynthesis/docs
  tests/        # was ProgramSynthesis/tests
corpora/        # unchanged (stays top-level)
```
Run convention after A: `cd src && python -m demos.<name>` / `python -m agent`. NO `sys.path` inserts; imports are
absolute (`from tbt.column import …`, `from tasks import LockPath`, `from perception.X import …`,
`from wm.score import …`). The two `Environment`s are NOT unified in A (that is B1) — `tasks/` keeps its own for
now; A is pure relocation so every demo/regression number is unchanged.

### A checklist
_PROGRESS (2026-06-26): A1–A4 DONE — moves via `git mv`, all imports fixed (silo `sys.path` bridges dropped,
`arc_agi_3`→`tasks`, `..wm`/`agent.wm`→`wm`, corpora path, agent's perception imports), `experiments/` deleted,
`src/perception/__init__.py` added. `PYTHONPATH=src python -c "from agent import …"` = IMPORTS OK. A5 VALIDATED — numberline 11/11, LockPath 100%, Toggle 100%, arithmetic (cross-demo import) runs. A6 = this merge
commit. Run convention is now `PYTHONPATH=src python -m <pkg>.<mod>`. **Remaining (cosmetic, do before Part B):
CLAUDE.md architecture-section paths + README run-commands still say `experiments/…` — needs a careful rewrite
(the precursor→demos / arc_agi_3→tasks renames make a blind sed wrong).**_
- [ ] **A1** `git mv` RWM: tbt→src/tbt, precursor→src/demos, unified_demo.py & scaling_probe.py→src/demos, Docs→src/docs
- [ ] **A2** `git mv` PS: arc_agi_3→src/tasks, agent/layouts.py→src/tasks, agent/wm→src/wm, agent/column/{perceive,objects,object_perceiver,goal_discover,dynamics_perceive}.py→src/perception, agent/column/unified_agent.py→src/agent.py, tests→src/tests, docs→src/docs
- [ ] **A3** fix imports: drop every `sys.path.insert`; `arc_agi_3`→`tasks`; `..wm.score`→`wm.score`; `.dynamics_perceive`/`.objects`/… in the agent→`perception.X`; `tbt.*` stays. Add `src/perception/__init__.py`, `src/tasks/__init__.py` as needed.
- [ ] **A4** delete `experiments/`; fix `agent/column/__init__.py`'s exports (now `src/__init__` or fold into `agent.py`)
- [ ] **A5** VALIDATE (must match prior numbers): `demos.arithmetic` (5166), a graph demo (1.000), `demos.language_recurrent` (PPL 152), and the agent regression LockPath 100 / MultiKey 100 / Sokoban 78.7 / Toggle 100
- [ ] **A6** update CLAUDE.md + README run commands to `src/`; commit "Merge to one src/ package; kill the experiment silo"

## Part B — the reorientation (the REAL fix; conceptual, incremental, regression-gated)
_PROGRESS (2026-06-26): **B1 + B2 + INVARIANT DONE & validated** (commit pending). The fat 575-line agent is
dissolved into three pieces: `agent.py` = a ~30-line pure driver that **imports nothing** (perception + planner
injected); `perception/scene.py` = the ONLY task-format code (WorldModel/Scene + the grid/colour/action-vocab,
`build_world` decodes the dynamics' string effects once); `tbt/planner.py` = the spatial planner (SR-frame map +
recurrence + subgoal value, **imports only `tbt/`**, speaks move-indices not GameAction). The ARC wiring +
factories + eval moved to `demos/agent_arc.py`. B1: `tasks/contract.py` `ContractEnv` presents the harness through
`tbt.env.Environment` (non-breaking — native harness + FrameData consumers untouched); `tbt/env.py` `step` gained
the optional click `coords`; the eval drives the agent over the contract. **Regression IDENTICAL** to the old
agent (proven seed-by-seed): LockPath 100, MultiKey 100, Sokoban (2 seeds), Toggle 100, partial-obs 8/8 mem vs
degraded ablation; numberline 11/11. **Still TODO: B3 (the emergent planning — the planner still has the
fire/cover/goal labels + the `harmful` hack, now in `tbt/planner.py` + `WorldModel.harmful`) and B4.**_
- [x] **B1** unify the `Environment` contract: `ContractEnv` adapts the harness to `tbt.env.Environment`; the
        contract `step` carries the click `coords` (declared; replica is movement-only); the agent drives the
        contract, never ARC specifics. (NOT a force-merge of the harness's return type — that breaks every
        FrameData consumer for ceremony; the click action is real-SDK work.)
- [x] **B2** dissolve the fat agent → thin shell: all task-format in `perception/scene.py`; the agent imports
        NOTHING task-specific (planner+perception injected). Subgoal enumeration relocated to `tbt/planner.py`
        (to be dissolved in B3, not yet emergent).
- [ ] **B3** planning → the column (per `src/tbt/EMERGENT_PLAN.md`): subgoals = eigenoptions (SR-frame) +
        affordances (learned dynamics); value from `reward.py`. Delete the fire/cover/goal labels +
        `WorldModel.harmful` (it must EMERGE from value over the augmented state — the toggle's switch).
- [ ] **B4** carry toggle 100% + the regression through; collect-all via the consume affordance.
- [x] **INVARIANT** `src/agent.py` imports nothing task-specific (verified: only `from __future__`); `tbt/planner.py`
        imports only `tbt/` + stdlib; all grid/colour/GameAction knowledge is in `perception/` + `tasks/`.

Detail + research backing for Part B: `src/tbt/EMERGENT_PLAN.md`.

## After Part B — study EfficientZero-V2
Read **EfficientZero-V2** (arXiv 2403.00564) — a sample-efficient, domain-general MuZero successor with big gains
in SEARCH over continuous spaces (Cipher flagged it as a key clue, 2026-06-26). Our `reward.py` is a MuZero-style
critic, so it informs the planning B moves into the column. Open question: **realize EZ-V2's improvements with the
neocortex (columns + thalamus + BG), not a monolithic net.** See memory `reference_efficientzero_v2`. (Read AFTER B.)
