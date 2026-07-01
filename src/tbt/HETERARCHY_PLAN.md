# HETERARCHY_PLAN — from ONE correct column to many, communicating

*2026-06-30. Precondition MET: the single column is anatomically correct and its CMP message is well-formed
(`COLUMN_AUDIT.md` → Verification). This is the plan for step 2 (communication) and step 3 (the heterarchy). Like the
audit, the benchmark is **mechanism correctness**, not a game score — but the pay-off of a correct heterarchy is that
the C4-coupled games (MultiKey/LockPath) return CORRECTLY FACTORED, and cross-GAME transfer (the human-baseline lever)
becomes expressible. Spec sources: [[reference_hierarchy_substrate]], [[reference_tbt_layers_4_23]] (CMP voting),
[[reference_tbt_frames_and_hippocampus]] (the B/body frame), `TARGET_ARCHITECTURE.md` (factor-first), Mountcastle
(one algorithm, many inputs).*

## What the heterarchy IS — and the ONE non-negotiable principle
TBT is **many copies of the ONE column algorithm + communication.** Two consequences fix the whole build:
1. **No new mechanism.** A "PFC/task column" is NOT a different kind of unit — it is the SAME `CorticalColumn` fed a
   different INPUT (task-state instead of sensor-patch); specialisation EMERGES from inputs (Mountcastle), it is not
   designed in. So we NEVER write a bespoke task/relational module — we instantiate the column we already have.
2. **Heterarchy, not hierarchy.** Columns are peers that VOTE (lateral CMP), not a fixed top-down stack. "Higher" =
   longer-range / more-abstract input, but the message and the voting are identical at every level.

## FACTOR-FIRST — do we even need a 2nd column? (the gate before any of this)
`TARGET_ARCHITECTURE` is explicit: **try to FACTOR within one column first (L6 eigen-subspaces); allocate a 2nd column
(via the basal ganglia) only when the factors do not separate.** The single column already carries both signals we need
— `locate` (position, the SR-eigenframe) and `object_state` (the dynamic scene, L2/3). The open question the C2↔C4
coupling posed is whether ONE column's L6 can hold `(position, object_state)` as SEPARABLE eigen-subspaces (spatial +
task on one frame) or whether they interfere (the diffuse/joint-state problem that C2a's DG fix and the position-only
regression exposed).

**⇒ H0 (the decision experiment, do this FIRST):** feed one column the JOINT `(position, object_state)` transitions and
test whether its SR eigenframe factorises — i.e. some eigenvectors track position (invariant to object_state) and
others track object_state (invariant to position). *Mechanism test:* on MultiKey, `col.locate((pos, keys))` place codes
cluster by `pos` along one subspace and by `keys` along another (a subspace-projection test), and `col.value` over the
joint state solves the level. **If it factorises → NO 2nd column is needed** (the heterarchy collapses to one column
with a structured frame; document it, wire the planner over the joint state, done). **If it does NOT factorise → H1+**
(two columns). Either outcome is a real finding, not a score. This gate is why we do not presuppose multiple columns.

## The build order (each step gated by a MECHANISM test)

### H1 — communication between two CORRECT columns (only if H0 says we need it)
Before any task column, prove inter-column messaging works now that the message is well-formed. Two identical columns
sensing the SAME object from different vantage points must REACH CONSENSUS on its identity/pose.
- **Substrate (have):** `thalamus.bind`/`read`/`read_location` (content ⊗ location routing) and `L23.vote` (lateral CMP
  pooling of neighbours' hypotheses). The B/body-centric frame ([[reference_tbt_frames_and_hippocampus]]) is the shared
  coordinate the votes are cast in — add it as the frame `vote` pools over across columns.
- **Do:** route column A's and column B's `(object, pose)` hypotheses through the thalamus; each runs `L23.vote` over
  the pooled set; consensus settles faster + more accurately than either alone.
- **Test:** two columns, one ambiguous glance each, disagree individually; after one `vote` round through the thalamus
  they agree on the true object — and CONVERGE in fewer samples than a single column (the CMP speed-up, [[reference_tbt_pose_invariant_recognition]]).

### H2 — the TASK column (a 2nd instance of the SAME column, task-state input)
Instantiate a second `CorticalColumn` whose INPUT is not the sensor patch but the SPATIAL column's output message —
specifically `(recognised-object, object_state)`. Its L6 learns a cognitive map over TASK states (keys-collected,
doors-open — the reward-free relational structure), exactly [[reference_hierarchy_substrate]]'s "separate task-space map,
grid reused." No reward is fed to its map (subgoals are reward-free); value rides on top.
- **Test:** the task column's `locate(object_state)` place codes encode the TASK topology (adjacent board-states —
  differing by one key — are near; a board-state two keys apart is far), learned online, with NO spatial coordinates
  leaking in.

### H3 — the SPATIAL+TASK heterarchy wired (where C4 lands, correctly factored)
The two columns, composed through the existing routing:
- **Spatial column** (the current one): SR + L5 over POSITIONS; emits `(position, recognised-object, object_state)`.
- **Task column** (H2): value over `(position ⊗ object_state)`; the planner reads V from HERE.
- **Basal ganglia** allocates which column models which structure (it already does this — `select`/`reinforce`); the
  **thalamus** binds spatial-position ⊗ task-state so the planner sees the joint state WITHOUT either frame going diffuse
  (position stays a clean metric in the spatial column; keys-collected stays clean in the task column).
- **This is the C4 INTEGRATION.** MultiKey/LockPath now plan over `(position, object_state)` — position from the spatial
  map, keys-collected from the task map — so the board-state regression the audit found is fixed BY FACTORISATION, not
  by a joint blob. *Test (mechanism, then the games as a downstream check):* the planner distinguishes two board-states
  at the same position (picks different actions); then MultiKey 2/2 and LockPath return, factored (a check that the
  mechanism works, not the target).

### H4 — abstraction / cross-GAME transfer (the human-baseline lever)
With the task map in place, a THIRD column fed longer-range input (the sequence of object_states = the mechanic
"shape") learns the MECHANIC as an object in its own right — this is what makes [[reference_human_baseline_traces]]'s
cross-GAME transfer (Sokoban's push-lesson → LockPath) EXPRESSIBLE: transfer the recognised MECHANIC-object, not
weights. Deferred beyond the first heterarchy, but the architecture (one more column, same algorithm) is already set.
- **Test:** a mechanic learned in game X is RECOGNISED (L2/3, pose-invariant over its state-graph) in game Y, cutting
  discovery cost on Y's first level.

## Sequencing + guardrails
- **H0 gates everything** — if one column factorises, H1–H4 simplify to a structured single column. Run it first.
- Each column is `CorticalColumn(...)` — the SAME class. If any step tempts a bespoke module, STOP: that is the harness
  trap again ([[feedback_bitter_lesson]], [[feedback_one_model]]). The task column differs only in its INPUT.
- Communication reuses `thalamus.*` + `L23.vote` — do not reimplement routing/voting ([[feedback_reuse_canonical_components]]).
- Mechanism tests only; the games are a downstream CHECK that the mechanism composed, never the objective.
- Order: **H0 (factor decision) → H1 (comms) → H2 (task column) → H3 (heterarchy + C4) → H4 (transfer).**
