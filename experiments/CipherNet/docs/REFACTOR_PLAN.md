# CipherNet Refactor Plan

*Written 2026-04-14. Based on architectural review against TBT biology and analysis
of three structural gaps: missing feedback, too-local lateral voting, and
supervised IT imposing one-column-per-class.*

---

## What to Keep

The v9 symbolic column design is the right foundation:
- One-shot Hebbian learning (write at commit, never retrain)
- Evidence accumulation across fixations (not single-shot)
- Soft overlap scoring with Chebyshev-distance similarity
- CHL unlearn pass (free winner ≠ clamped winner → subtract)
- DoG → HOG/Gabor encoding pipeline
- Receptive field hierarchy auto-computed from grid sizes

None of these are being replaced.

---

## Three Structural Gaps

### Gap 1 — Supervised IT (one column per class)

`commit_supervised(label)` hardwires `minicolumn_index == digit_class`. There are
exactly 10 minicolumns with pre-named slots. This is not TBT — it is 10 named
buckets. A real TBT column is unsupervised; the label association lives outside the
column in a dedicated readout layer.

**Fix (Phase 3):** IT uses `commit()` (unsupervised WTA, `n_mini=20`). `OutputCortex`
handles the Hebbian label mapping. `OutputCortex` was already fully implemented in
`output_cortex.py`; it just was not wired into the training/inference loop.

### Gap 2 — No feedback (missing white matter)

`Cortex.learn()` and `Cortex.classify()` are pure forward passes: V1 → IT, stop.
IT's tentative winner never reaches V1. Without feedback, V1 must resolve all
ambiguity from local evidence alone; IT's hypothesis provides no guidance.

Also: horizontal white matter connects columns *within* a layer across long distances,
not just ±1 neighbours. The existing `_lateral_pass()` only propagates to adjacent
columns.

**Fix (Phase 1 + 2):**
- Phase 1: `_global_lateral_pass()` — full-layer plurality vote, every column gets a
  small bonus for the most common tentative winner in the whole layer.
- Phase 2: `_feedback_pass()` — after each fixation's forward sweep, IT's winner's
  model predicts which V1 minicolumn should win at each RF slot; that minicolumn
  gets `receive_feedback_by_index(bonus)` added to its evidence before the next
  fixation's observations arrive.

### Gap 3 — Efference copy stopped at V1

The saccade displacement is used in V1 via `MiniColumn.predict(loc, displacement)`,
but V2/IT have no awareness of the motor command. With unsupervised IT (Phase 3), IT
accumulates a real model of V1 winner codes. The feedback pass (Phase 2) then *is* the
efference copy for IT: IT's prediction of what V1 will show IS IT's best guess given
the current motor state.

---

## Phase Plan

### Phase 1 — Full-layer lateral voting (DONE)

**Files:** `cortex.py`

Add `_global_lateral_pass(layer, bonus=0.05)` called after `_lateral_pass()` at
each fixation in both `learn()` and `classify()`. Counts votes across all columns
in a layer; every column gets a small evidence bonus for the plurality winner.

Implements the long-range horizontal white matter connections that `_lateral_pass()`'s
±1 neighbour radius could not reach.

### Phase 2 — Feedback predictions: IT → V1 (DONE)

**Files:** `column.py`, `cortex.py`

`MacroColumn.receive_feedback_by_index(mini_idx, bonus)` directly increments
`self._evidence[mini_idx]`. No new persistent state needed — evidence is already
a mutable running accumulator.

`Cortex._feedback_pass(layer_tentatives, feedback_bonus=0.3)` iterates upper→lower
layer pairs, reads each upper column's winner minicolumn's `_best[loc]` predictions
(stored as str(v1_winner_idx)), and calls `receive_feedback_by_index` on the
corresponding lower column.

Called at the end of each fixation's loop body in both `learn()` and `classify()`.

### Phase 3 — Unsupervised IT + OutputCortex (DONE)

**Files:** `cortex.py`, `guided.yaml`

IT layer becomes unsupervised (`supervised: false`, `n_mini: 20`). `Cortex` holds
`self._output_cortex: OutputCortex`. In `learn()`, after committing all layers, the
IT winners are passed to `output_cortex.learn(col_idx, sdr, label)`. In `classify()`,
`output_cortex.classify(active)` produces the final label prediction; `tentative_winner()`
SDRs feed the early-stopping path.

### Phase 4 — Efference copy through hierarchy

**Files:** `column.py`

When Phase 2 feedback is running and IT is unsupervised (Phase 3), the temporal bonus
mechanism in `observe_multi()` already checks whether the previous leader's predictions
matched the current observations. This becomes a genuine cross-layer credit signal once
IT has a real model.

Optional enhancement: merge the top-down prior signal into the temporal bonus
calculation so that a column whose IT-predicted mini matches its arriving observation
gets `TEMPORAL_BONUS` rather than just `CONTINUITY_BONUS`.

### Phase 5 — Temporal context cells

**Files:** `column.py`, `cortex.py`

`MiniColumn._model` currently maps `loc → Counter[feat]`. Change to
`(loc, context_cell) → Counter[feat]` where `context_cell` is the previous
fixation's tentative winner index modulo `n_context` (default 4).

Fixes: two objects sharing feature F at location L are currently indistinguishable
after 1 fixation. With context cells they activate different cells in different
sequence contexts.

Config addition: `n_context: 4` per layer (default 1 = current behaviour).

### Phase 6 — Fourier location encoding

**Files:** `reference_frames.py`, `column.py`

`RetinotopicFrame.position_key()` currently returns `(gy, gx)` integers.
Change to return a tuple of floats:

```
loc_vec(x, y) = [cos(2π f x/S), sin(2π f x/S), cos(2π f y/S), sin(2π f y/S)]
                for f in [1.0, 2.0, 4.0]   →  12-float tuple
```

`MiniColumn._loc_sim()` for 12-float tuples: cosine similarity (dot product,
since vectors are unit-normalised). Removes the need for the ±1 neighbour search
in `predict()`.

Config: `encoding: 'fourier'` vs `encoding: 'grid'` (backward-compatible).

---

## What NOT to Do

- **Do not rebuild the ARCHITECTURE.md physics engine** (oscillatory timing,
  L4/L5/L6/L23 layers, dendritic segments, graph.step()). That was v8 — it could
  not learn single-digit succession after 100 epochs. v9 symbolic columns are the
  foundation.
- **Do not add PFC, BG, or thalamus** until MNIST exceeds 85%. Those are for
  sequencing and action selection, not object recognition.
- **Do not add V2** as a separate layer yet — the V1→IT architecture is simpler to
  debug. Add V2 after Phase 3 produces a clean accuracy baseline.
- **Do not implement full HDC binding** until Phases 5 and 6 are complete and
  the location encoding is already vectorised.

---

## Implementation Order Rationale

| Phase | Prerequisite | Why ordered here |
|-------|--------------|-----------------|
| 1 (global lateral) | None | Simple, independent, measurable |
| 2 (feedback) | None | Independent of Phase 3; improves V1 consistency |
| 3 (unsupervised IT) | Phase 2 recommended first | Feedback assumes IT has a real model (not named slots) |
| 4 (efference copy) | Phases 2+3 | Feedback IS the efference copy once IT is unsupervised |
| 5 (temporal context) | Phase 3 | Context signal is the previous winner, needs stable unsupervised winners |
| 6 (Fourier locs) | None | Independent; do last to avoid noise during earlier debugging |
