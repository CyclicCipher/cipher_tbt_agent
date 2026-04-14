# CipherNet Refactor Plan

*Generated 2026-04-12*

Closes five architectural gaps between CipherNet and biologically-grounded
Thousand Brains Theory (TBT). Phases are ordered by risk and dependency.

---

## Phase 1 — Contrastive Hebbian Learning at V1

**Replaces teacher forcing**, which was only the clamped phase of CHL.

CHL has two phases per training image:

```
Free phase    :  run normal WTA                → record free_winner
Clamped phase :  force correct minicolumn      → write associations
Contrastive   :  if free_winner ≠ clamped      → unlearn from free_winner
```

The subtraction (unlearn) is what distinguishes CHL from teacher forcing. It
prevents incorrect V1 minicolumns — those that win the WTA for the wrong
class — from accumulating spurious associations that are never cleaned up.

**Biologically:** free phase = feedforward pass with top-down feedback turned
off. Clamped phase = feedforward + top-down label forcing from IT back to V1.
The calcium concentration signal (Shouval et al. 2002) gates LTP vs LTD.
Mathematically equivalent to predictive coding (Whittington & Bogacz 2017).

**Files changed:**
- `column.py` — `MiniColumn.unlearn_one()`, `MacroColumn.commit_chl()`
- `cortex.py` — `Layer.chl: bool`, CHL branch in `learn()` commit
- `configs/guided.yaml` — `chl: true` on V1

**Expected gain:** ~10–15pp. V1 minicolumns learn class-consistent spatial
models; IT receives stable feature codes per class across all training images.

**Status: IMPLEMENTED**

---

## Phase 2 — Distance-Decay Location Similarity

**Replaces** hardcoded ±1 neighbour search in `overlap_score`.

The old `_NEIGHBOURS = ((0,0),(1,0),…)` gave equal weight to any location
within ±1 bin and zero weight beyond. The new `_loc_sim` uses Chebyshev
distance decay `1/(1+dist)`:

```
dist=0 (exact match)       → sim=1.0
dist=1 (±1 bin ≈ ±2 px)    → sim=0.5
dist=2 (±2 bins ≈ ±4 px)   → sim=0.33
dist=3 (±3 bins ≈ ±6 px)   → sim=0.25
```

Storage key remains the 2D grid key (`RESOLUTION=2`). This is essential:
Fourier keys were tried but caused nearby centroids (which vary ±3px across
MNIST instances) to map to distinct keys, bloating `_loc_total` from ~9 to
~300+ entries per minicolumn and making `overlap_score` 60× slower.

The distance-decay approach achieves the same smooth spatial generalisation
without blowing up the key space, and preserves temporal prediction (2D keys
still support displacement arithmetic in `predict()`).

**Files changed:**
- `column.py` — `_loc_sim()` with 2D distance-decay (replaces `_NEIGHBOURS`);
  `len(location) != 2` guard in `predict()` retained for non-2D frame types
- `reference_frames.py` — `encoding` param added (kept for future use);
  Fourier path implemented but not used in `guided.yaml`
- `cortex.py` — pass `encoding` from config to `_make_frame`
- `configs/guided.yaml` — `encoding: grid` (default; Fourier deactivated)

**Expected gain:** ~3–7pp vs old hardcoded ±1 search, without speed regression.

**Status: IMPLEMENTED**

---

## Phase 3 — HDC Feature–Location Binding *(DEFERRED)*

**Current:** `_model[loc][feat]` — nested dict. Features and locations
co-stored; separately retrievable.

**Correct:** `object_vec += hadamard(feat_vec, loc_vec)`. Overlap = dot product.
Nearby-location generalisation is automatic via vector geometry, not a
hardcoded neighbourhood. Object model = superposition of all bindings.

**Why deferred:** Requires replacing the entire MiniColumn API — all Counter
storage, every caller in `cortex.py`, and the `diagnose()` method. High risk.
Gate on empirical results from Phases 1 and 2.

**New file needed when implemented:** `hdc_memory.py`
(`HDCItemMemory`, `HDCLocMemory`, `HDCFeatureEmbed`).

---

## Phase 4 — Temporal Memory Cells *(DEFERRED)*

**Current:** One Counter per location per minicolumn. No temporal context.

**Correct:** N cells per location; active cell selected by prior context
(previous fixation's active cells). Two objects sharing a feature at a location
activate different cells and are disambiguated after 2–3 fixations.

**Why deferred:** Requires renaming `MiniColumn` → `TemporalCell` and wrapping
N of them in a new `MiniColumn`. High risk of observation fragmentation at 5K
training examples (each temporal cell sees only 1/N observations). Gate on
empirical benefit after Phase 1+2 results.

---

## Phase 5 — Lateral Inhibition Between IT Columns

**Independent of all other phases. Low risk.**

IT's 3 columns (grid 3×1) vote independently. A lateral pass after each
fixation gives each column a small evidence bonus for minicolumns that are
currently winning in adjacent columns — L2/3-style consistency pressure.

**Implementation:** Snapshot tentative winners across the grid, then apply
bonus in a single pass (snapshot prevents order-dependency).

**Files changed:**
- `column.py` — `MacroColumn.apply_lateral_input()`
- `cortex.py` — `Layer.lateral_bonus: float`, `_lateral_pass()` static method,
  called after each fixation's observe step in `learn()` and `classify()`
- `configs/guided.yaml` — `lateral_bonus: 0.1` on IT

**Expected gain:** ~1–3pp on confusable digit pairs (3/8, 4/9, 7/1).

**Status: IMPLEMENTED**

---

## Dependency Map

```
Phase 1 (CHL)                ── independent ✓
Phase 2 (Fourier locs)       ── independent ✓
Phase 3 (HDC binding)        ── requires Phase 2 (position_vec()) — DEFERRED
Phase 4 (Temporal cells)     ── benefits from Phase 1+2 stability  — DEFERRED
Phase 5 (Lateral inhibition) ── independent ✓
```

## Summary Table

| Phase | Description          | Status      | Expected gain |
|-------|----------------------|-------------|---------------|
| 1     | CHL at V1            | IMPLEMENTED | +10–15pp      |
| 2     | Fourier location keys| IMPLEMENTED | +5–10pp       |
| 3     | HDC binding          | DEFERRED    | TBD           |
| 4     | Temporal cells       | DEFERRED    | TBD           |
| 5     | Lateral inhibition   | IMPLEMENTED | +1–3pp        |
