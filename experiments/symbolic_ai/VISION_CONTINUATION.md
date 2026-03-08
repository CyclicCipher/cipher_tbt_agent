# Vision Pipeline Continuation (next session)

**Date written:** 2026-03-08
**Priority:** Optimization and parallelization of `sequence_pipeline.py` and `vision_pipeline.py`.
**Motivation:** n_images=100 demo timed out (>1 hour). n_images=30 takes ~6 minutes.
Root cause: 1.07 billion pure-Python iterations in `_precompute_all_soft`.

---

## Completed this session

- **V5** — object-relative 'D{dr},{dc}' position tokens (commit `eefb9a8`)
- **V6** — classification chain with Laplace-smoothed Naive Bayes (commit `6f9313e`)
  - `teach_class()`, `classify()`, `evaluate_classification()`, `top_features()`
  - `_feat_counts` dict (raw counts, separate from ai — needed for Laplace smoothing)
- **Content hash bug fixed** — removed inner `_to_gray_f32(patch)` call that collapsed
  all 1×1 foveal pixel hashes to '00'. Patch is already globally-normalised float32.
- Merged `claude/naughty-williamson` → `main` (fast-forward). Pushed.

---

## Next session tasks

### Task 1 (DONE 2026-03-08) — Vectorize `_precompute_all_soft` in `sequence_pipeline.py`

**File:** `experiments/symbolic_ai/sequence_pipeline.py`
**Function:** `_precompute_all_soft` (line ~131) and `_ask_soft` (line ~102)

**Problem:** For K=64, arity=3 (wgc_soft):
```
262,144 calls to _ask_soft × 4096 dict entries each = 1.07 BILLION Python iterations
```

**Fix:** Replace with numpy tensor contractions. The soft retrieval formula:
```
result[q1,q2,q3,v] = Σ_{k1,k2,k3} S[q1,k1]*S[q2,k2]*S[q3,k3]*D[k1,k2,k3,v]
```
decomposes into three sequential `np.tensordot` calls (one per dimension).

**New function** (add after `_precompute_all_soft`, call it instead in `fit()`):
```python
def _precompute_all_soft_numpy(dist_cache, sim_matrix, K, arity):
    """Vectorized replacement: numpy tensor contractions instead of Python loops.
    Speedup ~2000× for arity=3, K=64. Falls back to _precompute_all_soft if numpy missing."""
    try:
        import numpy as np
    except ImportError:
        return _precompute_all_soft(dist_cache, sim_matrix, K, arity)

    S = np.array(sim_matrix, dtype=np.float32)  # (K, K)

    all_outputs = sorted({o for d in dist_cache.values() for o in d})
    out_idx = {o: i for i, o in enumerate(all_outputs)}
    V = len(all_outputs)
    if V == 0:
        return {tuple(str(i) for i in ids): None
                for ids in itertools.product(range(K), repeat=arity)}

    # Build dense tensors
    D = np.zeros((K,) * arity + (V,), dtype=np.float32)
    mask = np.zeros((K,) * arity, dtype=np.float32)
    for stored_key, out_dist in dist_cache.items():
        try:
            idx = tuple(int(s) for s in stored_key)
        except (ValueError, TypeError):
            continue
        if any(i < 0 or i >= K for i in idx):
            continue
        total = sum(out_dist.values())
        if total < 1e-12:
            continue
        mask[idx] = 1.0
        for out_token, prob in out_dist.items():
            if out_token in out_idx:
                D[idx + (out_idx[out_token],)] = prob

    # Sequential tensordot: contract each key dimension with S → query dimension
    result = D
    weight = mask
    for dim in range(arity):
        result = np.tensordot(S, result, axes=([1], [dim]))  # new query dim at front
        weight = np.tensordot(S, weight, axes=([1], [dim]))
    # Dims are reversed after arity contractions — transpose back
    result = np.transpose(result, list(range(arity - 1, -1, -1)) + [arity])
    weight = np.transpose(weight, list(range(arity - 1, -1, -1)))

    # Normalize
    normed = result / np.where(weight > 1e-12, weight, 1.0)[..., np.newaxis]

    # Convert back to dict format expected by predict_e3 / logprob
    soft = {}
    for ids in itertools.product(range(K), repeat=arity):
        qk = tuple(str(i) for i in ids)
        if float(weight[ids]) < 1e-12:
            soft[qk] = None
        else:
            d = {all_outputs[vi]: float(p) for vi, p in enumerate(normed[ids]) if p > 1e-12}
            soft[qk] = d if d else None
    return soft
```

In `fit()`, replace:
```python
self._nc_soft  = _precompute_all_soft(nc_cache,  sim_matrix, self._K, 2)
self._wgc_soft = _precompute_all_soft(wgc_cache, sim_matrix, self._K, 3)
```
With:
```python
self._nc_soft  = _precompute_all_soft_numpy(nc_cache,  sim_matrix, self._K, 2)
self._wgc_soft = _precompute_all_soft_numpy(wgc_cache, sim_matrix, self._K, 3)
```

**Memory:** K=64, V=72 → D tensor = 75 MB, peak ~200 MB. Acceptable.
**Expected speedup:** ~2000× (32s → ~16ms).

---

### Task 2 (DONE 2026-03-08) — Vectorize `foveal_sequence` inner loop in `vision_pipeline.py`

**File:** `experiments/symbolic_ai/vision_pipeline.py`
**Function:** `foveal_sequence` (line ~439), the double `for r, for c` loop

**Problem:** 9216 individual numpy `_quantize` calls on 1×1 arrays per crop.
Each numpy call has ~2-5μs fixed overhead → 18-46ms per crop.
300 crops (100 images × 3 fixations) = 5.5-13.8s just for _quantize.

**Fix:** Replace the entire double loop body. Add module-level LUT:
```python
# Add near top of file, after imports:
_GRAY_LUT_3BIT = ['00', '10', '20', '30', '40', '50', '60', '70']
```

Replace the `for r in range(crop_rows): for c in range(crop_cols):` block with:
```python
import numpy as np

# --- Vectorized content hashes (fine_patch=1 only) ---
if fine_patch == 1:
    levels = (1 << quant_bits) - 1
    q = np.clip(np.round(crop * levels), 0, levels).astype(np.uint8)
    lut = _GRAY_LUT_3BIT if quant_bits == 3 else None
    if lut:
        content_hashes = [lut[v] for v in q.ravel()]
    else:
        content_hashes = [_quantize(crop[r:r+1, c:c+1], quant_bits)
                          for r in range(crop_rows) for c in range(crop_cols)]
else:
    content_hashes = [_quantize(crop[r*fine_patch:(r+1)*fine_patch,
                                     c*fine_patch:(c+1)*fine_patch], quant_bits)
                      for r in range(crop_rows) for c in range(crop_cols)]

# --- Vectorized position tokens ---
rs = np.arange(crop_rows)
cs = np.arange(crop_cols)
if relative_pos:
    dr_bins = np.clip(
        (((rs - anchor_r) * n_pos_bins) / crop_rows).astype(int), -half, half - 1)
    dc_bins = np.clip(
        (((cs - anchor_c) * n_pos_bins) / crop_cols).astype(int), -half, half - 1)
    dr_grid, dc_grid = np.meshgrid(dr_bins, dc_bins, indexing='ij')
    pos_tokens = [f'D{dr},{dc}' for dr, dc in
                  zip(dr_grid.ravel().tolist(), dc_grid.ravel().tolist())]
else:
    pr_bins = (rs / crop_rows * n_pos_bins).astype(int)
    pc_bins = (cs / crop_cols * n_pos_bins).astype(int)
    pr_grid, pc_grid = np.meshgrid(pr_bins, pc_bins, indexing='ij')
    pos_tokens = [f'P{pr},{pc}' for pr, pc in
                  zip(pr_grid.ravel().tolist(), pc_grid.ravel().tolist())]

# Interleave pos + content
tokens = [v for pair in zip(pos_tokens, content_hashes) for v in pair]
```

**Expected speedup: 5-10× for foveal_sequence.**

---

### Task 3 (DONE 2026-03-08) — ExampleStore indexed lookup in `engine.py`

The `ask()` method does O(n) linear scan over `store.examples`.
After E0+E1 with 427K bigrams, each scan costs O(427K).

**Fix:** Added `_index: Dict[tuple, tuple]` field to `ExampleStore` (init=False).
- `add()` populates index; keeps first match (consistent with old linear scan)
- `lookup()` does O(1) dict lookup; falls back to O(n) for unhashable inputs (numpy)
- `ask()` and `ask_dist()` in engine.py now call `store.lookup(query)` instead of loop
- `_inputs_equal` in engine.py is now unused (left in place)

---

## Parallelization opportunities (after single-thread optimization)

1. **Parallel fixation processing:** `fit_images` processes fixations serially.
   `concurrent.futures.ThreadPoolExecutor` for `foveal_sequence` calls (GIL-free for numpy).

2. **Parallel image processing:** Each image's foveal sequences are independent.
   Can process images in batches with `ProcessPoolExecutor`.

3. **Batch `_precompute_all_soft_numpy`:** Already vectorized across all K^arity queries.
   No further parallelization needed after Task 1.

---

## Verification

After implementing Tasks 1+2:
- Run `python vision_pipeline.py --classify --n_images 30`
- Should complete in <30 seconds (was ~6 minutes)
- Check that vocab now includes >2 content tokens ('00'..'70')
- Classification accuracy should be >50% on horizontal/vertical stripes
