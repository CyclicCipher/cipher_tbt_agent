# Algorithmic Changes from Standard Mamba3

This file documents our departures from the standard Mamba3 architecture (Dao & Gu 2024) and the empirical evidence motivating each change.

---

## 1. PoPE Replaces RoPE (2026-02-13)

**Change:** Default positional embedding switched from RoPE (Rotary Position Embeddings) to PoPE (Polar Positional Embeddings) in `Mamba3Config.use_pope`.

**What PoPE does:** Encodes positions as phase angles on the complex unit circle using learnable frequencies with polar decomposition, rather than RoPE's fixed sinusoidal rotations. The key difference is that PoPE separates magnitude and phase, making distance information scale-invariant.

**Motivation:** Diagnostic comparison across three retrieval tasks showed PoPE dominates RoPE with no speed penalty.

### Benchmark Results (seq_len=16, d_model=64, d_state=32, 2 layers, 100 epochs)

| Task | RoPE test | PoPE test | Delta |
|------|-----------|-----------|-------|
| What (content matching) | 5.5% | 99.2% | +93.7% |
| What+Where (indirect indexing) | 11.7% | 87.3% | +75.6% |
| Where (position retrieval) | 7.5% | 14.8% | +7.3% |

PoPE also uses fewer parameters (157,272 vs 161,432) and trains slightly faster (85.9s vs 89.7s on the Where task).

### Per-Position Accuracy on Where Task

The Where task asks: "given a sequence [t0, t1, ..., t14, pos], output t_pos". Per-position diagnosis reveals the SSM memory decay profile:

```
PoPE — Per-position accuracy:
 Pos  Dist     Acc
   0    15   13.6%   (anomalous — possible BOC privilege)
   1    14    7.6%
   2    13    8.0%
   3    12    5.4%
   4    11    6.6%
   5    10    6.8%
   6     9    4.6%
   7     8    5.4%
   8     7    7.8%
   9     6    6.0%
  10     5    9.6%
  11     4   10.6%
  12     3   15.4%
  13     2   30.8%
  14     1   68.8%   ← nearest position

RoPE — Per-position accuracy:
 Pos  Dist     Acc
  13     2   18.4%
  14     1   20.6%   ← nearest position
  (all others: 3-6%, near random chance of 3.1%)
```

### Key Findings

1. **PoPE's advantage is retrieval fidelity, not memory depth.** Both architectures have the same ~2 position effective memory window. But PoPE retrieves 68.8% accuracy at distance 1 vs RoPE's 20.6%. PoPE helps the SSM use what it remembers, not remember further back.

2. **The Where task ceiling (~14.5%) is explained by memory decay.** PoPE gets 68.8% at distance 1, 30.8% at distance 2, then exponential decay. Averaged over 15 uniformly-sampled query positions, this yields ~14%.

3. **Position 0 anomaly:** PoPE shows 13.6% at position 0 (higher than positions 1-9). Beginning-of-sequence may receive privileged encoding in the initial hidden state.

4. **SSM memory is the bottleneck.** With d_state=32, the model can only effectively retrieve from ~1-2 most recent positions. This is the primary limitation for tasks requiring arbitrary-position access.

### Drawbacks

None observed. PoPE matches or exceeds RoPE on every metric tested.

### Files Changed

- `mamba3_block.py`: `Mamba3Config.use_pope` default changed from `False` to `True`
- `test_pope_vs_rope.py`: Added `--diagnose_where` flag and `diagnose_where_by_position()` for per-position accuracy breakdown
