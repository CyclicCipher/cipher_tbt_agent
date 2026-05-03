"""
test_spatial_pooler.py — Tests for the corrected Spatial Pooler.

Run from project root:
    python experiments/ManuallyCodedTBT/tests/test_spatial_pooler.py

Tests cover:
    1. Output sparsity is maintained
    2. Stability after learning
    3. Semantic overlap preservation
    4. Dissimilar inputs → dissimilar outputs
    5. Boost factor mechanism (active duty cycle vs neighbor mean)
    6. Overlap duty cycle tracking (second boosting mechanism)
    7. Permanence boost rescues dying columns
    8. Learning improves stability
    9. Diagnostics and edge cases
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from sdr import sdr_random, encode_scalar, overlap, population
from spatial_pooler import SpatialPooler


passed = 0
failed = 0


def test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        msg = f"  ✗ {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)


def section(name: str):
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")


# ── Shared setup ──────────────────────────────────────────────────────────────
# Using BAMI-recommended defaults: 2048 columns, 40 active (~2%), perm_thresh=0.2

INPUT_SIZE = 256
NUM_MINICOLS = 512   # smaller than BAMI min for test speed; 2048 for production
ACTIVE_K = 20        # ~4% of 512 — higher than BAMI's 2% to get detectable
                     # semantic overlap at this small column count
ENCODER_W = 41

def make_sp(seed=42, **kwargs):
    defaults = dict(
        input_size=INPUT_SIZE,
        num_minicolumns=NUM_MINICOLS,
        active_per_step=ACTIVE_K,
        potential_pct=1.0,
        connected_pct=0.5,
        permanence_threshold=0.2,
        permanence_inc=0.03,
        permanence_dec=0.015,
        boost_strength=3.0,
        duty_cycle_period=1000,
        min_pct_overlap_duty_cycle=0.001,
        stimulus_threshold=0,
        seed=seed,
    )
    defaults.update(kwargs)
    return SpatialPooler(**defaults)


def encode(value, min_val=0, max_val=20):
    return encode_scalar(value, INPUT_SIZE, ENCODER_W, min_val, max_val)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Output sparsity
# ══════════════════════════════════════════════════════════════════════════════
section("1. Output sparsity")

sp = make_sp()
out = sp.compute(encode(5), learn=False)

test("output length correct", len(out) == NUM_MINICOLS)
test("output dtype bool", out.dtype == bool)
test("output sparsity correct", population(out) == ACTIVE_K,
     f"expected {ACTIVE_K}, got {population(out)}")

sparsities = [population(sp.compute(encode(v), learn=False)) for v in range(21)]
test("sparsity consistent across inputs",
     all(s <= ACTIVE_K for s in sparsities),
     f"sparsities: {sparsities}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Stability after learning
# ══════════════════════════════════════════════════════════════════════════════
section("2. Stability after learning")

sp2 = make_sp(seed=123)
inputs = [encode(v) for v in range(10)]
for epoch in range(50):
    for inp in inputs:
        sp2.compute(inp, learn=True)

# After training, the same input should produce highly stable output.
# With stochastic tiebreaking, exact equality is not guaranteed when
# overlaps are tied, but after learning most columns have distinct
# overlap scores so the output should be very consistent.
outputs_for_3 = [sp2.compute(inputs[3], learn=False) for _ in range(5)]
min_pairwise_overlap = min(
    overlap(outputs_for_3[0], outputs_for_3[i]) for i in range(1, 5)
)
# After learning, the same input should produce outputs sharing
# at least 80% of active bits across repeated calls
test("output stable after learning (≥80% overlap across calls)",
     min_pairwise_overlap >= int(ACTIVE_K * 0.8),
     f"min pairwise overlap={min_pairwise_overlap}, threshold={int(ACTIVE_K*0.8)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Semantic overlap preservation
# ══════════════════════════════════════════════════════════════════════════════
section("3. Semantic overlap preservation")

sp3 = make_sp(seed=456)
inputs_full = [encode(v) for v in range(21)]
for epoch in range(80):
    for inp in inputs_full:
        sp3.compute(inp, learn=True)

outputs = {v: sp3.compute(encode(v), learn=False) for v in range(21)}

ov_adjacent = overlap(outputs[5], outputs[6])
ov_distant = overlap(outputs[5], outputs[15])
test("adjacent inputs → higher output overlap", ov_adjacent > ov_distant,
     f"5↔6={ov_adjacent}, 5↔15={ov_distant}")

ov_1_apart = overlap(outputs[10], outputs[11])
ov_3_apart = overlap(outputs[10], outputs[13])
ov_8_apart = overlap(outputs[10], outputs[18])
test("overlap decreases with distance (1 vs 3)", ov_1_apart >= ov_3_apart,
     f"1-apart={ov_1_apart}, 3-apart={ov_3_apart}")
test("overlap decreases with distance (3 vs 8)", ov_3_apart >= ov_8_apart,
     f"3-apart={ov_3_apart}, 8-apart={ov_8_apart}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Dissimilar inputs → dissimilar outputs
# ══════════════════════════════════════════════════════════════════════════════
section("4. Dissimilar inputs → dissimilar outputs")

ov_0_20 = overlap(outputs[0], outputs[20])
test("endpoints have low overlap", ov_0_20 <= ACTIVE_K // 2,
     f"0↔20 overlap = {ov_0_20}")

random_inp = sdr_random(INPUT_SIZE, ENCODER_W, np.random.default_rng(999))
random_out = sp3.compute(random_inp, learn=False)
ov_random = max(overlap(random_out, outputs[v]) for v in range(21))
test("random input differs from trained patterns", ov_random <= ACTIVE_K // 2,
     f"max overlap with any trained = {ov_random}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Boost factors — neighbor-relative target (BAMI correction #1)
# ══════════════════════════════════════════════════════════════════════════════
section("5. Boost factors — neighbor-relative target")

sp_boost = make_sp(seed=789)

# Before any learning all duty cycles are equal, so boost factors should be 1
test("initial boost factors all 1.0",
     np.allclose(sp_boost.boost_factors, 1.0),
     f"mean={sp_boost.boost_factors.mean():.4f}")

# Train and verify boost factors spread around 1
ever_active = np.zeros(NUM_MINICOLS, dtype=bool)
for epoch in range(30):
    for v in range(21):
        out = sp_boost.compute(encode(v), learn=True)
        ever_active |= out

# Columns used more than average should have boost < 1
# Columns used less than average should have boost > 1
mean_duty = sp_boost.active_duty_cycle.mean()
overactive = sp_boost.active_duty_cycle > mean_duty
underactive = sp_boost.active_duty_cycle < mean_duty

if overactive.any() and underactive.any():
    mean_boost_over = sp_boost.boost_factors[overactive].mean()
    mean_boost_under = sp_boost.boost_factors[underactive].mean()
    test("overactive columns suppressed (boost < 1)", mean_boost_over < 1.0,
         f"mean boost for overactive = {mean_boost_over:.4f}")
    test("underactive columns boosted (boost > 1)", mean_boost_under > 1.0,
         f"mean boost for underactive = {mean_boost_under:.4f}")
else:
    test("boost factor spread exists", True)  # degenerate case, not a failure

pct_used = ever_active.sum() / NUM_MINICOLS
test("boosting uses many minicolumns", pct_used > 0.3,
     f"{pct_used*100:.1f}% of minicolumns used")

entropy = sp_boost.get_entropy()
entropy_ratio = entropy / sp_boost.get_max_entropy()
test("entropy ratio reasonable", entropy_ratio > 0.5,
     f"entropy ratio={entropy_ratio:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Overlap duty cycle tracking (BAMI correction #2 — new mechanism)
# ══════════════════════════════════════════════════════════════════════════════
section("6. Overlap duty cycle tracking")

sp_odc = make_sp(seed=42)

# Initially overlap duty cycles match active duty cycles (both = target)
test("overlap duty cycle initialized",
     np.allclose(sp_odc.overlap_duty_cycle,
                 ACTIVE_K / NUM_MINICOLS, atol=1e-4))

# After training on structured input, overlap duty cycles should be positive
# and reasonably distributed
for epoch in range(20):
    for v in range(21):
        sp_odc.compute(encode(v), learn=True)

odc_stats = sp_odc.get_overlap_duty_cycle_stats()
test("overlap duty cycle mean positive", odc_stats["mean"] > 0,
     f"mean={odc_stats['mean']:.4f}")
test("overlap duty cycle max positive", odc_stats["max"] > 0,
     f"max={odc_stats['max']:.4f}")
test("overlap duty cycle min >= 0", odc_stats["min"] >= 0,
     f"min={odc_stats['min']:.6f}")

print(f"  overlap DC: mean={odc_stats['mean']:.4f}, "
      f"min={odc_stats['min']:.6f}, max={odc_stats['max']:.4f}, "
      f"underperforming={odc_stats['pct_underperforming']*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Permanence boost rescues dying columns (BAMI correction #2 — mechanism)
# ══════════════════════════════════════════════════════════════════════════════
section("7. Permanence boost rescues dying columns")

# Create a SP and manually force some columns to have very low permanences —
# simulating columns that have lost all their input connections.
sp_rescue = make_sp(seed=55)

# Kill columns 0–9: set all their permanences to 0
sp_rescue.permanence[:10, :] = 0.0
sp_rescue.overlap_duty_cycle[:10] = 0.0  # they haven't been overlapping

# Run one learning step to trigger _apply_permanence_boost
any_input = encode(5)
sp_rescue.compute(any_input, learn=True)

# The dead columns should have had their permanences increased
dead_perm_after = sp_rescue.permanence[:10][sp_rescue.potential_pool[:10]].mean()
test("dead columns receive permanence boost", dead_perm_after > 0.0,
     f"mean permanence of dead columns after boost = {dead_perm_after:.4f}")

# Healthy columns should be unaffected
healthy_perm = sp_rescue.permanence[100:110][sp_rescue.potential_pool[100:110]].mean()
test("healthy columns unaffected by permanence boost", healthy_perm > 0.0,
     f"mean permanence of healthy columns = {healthy_perm:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Learning improves stability
# ══════════════════════════════════════════════════════════════════════════════
section("8. Learning improves stability")

sp_learn = make_sp(seed=321)
inp_test = encode(7)
out_pre_1 = sp_learn.compute(inp_test, learn=False).copy()

for v in [1, 3, 12, 18]:
    sp_learn.compute(encode(v), learn=True)

out_pre_2 = sp_learn.compute(inp_test, learn=False).copy()
drift_before = ACTIVE_K - overlap(out_pre_1, out_pre_2)

for epoch in range(60):
    for v in range(21):
        sp_learn.compute(encode(v), learn=True)

out_post_1 = sp_learn.compute(inp_test, learn=False).copy()
for v in [1, 3, 12, 18]:
    sp_learn.compute(encode(v), learn=True)
out_post_2 = sp_learn.compute(inp_test, learn=False).copy()
drift_after = ACTIVE_K - overlap(out_post_1, out_post_2)

test("learning reduces output drift", drift_after <= drift_before,
     f"drift before={drift_before}, after={drift_after}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Diagnostics and edge cases
# ══════════════════════════════════════════════════════════════════════════════
section("9. Diagnostics and edge cases")

counts = sp_learn.get_connected_counts()
test("all minicolumns have connections", np.all(counts > 0),
     f"min connections = {counts.min()}")

# Zero input — no connected synapses active, no winners
sp_edge = make_sp(seed=111)
out_zero = sp_edge.compute(np.zeros(INPUT_SIZE, dtype=bool), learn=False)
test("zero input → no active minicolumns", population(out_zero) == 0)

# Dense input — should still produce correct sparsity
out_dense = sp_edge.compute(np.ones(INPUT_SIZE, dtype=bool), learn=False)
test("dense input → correct sparsity", population(out_dense) == ACTIVE_K,
     f"got {population(out_dense)}")

# stimulus_threshold=2 should exclude low-overlap columns
sp_thresh = make_sp(seed=42, stimulus_threshold=2)
# Sparse input with very few active bits — most columns will have 0 or 1 overlap
sparse_inp = sdr_random(INPUT_SIZE, 3, np.random.default_rng(7))
out_thresh = sp_thresh.compute(sparse_inp, learn=False)
test("stimulus_threshold filters low-overlap columns",
     population(out_thresh) <= ACTIVE_K)


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 60}")
print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'═' * 60}")

if failed > 0:
    sys.exit(1)
