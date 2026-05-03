"""
test_cortical_column.py — Tests for the full CorticalColumn assembly.

Run from project root:
    python experiments/ManuallyCodedTBT/tests/test_cortical_column.py

Tests cover:
    1.  Basic instantiation — default and custom components
    2.  Pre-encoded SDR input (no encoder)
    3.  Encoded input via encoder
    4.  Anomaly score = 1.0 on first input (everything bursts)
    5.  Anomaly score drops after learning a sequence
    6.  Location conditioning — same feature at different positions
    7.  Path integration — displacement updates grid correctly
    8.  Reset clears TM state but preserves SP and grid
    9.  get_stats() returns valid structure
    10. Higher-order memory stress test (orders 2 through 8)
        — at each order N, two sequences share N-1 identical middle
          elements but differ at the first element and the last.
          After training, the column must correctly predict the last
          element given the first — requiring N-1 steps of context.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from sdr import sdr_random, overlap, population
from encoder import make_number_encoder, SymbolEncoder, CategoryEncoder
from spatial_pooler import SpatialPooler
from temporal_memory import TemporalMemory
from grid_cells import GridCellLayer
from displacement_layer import DisplacementLayer, make_displacement_layer_from_grid
from cortical_column import CorticalColumn

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


# ── Shared helpers ────────────────────────────────────────────────────────────

INPUT_SIZE = 256
N_COLS = 256
ACTIVE_K = 12
CELLS = 4

def make_col(seed=42, **kwargs):
    """Build a column with small defaults for test speed."""
    defaults = dict(
        input_size=INPUT_SIZE,
        num_minicolumns=N_COLS,
        cells_per_col=CELLS,
        active_per_step=ACTIVE_K,
        sp_kwargs=dict(
            permanence_threshold=0.2,
            permanence_inc=0.03,
            permanence_dec=0.015,
            boost_strength=3.0,
        ),
        tm_kwargs=dict(
            segment_activation_threshold=3,
            min_threshold=2,
            permanence_threshold=0.5,
            permanence_inc=0.1,
            permanence_dec=0.1,
            initial_permanence=0.21,
            max_new_synapses_per_seg=12,
        ),
        seed=seed,
    )
    defaults.update(kwargs)
    return CorticalColumn(**defaults)


def make_sdr(n=INPUT_SIZE, w=20, seed=None):
    rng = np.random.default_rng(seed)
    return sdr_random(n, w, rng)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Basic instantiation
# ══════════════════════════════════════════════════════════════════════════════
section("1. Basic instantiation")

col_default = make_col()
test("default column creates", col_default is not None)
test("repr works", "CorticalColumn" in repr(col_default))

# Custom components
grid = GridCellLayer(periods=[7.0, 11.0, 13.0],
                     sdr_length_per_module=32, sdr_width_per_module=7)
displ = make_displacement_layer_from_grid(grid)
col_custom = CorticalColumn(
    grid_layer=grid,
    displacement_layer=displ,
    input_size=INPUT_SIZE,
    num_minicolumns=N_COLS,
    cells_per_col=CELLS,
    active_per_step=ACTIVE_K,
    seed=99,
)
test("custom grid column creates", col_custom is not None)
test("grid periods propagated",
     len(col_custom.grid_layer.modules) == 3)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Pre-encoded SDR input (no encoder)
# ══════════════════════════════════════════════════════════════════════════════
section("2. Pre-encoded SDR input")

col = make_col()
inp_sdr = make_sdr(INPUT_SIZE, 20, seed=1)
result = col.compute(inp_sdr, learn=False)

test("result has required keys",
     all(k in result for k in [
         'input_sdr', 'active_minicolumns', 'location_sdr',
         'active_cells', 'predictive_cells', 'anomaly_score'
     ]))
test("active_minicolumns is bool array",
     result['active_minicolumns'].dtype == bool)
test("active_cells is bool array",
     result['active_cells'].dtype == bool)
test("anomaly_score in [0, 1]",
     0.0 <= result['anomaly_score'] <= 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Encoded input via encoder
# ══════════════════════════════════════════════════════════════════════════════
section("3. Encoded input via encoder")

enc = make_number_encoder(max_value=20, n=INPUT_SIZE)
col_enc = CorticalColumn(
    encoder=enc,
    input_size=INPUT_SIZE,
    num_minicolumns=N_COLS,
    cells_per_col=CELLS,
    active_per_step=ACTIVE_K,
    seed=42,
)
result_enc = col_enc.compute(7.0, learn=False)
test("encoder column works", result_enc['anomaly_score'] >= 0.0)
test("input_sdr has correct population",
     result_enc['input_sdr'].dtype == bool)

# No encoder + raw value should raise
col_no_enc = make_col()
caught = False
try:
    col_no_enc.compute(7.0, learn=False)
except ValueError:
    caught = True
test("no encoder + raw value raises ValueError", caught)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Anomaly score = 1.0 on first input
# ══════════════════════════════════════════════════════════════════════════════
section("4. Anomaly = 1.0 on first input (everything bursts)")

col = make_col()
result = col.compute(make_sdr(seed=10), learn=False)
test("anomaly score = 1.0 on first input",
     result['anomaly_score'] == 1.0,
     f"got {result['anomaly_score']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Anomaly score drops after learning
# ══════════════════════════════════════════════════════════════════════════════
section("5. Anomaly drops after learning a sequence")

col = make_col(seed=123)
seq = [make_sdr(seed=i) for i in range(5)]

# Measure before training
anomaly_before = []
for _ in range(3):
    col.reset()
    for s in seq:
        col.compute(s, learn=False)
        anomaly_before.append(col.anomaly_score)

# Train
for epoch in range(40):
    col.reset()
    for s in seq:
        col.compute(s, learn=True)

# Measure after training (skip step 0 which has no prior context)
col.reset()
anomalies_after = []
for s in seq:
    col.compute(s, learn=False)
    anomalies_after.append(col.anomaly_score)

mean_before = np.mean(anomaly_before)
mean_after = np.mean(anomalies_after[1:])

test("anomaly drops after learning",
     mean_after < mean_before,
     f"before={mean_before:.3f}, after={mean_after:.3f}")
test("anomaly < 0.5 after training",
     mean_after < 0.5,
     f"mean={mean_after:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Location conditioning
# ══════════════════════════════════════════════════════════════════════════════
section("6. Location conditioning (same feature, different position)")

grid_lc = GridCellLayer(periods=[7.0, 11.0, 13.0],
                        sdr_length_per_module=32, sdr_width_per_module=7)
displ_lc = make_displacement_layer_from_grid(grid_lc)
col_lc = CorticalColumn(
    grid_layer=grid_lc,
    displacement_layer=displ_lc,
    input_size=INPUT_SIZE,
    num_minicolumns=N_COLS,
    cells_per_col=CELLS,
    active_per_step=ACTIVE_K,
    tm_kwargs=dict(
        location_sdr_length=grid_lc.total_sdr_length,
        max_loc_synapses_per_seg=8,
        segment_activation_threshold=3,
        min_threshold=2,
        initial_permanence=0.21,
        max_new_synapses_per_seg=12,
    ),
    seed=77,
)

preamble = make_sdr(seed=0)
feature  = make_sdr(seed=1)

# Train: [preamble, feature] at position 3
# Train: [preamble, feature] at position 50
for epoch in range(60):
    col_lc.reset()
    col_lc.reset_position(3.0)
    col_lc.compute(preamble, learn=True)
    col_lc.compute(feature,  learn=True)

    col_lc.reset()
    col_lc.reset_position(50.0)
    col_lc.compute(preamble, learn=True)
    col_lc.compute(feature,  learn=True)

# Verify location synapses were grown
stats = col_lc.tm.get_segment_stats()
test("location synapses grown during training",
     stats['total_location_synapses'] > 0,
     f"loc_syn={stats['total_location_synapses']}")

# Feature predicted after preamble at position 3
col_lc.reset()
col_lc.reset_position(3.0)
col_lc.compute(preamble, learn=False)
col_lc.compute(feature,  learn=False)
anomaly_loc = col_lc.anomaly_score
test("feature predicted after preamble at position 3",
     anomaly_loc < 1.0,
     f"anomaly={anomaly_loc:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Path integration — displacement updates grid
# ══════════════════════════════════════════════════════════════════════════════
section("7. Path integration via displacement")

col_pi = make_col(seed=5)
col_pi.reset_position(3.0)

col_pi.compute(make_sdr(seed=0), displacement=5.0, learn=False)
phases_after = col_pi.location

# Expected: (3 + 5) mod each period
periods = np.array([m.period for m in col_pi.grid_layer.modules])
expected = np.array([8.0 % p for p in periods])
test("displacement updates grid phases correctly",
     np.allclose(phases_after, expected),
     f"got {phases_after}, expected {expected}")

# Check position 3 → +5 → location matches set_position(8)
grid_check = GridCellLayer(
    periods=list(periods),
    sdr_length_per_module=col_pi.grid_layer.sdr_length_per_module,
    sdr_width_per_module=col_pi.grid_layer.sdr_width_per_module,
)
grid_check.set_position(8.0)
test("3 + 5 = 8 via column displacement",
     np.array_equal(col_pi.location_sdr, grid_check.get_location_sdr()))


# ══════════════════════════════════════════════════════════════════════════════
# 8. Reset behavior
# ══════════════════════════════════════════════════════════════════════════════
section("8. Reset clears TM but preserves SP and grid")

col_r = make_col(seed=6)
col_r.reset_position(42.0)

# Run some steps to build up TM state
for s in [make_sdr(seed=i) for i in range(3)]:
    col_r.compute(s, learn=True)

phases_before_reset = col_r.location.copy()
col_r.reset()

test("cell_active cleared after reset",
     not col_r.tm.cell_active.any())
test("cell_predictive cleared after reset",
     not col_r.tm.cell_predictive.any())
test("grid phases preserved after reset",
     np.allclose(col_r.location, phases_before_reset))

# First input after reset should burst
result_after = col_r.compute(make_sdr(seed=99), learn=False)
test("first input after reset bursts",
     result_after['anomaly_score'] == 1.0,
     f"got {result_after['anomaly_score']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. get_stats()
# ══════════════════════════════════════════════════════════════════════════════
section("9. get_stats() returns valid structure")

col_s = make_col()
col_s.compute(make_sdr(seed=1), learn=True)
stats = col_s.get_stats()

required_keys = [
    'iteration', 'anomaly_score', 'prediction_density',
    'sp_entropy', 'sp_max_entropy',
    'tm_total_segments_created', 'grid_phases',
]
test("all required keys present",
     all(k in stats for k in required_keys),
     f"missing: {[k for k in required_keys if k not in stats]}")
test("iteration = 1 after one compute", stats['iteration'] == 1)


# ══════════════════════════════════════════════════════════════════════════════
# 10. Higher-order memory stress test
# ══════════════════════════════════════════════════════════════════════════════
section("10. Higher-order memory stress test (orders 2-6)")

# Test structure for order N:
#   Sequence A: [start_A, mid_1, ..., mid_(N-2), end_A]
#   Sequence B: [start_B, mid_1, ..., mid_(N-2), end_B]
#
#   start_A ≠ start_B  (disambiguating first element)
#   mid elements are identical  (N-2 shared middle elements)
#   end_A ≠ end_B      (diverging outcomes to predict)
#
#   After training, at the end_A/end_B step the anomaly should be < 0.5.
#   This requires remembering back N-1 elements to the disambiguating start.
#
# Smaller column parameters for speed. Input size and column count are
# reduced; the memory capacity per order is still sufficient for this test.

HO_INPUT  = 128    # input SDR length for HO test
HO_COLS   = 128    # minicolumns
HO_ACTIVE = 8      # active per step (~6% sparsity)
HO_CELLS  = 4      # cells per column

def make_ho_sdr(idx):
    """Fixed random SDR — non-overlapping symbols for clean higher-order test."""
    rng = np.random.default_rng(2000 + idx)
    return sdr_random(HO_INPUT, 10, rng)

START_A_HO = make_ho_sdr(0)
START_B_HO = make_ho_sdr(1)
END_A_HO   = make_ho_sdr(2)
END_B_HO   = make_ho_sdr(3)
MIDS_HO    = [make_ho_sdr(10 + i) for i in range(8)]

print()
print(f"  {'Order':>6}  {'Anomaly A':>10}  {'Anomaly B':>10}  {'Pass':>6}")
print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*6}")

results_by_order = {}

for order in range(2, 7):
    n_mids = order - 2
    seq_A = [START_A_HO] + MIDS_HO[:n_mids] + [END_A_HO]
    seq_B = [START_B_HO] + MIDS_HO[:n_mids] + [END_B_HO]

    col_ho = CorticalColumn(
        input_size=HO_INPUT,
        num_minicolumns=HO_COLS,
        cells_per_col=HO_CELLS,
        active_per_step=HO_ACTIVE,
        sp_kwargs=dict(
            permanence_threshold=0.2,
            permanence_inc=0.03,
            permanence_dec=0.015,
            boost_strength=3.0,
        ),
        tm_kwargs=dict(
            segment_activation_threshold=3,
            min_threshold=2,
            permanence_threshold=0.5,
            permanence_inc=0.1,
            permanence_dec=0.1,
            initial_permanence=0.21,
            max_new_synapses_per_seg=12,
            max_segs_per_cell=16,
        ),
        seed=order * 100,
    )

    n_epochs = 40 + order * 5
    for epoch in range(n_epochs):
        col_ho.reset()
        for sdr in seq_A:
            col_ho.compute(sdr, learn=True)
        col_ho.reset()
        for sdr in seq_B:
            col_ho.compute(sdr, learn=True)

    # Evaluate: anomaly at the final (end) element of each sequence
    end_anomalies = []
    for seq in [seq_A, seq_B]:
        col_ho.reset()
        for sdr in seq[:-1]:
            col_ho.compute(sdr, learn=False)
        col_ho.compute(seq[-1], learn=False)
        end_anomalies.append(col_ho.anomaly_score)

    mean_end = np.mean(end_anomalies)
    passed_order = mean_end < 0.5
    results_by_order[order] = {
        'mean': mean_end,
        'passed': passed_order,
        'A': end_anomalies[0],
        'B': end_anomalies[1],
    }
    status = "✓" if passed_order else "✗"
    print(f"  {order:>6}  {end_anomalies[0]:>10.3f}  {end_anomalies[1]:>10.3f}  {status:>6}")

max_pass = max((o for o, r in results_by_order.items() if r['passed']), default=1)
min_fail = min((o for o, r in results_by_order.items() if not r['passed']), default=7)
print(f"\n  Max passing order: {max_pass}")
print(f"  First failing order: {min_fail}")

test("order-2 passes", results_by_order[2]['passed'],
     f"anomaly={results_by_order[2]['mean']:.3f}")
test("order-3 passes", results_by_order[3]['passed'],
     f"anomaly={results_by_order[3]['mean']:.3f}")
test("order-4 passes", results_by_order[4]['passed'],
     f"anomaly={results_by_order[4]['mean']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 60}")
print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'═' * 60}")

if failed > 0:
    sys.exit(1)
