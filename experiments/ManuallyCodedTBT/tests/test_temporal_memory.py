"""
test_temporal_memory.py — Tests for Temporal Memory.

Run from project root:
    python experiments/ManuallyCodedTBT/tests/test_temporal_memory.py

Tests cover:
    1. Output shape and basic properties
    2. Bursting on novel input (no prediction = all cells in column fire)
    3. Prediction after learning a sequence (anomaly score drops)
    4. Higher-order memory (same element, different context → different cell)
    5. Prediction density stays low
    6. Reset clears all state
    7. Segment and synapse growth
    8. Incorrect predictions are punished
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from temporal_memory import TemporalMemory


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


def make_col_sdr(n_cols: int, active: list) -> np.ndarray:
    """Create a minicolumn SDR with specified columns active."""
    sdr = np.zeros(n_cols, dtype=bool)
    sdr[active] = True
    return sdr


# ── Shared setup ──────────────────────────────────────────────────────────────

N_COLS = 64
CELLS = 4

def make_tm(seed=42, **kwargs):
    defaults = dict(
        num_minicolumns=N_COLS,
        cells_per_col=CELLS,
        max_segs_per_cell=32,
        max_synapses_per_seg=32,
        segment_activation_threshold=3,  # below winner cell count (~5) per step
        min_threshold=2,
        permanence_threshold=0.5,
        permanence_inc=0.1,
        permanence_dec=0.1,
        permanence_punish=0.01,
        initial_permanence=0.21,
        max_new_synapses_per_seg=15,
        seed=seed,
    )
    defaults.update(kwargs)
    return TemporalMemory(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Output shape and basic properties
# ══════════════════════════════════════════════════════════════════════════════
section("1. Output shape and basic properties")

tm = make_tm()
inp = make_col_sdr(N_COLS, [0, 5, 10, 15, 20])
out = tm.compute(inp, learn=False)

test("output length = total_cells", len(out) == N_COLS * CELLS)
test("output dtype bool", out.dtype == bool)
test("output has active cells", out.sum() > 0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Bursting on novel input
# ══════════════════════════════════════════════════════════════════════════════
section("2. Bursting on novel input")

tm2 = make_tm()
inp = make_col_sdr(N_COLS, [0, 5, 10])

# First input ever — no predictions exist — all active columns should burst
out = tm2.compute(inp, learn=False)

# Each active column (3 cols) should have all CELLS cells active
active_cols = [0, 5, 10]
for col in active_cols:
    col_cells = out[col * CELLS:(col + 1) * CELLS]
    test(f"col {col} bursts (all {CELLS} cells active)", col_cells.all(),
         f"active: {col_cells.sum()}/{CELLS}")

# Inactive columns should have no active cells
inactive_col_cells = out[1 * CELLS:2 * CELLS]
test("inactive col has no active cells", not inactive_col_cells.any())

# Anomaly score should be 1.0 (everything burst)
test("anomaly score = 1.0 on first input", tm2.get_anomaly_score() == 1.0,
     f"got {tm2.get_anomaly_score():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Prediction after learning a sequence
# ══════════════════════════════════════════════════════════════════════════════
section("3. Prediction after learning — anomaly score drops")

tm3 = make_tm(seed=1)

# Define a simple sequence: A → B → C → D
# Each element is a set of active minicolumns
seq = [
    make_col_sdr(N_COLS, [0, 1, 2, 3, 4]),
    make_col_sdr(N_COLS, [10, 11, 12, 13, 14]),
    make_col_sdr(N_COLS, [20, 21, 22, 23, 24]),
    make_col_sdr(N_COLS, [30, 31, 32, 33, 34]),
]

# Measure anomaly score before training
tm3.reset()
anomaly_before = []
for _ in range(3):
    for step in seq:
        tm3.compute(step, learn=False)
        anomaly_before.append(tm3.get_anomaly_score())

# Train on the sequence for many epochs
for epoch in range(30):
    tm3.reset()
    for step in seq:
        tm3.compute(step, learn=True)

# Measure anomaly score after training (skip first step — no prior context)
tm3.reset()
anomaly_after = []
for step in seq:
    tm3.compute(step, learn=False)
    anomaly_after.append(tm3.get_anomaly_score())

# After the first step, subsequent steps should be predicted
mean_before = np.mean(anomaly_before)
mean_after = np.mean(anomaly_after[1:])  # skip step 0 which has no context

test("anomaly score drops after learning",
     mean_after < mean_before,
     f"before={mean_before:.3f}, after={mean_after:.3f}")

test("sequence is predicted (anomaly < 0.5 after training)",
     mean_after < 0.5,
     f"mean anomaly after step 0: {mean_after:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Higher-order memory — same element, different context → different cell
# ══════════════════════════════════════════════════════════════════════════════
section("4. Higher-order memory")

# Two sequences sharing element B: A→B→C and X→B→Y
# After learning, B should activate different cells depending on whether
# A or X came before it — different cells encode different sequence contexts.
tm4 = make_tm(seed=2)

A = make_col_sdr(N_COLS, [0, 1, 2, 3, 4])
B = make_col_sdr(N_COLS, [10, 11, 12, 13, 14])
C = make_col_sdr(N_COLS, [20, 21, 22, 23, 24])
X = make_col_sdr(N_COLS, [30, 31, 32, 33, 34])
Y = make_col_sdr(N_COLS, [40, 41, 42, 43, 44])

for epoch in range(40):
    tm4.reset()
    for step in [A, B, C]:
        tm4.compute(step, learn=True)
    tm4.reset()
    for step in [X, B, Y]:
        tm4.compute(step, learn=True)

# Now present A then B — record which B cells activate
tm4.reset()
tm4.compute(A, learn=False)
B_after_A = tm4.compute(B, learn=False).copy()

# Now present X then B — record which B cells activate
tm4.reset()
tm4.compute(X, learn=False)
B_after_X = tm4.compute(B, learn=False).copy()

# The B cells in column 10-14 should differ between the two contexts
# (different cells within those minicolumns should fire)
b_cols = list(range(10, 15))
b_cells_A = np.concatenate([B_after_A[c*CELLS:(c+1)*CELLS] for c in b_cols])
b_cells_X = np.concatenate([B_after_X[c*CELLS:(c+1)*CELLS] for c in b_cols])

n_shared = (b_cells_A & b_cells_X).sum()
n_total_A = b_cells_A.sum()
n_total_X = b_cells_X.sum()

# Perfect higher-order: 0 shared. In practice expect significantly less than all.
# We test that they are not identical (not all cells shared)
test("B activates different cells in different contexts",
     n_shared < min(n_total_A, n_total_X),
     f"shared={n_shared}, A_active={n_total_A}, X_active={n_total_X}")

print(f"  B after A: {n_total_A} cells active, "
      f"B after X: {n_total_X} cells active, shared: {n_shared}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Prediction density stays low
# ══════════════════════════════════════════════════════════════════════════════
section("5. Prediction density stays low")

tm5 = make_tm(seed=3)
seq5 = [make_col_sdr(N_COLS, [i, i+1, i+2, i+3]) for i in range(0, 40, 8)]

densities = []
for epoch in range(20):
    tm5.reset()
    for step in seq5:
        tm5.compute(step, learn=True)
        densities.append(tm5.get_prediction_density())

mean_density = np.mean(densities)
test("prediction density stays sparse", mean_density < 0.3,
     f"mean density = {mean_density:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Reset clears all state
# ══════════════════════════════════════════════════════════════════════════════
section("6. Reset clears all cell state")

tm6 = make_tm()
for step in [make_col_sdr(N_COLS, [0, 1, 2])]:
    tm6.compute(step, learn=True)

tm6.reset()
test("cell_active cleared", not tm6.cell_active.any())
test("cell_predictive cleared", not tm6.cell_predictive.any())
test("cell_winner cleared", not tm6.cell_winner.any())
test("prev_active cleared", not tm6.prev_active.any())

# After reset, first input should burst again (no context)
out_after_reset = tm6.compute(make_col_sdr(N_COLS, [0, 1, 2]), learn=False)
cols = [0, 1, 2]
all_burst = all(
    out_after_reset[c*CELLS:(c+1)*CELLS].all() for c in cols
)
test("first input after reset bursts", all_burst)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Segment and synapse growth
# ══════════════════════════════════════════════════════════════════════════════
section("7. Segment and synapse growth")

tm7 = make_tm(seed=5)
seq7 = [make_col_sdr(N_COLS, [0, 5, 10]), make_col_sdr(N_COLS, [1, 6, 11])]

stats_before = tm7.get_segment_stats()
for epoch in range(10):
    tm7.reset()
    for step in seq7:
        tm7.compute(step, learn=True)
stats_after = tm7.get_segment_stats()

test("segments are created during learning",
     stats_after["total_segments_created"] > stats_before["total_segments_created"],
     f"before={stats_before['total_segments_created']}, "
     f"after={stats_after['total_segments_created']}")

test("synapses are grown during learning",
     stats_after["total_cell_synapses"] > 0,
     f"total synapses = {stats_after['total_cell_synapses']}")

print(f"  segs created: {stats_after['total_segments_created']}, "
      f"synapses: {stats_after['total_cell_synapses']}, "
      f"mean segs/cell: {stats_after['mean_segs_per_cell']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Incorrect predictions are punished (permanences decay)
# ══════════════════════════════════════════════════════════════════════════════
section("8. Incorrect predictions are punished")

tm8 = make_tm(seed=6)
seq8 = [make_col_sdr(N_COLS, [0, 1, 2]), make_col_sdr(N_COLS, [10, 11, 12])]

# Train so predictions exist
for epoch in range(15):
    tm8.reset()
    for step in seq8:
        tm8.compute(step, learn=True)

# Present seq8[0] WITHOUT learning to create predictive state cleanly
# (no permanence modification at this step)
tm8.reset()
tm8.compute(seq8[0], learn=False)

# Confirm predictions are present after seq8[0]
n_predictive = tm8.cell_predictive.sum()
test("predictions exist before wrong input",
     n_predictive > 0,
     f"predictive cells = {n_predictive}")

# Record permanences AFTER seq8[0] but BEFORE wrong input
perm_before = tm8.seg_syn_perm.copy()

# Present a wrong input with learning — punishment should reduce some permanences
wrong = make_col_sdr(N_COLS, [50, 51, 52])
tm8.compute(wrong, learn=True)

perm_after = tm8.seg_syn_perm.copy()

# Permanences of incorrectly predicting segments should have decreased somewhere
# (only valid synapses matter — ignore empty slots which stay at 0.0)
valid = tm8.seg_syn_cells >= 0
any_decreased = (perm_after[valid] < perm_before[valid]).any()
test("some permanences decreased due to punishment", any_decreased)


# ══════════════════════════════════════════════════════════════════════════════
# 9. Location conditioning — same feature, different location → different cell
# ══════════════════════════════════════════════════════════════════════════════
section("9. Location conditioning (TBT extension)")

from grid_cells import GridCellLayer
from sdr import overlap as sdr_overlap

gc = GridCellLayer(periods=[7.0, 11.0], sdr_length_per_module=32, sdr_width_per_module=7)
LOC_LEN = gc.total_sdr_length  # 64

tm_loc = TemporalMemory(
    num_minicolumns=N_COLS,
    cells_per_col=CELLS,
    location_sdr_length=LOC_LEN,
    max_loc_synapses_per_seg=8,
    segment_activation_threshold=3,
    min_threshold=2,
    permanence_threshold=0.5,
    permanence_inc=0.1,
    permanence_dec=0.1,
    initial_permanence=0.21,
    max_new_synapses_per_seg=10,
    max_new_loc_synapses_per_seg=6,
    seed=99,
)

preamble = make_col_sdr(N_COLS, [0, 1, 2, 3, 4])
feature  = make_col_sdr(N_COLS, [10, 11, 12, 13, 14])

gc.set_position(3.0)
loc_A = gc.get_location_sdr()
gc.set_position(50.0)
loc_B = gc.get_location_sdr()

loc_overlap = sdr_overlap(loc_A, loc_B)
test("location SDRs at pos 3 and 50 are distinct", loc_overlap < LOC_LEN // 4,
     f"overlap = {loc_overlap}")

# Train 2-step sequences so prev_winner cells exist when feature is learned:
#   [preamble @ loc_A] → [feature @ loc_A]
#   [preamble @ loc_B] → [feature @ loc_B]
for epoch in range(60):
    tm_loc.reset()
    tm_loc.compute(preamble, location_sdr=loc_A, learn=True)
    tm_loc.compute(feature,  location_sdr=loc_A, learn=True)
    tm_loc.reset()
    tm_loc.compute(preamble, location_sdr=loc_B, learn=True)
    tm_loc.compute(feature,  location_sdr=loc_B, learn=True)

stats_loc = tm_loc.get_segment_stats()
test("location synapses were grown",
     stats_loc["total_location_synapses"] > 0,
     f"location synapses = {stats_loc['total_location_synapses']}")
test("cell synapses were also grown",
     stats_loc["total_cell_synapses"] > 0,
     f"cell synapses = {stats_loc['total_cell_synapses']}")

print(f"  cell synapses: {stats_loc['total_cell_synapses']}, "
      f"location synapses: {stats_loc['total_location_synapses']}")

# Feature should be predicted after preamble at location A
tm_loc.reset()
tm_loc.compute(preamble, location_sdr=loc_A, learn=False)
tm_loc.compute(feature,  location_sdr=loc_A, learn=False)
anomaly_A = tm_loc.get_anomaly_score()
test("feature predicted after preamble at location A",
     anomaly_A < 1.0,
     f"anomaly = {anomaly_A:.3f}")

# Backward compatibility
tm_plain = make_tm(seed=42)
out_plain = tm_plain.compute(make_col_sdr(N_COLS, [0, 1, 2]), learn=False)
test("plain TM (no location) still works", out_plain.sum() > 0)


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 60}")
print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'═' * 60}")

if failed > 0:
    sys.exit(1)