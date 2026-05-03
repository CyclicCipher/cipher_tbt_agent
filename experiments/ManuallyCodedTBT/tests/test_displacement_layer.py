"""
test_displacement_layer.py — Tests for the Displacement Layer (L5b).

Run from project root:
    python experiments/ManuallyCodedTBT/tests/test_displacement_layer.py

Tests cover:
    1. Basic module properties and SDR output
    2. Displacement encoding: adjacent displacements have overlapping SDRs
    3. Negative displacements work natively (subtraction)
    4. apply_to updates a GridCellLayer correctly
    5. 3+5=8 via displacement applied to grid layer
    6. Subtraction: 8-5=3
    7. Commutativity: applying d1 then d2 = applying d2 then d1
    8. make_displacement_layer_from_grid produces compatible layer
    9. Period mismatch raises error
    10. Full round-trip: position → displacement → new position
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from sdr import overlap, population
from grid_cells import GridCellLayer
from displacement_layer import DisplacementLayer, make_displacement_layer_from_grid

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

PERIODS = [7.0, 11.0, 13.0]
SDR_LEN = 64
SDR_W = 13


def make_grid():
    return GridCellLayer(
        periods=PERIODS,
        sdr_length_per_module=SDR_LEN,
        sdr_width_per_module=SDR_W,
    )


def make_displ():
    return DisplacementLayer(
        periods=PERIODS,
        sdr_length_per_module=SDR_LEN,
        sdr_width_per_module=SDR_W,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. Basic properties
# ══════════════════════════════════════════════════════════════════════════════
section("1. Basic module properties")

d = make_displ()
test("num_modules", d.num_modules == 3)
test("total_sdr_length", d.total_sdr_length == 3 * SDR_LEN)

d.set_displacement(5.0)
phases = d.get_phases()
expected = np.array([5.0 % 7.0, 5.0 % 11.0, 5.0 % 13.0])
test("phases correct for displacement 5",
     np.allclose(phases, expected),
     f"got {phases}, expected {expected}")

sdr = d.get_displacement_sdr()
test("sdr length correct", len(sdr) == d.total_sdr_length)
test("sdr dtype bool", sdr.dtype == bool)
test("sdr population correct", population(sdr) == 3 * SDR_W,
     f"got {population(sdr)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Adjacent displacements overlap
# ══════════════════════════════════════════════════════════════════════════════
section("2. Adjacent displacements have overlapping SDRs")

d1, d2 = make_displ(), make_displ()
d1.set_displacement(5.0)
d2.set_displacement(6.0)
d3 = make_displ()
d3.set_displacement(12.0)

sdr_5 = d1.get_displacement_sdr()
sdr_6 = d2.get_displacement_sdr()
sdr_12 = d3.get_displacement_sdr()

ov_adj = overlap(sdr_5, sdr_6)
ov_far = overlap(sdr_5, sdr_12)

test("adjacent displacements have higher overlap than distant",
     ov_adj > ov_far,
     f"5↔6={ov_adj}, 5↔12={ov_far}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Negative displacements (subtraction)
# ══════════════════════════════════════════════════════════════════════════════
section("3. Negative displacements work natively")

d_neg = make_displ()
d_neg.set_displacement(-3.0)
phases_neg = d_neg.get_phases()

# -3 mod 7 = 4, -3 mod 11 = 8, -3 mod 13 = 10
expected_neg = np.array([(-3.0) % 7.0, (-3.0) % 11.0, (-3.0) % 13.0])
test("negative displacement phases correct",
     np.allclose(phases_neg, expected_neg),
     f"got {phases_neg}, expected {expected_neg}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. apply_to updates GridCellLayer correctly
# ══════════════════════════════════════════════════════════════════════════════
section("4. apply_to updates GridCellLayer")

grid = make_grid()
displ = make_displ()

grid.set_position(10.0)
displ.set_displacement(5.0)
displ.apply_to(grid)

phases_after = grid.get_phases()
expected_after = np.array([15.0 % 7.0, 15.0 % 11.0, 15.0 % 13.0])
test("grid phases correct after apply_to",
     np.allclose(phases_after, expected_after),
     f"got {phases_after}, expected {expected_after}")

# Should match set_position(15) directly
grid2 = make_grid()
grid2.set_position(15.0)
test("apply_to gives same result as set_position(15)",
     np.array_equal(grid.get_location_sdr(), grid2.get_location_sdr()))


# ══════════════════════════════════════════════════════════════════════════════
# 5. 3 + 5 = 8 via displacement applied to grid layer
# ══════════════════════════════════════════════════════════════════════════════
section("5. Addition: 3 + 5 = 8")

for a, b in [(3, 5), (0, 8), (7, 3), (50, 50), (13, 13)]:
    grid_sum = make_grid()
    displ_sum = make_displ()

    grid_sum.set_position(float(a))
    displ_sum.apply_displacement_to(float(b), grid_sum)

    grid_direct = make_grid()
    grid_direct.set_position(float(a + b))

    test(f"{a} + {b} = {a+b} via displacement layer",
         np.array_equal(grid_sum.get_location_sdr(),
                        grid_direct.get_location_sdr()))


# ══════════════════════════════════════════════════════════════════════════════
# 6. Subtraction: 8 - 5 = 3
# ══════════════════════════════════════════════════════════════════════════════
section("6. Subtraction: negative displacement")

for a, b in [(8, 5), (10, 3), (100, 37)]:
    grid_sub = make_grid()
    displ_sub = make_displ()

    grid_sub.set_position(float(a))
    displ_sub.apply_displacement_to(float(-b), grid_sub)

    grid_direct = make_grid()
    grid_direct.set_position(float(a - b))

    test(f"{a} - {b} = {a-b} via negative displacement",
         np.array_equal(grid_sub.get_location_sdr(),
                        grid_direct.get_location_sdr()))


# ══════════════════════════════════════════════════════════════════════════════
# 7. Commutativity: d1 then d2 = d2 then d1
# ══════════════════════════════════════════════════════════════════════════════
section("7. Commutativity of sequential displacements")

d1_val, d2_val = 7.0, 13.0
start = 20.0

grid_ab = make_grid()
displ_ab = make_displ()
grid_ab.set_position(start)
displ_ab.apply_displacement_to(d1_val, grid_ab)
displ_ab.apply_displacement_to(d2_val, grid_ab)

grid_ba = make_grid()
displ_ba = make_displ()
grid_ba.set_position(start)
displ_ba.apply_displacement_to(d2_val, grid_ba)
displ_ba.apply_displacement_to(d1_val, grid_ba)

test("d1 then d2 = d2 then d1",
     np.array_equal(grid_ab.get_location_sdr(),
                    grid_ba.get_location_sdr()))


# ══════════════════════════════════════════════════════════════════════════════
# 8. make_displacement_layer_from_grid
# ══════════════════════════════════════════════════════════════════════════════
section("8. make_displacement_layer_from_grid")

grid_ref = make_grid()
displ_from_grid = make_displacement_layer_from_grid(grid_ref)

test("periods match", [m.period for m in displ_from_grid.modules] ==
     [m.period for m in grid_ref.modules])
test("sdr_length matches",
     displ_from_grid.sdr_length_per_module == grid_ref.sdr_length_per_module)
test("sdr_width matches",
     displ_from_grid.sdr_width_per_module == grid_ref.sdr_width_per_module)

# Should be able to apply to the grid without error
grid_ref.set_position(10.0)
displ_from_grid.apply_displacement_to(5.0, grid_ref)
grid_check = make_grid()
grid_check.set_position(15.0)
test("apply via make_displacement_layer_from_grid works",
     np.array_equal(grid_ref.get_location_sdr(),
                    grid_check.get_location_sdr()))


# ══════════════════════════════════════════════════════════════════════════════
# 9. Period mismatch raises error
# ══════════════════════════════════════════════════════════════════════════════
section("9. Period mismatch detection")

wrong_grid = GridCellLayer(
    periods=[7.0, 11.0, 17.0],  # different last period
    sdr_length_per_module=SDR_LEN,
    sdr_width_per_module=SDR_W,
)
displ_check = make_displ()
displ_check.set_displacement(5.0)

caught = False
try:
    displ_check.apply_to(wrong_grid)
except ValueError:
    caught = True
test("period mismatch raises ValueError", caught)

# Module count mismatch
wrong_count_grid = GridCellLayer(
    periods=[7.0, 11.0],
    sdr_length_per_module=SDR_LEN,
    sdr_width_per_module=SDR_W,
)
caught2 = False
try:
    displ_check.apply_to(wrong_count_grid)
except ValueError:
    caught2 = True
test("module count mismatch raises ValueError", caught2)


# ══════════════════════════════════════════════════════════════════════════════
# 10. Full round-trip: position → +d → -d → back to start
# ══════════════════════════════════════════════════════════════════════════════
section("10. Round-trip: +d then -d returns to start")

for start_pos, d_val in [(42, 15), (100, 37), (0, 77)]:
    grid_rt = make_grid()
    displ_rt = make_displ()

    grid_rt.set_position(float(start_pos))
    sdr_start = grid_rt.get_location_sdr().copy()

    displ_rt.apply_displacement_to(float(d_val), grid_rt)
    displ_rt.apply_displacement_to(float(-d_val), grid_rt)

    test(f"pos {start_pos}: +{d_val} then -{d_val} returns to start",
         np.array_equal(grid_rt.get_location_sdr(), sdr_start))


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 60}")
print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'═' * 60}")

if failed > 0:
    sys.exit(1)
