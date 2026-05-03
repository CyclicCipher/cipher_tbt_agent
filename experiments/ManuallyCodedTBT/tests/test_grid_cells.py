"""
test_grid_cells.py — Tests for the Grid Cell Layer.

Run from project root:
    python experiments/ManuallyCodedTBT/tests/test_grid_cells.py

Tests cover:
    1. Basic module properties
    2. Phase encoding: adjacent positions have overlapping SDRs
    3. Phase encoding: distant positions have low SDR overlap
    4. Periodic wrap-around: position 0 ≈ position period
    5. Path integration: set_position + integrate = correct position
    6. Addition is path integration: pos 3 + disp 5 = pos 8
    7. Subtraction via negative displacement
    8. Commutativity: d1 + d2 = d2 + d1
    9. Multi-module: coprime periods give unique representations
    10. Full layer SDR: distinct positions give distinct SDRs
    11. CRT reconstruction
    12. make_number_line_layer convenience constructor
    13. Large displacements wrap correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from sdr import overlap, population
from grid_cells import GridCellLayer, GridCellModule, make_number_line_layer


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

# Coprime periods: product = 7×11×13 = 1001, covers integers 0..1000
PERIODS = [7.0, 11.0, 13.0]
SDR_LEN = 64
SDR_W = 13  # > 64/7 ≈ 9.1, so adjacent positions in period-7 module overlap


def make_layer():
    return GridCellLayer(
        periods=PERIODS,
        sdr_length_per_module=SDR_LEN,
        sdr_width_per_module=SDR_W,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. Basic module properties
# ══════════════════════════════════════════════════════════════════════════════
section("1. Basic module properties")

mod = GridCellModule(period=7.0, sdr_length=64, sdr_width=13)
mod.set_phase(3.0)

sdr = mod.get_sdr()
test("module SDR length", len(sdr) == 64)
test("module SDR dtype bool", sdr.dtype == bool)
test("module SDR population", population(sdr) == 13)
test("phase set correctly", abs(mod.phase - 3.0) < 1e-9)

layer = make_layer()
test("layer num_modules", layer.num_modules == 3)
test("layer total SDR length", layer.total_sdr_length == 3 * SDR_LEN)
test("layer unique_range", layer.unique_range == 7 * 11 * 13,
     f"got {layer.unique_range}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Adjacent positions have overlapping SDRs (within a module)
# ══════════════════════════════════════════════════════════════════════════════
section("2. Adjacent positions → overlapping SDRs")

mod7 = GridCellModule(period=7.0, sdr_length=64, sdr_width=13)
mod7.set_phase(3.0)
sdr_3 = mod7.get_sdr()
mod7.set_phase(4.0)
sdr_4 = mod7.get_sdr()
mod7.set_phase(6.0)
sdr_6 = mod7.get_sdr()

ov_adj = overlap(sdr_3, sdr_4)
ov_far = overlap(sdr_3, sdr_6)

test("adjacent phases overlap", ov_adj > 0,
     f"overlap(3,4) = {ov_adj}")
test("adjacent > far overlap", ov_adj > ov_far,
     f"adj={ov_adj}, far={ov_far}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Distant positions have low SDR overlap (within a module)
# ══════════════════════════════════════════════════════════════════════════════
section("3. Distant positions → low SDR overlap")

mod7.set_phase(0.0)
sdr_0 = mod7.get_sdr()
mod7.set_phase(3.0)
sdr_3 = mod7.get_sdr()

ov_distant = overlap(sdr_0, sdr_3)
test("positions 3 apart have low overlap in period-7 module",
     ov_distant < SDR_W // 2,
     f"overlap = {ov_distant}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Periodic wrap-around: position ≡ position + period
# ══════════════════════════════════════════════════════════════════════════════
section("4. Periodic wrap-around")

mod7.set_phase(0.0)
sdr_at_0 = mod7.get_sdr()
mod7.set_phase(7.0)  # same as 0 mod 7
sdr_at_7 = mod7.get_sdr()
mod7.set_phase(14.0)  # same as 0 mod 7
sdr_at_14 = mod7.get_sdr()

test("phase(0) == phase(7) in period-7 module",
     np.array_equal(sdr_at_0, sdr_at_7))
test("phase(0) == phase(14) in period-7 module",
     np.array_equal(sdr_at_0, sdr_at_14))

# Same for the full layer
layer = make_layer()
layer.set_position(3.0)
sdr_pos3 = layer.get_location_sdr()
layer.set_position(3.0 + 7 * 11 * 13)  # + full period of the system
sdr_pos3_wrap = layer.get_location_sdr()
test("position + full_period gives same layer SDR",
     np.array_equal(sdr_pos3, sdr_pos3_wrap))


# ══════════════════════════════════════════════════════════════════════════════
# 5. Path integration: set_position + integrate = correct position
# ══════════════════════════════════════════════════════════════════════════════
section("5. Path integration correctness")

layer = make_layer()
layer.set_position(10.0)
layer.integrate(5.0)
phases = layer.get_phases()

expected_phases = np.array([
    15.0 % 7.0,   # 15 mod 7 = 1
    15.0 % 11.0,  # 15 mod 11 = 4
    15.0 % 13.0,  # 15 mod 13 = 2
])
test("path integration phases correct",
     np.allclose(phases, expected_phases, atol=1e-9),
     f"got {phases}, expected {expected_phases}")

# SDR should match set_position(15)
layer2 = make_layer()
layer2.set_position(15.0)
sdr_direct = layer2.get_location_sdr()
sdr_integrated = layer.get_location_sdr()
test("set_position(10) + integrate(5) == set_position(15)",
     np.array_equal(sdr_direct, sdr_integrated))


# ══════════════════════════════════════════════════════════════════════════════
# 6. Addition is path integration: 3 + 5 = 8
# ══════════════════════════════════════════════════════════════════════════════
section("6. Addition is path integration")

for a, b in [(3, 5), (0, 8), (7, 3), (13, 13), (50, 50)]:
    layer_sum = make_layer()
    layer_sum.set_position(float(a))
    layer_sum.integrate(float(b))
    sdr_via_integration = layer_sum.get_location_sdr()

    layer_direct = make_layer()
    layer_direct.set_position(float(a + b))
    sdr_direct = layer_direct.get_location_sdr()

    test(f"{a} + {b} = {a+b} via path integration",
         np.array_equal(sdr_via_integration, sdr_direct))


# ══════════════════════════════════════════════════════════════════════════════
# 7. Subtraction via negative displacement
# ══════════════════════════════════════════════════════════════════════════════
section("7. Subtraction via negative displacement")

for a, b in [(8, 5), (10, 3), (100, 37)]:
    layer_sub = make_layer()
    layer_sub.set_position(float(a))
    layer_sub.integrate(float(-b))
    sdr_via_subtraction = layer_sub.get_location_sdr()

    layer_direct = make_layer()
    layer_direct.set_position(float(a - b))
    sdr_direct = layer_direct.get_location_sdr()

    test(f"{a} - {b} = {a-b} via negative displacement",
         np.array_equal(sdr_via_subtraction, sdr_direct))


# ══════════════════════════════════════════════════════════════════════════════
# 8. Commutativity: d1 + d2 = d2 + d1
# ══════════════════════════════════════════════════════════════════════════════
section("8. Displacement commutativity")

start = 20.0
d1, d2 = 7.0, 13.0

layer_ab = make_layer()
layer_ab.set_position(start)
layer_ab.integrate(d1)
layer_ab.integrate(d2)

layer_ba = make_layer()
layer_ba.set_position(start)
layer_ba.integrate(d2)
layer_ba.integrate(d1)

test("integrate(d1) then integrate(d2) == integrate(d2) then integrate(d1)",
     np.array_equal(layer_ab.get_location_sdr(), layer_ba.get_location_sdr()))


# ══════════════════════════════════════════════════════════════════════════════
# 9. Multi-module: coprime periods give unique representations
# ══════════════════════════════════════════════════════════════════════════════
section("9. Unique representations across range")

layer = make_layer()
sdrs = {}
collisions = 0
for pos in range(int(layer.unique_range)):
    layer.set_position(float(pos))
    key = tuple(layer.get_location_sdr().tolist())
    if key in sdrs:
        collisions += 1
    sdrs[key] = pos

test("no collisions in unique range",
     collisions == 0,
     f"{collisions} collisions out of {int(layer.unique_range)} positions")
test("number of unique SDRs = unique_range",
     len(sdrs) == int(layer.unique_range),
     f"unique={len(sdrs)}, range={int(layer.unique_range)}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. Full layer SDR: nearby positions have higher overlap than distant ones
# ══════════════════════════════════════════════════════════════════════════════
section("10. Layer SDR similarity structure")

layer = make_layer()
layer.set_position(50.0)
sdr_50 = layer.get_location_sdr()
layer.set_position(51.0)
sdr_51 = layer.get_location_sdr()
layer.set_position(100.0)
sdr_100 = layer.get_location_sdr()

ov_near = overlap(sdr_50, sdr_51)
ov_far = overlap(sdr_50, sdr_100)

test("adjacent positions have higher layer SDR overlap than distant",
     ov_near > ov_far,
     f"50↔51={ov_near}, 50↔100={ov_far}")

test("non-zero overlap between adjacent positions",
     ov_near > 0,
     f"overlap(50,51) = {ov_near}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. CRT reconstruction
# ══════════════════════════════════════════════════════════════════════════════
section("11. CRT position reconstruction")

layer = make_layer()
for target in [0, 1, 42, 100, 500, 999, 1000]:
    layer.set_position(float(target))
    reconstructed = layer.estimated_position()
    test(f"reconstruct position {target}",
         abs(reconstructed - target) < 0.5,
         f"got {reconstructed:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. make_number_line_layer convenience constructor
# ══════════════════════════════════════════════════════════════════════════════
section("12. make_number_line_layer")

nl = make_number_line_layer(max_value=100, num_modules=3)
test("number line layer created", nl is not None)
test("covers range ≥ 100", nl.unique_range >= 100,
     f"unique_range = {nl.unique_range}")

nl.set_position(37.0)
nl.integrate(8.0)
nl2 = make_number_line_layer(max_value=100, num_modules=3)
nl2.set_position(45.0)
test("number line: 37 + 8 = 45",
     np.array_equal(nl.get_location_sdr(), nl2.get_location_sdr()))


# ══════════════════════════════════════════════════════════════════════════════
# 13. Large displacements wrap correctly
# ══════════════════════════════════════════════════════════════════════════════
section("13. Large displacements wrap correctly")

layer = make_layer()
# After integrating the full unique range, we should be back at the start
layer.set_position(42.0)
sdr_start = layer.get_location_sdr().copy()
layer.integrate(float(int(layer.unique_range)))  # + full period
sdr_after_wrap = layer.get_location_sdr()
test("integrating full period returns to same position",
     np.array_equal(sdr_start, sdr_after_wrap))

# Very large negative displacement
layer.set_position(5.0)
sdr_5 = layer.get_location_sdr().copy()
layer.integrate(-5.0)  # should be at 0
layer2 = make_layer()
layer2.set_position(0.0)
test("5 - 5 = 0 via negative displacement",
     np.array_equal(layer.get_location_sdr(), layer2.get_location_sdr()))


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 60}")
print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'═' * 60}")

if failed > 0:
    sys.exit(1)
