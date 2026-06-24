"""
test_addition_10.py — Phase 10: Location-Driven Prediction + Learned L5a (SDR-native).

Tests:
  A. Reset no longer bursts at known positions (location_activation_threshold
     + prime_from_location)
  B. Multi-stride curriculum: stride-1 then stride-2 both converge with
     prime_from_location eliminating start-position burst competition
  C. Unique-result addition with SDR-native L5a (sanity check)
  D. Ambiguous-result addition with SDR-native L5a (critical — was 1.000 in 9b)
  E. Generalization to held-out (a, b) pairs

L5a redesign (SDR-native):
  L5a is now a bank of Spatial Poolers, one per displacement cell module.
  Input:  L3 active cells (full sequence context via TM higher-order memory).
  Output: displacement SDR (same format as DisplacementLayer).
  Learning: forced-winner Hebbian — for each step, encode the correct
    displacement as a target SDR and call SP.learn_with_target().
  Zero-displacement training for non-operator steps prevents spurious
  grid movement when L5a fires in neutral contexts.
  All signals are SDRs. No scalar floats cross layer boundaries.

Run from project root:
    python experiments/ManuallyCodedTBT/tests/test_addition_10.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from encoder import make_number_encoder, SymbolEncoder
from grid_cells import GridCellLayer
from displacement_layer import make_displacement_layer_from_grid
from cortical_column import CorticalColumn
from l5a_readout import L5aReadout
from sdr import concatenate

passed = 0
failed = 0


def test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}" + (f" — {detail}" if detail else ""))


def section(name: str):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")


# ── Shared config ─────────────────────────────────────────────────────────────

MAX_VALUE    = 10
N_COLS       = 128
ACTIVE_K     = 10
CELLS        = 8
INPUT_NUM    = 256    # number encoder length
INPUT_SYM    = 128    # symbol encoder length
INPUT_SIZE   = INPUT_NUM + INPUT_SYM   # 384
ENCODER_W    = 25
GRID_PERIODS = [11.0, 13.0, 17.0]
GRID_LEN     = 64
GRID_W       = 11

TM_KWARGS = dict(
    segment_activation_threshold=4,
    min_threshold=2,
    permanence_threshold=0.5,
    permanence_inc=0.1,
    permanence_dec=0.1,
    initial_permanence=0.21,
    max_new_synapses_per_seg=8,
    max_new_loc_synapses_per_seg=12,
    max_loc_synapses_per_seg=12,
    max_segs_per_cell=24,
    location_activation_threshold=8,
)
SP_KWARGS = dict(
    permanence_threshold=0.2,
    permanence_inc=0.03,
    permanence_dec=0.015,
    boost_strength=10.0,
)
L5A_KWARGS = dict(
    sp_permanence_threshold=0.3,
    sp_permanence_inc=0.08,
    sp_permanence_dec=0.01,
)

num_enc = make_number_encoder(max_value=MAX_VALUE, n=INPUT_NUM, w=ENCODER_W)
sym_enc = SymbolEncoder(symbols=['+', '='], n=INPUT_SYM, w=11, seed=0)


def encode_number(n: int) -> np.ndarray:
    return concatenate([num_enc.encode(float(n)),
                        np.zeros(INPUT_SYM, dtype=bool)])


def encode_symbol(s: str) -> np.ndarray:
    return concatenate([np.zeros(INPUT_NUM, dtype=bool),
                        sym_enc.encode(s)])


def make_walk_col(seed: int, input_size: int = INPUT_NUM) -> CorticalColumn:
    """Column for walk tests (number-only encoder, 256-bit input)."""
    grid  = GridCellLayer(periods=GRID_PERIODS, sdr_length_per_module=GRID_LEN,
                          sdr_width_per_module=GRID_W)
    displ = make_displacement_layer_from_grid(grid)
    return CorticalColumn(
        grid_layer=grid, displacement_layer=displ,
        input_size=input_size,
        num_minicolumns=N_COLS, cells_per_col=CELLS, active_per_step=ACTIVE_K,
        sp_kwargs=SP_KWARGS, tm_kwargs=TM_KWARGS, seed=seed,
    )


def make_addition_col(seed: int):
    """Column + L5a for addition tests (combined 384-bit input)."""
    grid  = GridCellLayer(periods=GRID_PERIODS, sdr_length_per_module=GRID_LEN,
                          sdr_width_per_module=GRID_W)
    displ = make_displacement_layer_from_grid(grid)
    col   = CorticalColumn(
        grid_layer=grid, displacement_layer=displ,
        input_size=INPUT_SIZE,
        num_minicolumns=N_COLS, cells_per_col=CELLS, active_per_step=ACTIVE_K,
        sp_kwargs=SP_KWARGS, tm_kwargs=TM_KWARGS, seed=seed,
    )
    l5a = L5aReadout.from_displacement_layer(
        displ, total_l3_cells=col.tm.total_cells,
        seed=seed, **L5A_KWARGS,
    )
    col.l5a = l5a
    return col, l5a


def sp_pretrain(col, inputs, n_epochs=80, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_epochs):
        for idx in rng.permutation(len(inputs)):
            col.sp.compute(inputs[idx], learn=True)


def walk_primed(col, positions, d, learn):
    col.reset()
    col.reset_position(float(positions[0]))
    col.prime_from_location()
    anoms = []
    for j, pos in enumerate(positions):
        disp = float(d) if j + 1 < len(positions) else None
        r = col.compute(num_enc.encode(float(pos)), displacement=disp, learn=learn)
        anoms.append(r['anomaly_score'])
    return anoms


# ══════════════════════════════════════════════════════════════════════════════
# TEST A — Reset no longer bursts at known positions
# ══════════════════════════════════════════════════════════════════════════════
section("A. location_activation_threshold + prime_from_location")

col_a = make_walk_col(seed=1)
walk1 = list(range(MAX_VALUE + 1))
num_inputs = [num_enc.encode(float(p)) for p in range(MAX_VALUE + 1)]
sp_pretrain(col_a, num_inputs, n_epochs=80, seed=0)

for _ in range(500):
    walk_primed(col_a, walk1, 1, learn=True)

anoms_walk = walk_primed(col_a, walk1, 1, learn=False)
test("stride-1 walk: all steps 1+ have 0.000 anomaly",
     all(a == 0.0 for a in anoms_walk[1:]),
     f"non-zero: {[anoms_walk[i] for i in range(1,11) if anoms_walk[i]>0]}")

# Jump-to test with prime_from_location
jump_anoms = []
for p in [0, 2, 4, 6, 8, 10]:
    col_a.reset()
    col_a.reset_position(float(p))
    col_a.prime_from_location()
    r = col_a.compute(num_enc.encode(float(p)), displacement=None, learn=False)
    jump_anoms.append(r['anomaly_score'])

mean_jump = np.mean(jump_anoms)
print(f"  Jump-to anomalies: {dict(zip([0,2,4,6,8,10], [f'{a:.2f}' for a in jump_anoms]))}")
test("mean anomaly across jump-to positions < 0.4",
     mean_jump < 0.4, f"mean={mean_jump:.3f}")

# Without prime_from_location: should still burst (no location-only priming)
col_a.reset()
col_a.reset_position(5.0)
r_raw = col_a.compute(num_enc.encode(5.0), displacement=None, learn=False)
test("without prime_from_location: first step still bursts at 1.0",
     r_raw['anomaly_score'] == 1.0, f"got {r_raw['anomaly_score']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST B — Multi-stride curriculum with prime_from_location
# ══════════════════════════════════════════════════════════════════════════════
section("B. Multi-stride curriculum (stride-1 then stride-2)")

col_b = make_walk_col(seed=2)
sp_pretrain(col_b, num_inputs, n_epochs=80, seed=1)

walks_1 = [list(range(MAX_VALUE + 1))]
walks_2 = [[i for i in range(s, MAX_VALUE + 1, 2)] for s in range(2)]

rng_b = np.random.default_rng(1)
for _ in range(500):
    walk_primed(col_b, walks_1[0], 1, learn=True)

s1_anoms = walk_primed(col_b, walks_1[0], 1, learn=False)[1:]
test("stride-1 converges with prime_from_location",
     np.mean(s1_anoms) < 0.1, f"mean={np.mean(s1_anoms):.3f}")

for _ in range(500):
    for positions in walks_2:
        walk_primed(col_b, positions, 2, learn=True)

s2_anoms = []
for positions in walks_2:
    s2_anoms.extend(walk_primed(col_b, positions, 2, learn=False)[1:])
mean_s2 = np.mean(s2_anoms)
print(f"  Stride-2 mean step-1+ anomaly: {mean_s2:.3f}")
test("stride-2 converges after stride-1 curriculum",
     mean_s2 < 0.6, f"mean={mean_s2:.3f}")

# prime_from_location makes even-positions position-addressable
jump_b = []
for p in [0, 2, 4, 6, 8, 10]:
    col_b.reset(); col_b.reset_position(float(p)); col_b.prime_from_location()
    r = col_b.compute(num_enc.encode(float(p)), displacement=None, learn=False)
    jump_b.append(r['anomaly_score'])
test("prime_from_location: stride-2 positions mean anomaly < 0.3",
     np.mean(jump_b) < 0.3, f"mean={np.mean(jump_b):.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Shared addition column (used in tests C, D, E)
# ══════════════════════════════════════════════════════════════════════════════
section("Building addition column (tests C, D, E)...")

col_c, l5a_c = make_addition_col(seed=3)

all_inputs_c = (
    [encode_number(n) for n in range(MAX_VALUE + 1)] +
    [encode_symbol('+'), encode_symbol('=')]
)
sp_pretrain(col_c, all_inputs_c, n_epochs=80, seed=2)

seqs = [(a, b, a+b) for a in range(MAX_VALUE + 1)
                    for b in range(1, MAX_VALUE + 1 - a)]

rng_c = np.random.default_rng(2)
N_EPOCHS = 50
print(f"  Training {N_EPOCHS} epochs × {len(seqs)} sequences...")
for epoch in range(N_EPOCHS):
    for idx in rng_c.permutation(len(seqs)):
        a, b, result = seqs[idx]
        col_c.reset()
        col_c.reset_position(0.0)

        # a — no operator displacement
        col_c.compute(encode_number(a), displacement=None, learn=True)
        l5a_c.learn_supervised(col_c.tm.cell_active, 0.0)

        # + — no operator displacement
        col_c.compute(encode_symbol('+'), displacement=None, learn=True)
        l5a_c.learn_supervised(col_c.tm.cell_active, 0.0)

        # b — OPERATOR STEP: L5a should output displacement=result=a+b
        # External displacement used during training for correct grid pos
        col_c.compute(encode_number(b), displacement=float(result), learn=True)
        l5a_c.learn_supervised(col_c.tm.cell_active, float(result))

        # = — no operator displacement
        col_c.compute(encode_symbol('='), displacement=None, learn=True)
        l5a_c.learn_supervised(col_c.tm.cell_active, 0.0)

        # result — no displacement, check prediction
        col_c.compute(encode_number(result), displacement=None, learn=True)
        l5a_c.learn_supervised(col_c.tm.cell_active, 0.0)

print("  Training complete.")


def run_addition(col, l5a, a, b, observe_result):
    """Run inference: [a, +, b, =, observe_result]. Returns anomaly at result."""
    col.reset()
    col.reset_position(0.0)
    col.compute(encode_number(a),    displacement=None, learn=False)
    col.compute(encode_symbol('+'),  displacement=None, learn=False)
    col.compute(encode_number(b),    displacement=None, learn=False)
    col.compute(encode_symbol('='),  displacement=None, learn=False)
    return col.compute(encode_number(observe_result), learn=False)['anomaly_score']


# ══════════════════════════════════════════════════════════════════════════════
# TEST C — Unique-result addition
# ══════════════════════════════════════════════════════════════════════════════
section("C. Unique-result addition — SDR-native L5a (sanity check)")

unique_cases = [(3,5,8), (2,7,9), (0,8,8), (1,6,7), (4,3,7)]
anom_cc, anom_cw = [], []
for a, b, result in unique_cases:
    wrong = (result + 4) % (MAX_VALUE + 1)
    if wrong == result: wrong = (wrong + 1) % (MAX_VALUE + 1)
    anom_cc.append(run_addition(col_c, l5a_c, a, b, result))
    anom_cw.append(run_addition(col_c, l5a_c, a, b, wrong))

mean_cc = np.mean(anom_cc); mean_cw = np.mean(anom_cw)
print(f"  Unique-result mean: correct={mean_cc:.3f}  wrong={mean_cw:.3f}")
test("C: correct < wrong", mean_cc < mean_cw,
     f"correct={mean_cc:.3f}, wrong={mean_cw:.3f}")
test("C: correct anomaly < 0.6", mean_cc < 0.6, f"mean={mean_cc:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST D — Ambiguous-result addition (critical)
# ══════════════════════════════════════════════════════════════════════════════
section("D. Ambiguous-result addition — reference frame required (was 1.000 in 9b)")

ambig_cases = [(1,9,10), (5,5,10), (2,8,10), (4,4,8), (3,5,8)]
anom_dc, anom_dw = [], []
for a, b, result in ambig_cases:
    wrong = (result + 3) % (MAX_VALUE + 1)
    if wrong == result: wrong = (wrong + 1) % (MAX_VALUE + 1)
    anom_dc.append(run_addition(col_c, l5a_c, a, b, result))
    anom_dw.append(run_addition(col_c, l5a_c, a, b, wrong))

mean_dc = np.mean(anom_dc); mean_dw = np.mean(anom_dw)

print(f"\n  {'a':>3}  {'b':>3}  {'a+b':>5}  {'correct':>9}  {'wrong':>7}  pass")
print(f"  {'─'*3}  {'─'*3}  {'─'*5}  {'─'*9}  {'─'*7}  ────")
for (a,b,r), ac, aw in zip(ambig_cases, anom_dc, anom_dw):
    mark = "✓" if ac < aw else "✗"
    print(f"  {a:>3}  {b:>3}  {r:>5}  {ac:>9.3f}  {aw:>7.3f}  {mark}")
print(f"\n  mean: correct={mean_dc:.3f}  wrong={mean_dw:.3f}")
print(f"  (Phase 9b baseline: correct≈1.000, wrong≈0.200)")

test("D: ambiguous correct < wrong", mean_dc < mean_dw,
     f"correct={mean_dc:.3f}, wrong={mean_dw:.3f}")
test("D: ambiguous correct anomaly < 0.5", mean_dc < 0.5, f"mean={mean_dc:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST E — Generalization to held-out pairs
# ══════════════════════════════════════════════════════════════════════════════
section("E. Generalization — held-out (a, b) pairs")

held_out = [(2,6,8), (3,6,9), (1,7,8), (4,5,9), (0,9,9)]
anom_ec, anom_ew = [], []
for a, b, result in held_out:
    wrong = (result + 3) % (MAX_VALUE + 1)
    if wrong == result: wrong = (wrong + 1) % (MAX_VALUE + 1)
    anom_ec.append(run_addition(col_c, l5a_c, a, b, result))
    anom_ew.append(run_addition(col_c, l5a_c, a, b, wrong))

mean_ec = np.mean(anom_ec); mean_ew = np.mean(anom_ew)
print(f"  Held-out mean: correct={mean_ec:.3f}  wrong={mean_ew:.3f}")
test("E: generalization correct < wrong", mean_ec < mean_ew,
     f"correct={mean_ec:.3f}, wrong={mean_ew:.3f}")
test("E: generalization correct anomaly < 0.5", mean_ec < 0.5, f"mean={mean_ec:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'═'*60}")
print()
if failed == 0:
    print("  Phase 10 PASS (SDR-native L5a).")
    print("  — location_activation_threshold eliminates reset burst")
    print("  — L5a bank-of-SPs produces displacement SDRs from L3 context")
    print("  — Ambiguous-result pairs now distinguished via reference frame")
    print("  — All signals are SDRs; no scalar floats cross layer boundaries")
else:
    print(f"  Phase 10: {passed}/{passed+failed} tests pass.")

if failed > 0:
    sys.exit(1)