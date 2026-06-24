"""
test_ablation_reference_frame.py — Is the system using the reference frame,
or rote-memorizing multiplication answers?

The concern: the TM has two simultaneous prediction pathways:
  (A) Cell→cell synapses: encode the full sequence [a, ×, b, =] as context,
      predict feature(a×b) purely from prior sequence pattern. This is
      rote memorization — the grid position is irrelevant.
  (B) Location synapses: encode position-based predictions. When the grid
      is at position p, predict feature(p). This is reference frame use —
      the prediction depends on WHERE the grid is, not WHAT sequence led there.

Both were trained simultaneously (external displacement was injected during
training, strengthening both pathways at once). The test results alone cannot
distinguish which pathway is doing the work.

Ablation design: three displacement conditions, tested on the same trained column.

  CONDITION 1 — Normal (L5a fires correctly):
    displacement = L5a output ≈ a×b
    Grid moves to ≈ position a×b.
    Anomaly at feature(a×b) should be LOW.

  CONDITION 2 — Zero displacement (reference frame ablated):
    displacement = 0.0  (grid stays at position 0)
    If rote memorization dominates: TM still predicts feature(a×b) from
      cell→cell synapses → anomaly STAYS LOW.
    If reference frame is essential: location SDR points to position 0,
      not a×b → anomaly RISES to ≈1.0.

  CONDITION 3 — Wrong displacement (grid moved to wrong position):
    displacement = wrong_pos  (some other position, not a×b)
    The TM's location synapses now point to feature(wrong_pos).
    If reference frame drives predictions:
      anomaly at feature(a×b)    = HIGH  (correct answer, wrong position)
      anomaly at feature(wrong_pos) = LOW  (wrong answer, but what grid predicts)
    This is the smoking gun: the system predicts the WRONG answer with
    confidence when its reference frame points to the wrong place.

If Condition 2 shows anomaly rising AND Condition 3 shows the wrong-position
feature predicted with low anomaly, the reference frame is the mechanism.

Run from project root:
    python experiments/ManuallyCodedTBT/tests/test_ablation_reference_frame.py
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


# ── Configuration (identical to test_multiplication.py) ───────────────────────

MAX_OPERAND  = 4
MAX_PRODUCT  = 16
GRID_PERIODS = [17.0, 19.0, 23.0]
N_COLS       = 128
ACTIVE_K     = 10
CELLS        = 8
INPUT_NUM    = 256
INPUT_SYM    = 128
INPUT_SIZE   = 384
ENCODER_W    = 25
GRID_LEN     = 64
GRID_W       = 11

num_enc = make_number_encoder(max_value=MAX_PRODUCT, n=INPUT_NUM, w=ENCODER_W)
sym_enc = SymbolEncoder(symbols=['×', '='], n=INPUT_SYM, w=11, seed=1)


def encode_number(n: int) -> np.ndarray:
    return concatenate([num_enc.encode(float(n)),
                        np.zeros(INPUT_SYM, dtype=bool)])


def encode_symbol(s: str) -> np.ndarray:
    return concatenate([np.zeros(INPUT_NUM, dtype=bool),
                        sym_enc.encode(s)])


# ── Build and train column ────────────────────────────────────────────────────

section("Training (identical to test_multiplication.py)")

grid  = GridCellLayer(periods=GRID_PERIODS, sdr_length_per_module=GRID_LEN,
                      sdr_width_per_module=GRID_W)
displ = make_displacement_layer_from_grid(grid)
col   = CorticalColumn(
    grid_layer=grid, displacement_layer=displ, input_size=INPUT_SIZE,
    num_minicolumns=N_COLS, cells_per_col=CELLS, active_per_step=ACTIVE_K,
    sp_kwargs=dict(permanence_threshold=0.2, permanence_inc=0.03,
                   permanence_dec=0.015, boost_strength=10.0),
    tm_kwargs=dict(segment_activation_threshold=4, min_threshold=2,
                   permanence_threshold=0.5, permanence_inc=0.1,
                   permanence_dec=0.1, initial_permanence=0.21,
                   max_new_synapses_per_seg=8, max_new_loc_synapses_per_seg=12,
                   max_loc_synapses_per_seg=12, max_segs_per_cell=24,
                   location_activation_threshold=8),
    seed=7,
)
l5a = L5aReadout.from_displacement_layer(
    displ, total_l3_cells=col.tm.total_cells,
    sp_permanence_threshold=0.3, sp_permanence_inc=0.08,
    sp_permanence_dec=0.01, seed=7,
)
col.l5a = l5a
rng = np.random.default_rng(0)

# SP pre-training
all_inputs = ([encode_number(n) for n in range(MAX_PRODUCT + 1)] +
              [encode_symbol('×'), encode_symbol('=')])
print(f"  SP pre-training...")
for _ in range(80):
    for idx in rng.permutation(len(all_inputs)):
        col.sp.compute(all_inputs[idx], learn=True)

# Walk training
walk = list(range(MAX_PRODUCT + 1))
print(f"  Walk training (positions 0..{MAX_PRODUCT}, 500 epochs)...")
for _ in range(500):
    col.reset()
    col.reset_position(0.0)
    col.prime_from_location()
    for j, pos in enumerate(walk):
        col.compute(encode_number(pos),
                    displacement=1.0 if j + 1 < len(walk) else None,
                    learn=True)

# L5a + TM training on multiplication sequences
all_seqs = [(a, b, a*b) for a in range(1, MAX_OPERAND+1)
                         for b in range(1, MAX_OPERAND+1)]
print(f"  Multiplication training ({len(all_seqs)} sequences, 300 epochs)...")
for _ in range(300):
    for idx in rng.permutation(len(all_seqs)):
        a, b, product = all_seqs[idx]
        col.reset()
        col.reset_position(0.0)
        col.compute(encode_number(a),    displacement=None,            learn=True)
        l5a.learn_supervised(col.tm.cell_active, 0.0)
        col.compute(encode_symbol('×'),  displacement=None,            learn=True)
        l5a.learn_supervised(col.tm.cell_active, 0.0)
        col.compute(encode_number(b),    displacement=float(product),  learn=True)
        l5a.learn_supervised(col.tm.cell_active, float(product))
        col.compute(encode_symbol('='),  displacement=None,            learn=True)
        l5a.learn_supervised(col.tm.cell_active, 0.0)
        col.compute(encode_number(product), displacement=None,         learn=True)
        l5a.learn_supervised(col.tm.cell_active, 0.0)

print("  Training complete.")

# ── Test cases ────────────────────────────────────────────────────────────────

# Use pairs we know work in the normal condition
test_cases = [(2, 3, 6), (2, 4, 8), (3, 3, 9), (4, 4, 16), (1, 3, 3)]


def run_sequence(a, b, product, displacement_override, observe_result):
    """Run [a, ×, b, =, observe_result] with displacement_override at step b.

    displacement_override=None  → L5a fires normally (reference frame intact)
    displacement_override=0.0   → grid stays at position 0 (reference frame ablated)
    displacement_override=float → grid moves to arbitrary position
    """
    col.reset()
    col.reset_position(0.0)
    col.compute(encode_number(a),       displacement=None,              learn=False)
    col.compute(encode_symbol('×'),     displacement=None,              learn=False)
    col.compute(encode_number(b),       displacement=displacement_override, learn=False)
    col.compute(encode_symbol('='),     displacement=None,              learn=False)
    return col.compute(encode_number(observe_result), learn=False)['anomaly_score']


# ══════════════════════════════════════════════════════════════════════════════
# CONDITION 1 — Normal: L5a fires, grid moves to a×b
# ══════════════════════════════════════════════════════════════════════════════
section("CONDITION 1 — Normal (L5a displacement active)")
print("""
  L5a fires normally. Grid should move to position a×b.
  Anomaly at feature(a×b) should be LOW.
""")

cond1_anoms = []
for a, b, product in test_cases:
    anom = run_sequence(a, b, product, None, product)
    cond1_anoms.append(anom)
    print(f"  {a}×{b}={product:>2}: anomaly at feature({product}) = {anom:.3f}")

mean_cond1 = np.mean(cond1_anoms)
print(f"\n  Mean anomaly (correct answer) = {mean_cond1:.3f}")
test("Condition 1: correct answer has low anomaly (<0.5)", mean_cond1 < 0.5,
     f"mean={mean_cond1:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# CONDITION 2 — Zero displacement: reference frame ablated
# ══════════════════════════════════════════════════════════════════════════════
section("CONDITION 2 — Zero displacement (reference frame ablated)")
print("""
  displacement=0.0 at step b. Grid stays at position 0.
  Location SDR at the '=' step points to position 0, not a×b.

  If ROTE MEMORIZATION: cell→cell synapses predict feature(a×b) anyway.
                        Anomaly stays LOW.
  If REFERENCE FRAME:   location SDR at position 0 doesn't predict feature(a×b).
                        Anomaly RISES.
""")

cond2_anoms = []
for a, b, product in test_cases:
    anom = run_sequence(a, b, product, 0.0, product)
    cond2_anoms.append(anom)
    print(f"  {a}×{b}={product:>2}: anomaly at feature({product}) = {anom:.3f}  "
          f"{'← HIGH (ref frame)' if anom > 0.7 else '← low (rote mem)'}")

mean_cond2 = np.mean(cond2_anoms)
print(f"\n  Mean anomaly (correct answer, zero displacement) = {mean_cond2:.3f}")
print(f"  vs Condition 1 mean = {mean_cond1:.3f}")
print(f"  Rise in anomaly = {mean_cond2 - mean_cond1:.3f}")

test("Condition 2: anomaly RISES when displacement is zeroed",
     mean_cond2 > mean_cond1 + 0.2,
     f"cond1={mean_cond1:.3f}, cond2={mean_cond2:.3f}, rise={mean_cond2-mean_cond1:.3f}")
test("Condition 2: anomaly > 0.5 (reference frame is needed)",
     mean_cond2 > 0.5,
     f"mean={mean_cond2:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# CONDITION 3 — Wrong displacement: grid moved to wrong position
# ══════════════════════════════════════════════════════════════════════════════
section("CONDITION 3 — Wrong displacement (grid moved to wrong position)")
print("""
  displacement = wrong_pos (a position ≠ a×b, but one that was walk-trained).
  The grid moves to wrong_pos. Location SDR points to feature(wrong_pos).

  If REFERENCE FRAME drives predictions:
    anomaly at feature(a×b)       = HIGH  (correct answer, wrong grid pos)
    anomaly at feature(wrong_pos) = LOW   (WRONG answer, but grid predicts it)

  This is the definitive test: if the system predicts the WRONG answer
  with confidence when the grid points to the wrong place, the reference
  frame is the mechanism, not the sequence pattern.

  If ROTE MEMORIZATION:
    anomaly at feature(a×b)       = LOW   (cell→cell synapses predict it)
    anomaly at feature(wrong_pos) = HIGH  (wrong answer, not in training)
""")

print(f"  {'a×b':>5}  {'wrong':>6}  {'anom@correct':>14}  {'anom@wrong_pos':>15}  verdict")
print(f"  {'─'*5}  {'─'*6}  {'─'*14}  {'─'*15}  ───────")

anom_at_correct_list = []
anom_at_wrong_list   = []

for a, b, product in test_cases:
    # Pick a wrong position that was in the walk (so it has trained location synapses)
    # and is far enough from product to be clearly distinct
    wrong_pos = ((product + 5) % (MAX_PRODUCT + 1))
    # Make sure wrong_pos is actually different
    if wrong_pos == product:
        wrong_pos = (wrong_pos + 3) % (MAX_PRODUCT + 1)
    wrong_feature = wrong_pos  # on the number line, feature at pos p is p

    anom_correct  = run_sequence(a, b, product,      float(wrong_pos), product)
    anom_wrong    = run_sequence(a, b, wrong_feature, float(wrong_pos), wrong_feature)

    anom_at_correct_list.append(anom_correct)
    anom_at_wrong_list.append(anom_wrong)

    verdict = ("REF FRAME ✓" if anom_correct > 0.5 and anom_wrong < 0.5
               else "ROTE MEM ✗"  if anom_correct < 0.5 and anom_wrong > 0.5
               else "MIXED")
    print(f"  {a}×{b}={product:>2}  →{wrong_pos:>5}  "
          f"{anom_correct:>14.3f}  {anom_wrong:>15.3f}  {verdict}")

mean_correct_cond3 = np.mean(anom_at_correct_list)
mean_wrong_cond3   = np.mean(anom_at_wrong_list)
print(f"\n  Mean anomaly at correct answer (grid→wrong pos): {mean_correct_cond3:.3f}")
print(f"  Mean anomaly at wrong_pos feature (grid→wrong pos): {mean_wrong_cond3:.3f}")

test("Condition 3: correct answer has HIGH anomaly when grid is at wrong pos",
     mean_correct_cond3 > 0.5,
     f"mean={mean_correct_cond3:.3f}")
test("Condition 3: wrong-position feature has LOWER anomaly than correct answer",
     mean_wrong_cond3 < mean_correct_cond3,
     f"correct={mean_correct_cond3:.3f}, wrong_pos={mean_wrong_cond3:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary and interpretation
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'═'*60}")

print(f"""
  Summary of anomaly scores:

    Condition                       Mean anomaly at 'correct' answer
    ─────────────────────────────   ────────────────────────────────
    1. Normal (L5a fires)           {mean_cond1:.3f}  ← low = system works
    2. Zero displacement            {mean_cond2:.3f}  ← {'HIGH = reference frame needed' if mean_cond2 > 0.5 else 'low = rote memorization'}
    3. Wrong displacement           {mean_correct_cond3:.3f}  ← {'HIGH = follows grid, not sequence' if mean_correct_cond3 > 0.5 else 'low = rote memorization'}

    Condition 3 anomaly at wrong-pos feature: {mean_wrong_cond3:.3f}
      ← {'LOWER than correct = system predicts where grid IS, not a×b' if mean_wrong_cond3 < mean_correct_cond3 else 'higher = not using reference frame'}

  Interpretation:""")

ref_frame_evidence = (mean_cond2 > mean_cond1 + 0.2 and
                      mean_cond2 > 0.5 and
                      mean_correct_cond3 > 0.5 and
                      mean_wrong_cond3 < mean_correct_cond3)

if ref_frame_evidence:
    print("""
  The reference frame is the primary prediction mechanism.

  Evidence:
    • Zeroing displacement (Cond 2) raises anomaly substantially — the
      correct answer is NOT predicted by cell→cell synapses alone. The
      grid's location SDR is required.
    • Moving the grid to a wrong position (Cond 3) causes the system to
      predict the WRONG answer with lower anomaly than the correct answer.
      The prediction follows the grid, not the learned sequence pattern.

  This confirms the operation is structurally equivalent to multiplication
  via the reference frame: L5a maps sequence context → displacement,
  the grid moves to position a×b, and the TM reads feature(a×b) from
  location synapses — not from the multiplication table memorized as
  a sequence pattern.""")
else:
    print("""
  The evidence for reference frame use is mixed or weak.
  Rote memorization may be a significant contributor.
  Interpretation requires examining per-pair results above.""")

if failed > 0:
    sys.exit(1)
