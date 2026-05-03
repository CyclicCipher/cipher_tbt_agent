"""
test_addition_9b.py — Phase 9b: Learned Addition (No Displacement Injection).

This test defines the NEXT IMPLEMENTATION MILESTONE. It is expected to
FAIL with the current architecture. The failure IS the specification.

What this tests:
  Whether a single cortical column can learn that the symbol "+" means
  "treat the next number as a displacement to apply to the reference frame."

  In Phase 9a, the displacement was injected externally — we manually
  called col.compute(number, displacement=d). This is architecturally
  equivalent to another column's L5b providing the displacement.

  In Phase 9b, NO displacement is injected. The column must itself learn:
    - Sequences: [3, +, 5] → the "+" symbol context means the next
      input (5) should update the grid cell reference frame, not just
      be encoded as a sensory feature
    - After learning "+", observe the result (8) at the new position

  What would be required for this to work:
    1. L5b displacement cells must learn: "when L3 context = 'after +',
       and L4 current input = number N, output displacement N to L6a."
    2. L5b is currently not implemented — displacement is external.
    3. Even with L5b implemented, the learning rule is unknown:
       what error signal tells L5b its displacement was correct?

Input encoding:
  Numbers 0-20 encoded as ScalarEncoder SDRs.
  Operators (+, =) encoded as CategoryEncoder SDRs.
  Both concatenated into a MultiEncoder for the column.

Training sequences:
  [a, +, b, =, a+b]
  The column sees: number a, then the + symbol, then number b,
  then the = symbol, then the result a+b.

Test:
  Present [a, +, b, =] and check whether the column's predictive
  cells match the minicolumns for feature(a+b). Compare anomaly
  at the result step for correct vs wrong answer.

Expected result: FAIL (anomaly is similar for correct and wrong answer
because the column has no mechanism to apply displacement from "+").

Documenting the failure state is the purpose of this test.

Run from project root:
    python experiments/ManuallyCodedTBT/tests/test_addition_9b.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from encoder import make_number_encoder, SymbolEncoder, MultiEncoder
from grid_cells import GridCellLayer
from displacement_layer import make_displacement_layer_from_grid
from cortical_column import CorticalColumn

passed = 0
failed = 0
expected_failures = 0


def test(name: str, condition: bool, detail: str = "", expected_to_fail: bool = False):
    global passed, failed, expected_failures
    if expected_to_fail:
        expected_failures += 1
        # For expected failures: pass = condition is False (correctly failing)
        if not condition:
            passed += 1
            print(f"  ✓ (expected fail) {name}")
        else:
            failed += 1
            print(f"  ! (unexpected pass!) {name}"
                  + (f"  — {detail}" if detail else ""))
    else:
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            failed += 1
            print(f"  ✗ {name}" + (f"  — {detail}" if detail else ""))


def section(name: str):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")


# ── Configuration ─────────────────────────────────────────────────────────────

MAX_VALUE    = 10
INPUT_SIZE   = 384   # 256 number + 128 operator
N_COLS       = 256
ACTIVE_K     = 15
CELLS        = 4
GRID_PERIODS = [11.0, 13.0, 17.0]
GRID_LEN     = 64
GRID_W       = 11
SP_PRETRAIN  = 80
TRAIN_EPOCHS = 40

# ── Encoders ──────────────────────────────────────────────────────────────────

num_enc = make_number_encoder(max_value=MAX_VALUE, n=256, w=21)
sym_enc = SymbolEncoder(symbols=['+', '=', '?'], n=128, w=11, seed=0)
enc = MultiEncoder([num_enc, sym_enc])
# Symbol encoding: numbers use sym 0 ('+'s bits zeroed? no, MultiEncoder
# concatenates. For number inputs use the dummy symbol index.
# For operator inputs the number part is zeroed.
# Simpler: use separate encoding for numbers vs operators.

def encode_number(n):
    """Encode a number with zero operator bits."""
    num_sdr = num_enc.encode(float(n))
    sym_sdr = np.zeros(128, dtype=bool)
    from sdr import concatenate
    return concatenate([num_sdr, sym_sdr])

def encode_symbol(sym):
    """Encode an operator with zero number bits."""
    num_sdr = np.zeros(256, dtype=bool)
    sym_sdr = sym_enc.encode(sym)
    from sdr import concatenate
    return concatenate([num_sdr, sym_sdr])


# ── Build column ──────────────────────────────────────────────────────────────

grid = GridCellLayer(
    periods=GRID_PERIODS,
    sdr_length_per_module=GRID_LEN,
    sdr_width_per_module=GRID_W,
)
displ = make_displacement_layer_from_grid(grid)
col = CorticalColumn(
    grid_layer=grid,
    displacement_layer=displ,
    input_size=INPUT_SIZE,
    num_minicolumns=N_COLS,
    cells_per_col=CELLS,
    active_per_step=ACTIVE_K,
    sp_kwargs=dict(
        permanence_threshold=0.2,
        permanence_inc=0.03,
        permanence_dec=0.015,
        boost_strength=10.0,
    ),
    tm_kwargs=dict(
        segment_activation_threshold=3,
        min_threshold=2,
        permanence_threshold=0.5,
        permanence_inc=0.1,
        permanence_dec=0.1,
        initial_permanence=0.21,
        max_new_synapses_per_seg=12,
        max_segs_per_cell=32,
    ),
    seed=42,
)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════
section("Training — [a, +, b, =, a+b] sequences")

rng = np.random.default_rng(0)

# SP pre-training on all input types
print(f"  SP pre-training: {SP_PRETRAIN} epochs...")
all_inputs = ([encode_number(n) for n in range(MAX_VALUE + 1)] +
              [encode_symbol('+'), encode_symbol('=')])
for epoch in range(SP_PRETRAIN):
    for idx in rng.permutation(len(all_inputs)):
        col.sp.compute(all_inputs[idx], learn=True)

# Build all training sequences [a, +, b, =, a+b] for a+b <= MAX_VALUE
sequences = []
for a in range(MAX_VALUE + 1):
    for b in range(1, MAX_VALUE + 1 - a):
        sequences.append((a, b, a + b))

print(f"  Training {len(sequences)} sequences × {TRAIN_EPOCHS} epochs...")
print(f"  NOTE: No displacement injected — column must learn '+' rule itself.")

for epoch in range(TRAIN_EPOCHS):
    for idx in rng.permutation(len(sequences)):
        a, b, result = sequences[idx]
        col.reset()
        col.reset_position(0.0)  # grid always starts at 0
        # Sequence: a, +, b, =, result — NO displacement injected anywhere
        col.compute(encode_number(a),    displacement=None, learn=True)
        col.compute(encode_symbol('+'),  displacement=None, learn=True)
        col.compute(encode_number(b),    displacement=None, learn=True)
        col.compute(encode_symbol('='),  displacement=None, learn=True)
        col.compute(encode_number(result), displacement=None, learn=True)

sp_ent = col.sp.get_entropy() / col.sp.get_max_entropy()
seg = col.tm.get_segment_stats()
print(f"  SP entropy ratio: {sp_ent:.3f}")
print(f"  TM: {seg['total_segments_created']} segments")


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════
section("Evaluation — rote memorization vs compositional addition")

# Test pairs
test_cases = [
    (3, 5,  8),
    (1, 9, 10),
    (4, 4,  8),
    (2, 7,  9),
    (5, 5, 10),
    (0, 8,  8),
]

print("  Computing anomaly at result step for all test cases...")
anomaly_correct = []
anomaly_wrong   = []
for a, b, result in test_cases:
    wrong = result + 3
    if wrong > MAX_VALUE:
        wrong = result - 3

    def run_seq_q(observe_result):
        col.reset()
        col.reset_position(0.0)
        col.compute(encode_number(a),   displacement=None, learn=False)
        col.compute(encode_symbol('+'), displacement=None, learn=False)
        col.compute(encode_number(b),   displacement=None, learn=False)
        col.compute(encode_symbol('='), displacement=None, learn=False)
        return col.compute(encode_number(observe_result), learn=False)['anomaly_score']

    anomaly_correct.append(run_seq_q(result))
    anomaly_wrong.append(run_seq_q(wrong))

mean_c = np.mean(anomaly_correct)
mean_w = np.mean(anomaly_wrong)
print(f"  Overall mean: correct={mean_c:.3f}  wrong={mean_w:.3f}")

# These tests distinguish rote memorization from compositional addition.
# The column CAN memorize specific (a,b)→result pairs via higher-order TM.
# The column CANNOT generalize when multiple (a,b) pairs share the same result,
# because without the reference frame it has no way to distinguish paths.

# Cases with UNIQUE results — TM can rote-memorize these
unique_result_cases = [(3, 5, 8), (2, 7, 9), (0, 8, 8)]
# Cases with AMBIGUOUS results — multiple (a,b) pairs share the result
# e.g. result=10: (1,9), (2,8), (3,7), (4,6), (5,5) all produce 10
ambiguous_result_cases = [(1, 9, 10), (5, 5, 10), (4, 4, 8)]

unique_correct = []
unique_wrong   = []
ambig_correct  = []
ambig_wrong    = []

all_eval = unique_result_cases + ambiguous_result_cases
for a, b, result in all_eval:
    wrong = result + 3
    if wrong > MAX_VALUE:
        wrong = result - 3

    def run_seq(observe_result):
        col.reset()
        col.reset_position(0.0)
        col.compute(encode_number(a),   displacement=None, learn=False)
        col.compute(encode_symbol('+'), displacement=None, learn=False)
        col.compute(encode_number(b),   displacement=None, learn=False)
        col.compute(encode_symbol('='), displacement=None, learn=False)
        return col.compute(encode_number(observe_result), learn=False)['anomaly_score']

    ac = run_seq(result)
    aw = run_seq(wrong)
    if (a, b, result) in unique_result_cases:
        unique_correct.append(ac); unique_wrong.append(aw)
    else:
        ambig_correct.append(ac); ambig_wrong.append(aw)

mean_uc = np.mean(unique_correct)
mean_uw = np.mean(unique_wrong)
mean_ac = np.mean(ambig_correct)
mean_aw = np.mean(ambig_wrong)

print(f"\n  UNIQUE-RESULT cases (TM can memorize):")
for (a,b,r), ac, aw in zip(unique_result_cases, unique_correct, unique_wrong):
    mark = "✓" if ac < aw else "✗"
    print(f"    {a}+{b}={r}: correct={ac:.3f} wrong={aw:.3f} {mark}")
print(f"  mean: correct={mean_uc:.3f}  wrong={mean_uw:.3f}")

print(f"\n  AMBIGUOUS-RESULT cases (needs reference frame to distinguish):")
for (a,b,r), ac, aw in zip(ambiguous_result_cases, ambig_correct, ambig_wrong):
    mark = "✓" if ac < aw else "✗"
    print(f"    {a}+{b}={r}: correct={ac:.3f} wrong={aw:.3f} {mark}")
print(f"  mean: correct={mean_ac:.3f}  wrong={mean_aw:.3f}")

# Unique-result cases: TM rote memorization may work — NOT the key test
test(
    "UNIQUE results: TM can rote-memorize some (a,b)→result pairs",
    mean_uc < mean_uw,
    f"correct={mean_uc:.3f}, wrong={mean_uw:.3f}",
    expected_to_fail=False,  # this may or may not pass — both are acceptable
)

# Ambiguous-result cases: this is where the reference frame is REQUIRED
# The column should FAIL here — expected failure = expected_to_fail=True
test(
    "AMBIGUOUS results: fails without reference frame (EXPECTED TO FAIL)",
    mean_ac < 0.5 and mean_ac < mean_aw,
    f"correct={mean_ac:.3f}, wrong={mean_aw:.3f}",
    expected_to_fail=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Diagnosis section
# ══════════════════════════════════════════════════════════════════════════════
section("Diagnosis: what the column IS and IS NOT learning")

print(f"""
  What the column DOES learn (from temporal memory):
    - Sequence pattern: a → '+' → b → '=' → a+b
    - The '=' symbol follows 'b' in the sequence
    - After '=', some number follows
    - The TM may partially learn that specific (a,b) pairs
      predict specific results (as rote memorization, not composition)

  What the column CANNOT learn without L5b:
    - That '+' means "apply the next number as a displacement"
    - That the grid cell reference frame should be at position a+b
      when the result is observed
    - Compositional generalization: predicting 3+5=8 from learning
      3+5 and 1+7=8 separately requires the reference frame

  Missing component: Layer 5b displacement cells.
    L5b must learn: given TM context "after +" and current L4 input
    (number b), output displacement b to L6a. This is the rule that
    converts operator context into reference frame updates.

  NEXT MILESTONE (Phase 10):
    Implement learned L5b. Design candidates:
      A. Supervised: train L5b with (L3 context, L4 input) → displacement
         pairs. Requires an external supervision signal.
      B. Hebbian: L5b synapses strengthen when the displacement output
         leads to a correctly predicted next state (predictive coding).
      C. Error-driven: use the anomaly score at the result step as an
         error signal to adjust L5b's displacement output.
    Option B or C aligns with the biological Hebbian/predictive coding
    framework and avoids backpropagation.
""")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"{'═'*60}")
print(f"  RESULTS: {passed} passed (of {passed+failed} total, "
      f"{expected_failures} expected failures)")
print(f"{'═'*60}")
print()
print("  Phase 9b correctly demonstrates the limitation:")
print("  Without learned L5b displacement cells, the column cannot")
print("  learn that '+' means 'apply the next number as a displacement'.")
print()
print("  This failing test IS the specification for Phase 10 (learned L5b).")
