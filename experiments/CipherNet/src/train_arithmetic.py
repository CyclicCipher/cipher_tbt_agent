"""CipherNet Arithmetic Training — learn math from token streams.

The system receives raw character tokens with no inherent meaning,
just like a transformer. It must discover:
  1. Digit → magnitude mapping (ANS grounding)
  2. Succession (after 3 comes 4)
  3. Place value (digit position determines magnitude)
  4. Carry rule (9+1 = 10, digit rollover)
  5. Addition and subtraction

The only innate knowledge: the ANS (approximate numerosity sense)
with 3 broadly-tuned columns at ~1, ~3, ~9.

Usage:
    python train_arithmetic.py --stage succession
    python train_arithmetic.py --stage addition
    python train_arithmetic.py --stage all
"""
from __future__ import annotations

import argparse
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from brain import Brain
from graph import TEMPORAL
from train import setup_brain, stage_ans_grounding


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def train_example(brain, input_tokens: str, output_tokens: str,
                  settle_steps: int = 20, lr: float = 0.02):
    """Train one example: feed input, clamp output, settle, learn.

    Input tokens are fed sequentially. Output tokens are clamped
    at the output cortex. The system must learn to produce the
    output from the input through its own dynamics.
    """
    graph = brain.graph
    priors = brain.priors

    # Feed input tokens one at a time.
    graph.reset_activations()
    # Reset GPi tonic
    for key in ['gpi_0', 'gpi_1', 'gpi_2', 'gpi_3', 'gpi_4']:
        if key in priors.get('basal_ganglia', {}):
            graph.activate(priors['basal_ganglia'][key], 0.8)

    for tok in input_tokens:
        brain.feed(tok, n_steps=2)

    # Build clamp: input tokens at full strength + output clamped.
    clamp = {}
    for tok in set(input_tokens):
        col = brain.tio._input_columns.get(tok)
        if col:
            clamp[col['L4']] = 1.0

    # Clamp first output character at output cortex.
    # For multi-char outputs, we train one char at a time.
    out_node = brain.tio._output_token_map.get(output_tokens[0])
    if out_node is not None:
        clamp[out_node] = 1.0

    # Suppress wrong digit outputs by clamping to 0.
    # Biology: PV+ basket cells in motor cortex actively suppress
    # non-selected actions during learning. Without this, non-target
    # outputs develop positive error and their incoming edges get
    # STRENGTHENED (echo interference: digit maps to itself).
    # Clamping to 0 creates negative error → weakens spurious edges.
    # Only suppress digit outputs (0-9), not all 49 tokens.
    for tok, nid in brain.tio._output_token_map.items():
        if tok != output_tokens[0] and tok in '0123456789':
            clamp[nid] = 0.0

    brain.settle(n_steps=settle_steps, clamp=clamp)
    graph.learn(learning_rate=lr, synaptogenesis=False, weight_decay=0.0)


def test_example(brain, input_tokens: str, expected: str,
                 n_steps: int = 10) -> tuple[str, bool]:
    """Test one example: feed input, read output.

    Uses settle() (PC inference) so the WTA inhibitor has time to
    suppress non-winners. Clamp only inputs (no output clamping).
    """
    graph = brain.graph
    priors = brain.priors

    graph.reset_activations()
    for key in ['gpi_0', 'gpi_1', 'gpi_2', 'gpi_3', 'gpi_4']:
        if key in priors.get('basal_ganglia', {}):
            graph.activate(priors['basal_ganglia'][key], 0.8)

    for tok in input_tokens:
        brain.feed(tok, n_steps=2)

    # Clamp inputs during inference (question persists).
    clamp = {}
    for tok in set(input_tokens):
        col = brain.tio._input_columns.get(tok)
        if col:
            clamp[col['L4']] = 1.0

    # Use settle (PC inference) — gives the WTA inhibitor time to
    # suppress non-winners, unlike raw step() which is feed mode.
    brain.settle(n_steps=n_steps, clamp=clamp)

    out, act = brain.read_output()
    correct = out == expected[0] if expected else False
    return out or '?', correct


# -----------------------------------------------------------------------
# Multi-digit helpers (plan-then-execute, not autoregressive)
# -----------------------------------------------------------------------

def _efference_copy_phase_advance(brain):
    """TBT efference copy: advance position by one theta step.

    Biology: when motor output fires, L5 sends a DISPLACEMENT signal
    through the thalamus to L6 of sensory columns AND to the PFC
    sequencer. This displacement says "position += 1" — NOT "I
    produced digit X."

    Two effects:
    1. L6 grid cells advance phase (location update via path integration)
    2. PFC sequencer advances phase (position gating for output cortex)

    The sequencer's phase determines WHICH input position the output
    cortex responds to. Only digits whose phase matches the sequencer
    can win the WTA. This is TBT: grid cell location gates sensory
    interpretation.

    Reference: Hawkins et al. 2018, Lewis et al. 2019.
    """
    import math
    phase_step = 2 * math.pi / 8  # THETA_PERIOD = 8
    graph = brain.graph
    # Advance L6 phases (grid cell path integration).
    for char in '0123456789':
        col = brain.tio._input_columns.get(char)
        if col:
            l6_node = graph.get_node(col['L6'])
            if l6_node:
                l6_node.phase = (l6_node.phase + phase_step) % (2 * math.pi)
    # Advance PFC sequencer phase (position gating for output).
    seq_key = brain.priors.get('pfc', {}).get('sequencer:L5')
    if seq_key is not None:
        seq_node = graph.get_node(seq_key)
        if seq_node:
            seq_node.phase = (seq_node.phase + phase_step) % (2 * math.pi)
            # Activate the sequencer so its phase reaches output cortex.
            seq_node.activation = max(seq_node.activation, 0.5)
    # One step to propagate.
    graph.step()


def _init_sequencer_phase(brain, input_str: str):
    """Set the PFC sequencer phase to match the FIRST input token's phase.

    The sequencer starts at the same phase as the leftmost (most
    significant) digit. The efference copy then advances it through
    subsequent positions. This aligns the sequencer's position code
    with the input position encoding.
    """
    if not input_str:
        return
    first_col = brain.tio._input_columns.get(input_str[0])
    if first_col:
        first_l4 = brain.graph.get_node(first_col['L4'])
        if first_l4:
            seq_key = brain.priors.get('pfc', {}).get('sequencer:L5')
            if seq_key is not None:
                seq_node = brain.graph.get_node(seq_key)
                if seq_node:
                    seq_node.phase = first_l4.phase
                    seq_node.activation = 0.5


def train_multi_digit(brain, input_str: str, output_str: str,
                      settle_steps: int = 25, lr: float = 0.02):
    """Train one multi-digit example with left-to-right position training.

    The graph state persists between output positions — NO reset.
    After settling for position 0 (leftmost digit), the graph encodes
    "I just produced X at the leftmost position." This context helps
    the next settle produce the correct digit for position 1.

    The carry signal lives in residual activations between positions.
    This is how the system learns the carry rule from examples.
    """
    graph = brain.graph
    priors = brain.priors

    # Reset and set GPi tonic.
    graph.reset_activations()
    for key in ['gpi_0', 'gpi_1', 'gpi_2', 'gpi_3', 'gpi_4']:
        if key in priors.get('basal_ganglia', {}):
            graph.activate(priors['basal_ganglia'][key], 0.8)

    # Feed all input tokens.
    for tok in input_str:
        brain.feed(tok, n_steps=2)

    # Build input clamp (persists across all positions).
    input_clamp = {}
    for tok in set(input_str):
        col = brain.tio._input_columns.get(tok)
        if col:
            input_clamp[col['L4']] = 1.0

    # Initialize sequencer to first input position (leftmost digit).
    _init_sequencer_phase(brain, input_str)

    # Train each output position LEFT TO RIGHT.
    for i, out_char in enumerate(output_str):
        clamp = dict(input_clamp)  # copy input clamp

        # Clamp target output.
        out_node = brain.tio._output_token_map.get(out_char)
        if out_node is not None:
            clamp[out_node] = 1.0

        # Suppress non-target digit outputs (PV+ basket cell inhibition).
        for tok, nid in brain.tio._output_token_map.items():
            if tok != out_char and tok in '0123456789':
                clamp[nid] = 0.0

        brain.settle(n_steps=settle_steps, clamp=clamp)
        graph.learn(learning_rate=lr, synaptogenesis=False, weight_decay=0.0)

        # TBT efference copy: DISPLACEMENT, not content.
        # After producing output, advance L6 phases of all input columns
        # by one theta step. This tells every column "position advanced"
        # without specifying WHAT was produced. L6 then predicts what to
        # expect at the next position (via L6→L4 feedback edges).
        # Biology: motor L5 → thalamus → sensory L6 (displacement signal).
        brain.tio.clear_output_with_inhibitor()
        _efference_copy_phase_advance(brain)

    # Train EOS after all digits.
    eos_node = brain.tio._output_token_map.get('<EOS>')
    if eos_node is not None:
        clamp = dict(input_clamp)
        clamp[eos_node] = 1.0
        for tok, nid in brain.tio._output_token_map.items():
            if tok in '0123456789':
                clamp[nid] = 0.0
        brain.settle(n_steps=settle_steps, clamp=clamp)
        graph.learn(learning_rate=lr * 0.5, synaptogenesis=False, weight_decay=0.0)


def test_multi_digit(brain, input_str: str, expected: str,
                     settle_steps: int = 20, max_digits: int = 6) -> tuple[str, bool]:
    """Test multi-digit output via sequential position readout.

    Mirrors the training protocol: for each position, settle with
    input clamped and read the output. The graph state carries over
    between positions (no reset), so carry information persists.

    This is plan-then-execute: each settle pre-computes the digit
    for the current position. The sequential readout is just reading
    from the pre-computed plan, not autoregressive computation.
    """
    graph = brain.graph
    priors = brain.priors

    graph.reset_activations()
    for key in ['gpi_0', 'gpi_1', 'gpi_2', 'gpi_3', 'gpi_4']:
        if key in priors.get('basal_ganglia', {}):
            graph.activate(priors['basal_ganglia'][key], 0.8)

    for tok in input_str:
        brain.feed(tok, n_steps=2)

    # Input clamp persists throughout.
    input_clamp = {}
    for tok in set(input_str):
        col = brain.tio._input_columns.get(tok)
        if col:
            input_clamp[col['L4']] = 1.0

    # Initialize sequencer to first input position.
    _init_sequencer_phase(brain, input_str)

    # Sequential position readout with TBT displacement efference copy.
    n_output = len(expected) if expected else max_digits
    collected = []
    for pos in range(n_output):
        brain.settle(n_steps=settle_steps, clamp=input_clamp)
        out, act = brain.read_output()

        if out is None or act < 0.01:
            collected.append('?')
        else:
            collected.append(out)

        # TBT efference copy: advance position, not content.
        brain.tio.clear_output_with_inhibitor()
        _efference_copy_phase_advance(brain)

    produced = ''.join(collected)
    return produced, produced == expected


def _trace_multi_digit(brain, input_str: str, max_digits: int = 6):
    """Diagnostic: trace multi-digit output with detailed activations."""
    graph = brain.graph
    priors = brain.priors

    graph.reset_activations()
    for key in ['gpi_0', 'gpi_1', 'gpi_2', 'gpi_3', 'gpi_4']:
        if key in priors.get('basal_ganglia', {}):
            graph.activate(priors['basal_ganglia'][key], 0.8)

    for tok in input_str:
        brain.feed(tok, n_steps=2)

    _init_sequencer_phase(brain, input_str)

    input_clamp = {}
    for tok in set(input_str):
        col = brain.tio._input_columns.get(tok)
        if col:
            input_clamp[col['L4']] = 1.0

    print(f"  Trace '{input_str}':")
    collected = []
    for step in range(max_digits):
        brain.settle(n_steps=15, clamp=input_clamp)

        # Top 5 output activations.
        top = []
        for nid, token in brain.tio._output_node_tokens.items():
            node = graph.get_node(nid)
            if node and node.activation > 0.001:
                top.append((node.activation, node.phase, token))
        top.sort(reverse=True)
        top5 = top[:5]

        out, act = brain.read_output()
        phase_str = ''
        if out and out != '<EOS>':
            out_node = brain.tio._output_token_map.get(out)
            if out_node:
                n = graph.get_node(out_node)
                if n:
                    phase_str = f' phase={n.phase:.2f}'

        detail = ' '.join(f"{t}:{a:.3f}" for a, p, t in top5)
        print(f"    Pos {step}: winner='{out}' act={act:.3f}{phase_str}  [{detail}]")

        if out is None or out == '<EOS>' or act < 0.03:
            break
        collected.append(out)
        brain.tio.clear_output_with_inhibitor()
        _efference_copy_phase_advance(brain)

    print(f"    Produced: '{''.join(collected)}'")
    return ''.join(collected)


# -----------------------------------------------------------------------
# Stage 1: Succession (single digit)
# -----------------------------------------------------------------------

def _diagnose_weights(brain, digit: int):
    """Print successor vs echo edge weights for a digit."""
    graph = brain.graph
    priors = brain.priors
    col = brain.tio._input_columns[str(digit)]
    l23 = col['L23']

    successor = digit + 1 if digit < 9 else None
    echo_node = priors['output_cortex'][f'out:{digit}']
    succ_node = priors['output_cortex'][f'out:{successor}'] if successor is not None else None

    echo_w = 0.0
    succ_w = 0.0
    for edge in graph._outgoing.get(l23, []):
        if edge.target == echo_node:
            echo_w = edge.weight
        if succ_node is not None and edge.target == succ_node:
            succ_w = edge.weight

    return echo_w, succ_w


def stage_succession(brain: Brain, epochs: int = 100, verbose: bool = False):
    """Learn succession: feed digit N, output digit N+1.

    Training data: "0" → "1", "1" → "2", ..., "8" → "9"
    The system must learn the number line from these examples.
    The ANS provides approximate magnitude; succession provides
    the exact ordering.
    """
    print("\n=== Stage 1: Succession ===")

    # Training pairs: input digit → next digit
    train_pairs = [(str(d), str(d + 1)) for d in range(9)]

    brain.attend(1.0)

    for epoch in range(epochs):
        random.shuffle(train_pairs)
        for inp, out in train_pairs:
            train_example(brain, inp, out, settle_steps=15, lr=0.02)

        if epoch % 10 == 0:
            correct = 0
            details = []
            for inp, out in train_pairs:
                produced, ok = test_example(brain, inp, out, n_steps=10)
                if ok:
                    correct += 1
                details.append(f"{inp}->{produced}{'OK' if ok else 'X'}")
            print(f"  Epoch {epoch:3d}: {correct}/9  {' '.join(details)}")

            # Weight diagnostics for sample digits.
            if verbose or epoch % 20 == 0:
                for d in [1, 4, 7]:
                    echo_w, succ_w = _diagnose_weights(brain, d)
                    margin = succ_w - echo_w
                    print(f"    char:{d} -> echo={echo_w:.3f} succ={succ_w:.3f} "
                          f"margin={margin:+.3f}")

    # Final test
    correct = 0
    results = []
    for inp, out in train_pairs:
        produced, ok = test_example(brain, inp, out, n_steps=10)
        if ok:
            correct += 1
        results.append(f"{inp}->{produced}({'OK' if ok else 'X'})")
    print(f"  Final: {correct}/9  {' '.join(results)}")
    # Final weight diagnostics.
    print("  Weight diagnostics (echo vs successor):")
    for d in range(9):
        echo_w, succ_w = _diagnose_weights(brain, d)
        margin = succ_w - echo_w
        print(f"    char:{d} -> echo={echo_w:.3f} succ={succ_w:.3f} "
              f"margin={margin:+.3f}")


# -----------------------------------------------------------------------
# Stage 2a: Succession + EOS (retrain with multi-digit protocol)
# -----------------------------------------------------------------------

def stage_succession_eos(brain: Brain, epochs: int = 20, verbose: bool = False):
    """Retrain single-digit succession using multi-digit protocol + EOS.

    Establishes the EOS pattern: after the last significant digit,
    produce the end-of-sequence token. Uses train_multi_digit which
    trains EOS after each example.
    """
    print("\n=== Stage 2a: Succession + EOS ===")
    train_pairs = [(str(d), str(d + 1)) for d in range(9)]
    brain.attend(1.0)

    for epoch in range(epochs):
        random.shuffle(train_pairs)
        for inp, out in train_pairs:
            train_multi_digit(brain, inp, out, settle_steps=15, lr=0.02)

        if epoch % 10 == 0:
            correct = 0
            for inp, out in train_pairs:
                produced, ok = test_multi_digit(brain, inp, out, settle_steps=10)
                if ok:
                    correct += 1
            print(f"  Epoch {epoch:3d}: {correct}/9 succession+EOS")

    # Final
    correct = 0
    for inp, out in train_pairs:
        produced, ok = test_multi_digit(brain, inp, out, settle_steps=10)
        if ok:
            correct += 1
    print(f"  Final: {correct}/9")


# -----------------------------------------------------------------------
# Stage 2b: Two-digit no-carry succession
# -----------------------------------------------------------------------

def stage_nocarry(brain: Brain, epochs: int = 80, verbose: bool = False):
    """Two-digit succession without carry: 10->11, 15->16, 23->24, etc.

    Teaches multi-digit output ordering (tens then ones, left to right)
    and the "echo" pattern (tens digit stays when ones digit < 9).
    """
    print("\n=== Stage 2b: Two-Digit No-Carry ===")

    # Train: tens digits 1-4, ones digits 0-8
    train_pairs = []
    for tens in range(1, 5):
        for ones in range(0, 9):
            train_pairs.append((f"{tens}{ones}", f"{tens}{ones + 1}"))
    # Holdout: tens digits 5-8
    holdout = []
    for tens in range(5, 9):
        for ones in range(0, 9):
            holdout.append((f"{tens}{ones}", f"{tens}{ones + 1}"))

    print(f"  Train: {len(train_pairs)}, Holdout: {len(holdout)}")
    brain.attend(1.0)

    for epoch in range(epochs):
        random.shuffle(train_pairs)
        for inp, out in train_pairs:
            train_multi_digit(brain, inp, out, settle_steps=20, lr=0.02)

        if epoch % 10 == 0:
            # Train accuracy (sample)
            t_correct = 0
            sample = train_pairs[:20]
            for inp, out in sample:
                _, ok = test_multi_digit(brain, inp, out, settle_steps=15)
                if ok:
                    t_correct += 1
            # Holdout accuracy (sample)
            h_correct = 0
            h_sample = holdout[:20]
            for inp, out in h_sample:
                _, ok = test_multi_digit(brain, inp, out, settle_steps=15)
                if ok:
                    h_correct += 1
            print(f"  Epoch {epoch:3d}: train {t_correct}/{len(sample)}, "
                  f"holdout {h_correct}/{len(h_sample)}")
            if verbose and epoch % 20 == 0:
                for inp, out in [("12", "13"), ("23", "24"), ("41", "42")]:
                    _trace_multi_digit(brain, inp)


# -----------------------------------------------------------------------
# Stage 2c: Two-digit carry succession
# -----------------------------------------------------------------------

def stage_carry_digits(brain: Brain, epochs: int = 120, verbose: bool = False):
    """Two-digit succession WITH carry: 19->20, 29->30, 9->10, etc.

    Mixed with no-carry examples. Carry examples repeated 4x per epoch.
    """
    print("\n=== Stage 2c: Two-Digit Carry ===")

    # No-carry examples (review)
    nocarry = []
    for tens in range(1, 5):
        for ones in range(0, 9):
            nocarry.append((f"{tens}{ones}", f"{tens}{ones + 1}"))

    # Carry examples
    carry = [("9", "10")]  # single->double
    for tens in range(1, 5):
        carry.append((f"{tens}9", f"{tens + 1}0"))

    # Holdout carry: unseen tens digits
    holdout_carry = []
    for tens in range(5, 9):
        holdout_carry.append((f"{tens}9", f"{tens + 1}0"))
    holdout_nocarry = []
    for tens in range(5, 9):
        for ones in range(0, 9):
            holdout_nocarry.append((f"{tens}{ones}", f"{tens}{ones + 1}"))

    print(f"  NoCarry: {len(nocarry)}, Carry: {len(carry)} (x4), "
          f"Holdout carry: {len(holdout_carry)}, Holdout nocarry: {len(holdout_nocarry)}")
    brain.attend(1.0)

    for epoch in range(epochs):
        # Build epoch batch: nocarry 1x + carry 4x
        batch = list(nocarry) + carry * 4
        random.shuffle(batch)
        for inp, out in batch:
            train_multi_digit(brain, inp, out, settle_steps=20, lr=0.02)

        if epoch % 15 == 0:
            # Carry accuracy
            c_correct = 0
            for inp, out in carry:
                _, ok = test_multi_digit(brain, inp, out, settle_steps=15)
                if ok:
                    c_correct += 1
            # No-carry accuracy (sample)
            nc_correct = 0
            nc_sample = nocarry[:15]
            for inp, out in nc_sample:
                _, ok = test_multi_digit(brain, inp, out, settle_steps=15)
                if ok:
                    nc_correct += 1
            # Holdout carry
            hc_correct = 0
            for inp, out in holdout_carry:
                _, ok = test_multi_digit(brain, inp, out, settle_steps=15)
                if ok:
                    hc_correct += 1
            print(f"  Epoch {epoch:3d}: carry {c_correct}/{len(carry)}, "
                  f"nocarry {nc_correct}/{len(nc_sample)}, "
                  f"holdout_carry {hc_correct}/{len(holdout_carry)}")
            if verbose:
                for inp, out in [("9", "10"), ("19", "20"), ("49", "50")]:
                    _trace_multi_digit(brain, inp)


# -----------------------------------------------------------------------
# Stage 2d: 99 -> 100 (triple-digit carry chain)
# -----------------------------------------------------------------------

def stage_carry_99(brain: Brain, epochs: int = 60, verbose: bool = False):
    """Learn 99->100: double carry producing three-digit output."""
    print("\n=== Stage 2d: 99 -> 100 ===")

    # Review carry + nocarry + the new 99->100
    review_nocarry = [(f"{t}{o}", f"{t}{o+1}") for t in range(1, 5) for o in range(0, 9)]
    review_carry = [("9", "10")] + [(f"{t}9", f"{t+1}0") for t in range(1, 5)]
    triple = [("99", "100")]

    brain.attend(1.0)

    for epoch in range(epochs):
        batch = review_nocarry + review_carry * 2 + triple * 6
        random.shuffle(batch)
        for inp, out in batch:
            train_multi_digit(brain, inp, out, settle_steps=25, lr=0.02)

        if epoch % 10 == 0:
            # Test 99->100
            produced, ok = test_multi_digit(brain, "99", "100", settle_steps=20)
            # Test carry review
            c_correct = 0
            for inp, out in review_carry:
                _, ok2 = test_multi_digit(brain, inp, out, settle_steps=15)
                if ok2:
                    c_correct += 1
            print(f"  Epoch {epoch:3d}: 99->'{produced}' ({'OK' if ok else 'X'}), "
                  f"carry_review {c_correct}/{len(review_carry)}")
            if verbose:
                _trace_multi_digit(brain, "99")


# -----------------------------------------------------------------------
# Stage 2e: OOD evaluation (no training)
# -----------------------------------------------------------------------

def stage_ood(brain: Brain, verbose: bool = True, **kwargs):
    """Test out-of-distribution generalization. No training."""
    print("\n=== Stage 2e: OOD Evaluation ===")

    brain.attend(1.0)
    ood_tests = [
        # Near OOD: unseen 2-digit
        ("59", "60"), ("69", "70"), ("79", "80"), ("89", "90"),
        ("51", "52"), ("67", "68"), ("83", "84"),
        # Far OOD: never-seen digit counts
        ("99", "100"),
        ("199", "200"), ("299", "300"), ("599", "600"),
        ("999", "1000"),
        ("9999", "10000"),
    ]

    correct = 0
    for inp, expected in ood_tests:
        produced, ok = test_multi_digit(brain, inp, expected, settle_steps=25)
        tag = "OK" if ok else "X"
        print(f"  {inp} -> '{produced}' (expected '{expected}') [{tag}]")
        if ok:
            correct += 1
        if verbose:
            _trace_multi_digit(brain, inp)

    print(f"\n  OOD total: {correct}/{len(ood_tests)}")


# -----------------------------------------------------------------------
# Stage 3: Addition (single digit)
# -----------------------------------------------------------------------

def stage_addition(brain: Brain, epochs: int = 200, verbose: bool = False):
    """Learn addition: "3+4=" → "7".

    Single-digit operands, single-digit results (sum ≤ 9).
    """
    print("\n=== Stage 3: Addition ===")

    all_pairs = [(a, b) for a in range(10) for b in range(10) if a + b <= 9]
    random.seed(42)
    random.shuffle(all_pairs)
    holdout = all_pairs[:10]
    train = all_pairs[10:]

    print(f"  Train: {len(train)}, Holdout: {len(holdout)}")

    brain.attend(1.0)

    for epoch in range(epochs):
        random.shuffle(train)
        for a, b in train:
            inp = f"{a}+{b}="
            out = str(a + b)
            train_example(brain, inp, out, settle_steps=20, lr=0.02)

        if epoch % 30 == 0:
            correct = 0
            for a, b in train[:20]:
                inp = f"{a}+{b}="
                out = str(a + b)
                _, ok = test_example(brain, inp, out, n_steps=10)
                if ok:
                    correct += 1

            h_correct = 0
            for a, b in holdout:
                inp = f"{a}+{b}="
                out = str(a + b)
                _, ok = test_example(brain, inp, out, n_steps=10)
                if ok:
                    h_correct += 1

            print(f"  Epoch {epoch:3d}: train {correct}/20, "
                  f"holdout {h_correct}/{len(holdout)}")


# -----------------------------------------------------------------------
# Stage 4: Subtraction (single digit)
# -----------------------------------------------------------------------

def stage_subtraction(brain: Brain, epochs: int = 200, verbose: bool = False):
    """Learn subtraction: "7-4=" → "3"."""
    print("\n=== Stage 4: Subtraction ===")

    all_pairs = [(a, b) for a in range(10) for b in range(a + 1)]
    random.seed(43)
    random.shuffle(all_pairs)
    holdout = all_pairs[:10]
    train = all_pairs[10:]

    print(f"  Train: {len(train)}, Holdout: {len(holdout)}")

    brain.attend(1.0)

    for epoch in range(epochs):
        random.shuffle(train)
        for a, b in train:
            inp = f"{a}-{b}="
            out = str(a - b)
            train_example(brain, inp, out, settle_steps=20, lr=0.02)

        if epoch % 30 == 0:
            correct = 0
            for a, b in train[:20]:
                inp = f"{a}-{b}="
                out = str(a - b)
                _, ok = test_example(brain, inp, out, n_steps=10)
                if ok:
                    correct += 1

            h_correct = 0
            for a, b in holdout:
                inp = f"{a}-{b}="
                out = str(a - b)
                _, ok = test_example(brain, inp, out, n_steps=10)
                if ok:
                    h_correct += 1

            print(f"  Epoch {epoch:3d}: train {correct}/20, "
                  f"holdout {h_correct}/{len(holdout)}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

STAGES = {
    'grounding': ['ans_grounding'],
    'succession': ['ans_grounding', 'succession'],
    'succession_eos': ['ans_grounding', 'succession', 'succession_eos'],
    'nocarry': ['ans_grounding', 'succession', 'succession_eos', 'nocarry'],
    'carry': ['ans_grounding', 'succession', 'succession_eos', 'nocarry', 'carry_digits'],
    'carry_99': ['ans_grounding', 'succession', 'succession_eos', 'nocarry', 'carry_digits', 'carry_99'],
    'carry_ood': ['ans_grounding', 'succession', 'succession_eos', 'nocarry', 'carry_digits', 'carry_99', 'ood'],
    'addition': ['ans_grounding', 'succession', 'addition'],
    'subtraction': ['ans_grounding', 'succession', 'addition', 'subtraction'],
}

STAGE_FUNCS = {
    'ans_grounding': stage_ans_grounding,
    'succession': stage_succession,
    'succession_eos': stage_succession_eos,
    'nocarry': stage_nocarry,
    'carry_digits': stage_carry_digits,
    'carry_99': stage_carry_99,
    'ood': stage_ood,
    'addition': stage_addition,
    'subtraction': stage_subtraction,
}


def main():
    parser = argparse.ArgumentParser(description="CipherNet Arithmetic Training")
    parser.add_argument('--stage', type=str, default='succession',
                        choices=list(STAGES.keys()))
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    brain = setup_brain()

    stage_list = STAGES[args.stage]
    for stage_name in stage_list:
        func = STAGE_FUNCS[stage_name]
        kwargs = {'verbose': args.verbose}
        if args.epochs and stage_name == stage_list[-1]:
            kwargs['epochs'] = args.epochs
        func(brain, **kwargs)

    print(f"\n{'=' * 60}")
    print(f"Training complete. Graph: {brain.graph.summary()}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
