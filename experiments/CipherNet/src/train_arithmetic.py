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

    # Suppress wrong output tokens.
    for tok, nid in brain.tio._output_token_map.items():
        if tok != output_tokens[0]:
            # Don't hard-clamp to 0 — let inhibitor handle it.
            pass

    brain.settle(n_steps=settle_steps, clamp=clamp)
    graph.learn(learning_rate=lr, synaptogenesis=False, weight_decay=0.0)


def test_example(brain, input_tokens: str, expected: str,
                 n_steps: int = 10) -> tuple[str, bool]:
    """Test one example: feed input, read output. No clamping."""
    graph = brain.graph
    priors = brain.priors

    graph.reset_activations()
    for key in ['gpi_0', 'gpi_1', 'gpi_2', 'gpi_3', 'gpi_4']:
        if key in priors.get('basal_ganglia', {}):
            graph.activate(priors['basal_ganglia'][key], 0.8)

    for tok in input_tokens:
        brain.feed(tok, n_steps=2)

    # Clamp inputs during inference (question persists).
    for tok in set(input_tokens):
        col = brain.tio._input_columns.get(tok)
        if col:
            graph.activate(col['L4'], 1.0)

    brain.step(n_steps)

    out, act = brain.read_output()
    correct = out == expected[0] if expected else False
    return out or '?', correct


# -----------------------------------------------------------------------
# Stage 1: Succession (single digit)
# -----------------------------------------------------------------------

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

        if epoch % 20 == 0:
            correct = 0
            for inp, out in train_pairs:
                _, ok = test_example(brain, inp, out, n_steps=8)
                if ok:
                    correct += 1
            print(f"  Epoch {epoch:3d}: {correct}/9 succession correct")

    # Final test
    correct = 0
    results = []
    for inp, out in train_pairs:
        produced, ok = test_example(brain, inp, out, n_steps=8)
        if ok:
            correct += 1
        results.append(f"{inp}->{produced}({'OK' if ok else 'X'})")
    print(f"  Final: {correct}/9  {' '.join(results)}")


# -----------------------------------------------------------------------
# Stage 2: Multi-digit succession (carry rule)
# -----------------------------------------------------------------------

def stage_carry(brain: Brain, epochs: int = 150, verbose: bool = False):
    """Learn the carry rule: 9→10, 19→20, 99→100.

    Training data includes single-digit rollover and two-digit examples.
    The system must learn:
    - When the ones digit is 9, it becomes 0 and the tens digit increments
    - Multi-character output (autoregressive)
    """
    print("\n=== Stage 2: Carry Rule ===")

    # Mix of single-digit and two-digit succession
    train_pairs = []
    # Single digit (review)
    for d in range(9):
        train_pairs.append((str(d), str(d + 1)))
    # Two-digit no carry
    for tens in range(1, 5):
        for ones in range(0, 9):
            train_pairs.append((f"{tens}{ones}", f"{tens}{ones + 1}"))
    # Two-digit WITH carry
    for tens in range(1, 5):
        train_pairs.append((f"{tens}9", f"{tens + 1}0"))
    # Special: 9 → 10 (single to double digit)
    train_pairs.append(("9", "10"))

    # Holdout: unseen tens digits
    holdout = [(f"{t}{o}", f"{t}{o+1}") for t in [6, 7] for o in range(9)]
    holdout += [(f"{t}9", f"{t+1}0") for t in [6, 7]]

    brain.attend(1.0)

    for epoch in range(epochs):
        random.shuffle(train_pairs)
        for inp, out in train_pairs:
            # For multi-char output, train on first char only for now
            train_example(brain, inp, out, settle_steps=15, lr=0.02)

        if epoch % 30 == 0:
            correct = 0
            total = min(20, len(train_pairs))
            for inp, out in train_pairs[:total]:
                _, ok = test_example(brain, inp, out, n_steps=10)
                if ok:
                    correct += 1

            # Holdout
            h_correct = 0
            for inp, out in holdout[:10]:
                _, ok = test_example(brain, inp, out, n_steps=10)
                if ok:
                    h_correct += 1

            print(f"  Epoch {epoch:3d}: train {correct}/{total}, "
                  f"holdout {h_correct}/10")


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
    'carry': ['ans_grounding', 'succession', 'carry'],
    'addition': ['ans_grounding', 'succession', 'addition'],
    'subtraction': ['ans_grounding', 'succession', 'addition', 'subtraction'],
    'all': ['ans_grounding', 'succession', 'carry', 'addition', 'subtraction'],
}

STAGE_FUNCS = {
    'ans_grounding': stage_ans_grounding,
    'succession': stage_succession,
    'carry': stage_carry,
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
