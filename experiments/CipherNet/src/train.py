"""CipherNet Training — predictive coding on arithmetic.

The training loop:
  1. Feed input tokens (activate input columns)
  2. Clamp desired output (the "goal" = prediction of what output will be)
  3. Settle: find activations that minimize prediction error with
     both input and output clamped (prospective configuration)
  4. Learn: adjust weights based on settled prediction errors
  5. Test: free inference (no output clamp) to check if learned

This is the TEACHER. It feeds tokens, sets goals, and triggers learning.
It NEVER orchestrates brain regions during computation.

Usage:
    python train.py --stage addition --epochs 50
    python train.py --stage all
"""
from __future__ import annotations

import argparse
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from brain import Brain
from graph import TEMPORAL, BINDING, SPATIAL


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def build_clamp(brain, input_tokens, output_token):
    """Build a clamp dict: fix input L4 nodes + output node."""
    clamp = {}
    # Clamp input columns at their L4 activation.
    for tok in input_tokens:
        col = brain.tio._input_columns.get(tok)
        if col:
            node = brain.graph.get_node(col['L4'])
            if node and node.activation > 0.01:
                clamp[col['L4']] = node.activation
    # Clamp desired output.
    out_node = brain.tio._output_token_map.get(output_token)
    if out_node is not None:
        clamp[out_node] = 1.0
    return clamp


def evaluate_pairs(brain, pairs, op_char, settle_steps=15):
    """Evaluate accuracy on (a, b) pairs — free inference (no clamp)."""
    correct = 0
    for a, b in pairs:
        if op_char == '+':
            expected = str(a + b)
        elif op_char == '-':
            expected = str(a - b)
        elif op_char == '*':
            expected = str(a * b)
        elif op_char == '/':
            expected = str(a // b)
        else:
            continue

        # Only test single-char answers for now.
        if len(expected) > 1:
            continue

        brain.graph.reset_activations()
        for tok in f"{a}{op_char}{b}=":
            brain.feed(tok, n_steps=1)
        brain.settle(n_steps=settle_steps)

        out, act = brain.read_output()
        if out == expected:
            correct += 1

    total = sum(1 for a, b in pairs
                if len(str(a + b if op_char == '+' else
                           a - b if op_char == '-' else
                           a * b if op_char == '*' else
                           a // b)) == 1)
    return correct, max(total, 1)


# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------

def setup_brain() -> Brain:
    """Create brain with minimal innate wiring."""
    brain = Brain(default_decay=0.5)
    graph = brain.graph
    priors = brain.priors

    # Create input columns.
    for char in '0123456789+-*/^=() ':
        brain.tio.get_or_create_input_column(char)

    # Innate: digit columns -> ANS input (sensory pathway).
    ans_input = priors['ans']['input_a']
    for d in range(10):
        col = brain.tio._input_columns[str(d)]
        graph.add_edge(col['L5'], ans_input, edge_type=BINDING, weight=0.05)

    # Innate: ALL digit columns -> ALL output digits (weak).
    # Feedforward: L2/3 (error signal) -> output cortex.
    # Like cortical neurons having many potential synapses, mostly
    # silent. Learning selects which become functional.
    for d_in in range(10):
        col = brain.tio._input_columns[str(d_in)]
        for d_out in range(10):
            out_node = priors['output_cortex'][f'out:{d_out}']
            w = 0.08 if d_in == d_out else 0.02
            graph.add_edge(col['L23'], out_node, edge_type=TEMPORAL, weight=w)

    # Innate: '=' -> output cortex disinhibition.
    eq_col = brain.tio._input_columns['=']
    graph.add_edge(eq_col['L5'], priors['output_cortex']['inhibitor'],
                   edge_type=TEMPORAL, weight=-0.1)

    # Short-range cortical connections (predictive coding wiring):
    # Feedforward (errors UP): L2/3 of lower -> L4 of higher
    # Feedback (predictions DOWN): L5 of higher -> L2/3 of lower (skip L4!)
    for i in range(9):
        col_a = brain.tio._input_columns[str(i)]
        col_b = brain.tio._input_columns[str(i + 1)]
        # Feedforward: errors propagate in both directions between neighbors
        graph.add_edge(col_a['L23'], col_b['L4'], edge_type=TEMPORAL, weight=0.01)
        graph.add_edge(col_b['L23'], col_a['L4'], edge_type=TEMPORAL, weight=0.01)
        # Feedback: predictions propagate, skip L4, target L2/3
        graph.add_edge(col_a['L5'], col_b['L23'], edge_type=TEMPORAL, weight=0.01)
        graph.add_edge(col_b['L5'], col_a['L23'], edge_type=TEMPORAL, weight=0.01)

    # Backward prediction edges from output cortex to digit columns.
    # Output cortex predicts what input it expects (generative model).
    # These go to L2/3 (NOT L4) — predictions skip the error layer.
    for d_out in range(10):
        out_node = priors['output_cortex'][f'out:{d_out}']
        for d_in in range(10):
            col = brain.tio._input_columns[str(d_in)]
            graph.add_edge(out_node, col['L23'], edge_type=TEMPORAL, weight=0.01)

    print(f"Brain setup: {graph.summary()}")
    return brain


# -----------------------------------------------------------------------
# ANS Grounding
# -----------------------------------------------------------------------

def stage_ans_grounding(brain: Brain, epochs: int = 50, verbose: bool = False):
    """Teach digit -> magnitude via ANS."""
    print("\n=== ANS Grounding ===")
    graph = brain.graph
    ans_input = brain.priors['ans']['input_a']

    for epoch in range(epochs):
        for d in range(10):
            graph.reset_activations()
            brain.feed(str(d), n_steps=2)
            graph.activate(ans_input, d / 9.0 if d > 0 else 0.05)
            brain.step(3)
            graph.learn(learning_rate=0.01, synaptogenesis=False,
                        edge_types={BINDING, TEMPORAL})

    print(f"  Done. Edges: {graph.edge_count()}")


# -----------------------------------------------------------------------
# Addition — the main event
# -----------------------------------------------------------------------

def stage_addition(brain: Brain, epochs: int = 100, verbose: bool = False):
    """Teach single-digit addition via predictive coding.

    Training loop:
    1. Feed "A+B=" tokens (activate input columns)
    2. Clamp desired output (goal prediction)
    3. Settle with both input + output clamped (prospective config)
    4. Learn from settled prediction errors
    5. Periodically test free inference (no output clamp)
    """
    print("\n=== Addition (Predictive Coding) ===")
    graph = brain.graph

    # All single-digit pairs with single-char result.
    small_pairs = [(a, b) for a in range(10) for b in range(10)
                   if a + b <= 9]
    random.seed(42)
    random.shuffle(small_pairs)
    holdout = small_pairs[:10]
    train = small_pairs[10:]

    print(f"  Train: {len(train)}, Holdout: {len(holdout)}")

    for epoch in range(epochs):
        random.shuffle(train)
        total_error = 0.0

        for a, b in train:
            expected = str(a + b)
            tokens = f"{a}+{b}="

            # 1. Feed input tokens.
            graph.reset_activations()
            for tok in tokens:
                brain.feed(tok, n_steps=1)

            # 2. Build clamp: fix inputs + desired output.
            clamp = build_clamp(brain, tokens, expected)

            # 3. Settle: find activations minimizing prediction error.
            brain.settle(n_steps=20, clamp=clamp)

            total_error += graph.total_error()

            # 4. Learn once from the settled state.
            graph.learn(learning_rate=0.005, synaptogenesis=False)

        # Evaluate holdout (free inference, no clamp).
        if epoch % 5 == 0:
            h_correct, h_total = evaluate_pairs(brain, holdout, '+', 15)
            t_correct, t_total = evaluate_pairs(brain, train[:20], '+', 15)
            avg_err = total_error / len(train)
            print(f"  Epoch {epoch}: avg_err={avg_err:.2f}, "
                  f"train {t_correct}/{t_total}, "
                  f"holdout {h_correct}/{h_total}, "
                  f"edges {graph.edge_count()}")

    h_correct, h_total = evaluate_pairs(brain, holdout, '+', 15)
    print(f"  Final holdout: {h_correct}/{h_total}")


# -----------------------------------------------------------------------
# Subtraction
# -----------------------------------------------------------------------

def stage_subtraction(brain: Brain, epochs: int = 80, verbose: bool = False):
    """Teach subtraction: "7-4=" -> "3"."""
    print("\n=== Subtraction (Predictive Coding) ===")
    graph = brain.graph

    all_pairs = [(a, b) for a in range(10) for b in range(a + 1)
                 if a - b <= 9]
    random.seed(43)
    random.shuffle(all_pairs)
    holdout = all_pairs[:10]
    train = all_pairs[10:]

    print(f"  Train: {len(train)}, Holdout: {len(holdout)}")

    for epoch in range(epochs):
        random.shuffle(train)
        for a, b in train:
            expected = str(a - b)
            tokens = f"{a}-{b}="

            graph.reset_activations()
            for tok in tokens:
                brain.feed(tok, n_steps=1)

            clamp = build_clamp(brain, tokens, expected)
            brain.settle(n_steps=20, clamp=clamp)
            graph.learn(learning_rate=0.01, synaptogenesis=True,
                        synapse_threshold=0.3, synapse_weight=0.05)

        if epoch % 10 == 0:
            h_correct, h_total = evaluate_pairs(brain, holdout, '-', 15)
            print(f"  Epoch {epoch}: holdout {h_correct}/{h_total}")

    h_correct, h_total = evaluate_pairs(brain, holdout, '-', 15)
    print(f"  Final holdout: {h_correct}/{h_total}")


# -----------------------------------------------------------------------
# Multiplication
# -----------------------------------------------------------------------

def stage_multiplication(brain: Brain, epochs: int = 100, verbose: bool = False):
    """Teach multiplication: "3*4=" -> "12" (single-char results first)."""
    print("\n=== Multiplication (Predictive Coding) ===")
    graph = brain.graph

    # Single-char results only for now.
    small_pairs = [(a, b) for a in range(10) for b in range(10)
                   if a * b <= 9]
    random.seed(44)
    random.shuffle(small_pairs)
    holdout = small_pairs[:8]
    train = small_pairs[8:]

    print(f"  Train: {len(train)}, Holdout: {len(holdout)}")

    for epoch in range(epochs):
        random.shuffle(train)
        for a, b in train:
            expected = str(a * b)
            tokens = f"{a}*{b}="

            graph.reset_activations()
            for tok in tokens:
                brain.feed(tok, n_steps=1)

            clamp = build_clamp(brain, tokens, expected)
            brain.settle(n_steps=20, clamp=clamp)
            graph.learn(learning_rate=0.01, synaptogenesis=True,
                        synapse_threshold=0.3, synapse_weight=0.05)

        if epoch % 10 == 0:
            h_correct, h_total = evaluate_pairs(brain, holdout, '*', 15)
            print(f"  Epoch {epoch}: holdout {h_correct}/{h_total}")


# -----------------------------------------------------------------------
# PEMDAS
# -----------------------------------------------------------------------

def generate_pemdas_expressions():
    """Generate two-operator expressions where precedence matters."""
    expressions = []
    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 6):
                # a + b * c
                result = a + b * c
                if result <= 9:
                    expressions.append((f"{a}+{b}*{c}=", str(result)))
                # a * b + c
                result = a * b + c
                if result <= 9:
                    expressions.append((f"{a}*{b}+{c}=", str(result)))
    return list(set(expressions))


def stage_pemdas(brain: Brain, epochs: int = 100, verbose: bool = False):
    """Teach PEMDAS order of operations."""
    print("\n=== PEMDAS (Predictive Coding) ===")
    graph = brain.graph

    all_data = generate_pemdas_expressions()
    random.seed(46)
    random.shuffle(all_data)
    holdout = all_data[:min(20, len(all_data) // 4)]
    train = all_data[len(holdout):]

    print(f"  Train: {len(train)}, Holdout: {len(holdout)}")

    for epoch in range(epochs):
        random.shuffle(train)
        for expr, expected in train:
            graph.reset_activations()
            for tok in expr:
                brain.feed(tok, n_steps=1)

            clamp = build_clamp(brain, expr, expected)
            brain.settle(n_steps=30, clamp=clamp)
            graph.learn(learning_rate=0.01, synaptogenesis=True,
                        synapse_threshold=0.3, synapse_weight=0.05)

        if epoch % 10 == 0:
            correct = 0
            for expr, expected in holdout:
                graph.reset_activations()
                for tok in expr:
                    brain.feed(tok, n_steps=1)
                brain.settle(n_steps=20)
                out, _ = brain.read_output()
                if out == expected:
                    correct += 1
            print(f"  Epoch {epoch}: holdout {correct}/{len(holdout)}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

STAGES = {
    'grounding': ['ans_grounding'],
    'addition': ['ans_grounding', 'addition'],
    'subtraction': ['ans_grounding', 'addition', 'subtraction'],
    'multiplication': ['ans_grounding', 'addition', 'subtraction',
                        'multiplication'],
    'pemdas': ['ans_grounding', 'addition', 'subtraction',
               'multiplication', 'pemdas'],
    'all': ['ans_grounding', 'addition', 'subtraction',
            'multiplication', 'pemdas'],
}

STAGE_FUNCS = {
    'ans_grounding': stage_ans_grounding,
    'addition': stage_addition,
    'subtraction': stage_subtraction,
    'multiplication': stage_multiplication,
    'pemdas': stage_pemdas,
}


def main():
    parser = argparse.ArgumentParser(description="CipherNet Training")
    parser.add_argument('--stage', type=str, default='addition',
                        choices=list(STAGES.keys()),
                        help='Which stage(s) to run')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epoch count')
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

    print(f"\n{'='*60}")
    print(f"Training complete. Graph: {brain.graph.summary()}")
    print(f"Total prediction error: {brain.graph.total_error():.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
