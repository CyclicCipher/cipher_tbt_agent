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


def evaluate_pairs(brain, pairs, op_char, n_steps=10):
    """Evaluate accuracy — single forward pass (no settle, no clamp).

    At test time, a trained PCN becomes a feedforward network.
    Just push input through the weights and read the output.
    """
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

        if len(expected) > 1:
            continue

        # Forward pass: feed input, step, read output. No settle.
        brain.graph.reset_activations()
        for tok in f"{a}{op_char}{b}=":
            brain.feed(tok, n_steps=2)
        brain.step(n_steps)

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

    # Innate: digit columns → ANS number columns (sensory pathway).
    # Each digit connects to the ANS column closest to its numerosity.
    # Connections are weak — the digit-to-numerosity mapping is LEARNED.
    ans_cols = {'num1': priors['ans']['num1:L4'],
                'num3': priors['ans']['num3:L4'],
                'num9': priors['ans']['num9:L4']}
    for d in range(10):
        col = brain.tio._input_columns[str(d)]
        # Initial connections from L23 (gamma, fast) to ANS columns.
        # Using TEMPORAL not BINDING for stronger signal propagation.
        for ans_name, ans_nid in ans_cols.items():
            graph.add_edge(col['L23'], ans_nid, edge_type=TEMPORAL, weight=0.1)

    # === DOMAIN-GENERAL SENSORY PATHWAY ===
    #
    # Two parallel paths to token cortex (like cochlea → A1):
    #
    # 1. TEMPORAL PATH: input relay → relay_token → temporal columns
    #    Carries "something arrived" signal (onset/transition/sustained).
    #    Undifferentiated — same for all tokens.
    #
    # 2. IDENTITY PATH: input column L23 → identity columns (DIRECT)
    #    Carries "THIS SPECIFIC token arrived" signal.
    #    All-to-all, initially weak. Columns specialize through competitive
    #    learning (lateral inhibition forces WTA → each column becomes
    #    selective for specific tokens). This is the tokentopic map.
    #
    # Path 1: relay_token (temporal dynamics)
    token_relay = priors['thalamus']['relay_token']
    for char in '0123456789+-*/=':
        col = brain.tio._input_columns[char]
        relay = col.get('relay')
        if relay is not None:
            graph.add_edge(relay, token_relay, edge_type=TEMPORAL, weight=0.5)

    # Path 2: identity edges via RELAY (tokentopic map)
    # Input column RELAY → identity column L4. Using the relay (not L23)
    # because relay nodes have role='relay' which gets ACh ENHANCEMENT.
    # Direct L23→L4 cross-subgraph edges get 70% ACh SUPPRESSION
    # (muscarinic intracortical lateral suppression) — too weak to work.
    # Biology: A1 receives from thalamus (MGB), not lateral cortex.
    #
    # STOCHASTIC SYMMETRY BREAKING: randomized weights so different
    # identity columns start with different sensitivities.
    import random as _rng
    _rng.seed(42)  # reproducible initialization
    tc_priors = priors.get('token_cortex', {})
    for char in '0123456789+-*/=':
        col = brain.tio._input_columns[char]
        relay = col.get('relay')
        if relay is None:
            continue
        for i in range(8):
            id_l4_key = f'id{i}:L4'
            if id_l4_key in tc_priors:
                # Random weight: mean 0.1, range [0.02, 0.18]
                w = 0.02 + _rng.random() * 0.16
                graph.add_edge(relay, tc_priors[id_l4_key],
                               edge_type=TEMPORAL, weight=w)

    # 2. WM stripes → output cortex (excitatory readout)
    # ALL L5 cells of each WM stripe → ALL output nodes. The population
    # pattern across L5 cells encodes identity. Each L5 cell contributes
    # to the output drive. Learned weights specialize which patterns
    # drive which outputs.
    for wm_name in ['wm0', 'wm1', 'wm2']:
        pfc_nodes = priors['pfc']
        # Find all L5 cells for this WM stripe
        l5_cells = [nid for key, nid in pfc_nodes.items()
                    if key.startswith(f'{wm_name}:L5:')]
        if not l5_cells:
            # Fallback: single-cell column (backward compat)
            l5_key = f'{wm_name}:L5'
            if l5_key in pfc_nodes:
                l5_cells = [pfc_nodes[l5_key]]
        for l5_nid in l5_cells:
            for node_key, node_id in priors['output_cortex'].items():
                if node_key.startswith('out:'):
                    # Weight scaled by 1/n_cells to keep total drive
                    # similar to single-cell (24 cells × 0.005 ≈ old 3 × 0.05).
                    graph.add_edge(l5_nid, node_id,
                                   edge_type=TEMPORAL, weight=0.005)

    # 3. Input columns → BG striatum (gating control)
    # All-to-all: any input can learn to gate any WM stripe.
    # The BG learns via dopamine which input should gate which stripe.
    # This is domain-general — same mechanism for digits, letters, etc.
    for char in '0123456789+-*/=':
        col = brain.tio._input_columns[char]
        for stripe in range(3):  # WM stripes 0-2
            d1_key = f'd1_go_{stripe}'
            d2_key = f'd2_nogo_{stripe}'
            if d1_key in priors.get('basal_ganglia', {}):
                graph.add_edge(col['L5'], priors['basal_ganglia'][d1_key],
                               edge_type=TEMPORAL, weight=0.05)
            if d2_key in priors.get('basal_ganglia', {}):
                graph.add_edge(col['L5'], priors['basal_ganglia'][d2_key],
                               edge_type=TEMPORAL, weight=0.05)

    # Innate: '=' -> output cortex disinhibition.
    eq_col = brain.tio._input_columns['=']
    graph.add_edge(eq_col['L5'], priors['output_cortex']['inhibitor'],
                   edge_type=TEMPORAL, weight=-0.1)

    # === INPUT COLUMN LATERAL INHIBITION ===
    # Like the output cortex inhibitor: one inhibitor node that creates
    # winner-take-all competition between input columns. When a new
    # token arrives, the inhibitor suppresses old tokens. Only the most
    # recently activated columns survive.
    # This is the biological lateral inhibition that prevents state
    # saturation (the Mamba selective forgetting equivalent).
    input_inhibitor = graph.add_node(
        label="input_inhibitor", subgraph="thalamus",
        role="inhibitor")
    for char in '0123456789+-*/=':
        col = brain.tio._input_columns[char]
        # L23 drives the inhibitor (excitatory, directed).
        graph.add_edge(col['L23'], input_inhibitor,
                       edge_type=TEMPORAL, weight=0.15)
        # Inhibitor suppresses L23 (inhibitory, directed).
        # Negative TEMPORAL = directed inhibition (interneuron).
        # Weight tuned: strong enough to suppress unfed columns,
        # weak enough that clamped L4 input (1.0) overcomes it.
        graph.add_edge(input_inhibitor, col['L23'],
                       edge_type=TEMPORAL, weight=-0.3)

    # Backward prediction edges from output cortex to digit columns.
    # Output cortex predicts what input it expects (generative model).
    # Routed to L6 (feedback layer) → L6 generates prediction for L4.
    # Same-digit feedback is stronger (0.05) to support multi-digit
    # autoregressive context: when "2" is produced, char:2 gets a
    # strong feedback signal that influences the next position's output.
    for d_out in range(10):
        out_node = priors['output_cortex'][f'out:{d_out}']
        for d_in in range(10):
            col = brain.tio._input_columns[str(d_in)]
            w = 0.05 if d_in == d_out else 0.01
            graph.add_edge(out_node, col['L6'], edge_type=TEMPORAL, weight=w)

    # === EOS CONNECTIVITY ===
    # Weak edges from digit columns to EOS output node.
    # Without any pathway, EOS can never fire. The system learns
    # WHEN to produce EOS through training (after last significant digit).
    eos_node = priors['output_cortex']['out:EOS']
    for d in range(10):
        col = brain.tio._input_columns[str(d)]
        graph.add_edge(col['L23'], eos_node, edge_type=TEMPORAL, weight=0.01)

    # Helper: find all cells for a multi-cell layer.
    # E.g., _find_cells(priors['broca'], 'ba44a:L5') returns all L5 cell nids.
    def _find_cells(prior_dict, prefix):
        """Find all cell node IDs matching a prefix (e.g., 'ba44a:L5')."""
        cells = [nid for key, nid in prior_dict.items()
                 if key.startswith(f'{prefix}:') and key[len(prefix)+1:].isdigit()]
        if not cells and prefix in prior_dict:
            cells = [prior_dict[prefix]]  # single-cell fallback
        return cells

    # === WORKSPACE CONJUNCTION PATH ===
    # Broca columns → output cortex (all L5 cells → all output nodes).
    for broca_col in ['ba44a', 'ba44p', 'ba45']:
        l5_cells = _find_cells(priors['broca'], f'{broca_col}:L5')
        for l5_nid in l5_cells:
            for d in range(10):
                out_node = priors['output_cortex'][f'out:{d}']
                graph.add_edge(l5_nid, out_node, edge_type=TEMPORAL, weight=0.02)

    # Broca → digit column L6 (backward predictions, first L5 cell only).
    for broca_col in ['ba44a', 'ba45']:
        l5_cells = _find_cells(priors['broca'], f'{broca_col}:L5')
        if l5_cells:
            for d in range(10):
                col = brain.tio._input_columns[str(d)]
                graph.add_edge(l5_cells[0], col['L6'],
                               edge_type=TEMPORAL, weight=0.01)

    print(f"Brain setup: {graph.summary()}")
    return brain


# -----------------------------------------------------------------------
# ANS Grounding
# -----------------------------------------------------------------------

def stage_ans_grounding(brain: Brain, epochs: int = 50, verbose: bool = False):
    """Teach digit → ANS numerosity association.

    Present each digit alongside activation of the closest ANS
    number column. The system learns which digit maps to which
    approximate numerosity.
    """
    print("\n=== ANS Grounding ===")
    graph = brain.graph
    ans = brain.priors['ans']

    # Map digits to closest ANS column (log-spaced: 1, 3, 9)
    # Digits 0-2 → num1, digits 3-5 → num3, digits 6-9 → num9
    digit_to_ans = {}
    for d in range(10):
        if d <= 2:
            digit_to_ans[d] = ans['num1:L4']
        elif d <= 5:
            digit_to_ans[d] = ans['num3:L4']
        else:
            digit_to_ans[d] = ans['num9:L4']

    for epoch in range(epochs):
        for d in range(10):
            graph.reset_activations()
            brain.feed(str(d), n_steps=2)
            # Settle with digit + correct ANS column clamped.
            # Wrong ANS columns clamped to 0.
            clamp = {brain.tio._input_columns[str(d)]['L4']: 1.0,
                     digit_to_ans[d]: 1.0}
            for ans_nid in digit_to_ans.values():
                if ans_nid != digit_to_ans[d]:
                    clamp[ans_nid] = 0.0
            brain.settle(n_steps=10, clamp=clamp)
            graph.learn(learning_rate=0.02, synaptogenesis=False)

    print(f"  Done. Edges: {graph.edge_count()}")


# -----------------------------------------------------------------------
# Addition — the main event
# -----------------------------------------------------------------------

def stage_addition(brain: Brain, epochs: int = 100, verbose: bool = False):
    """Teach single-digit addition via pure local predictive coding.

    The output clamp IS a prediction (active inference: "I predict
    the answer is 7"). The system settles to minimize prediction
    error everywhere. Each node's LOCAL error drives its own learning.
    No backward sweep. No global error signal.

    Training loop:
    1. Feed "A+B=" tokens
    2. Settle with input AND output clamped
       (input = sensory evidence, output = goal/prediction)
    3. Learn from LOCAL prediction errors at every node
    4. Test: settle with input only (no output clamp)
    """
    print("\n=== Addition (Pure Local Predictive Coding) ===")
    graph = brain.graph

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

            # 2. Settle with BOTH input AND output clamped.
            #    Input clamp = sensory evidence ("this is 3+4=")
            #    Output clamp = prediction/goal ("I predict the answer is 7")
            #    This IS active inference: the system adjusts internal
            #    state to be consistent with both the question and answer.
            #    Every node computes its own LOCAL prediction error.
            clamp = {}
            for tok in set(tokens):
                col = brain.tio._input_columns.get(tok)
                if col:
                    clamp[col['L4']] = 1.0
            # Output = goal prediction.
            out_node = brain.tio._output_token_map.get(expected)
            if out_node is not None:
                clamp[out_node] = 1.0

            brain.settle(n_steps=20, clamp=clamp)
            total_error += graph.total_error()

            # 3. Learn from LOCAL errors — each node's own
            #    (sensory - prediction) drives its own weight updates.
            #    No backward sweep. No global signal. Pure local.
            graph.learn(learning_rate=0.01, synaptogenesis=False,
                        weight_decay=0.0)

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
