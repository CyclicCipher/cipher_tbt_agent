"""
Structural operations for the Compositor's categorical graph (Phase 4).

These are NON-GRADIENT operations that update the graph's discrete structure.
Backpropagation is necessary but not sufficient -- it can refine edge weights
but cannot efficiently discover which edges should exist from scratch.

Two types of structural operation:
1. SEEDING: Analyze training data to detect consistent token transitions.
   Set edge logits directly. Run before training begins.
2. COMPOSITION CLOSURE: If edges A->B and B->C both exist, strengthen A->C.
   Run periodically during training.

Both operations are general-purpose: they detect statistical regularities
and categorical structure, not task-specific patterns.
"""

import math
import torch
from collections import defaultdict

from data import encode, VOCAB_SIZE, BOS_ID, EOS_ID, PAD_ID, TOKEN_TO_ID


def seed_oracle_graph(graph, verbose: bool = True):
    """Seed the graph with PERFECT knowledge for the toy tasks.

    This is a diagnostic tool: if the model can't use oracle knowledge,
    the architecture is broken. If it can, the problem is in graph learning.

    Relation 0 — Digit successor: d -> d+1 for d=0..8
    Relation 1 — Carry/wrap: 9 -> 0 (tens digit increments)
    Relation 2 — Operator structure:
        '+' -> digits (operand context), '=' -> digits (result context),
        ',' -> digits (delimiter context)
    Relation 3 — Reverse successor: d+1 -> d (for comparison/subtraction)

    All oracle edges set at logit=6.0 (sigmoid ~ 0.9975).
    All non-oracle edges left at init (-4.0, sigmoid ~ 0.018).
    """
    N = graph.n_nodes
    K = graph.n_relations
    LOGIT_STRONG = 6.0

    # Token IDs for digits
    d = {i: TOKEN_TO_ID[str(i)] for i in range(10)}

    # --- Relation 0: digit successor ---
    if K > 0:
        for i in range(9):
            graph.A.data[0, d[i], d[i + 1]] = LOGIT_STRONG
        if verbose:
            print(f"  Relation 0 (successor): 9 edges at logit={LOGIT_STRONG}")

    # --- Relation 1: carry (9->0) and tens-digit transitions ---
    if K > 1:
        # 9 wraps to 0 (units digit in carry)
        graph.A.data[1, d[9], d[0]] = LOGIT_STRONG
        # When a number crosses a tens boundary, the tens digit increments
        # e.g., 19->20: tens digit goes 1->2
        for i in range(9):
            graph.A.data[1, d[i], d[i + 1]] = LOGIT_STRONG * 0.5  # weaker, context-dependent
        if verbose:
            print(f"  Relation 1 (carry/tens): 10 edges")

    # --- Relation 2: operator-to-digit context ---
    if K > 2:
        comma_id = TOKEN_TO_ID[',']
        plus_id = TOKEN_TO_ID['+']
        eq_id = TOKEN_TO_ID['=']
        lt_id = TOKEN_TO_ID['<']
        gt_id = TOKEN_TO_ID['>']

        # Comma precedes digits (next number starts)
        for i in range(10):
            graph.A.data[2, comma_id, d[i]] = LOGIT_STRONG * 0.5
        # = precedes digits (result starts)
        for i in range(10):
            graph.A.data[2, eq_id, d[i]] = LOGIT_STRONG * 0.5
        # + is between digits
        for i in range(10):
            graph.A.data[2, d[i], plus_id] = LOGIT_STRONG * 0.3
            graph.A.data[2, plus_id, d[i]] = LOGIT_STRONG * 0.3
        # < and > relate to ordering
        for i in range(10):
            graph.A.data[2, d[i], lt_id] = LOGIT_STRONG * 0.3
            graph.A.data[2, d[i], gt_id] = LOGIT_STRONG * 0.3
        if verbose:
            print(f"  Relation 2 (operator context): ~50 edges")

    # --- Relation 3: reverse successor (predecessor) ---
    if K > 3:
        for i in range(9):
            graph.A.data[3, d[i + 1], d[i]] = LOGIT_STRONG
        graph.A.data[3, d[0], d[9]] = LOGIT_STRONG  # 0 wraps to 9
        if verbose:
            print(f"  Relation 3 (predecessor): 10 edges at logit={LOGIT_STRONG}")

    if verbose:
        P = graph.get_edge_probs()
        n_strong = (P > 0.5).sum().item()
        n_above_01 = (P > 0.1).sum().item()
        print(f"  Oracle total: {n_strong} strong edges (P>0.5), "
              f"{n_above_01} edges > 0.1")


def seed_graph_from_data(graph, sequences: list[str], verbose: bool = True):
    """Seed graph edges from training data using two strategies.

    Strategy 1 (relations 0-1): Raw token co-occurrence at offsets 1-2.
    Strategy 2 (relations 2-3): Parsed value-level transitions.
        Parses sequences to extract NUMBER values (not characters), then
        creates edges for successor relationships between DIGIT TOKENS.
        This handles the key insight: "4,5,6" means succ(4)=5, succ(5)=6
        at the semantic level, not just character-level co-occurrence.
    Remaining relations (4+): left for gradient descent.
    """
    N = graph.n_nodes
    K = graph.n_relations

    if verbose:
        print(f"  Strategy 1: token co-occurrence (relations 0-1)")

    # --- Strategy 1: raw offset co-occurrence (relations 0-1) ---
    for offset in range(1, min(3, K)):
        relation_idx = offset - 1
        pair_count = defaultdict(int)
        src_count = defaultdict(int)

        for seq in sequences:
            ids = encode(seq)
            for i in range(len(ids) - offset):
                src = ids[i]
                tgt = ids[i + offset]
                if src < N and tgt < N and src != tgt:
                    pair_count[(src, tgt)] += 1
                    src_count[src] += 1

        n_set = 0
        for (src, tgt), count in pair_count.items():
            if src_count[src] < 5:
                continue
            p = count / src_count[src]
            if p < 0.03:
                continue
            logit = math.log(p / max(1 - p, 1e-6))
            logit = max(min(logit, 5.0), -3.0)
            graph.A.data[relation_idx, src, tgt] = logit
            n_set += 1

        if verbose:
            P_k = torch.sigmoid(graph.A.data[relation_idx])
            P_k[graph.diag_mask[relation_idx]] = 0
            n_strong = (P_k > 0.5).sum().item()
            print(f"    Relation {relation_idx} (offset {offset}): "
                  f"{n_set} edges, {n_strong} strong")

    # --- Strategy 2: parsed value-level transitions ---
    if verbose:
        print(f"  Strategy 2: parsed value transitions (relations 2-3)")

    _seed_succession(graph, sequences, relation_idx=min(2, K-1), verbose=verbose)
    if K > 3:
        _seed_operator_structure(graph, sequences, relation_idx=3, verbose=verbose)


def _seed_succession(graph, sequences, relation_idx, verbose=True):
    """Detect successor relationships from comma-separated number sequences.

    Parses sequences like "4,5,6,7" to extract that 5 follows 4, 6 follows 5,
    etc. For single-digit numbers, creates direct edges between digit tokens.
    For multi-digit numbers, records the last-digit-to-first-digit transitions
    (the character-level boundary between consecutive numbers).

    This is general-purpose: it detects "token A is consistently followed by
    token B across a delimiter" without knowing what succession means.
    """
    from data import TOKEN_TO_ID
    N = graph.n_nodes

    # Parse comma-separated sequences to find CONSECUTIVE number transitions
    # Key: only count (N, N+1) pairs — direct succession, not all co-occurrence
    succ_count = defaultdict(int)  # (last_char_of_N, first_char_of_N+1) -> count
    pair_total = defaultdict(int)  # (src_digit,) -> total times it appears as predecessor

    for seq in sequences:
        if ',' not in seq:
            continue
        parts = seq.split(',')
        for i in range(len(parts) - 1):
            a_str = parts[i].strip()
            b_str = parts[i + 1].strip()
            if not a_str or not b_str:
                continue
            # Verify these are actually consecutive numbers (N, N+1)
            try:
                a_num = int(a_str)
                b_num = int(b_str)
            except ValueError:
                continue
            if b_num != a_num + 1:
                continue  # only count true successors

            # For single-digit -> single-digit: direct successor edge
            if len(a_str) == 1 and len(b_str) == 1:
                a_id = TOKEN_TO_ID.get(a_str)
                b_id = TOKEN_TO_ID.get(b_str)
                if a_id is not None and b_id is not None:
                    succ_count[(a_id, b_id)] += 1
                    pair_total[a_id] += 1
            # For multi-digit: last digit -> first digit of successor
            last_char = a_str[-1]
            first_char = b_str[0]
            if last_char.isdigit() and first_char.isdigit():
                a_id = TOKEN_TO_ID.get(last_char)
                b_id = TOKEN_TO_ID.get(first_char)
                if a_id is not None and b_id is not None:
                    if len(a_str) > 1 or len(b_str) > 1:  # avoid double-counting single-digit
                        succ_count[(a_id, b_id)] += 1
                        pair_total[a_id] += 1

    # Set edges using conditional probability: P(tgt | src, is_successor)
    n_set = 0
    for (src, tgt), count in succ_count.items():
        if src >= N or tgt >= N or src == tgt:
            continue
        if count < 3:
            continue
        # Conditional probability: when src is the last digit of a number,
        # how often is tgt the first digit of the successor?
        total = pair_total.get(src, 1)
        p = count / total
        logit = math.log(max(p, 0.05) / max(1 - p, 0.05))
        logit = max(min(logit, 4.0), -2.0)
        graph.A.data[relation_idx, src, tgt] = logit
        n_set += 1

    if verbose:
        P_k = torch.sigmoid(graph.A.data[relation_idx])
        P_k[graph.diag_mask[relation_idx]] = 0
        n_strong = (P_k > 0.5).sum().item()
        print(f"    Relation {relation_idx} (succession): "
              f"{n_set} edges, {n_strong} strong")

        from inspect_graph import node_label
        # Show digit-to-digit succession edges
        digit_start = 3  # '0' is at index 3 in ALL_TOKENS
        for d in range(9):
            src = digit_start + d
            tgt = digit_start + d + 1
            p = P_k[src, tgt].item()
            if p > 0.01:
                print(f"      {d}->{d+1}: P={p:.3f} "
                      f"(count={succ_count.get((src, tgt), 0)})")


def _seed_operator_structure(graph, sequences, relation_idx, verbose=True):
    """Detect operator-operand relationships from arithmetic sequences.

    Parses "3+5=8" to find that '+' connects two operands and '=' precedes
    a result. Creates edges from operators to the tokens they structurally
    relate to.

    General-purpose: detects "token X consistently appears between tokens
    of type Y" without knowing what addition means.
    """
    from data import TOKEN_TO_ID
    N = graph.n_nodes

    # Detect operator-context patterns
    op_context = defaultdict(lambda: defaultdict(int))

    for seq in sequences:
        ids = encode(seq)
        for i, tid in enumerate(ids):
            # For each operator token, record what appears before/after
            tok = None
            for ch in ['+', '=', '<', '>']:
                if tid == TOKEN_TO_ID.get(ch):
                    tok = ch
                    break
            if tok is None:
                continue
            # Token before operator
            if i > 0 and ids[i-1] < N:
                op_context[(tid, 'before')][ids[i-1]] += 1
            # Token after operator
            if i + 1 < len(ids) and ids[i+1] < N:
                op_context[(tid, 'after')][ids[i+1]] += 1

    n_set = 0
    for (op_id, direction), targets in op_context.items():
        total = sum(targets.values())
        for tgt_id, count in targets.items():
            if op_id >= N or tgt_id >= N or op_id == tgt_id:
                continue
            p = count / total
            if p < 0.05 or count < 3:
                continue
            logit = math.log(max(p, 0.05) / max(1 - p, 0.05))
            logit = max(min(logit, 3.0), -2.0)
            if direction == 'before':
                graph.A.data[relation_idx, tgt_id, op_id] = logit  # operand -> op
            else:
                graph.A.data[relation_idx, op_id, tgt_id] = logit  # op -> result
            n_set += 1

    if verbose:
        P_k = torch.sigmoid(graph.A.data[relation_idx])
        P_k[graph.diag_mask[relation_idx]] = 0
        n_strong = (P_k > 0.5).sum().item()
        print(f"    Relation {relation_idx} (operators): "
              f"{n_set} edges, {n_strong} strong")


def composition_closure(graph, threshold: float = 0.3, verbose: bool = False):
    """Strengthen 2-hop paths via transitive closure.

    For each relation k: if P(A->B) > threshold and P(B->C) > threshold,
    and P(A->C) < threshold, then set P(A->C) to the path strength
    min(P(A->B), P(B->C)).

    This is categorical composition: if morphisms f: A->B and g: B->C
    exist, their composite g.f: A->C must also exist.

    Uses the weakest-link criterion (min) rather than product because
    the strength of a composed path should be limited by its weakest step.
    """
    with torch.no_grad():
        P = graph.get_edge_probs()  # (K, N, N)
        n_strengthened = 0

        for k in range(graph.n_relations):
            Pk = P[k]  # (N, N)
            N = Pk.shape[0]

            # For each (i, j), find the strongest 2-hop path through any m
            # path_strength[i, m, j] = min(P[i,m], P[m,j])
            # best_path[i, j] = max_m path_strength[i, m, j]
            path_strength = torch.min(
                Pk.unsqueeze(2),  # (N, N, 1) = P[i,m]
                Pk.unsqueeze(0),  # (1, N, N) = P[m,j]
            )  # (N, N, N)
            best_path = path_strength.max(dim=1).values  # (N, N)

            # Where best_path > threshold but direct edge < threshold
            strong_path = best_path > threshold
            weak_direct = Pk < threshold
            no_self = ~graph.diag_mask[k]
            strengthen = strong_path & weak_direct & no_self

            if strengthen.any():
                target_p = best_path[strengthen].clamp(min=0.1, max=0.95)
                target_logit = torch.log(target_p / (1 - target_p))
                graph.A.data[k][strengthen] = target_logit
                n_strengthened += strengthen.sum().item()

        if verbose and n_strengthened > 0:
            print(f"  Composition closure: {n_strengthened} edges strengthened")

    return n_strengthened


def run_structural_ops(graph, epoch: int, interval: int = 20,
                       threshold: float = 0.3, verbose: bool = False):
    """Run periodic structural operations during training.

    Called every `interval` epochs. Currently runs composition closure.
    Future: chunking, colimit detection.
    """
    if epoch % interval != 0:
        return 0

    n = composition_closure(graph, threshold=threshold, verbose=verbose)
    return n


def graph_stats(graph) -> dict:
    """Quick statistics about the graph's edge distribution."""
    with torch.no_grad():
        P = graph.get_edge_probs()
        stats = {
            "max_edge": P.max().item(),
            "mean_edge": P.mean().item(),
            "n_above_0.1": (P > 0.1).sum().item(),
            "n_above_0.3": (P > 0.3).sum().item(),
            "n_above_0.5": (P > 0.5).sum().item(),
            "n_above_0.8": (P > 0.8).sum().item(),
        }
    return stats
