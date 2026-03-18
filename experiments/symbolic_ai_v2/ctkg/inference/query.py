"""
Relational query: single-hop and multi-hop graph traversal over the CTKG.

A *single-hop relational query* is a structured wrapper around `Predictor.generate`:
given a query prefix (e.g. ['succ', '5', 'eq']), generate the answer tokens and
record the chain confidence.

A *multi-hop relational query* chains n single-hop queries with the same operator:
the output digits of hop k become the input digits for hop k+1.

    multi_hop(['succ', '5', 'eq'], n_hops=2):
        hop 1: generate(['succ','5','eq']) → ['6']
        hop 2: generate(['succ','6','eq']) → ['7']
        answer: ['7']

The operator is extracted from the initial prefix (first non-digit, non-eq token).

See CTKG_ARCHITECTURE.md §Phase 6 for the full specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.inference.predict import Predictor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Result of a single-hop or multi-hop relational query.

    Attributes
    ----------
    answer_tokens:
        Generated output tokens (digits only, <eos> stripped).
    confidence:
        Product of the maximum probability at each generation step.
        Approximates the chain probability of the answer.  Range: (0, 1].
    n_hops:
        Number of hops executed.
    """

    answer_tokens: list[str] = field(default_factory=list)
    confidence: float = 1.0
    n_hops: int = 1

    def __repr__(self) -> str:
        ans = "".join(self.answer_tokens)
        return (
            f"QueryResult(answer={ans!r}, conf={self.confidence:.3f}, "
            f"n_hops={self.n_hops})"
        )


# ---------------------------------------------------------------------------
# Single-hop query
# ---------------------------------------------------------------------------

def relational_query(
    predictor: Predictor,
    query_prefix: list[str],
    eos: str = "<eos>",
    max_steps: int = 20,
) -> QueryResult:
    """Execute a single-hop relational query.

    Parameters
    ----------
    predictor:
        A fitted Predictor instance.
    query_prefix:
        Token sequence up to and including 'eq', e.g. ['succ', '5', 'eq'].
    eos:
        End-of-sequence token.
    max_steps:
        Maximum tokens to generate.

    Returns
    -------
    QueryResult with answer_tokens (digits only), chain confidence, n_hops=1.
    """
    current = list(query_prefix)
    answer: list[str] = []
    chain_conf = 1.0

    for _ in range(max_steps):
        dist = predictor.predict_next(current)
        if not dist:
            break
        next_tok = max(dist, key=lambda x: dist[x])
        max_prob = dist[next_tok]
        chain_conf *= max_prob
        if next_tok == eos:
            break
        answer.append(next_tok)
        current.append(next_tok)

    # Keep only digit tokens (strip any non-digit that sneaked in)
    digit_answer = [t for t in answer if t in "0123456789"]
    return QueryResult(answer_tokens=digit_answer, confidence=chain_conf, n_hops=1)


# ---------------------------------------------------------------------------
# Multi-hop query
# ---------------------------------------------------------------------------

def _extract_operator(
    prefix: list[str],
    predictor: Optional[Predictor] = None,
) -> Optional[str]:
    """Extract the operator atom from a query prefix.

    Uses the predictor's discovered op_atoms set if available; otherwise
    falls back to identifying the first non-digit, non-structural token
    that precedes 'eq'.  No hardcoded operator names.
    """
    known_ops = predictor._op_atoms if predictor is not None else frozenset()
    structural = {"eq", "<eos>"}
    digit_set = frozenset("0123456789")

    for tok in prefix:
        if tok in structural:
            break
        if tok in known_ops:
            return tok
        # Fallback: first token that is neither a digit nor a structural token
        if tok not in digit_set and tok not in structural:
            return tok
    return None


def multi_hop_query(
    predictor: Predictor,
    initial_prefix: list[str],
    n_hops: int,
    eos: str = "<eos>",
    max_steps: int = 20,
) -> QueryResult:
    """Execute a multi-hop relational query by chaining n single-hop queries.

    Parameters
    ----------
    predictor:
        A fitted Predictor instance.
    initial_prefix:
        Starting prefix, e.g. ['succ', '5', 'eq'].
    n_hops:
        Number of hops to chain.  n_hops=1 is equivalent to relational_query.
    eos:
        End-of-sequence token.
    max_steps:
        Maximum tokens per generation step.

    Returns
    -------
    QueryResult after n_hops applications of the same operator.

    Notes
    -----
    The operator is extracted from initial_prefix.  If no operator is found,
    returns a QueryResult with empty answer_tokens and confidence 0.0.
    """
    op = _extract_operator(initial_prefix, predictor=predictor)
    if op is None or n_hops < 1:
        return QueryResult(answer_tokens=[], confidence=0.0, n_hops=0)

    current_prefix = list(initial_prefix)
    cumulative_conf = 1.0

    for hop in range(n_hops):
        result = relational_query(predictor, current_prefix, eos=eos, max_steps=max_steps)
        cumulative_conf *= result.confidence
        if not result.answer_tokens:
            return QueryResult(
                answer_tokens=[],
                confidence=cumulative_conf,
                n_hops=hop + 1,
            )
        # Build next prefix: [op] + answer_digits + ['eq']
        current_prefix = [op] + result.answer_tokens + ["eq"]

    return QueryResult(
        answer_tokens=result.answer_tokens,
        confidence=cumulative_conf,
        n_hops=n_hops,
    )
