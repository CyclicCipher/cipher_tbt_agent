"""
Expression parser: convert flat token sequences ↔ Expr trees.

The corpus uses prefix (Polish) notation: the operator precedes its arguments.
Digit tokens ('0'..'9') are always arity-0 atoms; multi-digit numbers in the
corpus are stored as separate consecutive digit tokens.  This parser treats each
digit token as an independent atom, which is correct for the LHS (input) of
every corpus sequence.  Multi-digit RHS results are not parsed as single atoms —
rule discovery simply skips sequences whose RHS is not a well-formed prefix
expression.

ArityTable
----------
Maps operator names (strings) to their arity (int, ≥ 0).
Arity 0 = atom (leaf node).  Arity k = operator that takes k sub-expressions.
TERMINATORS are NOT in the ArityTable — they mark segment boundaries and are
consumed by the caller (split_on_terminators), not by the recursive parser.
"""
from __future__ import annotations

from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.term_algebra import Expr, atom, node


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

ArityTable = dict[str, int]   # token → number of sub-expression arguments

# Tokens that act as segment separators, never part of an expression
TERMINATORS: frozenset[str] = frozenset({
    'eq', 'step', 'ans', 'dx', '<eos>', 'carry', 'and',
})
# NOTE: 'at' is intentionally NOT a terminator.  It appears inline in eval
# sequences as a structural keyword: eval A x B at C eq Y.  Treating 'at' as
# a terminator would split the eval arguments across segments, preventing
# _cata_predict from building eval(A, x, B, at, C) as a 5-arg tree.

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _ParseError(Exception):
    """Raised internally when a prefix parse fails."""


def _try_parse(tokens: list[str],
               arities: ArityTable,
               pos: int) -> tuple[Expr, int]:
    """
    Recursive-descent prefix parser.

    Returns (parsed_Expr, next_position) or raises _ParseError.
    Raises _ParseError if the token at pos has no known arity.
    """
    if pos >= len(tokens):
        raise _ParseError(f"unexpected end of token list at position {pos}")

    tok = tokens[pos]

    if tok in TERMINATORS:
        raise _ParseError(f"terminator '{tok}' encountered mid-parse at position {pos}")

    arity = arities.get(tok)
    if arity is None:
        raise _ParseError(f"unknown arity for token '{tok}'")

    if arity == 0:
        return atom(tok), pos + 1

    args: list[Expr] = []
    cur = pos + 1
    for i in range(arity):
        try:
            arg, cur = _try_parse(tokens, arities, cur)
        except _ParseError as e:
            raise _ParseError(
                f"failed to parse arg {i} of '{tok}': {e}"
            ) from e
        args.append(arg)

    return node(tok, *args), cur


def _split_on_terminators(
    tokens: list[str],
    terminators: frozenset[str],
) -> list[list[str]]:
    """Split a token sequence on terminator tokens, returning non-empty segments."""
    segments: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if tok in terminators:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        segments.append(current)
    return segments



# ---------------------------------------------------------------------------
# Public API — parse / unparse
# ---------------------------------------------------------------------------

def parse(tokens: list[str], arities: ArityTable) -> Optional[Expr]:
    """
    Parse a prefix token sequence into an Expr tree.

    Returns the parsed Expr if the ENTIRE token list is consumed, else None.
    Unknown tokens cause None to be returned (no exception).

    Usage: pass the segment BETWEEN terminators, not the full sequence.
    To parse a complete sequence, call split_on_terminators first.
    """
    try:
        expr, consumed = _try_parse(tokens, arities, 0)
    except _ParseError:
        return None
    if consumed != len(tokens):
        return None   # trailing tokens — ambiguous or wrong arity
    return expr


def parse_full(tokens: list[str],
               arities: ArityTable) -> tuple[Optional[Expr], Optional[Expr]]:
    """
    Parse a complete corpus sequence `[input_tokens 'eq' output_tokens]`.

    Returns (input_expr, output_expr).  Either may be None if the corresponding
    segment is not parseable as a complete prefix expression.

    Handles the common case where the sequence contains exactly one 'eq'.
    Sequences with 'step'/'ans'/'dx' terminators are split on ALL terminators
    and (input, output) are taken as the first and last segment respectively.
    """
    segs = _split_on_terminators(tokens, TERMINATORS)
    if len(segs) < 2:
        return None, None
    input_expr  = parse(segs[0],  arities)
    output_expr = parse(segs[-1], arities)
    return input_expr, output_expr


def unparse(expr: Expr) -> list[str]:
    """
    Unparse an Expr tree into a prefix token sequence.

    Pre-order traversal: emit the head, then recursively unparse each argument.
    Inverse of parse: unparse(parse(seq, arities)) == seq for any parseable seq.
    """
    from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
    tokens: list[str] = [TOKEN_GRAPH.decode(expr.head)]
    for arg in expr.args:
        tokens.extend(unparse(arg))
    return tokens


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def split_on_terminators(
    tokens: list[str],
    terminators: Optional[frozenset[str]] = None,
) -> list[list[str]]:
    """Public wrapper around _split_on_terminators."""
    return _split_on_terminators(tokens, terminators or TERMINATORS)


def normalize_surface(tokens: list[str], norm_map: dict[str, str]) -> list[str]:
    """Apply a surface-form normalization map to a token sequence.

    Each token is replaced by its canonical form as given by norm_map.
    Tokens not in the map are left unchanged.

    This is the only layer where surface forms are touched; every layer above
    this function sees only canonical forms.

    Parameters
    ----------
    tokens   : raw token sequence (e.g. from the corpus)
    norm_map : surface_form → canonical_form (e.g. {'five': '5', 'cinq': '5'})

    Returns the normalized token sequence.

    Example
    -------
    >>> normalize_surface(['add', 'five', 'three', 'eq', '8'], {'five': '5', 'three': '3'})
    ['add', '5', '3', 'eq', '8']
    """
    return [norm_map.get(tok, tok) for tok in tokens]
