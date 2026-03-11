"""variable_binding.py -- Phase 17b / Phase 19: Symbolic regression and variable binding.

Gary Marcus's critique: algebraic generalisation requires universally quantified
variables.  This module adds the minimum machinery for that.

Design principle (representation independence):
  Rules operate on GRAPH-DERIVED RANKS, not on atom names.  The successor
  endofunctor map {atom_0 → atom_1, atom_1 → atom_2, ...} IS the total order
  on atoms.  Rank is the position in this chain, derivable from graph topology
  alone, independent of atom names.  This means the algorithm works identically
  for math ('0', '1', '2', ...), Danganronpa ('chapter_1', 'chapter_2', ...),
  music ('C4', 'D4', 'E4', ...) or any other ordered domain.

Three components:
  1. RelationalRule       -- structure-only rules (identity, constant, commutative)
  2. AlgebraicRule        -- arithmetic rules fit on rank pairs (Phase 17b)
  3. Chain detection      -- _is_chain, _build_rank_map (Phase 19 Level 1 gate)

Hypothesis space for AlgebraicRule:
  Programs are enumerated by a grammar-based synthesizer (_unary_programs,
  _binary_programs).  Any formula expressible in the arithmetic grammar is
  found automatically without modifying the search code.

  Grammar (unary):
    M = N + k | k*N | N^k | isqrt(N) | N//k | ...

  Grammar (binary):
    M = N1+N2 | N1-N2 | N1*N2 | N1^N2 | N1//N2
      | k*(N1±N2) | k*N1*N2 | N1*N2//k | N1^p*N2^q | k*N1^p*N2^q | ...

Search strategy: enumerate grammar programs in complexity order (simpler
programs first); accept the first with ZERO residual on ALL training pairs.
This is principled program synthesis — complete up to grammar depth and
guaranteed to find the simplest consistent formula.

Public API:
  RelationalRule                              -- dataclass
  AlgebraicRule                               -- dataclass
  fit_relational_rule(op, ef_map, mg)         -- fit structure-only rule
  fit_rule(op, ef_map, mg, rank_map)          -- fit arithmetic rule on ranks
  build_variable_binding(mg, topo)            -- run all discovery; stores on mg
  predict_via_relational_rule(mg, atom_seq)   -- relational rule prediction
  predict_via_variable_binding(mg, ctx_id, etype) -- unification + rank eval
  predict_via_frame_match(mg, atom_values)    -- raw buffer frame matching
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Callable

from ..core.morphism import MorphismGraph, Atom


# ── RelationalRule ─────────────────────────────────────────────────────────────

@dataclass
class RelationalRule:
    """A structural rule over slot positions in a composition frame.

    Requires zero knowledge of atom contents -- operates purely on atom
    identity (same/different slot positions).  Sits BELOW AlgebraicRule
    in the hierarchy: needs less information and is more reliable.

    Attributes
    ----------
    op          : operator atom value (e.g. 'id', 'reflect')
    arity       : number of argument slots (1 or 2)
    relation    : 'identity'    -- f(X) = X  (result == arg)
                  'constant'   -- f(X) = C  (same result for all args)
                  'commutative'-- f(a,b) = f(b,a)  (result swaps with args)
    constant_id : atom ID of the fixed result (only for 'constant')
    evidence    : number of training frames confirming this rule
    """
    op:          str
    arity:       int
    relation:    str          # 'identity' | 'constant' | 'commutative'
    constant_id: Optional[int] = None
    evidence:    int = 0

    @property
    def confidence(self) -> float:
        return self.evidence / (self.evidence + 1)


# ── AlgebraicRule ──────────────────────────────────────────────────────────────

@dataclass
class AlgebraicRule:
    """A discovered algebraic rule with universally quantified variables.

    All arithmetic operates on GRAPH-DERIVED RANKS, not atom names.
    fn(rank_arg) -> rank_result  (unary)
    fn(rank_arg1, rank_arg2) -> rank_result  (binary)

    Attributes
    ----------
    op         : operator atom value (e.g. 'succ', 'add')
    arity      : 1 = unary, 2 = binary
    formula    : human-readable string (e.g. 'M = N + 1')
    fn         : callable over ranks
    evidence   : number of training rank-pairs confirming this rule
    fixed      : if True, falsification raises SheafViolation (homeostatic prior)
    """
    op:       str
    arity:    int
    formula:  str
    fn:       Callable
    evidence: int
    fixed:    bool = False

    @property
    def confidence(self) -> float:
        return self.evidence / (self.evidence + 1)


# ── Chain detection ────────────────────────────────────────────────────────────

def _is_chain(ef_map: dict) -> bool:
    """Return True if ef_map is a single connected acyclic chain.

    A chain: exactly one root (total order with unique minimum), every atom
    appears at most once as a VALUE (no branching), and the graph is acyclic.
    The forest case (multiple disconnected chains) is REJECTED because multiple
    roots would each receive rank 0, producing an ambiguous ordering.

    Used to detect operators like 'succ' that impose a total order on their
    argument atoms.  Only valid for unary ef_maps ({arg_id: result_id}).
    Binary maps always return False.
    """
    if not ef_map:
        return False
    # Binary operators are not chains.
    if isinstance(next(iter(ef_map)), tuple):
        return False
    # No two keys may share the same value (no branching).
    vals = list(ef_map.values())
    if len(set(vals)) != len(vals):
        return False
    # Exactly one root (single connected component, not a forest).
    keys    = set(ef_map.keys())
    val_set = set(vals)
    roots   = keys - val_set
    if len(roots) != 1:
        return False  # 0 roots = cycle; >1 roots = forest — both rejected
    # Walk the single chain to verify no cycle.
    root    = next(iter(roots))
    visited: set[int] = set()
    node = root
    while node in ef_map:
        if node in visited:
            return False  # cycle
        visited.add(node)
        node = ef_map[node]
    return True


def _build_rank_map(ef_map: dict) -> tuple[dict[int, int], dict[int, int]]:
    """Walk a chain endofunctor map and assign integer ranks to each atom.

    Starts from the chain root (atom that is a key but not a value) and walks
    forward, assigning rank 0, 1, 2, ...

    Returns
    -------
    (rank_map, inv_rank_map)
      rank_map    : {atom_id → rank}  position in the chain
      inv_rank_map: {rank → atom_id}  inverse lookup
    """
    keys    = set(ef_map.keys())
    val_set = set(ef_map.values())
    roots   = keys - val_set

    rank_map:     dict[int, int] = {}
    inv_rank_map: dict[int, int] = {}

    for root in roots:
        rank = 0
        node: Optional[int] = root
        while node is not None:
            rank_map[node]     = rank
            inv_rank_map[rank] = node
            rank += 1
            node = ef_map.get(node)

    return rank_map, inv_rank_map


# ── Rank helpers ───────────────────────────────────────────────────────────────

def _atom_rank(
    mg: MorphismGraph,
    atom_id: int,
    rank_map: dict[int, int],
) -> Optional[int]:
    """Return the rank of atom_id.

    Priority:
      1. Graph-structural rank from rank_map (representation-independent;
         derived from chain-structured endofunctors like 'succ').
      2. int(atom.value) fallback for numeric-named atoms.  This applies
         when the training corpus is SPARSE (doesn't form a complete chain).
         In a proper deployment, training data should include a complete
         chain, making the graph-structural path fully sufficient.
    """
    rank = rank_map.get(atom_id)
    if rank is not None:
        return rank
    sym = mg.symbols[atom_id]
    if isinstance(sym, Atom):
        try:
            return int(sym.value)
        except (ValueError, TypeError):
            pass
    return None


def _value_rank(
    s: str,
    mg: MorphismGraph,
    rank_map: dict[int, int],
) -> Optional[int]:
    """Convert a string atom value to its rank.  Returns None on failure."""
    atom_id = mg.atoms.get(s)
    if atom_id is None:
        return None
    return _atom_rank(mg, atom_id, rank_map)


# ── Grammar-based program synthesis ────────────────────────────────────────────
#
# Programs are enumerated by complexity (number of AST nodes).  Any formula
# expressible in the arithmetic grammar is found automatically — no domain
# knowledge is encoded in the search order beyond "simpler formulas first".
#
# Grammar (unary):   M := N + k | k*N | N^k | isqrt(N) | N//k | ...
# Grammar (binary):  M := N1+N2 | N1-N2 | N1*N2 | k*N1*N2 | N1*N2^p//k | ...
#
# The enumerate_* generators yield (formula_str, callable) in complexity order.
# _fit_unary/_fit_binary accept the first formula with zero residual on ALL data.
# This is the standard "enumerate shortest consistent program" approach from
# program synthesis / inductive logic programming.

def _enumerate_unary(k_max: int = 20):
    """Yield (formula, fn) for unary programs M = f(N), simplest first."""
    # Complexity 1: shift by integer constant (covers succ k=+1, pred k=-1)
    for k in list(range(1, k_max + 1)) + list(range(-1, -k_max - 1, -1)):
        yield f"M = N + {k}", lambda n, _k=k: n + _k

    # Complexity 1: scale by positive integer
    for k in range(2, k_max + 1):
        yield f"M = {k} * N", lambda n, _k=k: _k * n

    # Complexity 1: integer power (guarded against overflow)
    for k in range(2, 7):
        yield f"M = N ^ {k}", lambda n, _k=k: n ** _k if n >= 0 else None

    # Complexity 1: integer square root  (inverse of N^2)
    yield "M = isqrt(N)", lambda n: math.isqrt(n) if n >= 0 else None

    # Complexity 1: floor division by constant
    for k in range(2, k_max + 1):
        yield f"M = N // {k}", lambda n, _k=k: n // _k if n >= 0 else None

    # Complexity 2: scale then shift  (M = k*N + j)
    for k in range(2, 6):
        for j in range(1, 6):
            yield f"M = {k}*N + {j}", lambda n, _k=k, _j=j: _k * n + _j
            yield f"M = {k}*N - {j}", lambda n, _k=k, _j=j: _k * n - _j


def _enumerate_binary(k_max: int = 20):
    """Yield (formula, fn) for binary programs M = f(N1, N2), simplest first."""
    # Complexity 0: two-variable arithmetic
    yield "M = N1 + N2",  lambda n1, n2: n1 + n2
    yield "M = N1 - N2",  lambda n1, n2: n1 - n2
    yield "M = N2 - N1",  lambda n1, n2: n2 - n1
    yield "M = N1 * N2",  lambda n1, n2: n1 * n2
    yield "M = N1 ^ N2",  lambda n1, n2: n1 ** n2 if n1 >= 0 and 0 <= n2 <= 9 else None
    yield "M = N1 // N2", lambda n1, n2: n1 // n2 if n2 != 0 else None

    # Complexity 1: multiply by constant k  (e.g. pe = 10*rho*h)
    for k in range(2, k_max + 1):
        yield f"M = {k} * N1 * N2",   lambda n1, n2, _k=k: _k * n1 * n2
        yield f"M = {k} * (N1 + N2)", lambda n1, n2, _k=k: _k * (n1 + n2)
        yield f"M = {k} * (N1 - N2)", lambda n1, n2, _k=k: _k * (n1 - n2)

    # Complexity 1: floor-divide product by constant  (e.g. ke = rho*v²//2)
    for k in range(2, k_max + 1):
        yield f"M = N1 * N2 // {k}", lambda n1, n2, _k=k: n1 * n2 // _k

    # Complexity 1: additive constant
    for k in range(1, k_max + 1):
        yield f"M = N1 * N2 + {k}", lambda n1, n2, _k=k: n1 * n2 + _k
        yield f"M = N1 * N2 - {k}", lambda n1, n2, _k=k: n1 * n2 - _k
        yield f"M = N1 + N2 + {k}", lambda n1, n2, _k=k: n1 + n2 + _k

    # Complexity 2: one variable raised to a small power
    for p in range(2, 5):
        yield f"M = N1 * N2^{p}", lambda n1, n2, _p=p: n1 * n2 ** _p if n2 >= 0 else None
        yield f"M = N1^{p} * N2", lambda n1, n2, _p=p: n1 ** _p * n2 if n1 >= 0 else None
        # with constant divisor  (e.g. ke = rho*v²//2)
        for k in range(2, 11):
            yield (f"M = N1 * N2^{p} // {k}",
                   lambda n1, n2, _p=p, _k=k: n1 * n2 ** _p // _k if n2 >= 0 else None)
            yield (f"M = {k} * N1 * N2^{p}",
                   lambda n1, n2, _p=p, _k=k: _k * n1 * n2 ** _p if n2 >= 0 else None)


def _enumerate_ternary(k_max: int = 20):
    """Yield (formula, fn) for ternary programs M = f(N1, N2, N3), simplest first.

    N1, N2, N3 are the three argument ranks; M is the result rank.
    Covers the Bernoulli family: M = isqrt(N1 + N2^p - N3), etc.
    """
    # Complexity 0: linear combinations
    yield "M = N1 + N2 - N3", lambda n1, n2, n3: n1 + n2 - n3
    yield "M = N1 - N2 + N3", lambda n1, n2, n3: n1 - n2 + n3
    yield "M = N1 + N2 + N3", lambda n1, n2, n3: n1 + n2 + n3
    yield "M = N1 * N2 * N3", lambda n1, n2, n3: n1 * n2 * n3

    # Complexity 1: isqrt applied to linear combination
    yield "M = isqrt(N1 + N2 - N3)", lambda n1, n2, n3: (
        math.isqrt(n1 + n2 - n3) if n1 + n2 - n3 >= 0 else None)
    yield "M = isqrt(N1 - N2 + N3)", lambda n1, n2, n3: (
        math.isqrt(n1 - n2 + n3) if n1 - n2 + n3 >= 0 else None)

    # Complexity 2: quadratic in one argument  (covers Bernoulli P + v² = const)
    for p in range(2, 5):
        yield (f"M = isqrt(N1 + N2^{p} - N3)",
               lambda n1, n2, n3, _p=p: (
                   math.isqrt(n1 + n2 ** _p - n3)
                   if n2 >= 0 and n1 + n2 ** _p - n3 >= 0 else None))
        yield (f"M = isqrt(N1 + N3^{p} - N2)",
               lambda n1, n2, n3, _p=p: (
                   math.isqrt(n1 + n3 ** _p - n2)
                   if n3 >= 0 and n1 + n3 ** _p - n2 >= 0 else None))
        yield (f"M = isqrt(N2^{p} + N3 - N1)",
               lambda n1, n2, n3, _p=p: (
                   math.isqrt(n2 ** _p + n3 - n1)
                   if n2 >= 0 and n2 ** _p + n3 - n1 >= 0 else None))
        # with constant divisor
        for k in range(2, 6):
            yield (f"M = isqrt(N1 + N2^{p} - N3) // {k}",
                   lambda n1, n2, n3, _p=p, _k=k: (
                       math.isqrt(n1 + n2 ** _p - n3) // _k
                       if n2 >= 0 and n1 + n2 ** _p - n3 >= 0 else None))


def _fit_programs(examples, enumerator, min_examples: int = 2):
    """Generic program synthesis: accept first formula with zero residual.

    examples : list of input-tuples (values vary by arity)
               unary:   [(n, m), ...]
               binary:  [(n1, n2, m), ...]
               ternary: [(n1, n2, n3, m), ...]
    enumerator: callable returning an iterator of (formula, fn) pairs.
    Returns (formula, fn) or None.
    """
    if len(examples) < min_examples:
        return None
    for formula, fn in enumerator():
        try:
            if all(fn(*ex[:-1]) == ex[-1] for ex in examples):
                return formula, fn
        except Exception:
            continue
    return None


def _fit_unary(pairs: list[tuple[int, int]]) -> Optional[tuple[str, Callable]]:
    """Fit a unary program M = f(N) via grammar-based synthesis."""
    return _fit_programs(pairs, _enumerate_unary, min_examples=2)


def _fit_binary(triples: list[tuple[int, int, int]]) -> Optional[tuple[str, Callable]]:
    """Fit a binary program M = f(N1, N2) via grammar-based synthesis."""
    return _fit_programs(triples, _enumerate_binary, min_examples=3)


def _fit_ternary(quads: list[tuple[int, int, int, int]]) -> Optional[tuple[str, Callable]]:
    """Fit a ternary program M = f(N1, N2, N3) via grammar-based synthesis."""
    return _fit_programs(quads, _enumerate_ternary, min_examples=3)


# ── RelationalRule discovery ───────────────────────────────────────────────────

def fit_relational_rule(
    op:     str,
    ef_map: dict,
    mg:     MorphismGraph,
) -> Optional[RelationalRule]:
    """Fit a structure-only RelationalRule to one operator's endofunctor map.

    Checks identity, constant, and commutativity using only atom IDs -- zero
    content knowledge required.  Requires at least 2 entries.
    """
    if not ef_map or len(ef_map) < 2:
        return None

    first_key = next(iter(ef_map))
    is_binary  = isinstance(first_key, tuple) and len(first_key) == 2
    is_ternary = isinstance(first_key, tuple) and len(first_key) == 3

    if is_ternary:
        # No structural rules discovered yet for ternary operators; skip.
        return None
    elif not is_binary:
        # Unary: check identity (result == arg for ALL pairs)
        if all(res_id == arg_id for arg_id, res_id in ef_map.items()):
            return RelationalRule(op=op, arity=1, relation='identity',
                                  evidence=len(ef_map))
        # Unary: check constant (same result for ALL args)
        results = set(ef_map.values())
        if len(results) == 1:
            return RelationalRule(op=op, arity=1, relation='constant',
                                  constant_id=next(iter(results)),
                                  evidence=len(ef_map))
    else:
        # Binary: check commutativity (f(a,b) = f(b,a) for all pairs where both exist)
        commutative_pairs = 0
        total_pairs = 0
        for (a1, a2), res in ef_map.items():
            swapped = ef_map.get((a2, a1))
            if swapped is not None:
                total_pairs += 1
                if swapped == res:
                    commutative_pairs += 1
        if total_pairs >= 2 and commutative_pairs == total_pairs:
            return RelationalRule(op=op, arity=2, relation='commutative',
                                  evidence=len(ef_map))

    return None


# ── AlgebraicRule discovery ────────────────────────────────────────────────────

def fit_rule(
    op:       str,
    ef_map:   dict,
    mg:       MorphismGraph,
    rank_map: Optional[dict[int, int]] = None,
) -> Optional[AlgebraicRule]:
    """Fit an AlgebraicRule to one operator's endofunctor map.

    Uses graph-derived ranks, not atom names.  Only atoms in rank_map are
    used for fitting; if fewer than min_evidence rank-mapped pairs exist,
    returns None.

    rank_map defaults to mg._rank_map if not supplied.
    """
    if not ef_map:
        return None

    if rank_map is None:
        rank_map = getattr(mg, '_rank_map', {})

    first_key = next(iter(ef_map))
    is_binary  = isinstance(first_key, tuple) and len(first_key) == 2
    is_ternary = isinstance(first_key, tuple) and len(first_key) == 3

    if is_ternary:
        quads: list[tuple[int, int, int, int]] = []
        for (a1_id, a2_id, a3_id), res_id in ef_map.items():
            n1 = _atom_rank(mg, a1_id,  rank_map)
            n2 = _atom_rank(mg, a2_id,  rank_map)
            n3 = _atom_rank(mg, a3_id,  rank_map)
            m  = _atom_rank(mg, res_id, rank_map)
            if n1 is not None and n2 is not None and n3 is not None and m is not None:
                quads.append((n1, n2, n3, m))
        result = _fit_ternary(quads)
        if result:
            formula, fn = result
            return AlgebraicRule(op=op, arity=3, formula=formula,
                                 fn=fn, evidence=len(quads))

    elif is_binary:
        triples: list[tuple[int, int, int]] = []
        for (a1_id, a2_id), res_id in ef_map.items():
            n1 = _atom_rank(mg, a1_id,  rank_map)
            n2 = _atom_rank(mg, a2_id,  rank_map)
            m  = _atom_rank(mg, res_id, rank_map)
            if n1 is not None and n2 is not None and m is not None:
                triples.append((n1, n2, m))
        result = _fit_binary(triples)
        if result:
            formula, fn = result
            return AlgebraicRule(op=op, arity=2, formula=formula,
                                 fn=fn, evidence=len(triples))

    else:
        pairs: list[tuple[int, int]] = []
        for arg_id, res_id in ef_map.items():
            n = _atom_rank(mg, arg_id,  rank_map)
            m = _atom_rank(mg, res_id, rank_map)
            if n is not None and m is not None:
                pairs.append((n, m))
        result = _fit_unary(pairs)
        if result:
            formula, fn = result
            return AlgebraicRule(op=op, arity=1, formula=formula,
                                 fn=fn, evidence=len(pairs))

    return None


# ── Public API ─────────────────────────────────────────────────────────────────

def build_variable_binding(
    mg:   MorphismGraph,
    topo,
) -> dict[str, AlgebraicRule]:
    """Run symbolic regression on all endofunctor maps.

    Step 1: detect chain-structured unary endofunctors and build the global
    rank_map.  An atom 'has ordinal type' iff it appears in this map.
    Step 2: fit RelationalRules (structure-only, no content needed).
    Step 3: fit AlgebraicRules (arithmetic on ranks, gated by rank_map membership).

    After this call:
      mg._rank_map         = {atom_id: rank}
      mg._inv_rank_map     = {rank: atom_id}
      mg._relational_rules = {op_name: RelationalRule}
      mg._algebraic_rules  = {op_name: AlgebraicRule}

    Returns mg._algebraic_rules.
    """
    if not getattr(mg, '_endofunctors', None):
        from .rule_store import build_rule_store
        build_rule_store(mg, topo)

    # Step 1: build unified rank map from chain-structured unary endofunctors.
    # Only add a chain if its ranks don't conflict with already-assigned ranks.
    # Example: succ gives atom('0')→0 and pred gives atom('0')→24 — if both
    # were merged, the map would be corrupted.  The first (longest) chain wins.
    rank_map:     dict[int, int] = {}
    inv_rank_map: dict[int, int] = {}

    # Sort by chain length (longest first) so the most informative chain wins.
    chain_ops = [
        (op, ef_map)
        for op, ef_map in mg._endofunctors.items()
        if ef_map and _is_chain(ef_map)
    ]
    chain_ops.sort(key=lambda kv: -len(kv[1]))

    for _op, ef_map in chain_ops:
        rm, irm = _build_rank_map(ef_map)
        # Skip if any atom in this chain is already assigned a DIFFERENT rank.
        if any(
            atom_id in rank_map and rank_map[atom_id] != rank
            for atom_id, rank in rm.items()
        ):
            continue
        rank_map.update(rm)
        inv_rank_map.update(irm)

    mg._rank_map     = rank_map
    mg._inv_rank_map = inv_rank_map

    # Step 2: fit relational rules (structure-only).
    relational_rules: dict[str, RelationalRule] = {}
    for op, ef_map in mg._endofunctors.items():
        rule = fit_relational_rule(op, ef_map, mg)
        if rule is not None:
            relational_rules[op] = rule
    mg._relational_rules = relational_rules

    # Step 3: fit algebraic rules (rank-based arithmetic).
    rules: dict[str, AlgebraicRule] = {}
    for op, ef_map in mg._endofunctors.items():
        rule = fit_rule(op, ef_map, mg, rank_map)
        if rule is not None:
            rules[op] = rule

    mg._algebraic_rules = rules
    return rules


def _result_atom(
    mg:          MorphismGraph,
    result_rank: int,
    inv_rank_map: dict[int, int],
    confidence:  float,
    max_rank:    int = 10_000,
) -> dict[int, float]:
    """Convert a result rank to {atom_id: confidence}.

    Lookup order:
      1. inv_rank_map (graph-structural chain, representation-independent)
      2. mg.atoms.get(str(result_rank)) — numeric-name convention fallback
      3. mg.get_or_create_atom() — create a predicted atom if still not found
    """
    if abs(result_rank) > max_rank:
        return {}
    result_id = inv_rank_map.get(result_rank)
    if result_id is None:
        # Numeric-name fallback: str(rank) may be an existing atom's name.
        result_id = mg.atoms.get(str(result_rank))
    if result_id is None:
        # Novel result: create a predicted atom.
        result_id = mg.get_or_create_atom(str(result_rank), coarse_type='num')
    return {result_id: confidence}


def predict_via_relational_rule(
    mg:        MorphismGraph,
    atom_seq:  list[tuple[int, str]],
) -> dict[int, float]:
    """Apply a discovered RelationalRule to the atom sequence.

    atom_seq: [(atom_id, atom_value_str), ...] ending with 'eq'.
    Returns {result_atom_id: confidence} or {}.
    """
    relational_rules = getattr(mg, '_relational_rules', None)
    if not relational_rules or not atom_seq:
        return {}
    if atom_seq[-1][1] != 'eq':
        return {}

    n = len(atom_seq)

    # Unary frame: [op, arg, eq]
    if n == 3:
        op_val = atom_seq[0][1]
        arg_id = atom_seq[1][0]
        rule   = relational_rules.get(op_val)
        if rule is not None and rule.arity == 1:
            if rule.relation == 'identity':
                return {arg_id: rule.confidence}
            if rule.relation == 'constant' and rule.constant_id is not None:
                return {rule.constant_id: rule.confidence}

    # Binary frame: [op, arg1, arg2, eq]
    if n == 4:
        op_val  = atom_seq[0][1]
        arg1_id = atom_seq[1][0]
        arg2_id = atom_seq[2][0]
        rule    = relational_rules.get(op_val)
        if rule is not None and rule.arity == 2:
            if rule.relation == 'commutative':
                # If (arg2, arg1) is in the endofunctor, use its result.
                ef_map = getattr(mg, '_endofunctors', {}).get(op_val, {})
                swapped = ef_map.get((arg2_id, arg1_id))
                if swapped is not None:
                    return {swapped: rule.confidence}

    return {}


def predict_via_variable_binding(
    mg:      MorphismGraph,
    ctx_id:  int,
    etype:   int,
) -> dict[int, float]:
    """Unification + algebraic rule evaluation on rank-based composition context.

    Decomposes ctx_id to its constituent atom sequence, matches against known
    operator frames, applies the discovered algebraic rule in rank space, and
    returns {result_atom_id: confidence}.

    Returns {} if:
      - no algebraic rules have been discovered
      - ctx_id does not match any known frame
      - the rule evaluation raises an exception
    """
    rules = getattr(mg, '_algebraic_rules', None)
    if not rules:
        return {}

    rank_map     = getattr(mg, '_rank_map',     {})
    inv_rank_map = getattr(mg, '_inv_rank_map', {})

    # Decompose ctx_id to its constituent atoms (left-to-right)
    from .rule_store import _decompose
    atom_seq: list[tuple[int, str]] = []
    _decompose(mg, ctx_id, atom_seq)

    if not atom_seq:
        return {}

    n = len(atom_seq)

    # ── Ternary frame without 'eq': [op, arg1, arg2, arg3] ────────────────────
    # Compact operators like 'bernoulli' have no 'eq' separator;
    # the result immediately follows the last argument.
    if n == 4 and atom_seq[-1][1] != 'eq':
        op_val  = atom_seq[0][1]
        arg1_id = atom_seq[1][0]
        arg2_id = atom_seq[2][0]
        arg3_id = atom_seq[3][0]
        rule    = rules.get(op_val)
        if rule is not None and rule.arity == 3:
            n1 = _atom_rank(mg, arg1_id, rank_map)
            n2 = _atom_rank(mg, arg2_id, rank_map)
            n3 = _atom_rank(mg, arg3_id, rank_map)
            if n1 is not None and n2 is not None and n3 is not None:
                try:
                    result_rank = rule.fn(n1, n2, n3)
                    return _result_atom(mg, result_rank, inv_rank_map, rule.confidence)
                except Exception:
                    pass

    # Frame must end with the equality marker for eq-delimited frames
    if atom_seq[-1][1] != 'eq':
        return {}

    # ── Unary frame: [op, arg, eq] ─────────────────────────────────────────────
    if n == 3:
        op_val = atom_seq[0][1]
        arg_id = atom_seq[1][0]
        rule   = rules.get(op_val)
        if rule is not None and rule.arity == 1:
            arg_rank = _atom_rank(mg, arg_id, rank_map)
            if arg_rank is not None:
                try:
                    result_rank = rule.fn(arg_rank)
                    return _result_atom(mg, result_rank, inv_rank_map, rule.confidence)
                except Exception:
                    pass

    # ── Binary frame: [op, arg1, arg2, eq] ────────────────────────────────────
    if n == 4:
        op_val  = atom_seq[0][1]
        arg1_id = atom_seq[1][0]
        arg2_id = atom_seq[2][0]
        rule    = rules.get(op_val)
        if rule is not None and rule.arity == 2:
            n1 = _atom_rank(mg, arg1_id, rank_map)
            n2 = _atom_rank(mg, arg2_id, rank_map)
            if n1 is not None and n2 is not None:
                try:
                    result_rank = rule.fn(n1, n2)
                    return _result_atom(mg, result_rank, inv_rank_map, rule.confidence)
                except Exception:
                    pass

    return {}


def predict_via_frame_match(
    mg:          MorphismGraph,
    atom_values: list[str],
) -> dict[int, float]:
    """Variable binding directly on raw atom-value strings (frame matching).

    Fallback used when the multilevel composition context has collapsed.
    Uses mg._rank_map to convert atom strings to ranks (representation-independent).

    atom_values: the FULL atom buffer (any length).  The function checks the
      LAST 3, LAST 4, and LAST 5 elements for known operator frames, so it
      works whether the buffer contains a raw [op arg eq] suffix or a longer
      sequence like [what is add N1 N2 eq].

    Returns {result_atom_id: confidence} or {} if no frame matches.
    """
    rules = getattr(mg, '_algebraic_rules', None)
    if not rules or not atom_values:
        return {}

    rank_map     = getattr(mg, '_rank_map',     {})
    inv_rank_map = getattr(mg, '_inv_rank_map', {})

    # ── Ternary without 'eq': check last 4 atoms [op, a1, a2, a3] ─────────────
    if len(atom_values) >= 4 and atom_values[-1] != 'eq':
        tail4 = atom_values[-4:]
        op_val = tail4[0]
        rule   = rules.get(op_val)
        if rule is not None and rule.arity == 3:
            n1 = _value_rank(tail4[1], mg, rank_map)
            n2 = _value_rank(tail4[2], mg, rank_map)
            n3 = _value_rank(tail4[3], mg, rank_map)
            if n1 is not None and n2 is not None and n3 is not None:
                try:
                    result_rank = rule.fn(n1, n2, n3)
                    return _result_atom(mg, result_rank, inv_rank_map, rule.confidence)
                except Exception:
                    pass

    if atom_values[-1] != 'eq':
        return {}

    # ── Unary frame: check last 3 atoms [op, arg, eq] ─────────────────────────
    if len(atom_values) >= 3:
        tail3  = atom_values[-3:]
        op_val = tail3[0]
        rule   = rules.get(op_val)
        if rule is not None and rule.arity == 1:
            arg_rank = _value_rank(tail3[1], mg, rank_map)
            if arg_rank is not None:
                try:
                    result_rank = rule.fn(arg_rank)
                    return _result_atom(mg, result_rank, inv_rank_map, rule.confidence)
                except Exception:
                    pass

    # ── Binary frame: check last 4 atoms [op, arg1, arg2, eq] ─────────────────
    if len(atom_values) >= 4:
        tail4    = atom_values[-4:]
        op_val   = tail4[0]
        arg1_str = tail4[1]
        arg2_str = tail4[2]
        rule     = rules.get(op_val)
        if rule is not None and rule.arity == 2:
            n1 = _value_rank(arg1_str, mg, rank_map)
            n2 = _value_rank(arg2_str, mg, rank_map)
            if n1 is not None and n2 is not None:
                try:
                    result_rank = rule.fn(n1, n2)
                    return _result_atom(mg, result_rank, inv_rank_map, rule.confidence)
                except Exception:
                    pass

    # ── Variadic fold: [..., vadd, N1, ..., Nk, eq] for k >= 3 ───────────────
    # Applies the binary vadd/add rule iteratively: result = N1 + N2 + ... + Nk.
    # The binary frame above handles k == 2; this handles k >= 3.
    if 'vadd' in atom_values:
        vadd_pos = len(atom_values) - 1 - list(reversed(atom_values)).index('vadd')
        between = atom_values[vadd_pos + 1 : -1]   # tokens between vadd and eq
        if len(between) >= 3:
            num_ranks = [_value_rank(v, mg, rank_map) for v in between]
            if all(r is not None for r in num_ranks):
                vadd_rule = rules.get('vadd') or rules.get('add')
                if vadd_rule is not None and vadd_rule.arity == 2:
                    try:
                        acc = num_ranks[0]
                        for r in num_ranks[1:]:
                            acc = vadd_rule.fn(acc, r)
                        out = _result_atom(mg, acc, inv_rank_map, vadd_rule.confidence)
                        if out:
                            return out
                    except Exception:
                        pass

    # ── NL numeral scan: implicit-operator fallback for word problems ──────────
    # When the buffer ends with 'eq' and contains >= 2 numerals but no explicit
    # operator frame matched above, try binary rules on the last two numerals.
    # Handles NL word problems where the operator is encoded in prose:
    #   'alice has 3 apples bob gives her 4 how many eq'
    #   -> last two numerals: (3, 4)  -> add(3, 4) = 7
    # Priority order: add > sub > mul  (add is the default word-problem operation).
    # Confidence is discounted vs. explicit frames so Hopf memorisation wins
    # for training instances while the rule wins for novel (N1, N2) pairs.
    num_strs = [v for v in atom_values[:-1]
                if _value_rank(v, mg, rank_map) is not None
                and _is_numeral_str(v)]
    if len(num_strs) >= 2:
        n1 = _value_rank(num_strs[-2], mg, rank_map)
        n2 = _value_rank(num_strs[-1], mg, rank_map)
        if n1 is not None and n2 is not None:
            for rule_name in ('add', 'sub', 'mul'):
                rule = rules.get(rule_name)
                if rule is not None and rule.arity == 2:
                    try:
                        result_rank = rule.fn(n1, n2)
                        out = _result_atom(mg, result_rank, inv_rank_map,
                                           rule.confidence * 0.8)
                        if out:
                            return out
                    except Exception:
                        pass

    return {}


def _is_numeral_str(s: str) -> bool:
    """Return True if s looks like a number (int or float)."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False
