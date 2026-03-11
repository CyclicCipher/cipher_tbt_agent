"""backward_chainer.py -- Phase 18c: Adjunction-based backward chaining.

Biological analog: the anterior cingulate cortex detects mismatch between
expected and observed outcomes; the cerebellum maintains an inverse model
("given desired output, compute required input").

Active inference framing: the conservation law is a HARD prior — not revisable
by new data.  When the goal is a conserved quantity, the agent must ACT to
find the missing value by running the inverse model (adjunction).  This is
pure action, zero learning: the generative model's inverse computes what the
world must be like to satisfy the prior.

Algorithm:
  Given atom_values (the last N atoms before the unknown token to predict):
  1. Scan for known conservation frames (e.g. 'conserve op A B eq op C').
  2. Evaluate the LHS expression to get the conserved total.
  3. Use the discovered adjunction (e.g. sub ⊣ add) to invert the RHS operator
     and solve for the unknown: ? = adj_op(total, C).
  4. Also handle generic single-eq constraints: LHS eq op C → ? = adj(LHS, C).

Adjunction storage (mg._adjunctions):
  List of (F, G, coverage) where G(F(A, B), B) ≈ A for all training instances.
  For (F='add', G='sub'): sub(add(A,B), B) = A — sub is the right adjoint of add.

Public API:
  BackwardChainer(mg)
    solve(atom_values)              -- main entry point
    _find_adjunction(op)            -- look up adj(op) from mg._adjunctions
    _try_conservation(buf, rules, adjs)
    _try_binary_constraint(buf, rules, adjs)
"""

from __future__ import annotations

from typing import Optional

from ..core.morphism import MorphismGraph
from .rule_chainer import _parse_prefix, _try_rank, _try_int


class BackwardChainer:
    """Adjunction-based backward chaining for constraint satisfaction.

    Used as back-off level 0e in the prediction chain.  Fires when the buffer
    ends with an operator + known operand (not 'eq'), indicating the agent
    knows the total and one argument and must infer the other.
    """

    def __init__(self, mg: MorphismGraph) -> None:
        self.mg = mg

    def solve(self, atom_values: list[str]) -> dict[int, float]:
        """Attempt backward chaining on atom_values.

        atom_values: recent atom strings before the unknown token to predict
                     (does NOT need to end with 'eq').

        Returns {result_atom_id: confidence} or {}.
        """
        adjunctions = getattr(self.mg, '_adjunctions', None)
        rules       = getattr(self.mg, '_algebraic_rules', None)
        if not adjunctions or not rules or not atom_values:
            return {}

        inv_rank_map = getattr(self.mg, '_inv_rank_map', {})

        # Pattern 1: conservation frame
        val = self._try_conservation(atom_values, rules, adjunctions)
        if val is not None:
            result_rank, conf = val
            result_id = inv_rank_map.get(result_rank)
            if result_id is None and abs(result_rank) <= 10_000:
                result_id = self.mg.get_or_create_atom(str(result_rank), 'num')
            if result_id is not None:
                return {result_id: conf}

        # Pattern 2: generic single-eq constraint
        val = self._try_binary_constraint(atom_values, rules, adjunctions)
        if val is not None:
            result_rank, conf = val
            result_id = inv_rank_map.get(result_rank)
            if result_id is None and abs(result_rank) <= 10_000:
                result_id = self.mg.get_or_create_atom(str(result_rank), 'num')
            if result_id is not None:
                return {result_id: conf}

        return {}

    # ── Pattern 1: conservation ────────────────────────────────────────────────

    def _try_conservation(
        self,
        buf:        list[str],
        rules:      dict,
        adjunctions: list,
    ) -> Optional[tuple[int, float]]:
        """Match 'conserve op A B eq op2 C' and solve for the unknown D.

        The conservation invariant says op(A, B) = op2(C, D).
        We know total = op(A, B) from the LHS.
        We know op2 and C from the RHS prefix (the last tokens in buf).
        We solve: D = adj_op2(total, C).

        Searches for 'conserve' within buf to handle buffer-offset cases
        (the frame may not start at index 0 if the buffer partially covers it).
        """
        for i, tok in enumerate(buf):
            if tok != 'conserve':
                continue
            # Minimum structure: conserve LHS_op A B eq RHS_op C
            # Indices:           i      i+1    i+2 i+3 i+4 i+5  i+6
            if i + 6 >= len(buf):
                continue

            eq_tok = buf[i + 4]
            if eq_tok != 'eq':
                continue

            lhs_op = buf[i + 1]
            rhs_op = buf[i + 5]
            a_str  = buf[i + 2]
            b_str  = buf[i + 3]
            c_str  = buf[i + 6]

            # Evaluate LHS: try as a two-argument prefix expression.
            lhs_tokens = [lhs_op, a_str, b_str]
            lhs_val, lhs_npos = _parse_prefix(lhs_tokens, 0, rules, self.mg)
            if lhs_val is None or lhs_npos != len(lhs_tokens):
                continue

            c_val = _try_rank(c_str, self.mg)
            if c_val is None:
                continue

            adj_op = self._find_adjunction(rhs_op, adjunctions)
            if adj_op is None:
                continue

            adj_rule = rules.get(adj_op)
            if adj_rule is None or adj_rule.arity != 2:
                continue

            try:
                result_val = adj_rule.fn(lhs_val, c_val)
                if abs(result_val) <= 10_000:
                    return result_val, adj_rule.confidence
            except Exception:
                pass

        return None

    # ── Pattern 2: generic single-eq constraint ────────────────────────────────

    def _try_binary_constraint(
        self,
        buf:        list[str],
        rules:      dict,
        adjunctions: list,
    ) -> Optional[tuple[int, float]]:
        """Match '...LHS... eq op C [unary_op]' and solve for the unknown.

        Scans buf for 'eq'; evaluates the LHS prefix; the RHS suffix (after
        'eq') is:
          - Length 2: [op, C]         → solve op(?, C) = LHS via adj(op)
          - Length 3: [op, C, uop]    → solve op(C, uop(?)) = LHS via chained
                                        adjunctions: (1) adj(op)(LHS,C) = uop(?),
                                        (2) adj(uop)(uop(?)) = ?
            Both adjunctions must be in mg._adjunctions (discovered, not hardcoded).
        """
        for eq_i in range(1, len(buf) - 2):
            if buf[eq_i] != 'eq':
                continue

            rhs = buf[eq_i + 1:]
            if not rhs:
                continue

            # Evaluate LHS (shared for all RHS lengths).
            lhs_tokens = buf[:eq_i]
            if not lhs_tokens:
                continue
            lhs_val, lhs_npos = _parse_prefix(lhs_tokens, 0, rules, self.mg)
            if lhs_val is None or lhs_npos != len(lhs_tokens):
                continue

            if len(rhs) == 2:
                # Simple case: LHS eq op C → ? = adj(op)(LHS, C)
                rhs_op = rhs[0]
                c_val  = _try_rank(rhs[1], self.mg)
                if c_val is None:
                    continue
                adj_op = self._find_adjunction(rhs_op, adjunctions)
                if adj_op is None:
                    continue
                adj_rule = rules.get(adj_op)
                if adj_rule is None or adj_rule.arity != 2:
                    continue
                try:
                    result_val = adj_rule.fn(lhs_val, c_val)
                    if abs(result_val) <= 10_000:
                        return result_val, adj_rule.confidence
                except Exception:
                    pass

            elif len(rhs) == 3:
                # Chained case: LHS eq bin_op C unary_op
                # Meaning: bin_op(C, unary_op(?)) = LHS
                # Step 1: adj(bin_op)(LHS, C) = unary_op(?)
                # Step 2: adj(unary_op)(unary_op(?)) = ?
                bin_op  = rhs[0]
                c_val   = _try_rank(rhs[1], self.mg)
                unary_op = rhs[2]
                if c_val is None:
                    continue
                # Step 1: invert the binary operator
                adj_bin = self._find_adjunction(bin_op, adjunctions)
                if adj_bin is None:
                    continue
                adj_bin_rule = rules.get(adj_bin)
                if adj_bin_rule is None or adj_bin_rule.arity != 2:
                    continue
                try:
                    intermediate = adj_bin_rule.fn(lhs_val, c_val)
                except Exception:
                    continue
                # Step 2: invert the unary operator (adj must have arity 1)
                adj_unary = self._find_adjunction(unary_op, adjunctions)
                if adj_unary is None:
                    continue
                adj_unary_rule = rules.get(adj_unary)
                if adj_unary_rule is None or adj_unary_rule.arity != 1:
                    continue
                try:
                    result_val = adj_unary_rule.fn(intermediate)
                    if isinstance(result_val, float) and result_val == int(result_val):
                        result_val = int(result_val)
                    if isinstance(result_val, int) and abs(result_val) <= 10_000:
                        return result_val, min(adj_bin_rule.confidence, adj_unary_rule.confidence)
                except Exception:
                    pass

        return None

    # ── Adjunction lookup ──────────────────────────────────────────────────────

    def _find_adjunction(self, op: str, adjunctions: list) -> Optional[str]:
        """Return the adjoint of op from mg._adjunctions, or None.

        mg._adjunctions is a list of (F, G, coverage) tuples where
        G(F(A, B), B) = A — G is the right adjoint of F.

        To invert op=F: use G.
        To invert op=G: use F (left adjoint — useful when roles are symmetric,
        e.g. sub and add are mutual adjoints in an informal sense).
        """
        for F, G, _cov in adjunctions:
            if F == op:
                return G
            if G == op:
                return F
        return None
