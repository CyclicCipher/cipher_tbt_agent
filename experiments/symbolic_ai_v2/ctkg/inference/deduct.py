"""
Deduction via path traversal in MorphismGraph (Stage 4).

The DeductionEngine performs in-context symbolic deduction:
  - It parses a prefix for explicit implication rules ([rule_tok, A, B])
  - It parses the known premise ([given_tok, X])
  - It does BFS through the local implication graph to find the conclusion
  - This is the "path-finding in MorphismGraph" specified in the Phase XXV roadmap

Format handled
--------------
  [rule_tok, A, B, rule_tok, B, C, ..., given_tok, X, conclude_tok]
  → predict: the reachable conclusion from X after at most max_depth hops

All tokens are treated as opaque identifiers — no string comparison on content.
The role tokens (rule_tok, given_tok, conclude_tok) are the ONLY tokens compared
by name, at the prefix boundary.  Content tokens A, B, C, X are anonymous.

Supports D-1 (1-hop), D-2 (2-hop), D-3 (3-hop) from the I/D/A benchmark.

Iron Law compliance
-------------------
No string comparisons are performed on CONTENT tokens (A, B, X, etc.).
Only the three structural role tokens (rule_tok, given_tok, conclude_tok) are
matched by string equality at the boundary.  All other token identity is
preserved opaquely — the algorithm is symbol-invariant by construction.
"""

from __future__ import annotations

from collections import deque
from typing import Optional


class DeductionEngine:
    """In-context deductive reasoner using implication graph path traversal.

    Each call to ``predict`` builds a fresh local implication graph from the
    rules extracted from the prefix, then performs BFS to find the conclusion.
    No training memory is used — reasoning is purely in-context.

    This is the Stage 4 implementation of the path-traversal deduction level.
    The implication graph is semantically a sub-graph of the MorphismGraph
    (IMPLIES morphisms), built and traversed on-demand from the prefix.

    Parameters
    ----------
    rule_tok : str
        Token that introduces an implication rule pair.
        Format in prefix: [rule_tok, antecedent, consequent, ...]
    given_tok : str
        Token that introduces a known premise.
        Format in prefix: [given_tok, premise, ...]
    conclude_tok : str
        Token that signals a query ("what follows from the premises?").
        The engine fires when this is the last token in the prefix.
    max_depth : int
        Maximum BFS depth (number of implication hops).  Default 10.
    """

    def __init__(
        self,
        rule_tok: str,
        given_tok: str,
        conclude_tok: str,
        max_depth: int = 10,
    ) -> None:
        self._rule_tok = rule_tok
        self._given_tok = given_tok
        self._conclude_tok = conclude_tok
        self._max_depth = max_depth

    def predict(self, prefix: list[str]) -> Optional[dict[str, float]]:
        """Return {conclusion: 1.0} if a deductive conclusion is found.

        Parses all ``[rule_tok, A, B]`` pairs and the ``[given_tok, X]`` premise
        from *prefix*, builds a local implication graph, and performs BFS from
        X to find the deepest reachable conclusion.

        Only fires when *conclude_tok* is the last token in *prefix*.

        Parameters
        ----------
        prefix : list of string tokens observed so far.

        Returns
        -------
        dict[str, float] with a single entry {conclusion: 1.0}, or None.
        """
        # Engine fires only when we just saw conclude_tok.
        if not prefix or prefix[-1] != self._conclude_tok:
            return None

        # --- Parse the prefix ---
        rules: list[tuple[str, str]] = []   # (antecedent, consequent)
        premises: list[str] = []
        i = 0
        while i < len(prefix) - 1:   # -1: exclude trailing conclude_tok
            tok = prefix[i]
            if tok == self._rule_tok and i + 2 < len(prefix):
                antecedent = prefix[i + 1]
                consequent = prefix[i + 2]
                rules.append((antecedent, consequent))
                i += 3
            elif tok == self._given_tok and i + 1 < len(prefix):
                premises.append(prefix[i + 1])
                i += 2
            else:
                i += 1

        if not premises or not rules:
            return None

        # --- Build local adjacency list (the implication graph) ---
        graph: dict[str, list[str]] = {}
        for antecedent, consequent in rules:
            graph.setdefault(antecedent, []).append(consequent)

        # --- BFS from each premise ---
        # visited tracks all nodes reachable from premises (including premises).
        # reachable collects non-premise nodes (i.e. actual conclusions).
        visited: set[str] = set(premises)
        frontier: deque[tuple[str, int]] = deque(
            (p, 0) for p in premises
        )
        reachable: list[tuple[str, int]] = []  # (node, depth)

        while frontier:
            node, depth = frontier.popleft()
            if depth > 0:
                reachable.append((node, depth))
            if depth >= self._max_depth:
                continue
            for succ_node in graph.get(node, []):
                if succ_node not in visited:
                    visited.add(succ_node)
                    frontier.append((succ_node, depth + 1))

        if not reachable:
            return None

        # Return the most deeply reachable conclusion (longest derivation chain).
        # Ties broken by order of discovery (BFS order = insertion order).
        reachable.sort(key=lambda x: -x[1])
        conclusion = reachable[0][0]
        return {conclusion: 1.0}

    def predict_chain(
        self,
        rules: list[tuple[str, str]],
        premises: list[str],
    ) -> Optional[list[str]]:
        """Return the full deduction chain from premises to conclusion.

        Parameters
        ----------
        rules    : list of (antecedent, consequent) implication pairs.
        premises : starting facts.

        Returns
        -------
        list[str] of nodes along the path from the premise to the deepest
        conclusion, or None if no conclusion reachable.
        """
        if not premises or not rules:
            return None

        graph: dict[str, list[str]] = {}
        for ant, con in rules:
            graph.setdefault(ant, []).append(con)

        # BFS tracking parent pointers for path reconstruction.
        visited: set[str] = set(premises)
        parent: dict[str, Optional[str]] = {p: None for p in premises}
        frontier: deque[tuple[str, int]] = deque(
            (p, 0) for p in premises
        )
        deepest: Optional[tuple[str, int]] = None

        while frontier:
            node, depth = frontier.popleft()
            if depth > 0:
                if deepest is None or depth > deepest[1]:
                    deepest = (node, depth)
            if depth >= self._max_depth:
                continue
            for succ_node in graph.get(node, []):
                if succ_node not in visited:
                    visited.add(succ_node)
                    parent[succ_node] = node
                    frontier.append((succ_node, depth + 1))

        if deepest is None:
            return None

        # Reconstruct path from deepest conclusion back to a premise.
        path: list[str] = []
        cur: Optional[str] = deepest[0]
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        return path


class TypedDeductionEngine(DeductionEngine):
    """DeductionEngine extended with value-dependent type checking (D-8).

    When performing BFS over the implication graph, checks that each token
    satisfies its type constraint before following outgoing edges.

    The "dependent" property: the set of reachable conclusions depends on the
    VALUES of the tokens, not just their names.  Two tokens with the same name
    but different numeric values may have different type membership and thus
    different reachable conclusions.

    Parameters
    ----------
    rule_tok, given_tok, conclude_tok, max_depth : same as DeductionEngine.

    New method
    ----------
    predict_with_types(prefix, value_context, type_constraints) ->
        Optional[dict[str, float]]
        Like predict(), but filters BFS edges based on type constraints.

        value_context     : dict[str, float] mapping token name → numeric value.
        type_constraints  : dict[str, Callable[[float], bool]] mapping token name
                            → predicate.  If a BFS node X fails its predicate,
                            outgoing edges from X are blocked (type violation).

    Iron Law compliance
    -------------------
    Type predicates are Python callables applied to float values — no string
    comparisons on domain token names.  The predicate is stored and called by
    the engine; the engine never inspects the token name to decide which
    predicate to use.

    Bitter Lesson compliance
    ------------------------
    The type context is data (dict of callables), not code.  The engine applies
    whatever predicates it is given; it has no built-in notion of 'velocity',
    'mass', or any domain concept.
    """

    def predict_with_types(
        self,
        prefix: list,
        value_context: dict | None = None,
        type_constraints: dict | None = None,
    ):
        """BFS deduction with value-dependent type checking.

        Parameters
        ----------
        prefix            : same as predict().
        value_context     : dict[str, float].  Token name → numeric value.
                            Tokens absent from value_context are treated as
                            having value 0.0.
        type_constraints  : dict[str, Callable[[float], bool]].
                            Token name → predicate.  Absent tokens have no
                            constraint (predicate always True).

        Returns
        -------
        Same as predict(): {conclusion: 1.0} or None.
        Tokens whose type constraint fails are NOT followed in BFS.
        """
        if not prefix or prefix[-1] != self._conclude_tok:
            return None

        vc = value_context or {}
        tc = type_constraints or {}

        # Parse prefix — same logic as predict()
        rules: list = []
        premises: list = []
        i = 0
        while i < len(prefix) - 1:
            tok = prefix[i]
            if tok == self._rule_tok and i + 2 < len(prefix):
                rules.append((prefix[i + 1], prefix[i + 2]))
                i += 3
            elif tok == self._given_tok and i + 1 < len(prefix):
                premises.append(prefix[i + 1])
                i += 2
            else:
                i += 1

        if not premises or not rules:
            return None

        # Build implication graph
        graph: dict = {}
        for ant, con in rules:
            graph.setdefault(ant, []).append(con)

        def _type_ok(token: str) -> bool:
            """Check if token satisfies its type constraint, if any."""
            pred = tc.get(token)
            if pred is None:
                return True
            value = vc.get(token, 0.0)
            return pred(value)

        # BFS with type gating
        from collections import deque
        visited: set = set(premises)
        frontier: deque = deque((p, 0) for p in premises)
        reachable: list = []

        while frontier:
            node, depth = frontier.popleft()
            if depth > 0:
                reachable.append((node, depth))
            if depth >= self._max_depth:
                continue
            # Only follow edges if THIS node passes its type check
            if not _type_ok(node):
                continue
            for succ_node in graph.get(node, []):
                if succ_node not in visited:
                    visited.add(succ_node)
                    frontier.append((succ_node, depth + 1))

        if not reachable:
            return None

        reachable.sort(key=lambda x: -x[1])
        conclusion = reachable[0][0]
        return {conclusion: 1.0}
