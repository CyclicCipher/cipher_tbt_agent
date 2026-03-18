"""
Compression tree accumulation.

Phase 4 of the CTKG pipeline (GraphZip / SEQUITUR) produces grammar rules of the
form R → (child_1, ..., child_k).  `accumulate_tree` replays grammar derivations
to build the compression tree for each sequence.

The **compression tree** for a typed sequence is the tree of rule applications that
reproduces the sequence from its non-terminal root.  Each internal node carries:
  - The rule_id that was applied at this node
  - The position in the original sequence
  - A list of child nodes (recursively)
  - The type distribution at this node

Phase XI: MultiMorphism and OperadStructure removed — both are degenerate special
cases of Relation (see ctkg/learning/relation_store.py).  accumulate_tree now
returns the list of root CompressionTreeNodes directly.

Reference: Architecture §Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.concept_lattice import ConceptId
from experiments.symbolic_ai_v2.ctkg.learning.graph_grammar import (
    Grammar,
    GrammarRule,
    _is_terminal,
    _is_nonterminal,
    _terminal_id,
    _nonterminal_id,
)


# Type alias for a probability distribution over concept IDs
TypeDist = dict[ConceptId, float]


@dataclass
class CompressionTreeNode:
    """One node in a compression tree.

    Terminal nodes correspond to atoms in the original sequence.
    Non-terminal nodes correspond to grammar rule applications.

    Attributes
    ----------
    rule_id:
        Grammar rule ID, or -1 for terminal nodes.
    position:
        Start position in the original sequence (0-indexed).
    children:
        Child nodes (empty for terminal nodes).
    type_dist:
        Type distribution at this node (concept_id → probability).
    symbol:
        The raw symbol ID as produced by `graph_grammar.py` (terminal or
        non-terminal).
    """

    rule_id: int
    position: int
    children: list["CompressionTreeNode"] = field(default_factory=list)
    type_dist: TypeDist = field(default_factory=dict)
    symbol: int = -1

    @property
    def is_terminal(self) -> bool:
        return len(self.children) == 0

    def depth(self) -> int:
        """Depth of the subtree rooted at this node (0 = leaf)."""
        if self.is_terminal:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def __repr__(self) -> str:
        tag = "leaf" if self.is_terminal else f"rule={self.rule_id}"
        return f"Node({tag}, pos={self.position}, children={len(self.children)})"


# ---------------------------------------------------------------------------
# accumulate_tree()
# ---------------------------------------------------------------------------

def accumulate_tree(
    typed_sequences: list[list[int]],
    grammar: Grammar,
    concept_type_fn: Optional[object] = None,
) -> list[CompressionTreeNode]:
    """Replay grammar derivations and build compression trees.

    For each typed sequence, replays the grammar derivation to produce the
    root CompressionTreeNode for that sequence.

    Parameters
    ----------
    typed_sequences:
        List of typed sequences (each is a list of symbol IDs as produced by
        `graph_grammar.typed_corpus_from_lattice()` or similar).  Terminal
        symbols encode concept IDs via `_t(concept_id)`; non-terminal symbols
        encode rule IDs via `_nt(rule_id)`.
    grammar:
        The Grammar returned by `graph_grammar.sequitur()` or
        `graph_grammar.corpus_grammar()`.
    concept_type_fn:
        Optional callable `concept_id → TypeDist` for computing the type
        distribution of a terminal concept.  If None, a point mass is used.

    Returns
    -------
    List of root CompressionTreeNodes, one per symbol in each typed sequence.
    """
    roots: list[CompressionTreeNode] = []

    # Build a rule lookup: rule_id → GrammarRule
    rule_by_id: dict[int, GrammarRule] = {r.rule_id: r for r in grammar.rules}

    def _term_dist(concept_id: ConceptId) -> TypeDist:
        if concept_type_fn is not None:
            try:
                return dict(concept_type_fn(concept_id))
            except Exception:
                pass
        return {concept_id: 1.0}

    def _expand(sym: int, pos: int) -> CompressionTreeNode:
        """Recursively expand a symbol to a CompressionTreeNode."""
        if _is_terminal(sym):
            c_id = _terminal_id(sym)
            return CompressionTreeNode(
                rule_id=-1,
                position=pos,
                children=[],
                type_dist=_term_dist(c_id),
                symbol=sym,
            )
        # Non-terminal: look up the rule and expand each child
        rule_id = _nonterminal_id(sym)
        rule = rule_by_id.get(rule_id)
        if rule is None:
            return CompressionTreeNode(
                rule_id=rule_id,
                position=pos,
                children=[],
                type_dist={},
                symbol=sym,
            )
        children: list[CompressionTreeNode] = []
        cur_pos = pos
        for child_sym in rule.body:
            child_node = _expand(child_sym, cur_pos)
            children.append(child_node)
            cur_pos += _subtree_span(child_sym, rule_by_id)

        child_dists = [c.type_dist for c in children]
        combined = _mean_dist(child_dists)

        return CompressionTreeNode(
            rule_id=rule_id,
            position=pos,
            children=children,
            type_dist=combined,
            symbol=sym,
        )

    for typed_seq in typed_sequences:
        pos = 0
        for sym in typed_seq:
            roots.append(_expand(sym, pos))
            pos += _subtree_span(sym, rule_by_id)

    return roots


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subtree_span(sym: int, rule_by_id: dict[int, GrammarRule]) -> int:
    """Compute the number of terminal leaves in the subtree of sym.

    For terminal symbols, span = 1.
    For non-terminals, span = sum of spans of all children (recursive).
    Cycles (self-referential rules) are guarded against with a visited set.
    """
    visited: set[int] = set()

    def _span(s: int) -> int:
        if _is_terminal(s):
            return 1
        rule_id = _nonterminal_id(s)
        if rule_id in visited:
            return 1  # cycle guard
        visited.add(rule_id)
        rule = rule_by_id.get(rule_id)
        if rule is None:
            return 1
        return sum(_span(child) for child in rule.body)

    return _span(sym)


def _mean_dist(dists: list[TypeDist]) -> TypeDist:
    """Element-wise mean of a list of type distributions."""
    if not dists:
        return {}
    combined: dict[int, float] = {}
    for d in dists:
        for c_id, w in d.items():
            combined[c_id] = combined.get(c_id, 0.0) + w
    total = sum(combined.values())
    if total <= 0.0:
        return {}
    n = len(dists)
    return {c_id: w / total for c_id, w in combined.items()}
