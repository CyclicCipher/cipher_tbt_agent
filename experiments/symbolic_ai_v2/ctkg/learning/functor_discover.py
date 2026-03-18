"""
Phase IX — Functorial Variable Discovery.

Discovers functorial variables — variables whose substitution sets form
consistent partitions across multiple rewrite rules, indicating a latent
functor pair with a natural transformation between them.

Example (word analogies):
  Rule R: analogy(X, Y) → eq(X, Y)
  Corpus: analogy(man, woman), analogy(king, queen), analogy(husband, wife)
  Observed bindings: X ∈ {man, king, husband},  Y ∈ {woman, queen, wife}
  Consistent bijection: man↔woman, king↔queen, husband↔wife
  → FunctorCandidate(partition_a={man,king,husband}, partition_b={woman,queen,wife},
                     bijection={man:woman, king:queen, husband:wife}, evidence=3)

CT reference (CT_REFERENCE.md §3, §6):
  A FunctorCandidate represents the evidence for a natural transformation
  η: F_a → F_b where F_a maps abstract role objects to partition_a values
  and F_b maps them to partition_b values.  The bijection is the components
  η_c: F_a(c) → F_b(c) at each abstract role object c.

API:
  collect_variable_values(rules, corpus_examples)  → dict
  cluster_consistent_partitions(variable_values)   → list[FunctorCandidate]
  register_as_nat_trans(candidate, kg)             → NaturalTransformation

See FIXING_GENERALIZATION_PART2.md §Phase IX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.rewrite import RewriteRule
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import Expr, match


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FunctorCandidate:
    """Evidence for a functor pair with natural transformation.

    Represents the discovered bijection between two consistent value sets
    across multiple rules.

    Attributes
    ----------
    partition_a:
        The domain partition — all values taken by the 'source' variable role.
        e.g. frozenset({'man', 'king', 'husband'})
    partition_b:
        The codomain partition — all values taken by the 'target' variable role.
        e.g. frozenset({'woman', 'queen', 'wife'})
    bijection:
        The consistent bijection a → b.
        e.g. {'man': 'woman', 'king': 'queen', 'husband': 'wife'}
    supporting_rules:
        Identifiers of the rules that provided evidence for this partition.
    evidence:
        Total number of supporting (rule, example) instances.
    """

    partition_a: frozenset
    partition_b: frozenset
    bijection: dict[str, str] = field(default_factory=dict)
    supporting_rules: list[str] = field(default_factory=list)
    evidence: int = 0

    def __repr__(self) -> str:
        return (
            f"FunctorCandidate(a={set(self.partition_a)}, "
            f"b={set(self.partition_b)}, ev={self.evidence})"
        )


@dataclass
class NaturalTransformation:
    """A discovered natural transformation η: F_a → F_b.

    Produced by register_as_nat_trans() from a FunctorCandidate.

    Attributes
    ----------
    name:
        Unique identifier for this natural transformation.
    partition_a:
        Domain values (F_a's image).
    partition_b:
        Codomain values (F_b's image).
    components:
        η components: dict[abstract_role → morphism].  For finite sets,
        each morphism is just the target value: components[a] = b means η(a) = b.
    evidence:
        Number of training instances supporting this transformation.
    """

    name: str
    partition_a: frozenset
    partition_b: frozenset
    components: dict[str, str] = field(default_factory=dict)
    evidence: int = 0

    def __repr__(self) -> str:
        return f"NatTrans({self.name}, ev={self.evidence})"


# ---------------------------------------------------------------------------
# collect_variable_values
# ---------------------------------------------------------------------------

def collect_variable_values(
    rules: list[RewriteRule],
    corpus_examples: list[Expr],
) -> dict[tuple, list[str]]:
    """Collect substitution values for each (rule_id, var_name) pair.

    For each rule, tries to match each corpus example against the rule's lhs.
    When a match succeeds, records the binding for each pattern variable.

    Parameters
    ----------
    rules:
        List of RewriteRules.  Each rule's lhs may contain var() pattern nodes.
    corpus_examples:
        List of Expr trees (from term_algebra) to match against.

    Returns
    -------
    dict mapping (rule_id_str, var_name) → list of observed values.
    rule_id_str is `repr(rule)` or the rule's algebra_name if set, else its index.
    """
    result: dict[tuple, list[str]] = {}

    for idx, rule in enumerate(rules):
        rule_id = rule.algebra_name if rule.algebra_name else f'rule_{idx}'
        for example in corpus_examples:
            bindings = match(rule.lhs, example)
            if bindings is None:
                continue
            for var_name, value in bindings.items():
                key = (rule_id, var_name)
                # Extract the head (token value) from a var-bound Expr
                if isinstance(value, Expr):
                    from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
                    val_str = TOKEN_GRAPH.decode(value.head)
                else:
                    val_str = str(value)
                if key not in result:
                    result[key] = []
                result[key].append(val_str)

    return result


# ---------------------------------------------------------------------------
# cluster_consistent_partitions
# ---------------------------------------------------------------------------

def cluster_consistent_partitions(
    variable_values: dict[tuple, list[str]],
) -> list[FunctorCandidate]:
    """Find pairs of variable slots with a consistent bijection.

    Two variable slots (rule_i, var_a) and (rule_i, var_b) from the SAME
    rule form a consistent bijection candidate if:
      1. Both slots have the same number of observed values (same arity).
      2. The values co-occur in the same positions across examples (i.e.
         the i-th value of var_a always pairs with the i-th value of var_b).
      3. The resulting mapping is injective (a bijection on observed values).

    Additionally, candidates from DIFFERENT rules are merged if they share
    the same bijection (same (a, b) pair mapping), increasing evidence.

    Parameters
    ----------
    variable_values:
        Output of collect_variable_values():
        {(rule_id, var_name): [values_observed_in_order]}.

    Returns
    -------
    List of FunctorCandidate objects, one per distinct discovered bijection.
    Sorted by evidence descending.
    """
    # Group keys by rule_id
    rules_to_vars: dict[str, list[str]] = {}
    for (rule_id, var_name) in variable_values:
        if rule_id not in rules_to_vars:
            rules_to_vars[rule_id] = []
        if var_name not in rules_to_vars[rule_id]:
            rules_to_vars[rule_id].append(var_name)

    # For each rule, check all pairs of variables for consistent bijections
    # bijection_key → FunctorCandidate (for merging across rules)
    canon_candidates: dict[tuple, FunctorCandidate] = {}

    for rule_id, var_names in rules_to_vars.items():
        if len(var_names) < 2:
            continue

        for i in range(len(var_names)):
            for j in range(i + 1, len(var_names)):
                va, vb = var_names[i], var_names[j]
                vals_a = variable_values.get((rule_id, va), [])
                vals_b = variable_values.get((rule_id, vb), [])

                if not vals_a or not vals_b or len(vals_a) != len(vals_b):
                    continue

                # Build proposed bijection from positional co-occurrence
                bij_ab: dict[str, str] = {}
                bij_ba: dict[str, str] = {}
                consistent = True
                for a_val, b_val in zip(vals_a, vals_b):
                    if a_val in bij_ab and bij_ab[a_val] != b_val:
                        consistent = False
                        break
                    if b_val in bij_ba and bij_ba[b_val] != a_val:
                        consistent = False
                        break
                    bij_ab[a_val] = b_val
                    bij_ba[b_val] = a_val

                if not consistent:
                    continue
                if len(bij_ab) < 2:
                    # Trivial (single-value) bijections provide no evidence
                    continue

                # Canonical key: sorted tuple of (a, b) pairs
                bij_key = tuple(sorted(bij_ab.items()))
                part_a = frozenset(bij_ab.keys())
                part_b = frozenset(bij_ab.values())

                if bij_key not in canon_candidates:
                    canon_candidates[bij_key] = FunctorCandidate(
                        partition_a=part_a,
                        partition_b=part_b,
                        bijection=dict(bij_ab),
                        supporting_rules=[rule_id],
                        evidence=len(vals_a),
                    )
                else:
                    cand = canon_candidates[bij_key]
                    if rule_id not in cand.supporting_rules:
                        cand.supporting_rules.append(rule_id)
                    cand.evidence += len(vals_a)

    return sorted(canon_candidates.values(), key=lambda c: c.evidence, reverse=True)


# ---------------------------------------------------------------------------
# register_as_nat_trans
# ---------------------------------------------------------------------------

def register_as_nat_trans(
    candidate: FunctorCandidate,
    kg: dict,
    name: Optional[str] = None,
) -> NaturalTransformation:
    """Convert a FunctorCandidate to a NaturalTransformation and register it.

    Creates a NaturalTransformation η: F_a → F_b whose components are the
    bijection mapping.  Registers it in `kg` under the key `name`.

    Parameters
    ----------
    candidate:
        FunctorCandidate to register.
    kg:
        Knowledge graph registry — any dict mapping name → NaturalTransformation.
        (May be a MorphismGraph or a plain dict for testing.)
    name:
        Optional name for the transformation.  If None, a name is auto-generated
        from the partition sizes and evidence.

    Returns
    -------
    The NaturalTransformation that was registered.
    """
    if name is None:
        # Auto-generate a stable name from sorted bijection pairs
        pairs = '_'.join(
            f'{a}-{b}' for a, b in sorted(candidate.bijection.items())[:3]
        )
        name = f'nat_trans_{pairs}'

    nat_trans = NaturalTransformation(
        name=name,
        partition_a=candidate.partition_a,
        partition_b=candidate.partition_b,
        components=dict(candidate.bijection),
        evidence=candidate.evidence,
    )

    # Register in the knowledge graph
    if isinstance(kg, dict):
        kg[name] = nat_trans
    elif hasattr(kg, 'add_nat_trans'):
        kg.add_nat_trans(nat_trans)
    else:
        # Fallback: treat as dict-like
        kg[name] = nat_trans

    return nat_trans
