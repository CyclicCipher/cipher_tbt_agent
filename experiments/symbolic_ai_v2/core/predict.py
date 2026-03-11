"""Prediction and generation for a trained MorphismGraph.

Bottom-up (perceptual) — observing a sequence and predicting what comes next:
  predict()              — P(next | context, etype): ranked (symbol_id, prob) list
  predict_by_value()     — same, returns (string_value, prob) pairs for atoms only
  perplexity()           — cross-entropy bits/token on a test corpus
  perplexity_multilevel()— same, uses full composition hierarchy as context

Top-down (generative) — starting from a learned composition and expanding it:
  generate()             — decompose a high-level symbol to atoms (BLUEPRINT §generate)
  generate_sequence()    — alternate predict → sample → generate in a loop
  predict_sequence()     — like predict(), but expands compositions to atom sequences

Prediction back-off chain (BLUEPRINT.md §"predict()"):
  0a. Endofunctor table (Phase 17a): exact lookup for seen (input -> output) pairs.
      Active when mg._endofunctors is populated (call build_rule_store(mg, topo)).
  0b. Relational rule (Phase 19 Level 2): structure-only rule (identity, constant,
      commutative).  Zero content knowledge -- pure slot-position equality.
      Active when mg._relational_rules is populated (call build_variable_binding).
  0c. Variable binding (Phase 17b): apply discovered formula to unseen inputs.
      Active when mg._algebraic_rules is populated (call build_variable_binding).
  0d. Frame match (Phase 17b): raw atom buffer [op, arg, eq] → apply rule directly.
  0e. Rule chaining (Phase 18a): depth-limited recursive prefix evaluation.
      Handles multi-step arithmetic (e.g. eval 2 x 3 at 5 eq → 13).
  0f. Backward chaining (Phase 18c): adjunction-based constraint solving.
      Handles conservation laws (e.g. conserve add 3 4 eq add 5 → 2).
  1. Hopf-smoothed composition context: normalised edge counts, O(degree).
  2. CTKG type back-off: pool edges from same-type atoms via FCA type-group.
     Only active when mg._ctkg is populated (auto-wired via MorphismGraph(topo)).
  3. Corpus-wide marginal: uniform over all observed targets for this edge type.

Phase 18b — SequenceGoal:
  When the back-off chain predicts a COMPOSITION (multi-token answer), the
  result can be expressed as a SequenceGoal: the predicted atom sequence.
  predict_sequence() returns these expanded predictions.
  The _accuracy function in math_benchmark credits a composition prediction
  if the true last token appears as the final atom of the decomposition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .morphism import MorphismGraph
from .topology import Topology


# ── Phase 18b: SequenceGoal ────────────────────────────────────────────────────

@dataclass
class SequenceGoal:
    """A predicted multi-token answer: a sequence of atom IDs.

    Arises when the back-off chain's top prediction is a Composition rather
    than a leaf Atom.  The composition is expanded to its constituent atoms
    via generate(), giving the full surface form of the answer.

    Attributes
    ----------
    atoms      : ordered list of atom IDs (leaf symbols, level == 0)
    confidence : the probability assigned to this prediction
    """
    atoms:      list[int]
    confidence: float

    def contains(self, atom_id: int) -> bool:
        """Return True if atom_id appears anywhere in the predicted sequence."""
        return atom_id in self.atoms

    def ends_with(self, atom_id: int) -> bool:
        """Return True if atom_id is the last atom in the predicted sequence."""
        return bool(self.atoms) and self.atoms[-1] == atom_id

    # Atom buffer size — wide enough for linear_eval (7 tokens),
    # conservation (7 tokens), Bernoulli (8 tokens), and NL word problems
    # (up to 12 tokens: alice has N1 apples bob gives her N2 how many eq).
    ATOM_BUF_SIZE: int = 16


def predict(
    mg: MorphismGraph,
    context_id: int,
    etype: int,
    n_top: int = 10,
    type_map: Optional[dict[int, str]] = None,
) -> list[tuple[int, float]]:
    """Return the top-n predicted next symbols as [(symbol_id, probability)].

    Uses the fast path (edge count distribution) if context_id has been seen.

    Falls back in order:
      1. CTKG type back-off: if type_map is provided and context_id belongs to
         a type-group, pool edges from all same-type atoms (FCA adjunction).
      2. Corpus-wide marginal: uniform over all observed targets for etype.

    n_top = 0 returns the full distribution (may be large).
    """
    dist = mg.predict_dist(context_id, etype)

    if not dist:
        # Back-off 1: CTKG type-group marginal (adjunction back-off)
        if type_map is not None:
            dist = _type_marginal_dist(mg, type_map, context_id, etype)

    if not dist:
        # Back-off 2: corpus-wide marginal over all sources
        dist = _marginal_dist(mg, etype)

    if not dist:
        return []

    ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    return ranked if n_top == 0 else ranked[:n_top]


def predict_by_value(
    mg: MorphismGraph,
    value: str,
    etype_name: str,
    topology: Topology,
    n_top: int = 10,
) -> list[tuple[str, float]]:
    """Convenience wrapper: returns [(string_value, probability)].

    Only atom symbols appear in the result (compositions are filtered out
    since they are internal abstractions, not surface observations).
    """
    etype = topology.registry.code(etype_name)
    sid   = mg.atoms.get(value)
    if sid is None:
        return []
    ranked_ids = predict(mg, sid, etype, n_top=0)
    result: list[tuple[str, float]] = []
    for tgt_id, prob in ranked_ids:
        sym = mg.symbols[tgt_id]
        from .morphism import Atom
        if isinstance(sym, Atom):
            result.append((sym.value, prob))
        if n_top > 0 and len(result) >= n_top:
            break
    return result


def perplexity(
    mg: MorphismGraph,
    sequences: list,
    topology: Topology,
) -> float:
    """Compute cross-entropy perplexity in bits/token on a list of sequences.

    Each sequence is passed through topology.stream_tokens().
    The first token of each sequence (edge_type = None) is skipped because
    there is no context from which to predict it.

    Back-off chain (applied in order when the previous level returns empty):
      1. Hopf-smoothed edge counts from mg.predict_dist()
      2. FCA type-group marginal from mg._ctkg  (if LiveCTKG was wired)
      3. Corpus-wide marginal over all observed targets for this edge type

    Returns bits/token.  Lower is better.  Baseline: log2(vocab_size).
    """
    # Pre-extract type_map once; O(|atoms|), negligible vs. the eval loop.
    type_map: Optional[dict] = (
        mg._ctkg.atom_type_map(mg) if mg._ctkg is not None else None
    )

    total_bits  = 0.0
    total_tokens = 0

    for seq in sequences:
        prev_id: Optional[int] = None
        for value, etype in topology.stream_tokens(seq):
            sid = mg.atoms.get(value)
            if sid is None:
                # Unseen atom: back-off to uniform over known atoms
                n_atoms = max(mg.n_atoms(), 1)
                bits = math.log2(n_atoms)
                if prev_id is not None:
                    total_bits  += bits
                    total_tokens += 1
                prev_id = None   # can't use unseen atom as context
                continue

            if prev_id is not None and etype is not None:
                # 1. Hopf-smoothed edge counts
                dist = mg.predict_dist(prev_id, etype)
                # 2. FCA type-group back-off (if LiveCTKG is wired)
                if not dist and type_map is not None:
                    dist = _type_marginal_dist(mg, type_map, prev_id, etype)
                # 3. Corpus-wide marginal
                if not dist:
                    dist = _marginal_dist(mg, etype)

                p = dist.get(sid, 0.0)
                if p <= 0.0:
                    n_tgts = max(len(dist) + 1, 1)
                    p = 1.0 / (n_tgts * 10)

                total_bits   += -math.log2(p)
                total_tokens += 1

            prev_id = sid

    if total_tokens == 0:
        return 0.0
    return total_bits / total_tokens


def perplexity_multilevel(
    mg: MorphismGraph,
    sequences: list,
    topology: Topology,
) -> float:
    """Cross-entropy perplexity using the full composition hierarchy as context.

    Mirrors the buffer compression that MorphismGraph._compress_buf_tail()
    applies during training:

      ctx_id starts as None.
      After observing atom sid via edge etype from ctx_id:
        - if (ctx_id, etype, sid) is a known composition C  → ctx_id = C
        - otherwise                                          → ctx_id = sid

    ctx_id tracks the deepest composition covering recent history, exactly
    as _buf[-1] does during training.

    Back-off chain (each level applied only when the previous returns empty):
      1. Hopf-smoothed composition context  predict_dist(ctx_id, etype)
      2. FCA type back-off on ctx_id        _type_marginal_dist(type_map, ctx_id)
      3. Raw atom bigram                    predict_dist(atom_id, etype)
      4. FCA type back-off on atom_id       _type_marginal_dist(type_map, atom_id)
      5. Corpus-wide marginal               _marginal_dist(etype)

    Steps 2 and 4 only apply when mg._ctkg is populated (i.e. the graph was
    created with MorphismGraph(topology)).  They pool edges from all symbols
    sharing the same FCA structural type — zero effect for sequence_1d
    (trivial single type), useful for multi-edge topologies (grid, foveal).

    Returns bits/token.  Lower is better.  Baseline: log2(vocab_size).
    """
    # Pre-extract type_map once from LiveCTKG if it was wired.
    type_map: Optional[dict] = (
        mg._ctkg.atom_type_map(mg) if mg._ctkg is not None else None
    )

    total_bits   = 0.0
    total_tokens = 0

    for seq in sequences:
        ctx_id:   Optional[int] = None
        atom_id:  Optional[int] = None
        atom_buf: list[str]     = []   # raw atom values for multi-level back-off

        for value, etype in topology.stream_tokens(seq):
            sid = mg.atoms.get(value)
            if sid is None:
                n_atoms = max(mg.n_atoms(), 1)
                if ctx_id is not None:
                    total_bits   += math.log2(n_atoms)
                    total_tokens += 1
                ctx_id  = None
                atom_id = None
                atom_buf.clear()
                continue

            if ctx_id is not None and etype is not None:
                # 0a. Algebraic endofunctor table (Phase 17a) — seen inputs, certainty 1.0
                dist = _predict_via_rules(mg, ctx_id, etype)
                # 0b. Relational rule — structure-only (identity, constant, commutative)
                if not dist:
                    dist = _predict_via_relational_rule(mg, atom_buf)
                # 0c. Variable binding via ctx_id decomposition (Phase 17b)
                if not dist:
                    dist = _predict_via_variable_binding(mg, ctx_id, etype)
                # 0d. Variable binding via raw atom buffer (novel inputs, no composition)
                if not dist:
                    dist = _predict_via_frame_match(mg, atom_buf)
                # 0e. Rule chaining — recursive prefix evaluation (Phase 18a)
                if not dist:
                    dist = _predict_via_chain(mg, atom_buf)
                # 0f. Backward chaining — adjunction constraint solving (Phase 18c)
                if not dist:
                    dist = _predict_via_backward_chain(mg, atom_buf)
                # 1. Hopf-smoothed composition context
                if not dist:
                    dist = mg.predict_dist(ctx_id, etype)
                # 2. FCA type back-off on composition context
                if not dist and type_map is not None:
                    dist = _type_marginal_dist(mg, type_map, ctx_id, etype)
                # 3. Raw atom bigram
                if not dist and atom_id is not None:
                    dist = mg.predict_dist(atom_id, etype)
                # 4. FCA type back-off on atom context
                if not dist and type_map is not None and atom_id is not None:
                    dist = _type_marginal_dist(mg, type_map, atom_id, etype)
                # 5. Corpus-wide marginal
                if not dist:
                    dist = _marginal_dist(mg, etype)

                p = dist.get(sid, 0.0)
                if p <= 0.0:
                    n_tgts = max(len(dist) + 1, 1)
                    p = 1.0 / (n_tgts * 10)

                total_bits   += -math.log2(p)
                total_tokens += 1

            # Advance atom buffer (Phase 18: keep last 8 atoms for wider frames)
            atom_buf.append(value)
            if len(atom_buf) > SequenceGoal.ATOM_BUF_SIZE:
                atom_buf.pop(0)

            if ctx_id is not None and etype is not None:
                comp = mg.rules_inv.get((ctx_id, etype, sid))
                ctx_id = comp if comp is not None else sid
            else:
                ctx_id = sid
            atom_id = sid

    if total_tokens == 0:
        return 0.0
    return total_bits / total_tokens


# ── Internal helpers ──────────────────────────────────────────────────────────

def _predict_via_rules(
    mg: MorphismGraph,
    ctx_id: int,
    etype: int,
) -> dict[int, float]:
    """Level-0 back-off: algebraic endofunctor lookup (Phase 17a).

    Delegates to reasoning.rule_store.predict_via_rules if mg._endofunctors
    has been populated by build_rule_store(mg, topo).  Returns {} otherwise.
    Imported lazily to avoid circular imports.
    """
    if not getattr(mg, '_endofunctors', None):
        return {}
    from ..reasoning.rule_store import predict_via_rules as _pvr
    return _pvr(mg, ctx_id, etype)


def _predict_via_relational_rule(
    mg:       MorphismGraph,
    atom_buf: list[str],
) -> dict[int, float]:
    """Level-0b back-off: relational rule (Phase 19 Level 2).

    Applies structure-only rules (identity, constant, commutative) that
    require zero content knowledge.  Builds an atom_seq from atom_buf by
    looking up atom IDs, then delegates to predict_via_relational_rule.
    Returns {} if mg._relational_rules is not populated or no frame matches.
    """
    if not getattr(mg, '_relational_rules', None) or not atom_buf:
        return {}
    atom_seq = []
    for val in atom_buf:
        aid = mg.atoms.get(val)
        if aid is None:
            return {}
        atom_seq.append((aid, val))
    from ..reasoning.variable_binding import predict_via_relational_rule as _prr
    return _prr(mg, atom_seq)


def _predict_via_frame_match(
    mg:          MorphismGraph,
    atom_buf:    list[str],
) -> dict[int, float]:
    """Level-0d back-off: frame match on raw atom buffer (Phase 17b).

    Used when the multilevel context collapsed (novel input, no composition).
    atom_buf holds the last 4 atom string values seen before the prediction.
    Returns {} if mg._algebraic_rules is not populated or no frame matches.
    """
    if not getattr(mg, '_algebraic_rules', None) or not atom_buf:
        return {}
    from ..reasoning.variable_binding import predict_via_frame_match as _pfm
    return _pfm(mg, atom_buf)


def _predict_via_variable_binding(
    mg: MorphismGraph,
    ctx_id: int,
    etype: int,
) -> dict[int, float]:
    """Level-0c back-off: algebraic variable binding (Phase 17b).

    Delegates to reasoning.variable_binding.predict_via_variable_binding if
    mg._algebraic_rules has been populated by build_variable_binding(mg, topo).
    Handles inputs NOT seen in training by applying the discovered formula.
    Returns {} otherwise.  Imported lazily to avoid circular imports.
    """
    if not getattr(mg, '_algebraic_rules', None):
        return {}
    from ..reasoning.variable_binding import predict_via_variable_binding as _pvvb
    return _pvvb(mg, ctx_id, etype)


def _predict_via_chain(
    mg:       MorphismGraph,
    atom_buf: list[str],
) -> dict[int, float]:
    """Level-0e back-off: recursive rule chaining (Phase 18a).

    Evaluates atom_buf as a multi-step arithmetic expression (e.g. linear
    polynomial: eval A x B at V eq → A*V+B).  Active when mg._algebraic_rules
    is populated.  Returns {} if no chain succeeds.
    """
    if not getattr(mg, '_algebraic_rules', None) or not atom_buf:
        return {}
    from ..reasoning.rule_chainer import RuleChainer
    return RuleChainer(mg).chain(atom_buf)


def _predict_via_backward_chain(
    mg:       MorphismGraph,
    atom_buf: list[str],
) -> dict[int, float]:
    """Level-0f back-off: adjunction-based backward chaining (Phase 18c).

    Solves for unknown operands using discovered adjunctions (e.g. sub ⊣ add).
    Handles conservation law patterns and generic binary constraints.
    Active when mg._adjunctions is populated (call build_rule_store).
    Returns {} if no constraint can be solved.
    """
    if not getattr(mg, '_adjunctions', None) or not atom_buf:
        return {}
    from ..reasoning.backward_chainer import BackwardChainer
    return BackwardChainer(mg).solve(atom_buf)


def _marginal_dist(mg: MorphismGraph, etype: int) -> dict[int, float]:
    """Marginal distribution P(tgt | etype) summed over all source symbols."""
    counts: dict[int, int] = {}
    for (src, et, tgt), cnt in mg.edges.items():
        if et == etype:
            counts[tgt] = counts.get(tgt, 0) + cnt
    total = sum(counts.values())
    if total == 0:
        return {}
    return {tgt: cnt / total for tgt, cnt in counts.items()}


def _type_marginal_dist(
    mg: MorphismGraph,
    type_map: dict[int, str],
    context_id: int,
    etype: int,
) -> dict[int, float]:
    """Type-group marginal: pool edges from atoms sharing context_id's type.

    Returns P(tgt | type(context_id), etype) by aggregating edge counts from
    every atom that belongs to the same CTKG type-group as context_id.

    This is the FCA adjunction back-off (Phase 10b / BLUEPRINT predict() step 3):
    when context_id has never been seen as a source, its structural type still
    constrains what tokens can follow it.  The pool comes from all atoms of the
    same type — a Kneser-Ney-style smoothing grounded in the CTKG type lattice.

    Returns {} if context_id has no type or if no same-type atoms have edges.
    """
    context_type = type_map.get(context_id)
    if context_type is None:
        return {}

    same_type = {aid for aid, tname in type_map.items() if tname == context_type}

    counts: dict[int, int] = {}
    for (src, et, tgt), cnt in mg.edges.items():
        if et == etype and src in same_type:
            counts[tgt] = counts.get(tgt, 0) + cnt

    total = sum(counts.values())
    if total == 0:
        return {}
    return {tgt: c / total for tgt, c in counts.items()}


# ── Phase 18b: predict_sequence ───────────────────────────────────────────────

def predict_sequence(
    mg:         MorphismGraph,
    context_id: int,
    etype:      int,
    n_top:      int = 5,
    type_map:   Optional[dict[int, str]] = None,
) -> list[SequenceGoal]:
    """Like predict(), but expands composition predictions to atom sequences.

    Returns the top-n predictions as SequenceGoal objects, each containing the
    full atom sequence obtained by decomposing the predicted symbol.

    For a leaf Atom prediction, the SequenceGoal contains a single atom.
    For a Composition prediction, generate() decomposes it to leaf atoms.

    This is the Phase 18b interface: it allows the model to express multi-token
    answers (e.g. 'd sq x eq -> [mul, 2, x]') as a single structured prediction
    rather than requiring the model to predict each atom individually.

    Parameters
    ----------
    mg          : trained MorphismGraph
    context_id  : current context symbol ID
    etype       : edge type for the next observation
    n_top       : number of top predictions to expand
    type_map    : optional FCA type-group map (from LiveCTKG.atom_type_map)

    Returns list of SequenceGoal (may be empty).
    """
    ranked = predict(mg, context_id, etype, n_top=n_top, type_map=type_map)
    result: list[SequenceGoal] = []
    for sym_id, prob in ranked:
        atoms = generate(mg, sym_id, target_level=0)
        result.append(SequenceGoal(atoms=atoms, confidence=prob))
    return result


# ── Top-down generation (BLUEPRINT.md §"generate() — top-down expansion") ─────

def generate(
    mg: MorphismGraph,
    goal_id: int,
    target_level: int = 0,
) -> list[int]:
    """Decompose a high-level symbol down to symbols at target_level.

    Analysis-by-synthesis: apply the coproduct Δ (inverse of composition)
    recursively until every result symbol is at or below target_level.

    BLUEPRINT §generate():
      1. If level(goal_id) <= target_level: return [goal_id].
      2. Look up the decomposition rule:  goal_id = (left →[etype]→ right).
      3. Recurse: generate(left, target_level) + generate(right, target_level).

    Each composition has exactly one decomposition rule in this implementation
    (created by create_composition()), so selection is deterministic.  The
    frequency-weighted sampling described in BLUEPRINT is relevant when a symbol
    is reachable via multiple rule paths — not possible in Graph-SEQUITUR because
    each (left, etype, right) triple maps to exactly one composition ID.

    Parameters
    ----------
    mg           : trained MorphismGraph
    goal_id      : symbol ID to expand (Atom or Composition)
    target_level : stop expanding when symbols reach this level (0 = raw atoms)

    Returns a list of symbol IDs, all with level <= target_level.
    """
    sym = mg.symbols[goal_id]
    if sym.level <= target_level:
        return [goal_id]

    rule = mg.rules.get(goal_id)
    if rule is None:
        # Atom at level > target_level, or rule was pruned — return as-is.
        return [goal_id]

    left, _etype, right = rule
    return generate(mg, left, target_level) + generate(mg, right, target_level)


def generate_until_eos(
    mg:       MorphismGraph,
    prompt:   list[str],
    topology: Topology,
    eos:      str = '<eos>',
    max_steps: int = 50,
) -> list[str]:
    """Greedy autoregressive generation from a prompt until EOS.

    Feeds the prompt tokens into the composition context (without updating the
    model — this is inference only), then greedily predicts the next token using
    the full back-off chain at each step, stopping when EOS is predicted or
    max_steps is reached.

    The back-off chain is the same as in perplexity_multilevel:
      0a. Endofunctor table
      0b/0c. Variable binding / frame match
      0d. Rule chaining
      0e. Backward chaining
      1. Hopf-smoothed edge counts
      2. CTKG type back-off
      3. Corpus-wide marginal

    Parameters
    ----------
    mg        : trained MorphismGraph (must have build_rule_store + build_variable_binding called)
    prompt    : list of atom string values forming the question / context
    topology  : topology used for edge types and stream_tokens
    eos       : EOS atom value (default '<eos>')
    max_steps : hard cap on generated tokens (prevents infinite loops)

    Returns
    -------
    Generated tokens AFTER the prompt (not including the prompt itself),
    up to and including EOS (or max_steps tokens if EOS never predicted).
    """
    from .morphism import Atom
    from ..reasoning.variable_binding import predict_via_frame_match

    etype_names = topology.registry.names()
    if not etype_names:
        return []
    all_etypes = [topology.registry.code(n) for n in etype_names]

    # Build context from prompt.  Use stream_tokens to get the correct edge type
    # for each token (crucial for math_topology where num/op/eq/var differ).
    pairs = list(topology.stream_tokens(prompt))

    ctx_id:   Optional[int] = None
    atom_buf: list[str] = []

    for value, etype in pairs:
        sid = mg.atoms.get(value)
        if sid is None:
            continue
        atom_buf.append(value)
        if len(atom_buf) > SequenceGoal.ATOM_BUF_SIZE:
            atom_buf.pop(0)
        if ctx_id is not None and etype is not None:
            comp = mg.rules_inv.get((ctx_id, etype, sid))
            ctx_id = comp if comp is not None else sid
        else:
            ctx_id = sid

    if ctx_id is None:
        return []

    output: list[str] = []

    for _ in range(max_steps):
        # Aggregate predictions across all edge types so that topologies with
        # multiple types (e.g. math_topology: op/num/var/eq) are handled correctly.
        # The back-off chain uses algebraic rules (etype-independent) first; for
        # the compositional back-offs we try every registered edge type.
        dist: dict[int, float] = {}

        # 0c. Frame match on atom buffer (etype-independent).
        frame_dist = predict_via_frame_match(mg, atom_buf)
        if frame_dist:
            dist = frame_dist
        # 0d. Rule chaining.
        if not dist:
            dist = _predict_via_chain(mg, atom_buf) or {}
        # 0e. Backward chaining.
        if not dist:
            dist = _predict_via_backward_chain(mg, atom_buf) or {}

        # 0a/0b. Endofunctor / variable binding (try all edge types, take best).
        if not dist:
            for et in all_etypes:
                d = _predict_via_rules(mg, ctx_id, et)
                if d:
                    dist = d
                    break
        if not dist:
            for et in all_etypes:
                d = _predict_via_variable_binding(mg, ctx_id, et)
                if d:
                    dist = d
                    break

        # 1. Hopf-smoothed composition context (all edge types).
        if not dist:
            merged: dict[int, float] = {}
            for et in all_etypes:
                d = mg.predict_dist(ctx_id, et)
                for k, v in d.items():
                    merged[k] = max(merged.get(k, 0.0), v)
            dist = merged

        # 2. CTKG type back-off (all edge types).
        if not dist:
            type_map = mg._ctkg.atom_type_map(mg) if mg._ctkg is not None else None
            if type_map is not None:
                merged = {}
                for et in all_etypes:
                    d = _type_marginal_dist(mg, type_map, ctx_id, et)
                    for k, v in d.items():
                        merged[k] = max(merged.get(k, 0.0), v)
                dist = merged

        # 3. Corpus-wide marginal (all edge types).
        if not dist:
            merged = {}
            for et in all_etypes:
                d = _marginal_dist(mg, et)
                for k, v in d.items():
                    merged[k] = max(merged.get(k, 0.0), v)
            dist = merged

        if not dist:
            break

        # Greedy decode: pick the highest-probability symbol and expand to atoms.
        best_id = max(dist, key=dist.get)
        atoms   = generate(mg, best_id, target_level=0)

        for aid in atoms:
            sym = mg.symbols[aid]
            if not isinstance(sym, Atom):
                continue
            token = sym.value
            output.append(token)
            atom_buf.append(token)
            if len(atom_buf) > SequenceGoal.ATOM_BUF_SIZE:
                atom_buf.pop(0)
            if token == eos:
                return output

        # Advance composition context: try each edge type and take whichever
        # gives a valid composition; fall back to raw symbol id.
        advanced = False
        for et in all_etypes:
            comp = mg.rules_inv.get((ctx_id, et, best_id))
            if comp is not None:
                ctx_id  = comp
                advanced = True
                break
        if not advanced:
            ctx_id = best_id

    return output


def generate_sequence(
    mg: MorphismGraph,
    start_value: str,
    topology: Topology,
    n_steps: int = 20,
    rng=None,
) -> list[str]:
    """Generate a sequence by alternating predict → sample → generate.

    At each step:
      1. predict_dist(ctx_id, etype) → sample next symbol from the distribution
      2. generate(sampled_id, target_level=0) → expand to atoms
      3. Append atom values to output; advance ctx_id along the composition chain

    This is the full analysis-by-synthesis loop:
      predict() selects WHAT comes next (the composition-level goal);
      generate() realises HOW it is expressed (the atom-level surface form).

    Parameters
    ----------
    mg          : trained MorphismGraph
    start_value : seed atom value
    topology    : topology used to pick the default edge type
    n_steps     : number of predict → generate steps
    rng         : optional random.Random instance; a fresh one is created if None

    Returns a list of atom string values (the generated surface sequence).
    """
    import random as _random
    from .morphism import Atom

    _rng = rng if rng is not None else _random.Random()

    etype_names = topology.registry.names()
    if not etype_names:
        return []
    default_etype = topology.registry.code(etype_names[0])

    sid = mg.atoms.get(start_value)
    if sid is None:
        return [start_value]

    output: list[str] = [start_value]
    ctx_id = sid

    for _ in range(n_steps):
        # 1. Predict next symbol distribution (full back-off chain)
        dist = mg.predict_dist(ctx_id, default_etype)
        if not dist:
            type_map = mg._ctkg.atom_type_map(mg) if mg._ctkg is not None else None
            if type_map is not None:
                dist = _type_marginal_dist(mg, type_map, ctx_id, default_etype)
        if not dist:
            dist = _marginal_dist(mg, default_etype)
        if not dist:
            break

        # 2. Sample from distribution
        keys    = list(dist.keys())
        weights = [dist[k] for k in keys]
        next_id = _rng.choices(keys, weights=weights, k=1)[0]

        # 3. Generate: expand next_id to atoms
        for aid in generate(mg, next_id, target_level=0):
            sym = mg.symbols[aid]
            if isinstance(sym, Atom):
                output.append(sym.value)

        # 4. Advance context along the composition hierarchy (multilevel)
        comp   = mg.rules_inv.get((ctx_id, default_etype, next_id))
        ctx_id = comp if comp is not None else next_id

    return output
