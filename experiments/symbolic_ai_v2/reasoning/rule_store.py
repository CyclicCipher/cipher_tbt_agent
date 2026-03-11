"""rule_store.py — Algebraic rule discovery from a trained MorphismGraph.

Phase 17a: Category-theoretic approaches.

Three discovery mechanisms, in order of CT sophistication:

  1. Endofunctor maps
     For each operator, extract a lookup table {input→output} from the
     composition hierarchy.  Unary: {arg_id: result_id}.  Binary:
     {(arg1_id, arg2_id): result_id}.  These are endofunctors on the
     number type-group; discovering them from data without being told the
     formula is the CT analogue of learning the rule by observation.

  2. Adjunction detection
     Two operators F, G form an adjunction (F ⊣ G) if G(F(A,B),B) = A
     holds on all training instances.  Discovered from the endofunctor
     maps by checking the round-trip identity on extracted pairs.
     Examples: add ⊣ sub,  mul ⊣ div,  succ ⊣ pred.

  3. Natural transformation discovery
     Two operators F, G are related by a natural transformation if their
     induced maps on the number type-group are related by a systematic
     offset or inversion.  Examples:
       - succ and pred are inverse natural transformations
       - add(·,k) and sub(·,k) are parameterised natural transformations
     These are registered as CTKG Adjunction nodes for downstream use.

  4. Higher-order relational reasoning
     Check whether one operator's rule is expressible as a composition
     of other discovered rules (e.g. mul = iterated add).  Requires
     integer arithmetic on atom values — extracted via _atom_int().
     Detected when: F(A) = G(G(A)) or G applied k times.

Public API:
  build_rule_store(mg, topo)           — run all discovery; stores results on mg
  predict_via_rules(mg, ctx_id, etype) — endofunctor lookup for one prediction

After calling build_rule_store(mg, topo), the MorphismGraph carries:
  mg._endofunctors  : dict[str, dict]   — per-operator endofunctor maps
  mg._adjunctions   : list[tuple]        — discovered (F_name, G_name) pairs
  mg._nat_transforms: list[tuple]        — discovered (F_name, G_name, relation)
  mg._higher_order  : list[tuple]        — (F_name, expressed_as) relations
"""

from __future__ import annotations

from typing import Optional

from ..core.morphism import MorphismGraph, Atom, Composition


# ── Atom value helpers ─────────────────────────────────────────────────────────

def _atom_value(mg: MorphismGraph, atom_id: int) -> Optional[str]:
    """Return string value of atom_id, or None if not an Atom."""
    sym = mg.symbols[atom_id]
    return sym.value if isinstance(sym, Atom) else None


def _atom_int(mg: MorphismGraph, atom_id: int) -> Optional[int]:
    """Return integer value of an atom if its string value parses as int."""
    v = _atom_value(mg, atom_id)
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _id_to_value_map(mg: MorphismGraph) -> dict[int, str]:
    """Return {atom_id: string_value} for all atoms."""
    return {sid: sym.value for sid, sym in enumerate(mg.symbols)
            if isinstance(sym, Atom)}


# ── Endofunctor extraction ─────────────────────────────────────────────────────

def extract_unary_pairs(
    mg: MorphismGraph,
    op_value: str,
    num_e: int,
    eq_e: int,
) -> dict[int, int]:
    """Extract {arg_id: result_id} endofunctor map for a unary operator.

    Matches the composition chain:  op --num_e--> arg --eq_e--> eq --num_e--> result
    Reads directly from mg.rules and mg._out; does not scan original sequences.

    Returns {} if the operator was not observed or no compositions formed.
    """
    op_id = mg.atoms.get(op_value)
    eq_id = mg.atoms.get('eq')
    if op_id is None or eq_id is None:
        return {}

    pairs: dict[int, int] = {}

    # Scan all composition rules for C1 = compose(op_id, num_e, arg_id)
    for comp_id, (left, etype, right) in mg.rules.items():
        if left != op_id or etype != num_e:
            continue
        arg_id = right
        C1 = comp_id

        # C2 = compose(C1, eq_e, eq_id)
        C2 = mg.rules_inv.get((C1, eq_e, eq_id))
        if C2 is None:
            continue

        # What follows C2 via num_e?  The most frequent target is the result.
        out = mg._out.get(C2, {}).get(num_e, {})
        if not out:
            continue

        best_result = max(out, key=out.get)
        pairs[arg_id] = best_result

    return pairs


def extract_binary_pairs(
    mg: MorphismGraph,
    op_value: str,
    num_e: int,
    eq_e: int,
) -> dict[tuple[int, int], int]:
    """Extract {(arg1_id, arg2_id): result_id} for a binary operator.

    Matches: op --num_e--> arg1 --num_e--> arg2 --eq_e--> eq --num_e--> result
    """
    op_id = mg.atoms.get(op_value)
    eq_id = mg.atoms.get('eq')
    if op_id is None or eq_id is None:
        return {}

    pairs: dict[tuple[int, int], int] = {}

    # C1 = compose(op_id, num_e, arg1_id)
    for comp_id1, (left1, etype1, right1) in mg.rules.items():
        if left1 != op_id or etype1 != num_e:
            continue
        arg1_id = right1
        C1 = comp_id1

        # Navigate forward: what arg2 values follow C1 via num_e?
        arg2_out = mg._out.get(C1, {}).get(num_e, {})
        for arg2_id in arg2_out:
            # C2 = compose(C1, num_e, arg2_id)
            C2 = mg.rules_inv.get((C1, num_e, arg2_id))
            if C2 is None:
                continue

            # C3 = compose(C2, eq_e, eq_id)
            C3 = mg.rules_inv.get((C2, eq_e, eq_id))
            if C3 is None:
                continue

            out = mg._out.get(C3, {}).get(num_e, {})
            if not out:
                continue

            best_result = max(out, key=out.get)
            pairs[(arg1_id, arg2_id)] = best_result

    return pairs


def extract_ternary_pairs(
    mg: MorphismGraph,
    op_value: str,
    num_e: int,
) -> dict[tuple[int, int, int], int]:
    """Extract {(arg1_id, arg2_id, arg3_id): result_id} for ternary operators.

    Matches sequences WITHOUT 'eq': op arg1 arg2 arg3 result.
    The result is the most frequent atom following the 4-token context.

    Used for operators like 'bernoulli' whose compact form is
      bernoulli P1 V1 P2 V2  (no 'eq' separator).
    """
    op_id = mg.atoms.get(op_value)
    if op_id is None:
        return {}

    pairs: dict[tuple[int, int, int], int] = {}

    # C1 = compose(op_id, num_e, arg1_id)
    for comp_id1, (left1, etype1, right1) in mg.rules.items():
        if left1 != op_id or etype1 != num_e:
            continue
        arg1_id = right1
        C1 = comp_id1

        # Navigate forward: what arg2 values follow C1 via num_e?
        arg2_out = mg._out.get(C1, {}).get(num_e, {})
        for arg2_id in arg2_out:
            C2 = mg.rules_inv.get((C1, num_e, arg2_id))
            if C2 is None:
                continue

            # What arg3 values follow C2 via num_e?
            arg3_out = mg._out.get(C2, {}).get(num_e, {})
            for arg3_id in arg3_out:
                C3 = mg.rules_inv.get((C2, num_e, arg3_id))
                if C3 is None:
                    continue

                # The result follows C3 via num_e (no 'eq')
                out = mg._out.get(C3, {}).get(num_e, {})
                if not out:
                    continue

                best_result = max(out, key=out.get)
                pairs[(arg1_id, arg2_id, arg3_id)] = best_result

    return pairs


def discover_endofunctors(mg: MorphismGraph, topo) -> dict[str, dict]:
    """Build endofunctor maps for all operators found in mg.atoms.

    Returns {op_value: endofunctor_map} where:
      - unary   operators map: {arg_id: result_id}
      - binary  operators map: {(arg1_id, arg2_id): result_id}
      - ternary operators map: {(arg1_id, arg2_id, arg3_id): result_id}

    An operator is treated as binary if its endofunctor map from the binary
    extractor is non-empty; otherwise the unary extractor is tried.
    """
    reg = topo.registry
    # Topology may not have all four math edge types; guard safely.
    try:
        num_e = reg.code('num')
        eq_e  = reg.code('eq')
    except KeyError:
        # Fallback: use first registered etype for both (e.g. sequence_1d)
        next_e = reg.code(reg.names()[0])
        num_e = eq_e = next_e

    # Known operator sets for the math domain.
    # Operators NOT in these sets (e.g. 'eval', 'conserve') are too complex
    # for simple frame matching — skip them in Phase 17a.
    _UNARY   = {'succ', 'pred', 'sq', 'sqrt'}
    _BINARY  = {'add', 'sub', 'mul', 'pow', 'ke', 'pe', 'vadd'}
    _TERNARY = {'bernoulli'}   # compact format: op A B C → result  (no 'eq')

    result: dict[str, dict] = {}

    for op_value in mg.atoms:
        if op_value in _UNARY:
            m = extract_unary_pairs(mg, op_value, num_e, eq_e)
            if m:
                result[op_value] = m
        elif op_value in _BINARY:
            m = extract_binary_pairs(mg, op_value, num_e, eq_e)
            if m:
                result[op_value] = m
        elif op_value in _TERNARY:
            m = extract_ternary_pairs(mg, op_value, num_e)
            if m:
                result[op_value] = m

    return result


# ── Adjunction detection ───────────────────────────────────────────────────────

def discover_adjunctions(
    mg: MorphismGraph,
    endofunctors: dict[str, dict],
) -> list[tuple[str, str, float]]:
    """Discover (F, G) adjunction pairs from endofunctor maps.

    Tests the round-trip identity: G(F(A, B), B) = A.
    For binary F, binary G: check sub(add(A,B), B) = A etc.
    For unary F, unary G:   check pred(succ(N)) = N etc.

    Returns list of (F_name, G_name, coverage) where coverage is the fraction
    of F's training pairs for which the round-trip holds.  Only pairs with
    coverage >= 0.9 are returned.
    """
    found: list[tuple[str, str, float]] = []

    id_to_val = _id_to_value_map(mg)

    # ── Unary pairs ───────────────────────────────────────────────────────────
    unary_ops = {op: m for op, m in endofunctors.items()
                 if m and not isinstance(next(iter(m)), tuple)}

    for F_name, F_map in unary_ops.items():
        for G_name, G_map in unary_ops.items():
            if F_name == G_name:
                continue
            # Check G(F(N)) = N for all N in F_map
            total = 0
            correct = 0
            for N_id, M_id in F_map.items():
                total += 1
                roundtrip = G_map.get(M_id)
                if roundtrip == N_id:
                    correct += 1
            if total > 0 and correct / total >= 0.9:
                found.append((F_name, G_name, correct / total))

    # ── Binary pairs (strictly 2-element tuple keys) ──────────────────────────
    binary_ops = {op: m for op, m in endofunctors.items()
                  if m and isinstance(next(iter(m)), tuple)
                  and len(next(iter(m))) == 2}

    for F_name, F_map in binary_ops.items():
        for G_name, G_map in binary_ops.items():
            if F_name == G_name:
                continue
            # Check G(F(A,B), B) = A: i.e. if F maps (A,B)→C, then G maps (C,B)→A
            total = 0
            correct = 0
            for (A_id, B_id), C_id in F_map.items():
                total += 1
                roundtrip = G_map.get((C_id, B_id))
                if roundtrip == A_id:
                    correct += 1
            if total > 0 and correct / total >= 0.9:
                found.append((F_name, G_name, correct / total))

    return found


# ── Natural transformation discovery ──────────────────────────────────────────

def discover_natural_transformations(
    mg: MorphismGraph,
    endofunctors: dict[str, dict],
) -> list[tuple[str, str, str, float]]:
    """Discover natural transformations between operator endofunctors.

    A natural transformation α: F → G is a coherent family of morphisms
    relating the two operators' induced maps on the number type-group.

    Checks two relations:
      (a) Constant-shift: G(N) - F(N) = k  for all N in common domain
          (detected by integer arithmetic on atom values)
      (b) Composition: G(N) = F(F(N))  for all N  (F iterated twice)

    Returns list of (F_name, G_name, relation_description, coverage).
    """
    found: list[tuple[str, str, str, float]] = []

    # Only unary operators support the simple shift / composition check
    unary_ops = {op: m for op, m in endofunctors.items()
                 if m and not isinstance(next(iter(m)), tuple)}

    for F_name, F_map in unary_ops.items():
        for G_name, G_map in unary_ops.items():
            if F_name >= G_name:   # avoid symmetric duplicates
                continue

            # Common domain: N_ids present in both maps
            common = set(F_map) & set(G_map)
            if len(common) < 3:
                continue

            # Collect (f_val, g_val) integer pairs for analysis
            pairs: list[tuple[int, int]] = []
            for N_id in common:
                f_val = _atom_int(mg, F_map[N_id])
                g_val = _atom_int(mg, G_map[N_id])
                if f_val is not None and g_val is not None:
                    pairs.append((f_val, g_val))

            if len(pairs) < 3:
                continue

            # Test (a): G(N) - F(N) = constant?
            diffs = [g - f for f, g in pairs]
            if len(set(diffs)) == 1:
                k = diffs[0]
                found.append((
                    F_name, G_name,
                    f"G(N) = F(N) + {k}  (constant shift)",
                    1.0,
                ))
                continue

            # Test (b): G(N) = F(F(N))?
            match = 0
            for N_id in common:
                M_id = F_map.get(N_id)          # F(N)
                if M_id is None:
                    continue
                FF_id = F_map.get(M_id)         # F(F(N))
                if FF_id == G_map.get(N_id):    # == G(N)?
                    match += 1
            coverage = match / max(len(common), 1)
            if coverage >= 0.9:
                found.append((
                    F_name, G_name,
                    f"G = F ∘ F  (F iterated twice)",
                    coverage,
                ))

    return found


# ── Higher-order relational reasoning ─────────────────────────────────────────

def discover_higher_order(
    mg: MorphismGraph,
    endofunctors: dict[str, dict],
) -> list[tuple[str, str, str]]:
    """Discover relations between relations (natural transformations of functors).

    Checks whether one binary operator's rule is expressible as iterated
    application of another:
      mul(N, k) = add(N, add(N, ... N)) — mul as iterated add
      pow(N, k) = mul(N, mul(N, ... N)) — pow as iterated mul

    For each (F_binary, G_binary) pair, checks: for small k (2..5),
    does F(N, k) == G applied k times to N hold for ≥80% of observed pairs?

    Returns list of (F_name, G_name, description).
    """
    found: list[tuple[str, str, str]] = []

    binary_ops = {op: m for op, m in endofunctors.items()
                  if m and isinstance(next(iter(m)), tuple)}
    unary_ops  = {op: m for op, m in endofunctors.items()
                  if m and not isinstance(next(iter(m)), tuple)}

    # Check mul = iterated add; pow = iterated mul
    candidate_pairs = [('mul', 'add'), ('pow', 'mul')]

    for F_name, G_name in candidate_pairs:
        F_map = binary_ops.get(F_name)
        G_map = binary_ops.get(G_name)
        if F_map is None or G_map is None:
            continue

        match = 0
        total = 0

        for (N_id, K_id), result_id in F_map.items():
            K_int = _atom_int(mg, K_id)
            N_int = _atom_int(mg, N_id)
            result_int = _atom_int(mg, result_id)
            if K_int is None or N_int is None or result_int is None:
                continue
            if K_int < 2 or K_int > 9:
                continue

            # Try: G applied K times to (N, N)
            # For add: add(N, N*(K-1)) but we test via the G_map
            # Accumulate: start with N, add N repeatedly
            acc_id = N_id
            ok = True
            for _ in range(K_int - 1):
                acc_id = G_map.get((acc_id, N_id))
                if acc_id is None:
                    ok = False
                    break

            total += 1
            if ok and acc_id == result_id:
                match += 1

        if total > 0 and match / total >= 0.8:
            found.append((
                F_name, G_name,
                f"{F_name}(N, k) = {G_name} applied k times to N  "
                f"(coverage {match}/{total})",
            ))

    return found


# ── Public API ─────────────────────────────────────────────────────────────────

def build_rule_store(mg: MorphismGraph, topo) -> None:
    """Run all Phase 17a discovery algorithms and store results on mg.

    After this call, mg carries:
      mg._endofunctors   : {op_value: {arg_id: result_id} or {(a,b): c}}
      mg._adjunctions    : [(F_name, G_name, coverage), ...]
      mg._nat_transforms : [(F_name, G_name, description, coverage), ...]
      mg._higher_order   : [(F_name, G_name, description), ...]

    Idempotent: safe to call multiple times (rebuilds from current state).
    """
    ef = discover_endofunctors(mg, topo)
    mg._endofunctors   = ef

    mg._adjunctions    = discover_adjunctions(mg, ef)
    mg._nat_transforms = discover_natural_transformations(mg, ef)
    mg._higher_order   = discover_higher_order(mg, ef)


def predict_via_rules(
    mg: MorphismGraph,
    ctx_id: int,
    etype: int,
) -> dict[int, float]:
    """Attempt to predict next atom using the stored endofunctor maps.

    Called as level-0 back-off in the prediction chain.  Returns a
    distribution {result_atom_id: 1.0} if an endofunctor rule matches,
    or {} if no match.

    Matching strategy: decompose ctx_id to its constituent atoms, then
    check whether the atom sequence ends in a known operator frame:
      Unary:  [op, arg, 'eq']       → look up _endofunctors[op][arg_id]
      Binary: [op, arg1, arg2, 'eq'] → look up _endofunctors[op][(a1,a2)]

    The check requires etype to be the 'num' edge type (predicting a number
    result).  Returns {} for all other edge types.
    """
    endofunctors = getattr(mg, '_endofunctors', None)
    if not endofunctors:
        return {}

    # Decompose ctx_id to the constituent atom (symbol_id, value) pairs
    atom_seq: list[tuple[int, str]] = []
    _decompose(mg, ctx_id, atom_seq)

    if not atom_seq:
        return {}

    # Frame must end with 'eq'
    if atom_seq[-1][1] != 'eq':
        return {}

    n = len(atom_seq)

    # ── Unary frame: [op, arg, eq] ────────────────────────────────────────────
    if n == 3:
        op_val  = atom_seq[0][1]
        arg_id  = atom_seq[1][0]
        ef_map  = endofunctors.get(op_val)
        if ef_map is not None and not isinstance(next(iter(ef_map), None), tuple):
            result_id = ef_map.get(arg_id)
            if result_id is not None:
                return {result_id: 1.0}

    # ── Binary frame: [op, arg1, arg2, eq] ───────────────────────────────────
    if n == 4:
        op_val  = atom_seq[0][1]
        arg1_id = atom_seq[1][0]
        arg2_id = atom_seq[2][0]
        ef_map  = endofunctors.get(op_val)
        if ef_map is not None and isinstance(next(iter(ef_map), None), tuple):
            result_id = ef_map.get((arg1_id, arg2_id))
            if result_id is not None:
                return {result_id: 1.0}

    return {}


def _decompose(
    mg: MorphismGraph,
    symbol_id: int,
    out: list[tuple[int, str]],
) -> None:
    """Recursively decompose symbol_id to its leaf atoms (left-to-right DFS).

    Appends (atom_id, atom_value) pairs to out.  Stops at Atoms or when
    the rule is absent (pruned composition).  Non-recursive depth limit
    prevents runaway on very deep compositions.
    """
    sym = mg.symbols[symbol_id]
    if isinstance(sym, Atom):
        out.append((symbol_id, sym.value))
        return
    rule = mg.rules.get(symbol_id)
    if rule is None:
        return   # pruned or missing — treat as leaf
    left, _etype, right = rule
    # Depth guard: don't expand past 20 atoms (all math frames are ≤ 6)
    if len(out) >= 20:
        return
    _decompose(mg, left,  out)
    _decompose(mg, right, out)
