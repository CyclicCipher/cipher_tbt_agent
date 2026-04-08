"""Category discovery from sensorimotor triples.

Given (feature, displacement, next_feature) triples accumulated by
a cortical column, discover the categorical structure:
- Objects: equivalence classes of features
- Morphisms: equivalence classes of displacements
- Composition: how morphisms chain (Cayley table)
- Classification: what algebraic structure (Z, Z², Z/nZ, etc.)

This is a PROGRAM that operates ON a column's data, not something
that lives inside the column. The column collects raw experience;
this program discovers structure from that experience.

Based on: automata learning (L*, Hopcroft), partition refinement,
Krohn-Rhodes decomposition, tensor analysis of transitions.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiscoveredCategory:
    """The result of category discovery: objects, morphisms, composition."""

    # Objects: equivalence classes of features.
    # {class_id: set of original features in that class}
    objects: dict[int, set[str]] = field(default_factory=dict)

    # Feature → object class assignment.
    feature_to_object: dict[str, int] = field(default_factory=dict)

    # Morphisms: equivalence classes of displacements.
    # {morphism_id: set of original displacements in that class}
    morphisms: dict[int, set] = field(default_factory=dict)

    # Displacement → morphism class assignment.
    displacement_to_morphism: dict = field(default_factory=dict)

    # Transition table: (object_id, morphism_id) → object_id
    # This IS the category's composition structure.
    transitions: dict[tuple[int, int], int] = field(default_factory=dict)

    # Composition table (Cayley table): (morph_id, morph_id) → morph_id
    # Discovered from chained transitions.
    composition: dict[tuple[int, int], int] = field(default_factory=dict)

    # Algebraic classification (if determined).
    algebra_type: str = "unknown"

    def n_objects(self) -> int:
        return len(self.objects)

    def n_morphisms(self) -> int:
        return len(self.morphisms)

    def describe(self) -> str:
        lines = [
            f"Category: {self.algebra_type}",
            f"  Objects: {self.n_objects()} classes",
            f"  Morphisms: {self.n_morphisms()} classes",
            f"  Transitions: {len(self.transitions)} entries",
            f"  Compositions: {len(self.composition)} entries",
        ]
        return "\n".join(lines)


def discover_category(triples: list[tuple[str, Any, str]],
                      verbose: bool = False) -> DiscoveredCategory:
    """Discover categorical structure from (feature, displacement, feature) triples.

    The main entry point. Runs all four stages:
    1. Object discovery (partition refinement)
    2. Morphism discovery (displacement equivalence)
    3. Composition table (Cayley table from chained transitions)
    4. Algebraic classification

    Args:
        triples: list of (feature_a, displacement, feature_b)
        verbose: print progress

    Returns:
        DiscoveredCategory with objects, morphisms, composition.
    """
    if not triples:
        return DiscoveredCategory()

    cat = DiscoveredCategory()

    # Collect unique features and displacements.
    features = set()
    displacements = set()
    for a, d, b in triples:
        features.add(a)
        features.add(b)
        displacements.add(d)

    if verbose:
        print(f"  Raw: {len(features)} features, {len(displacements)} displacements, "
              f"{len(triples)} triples")

    # --- Stage 1: Object discovery (partition refinement) ---
    # Two features are equivalent if they have the same transition
    # signature: for every displacement, they map to the same target class.
    # This is the Hopcroft/Myhill-Nerode equivalence.

    # Build transition map: (feature, displacement) → set of target features.
    trans_map: dict[tuple[str, Any], set[str]] = defaultdict(set)
    for a, d, b in triples:
        trans_map[(a, d)].add(b)

    # Initial partition: all features in one class.
    # Refine by splitting classes where members have different transition targets.
    feature_list = sorted(features)
    disp_list = sorted(displacements, key=str)

    # Compute signature for each feature: tuple of (displacement → frozenset of targets).
    def signature(feat: str) -> tuple:
        sig = []
        for d in disp_list:
            targets = trans_map.get((feat, d), set())
            sig.append(frozenset(targets))
        return tuple(sig)

    # Group features by signature → equivalence classes.
    sig_to_class: dict[tuple, int] = {}
    class_id = 0
    for feat in feature_list:
        sig = signature(feat)
        if sig not in sig_to_class:
            sig_to_class[sig] = class_id
            cat.objects[class_id] = set()
            class_id += 1
        cid = sig_to_class[sig]
        cat.objects[cid].add(feat)
        cat.feature_to_object[feat] = cid

    # Iterative refinement: re-partition using class-level targets.
    # (One pass is usually sufficient for small alphabets.)
    changed = True
    max_iter = 10
    for iteration in range(max_iter):
        if not changed:
            break
        changed = False

        def class_signature(feat: str) -> tuple:
            sig = []
            for d in disp_list:
                targets = trans_map.get((feat, d), set())
                target_classes = frozenset(cat.feature_to_object.get(t, -1) for t in targets)
                sig.append(target_classes)
            return tuple(sig)

        new_objects: dict[int, set[str]] = {}
        new_f2o: dict[str, int] = {}
        csig_to_class: dict[tuple, int] = {}
        new_cid = 0
        for feat in feature_list:
            csig = class_signature(feat)
            if csig not in csig_to_class:
                csig_to_class[csig] = new_cid
                new_objects[new_cid] = set()
                new_cid += 1
            cid = csig_to_class[csig]
            new_objects[cid].add(feat)
            new_f2o[feat] = cid

        if len(new_objects) != len(cat.objects):
            changed = True
            cat.objects = new_objects
            cat.feature_to_object = new_f2o

    if verbose:
        print(f"  Stage 1: {cat.n_objects()} object classes")
        for cid, members in sorted(cat.objects.items()):
            sample = sorted(members)[:5]
            print(f"    Class {cid}: {sample}{'...' if len(members) > 5 else ''} "
                  f"({len(members)} members)")

    # --- Stage 2: Morphism discovery ---
    # Two displacements are equivalent if they induce the same mapping
    # between object classes.

    def displacement_signature(d: Any) -> dict[int, set[int]]:
        """What object class does each source class map to under d?"""
        mapping: dict[int, set[int]] = {}
        for a, dd, b in triples:
            if dd == d:
                src_class = cat.feature_to_object.get(a, -1)
                tgt_class = cat.feature_to_object.get(b, -1)
                if src_class not in mapping:
                    mapping[src_class] = set()
                mapping[src_class].add(tgt_class)
        return mapping

    disp_sigs: dict[Any, tuple] = {}
    for d in disp_list:
        sig = displacement_signature(d)
        # Convert to hashable form.
        disp_sigs[d] = tuple(sorted(
            (src, frozenset(tgts)) for src, tgts in sig.items()
        ))

    sig_to_morph: dict[tuple, int] = {}
    morph_id = 0
    for d in disp_list:
        sig = disp_sigs[d]
        if sig not in sig_to_morph:
            sig_to_morph[sig] = morph_id
            cat.morphisms[morph_id] = set()
            morph_id += 1
        mid = sig_to_morph[sig]
        cat.morphisms[mid].add(d)
        cat.displacement_to_morphism[d] = mid

    if verbose:
        print(f"  Stage 2: {cat.n_morphisms()} morphism classes")
        for mid, members in sorted(cat.morphisms.items()):
            sample = sorted(members, key=str)[:5]
            print(f"    Morphism {mid}: {sample}{'...' if len(members) > 5 else ''}")

    # --- Stage 3: Transition table ---
    # (object_class, morphism_class) → target_object_class
    # From the observed triples.

    for a, d, b in triples:
        src = cat.feature_to_object.get(a, -1)
        morph = cat.displacement_to_morphism.get(d, -1)
        tgt = cat.feature_to_object.get(b, -1)
        if src >= 0 and morph >= 0 and tgt >= 0:
            cat.transitions[(src, morph)] = tgt

    if verbose:
        print(f"  Stage 3: {len(cat.transitions)} transitions")

    # --- Stage 4: Composition table (Cayley table) ---
    # For morphism pairs (m1, m2), find m3 = m1∘m2 by chaining:
    # If (obj_a, m1) → obj_b AND (obj_b, m2) → obj_c,
    # then m1∘m2 should map obj_a → obj_c.
    # Find which morphism m3 has (obj_a, m3) → obj_c.

    morph_ids = sorted(cat.morphisms.keys())
    obj_ids = sorted(cat.objects.keys())

    for m1 in morph_ids:
        for m2 in morph_ids:
            # For each object, trace m1 then m2.
            composed_map: dict[int, int] = {}
            for obj in obj_ids:
                mid1 = cat.transitions.get((obj, m1))
                if mid1 is not None:
                    mid2 = cat.transitions.get((mid1, m2))
                    if mid2 is not None:
                        composed_map[obj] = mid2

            if not composed_map:
                continue

            # Find which morphism matches this composed mapping.
            composed_sig = tuple(sorted(
                (src, frozenset([tgt])) for src, tgt in composed_map.items()
            ))
            for m3 in morph_ids:
                m3_sig = tuple(sorted(
                    (src, frozenset([tgt]))
                    for (src, m), tgt in cat.transitions.items()
                    if m == m3
                ))
                if composed_sig == m3_sig:
                    cat.composition[(m1, m2)] = m3
                    break

    if verbose:
        print(f"  Stage 4: {len(cat.composition)} composition rules")

    # --- Algebraic classification ---
    _classify(cat, verbose)

    return cat


def _classify(cat: DiscoveredCategory, verbose: bool = False):
    """Attempt to classify the algebraic structure."""
    n_obj = cat.n_objects()
    n_morph = cat.n_morphisms()

    if n_obj == 0 or n_morph == 0:
        cat.algebra_type = "trivial"
        return

    # Check for identity morphism.
    identity = None
    for m in cat.morphisms:
        is_id = True
        for o in cat.objects:
            if cat.transitions.get((o, m)) != o:
                is_id = False
                break
        if is_id:
            identity = m
            break

    # Check if all morphisms have inverses (group vs monoid).
    is_group = identity is not None
    if is_group:
        for m in cat.morphisms:
            has_inverse = False
            for m_inv in cat.morphisms:
                if (cat.composition.get((m, m_inv)) == identity and
                        cat.composition.get((m_inv, m)) == identity):
                    has_inverse = True
                    break
            if not has_inverse:
                is_group = False
                break

    # Check commutativity.
    is_commutative = True
    for m1 in cat.morphisms:
        for m2 in cat.morphisms:
            c12 = cat.composition.get((m1, m2))
            c21 = cat.composition.get((m2, m1))
            if c12 is not None and c21 is not None and c12 != c21:
                is_commutative = False
                break

    # Check for cyclic structure (single generator).
    is_cyclic = False
    generator = None
    if is_group and n_morph > 1:
        for m in cat.morphisms:
            if m == identity:
                continue
            # Check if repeated application of m generates all morphisms.
            generated = {identity}
            current = m
            for _ in range(n_morph + 1):
                generated.add(current)
                next_m = cat.composition.get((current, m))
                if next_m is None:
                    break
                current = next_m
            if len(generated) == n_morph:
                is_cyclic = True
                generator = m
                break

    # Classify.
    if n_morph == 1:
        cat.algebra_type = "trivial (one morphism)"
    elif is_cyclic and n_obj == n_morph:
        cat.algebra_type = f"Z/{n_morph}Z (cyclic group, generator=morphism_{generator})"
    elif is_cyclic:
        cat.algebra_type = f"cyclic (generator=morphism_{generator})"
    elif is_group and is_commutative:
        cat.algebra_type = f"abelian group (order {n_morph})"
    elif is_group:
        cat.algebra_type = f"non-abelian group (order {n_morph})"
    elif is_commutative:
        cat.algebra_type = f"commutative monoid ({n_morph} elements)"
    else:
        cat.algebra_type = f"monoid ({n_morph} elements)"

    if verbose:
        print(f"  Classification: {cat.algebra_type}")
        if identity is not None:
            print(f"    Identity: morphism_{identity}")
        if generator is not None:
            print(f"    Generator: morphism_{generator}")


# -----------------------------------------------------------------------
# Convenience: discover from a column's accumulated triples
# -----------------------------------------------------------------------

def discover_from_column_triples(
    triples: list[tuple[str, tuple, str]],
    quantize_displacement: bool = True,
    verbose: bool = False,
) -> DiscoveredCategory:
    """Discover category from a column's (feature, displacement, feature) triples.

    Optionally quantizes displacement vectors to integer grid
    (avoids float precision issues in equivalence checking).
    """
    processed = []
    for feat_a, disp, feat_b in triples:
        if quantize_displacement and isinstance(disp, tuple):
            disp = tuple(round(d) for d in disp)
        processed.append((feat_a, disp, feat_b))
    return discover_category(processed, verbose=verbose)
