"""Domain classifier — identify which domain a problem belongs to.

Given observations (sets of predicate-value tuples), determine:
1. Which known domain this matches (if any)
2. Whether this is a new domain requiring discovery
3. Whether this is a variant of a known domain (functor candidate)

The classifier uses the PREDICATE SIGNATURE — which predicates appear,
their arities, and their value types — as a fingerprint for each domain.
This is domain-general: no hardcoded domain names or predicate names.

For a multi-purpose system, this is the "router" that decides where
to send incoming problems.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from .types import Domain, Predicate
import sys, os
_CATPLAN_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _CATPLAN_DIR not in sys.path:
    sys.path.insert(0, _CATPLAN_DIR)
from worlds.base import Observation, Demonstration


# ---------------------------------------------------------------------------
# Domain signature: a fingerprint for a domain
# ---------------------------------------------------------------------------

def domain_signature(domain: Domain) -> frozenset[tuple[str, int, str]]:
    """Extract a signature from a known domain.

    Signature = set of (predicate_name, arity, value_type).
    This uniquely identifies the domain's predicate vocabulary.
    """
    sig = set()
    for pred in domain.predicates.values():
        sig.add((pred.name, pred.arity, "bool"))
    for dp in domain.derived.values():
        sig.add((dp.name, len(dp.param_types), "bool"))
    return frozenset(sig)


def observation_signature(obs: Observation) -> frozenset[tuple[str, int, str]]:
    """Extract a signature from an observation.

    Infers predicate names, arities, and value types from the facts.
    """
    sig = set()
    for pred, args, value in obs.facts:
        if isinstance(value, bool):
            vtype = "bool"
        elif isinstance(value, (int, float)):
            vtype = "numeric"
        else:
            vtype = "other"
        sig.add((pred, len(args), vtype))
    return frozenset(sig)


def demonstration_signature(demo: Demonstration) -> frozenset[tuple[str, int, str]]:
    """Extract a signature from a demonstration (union of all observations)."""
    sig: set[tuple[str, int, str]] = set()
    for t in demo.transitions:
        sig |= observation_signature(t.before)
        sig |= observation_signature(t.after)
    return frozenset(sig)


# ---------------------------------------------------------------------------
# Domain classifier
# ---------------------------------------------------------------------------

class DomainClassifier:
    """Classifies observations into known domains or flags them as new."""

    def __init__(self):
        self._known_domains: dict[str, frozenset] = {}  # name -> signature
        self._domains: dict[str, Domain] = {}

    def register_domain(self, domain: Domain):
        """Register a known domain for future classification."""
        sig = domain_signature(domain)
        self._known_domains[domain.name] = sig
        self._domains[domain.name] = domain

    def classify(self, obs: Observation) -> tuple[str, float, str]:
        """Classify an observation.

        Returns (domain_name, confidence, explanation).
        - If confidence > 0.8: exact or near-exact match.
        - If 0.3 < confidence < 0.8: partial match (functor candidate).
        - If confidence < 0.3: new domain.
        """
        obs_sig = observation_signature(obs)
        if not obs_sig:
            return "unknown", 0.0, "empty observation"

        obs_preds = {(p, a) for p, a, _ in obs_sig}

        best_name = "unknown"
        best_score = 0.0
        best_explanation = "no matching domain"

        for name, domain_sig in self._known_domains.items():
            domain_preds = {(p, a) for p, a, _ in domain_sig}

            # Jaccard similarity on (predicate_name, arity) pairs.
            intersection = obs_preds & domain_preds
            union = obs_preds | domain_preds

            if not union:
                continue

            similarity = len(intersection) / len(union)

            # Bonus for exact predicate name matches (not just arity).
            obs_names = {p for p, _ in obs_preds}
            domain_names = {p for p, _ in domain_preds}
            name_overlap = len(obs_names & domain_names) / max(len(obs_names | domain_names), 1)

            score = 0.6 * similarity + 0.4 * name_overlap

            if score > best_score:
                best_score = score
                best_name = name
                n_match = len(intersection)
                n_obs = len(obs_preds)
                n_domain = len(domain_preds)
                best_explanation = (
                    f"matched {n_match}/{n_obs} obs predicates to "
                    f"{n_match}/{n_domain} domain predicates "
                    f"(Jaccard={similarity:.2f}, name_overlap={name_overlap:.2f})"
                )

        # Determine classification type.
        if best_score > 0.8:
            classification = best_name
        elif best_score > 0.3:
            classification = f"variant_of:{best_name}"
            best_explanation = f"partial match — functor candidate. {best_explanation}"
        else:
            classification = "new_domain"
            best_explanation = f"no sufficient match. {best_explanation}"

        return classification, best_score, best_explanation

    def classify_demonstration(self, demo: Demonstration) -> tuple[str, float, str]:
        """Classify a full demonstration trajectory."""
        if not demo.transitions:
            return "unknown", 0.0, "empty demonstration"

        # Use the union of all observations.
        all_facts: set = set()
        for t in demo.transitions:
            all_facts |= t.before.facts
            all_facts |= t.after.facts
        combined = Observation(facts=frozenset(all_facts))
        return self.classify(combined)

    def suggest_functor(self, source_domain: str,
                        obs: Observation) -> dict[str, str] | None:
        """If classified as a variant, suggest a predicate mapping (proto-functor).

        Returns {obs_predicate: domain_predicate} mapping based on
        arity matching and name similarity.
        """
        if source_domain not in self._domains:
            return None

        domain = self._domains[source_domain]
        obs_sig = observation_signature(obs)
        domain_sig = domain_signature(domain)

        obs_by_arity: dict[int, list[str]] = {}
        for p, a, _ in obs_sig:
            obs_by_arity.setdefault(a, []).append(p)

        domain_by_arity: dict[int, list[str]] = {}
        for p, a, _ in domain_sig:
            domain_by_arity.setdefault(a, []).append(p)

        mapping: dict[str, str] = {}
        for arity in obs_by_arity:
            if arity not in domain_by_arity:
                continue
            obs_preds = obs_by_arity[arity]
            dom_preds = domain_by_arity[arity]

            # Match by name similarity (exact match first, then prefix).
            for op in obs_preds:
                for dp in dom_preds:
                    if op == dp:
                        mapping[op] = dp
                        break
                    if op.startswith(dp) or dp.startswith(op):
                        mapping[op] = dp
                        break

        return mapping if mapping else None


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from engine.parser import parse_file
    from worlds.chemistry import WaterFormation
    from worlds.circuits import CircuitsWorld
    from worlds.physics import constant_velocity_scenario

    # Register known domains.
    classifier = DomainClassifier()

    for domain_file in [
        "domains/blocks/domain.catplan",
        "domains/logistics/domain.catplan",
        "domains/arithmetic/domain.catplan",
    ]:
        domains, _ = parse_file(domain_file)
        classifier.register_domain(domains[0])

    # Classify observations from unknown worlds.
    print("=== Domain Classification ===\n")

    print("Chemistry world:")
    w = WaterFormation()
    obs = w.observe()
    name, score, explanation = classifier.classify(obs)
    print(f"  -> {name} (score={score:.2f}): {explanation}\n")

    print("Circuits world:")
    w = CircuitsWorld()
    obs = w.observe()
    name, score, explanation = classifier.classify(obs)
    print(f"  -> {name} (score={score:.2f}): {explanation}\n")

    print("Physics world:")
    w = constant_velocity_scenario()
    obs = w.observe()
    name, score, explanation = classifier.classify(obs)
    print(f"  -> {name} (score={score:.2f}): {explanation}\n")
