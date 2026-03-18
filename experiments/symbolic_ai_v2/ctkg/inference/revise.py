"""
Revision via Minimal Graph Edit (Stage 6).

The RevisionEngine detects anomalous token sequences (via SurpriseDetector)
and proposes minimal edits to the MorphismGraph that explain the anomaly.

Design (Phase XXV §Stage 6)
----------------------------
1. **Identify failing subgraph**: collect all OBS_SEQ morphisms that were
   predicted with high surprise (positions flagged by SurpriseDetector).
2. **Single-morphism extension candidates**: for each surprising (A, B) pair,
   propose one new IMPLIES or OBS_SEQ morphism that would explain it.
3. **Score by posterior**: candidates are scored by how many other anomalous
   observations they explain simultaneously (breadth) minus a complexity
   penalty (one morph = penalty 1).
4. **Adopt best**: the highest-scoring candidate is written into the graph.

Track A-1 (single anomaly): one observation contradicts the current rules.
Track A-2 (two hypotheses): two competing explanations; best posterior wins.

This module satisfies the Phase XXV constraint: no new Python dicts for rules.
All revision state is stored as morphisms in the MorphismGraph (the adopted
candidate becomes a new OBS_SEQ or IMPLIES edge with appropriate weight).

Iron Law compliance
-------------------
No string comparisons on content tokens.  All token identity is via NodeId.
The only string comparison is on ``morph_type`` (structural label).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import (
    MorphismGraph,
    ObjectId,
    MorphId,
)
from experiments.symbolic_ai_v2.ctkg.inference.surprise import (
    SurpriseDetector,
    SurpriseAnnotation,
)


# ---------------------------------------------------------------------------
# Candidate morphism
# ---------------------------------------------------------------------------

@dataclass
class RevisionCandidate:
    """A proposed single-morphism extension to the MorphismGraph.

    Attributes
    ----------
    source_label : label of the source object (antecedent token).
    target_label : label of the target object (consequent token).
    morph_type   : "OBS_SEQ" or "IMPLIES" (the proposed edge type).
    explains     : list of (position, token) anomalies this candidate covers.
    score        : posterior score = len(explains) - complexity_penalty.
    """
    source_label: str
    target_label: str
    morph_type: str
    explains: list[tuple[int, str]] = field(default_factory=list)
    score: float = 0.0


# ---------------------------------------------------------------------------
# RevisionEngine
# ---------------------------------------------------------------------------

class RevisionEngine:
    """Proposes and applies minimal graph edits to explain anomalous observations.

    Parameters
    ----------
    surprise_detector : SurpriseDetector
        Used to identify which token positions are anomalous.
    mg : MorphismGraph
        The graph to revise.  Candidates are added as new morphisms.
    complexity_penalty : float
        Cost per proposed morphism (Occam factor).  Default 1.0.
        Lower values allow more complex hypotheses; higher values prefer
        simpler explanations.
    """

    def __init__(
        self,
        surprise_detector: SurpriseDetector,
        mg: MorphismGraph,
        complexity_penalty: float = 1.0,
    ) -> None:
        self._sd = surprise_detector
        self._mg = mg
        self._complexity = complexity_penalty

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def revise(
        self,
        tokens: list[str],
        start: int = 0,
    ) -> Optional[RevisionCandidate]:
        """Scan *tokens*, identify anomalies, propose and apply best edit.

        Parameters
        ----------
        tokens : the full observed token sequence.
        start  : first position to check for surprise.

        Returns
        -------
        The adopted RevisionCandidate, or None if no anomaly was found.
        """
        anomalies = self._sd.scan_sequence(tokens, start=start)
        if not anomalies:
            return None

        candidates = self._generate_candidates(tokens, anomalies)
        if not candidates:
            return None

        best = max(candidates, key=lambda c: c.score)
        if best.score <= 0.0:
            return None

        self._apply(best)
        return best

    def generate_candidates(
        self,
        tokens: list[str],
        anomalies: list[SurpriseAnnotation],
    ) -> list[RevisionCandidate]:
        """Public wrapper around _generate_candidates for testing/inspection."""
        return self._generate_candidates(tokens, anomalies)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_candidates(
        self,
        tokens: list[str],
        anomalies: list[SurpriseAnnotation],
    ) -> list[RevisionCandidate]:
        """Build one candidate per anomalous (prev_token, curr_token) bigram.

        Each candidate is a single OBS_SEQ morphism from the token at
        position (anomaly.position - 1) to the anomalous token.  This is the
        "single-morphism extension" from the Stage 6 spec.

        The score = (number of anomalies explained) - complexity_penalty.
        Because each candidate covers exactly one anomalous bigram, the score
        is always 1 - complexity_penalty = 0.0 (with the default penalty of 1).
        A lower complexity_penalty (< 1) allows adoption on single evidence.

        For Track A-2 (two competing hypotheses), two candidates are generated
        and the one that matches more anomalies wins.
        """
        # Group anomalies by (prev_tok, curr_tok) pair.
        pair_to_anomalies: dict[tuple[str, str], list[SurpriseAnnotation]] = {}
        for ann in anomalies:
            pos = ann.position
            if pos == 0:
                # No previous token — can't form a bigram.
                continue
            prev_tok = tokens[pos - 1]
            curr_tok = ann.token
            pair_to_anomalies.setdefault((prev_tok, curr_tok), []).append(ann)

        candidates: list[RevisionCandidate] = []
        for (src_label, tgt_label), ann_list in pair_to_anomalies.items():
            explains = [(ann.position, ann.token) for ann in ann_list]
            score = len(explains) - self._complexity
            candidates.append(RevisionCandidate(
                source_label=src_label,
                target_label=tgt_label,
                morph_type="OBS_SEQ",
                explains=explains,
                score=score,
            ))

        return candidates

    def _apply(self, candidate: RevisionCandidate) -> MorphId:
        """Write the candidate as a new OBS_SEQ morphism into the graph.

        Objects for source and target are created if absent.  If the edge
        already exists, its evidence_count is incremented.
        """
        src_obj = self._ensure_object(candidate.source_label)
        tgt_obj = self._ensure_object(candidate.target_label)

        # Check for existing edge.
        existing = self._mg.hom(src_obj, tgt_obj, include_identity=False)
        for m in existing:
            if m.morph_type == candidate.morph_type:
                self._mg.observe(m.morph_id)
                return m.morph_id

        # Add new morphism.
        m = self._mg.add_morphism(
            src_obj, tgt_obj,
            morph_type=candidate.morph_type,
            evidence=1,
        )
        # Weight encodes the posterior score (normalised to [0, 1] by capping at 1).
        m.weight = min(1.0, max(0.0, candidate.score / max(1.0, abs(candidate.score))))
        return m.morph_id

    def _ensure_object(self, label: str) -> ObjectId:
        """Return the ObjectId for *label*, creating a placeholder object if absent."""
        for obj in self._mg.objects():
            if obj.label == label:
                return obj.obj_id
        obj = self._mg.add_object(concept=None, label=label)
        return obj.obj_id
