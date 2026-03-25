"""
Trace — JSON trace export for activation flow visualisation.

Records the state of the graph at each phase of the observe→spread→select
pipeline. Exports to a JSON file that the HTML visualiser can render.

Each trace record captures one test question and its resolution:
  1. OBSERVATION: which tokens activated, at what level
  2. COOCCUR_SPREAD: which nodes activated via co-occurrence, how much
  3. SIGMA: which edges got sigma boosted, how much, Q·K value per target
  4. FORWARD_SPREAD: signal from context to candidates in select_action
  5. SELECTION: which candidate won, final scores, correct answer

Usage:
    tracer = Tracer(kg)
    tracer.start_record(correct_answer="7")
    tracer.capture_observation(current_nids)
    tracer.capture_cooccur_spread(before_acts, after_acts)
    tracer.capture_sigma(edges_with_sigma)
    tracer.capture_forward(fwd_scores, bwd_scores)
    tracer.capture_selection(chosen, scores)
    tracer.save("trace.json")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


NodeId = int


@dataclass
class TraceRecord:
    """One test question's full activation flow."""
    step: int = 0
    correct_answer: str = ""
    selected_answer: str = ""

    # Phase 1: observation
    observation_tokens: list[dict] = field(default_factory=list)  # [{label, nid, activation}]

    # Phase 2: co-occurrence spread
    cooccur_activations: list[dict] = field(default_factory=list)  # [{label, nid, before, after}]

    # Phase 3: sigma
    sigma_edges: list[dict] = field(default_factory=list)  # [{src, tgt, sigma, qk, role}]
    qk_values: list[dict] = field(default_factory=list)    # [{label, nid, qk}]

    # Phase 4: forward spread in select_action
    forward_scores: list[dict] = field(default_factory=list)  # [{label, nid, fwd, bwd, epistemic, total}]

    # Phase 5: selection
    all_scores: list[dict] = field(default_factory=list)  # [{label, score, is_correct}]


class Tracer:
    """Captures activation flow traces for visualisation."""

    def __init__(self) -> None:
        self._records: list[TraceRecord] = []
        self._current: TraceRecord | None = None
        self.enabled = False

    def start_record(self, step: int = 0, correct_answer: str = "") -> None:
        if not self.enabled:
            return
        self._current = TraceRecord(step=step, correct_answer=correct_answer)

    def capture_observation(self, kg, current_nids: list[NodeId]) -> None:
        if not self.enabled or self._current is None:
            return
        for nid in current_nids:
            node = kg._nodes.get(nid)
            self._current.observation_tokens.append({
                "label": kg.label_for_node(nid),
                "nid": nid,
                "activation": node.activation if node else 0.0,
            })

    def capture_cooccur(self, kg, before: dict[NodeId, float],
                        after: dict[NodeId, float]) -> None:
        if not self.enabled or self._current is None:
            return
        all_nids = set(before) | set(after)
        for nid in sorted(all_nids):
            b = before.get(nid, 0.0)
            a = after.get(nid, 0.0)
            if abs(a - b) > 0.01 or a > 0.01:
                self._current.cooccur_activations.append({
                    "label": kg.label_for_node(nid),
                    "nid": nid,
                    "before": round(b, 3),
                    "after": round(a, 3),
                })

    def capture_sigma(self, kg, active_nids: set[NodeId],
                      qk: dict[NodeId, float]) -> None:
        if not self.enabled or self._current is None:
            return
        # Record Q·K values for each active target.
        for nid in sorted(active_nids):
            v = qk.get(nid, 0.0)
            if v > 0.001:
                self._current.qk_values.append({
                    "label": kg.label_for_node(nid),
                    "nid": nid,
                    "qk": round(v, 3),
                })
        # Record top sigma edges.
        for edge in kg._edges.values():
            if edge.sigma > 0.01:
                self._current.sigma_edges.append({
                    "src": kg.label_for_node(edge.source),
                    "tgt": kg.label_for_node(edge.target),
                    "sigma": round(edge.sigma, 3),
                    "role": edge.role,
                })

    def capture_forward(self, kg, fwd: dict[NodeId, float],
                        bwd: dict[NodeId, float],
                        candidates: list[NodeId]) -> None:
        if not self.enabled or self._current is None:
            return
        for nid in candidates:
            n_edges = len(kg._outgoing.get(nid, ()))
            epistemic = 0.5 if n_edges == 0 else 1.0 / (1.0 + n_edges)
            f = fwd.get(nid, 0.0)
            b = bwd.get(nid, 0.0)
            self._current.forward_scores.append({
                "label": kg.label_for_node(nid),
                "nid": nid,
                "fwd": round(f, 4),
                "bwd": round(b, 4),
                "epistemic": round(epistemic, 4),
                "total": round(1.0 * f + 2.0 * b + 1.5 * epistemic, 4),
            })

    def capture_selection(self, selected_label: str) -> None:
        if not self.enabled or self._current is None:
            return
        self._current.selected_answer = selected_label
        self._records.append(self._current)
        self._current = None

    def save(self, path: str) -> None:
        """Save all records to a JSON file."""
        data = []
        for rec in self._records:
            data.append({
                "step": rec.step,
                "correct": rec.correct_answer,
                "selected": rec.selected_answer,
                "observation": rec.observation_tokens,
                "cooccur": rec.cooccur_activations,
                "qk": rec.qk_values,
                "sigma_edges": rec.sigma_edges[:50],  # limit for file size
                "forward": rec.forward_scores,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def records(self) -> list[TraceRecord]:
        return list(self._records)
