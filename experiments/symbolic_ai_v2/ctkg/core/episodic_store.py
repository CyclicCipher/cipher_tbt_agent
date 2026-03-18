"""Episodic store: surprise-driven event retention with MDL pruning.

An episodic event is one interaction step — a token sequence paired with the
prediction error the agent experienced at that step.  The store retains events
whose prediction error exceeds a configurable threshold, discarding routine
(predictable) steps.  When the buffer overflows, the lowest-surprise events
are pruned first.

Consolidation merges events with identical token sequences, collapsing
repeated experience into a single representative episode.

Priority replay returns events weighted by salience — the highest-salience
events are replayed most often.  Salience blends temporal recency with
prediction surprise:

    salience(e) = recency_alpha * recency(e) + (1 - recency_alpha) * surprise(e)

where recency(e) = 1 / (1 + age) and age = current_max_step - e.step.
This prevents old high-surprise events from dominating indefinitely.

Design notes:
  - No references to MorphismGraph or Predictor here.  The store is a pure
    data structure; the prediction error is computed externally (by the
    AgentLoop or benchmarks) and passed in as a float.
  - Thread-safety is not a concern at this scale.

See CTKG_ARCHITECTURE.md §Episodic for the full specification.
See ROADMAP.md Stage 5, Step 5.1 for the design decisions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EpisodicEvent:
    """One retained interaction step.

    Attributes
    ----------
    step:
        Global timestep index when the event was recorded.
    tokens:
        The token sequence observed at this step (flat list, MSB first).
    prediction_error:
        1 - max_prob from the predictor at this step.
        0.0 = perfect prediction (no surprise).
        1.0 = total surprise (uniform prediction).
    consolidated:
        True after consolidate() has merged duplicate sequences.
    """

    step: int
    tokens: list[str]
    prediction_error: float
    consolidated: bool = False

    def __repr__(self) -> str:
        toks = self.tokens[:5]
        ellipsis = "..." if len(self.tokens) > 5 else ""
        return (
            f"EpisodicEvent(step={self.step}, "
            f"tokens={toks}{ellipsis}, "
            f"pe={self.prediction_error:.3f})"
        )


# ---------------------------------------------------------------------------
# EpisodicStore
# ---------------------------------------------------------------------------

class EpisodicStore:
    """Surprise-driven episodic memory with MDL-motivated pruning.

    Parameters
    ----------
    surprise_threshold:
        Events with prediction_error <= this value are not stored.
        Default: 0.5 (store anything harder than a coin flip).
    max_events:
        Maximum number of events retained in the buffer.  When exceeded,
        the lowest-salience events are pruned.  None = unlimited.
    recency_alpha:
        Weight given to temporal recency vs. surprise in the salience score.
        salience = recency_alpha * recency + (1 - recency_alpha) * surprise
        where recency = 1 / (1 + age), age = current_max_step - event.step.
        Default: 0.3  (70% surprise-driven, 30% recency-driven).
    seed:
        Random seed for replay sampling.
    """

    def __init__(
        self,
        surprise_threshold: float = 0.5,
        max_events: int = 500,
        recency_alpha: float = 0.3,
        seed: int = 42,
    ) -> None:
        self._threshold = surprise_threshold
        self._max_events = max_events
        self._recency_alpha = recency_alpha
        self._rng = random.Random(seed)
        self._events: list[EpisodicEvent] = []

    # ------------------------------------------------------------------
    # Adding events
    # ------------------------------------------------------------------

    def add_event(
        self,
        step: int,
        tokens: list[str],
        prediction_error: float,
    ) -> Optional[EpisodicEvent]:
        """Store a new event if its surprise exceeds the threshold.

        Parameters
        ----------
        step:
            Global step index.
        tokens:
            Observed token sequence.
        prediction_error:
            Surprise signal; 0.0 = perfect prediction, 1.0 = total surprise.

        Returns
        -------
        The stored EpisodicEvent, or None if prediction_error <= threshold.
        """
        if prediction_error <= self._threshold:
            return None

        evt = EpisodicEvent(
            step=step,
            tokens=list(tokens),
            prediction_error=prediction_error,
        )
        self._events.append(evt)

        # Enforce capacity limit immediately
        if self._max_events is not None and len(self._events) > self._max_events:
            self.prune(self._max_events)

        return evt

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_recent(self, n: int = 20) -> list[EpisodicEvent]:
        """Return the n most recently added events (newest first).

        Parameters
        ----------
        n:
            Maximum number of events to return.

        Returns
        -------
        Events in reverse chronological order.
        """
        return list(reversed(self._events[-n:])) if self._events else []

    def replay_batch(self, n: int = 10) -> list[EpisodicEvent]:
        """Sample n events for replay, weighted by salience.

        Higher-salience events are sampled more often.  Salience blends
        temporal recency with prediction surprise:

            salience(e) = alpha * recency(e) + (1 - alpha) * surprise(e)

        Sampling is done with replacement so that small stores can still
        return n events.

        Parameters
        ----------
        n:
            Number of events to return.

        Returns
        -------
        List of n sampled EpisodicEvents (may contain duplicates if n > len).
        Empty list if the store is empty.
        """
        if not self._events:
            return []

        max_step = max(e.step for e in self._events)
        weights = [self._salience(e, max_step) for e in self._events]
        total_w = sum(weights)
        if total_w < 1e-12:
            # All equal weight — uniform sample
            return self._rng.choices(self._events, k=n)

        normalised = [w / total_w for w in weights]
        return self._rng.choices(self._events, weights=normalised, k=n)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def consolidate(self) -> int:
        """Merge events with identical token sequences.

        After consolidation, only the highest-surprise representative of
        each unique token sequence is kept; its prediction_error is set to
        the mean of all merged events.  Merged events are removed.

        Returns
        -------
        Number of events removed (merged).
        """
        before = len(self._events)
        groups: dict[tuple[str, ...], list[EpisodicEvent]] = {}
        for evt in self._events:
            key = tuple(evt.tokens)
            groups.setdefault(key, []).append(evt)

        kept: list[EpisodicEvent] = []
        for key, group in groups.items():
            # Use the most-recent event as representative, set avg PE
            rep = max(group, key=lambda e: e.step)
            rep.prediction_error = sum(e.prediction_error for e in group) / len(group)
            rep.consolidated = True
            kept.append(rep)

        # Preserve chronological order
        kept.sort(key=lambda e: e.step)
        self._events = kept
        return before - len(self._events)

    def prune(self, max_events: Optional[int] = None) -> int:
        """Remove lowest-salience events until at most max_events remain.

        Salience = alpha * recency + (1-alpha) * surprise.  The lowest-salience
        events are pruned first, preserving the most informative and most recent
        episodes.

        Parameters
        ----------
        max_events:
            Target buffer size.  Uses self._max_events if None.

        Returns
        -------
        Number of events removed.
        """
        limit = max_events if max_events is not None else self._max_events
        if limit is None or len(self._events) <= limit:
            return 0

        n_remove = len(self._events) - limit
        max_step = max(e.step for e in self._events)
        # Sort by salience ascending; remove lowest-salience events
        self._events.sort(key=lambda e: self._salience(e, max_step))
        self._events = self._events[n_remove:]
        # Restore chronological order
        self._events.sort(key=lambda e: e.step)
        return n_remove

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _salience(self, event: EpisodicEvent, max_step: int) -> float:
        """Compute salience for an event given the current maximum step.

        salience(e) = alpha * recency(e) + (1 - alpha) * surprise(e)

        where recency(e) = 1 / (1 + age)  and  age = max_step - e.step.

        Parameters
        ----------
        event:
            The event to score.
        max_step:
            The current maximum step in the store (used to compute age).

        Returns
        -------
        Salience ∈ (0, 1].
        """
        age = max_step - event.step
        recency = 1.0 / (1.0 + age)
        surprise = event.prediction_error
        return self._recency_alpha * recency + (1.0 - self._recency_alpha) * surprise

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        n = len(self._events)
        avg_pe = (
            sum(e.prediction_error for e in self._events) / n
            if n > 0 else 0.0
        )
        return (
            f"EpisodicStore(events={n}, "
            f"threshold={self._threshold:.2f}, "
            f"avg_pe={avg_pe:.3f})"
        )
