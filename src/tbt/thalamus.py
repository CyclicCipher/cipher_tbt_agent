"""The thalamus — the inter-column routing fabric (transthalamic; Sherman & Guillery).

A column owns ONE structural map. Composing two structures that interact — a number's DIGIT × its POSITION,
or (later) a task subgoal × a spatial goal — needs them kept FACTORED in separate columns and joined by a
thin channel, NOT fused into one fatter column (that would be the 2^K conjunctive explosion, architecture
doc §2). The thalamus is that channel. Here it provides the cross-column CONJUNCTION:

  bind : route column-C's CONTENT (What) and column-L's LOCATION (Where) and bind them into a register
         R = Σ  content ⊗ place   — Smolensky's tensor product / VSA binding, but across two columns.
  read : given a location in column-L (a context / goal-state), retrieve the content bound there from
         column-C — project the register onto the place code, match against C's content codebook.

The capacity win this buys is the whole reason for multi-column: one column caps at its content codebook
(feat_dim distinct symbols); factoring DIGIT × POSITION lets two small columns represent unbounded numbers
(place value). The same bind/read channel is the substrate for the goal-state control loop (task column
sets a goal-state in the spatial column; doc §5) — built on top of this, gated by the basal ganglia.

The register is returned to the caller (a transient "object" — one number, one bound scene), so many can be
held at once; the thalamus itself is the stateless router. Pure torch.
"""

from __future__ import annotations

import torch


class Thalamus:
    def bind(self, content_col, loc_col, items):
        """items = [(content_label, loc_node), ...]. Returns the conjunction register R = Σ content ⊗ place."""
        R = None
        for label, node in items:
            term = torch.outer(content_col.content_code(label), loc_col.place_code(node))
            R = term if R is None else R + term
        return R

    def read(self, R, content_col, loc_col, loc_node, threshold=0.5, default=None):
        """Bottom-up read: given a location in loc_col, retrieve the content bound there from content_col.
        Returns `default` when nothing is bound at that location (score below threshold)."""
        scores = content_col.L4.E @ (R @ loc_col.place_code(loc_node))
        best = int(scores.argmax())
        return best if scores[best] > threshold else default

    def read_location(self, R, content_col, loc_col, content_label, threshold=0.1, default=None):
        """TOP-DOWN goal-state SET (the §5 control-loop direction): given a CONTENT (a task subgoal),
        retrieve the LOCATION bound to it in loc_col — the task column setting a goal-state in the spatial
        column. The transpose of `read`. Returns loc_col's NODE INDEX (into its place codebook), or
        `default` if nothing is bound there."""
        v = R.t() @ content_col.content_code(content_label)         # the place bound to this content
        scores = loc_col.place @ v
        best = int(scores.argmax())
        return best if scores[best] > threshold else default
