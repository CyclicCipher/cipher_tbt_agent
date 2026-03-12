"""spectral_predict.py — Unified spectral predictor (Phase E).

Replaces the heuristic back-off chain (steps 0a–0f) in core/predict.py with
a principled 4-level predictor derived from the Hankel tensor architecture:

    1. Spectral exact lookup (core/state_space.py)
       ss.predict_dist(atom_buf[-k:]) — tries k = k_max down to k = 1.
       Zero residual for seen k-gram contexts.

    2. Spectral Kan extension (core/state_space.py)
       ss.predict_unseen(atom_buf[-k:]) — Yoneda-correct nearest-neighbour
       plus embedding delta.  No integer parsing; no token-name guards.
       Objects are defined by their co-occurrence geometry (Yoneda lemma).

    3. Cross-domain transfer (reasoning/intertwiner.py)
       transfer_predict(atom_buf[-k:], ss_source, ss_target, eta).
       Activates when an intertwiner is configured (ss_source, eta pair).

    4. Marginal
       Uniform over all atoms observed in the state space.

The SpectralPredictor is trained on raw sequences and optionally receives an
intertwiner configuration.  It does NOT modify the existing MorphismGraph
pipeline — both run in parallel during the migration phase
(ROADMAP_REDESIGN.md §IV.5).

Public API
----------
SpectralPredictor  — trainable predictor with 4-level back-off
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .hankel import HankelEstimator
from .state_space import StateSpace
from .topology import Topology
from ..reasoning.intertwiner import transfer_predict
from ..reasoning.frame_solver import FrameSolver


# ── Intertwiner configuration ─────────────────────────────────────────────────

@dataclass
class IntertwinedDomain:
    """A source domain state space and its intertwiner to the target domain.

    Attributes
    ----------
    ss_source : state space of the source (NL) domain
    eta       : intertwiner matrix (target.rank × source.rank)
    """
    ss_source: StateSpace
    eta:       np.ndarray


# ── SpectralPredictor ─────────────────────────────────────────────────────────

class SpectralPredictor:
    """Unified 5-level spectral predictor.

    Training
    --------
    Call SpectralPredictor.train(seqs, ...) or construct directly from a
    pre-built StateSpace.

    Prediction
    ----------
    predictor.predict(atom_buf) → dict[str, float]

    The 5-level back-off chain:
      1. Spectral exact lookup   — raw_dist k-gram hit
      2. FrameSolver             — adjunction-based (succ⊣pred, add⊣sub, self-adj)
      3. Spectral Kan extension  — nearest-neighbour in state space
      4. Cross-domain transfer   — intertwiner (if configured)
      5. Marginal                — uniform over observed vocabulary

    Generation
    ----------
    predictor.generate(prompt, max_steps, eos) → list[str]
    """

    def __init__(
        self,
        ss:            StateSpace,
        *,
        intertwined:   Optional[list[IntertwinedDomain]] = None,
        atom_buf_size: int = 16,
        frame_solver:  Optional[FrameSolver] = None,
    ) -> None:
        self.ss            = ss
        self.intertwined   = intertwined  or []
        self.atom_buf_size = atom_buf_size
        self._frame_solver = frame_solver

        # Pre-compute marginal distribution over all col_keys
        self._marginal: dict[str, float] = {}
        if ss.col_keys:
            p = 1.0 / len(ss.col_keys)
            self._marginal = {a: p for a in ss.col_keys}

    # ── Class-method constructor ──────────────────────────────────────────

    @classmethod
    def train(
        cls,
        seqs:   list[list[str]],
        topo:   Optional[Topology] = None,
        k_max:  int = 4,
        rank:   Optional[int] = None,
        *,
        intertwined:   Optional[list[IntertwinedDomain]] = None,
    ) -> 'SpectralPredictor':
        """Train a SpectralPredictor from raw sequences.

        Parameters
        ----------
        seqs         : list of token sequences (each a list of strings)
        topo         : Topology (accepted for API consistency, not used)
        k_max        : maximum context width for Hankel estimation
        rank         : SVD rank (None = automatic from singular value elbow)
        intertwined  : pre-built intertwiner configurations for transfer step
        """
        he = HankelEstimator(k_max=k_max)
        for seq in seqs:
            he.observe(seq, topo)
        ss = he.build_state_space(k_max=k_max, rank=rank)
        fs = FrameSolver.build(ss.raw_dist)
        return cls(ss, intertwined=intertwined, frame_solver=fs)

    # ── Core prediction ───────────────────────────────────────────────────

    def predict(self, atom_buf: list[str]) -> dict[str, float]:
        """4-level back-off prediction from a recent-atom buffer.

        Parameters
        ----------
        atom_buf : recent atom string values (most recent LAST, max self.atom_buf_size)

        Returns
        -------
        {atom_str: probability} — normalised distribution over next atoms.
        Returns marginal if all higher levels fail.
        """
        buf = atom_buf[-self.atom_buf_size:]

        # ── Steps 1+2: Exact lookup and FrameSolver, interleaved per k ───
        # At each context length k (longest first), try exact lookup then
        # FrameSolver before shrinking the context.  This prevents a shorter
        # k-gram exact hit (e.g. ('32','eq') → pred(32)=31) from shadowing a
        # FrameSolver adjunction answer at the full frame context
        # (e.g. ('succ','32','eq') → succ⊣pred gives 33).
        for k in range(self.ss.k_max, 0, -1):
            if len(buf) < k:
                continue
            ctx = tuple(buf[-k:])

            # Step 1: Spectral exact lookup
            dist = self.ss.predict_dist(ctx)
            if dist:
                return dist

            # Step 2: FrameSolver adjunction-based prediction
            if self._frame_solver is not None:
                dist = self._frame_solver.predict(ctx)
                if dist:
                    return dist

        # ── Step 3: Spectral Kan extension (Yoneda-correct) ───────────────
        if buf:
            fallback = self.ss.col_keys
            for k in range(min(self.ss.k_max, len(buf)), 0, -1):
                ctx = tuple(buf[-k:])
                dist = self.ss.predict_unseen(ctx, fallback)
                if dist:
                    return dist

        # ── Step 4: Cross-domain transfer ────────────────────────────────
        for dom in self.intertwined:
            for k in range(dom.ss_source.k_max, 0, -1):
                if len(buf) < k:
                    continue
                ctx = tuple(buf[-k:])
                dist = transfer_predict(ctx, dom.ss_source, self.ss, dom.eta)
                if dist:
                    return dist

        # ── Step 5: Marginal ─────────────────────────────────────────────
        return self._marginal.copy()

    # ── Perplexity ────────────────────────────────────────────────────────

    def perplexity(self, test_seq: list[str]) -> float:
        """Bits-per-token on a test sequence using the fast log-prob path.

        Uses ss.predict_log_prob() for seen contexts (O(1) dict lookup via
        raw_dist — no dict construction, no numpy multiply) and falls back
        to the full predict() chain for unseen contexts (rare at k≥1 for
        in-vocabulary tokens).

        Parameters
        ----------
        test_seq : flat list of atom strings (e.g. individual characters)

        Returns
        -------
        Average bits per token (base-2 entropy estimate on the test data).
        """
        import math
        total_nll = 0.0
        n_tokens  = 0
        buf: list[str] = []
        log2_vocab = math.log2(max(len(self.ss.col_keys), 2))

        for atom in test_seq:
            if buf:
                log_p = float('-inf')

                # Try exact MLE via predict_log_prob at each k (longest first)
                for k in range(self.ss.k_max, 0, -1):
                    if len(buf) < k:
                        continue
                    ctx = tuple(buf[-k:])
                    lp  = self.ss.predict_log_prob(ctx, atom)
                    if lp > float('-inf'):
                        log_p = lp
                        break

                # Fallback: full predict() chain (Kan extension / transfer /
                # marginal) for unseen contexts — rare for character-level data
                # after k=1 exact lookup succeeds almost always.
                if log_p == float('-inf'):
                    dist = self.predict(buf)
                    p    = dist.get(atom, 0.0)
                    log_p = math.log2(p) if p > 0.0 else -log2_vocab

                total_nll += -log_p
                n_tokens  += 1

            buf.append(atom)
            if len(buf) > self.atom_buf_size:
                buf = buf[-self.atom_buf_size:]

        return total_nll / max(n_tokens, 1)

    # ── Generation ────────────────────────────────────────────────────────

    def generate(
        self,
        prompt:    list[str],
        max_steps: int = 50,
        eos:       str = '<eos>',
    ) -> list[str]:
        """Autoregressive greedy generation.

        Feeds prompt into the atom buffer, then repeatedly:
          1. Predicts next token via self.predict(atom_buf)
          2. Appends the argmax token to output and atom_buf
          3. Stops on eos token or max_steps

        Parameters
        ----------
        prompt    : list of atom strings to prime the context
        max_steps : maximum tokens to generate after the prompt
        eos       : stop-generation token

        Returns
        -------
        List of generated tokens (not including the prompt).
        """
        atom_buf = list(prompt)[-self.atom_buf_size:]
        output: list[str] = []

        for _ in range(max_steps):
            dist = self.predict(atom_buf)
            if not dist:
                break
            next_tok = max(dist, key=dist.get)
            output.append(next_tok)
            if next_tok == eos:
                break
            atom_buf.append(next_tok)
            if len(atom_buf) > self.atom_buf_size:
                atom_buf = atom_buf[-self.atom_buf_size:]

        return output
