"""visual_symbol.py — Phase R6: visual symbol learning from pixel context statistics.

Architecture
============
Phase R6 applies the distributional hypothesis (Phase O) to pixel patches:

    Phase O:    word w appears in context c  →  POS category (syntactic)
    Phase R6:   patch p appears next to q    →  visual symbol category (glyph)

The SAME ``discover_categories_from_dists()`` machinery that discovered POS
categories from text (Phase O) is reused here — verbatim — with one change:
*tokens are 8×8 pixel-patch hashes rather than word strings.*

This demonstrates the generality of the distributional approach.  No labels,
no OCR model, no game-specific feature engineering.  Structure emerges from
sequential co-occurrence statistics alone.

Two-level discovery
===================
Level 1 — Glyph clusters (letter-like categories)
    1. Extract 8×8 patches from text-region foveal images.
    2. Quantize each patch → compact hex hash (3 bits/pixel; noise-tolerant).
    3. Build bigram context: P(neighbor_hash | patch_hash) within each row.
    4. ``discover_categories_from_dists()`` → glyph cluster assignments.
    Result: patches of the same letter are in the same cluster.

Level 2 — Word clusters (word-like categories)
    1. Convert foveal patches → glyph-id sequences (one per foveal frame).
    2. Build bigram context: P(next_glyph_sequence | current_glyph_sequence).
    3. ``discover_categories_from_dists()`` → word cluster assignments.
    Result: different renders of the same word are in the same cluster.

Level 3 — Causal cross-reference
    Which word clusters precede reward-generating actions?
    P(action | word_cluster) → cluster → probable command label.

Design principle
================
**THE MODEL MUST NEVER BE DESIGNED AROUND A SPECIFIC TASK. THE MODEL MUST BE GENERAL.**

Zero game-specific logic.  No hardcoded letter templates, no OCR weights,
no font database.  The only inputs are: (a) pixel patches from FovealAttention,
(b) action/reward tuples from the causal history.

See AIF_ROADMAP.md for the full Phase R implementation plan.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants (callers can override via VisualSymbolLearner constructor)
# ---------------------------------------------------------------------------

_DEFAULT_PATCH_SIZE: int = 8    # pixels per side of each glyph patch
_DEFAULT_STRIDE:     int = 8    # non-overlapping default
_DEFAULT_QUANT_BITS: int = 3    # bits per pixel in quantized hash → 8 grey levels


# ---------------------------------------------------------------------------
# VisualGlyphStream  (analogous to the token stream in Phase O)
# ---------------------------------------------------------------------------

@dataclass
class VisualGlyphStream:
    """Rolling stream of quantized patch tokens with bigram context statistics.

    Each :meth:`observe` call appends a horizontal row of patches (one foveal
    text-region image).  Context is the immediate left/right neighbours within
    each row — the visual equivalent of Phase O word bigrams.

    The stream is bounded at ``max_tokens`` to control memory; oldest entries
    are discarded when the cap is exceeded.

    Attributes
    ----------
    patch_size      Side length (pixels) of each square patch.
    stride          Horizontal + vertical step between patch origins.
    quant_bits      Quantization depth (2=coarse..4=fine; 3 recommended).
    max_tokens      Rolling window size (patch-token count).
    context_counts  {patch_hash: Counter({ctx_hash: count})} — raw bigrams.
    patch_counts    {patch_hash: total_observations}.
    stream          Ordered patch-hash sequence (most-recent ``max_tokens``).
    """

    patch_size:     int = _DEFAULT_PATCH_SIZE
    stride:         int = _DEFAULT_STRIDE
    quant_bits:     int = _DEFAULT_QUANT_BITS
    max_tokens:     int = 20_000

    # Mutable fields — use default_factory to avoid shared-state bugs.
    context_counts: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    patch_counts:   Dict[str, int]   = field(default_factory=Counter)
    stream:         List[str]        = field(default_factory=list)

    def observe(self, foveal: np.ndarray) -> int:
        """Process one foveal image; update context statistics.

        Parameters
        ----------
        foveal  H×W or H×W×C image array (any dtype).  Will be converted to
                float32 greyscale before patch extraction.

        Returns
        -------
        int  Number of patch tokens extracted from this frame.
        """
        gray    = _to_gray_f32(foveal)
        patches = _extract_patches(gray, self.patch_size, self.stride)
        if not patches:
            return 0

        hashes = [_quantize(p, self.quant_bits) for p in patches]

        for i, h in enumerate(hashes):
            self.patch_counts[h] += 1
            self.stream.append(h)
            # Symmetric bigram context within this text-row (left ↔ right).
            if i > 0:
                left = hashes[i - 1]
                self.context_counts[h][left]    += 1
                self.context_counts[left][h]    += 1

        # Trim stream to rolling window.
        if len(self.stream) > self.max_tokens:
            self.stream = self.stream[-self.max_tokens:]

        return len(hashes)

    @property
    def n_unique(self) -> int:
        """Number of unique patch hashes observed."""
        return len(self.patch_counts)

    @property
    def n_tokens(self) -> int:
        """Total tokens in the rolling stream."""
        return len(self.stream)

    def to_dists(
        self,
        min_count: int = 3,
    ) -> Tuple[Dict, Dict]:
        """Normalise context counts to probability distributions.

        Returns
        -------
        dists
            ``{(hash,): {(ctx_hash,): probability}}`` — ready for
            ``discover_categories_from_dists()``.
        input_counts
            ``{(hash,): observation_count}``.
        """
        dists:        Dict = {}
        input_counts: Dict = {}
        for h, ctx in self.context_counts.items():
            total = sum(ctx.values())
            if total < min_count:
                continue
            dists[(h,)]        = {(c,): cnt / total for c, cnt in ctx.items()}
            input_counts[(h,)] = self.patch_counts[h]
        return dists, input_counts

    def summary(self) -> str:
        """Human-readable status string."""
        top5 = sorted(self.patch_counts.items(), key=lambda kv: -kv[1])[:5]
        top_str = ', '.join(f'{h[:6]}..:{n}' for h, n in top5)
        return (
            f'VisualGlyphStream: {self.n_tokens} tokens, '
            f'{self.n_unique} unique hashes\n'
            f'  top-5 most-frequent: {top_str}'
        )


# ---------------------------------------------------------------------------
# VisualSymbolLearner  (main class)
# ---------------------------------------------------------------------------

class VisualSymbolLearner:
    """Learn visual symbol (glyph / word) categories from pixel context statistics.

    This is the Phase R6 component.  It wraps a :class:`VisualGlyphStream` and
    drives the two-level discovery process:

    * :meth:`observe`           — feed foveal text-region patches (learning).
    * :meth:`discover_glyphs`   — cluster patch hashes → letter-like categories.
    * :meth:`discover_words`    — cluster glyph sequences → word-like categories.
    * :meth:`decode`            — decode a foveal patch → glyph cluster ids.
    * :meth:`word_cluster`      — identify the word cluster for a foveal patch.
    * :meth:`causal_crossref`   — map word clusters → probable command labels.

    Parameters
    ----------
    patch_size         Side length of glyph patches in pixels.  Should match
                       the font's per-character grid in the target environment.
    stride             Step between patch origins.  Equal to ``patch_size``
                       gives non-overlapping tiles (default).
    quant_bits         Quantization depth.  3 bits → 8 grey levels; robust to
                       minor anti-aliasing variation.
    n_glyph_clusters   Target number of letter-like categories.  Set ≥ the
                       number of distinct characters visible on screen.
    n_word_clusters    Target number of word-like categories.  Set ≥ the
                       expected on-screen vocabulary size.
    max_tokens         Rolling window cap for the patch stream (patch count).
    """

    def __init__(
        self,
        patch_size:         int = _DEFAULT_PATCH_SIZE,
        stride:             int = _DEFAULT_STRIDE,
        quant_bits:         int = _DEFAULT_QUANT_BITS,
        n_glyph_clusters:   int = 32,
        n_word_clusters:    int = 64,
        max_tokens:         int = 20_000,
    ) -> None:
        self.patch_size       = patch_size
        self.stride           = stride
        self.quant_bits       = quant_bits
        self.n_glyph_clusters = n_glyph_clusters
        self.n_word_clusters  = n_word_clusters

        # Level-1 patch stream and glyph assignments.
        self._stream = VisualGlyphStream(
            patch_size = patch_size,
            stride     = stride,
            quant_bits = quant_bits,
            max_tokens = max_tokens,
        )
        self._glyph: Dict[str, int] = {}   # patch_hash → glyph_cluster_id

        # Level-2 word stream and word assignments.
        self._word_stream:  List[tuple]                = []
        self._word_ctx:     Dict[tuple, Dict[tuple, int]] = defaultdict(Counter)
        self._word_counts:  Dict[tuple, int]            = Counter()
        self._word: Dict[tuple, int] = {}              # glyph_seq_tuple → word_cluster_id

    # ------------------------------------------------------------------
    # Level-1: Observation and glyph discovery
    # ------------------------------------------------------------------

    def observe(self, foveal: np.ndarray) -> int:
        """Feed one foveal text-region patch into the glyph stream.

        Call this whenever FovealAttention.step() returns ``text_region=True``.

        Parameters
        ----------
        foveal  Foveal image patch (H, W) or (H, W, C), any dtype.

        Returns
        -------
        int  Number of patch tokens extracted.
        """
        return self._stream.observe(foveal)

    def discover_glyphs(self, min_observations: int = 3) -> Dict[str, int]:
        """Cluster patch hashes into glyph categories (Phase O on pixels).

        Uses ``discover_categories_from_dists()`` — identical to the
        function that discovers POS categories in Phase O.

        Parameters
        ----------
        min_observations  Minimum context observations for a patch hash to
                          be included in clustering.  Low-frequency hashes
                          are noise (partial/occluded characters) and are
                          excluded.

        Returns
        -------
        Dict[str, int]
            ``{patch_hash: glyph_cluster_id}``; empty if insufficient data.
        """
        # Lazy import to avoid circular dependency at module load.
        try:
            from synthesis import discover_categories_from_dists  # type: ignore[import]
        except ImportError:
            raise ImportError(
                'discover_categories_from_dists not found.  '
                'Ensure visual_symbol.py is loaded from within symbolic_ai/.'
            )

        dists, input_counts = self._stream.to_dists(min_count=min_observations)
        n_eligible = len(dists)
        if n_eligible < 2:
            return {}

        k = min(self.n_glyph_clusters, n_eligible)
        assignment = discover_categories_from_dists(
            dists,
            input_counts,
            n_clusters   = k,
            min_examples = min_observations,
            method       = 'kmeans',
        )
        self._glyph = {h: cid for (h,), cid in assignment.items()}
        return dict(self._glyph)

    def decode(self, foveal: np.ndarray) -> List[Optional[int]]:
        """Decode a foveal patch to a sequence of glyph cluster ids.

        Parameters
        ----------
        foveal  Foveal image (H, W) or (H, W, C).

        Returns
        -------
        List[Optional[int]]
            One entry per sub-patch, in left-to-right order.
            ``None`` means the patch hash was not assigned a cluster yet.
        """
        gray    = _to_gray_f32(foveal)
        patches = _extract_patches(gray, self.patch_size, self.stride)
        return [self._glyph.get(_quantize(p, self.quant_bits)) for p in patches]

    # ------------------------------------------------------------------
    # Level-2: Word observation and word discovery
    # ------------------------------------------------------------------

    def observe_word(self, foveal: np.ndarray) -> Optional[tuple]:
        """Decode ``foveal`` → glyph sequence and record for word clustering.

        Parameters
        ----------
        foveal  Foveal image.  Glyphs must have been discovered first.

        Returns
        -------
        tuple or None
            The classified glyph-id sequence; ``None`` if no glyphs known yet.
        """
        if not self._glyph:
            return None

        seq = tuple(g for g in self.decode(foveal) if g is not None)
        if not seq:
            return None

        self._word_counts[seq] += 1
        self._word_stream.append(seq)

        # Symmetric bigram context over the word stream.
        if len(self._word_stream) >= 2:
            prev = self._word_stream[-2]
            self._word_ctx[seq][prev]  += 1
            self._word_ctx[prev][seq]  += 1

        return seq

    def discover_words(self, min_observations: int = 3) -> Dict[tuple, int]:
        """Cluster glyph sequences into word-like categories.

        Same distributional clustering as :meth:`discover_glyphs`, one level
        up.  Each glyph sequence is treated as a "token"; its context is the
        surrounding sequences in the visual stream.

        Returns
        -------
        Dict[tuple, int]
            ``{glyph_seq_tuple: word_cluster_id}``; empty if insufficient data.
        """
        try:
            from synthesis import discover_categories_from_dists  # type: ignore[import]
        except ImportError:
            raise ImportError(
                'discover_categories_from_dists not found.  '
                'Ensure visual_symbol.py is loaded from within symbolic_ai/.'
            )

        dists:        Dict = {}
        input_counts: Dict = {}
        for seq, ctx in self._word_ctx.items():
            total = sum(ctx.values())
            if total < min_observations:
                continue
            dists[(seq,)]        = {(s,): cnt / total for s, cnt in ctx.items()}
            input_counts[(seq,)] = self._word_counts[seq]

        n_eligible = len(dists)
        if n_eligible < 2:
            return {}

        k = min(self.n_word_clusters, n_eligible)
        assignment = discover_categories_from_dists(
            dists,
            input_counts,
            n_clusters   = k,
            min_examples = min_observations,
            method       = 'kmeans',
        )
        self._word = {seq: cid for (seq,), cid in assignment.items()}
        return dict(self._word)

    def word_cluster(self, foveal: np.ndarray) -> Optional[int]:
        """Return the word cluster id for the glyph sequence in ``foveal``.

        Returns ``None`` if glyphs or words have not been discovered yet, or
        if the glyph sequence is unrecognised.
        """
        if not self._glyph or not self._word:
            return None
        seq = self.observe_word(foveal)
        if seq is None:
            return None
        return self._word.get(seq)

    # ------------------------------------------------------------------
    # Level-3: Causal cross-reference
    # ------------------------------------------------------------------

    def causal_crossref(
        self,
        action_history: List[Tuple[np.ndarray, str, float]],
        min_confidence: float = 0.6,
        min_support:    int   = 3,
    ) -> Dict[int, str]:
        """Map word clusters to their probable command/action labels.

        When the agent takes action A after observing visual word cluster W
        and receives reward > 0, W is a candidate label for A.

        The distributional hypothesis applied causally:
        "You shall know a visual word by the action it accompanies."

        Parameters
        ----------
        action_history  List of (foveal_image, action_str, reward) triples
                        collected during exploration.
        min_confidence  Minimum P(action | word_cluster) to report.
        min_support     Minimum co-occurrence count required.

        Returns
        -------
        Dict[int, str]
            ``{word_cluster_id: action_str}`` for high-confidence matches.
        """
        if not self._glyph:
            return {}

        cooc:          Dict[Tuple[int, str], int] = Counter()
        cluster_total: Dict[int, int]              = Counter()

        for foveal, action, reward in action_history:
            if reward <= 0.0:
                continue
            wc = self.word_cluster(foveal)
            if wc is None:
                continue
            cooc[(wc, action)]  += 1
            cluster_total[wc]   += 1

        result: Dict[int, str] = {}
        for wc, total in cluster_total.items():
            if total < min_support:
                continue
            # Best action for this cluster.
            best, best_cnt = '', 0
            for (cluster, action), cnt in cooc.items():
                if cluster == wc and cnt > best_cnt:
                    best, best_cnt = action, cnt
            if best and best_cnt / total >= min_confidence:
                result[wc] = best

        return result

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> str:
        """Human-readable summary of the current learning state."""
        n_gclusters = len(set(self._glyph.values()))
        n_wclusters = len(set(self._word.values()))
        return (
            f'VisualSymbolLearner:\n'
            f'  {self._stream.summary()}\n'
            f'  glyph clusters: {n_gclusters}/{self.n_glyph_clusters}\n'
            f'  word  clusters: {n_wclusters}/{self.n_word_clusters}\n'
            f'  word  stream:   {len(self._word_stream)} sequences'
        )


# ---------------------------------------------------------------------------
# Module-level convenience  (mirrors discover_goals API style)
# ---------------------------------------------------------------------------

def discover_visual_symbols(
    foveal_patches:   List[np.ndarray],
    n_glyph_clusters: int = 32,
    patch_size:       int = _DEFAULT_PATCH_SIZE,
    stride:           int = _DEFAULT_STRIDE,
    quant_bits:       int = _DEFAULT_QUANT_BITS,
    min_observations: int = 3,
) -> Tuple['VisualSymbolLearner', Dict[str, int]]:
    """Convenience: feed a batch of foveal patches and return discovered glyphs.

    Parameters
    ----------
    foveal_patches    List of foveal image arrays (H, W) or (H, W, C).
    n_glyph_clusters  Target number of glyph categories.
    patch_size        Size of each sub-patch.
    stride            Stride between sub-patches.
    quant_bits        Quantization depth.
    min_observations  Minimum observations to include a patch hash.

    Returns
    -------
    (learner, glyph_assignment)
        The fitted :class:`VisualSymbolLearner` and its glyph assignment dict.
    """
    learner = VisualSymbolLearner(
        patch_size       = patch_size,
        stride           = stride,
        quant_bits       = quant_bits,
        n_glyph_clusters = n_glyph_clusters,
    )
    for patch in foveal_patches:
        learner.observe(patch)
    assignment = learner.discover_glyphs(min_observations=min_observations)
    return learner, assignment


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_gray_f32(image: np.ndarray) -> np.ndarray:
    """Convert any image to float32 grayscale in [0, 1]."""
    if image.ndim == 2:
        img = image.astype(np.float32)
    elif image.ndim == 3 and image.shape[2] >= 3:
        # ITU-R BT.601 luminance weights.
        img = (
            0.299 * image[:, :, 0].astype(np.float32) +
            0.587 * image[:, :, 1].astype(np.float32) +
            0.114 * image[:, :, 2].astype(np.float32)
        )
    else:
        img = image[:, :, 0].astype(np.float32)

    mn, mx = float(img.min()), float(img.max())
    if mx - mn > 1e-6:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img)
    return img


def _extract_patches(
    image:      np.ndarray,
    patch_size: int,
    stride:     int,
) -> List[np.ndarray]:
    """Extract a grid of square patches from a 2D float32 image.

    Parameters
    ----------
    image       2D float32 array (H, W) in [0, 1].
    patch_size  Side length of each patch in pixels.
    stride      Step between patch origins (horizontal and vertical).

    Returns
    -------
    List of (patch_size, patch_size) float32 arrays in row-major order
    (top-left to bottom-right).  Empty list if the image is smaller than
    one patch.
    """
    H, W = image.shape[:2]
    if H < patch_size or W < patch_size:
        return []
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patches.append(image[y : y + patch_size, x : x + patch_size].copy())
    return patches


def _quantize(patch: np.ndarray, bits: int) -> str:
    """Quantize a float32 patch to a compact hex-encoded hash string.

    Each pixel is mapped to ``[0, 2^bits − 1]`` and the resulting integer
    array is byte-packed (two pixels per byte for bits ≤ 4).  The hash is
    returned as a hex string.

    Two patches of the same glyph rendered in the same font at the same
    pixel size will produce the same hash (or near-identical hashes with
    bits=3, providing tolerance for 1-level grey quantization noise).

    Parameters
    ----------
    patch  Float32 array (patch_size, patch_size) in [0, 1].
    bits   Quantization depth.  2 = very coarse (4 grey levels);
           3 = coarse (8 levels, recommended);
           4 = medium (16 levels).

    Returns
    -------
    str  Hex-encoded byte string (length = patch_size² × bits // 8).
    """
    levels = (1 << bits) - 1
    q = np.clip(np.round(patch.ravel() * levels), 0, levels).astype(np.uint8)
    if bits <= 4:
        # Pack pairs of quantized values into single bytes.
        if len(q) % 2 != 0:
            q = np.concatenate([q, np.zeros(1, dtype=np.uint8)])
        packed = (q[0::2] << 4) | q[1::2]
        return packed.tobytes().hex()
    else:
        return q.tobytes().hex()
