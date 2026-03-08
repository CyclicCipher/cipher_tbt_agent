"""language_prior.py -- Character bigram language model for OCR rescoring.

Learns the syntactic/graphemic structure of a language from raw character
sequences (no tokenization, no linguistic pre-processing).  The model is a
character bigram P(c_t | c_{t-1}) trained on ground-truth transcriptions.

This is used with GlyphReader.read_patch_topk() to decode a character
sequence via Viterbi decoding:

    combined_score(c_t) = alpha * log P(visual_patch | c_t)
                        + (1 - alpha) * log P(c_t | c_{t-1})

This is exactly a Hidden Markov Model:
  - Hidden states   = true characters
  - Observations    = pixel patches (GlyphReader top-K candidates)
  - Transition prob = character bigram language model
  - Emission prob   = GlyphReader visual confidence

Reference: Fritz et al. 2024 -- Hidden Markov models and the Bayes filter
in categorical probability (arXiv:2401.14669).  The Viterbi algorithm here
is the MAP inference algorithm for discrete HMMs.

Design principle: no language-specific rules, no dictionaries hardcoded.
Structure comes from data distributions alone.
"""
from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


_FLOOR_PROB: float = 1e-6   # probability floor for unseen bigrams
_LOG_FLOOR:  float = math.log(_FLOOR_PROB)
_BOS:        str   = "\x02"  # beginning-of-sequence sentinel
_EOS:        str   = "\x03"  # end-of-sequence sentinel


class LanguagePrior:
    """Character bigram language model + vocabulary for OCR rescoring.

    Usage
    -----
    Training (one-time)::

        prior = LanguagePrior()
        prior.train(["line of text 1", "line of text 2", ...])
        prior.save("lang_prior.pkl")

    Inference::

        prior = LanguagePrior.load("lang_prior.pkl")

        # candidates: List[List[(char, visual_score)]] from GlyphReader.read_patch_topk()
        best_string = prior.viterbi(candidates, alpha=0.6)

    Parameters
    ----------
    alpha   Weight of visual score vs language prior (see viterbi()).
            alpha=1.0 = pure visual (no language model).
            alpha=0.0 = pure language model (ignores GlyphReader).
            alpha=0.6 is a reasonable default.
    """

    def __init__(self) -> None:
        # Bigram counts/probs: _bigram_probs[c1][c2] = P(c2 | c1)
        self._bigram_probs: Dict[str, Dict[str, float]] = {}
        # Unigram frequencies: _char_freq[c] = P(c)
        self._char_freq: Dict[str, float] = {}
        # Word vocabulary (lowercased)
        self._vocab: Set[str] = set()
        # Precomputed numpy transition matrix (built lazily after load/train)
        self._chars:        Optional[List[str]]   = None
        self._char_to_idx:  Optional[Dict[str, int]] = None
        self._trans_logp:   Optional[np.ndarray]  = None   # (K, K) float32
        self._unigram_logp: Optional[np.ndarray]  = None   # (K,) float32
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts:          List[str],
        min_word_count: int = 2,
        verbose:        bool = True,
    ) -> "LanguagePrior":
        """Build character bigram model and vocabulary from raw text strings.

        No tokenization.  Each string is processed as a raw character sequence.
        The model learns transition probabilities between adjacent characters,
        including spaces and punctuation.

        Parameters
        ----------
        texts           Iterable of plain-text strings (one per GT line).
        min_word_count  Min occurrences for a word to enter the vocabulary.
        verbose         Print a training summary.
        """
        bigrams:     Dict[str, Counter] = defaultdict(Counter)
        char_counts: Counter            = Counter()
        word_counts: Counter            = Counter()

        n_chars_total = 0
        for text in texts:
            text = text.strip()
            if not text:
                continue
            prev = _BOS
            for ch in text:
                bigrams[prev][ch] += 1
                char_counts[ch]   += 1
                prev = ch
                n_chars_total += 1
            bigrams[prev][_EOS] += 1

            for w in text.split():
                word_counts[w.lower()] += 1

        # Normalise bigrams to probabilities
        self._bigram_probs = {}
        for c1, nexts in bigrams.items():
            total = sum(nexts.values())
            if total > 0:
                self._bigram_probs[c1] = {c2: cnt / total
                                          for c2, cnt in nexts.items()}

        # Character unigram frequency
        total_chars = sum(char_counts.values()) or 1
        self._char_freq = {c: cnt / total_chars for c, cnt in char_counts.items()}

        # Vocabulary
        self._vocab = {w for w, cnt in word_counts.items()
                       if cnt >= min_word_count}

        self._trained = True

        # Pre-build numpy matrix for fast Viterbi
        self._build_matrix()

        if verbose:
            n_bigrams = sum(len(v) for v in self._bigram_probs.values())
            print(f"LanguagePrior trained: {len(self._char_freq)} chars, "
                  f"{n_bigrams} bigrams, {len(self._vocab)} vocab words, "
                  f"{n_chars_total:,} training chars")

        return self

    # ------------------------------------------------------------------
    # Numpy transition matrix (precomputed for fast Viterbi)
    # ------------------------------------------------------------------

    def _build_matrix(self) -> None:
        """Precompute (K, K) log-prob transition matrix for numpy Viterbi."""
        chars = sorted(self._char_freq.keys())
        K = len(chars)
        char_to_idx = {c: i for i, c in enumerate(chars)}

        trans = np.full((K, K), _LOG_FLOOR, dtype=np.float32)
        for i, c1 in enumerate(chars):
            row = self._bigram_probs.get(c1, {})
            for j, c2 in enumerate(chars):
                p = row.get(c2, _FLOOR_PROB)
                trans[i, j] = math.log(max(p, _FLOOR_PROB))

        unigram = np.array(
            [math.log(max(self._char_freq.get(c, _FLOOR_PROB), _FLOOR_PROB))
             for c in chars],
            dtype=np.float32,
        )

        self._chars       = chars
        self._char_to_idx = char_to_idx
        self._trans_logp  = trans
        self._unigram_logp = unigram

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def viterbi(
        self,
        candidates: List[List[Tuple[str, float]]],
        alpha:      float = 0.6,
    ) -> str:
        """Viterbi MAP decoding over a character sequence.

        Finds the character string s* = argmax_s ∏_t P(visual_t | s_t) × P(s_t | s_{t-1}).

        Parameters
        ----------
        candidates  Per-position visual evidence: each element is a list of
                    (char, visual_score) pairs from GlyphReader.read_patch_topk().
                    Scores should sum to ~1.0 (normalized).
        alpha       Visual weight in [0, 1].  The combined score is:
                        alpha × log_visual + (1-alpha) × log_transition

        Returns
        -------
        Most probable character string (same length as candidates).
        """
        if not candidates or self._chars is None:
            # Fallback: greedy best visual candidate
            return "".join(
                (cands[0][0] if cands else "") for cands in candidates
            )

        T = len(candidates)
        K = len(self._chars)

        # Build visual log-prob array: (T, K)
        # vis_logp[t, k] = log P(visual observation t | char k)
        vis_logp = np.full((T, K), _LOG_FLOOR, dtype=np.float32)
        for t, cands in enumerate(candidates):
            for ch, score in cands:
                idx = self._char_to_idx.get(ch)
                if idx is not None:
                    vis_logp[t, idx] = math.log(max(score, _FLOOR_PROB))

        # Viterbi DP (vectorized)
        dp = np.full((T, K), -np.inf, dtype=np.float64)
        bp = np.zeros((T, K), dtype=np.int32)

        # t = 0: initialise with unigram prior + visual
        dp[0] = (alpha * vis_logp[0]
                 + (1.0 - alpha) * self._unigram_logp)

        # t > 0: forward pass
        # trans_logp[i, j] = log P(char_j | char_i)
        trans = self._trans_logp  # (K, K)
        for t in range(1, T):
            # prev: (K,) → broadcast to (K, K): prev[i] + trans[i, j]
            scores = dp[t - 1, :, np.newaxis] + (1.0 - alpha) * trans  # (K, K)
            bp[t] = np.argmax(scores, axis=0)                            # (K,)
            dp[t] = np.max(scores, axis=0) + alpha * vis_logp[t]        # (K,)

        # Backtrack
        path = np.empty(T, dtype=np.int32)
        path[T - 1] = int(np.argmax(dp[T - 1]))
        for t in range(T - 1, 0, -1):
            path[t - 1] = bp[t, path[t]]

        return "".join(self._chars[idx] for idx in path)

    def word_score(self, word: str) -> float:
        """Return 1.0 if word is in vocabulary, else 0.0."""
        return 1.0 if word.lower() in self._vocab else 0.0

    def char_log_prob(self, c_prev: str, c_next: str) -> float:
        """Log P(c_next | c_prev) with floor smoothing."""
        p = self._bigram_probs.get(c_prev, {}).get(c_next, _FLOOR_PROB)
        return math.log(max(p, _FLOOR_PROB))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save language model to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump({
                "bigram_probs": self._bigram_probs,
                "char_freq":    self._char_freq,
                "vocab":        self._vocab,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "LanguagePrior":
        """Load a previously trained LanguagePrior from disk."""
        with open(path, "rb") as f:
            d = pickle.load(f)
        lp = cls()
        lp._bigram_probs = d["bigram_probs"]
        lp._char_freq    = d["char_freq"]
        lp._vocab        = d["vocab"]
        lp._trained      = True
        lp._build_matrix()   # rebuild numpy arrays
        return lp

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        n_bigrams = sum(len(v) for v in self._bigram_probs.values())
        return (
            f"LanguagePrior: {len(self._char_freq)} chars, "
            f"{n_bigrams} bigrams, {len(self._vocab)} vocab words"
        )

    def top_bigrams(self, c: str, n: int = 5) -> List[Tuple[str, float]]:
        """Return top-N most likely characters after c."""
        row = self._bigram_probs.get(c, {})
        return sorted(row.items(), key=lambda x: -x[1])[:n]

    def __repr__(self) -> str:
        status = "trained" if self._trained else "untrained"
        return f"LanguagePrior({status}, K={len(self._char_freq)})"
