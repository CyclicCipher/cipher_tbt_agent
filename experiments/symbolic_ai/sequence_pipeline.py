"""sequence_pipeline.py — E0-E6 sequence learning for any discrete token stream.

Broca's area principle: one general algorithm, all domains.

    text:    sequences = [line.split() for line in corpus_lines]
    actions: sequences = [episode_actions for episode in game_log]
    music:   sequences = [note_ids for track in midi]
    vision:  sequences = [patch_ids for row in image]   # via VisionLearner

The architecture (domain-agnostic):

    E0: Teach next_token/prev_token bigrams → bidir clustering → K categories
    E1: next_cat(c1,c2)→c3  +  token_given_cat(c1,c2,c3)→tok   [K²+K³ entries]
    E2: Context-sensitive clusters: (c_prev, token) → ctx_cluster_id
    E3: Soft retrieval: sim(c,c') = exp(-T·JSD(succ_dist[c], succ_dist[c']))
    E4: slot_occupants(c_prev, c_next) → token   (paradigmatic axis)
    E5: Sense-splitting: k-means on (c_prev,c_next) context vectors
    E6: Beam-search composition: find chain of existing concepts explaining examples

Usage:
    from sequence_pipeline import SequenceLearner, sequences_from_texts

    learner = SequenceLearner(n_clusters=12)
    learner.fit(sequences_from_texts(train_lines))
    pred = learner.predict(('est', 'in'))
    results = learner.evaluate(sequences_from_texts(test_lines))
"""
from __future__ import annotations

import argparse
import collections
import itertools
import math
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if os.path.join(_HERE, '..') not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, '..'))

import io
if (hasattr(sys.stdout, 'buffer') and
        getattr(sys.stdout, 'encoding', 'utf-8').lower() != 'utf-8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')

from engine import SymbolicAI
from synthesis import discover_categories_from_dists

_HERE_CTKG = os.path.join(os.path.dirname(_HERE), 'ctkg')
if _HERE_CTKG not in sys.path:
    sys.path.insert(0, _HERE_CTKG)
from ctkg.graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Text convenience helper (the ONLY language-specific thing)
# ---------------------------------------------------------------------------

def sequences_from_texts(texts: list[str]) -> list[list[str]]:
    """Convert text lines to word-token sequences."""
    return [t.split() for t in texts]


# ---------------------------------------------------------------------------
# Memory diagnostics
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Return current process RSS in MB, or 0.0 if psutil is unavailable."""
    try:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / 1_048_576
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Internal helpers (pure functions — no class dependency)
# ---------------------------------------------------------------------------

def _jsd(p: dict, q: dict) -> float:
    """Jensen-Shannon divergence (base-2). 0=identical, 1=fully disjoint."""
    result = 0.0
    for k in set(p) | set(q):
        pk, qk = p.get(k, 0.0), q.get(k, 0.0)
        mk = 0.5 * (pk + qk)
        if mk < 1e-12:
            continue
        if pk > 1e-12:
            result += 0.5 * pk * math.log2(pk / mk)
        if qk > 1e-12:
            result += 0.5 * qk * math.log2(qk / mk)
    return result


def _precompute_dist_cache(ai: SymbolicAI, concept_name: str) -> dict:
    """Return {input_key: {output_key: probability}} for a stored concept."""
    store = ai.stores.get(concept_name)
    if store is None:
        return {}
    key_counts: dict = collections.defaultdict(
        lambda: collections.defaultdict(float))
    key_totals: dict = collections.defaultdict(float)
    for inp, out in store.examples:
        key_counts[inp][out] += 1
        key_totals[inp] += 1
    return {k: {o: c / key_totals[k] for o, c in v.items()}
            for k, v in key_counts.items()}


def _ask_soft(query_key: tuple, dist_cache: dict,
              sim_matrix: list, K: int) -> dict | None:
    """Similarity-weighted soft retrieval — distributional analogue of attention."""
    weighted: dict = collections.defaultdict(float)
    total_w = 0.0
    for stored_key, out_dist in dist_cache.items():
        if len(stored_key) != len(query_key):
            continue
        w = 1.0
        for qi_s, ki_s in zip(query_key, stored_key):
            try:
                qi, ki = int(qi_s), int(ki_s)
                w *= (sim_matrix[qi][ki]
                      if 0 <= qi < K and 0 <= ki < K
                      else (1.0 if qi == ki else 0.0))
            except (ValueError, TypeError):
                w *= (1.0 if qi_s == ki_s else 0.0)
            if w < 1e-12:
                break
        if w < 1e-12:
            continue
        for out, p in out_dist.items():
            weighted[out] += w * p
        total_w += w
    if total_w < 1e-12:
        return None
    return {k: v / total_w for k, v in weighted.items()}


def _precompute_all_soft(dist_cache: dict, sim_matrix: list,
                          K: int, arity: int) -> dict:
    """Precompute soft distributions for all K^arity query keys."""
    soft: dict = {}
    for ids in itertools.product(range(K), repeat=arity):
        qk = tuple(str(q) for q in ids)
        soft[qk] = _ask_soft(qk, dist_cache, sim_matrix, K)
    return soft


def _precompute_all_soft_numpy(dist_cache: dict, sim_matrix: list,
                                K: int, arity: int) -> dict:
    """Vectorized replacement: numpy tensor contractions instead of Python loops.
    Speedup ~2000× for arity=3, K=64. Falls back to _precompute_all_soft if numpy missing."""
    try:
        import numpy as np
    except ImportError:
        return _precompute_all_soft(dist_cache, sim_matrix, K, arity)

    S = np.array(sim_matrix, dtype=np.float32)  # (K, K)

    all_outputs = sorted({o for d in dist_cache.values() for o in d})
    out_idx = {o: i for i, o in enumerate(all_outputs)}
    V = len(all_outputs)
    if V == 0:
        return {tuple(str(i) for i in ids): None
                for ids in itertools.product(range(K), repeat=arity)}

    # Build dense tensors
    D = np.zeros((K,) * arity + (V,), dtype=np.float32)
    mask = np.zeros((K,) * arity, dtype=np.float32)
    for stored_key, out_dist in dist_cache.items():
        try:
            idx = tuple(int(s) for s in stored_key)
        except (ValueError, TypeError):
            continue
        if any(i < 0 or i >= K for i in idx):
            continue
        total = sum(out_dist.values())
        if total < 1e-12:
            continue
        mask[idx] = 1.0
        for out_token, prob in out_dist.items():
            if out_token in out_idx:
                D[idx + (out_idx[out_token],)] = prob

    # Sequential tensordot: contract each key dimension with S → query dimension
    result = D
    weight = mask
    for dim in range(arity):
        result = np.tensordot(S, result, axes=([1], [dim]))  # new query dim at front
        weight = np.tensordot(S, weight, axes=([1], [dim]))
    # Dims are reversed after arity contractions — transpose back
    result = np.transpose(result, list(range(arity - 1, -1, -1)) + [arity])
    weight = np.transpose(weight, list(range(arity - 1, -1, -1)))

    # Normalize
    normed = result / np.where(weight > 1e-12, weight, 1.0)[..., np.newaxis]

    # Convert back to dict format expected by predict_e3 / logprob
    soft = {}
    for ids in itertools.product(range(K), repeat=arity):
        qk = tuple(str(i) for i in ids)
        if float(weight[ids]) < 1e-12:
            soft[qk] = None
        else:
            d = {all_outputs[vi]: float(p) for vi, p in enumerate(normed[ids]) if p > 1e-12}
            soft[qk] = d if d else None
    return soft


# ---------------------------------------------------------------------------
# SequenceLearner
# ---------------------------------------------------------------------------

class SequenceLearner:
    """E0-E6 sequence learning pipeline for any discrete token sequences.

    Works on List[List[Any]] where Any is any hashable token type.
    All phases are optional: call fit() for E1-E3, then fit_paradigmatic(),
    fit_polysemy(), synthesize() as needed.
    """

    def __init__(self, ai: SymbolicAI | None = None, n_clusters: int = 12):
        self.ai = ai or SymbolicAI(KnowledgeGraph())
        self.n_clusters = n_clusters
        # Set by fit():
        self.assignment: dict = {}   # token → cluster_id
        self.clusters:   dict = {}   # cluster_id → [tokens]
        self._K: int = n_clusters
        # E3 state — lazy: soft distributions computed on first query, then cached.
        # Eager precomputation of all K³ combinations caused O(K³×V) memory blow-up
        # (e.g. K=64, V=5000 → 5 GB tensor; 24 GB for Latin corpus). Now we store
        # the raw dist caches and sim_matrix, and call _ask_soft() on demand.
        self._nc_soft:   dict = {}   # lazy cache: (c1,c2) → {c3: prob}
        self._wgc_soft:  dict = {}   # lazy cache: (c1,c2,c3) → {token: prob}
        self._nc_cache:  dict = {}   # raw {(c1,c2): {c3: prob}} from next_cat
        self._wgc_cache: dict = {}   # raw {(c1,c2,c3): {token: prob}} from token_given_cat
        self._sim_matrix: list = []  # K×K JSD-based similarity matrix
        self._alpha: float = 0.95
        # E4 state:
        self._frame_profiles: dict = {}

    # ---- E0: bigram discovery -----------------------------------------------

    def discover(self, sequences: list[list[Any]],
                 verbose: bool = False) -> None:
        """E0: Teach next_token / prev_token bigrams from sequences.

        After this, call fit() to cluster tokens and train E1-E3.
        Skip if you already have a pre-built AI (e.g. from discover_structure).
        """
        for name, it, ot in [
            ('next_token', ['token'], ['token']),
            ('prev_token', ['token'], ['token']),
        ]:
            if name not in ai_graph_concepts(self.ai):
                self.ai.add_concept(name=name, domain='sequence',
                                    input_type=it, output_type=ot)
        n_pairs = 0
        for seq in sequences:
            for i in range(len(seq) - 1):
                t1, t2 = str(seq[i]), str(seq[i + 1])
                self.ai.teach('next_token', (t1,), (t2,))
                self.ai.teach('prev_token', (t2,), (t1,))
                n_pairs += 1
        if verbose:
            print(f'  Taught {n_pairs:,} bigram pairs')

    # ---- E1-E3 ---------------------------------------------------------------

    def fit(self, sequences: list[list[Any]],
            bigram_concept: str | None = None,
            e3_temperature: float = 2.0,
            phases: str = 'e1e3',
            verbose: bool = True) -> None:
        """E1-E3: Cluster tokens, train category chain, build soft retrieval.

        Args:
            sequences:       Training sequences (List[List[token]]).
            bigram_concept:  Name of the next-token concept already in ai
                             (e.g. 'next_word_hier' from discover_structure).
                             If None: calls discover() first to teach next_token.
            e3_temperature:  JSD similarity sharpness (higher = more selective).
            phases:          Subset of phases: any combo of 'e1' 'e2' 'e3'.
            verbose:         Print progress.
        """
        run_e2 = 'e2' in phases
        run_e3 = 'e3' in phases

        if verbose:
            _m0 = _rss_mb()
            if _m0:
                print(f'  RSS at fit() start = {_m0:.0f} MB')

        # --- E0: ensure bigrams exist ---
        next_concept = bigram_concept or 'next_token'
        prev_concept = 'prev_' + next_concept.split('next_', 1)[-1]
        if next_concept not in ai_stores(self.ai):
            self.discover(sequences, verbose=verbose)
            next_concept = 'next_token'
            prev_concept = 'prev_token'
        if verbose:
            _m = _rss_mb()
            if _m:
                print(f'  RSS after E0 discover = {_m:.0f} MB')

        # --- Cluster tokens by bidir bigram distributions ---
        if verbose:
            print(f'  Clustering {self._K} token categories '
                  f'from {next_concept}/{prev_concept}...')
        result = self.ai.induce_hierarchy_bidir(
            next_concept, prev_concept, n_clusters=self._K)
        if 'error' in result:
            print(f'  WARNING: clustering failed: {result["error"]}')
            return
        self.assignment = result.get('assignment', {})
        self.clusters   = result.get('clusters', {})
        if verbose:
            print(f'  Vocab: {len(self.assignment):,}  '
                  f'Clusters: {len(self.clusters)}')

        # Free E0 bigram examples — clustering is done, raw pairs no longer needed.
        # _index dicts remain intact for O(1) lookups via ask()/ask_dist().
        for _c in ('next_token', 'prev_token'):
            _s = self.ai.stores.get(_c)
            if _s is not None:
                _s.examples.clear()
        if verbose:
            _m = _rss_mb()
            if _m:
                print(f'  RSS after E0 clear = {_m:.0f} MB')

        # --- E1: category chain ---
        _register_concepts(self.ai, [
            ('next_cat',       ['cat','cat'],      ['cat'],  'sequence'),
            ('token_given_cat',['cat','cat','cat'],['token'],'sequence'),
        ])
        n_used = n_skip = 0
        for seq in sequences:
            for i in range(len(seq) - 2):
                t1, t2, t3 = seq[i], seq[i + 1], seq[i + 2]
                c1 = self.assignment.get(str(t1))
                c2 = self.assignment.get(str(t2))
                c3 = self.assignment.get(str(t3))
                if c1 is None or c2 is None or c3 is None:
                    n_skip += 1; continue
                s1, s2, s3 = str(c1), str(c2), str(c3)
                self.ai.teach('next_cat',        (s1, s2),       (s3,))
                self.ai.teach('token_given_cat', (s1, s2, s3),   (str(t3),))
                n_used += 1
        if verbose:
            print(f'  E1: {n_used:,} trigrams used, {n_skip:,} OOV skipped')
            _m = _rss_mb()
            if _m:
                print(f'  RSS after E1 = {_m:.0f} MB')

        # --- E2: context-sensitive clusters ---
        if run_e2:
            self._ctx_assignment, self._ctx_clusters = _build_ctx_assignment(
                sequences, self.assignment, n_clusters=self._K, verbose=verbose)
            if self._ctx_assignment:
                _register_concepts(self.ai, [
                    ('next_cat_ctx',       ['cat','cat_ctx'],           ['cat_ctx'], 'sequence'),
                    ('token_given_cat_ctx',['cat','cat_ctx','cat_ctx'], ['token'],   'sequence'),
                ])
                _train_chain_ctx(self.ai, sequences, self.assignment,
                                 self._ctx_assignment, verbose=verbose)

        # --- E3: soft retrieval ---
        if run_e3:
            if verbose:
                print('  E3: Building successor distributions...')
            succ_dists = _build_succ_dists(self.ai, self.clusters, next_concept)
            sim_matrix = _build_sim_matrix(succ_dists, self._K, e3_temperature)
            nc_cache   = _precompute_dist_cache(self.ai, 'next_cat')
            wgc_cache  = _precompute_dist_cache(self.ai, 'token_given_cat')

            # Store caches for LAZY soft retrieval. Do NOT precompute all K³ keys:
            # that creates a (K,K,K,V) tensor + K³ Python dicts which is O(K³×V) memory.
            # At K=64, V=72 (vision) → ~1.5 GB; V=5000 (Latin text) → >20 GB.
            # Instead: compute _ask_soft() on first query, cache result.
            self._nc_cache   = nc_cache
            self._wgc_cache  = wgc_cache
            self._sim_matrix = sim_matrix
            self._nc_soft    = {}   # lazy: populated by _get_nc_soft()
            self._wgc_soft   = {}   # lazy: populated by _get_wgc_soft()

            if verbose:
                print(f'  E3: nc_cache {len(nc_cache)} keys  '
                      f'wgc_cache {len(wgc_cache)} keys  '
                      f'(lazy — soft dists computed on first query)')
                _mem = _rss_mb()
                if _mem:
                    print(f'  E3: RSS after caching = {_mem:.0f} MB')

            # Free E1 examples — all information is now in _nc_cache / _wgc_cache.
            for _c in ('next_cat', 'token_given_cat'):
                _s = self.ai.stores.get(_c)
                if _s is not None:
                    _s.examples.clear()

    # ---- Prediction ----------------------------------------------------------

    def predict(self, context: tuple) -> Any | None:
        """Predict next token given a 2-token context. Uses E3 with E1 fallback."""
        if len(context) < 2:
            return None
        t1, t2 = str(context[-2]), str(context[-1])
        return self._predict_e3(t1, t2) or self._predict_e1(t1, t2)

    def predict_dist(self, context: tuple) -> dict[Any, float]:
        """Full distribution over next tokens given context.

        Marginalises over c3: P(t3|c1,c2) = Σ_c3 P(c3|c1,c2) · P(t3|c1,c2,c3).
        """
        if len(context) < 2:
            return {}
        t1, t2 = str(context[-2]), str(context[-1])
        c1 = self.assignment.get(t1)
        c2 = self.assignment.get(t2)
        if c1 is None or c2 is None:
            return {}
        c3_dist = self._get_nc_soft((str(c1), str(c2)))
        if not c3_dist:
            return {}
        result: dict = {}
        for c3_key, p_c3 in c3_dist.items():
            c3 = c3_key[0] if isinstance(c3_key, tuple) else str(c3_key)
            t3_dist = self._get_wgc_soft((str(c1), str(c2), c3))
            if not t3_dist:
                continue
            for tok_key, p_t3 in t3_dist.items():
                tok = tok_key[0] if isinstance(tok_key, tuple) else str(tok_key)
                result[tok] = result.get(tok, 0.0) + p_c3 * p_t3
        return result

    def logprob(self, *tokens: Any) -> float | None:
        """Log₂ P(tokens[-1] | tokens[-3], tokens[-2]). None if not computable."""
        if len(tokens) < 3:
            return None
        t1, t2, t3 = str(tokens[-3]), str(tokens[-2]), str(tokens[-1])
        c1 = self.assignment.get(t1)
        c2 = self.assignment.get(t2)
        c3 = self.assignment.get(t3)
        if c1 is None or c2 is None or c3 is None:
            return None
        c3_dist = self._get_nc_soft((str(c1), str(c2)))
        if c3_dist is None:
            # E1 fallback
            cat_d = self.ai.ask_dist('next_cat', (str(c1), str(c2)))
            if cat_d is None:
                return None
            p_c3 = cat_d.get((str(c3),), 0.0)
            if p_c3 < 1e-12:
                return None
            word_d = self.ai.ask_dist('token_given_cat',
                                       (str(c1), str(c2), str(c3)))
            if word_d is None:
                return None
            p_t3 = word_d.get((t3,), 0.0)
            return math.log2(p_c3 * p_t3) if p_t3 > 1e-12 else None
        p_c3 = c3_dist.get((str(c3),), 0.0)
        if p_c3 < 1e-12:
            return None
        t3_dist = self._get_wgc_soft((str(c1), str(c2), str(c3)))
        if t3_dist is None:
            return None
        p_t3 = t3_dist.get((t3,), 0.0)
        return math.log2(p_c3 * p_t3) if p_t3 > 1e-12 else None

    # ---- Evaluation ----------------------------------------------------------

    def evaluate(self,
                 test_sequences: list[list[Any]],
                 train_pairs: set | None = None,
                 verbose: bool = True) -> dict:
        """Evaluate E3 vs E1 vs flat on test sequences.

        train_pairs: set of (t2, t3) pairs seen in training for unseen split.
        Returns dict with accuracy, unseen_accuracy, coverage, perplexity keys.
        """
        r = dict(total=0,
                 e3_correct=0,   e3_answered=0,
                 e1_correct=0,   e1_answered=0,
                 flat_correct=0, flat_answered=0,
                 unseen_total=0,
                 e3_unseen_correct=0, e3_unseen_answered=0,
                 e1_unseen_correct=0, e1_unseen_answered=0,
                 logloss=0.0, logloss_n=0)
        tp = train_pairs or set()

        for seq in test_sequences:
            for i in range(len(seq) - 2):
                t1, t2, t3 = str(seq[i]), str(seq[i+1]), str(seq[i+2])
                r['total'] += 1
                unseen = (t2, t3) not in tp
                if unseen:
                    r['unseen_total'] += 1

                # E3
                p_e3 = self._predict_e3(t1, t2)
                if p_e3 is not None:
                    r['e3_answered'] += 1
                    if p_e3 == t3:
                        r['e3_correct'] += 1
                    if unseen:
                        r['e3_unseen_answered'] += 1
                        if p_e3 == t3:
                            r['e3_unseen_correct'] += 1

                # E1
                p_e1 = self._predict_e1(t1, t2)
                if p_e1 is not None:
                    r['e1_answered'] += 1
                    if p_e1 == t3:
                        r['e1_correct'] += 1
                    if unseen:
                        r['e1_unseen_answered'] += 1
                        if p_e1 == t3:
                            r['e1_unseen_correct'] += 1

                # Flat bigram
                p_fl = self._predict_flat(t2)
                if p_fl is not None:
                    r['flat_answered'] += 1
                    if p_fl == t3:
                        r['flat_correct'] += 1

                # Log-prob (E3)
                lp = self.logprob(t1, t2, t3)
                if lp is not None:
                    r['logloss'] -= lp; r['logloss_n'] += 1

        if verbose:
            T, UT = r['total'], r['unseen_total']
            def pct(n, d): return f'{100*n/d:.1f}%' if d else 'N/A'
            def ppl(l, n): return f'{2**(l/n):.1f}' if n else 'N/A'
            print(f'\n  Test trigrams: {T:,}  (unseen: {UT:,})')
            print(f'  {"Metric":<36} {"Flat":>8} {"E1":>8} {"E3":>8}')
            print(f'  {"-"*60}')
            T_ = max(T, 1)
            print(f'  {"Coverage":<36} '
                  f'{pct(r["flat_answered"],T_):>8} '
                  f'{pct(r["e1_answered"],T_):>8} '
                  f'{pct(r["e3_answered"],T_):>8}')
            print(f'  {"Accuracy (of answered)":<36} '
                  f'{pct(r["flat_correct"],max(r["flat_answered"],1)):>8} '
                  f'{pct(r["e1_correct"],max(r["e1_answered"],1)):>8} '
                  f'{pct(r["e3_correct"],max(r["e3_answered"],1)):>8}')
            print(f'  {"Accuracy (of total)":<36} '
                  f'{pct(r["flat_correct"],T_):>8} '
                  f'{pct(r["e1_correct"],T_):>8} '
                  f'{pct(r["e3_correct"],T_):>8}')
            UT_ = max(UT, 1)
            print(f'\n  Unseen pairs:')
            print(f'  {"Accuracy (of answered, unseen)":<36} '
                  f'{"N/A":>8} '
                  f'{pct(r["e1_unseen_correct"],max(r["e1_unseen_answered"],1)):>8} '
                  f'{pct(r["e3_unseen_correct"],max(r["e3_unseen_answered"],1)):>8}')
            print(f'  {"Accuracy (of total unseen)":<36} '
                  f'{"N/A":>8} '
                  f'{pct(r["e1_unseen_correct"],UT_):>8} '
                  f'{pct(r["e3_unseen_correct"],UT_):>8}')
            print(f'  E3 perplexity: {ppl(r["logloss"], r["logloss_n"])}')

        # Backward-compat aliases (for parity_test.py and external callers)
        r['chain_answered']       = r['e1_answered']
        r['chain_correct']        = r['e1_correct']
        r['chain_unseen_answered']= r['e1_unseen_answered']
        r['chain_unseen_correct'] = r['e1_unseen_correct']
        r['chain_logloss']        = r['logloss']
        r['chain_logloss_n']      = r['logloss_n']
        r['flat_logloss']         = r['logloss']   # flat logloss not computed separately
        r['flat_logloss_n']       = r['logloss_n']
        r['e3_logloss']           = r['logloss']
        r['e3_logloss_n']         = r['logloss_n']
        r['flat_unseen_answered'] = 0
        r['flat_unseen_correct']  = 0
        return r

    # ---- E4: paradigmatic axis -----------------------------------------------

    def fit_paradigmatic(self, sequences: list[list[Any]],
                          verbose: bool = False) -> None:
        """E4: Learn slot_occupants(c_prev, c_next) → token.

        Captures paradigmatic substitutability: tokens that fill the same
        (category_before, category_after) frames are semantically similar.
        """
        _register_concepts(self.ai, [
            ('slot_occupants', ['cat', 'cat'], ['token'], 'sequence'),
        ])
        for seq in sequences:
            for i in range(len(seq) - 2):
                t1, t2, t3 = seq[i], seq[i+1], seq[i+2]
                c1 = self.assignment.get(str(t1))
                c3 = self.assignment.get(str(t3))
                if c1 is None or c3 is None:
                    continue
                self.ai.teach('slot_occupants', (str(c1), str(c3)), (str(t2),))

        # Build frame profiles: token → distribution over (c_prev, c_next)
        store = self.ai.stores.get('slot_occupants')
        if store is None:
            return
        tok_frames: dict = collections.defaultdict(collections.Counter)
        for (c1s, c3s), (tok,) in store.examples:
            tok_frames[tok][(c1s, c3s)] += 1
        self._frame_profiles = {}
        for tok, frame_counts in tok_frames.items():
            total = sum(frame_counts.values())
            self._frame_profiles[tok] = {
                k: v / total for k, v in frame_counts.items()}

    def paradigmatic_neighbors(self, token: Any, topn: int = 8,
                                temperature: float = 2.0) -> list[tuple]:
        """Tokens most substitutable with `token` in same frames (E4).

        Returns [(token, similarity), ...] sorted by descending similarity.
        Uses exp(-T·JSD) on frame-distribution profiles.
        """
        key = str(token)
        if key not in self._frame_profiles or not self._frame_profiles:
            return []
        p = self._frame_profiles[key]
        results = []
        for other, q in self._frame_profiles.items():
            if other == key:
                continue
            sim = math.exp(-temperature * _jsd(p, q))
            if sim > 1e-6:
                results.append((other, sim))
        return sorted(results, key=lambda x: -x[1])[:topn]

    # ---- E5: polysemy --------------------------------------------------------

    def fit_polysemy(self, sequences: list[list[Any]],
                      candidates: list | None = None,
                      n_senses: int = 2,
                      verbose: bool = False) -> None:
        """E5: Split ambiguous tokens into context-conditional senses.

        Performs k-means on (c_prev, c_next) context vectors for each token
        in `candidates` (or top-entropy tokens if None).
        Adds token_given_sense(c1, sense_label, c3) → token concept.
        """
        if not self.assignment:
            return

        # Collect (c1, c3) contexts for each token occurrence as middle word
        tok_contexts: dict = collections.defaultdict(list)
        for seq in sequences:
            for i in range(len(seq) - 2):
                t1, t2, t3 = seq[i], seq[i+1], seq[i+2]
                c1 = self.assignment.get(str(t1))
                c3 = self.assignment.get(str(t3))
                if c1 is None or c3 is None:
                    continue
                tok_contexts[str(t2)].append((c1, c3))

        if candidates is None:
            # Simple frequency-based selection: tokens with ≥20 occurrences
            candidates = [t for t, ctxs in tok_contexts.items()
                          if len(ctxs) >= 20][:30]

        _register_concepts(self.ai, [
            ('token_given_sense', ['cat', 'sense', 'cat'], ['token'], 'sequence'),
        ])

        for tok in candidates:
            ctxs = tok_contexts.get(tok, [])
            if len(ctxs) < n_senses * 2:
                continue
            # k-means on (c1, c3) pairs
            sense_map = _kmeans_contexts(ctxs, n_senses)
            for ctx_pair, sense_id in sense_map.items():
                label = f'{tok}_{sense_id}'
                # Teach: for each occurrence with this context → assign to sense
                for c1, c3 in ctxs:
                    if sense_map.get((c1, c3)) == sense_id:
                        self.ai.teach('token_given_sense',
                                      (str(c1), label, str(c3)), (tok,))

        if verbose:
            print(f'  E5: polysemy split on {len(candidates)} candidates')

    # ---- E6: meta-synthesis --------------------------------------------------

    def synthesize(self, examples: list[tuple],
                   max_depth: int = 3,
                   beam_width: int = 5,
                   min_coverage: float = 0.05,
                   verbose: bool = False) -> list[tuple]:
        """E6: Find composition chains explaining (input, output) examples.

        Returns [(chain_name, coverage, accuracy), ...] sorted by score.
        """
        concept_names = list(self.ai.stores.keys())
        all_found = []

        # Depth-1: single concepts
        beam = []
        for name in concept_names:
            cov, acc = _score_chain(self.ai, [name], examples)
            if cov >= min_coverage:
                beam.append(([name], cov, acc))
        if verbose:
            print(f'  Depth 1: {len(beam)} chains ≥{min_coverage:.0%} coverage')
        all_found.extend(beam)
        beam = sorted(beam, key=lambda x: -(x[1]*x[2]))[:beam_width]

        # Depth 2+: extend
        for depth in range(2, max_depth + 1):
            candidates = []
            for chain, _, _ in beam:
                for name in concept_names:
                    extended = chain + [name]
                    cov, acc = _score_chain(self.ai, extended, examples)
                    if cov >= min_coverage:
                        candidates.append((extended, cov, acc))
            if not candidates:
                break
            candidates.sort(key=lambda x: -(x[1]*x[2]))
            all_found.extend(candidates)
            beam = candidates[:beam_width]

        # Deduplicate and format
        seen, unique = set(), []
        for chain, cov, acc in sorted(all_found, key=lambda x: -(x[1]*x[2])):
            name = ' ∘ '.join(chain)
            if name not in seen:
                seen.add(name)
                unique.append((name, cov, acc))
        return unique

    # ---- Private helpers -----------------------------------------------------

    def _get_nc_soft(self, key: tuple) -> dict | None:
        """Lazy soft retrieval for next_cat. Computes and caches on first access."""
        if key not in self._nc_soft:
            self._nc_soft[key] = _ask_soft(key, self._nc_cache,
                                           self._sim_matrix, self._K)
        return self._nc_soft[key]

    def _get_wgc_soft(self, key: tuple) -> dict | None:
        """Lazy soft retrieval for token_given_cat. Computes and caches on first access."""
        if key not in self._wgc_soft:
            self._wgc_soft[key] = _ask_soft(key, self._wgc_cache,
                                            self._sim_matrix, self._K)
        return self._wgc_soft[key]

    def _predict_e1(self, t1: str, t2: str) -> str | None:
        c1 = self.assignment.get(t1)
        c2 = self.assignment.get(t2)
        if c1 is None or c2 is None:
            return None
        c3t = self.ai.ask('next_cat', (str(c1), str(c2)))
        if c3t is None:
            return None
        c3 = c3t[0] if isinstance(c3t, tuple) else str(c3t)
        w3t = self.ai.ask('token_given_cat', (str(c1), str(c2), c3))
        if w3t is None:
            return None
        return w3t[0] if isinstance(w3t, tuple) else str(w3t)

    def _predict_e3(self, t1: str, t2: str) -> str | None:
        c1 = self.assignment.get(t1)
        c2 = self.assignment.get(t2)
        if c1 is None or c2 is None:
            return None
        c3d = self._get_nc_soft((str(c1), str(c2)))
        if c3d is None:
            return None
        c3_tup = max(c3d, key=c3d.get)
        c3 = c3_tup[0] if isinstance(c3_tup, tuple) else str(c3_tup)
        wd = self._get_wgc_soft((str(c1), str(c2), c3))
        if wd is None:
            return None
        best = max(wd, key=wd.get)
        return best[0] if isinstance(best, tuple) else str(best)

    def _predict_flat(self, t2: str) -> str | None:
        # Try both concept names for compatibility
        for name in ('next_token', 'next_word_hier'):
            r = self.ai.ask(name, (t2,))
            if r is not None:
                return r[0] if isinstance(r, tuple) else str(r)
        return None


# ---------------------------------------------------------------------------
# Module-level helpers used by SequenceLearner
# ---------------------------------------------------------------------------

def ai_graph_concepts(ai: SymbolicAI) -> set:
    return set(ai.graph.concepts.keys())


def ai_stores(ai: SymbolicAI) -> set:
    return set(ai.stores.keys())


def _register_concepts(ai: SymbolicAI,
                        specs: list[tuple[str, list, list, str]]) -> None:
    """Register concepts in graph if not already present."""
    for name, in_types, out_types, domain in specs:
        if name not in ai.graph.concepts:
            ai.add_concept(name=name, domain=domain,
                           input_type=in_types, output_type=out_types)


def _build_succ_dists(ai: SymbolicAI, clusters: dict,
                       next_concept: str = 'next_token') -> dict:
    """Successor-distribution per cluster (task-relevant similarity for E3)."""
    succ: dict = {}
    for cid, members in clusters.items():
        merged: dict = collections.defaultdict(float)
        n = 0
        for tok in members:
            d = ai.ask_dist(next_concept, (str(tok),))
            if d is None:
                continue
            for out_tup, prob in d.items():
                k = out_tup[0] if isinstance(out_tup, tuple) else str(out_tup)
                merged[k] += prob
            n += 1
        succ[cid] = ({k: v / n for k, v in merged.items()} if n else {})
    return succ


def _build_sim_matrix(succ_dists: dict, K: int,
                       temperature: float = 2.0) -> list:
    """K×K similarity matrix: sim(i,j) = exp(-T·JSD(succ[i], succ[j]))."""
    m = [[0.0] * K for _ in range(K)]
    for i in range(K):
        di = succ_dists.get(i, {})
        for j in range(K):
            m[i][j] = (1.0 if i == j
                       else math.exp(-temperature * _jsd(di, succ_dists.get(j, {}))))
    return m


def _build_ctx_assignment(sequences: list, base_assignment: dict,
                            n_clusters: int, min_examples: int = 3,
                            verbose: bool = False) -> tuple[dict, dict]:
    """E2: Build context-sensitive (c_prev, token) cluster assignment."""
    raw: dict = collections.defaultdict(collections.Counter)
    for seq in sequences:
        for i in range(len(seq) - 2):
            t1, t2, t3 = str(seq[i]), str(seq[i+1]), str(seq[i+2])
            c1 = base_assignment.get(t1)
            if c1 is None or t2 not in base_assignment:
                continue
            raw[(c1, t2)][t3] += 1

    if not raw:
        return {}, {}

    dists = {}
    counts = {}
    for key, counter in raw.items():
        total = sum(counter.values())
        if total < 1:
            continue
        atom = (key,)
        dists[atom]  = {(w,): cnt / total for w, cnt in counter.items()}
        counts[atom] = total

    raw_asgn = discover_categories_from_dists(
        dists, counts, n_clusters=n_clusters, min_examples=min_examples)
    if not raw_asgn:
        return {}, {}

    clusters_raw: dict = collections.defaultdict(list)
    ctx_asgn: dict = {}
    for atom_key, cid in raw_asgn.items():
        key = atom_key[0]
        ctx_asgn[key] = cid
        clusters_raw[cid].append(key)

    by_size = sorted(clusters_raw.items(), key=lambda kv: -len(kv[1]))
    renumber = {old: new for new, (old, _) in enumerate(by_size)}
    ctx_asgn = {k: renumber[v] for k, v in ctx_asgn.items()}
    ctx_clusters = {renumber[old]: m for old, m in clusters_raw.items()}
    if verbose:
        print(f'  E2: {len(ctx_clusters)} context clusters')
    return ctx_asgn, ctx_clusters


def _train_chain_ctx(ai: SymbolicAI, sequences: list,
                      base_assignment: dict,
                      ctx_assignment: dict,
                      verbose: bool = False) -> None:
    """E2: Train next_cat_ctx and token_given_cat_ctx."""
    n_used = n_skip = 0
    for seq in sequences:
        for i in range(len(seq) - 2):
            t1, t2, t3 = str(seq[i]), str(seq[i+1]), str(seq[i+2])
            c1 = base_assignment.get(t1)
            c2b = base_assignment.get(t2)
            if c1 is None or c2b is None:
                n_skip += 1; continue
            c2_ctx = ctx_assignment.get((c1, t2))
            c3b    = base_assignment.get(t3)
            if c2_ctx is None or c3b is None:
                n_skip += 1; continue
            c3_ctx = ctx_assignment.get((c2b, t3))
            if c3_ctx is None:
                n_skip += 1; continue
            ai.teach('next_cat_ctx',
                     (str(c1), str(c2_ctx)), (str(c3_ctx),))
            ai.teach('token_given_cat_ctx',
                     (str(c1), str(c2_ctx), str(c3_ctx)), (t3,))
            n_used += 1
    if verbose:
        print(f'  E2 chain: {n_used:,} used, {n_skip:,} skipped')


def _kmeans_contexts(contexts: list[tuple], n_clusters: int,
                      max_iter: int = 20) -> dict:
    """K-means on (c1, c3) integer tuples. Returns {(c1,c3): cluster_id}."""
    import random
    all_keys = list(set(contexts))
    if len(all_keys) <= n_clusters:
        return {k: i % n_clusters for i, k in enumerate(all_keys)}

    # Frequency counts
    freq: dict = collections.Counter(contexts)
    unique = list(freq.keys())

    # Initialize centroids randomly
    rng = random.Random(42)
    centroids = rng.sample(unique, min(n_clusters, len(unique)))

    assignment = {}
    for _ in range(max_iter):
        # Assign
        new_asgn = {}
        for k in unique:
            best = min(range(len(centroids)),
                       key=lambda j: _manhattan(k, centroids[j]))
            new_asgn[k] = best
        if new_asgn == assignment:
            break
        assignment = new_asgn
        # Update centroids
        cluster_sums: dict = collections.defaultdict(lambda: [0, 0])
        cluster_counts: dict = collections.defaultdict(int)
        for k, cid in assignment.items():
            cluster_sums[cid][0] += k[0] * freq[k]
            cluster_sums[cid][1] += k[1] * freq[k]
            cluster_counts[cid]  += freq[k]
        centroids = [
            (round(cluster_sums[j][0] / max(cluster_counts[j], 1)),
             round(cluster_sums[j][1] / max(cluster_counts[j], 1)))
            for j in range(len(centroids))
        ]

    return assignment


def _manhattan(a: tuple, b: tuple) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def _score_chain(ai: SymbolicAI, chain: list[str],
                  examples: list[tuple]) -> tuple[float, float]:
    """Score a concept chain on (input_tuple, expected_output) examples."""
    n_ans = n_correct = 0
    for inp, expected in examples:
        current = inp
        for name in chain:
            r = ai.ask(name, current)
            if r is None:
                current = None; break
            current = r if isinstance(r, tuple) else (r,)
        if current is not None:
            n_ans += 1
            if current == expected or (current and current[0] == expected[0]):
                n_correct += 1
    n = len(examples)
    cov = n_ans / n if n else 0.0
    acc = n_correct / n_ans if n_ans else 0.0
    return cov, acc


# ---------------------------------------------------------------------------
# CLI entry point (text corpus demo)
# ---------------------------------------------------------------------------

def main() -> None:
    from discover_structure import run_pipeline, _stream_texts, _banner, _DATA_DIR

    try:
        from ocr_test import find_pairs  # type: ignore[import]
    except ImportError:
        import glob, random as _random
        def find_pairs(d, max_n=None, shuffle=False, seed=0):
            pairs = []
            for gt in glob.glob(os.path.join(d, '**', '*.gt.txt'), recursive=True):
                pairs.append((None, gt))
            if shuffle:
                rng = _random.Random(seed)
                rng.shuffle(pairs)
            return pairs[:max_n] if max_n else pairs

    p = argparse.ArgumentParser(description='SequenceLearner E0-E6 demo')
    p.add_argument('--corpus',     default='EarlyModernLatin')
    p.add_argument('--n_train',    type=int,   default=5000)
    p.add_argument('--n_clusters', type=int,   default=12)
    p.add_argument('--phases',     default='e1e3')
    p.add_argument('--seed',       type=int,   default=42)
    args = p.parse_args()

    d = (args.corpus if os.path.isdir(args.corpus)
         else os.path.join(_DATA_DIR, args.corpus))
    if not os.path.isdir(d):
        print(f'ERROR: corpus not found: {args.corpus!r}')
        sys.exit(1)

    total = args.n_train + max(args.n_train // 4, 100)
    pairs  = find_pairs(d, max_n=total, shuffle=True, seed=args.seed)
    texts  = _stream_texts(pairs)
    n      = min(args.n_train, int(len(texts) * 0.8))
    train  = texts[:n]; test = texts[n:]

    _banner('SequenceLearner E0-E6 Demo')
    print(f'  Corpus: {args.corpus}  train={n}  K={args.n_clusters}')

    # Teach multi-scale bigrams first (optional — gives richer representations)
    ai, _ = run_pipeline(train, n_levels=3, max_merges=500, verbose=False)

    learner = SequenceLearner(ai=ai, n_clusters=args.n_clusters)
    learner.fit(sequences_from_texts(train),
                bigram_concept='next_word_hier',
                phases=args.phases, verbose=True)

    # Build train_pairs for unseen evaluation
    train_pairs = set()
    for t in train:
        toks = t.split()
        for i in range(len(toks) - 1):
            train_pairs.add((toks[i], toks[i+1]))

    learner.evaluate(sequences_from_texts(test),
                     train_pairs=train_pairs, verbose=True)

    # E4 demo
    learner.fit_paradigmatic(sequences_from_texts(train), verbose=True)
    for probe in ('est', 'ex', 'cum'):
        nb = learner.paradigmatic_neighbors(probe, topn=4)
        if nb:
            nb_str = '  '.join(f'{w}({s:.2f})' for w, s in nb)
            print(f'  {probe!r:8} ≈  {nb_str}')


if __name__ == '__main__':
    main()
