"""relational_pipeline.py — Multi-relational structure learning without linearization.

Instead of turning 2D data into 1D sequences (which destroys spatial structure),
this module learns from (atom_a, relation, atom_b) triples directly.

Broca's area principle extends naturally: one algorithm, all relational domains.

    text:    ('king', 'IS_A', 'person'), ('king', 'HAS', 'crown'), ...
    image:   ('patch_ab3f', 'H', 'patch_c71e'), ('patch_ab3f', 'V', 'patch_09d2'), ...
    KB:      ('Paris', 'capital_of', 'France'), ('Paris', 'located_in', 'Europe'), ...

Architecture (domain-agnostic):

    E0: Compound bigrams (atom,) → ('relation:target',) → bidir clustering → K categories.
        An atom's signature = ALL of its relational neighbors, across ALL relation types.
    E1: next_cat_rel(c_src, relation) → c_tgt
        token_given_cat_rel(c_src, relation, c_tgt) → atom
    E3: Soft retrieval  sim(c,c') = exp(-T·JSD(succ_dist[c], succ_dist[c']))
        Mixed keys: category IDs get soft similarity; relation names get exact match.
        (Both handled automatically by _ask_soft — integer keys → soft, string keys → exact.)

Usage:
    from relational_pipeline import RelationalLearner

    triples = [('king', 'IS_A', 'person'), ('queen', 'IS_A', 'person'), ...]
    learner = RelationalLearner(n_clusters=12)
    learner.fit(triples)
    print(learner.predict('queen', 'IS_A'))     # → 'person'
    print(learner.atom_neighbors('king', topn=5))

For 2D images:
    from relational_pipeline import Image2DRelationalLearner
    learner = Image2DRelationalLearner(patch_size=8, n_clusters=16)
    learner.fit_images(images)
    clusters = learner.assignment   # patch_hash → cluster_id
"""
from __future__ import annotations

import collections
import math
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if os.path.join(_HERE, '..') not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, '..'))

from engine import SymbolicAI
from synthesis import _gap_threshold
from sequence_pipeline import (
    _jsd, _rss_mb, _ask_soft, _build_sim_matrix,
    _precompute_dist_cache,
    _register_concepts, ai_stores, ai_graph_concepts,
)

_HERE_CTKG = os.path.join(os.path.dirname(_HERE), 'ctkg')
if _HERE_CTKG not in sys.path:
    sys.path.insert(0, _HERE_CTKG)
from ctkg.graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# RelationalLearner
# ---------------------------------------------------------------------------

class RelationalLearner:
    """E0-E3 multi-relational structure learning from (atom, relation, atom) triples.

    The compound bigram trick:
        Each triple (a, r, b) becomes two bigrams:
            rel_next(a,) → (f'{r}:{b}',)
            rel_prev(b,) → (f'rev_{r}:{a}',)

        So an atom's distributional signature encodes ALL its relational neighbors
        across ALL relation types simultaneously. induce_hierarchy_bidir() then
        clusters atoms by these compound signatures — atoms with similar relational
        roles end up in the same category, regardless of surface form.

    E3 soft retrieval key structure for next_cat_rel:
        key = (str(c_src), relation_name)
        - c_src is an integer string → uses sim_matrix (soft)
        - relation_name is a non-integer string → uses exact match
        This is handled automatically by _ask_soft — no special casing needed.
    """

    def __init__(self, ai: SymbolicAI | None = None, n_clusters: int | None = None):
        self.ai = ai or SymbolicAI(KnowledgeGraph())
        self.n_clusters = n_clusters
        self.assignment: dict = {}   # atom_str → cluster_id (int)
        self.clusters:   dict = {}   # cluster_id → [atom_str, ...]
        self._K: int = n_clusters
        # E3 lazy state — same pattern as SequenceLearner to avoid O(K²×V) blow-up
        self._nc_soft:   dict = {}   # lazy cache: (c_src, rel) → {c_tgt: prob}
        self._wgc_soft:  dict = {}   # lazy cache: (c_src, rel, c_tgt) → {atom: prob}
        self._nc_cache:  dict = {}   # raw dist cache from next_cat_rel
        self._wgc_cache: dict = {}   # raw dist cache from token_given_cat_rel
        self._sim_matrix: list = []  # K×K JSD-based similarity matrix
        self._atom_bigrams: dict = {}   # (atom, rel) → {target: prob}  (fast path)
        self._rel_unigram:  dict = {}   # rel → {target: prob}  (OOV fallback)

    # ---- E0: bigram discovery -----------------------------------------------

    def discover(self, triples: list[tuple[Any, str, Any]],
                 verbose: bool = False) -> None:
        """E0: Teach compound bigrams from (atom_a, relation, atom_b) triples.

        Forward:  rel_next(a,) → (f'{r}:{b}',)
        Backward: rel_prev(b,) → (f'rev_{r}:{a}',)

        Both directions contribute to each atom's distributional signature,
        so clustering captures relational context symmetrically.
        """
        for name, it, ot in [
            ('rel_next', ['atom'], ['rel_atom']),
            ('rel_prev', ['atom'], ['rel_atom']),
        ]:
            if name not in ai_graph_concepts(self.ai):
                self.ai.add_concept(name=name, domain='relational',
                                    input_type=it, output_type=ot)
        n = 0
        for a, r, b in triples:
            a_s, r_s, b_s = str(a), str(r), str(b)
            self.ai.teach('rel_next', (a_s,), (f'{r_s}:{b_s}',))
            self.ai.teach('rel_prev', (b_s,), (f'rev_{r_s}:{a_s}',))
            n += 1
        if verbose:
            print(f'  E0: taught {n:,} triples as compound bigrams')

    # ---- Large-KG mode (no cluster pipeline) --------------------------------

    def _fit_large_kg(self, triples: list[tuple[Any, str, Any]],
                      verbose: bool = True) -> None:
        """Build atom bigrams + per-relation unigram in a single O(N) pass.

        Used when n_unique_atoms > max_atoms (e.g. FB15k-237 with 14K entities).
        The E0-E3 cluster pipeline would allocate a dense matrix of size
        n_atoms × n_compound_tokens ≈ 14K × 544K × 4 bytes ≈ 30 GB — unacceptable.

        For large KGs, atom-level bigrams ARE the right model:
          - Fast path:  P(t | h, r) from _atom_bigrams  (exact empirical dist)
          - OOV fallback: P(t | r) from _rel_unigram    (marginal over all h)
        No clustering is needed or useful: each entity is semantically distinct.
        """
        from collections import defaultdict, Counter as _Counter
        _atom_raw: dict = defaultdict(_Counter)  # (h, r) → Counter(t)
        _rel_raw:  dict = defaultdict(_Counter)  # r → Counter(t)

        for a, r, b in triples:
            a_s, r_s, b_s = str(a), str(r), str(b)
            _atom_raw[(a_s, r_s)][b_s] += 1
            _rel_raw[r_s][b_s] += 1

        # Save raw counts for update_online() before normalising.
        self._atom_counts: dict = dict(_atom_raw)   # (atom, rel) → Counter(target)
        self._rel_counts:  dict = dict(_rel_raw)    # rel → Counter(target)

        self._atom_bigrams = {}
        for (atom, rel), ctr in _atom_raw.items():
            total = sum(ctr.values())
            if total > 0:
                self._atom_bigrams[(atom, rel)] = {
                    k: v / total for k, v in ctr.items()}

        self._rel_unigram = {}
        for rel, ctr in _rel_raw.items():
            total = sum(ctr.values())
            if total > 0:
                self._rel_unigram[rel] = {k: v / total for k, v in ctr.items()}

        # assignment stays empty — no clustering
        self.assignment = {}
        self.clusters   = {}
        self._K         = 0

        if verbose:
            print(f'  atom_bigrams: {len(self._atom_bigrams):,} (h,r) pairs')
            print(f'  rel_unigram:  {len(self._rel_unigram):,} relations')
            m = _rss_mb()
            if m:
                print(f'  RSS after large-KG fit = {m:.0f} MB')

    # ---- E0-E3: fit ---------------------------------------------------------

    def fit(self, triples: list[tuple[Any, str, Any]],
            e3_temperature: float = 2.0,
            verbose: bool = True,
            max_atoms: int = 500) -> None:
        """Fit E0-E3 on a list of (atom_a, relation_name, atom_b) triples.

        Args:
            max_atoms: If the number of unique atoms exceeds this threshold,
                       skip the E0-E3 cluster pipeline (which builds a dense
                       n_atoms × n_compound_tokens matrix that OOMs for large KGs)
                       and use atom-level bigrams + per-relation unigram only.
                       Default 500 is safe for char/word level data; for FB15k-237
                       (14K entities) the cluster matrix would be ~31 GB.
        """
        # Count unique atoms before allocating anything
        atoms_seen: set = set()
        for a, r, b in triples:
            atoms_seen.add(str(a))
            atoms_seen.add(str(b))
        n_unique = len(atoms_seen)

        if n_unique > max_atoms:
            if verbose:
                print(f'  Large-KG mode: {n_unique:,} atoms > {max_atoms} — '
                      f'skipping cluster pipeline (would allocate '
                      f'~{n_unique * len(triples) * 4 // 1_000_000_000:.0f}+ GB), '
                      f'using atom bigrams + rel unigram only.')
            self._fit_large_kg(triples, verbose=verbose)
            return

        if verbose:
            m = _rss_mb()
            if m:
                print(f'  RSS at fit() start = {m:.0f} MB')

        # E0: discover compound bigrams
        if 'rel_next' not in ai_stores(self.ai):
            self.discover(triples, verbose=verbose)
        if verbose:
            m = _rss_mb()
            if m:
                print(f'  RSS after E0 discover = {m:.0f} MB')

        # Cluster atoms by compound bigram distributions
        if verbose:
            print(f'  Clustering {self._K} atom categories '
                  f'from rel_next/rel_prev...')
        result = self.ai.induce_hierarchy_bidir(
            'rel_next', 'rel_prev', n_clusters=self._K)
        if 'error' in result:
            print(f'  WARNING: clustering failed: {result["error"]}')
            return
        self.assignment = result.get('assignment', {})
        self.clusters   = result.get('clusters', {})
        self._K         = result.get('n_clusters', len(self.clusters))
        if verbose:
            print(f'  Atoms: {len(self.assignment):,}  '
                  f'Clusters: {self._K}')

        # Build E3 successor distributions BEFORE clearing E0 examples
        # (ask_dist uses store.examples, which is gone after .clear())
        if verbose:
            print('  E3: building successor distributions...')
        succ_dists = _build_rel_succ_dists(self.ai, self.clusters)
        self._succ_dists = succ_dists   # stored for R4 adapt_metric()
        sim_matrix = _build_sim_matrix(succ_dists, self._K, e3_temperature)

        # Free E0 bigram examples — clustering is complete
        for _c in ('rel_next', 'rel_prev'):
            _s = self.ai.stores.get(_c)
            if _s is not None:
                _s.examples.clear()
        if verbose:
            m = _rss_mb()
            if m:
                print(f'  RSS after E0 clear = {m:.0f} MB')

        # E1: category chain conditioned on relation name
        _register_concepts(self.ai, [
            ('next_cat_rel',        ['cat', 'rel'],       ['cat'],  'relational'),
            ('token_given_cat_rel', ['cat', 'rel', 'cat'],['atom'], 'relational'),
        ])
        # Build transition matrix (GeometryDetector) AND atom-level bigrams
        # (V-level prediction pathway) in a single O(N) pass.
        from collections import defaultdict, Counter as _Counter
        _trans_raw:   dict = defaultdict(lambda: defaultdict(_Counter))
        _atom_raw:    dict = defaultdict(_Counter)  # (atom, rel) → Counter(target)
        n_used = n_skip = 0
        for a, r, b in triples:
            a_s, r_s, b_s = str(a), str(r), str(b)
            c_a = self.assignment.get(a_s)
            c_b = self.assignment.get(b_s)
            # Atom-level bigram (always — even for OOV categories)
            _atom_raw[(a_s, r_s)][b_s] += 1
            if c_a is None or c_b is None:
                n_skip += 1
                continue
            self.ai.teach('next_cat_rel',
                          (str(c_a), r_s), (str(c_b),))
            self.ai.teach('token_given_cat_rel',
                          (str(c_a), r_s, str(c_b)), (b_s,))
            _trans_raw[r_s][str(c_a)][str(c_b)] += 1
            n_used += 1
        # Normalise atom bigrams and store.
        self._atom_counts: dict = dict(_atom_raw)  # raw counts for update_online()
        self._atom_bigrams: dict = {}
        for (atom, rel), ctr in _atom_raw.items():
            total = sum(ctr.values())
            if total > 0:
                self._atom_bigrams[(atom, rel)] = {
                    k: v / total for k, v in ctr.items()}
        # Build rel unigram counts (mirrors _fit_large_kg for update_online() compat).
        from collections import Counter as _Counter2
        _rel_raw2: dict = collections.defaultdict(_Counter2)
        for a, r, b in triples:
            _rel_raw2[str(r)][str(b)] += 1
        self._rel_counts: dict = dict(_rel_raw2)
        self._rel_unigram: dict = {
            r: {t: c / sum(ctr.values()) for t, c in ctr.items()}
            for r, ctr in _rel_raw2.items() if sum(ctr.values()) > 0
        }
        # Normalise and store for GeometryDetector.
        self._trans: dict = {}
        for rel, src_map in _trans_raw.items():
            self._trans[rel] = {}
            for c_src, ctr in src_map.items():
                total = sum(ctr.values())
                if total > 0:
                    self._trans[rel][c_src] = {c: n / total
                                               for c, n in ctr.items()}
        if verbose:
            print(f'  E1: {n_used:,} triples used, {n_skip:,} OOV skipped')
            m = _rss_mb()
            if m:
                print(f'  RSS after E1 = {m:.0f} MB')

        # Build E3 dist caches from E1 examples before clearing
        nc_cache  = _precompute_dist_cache(self.ai, 'next_cat_rel')
        wgc_cache = _precompute_dist_cache(self.ai, 'token_given_cat_rel')

        # Store for lazy retrieval
        self._nc_cache   = nc_cache
        self._wgc_cache  = wgc_cache
        self._sim_matrix = sim_matrix
        self._nc_soft    = {}
        self._wgc_soft   = {}

        if verbose:
            print(f'  E3: nc_cache {len(nc_cache)} keys  '
                  f'wgc_cache {len(wgc_cache)} keys  (lazy — computed on first query)')
            m = _rss_mb()
            if m:
                print(f'  RSS after E3 cache = {m:.0f} MB')

        # Free E1 examples — all information now in nc_cache / wgc_cache
        for _c in ('next_cat_rel', 'token_given_cat_rel'):
            _s = self.ai.stores.get(_c)
            if _s is not None:
                _s.examples.clear()

    # ---- Prediction ---------------------------------------------------------

    def predict_chain(self, atom: Any, relations: list[str]) -> Any | None:
        """Multi-hop prediction: atom -r1→ ? -r2→ ? ... -rn→ target.

        Traces through each relation in order, piping the output category into
        the next hop's input.  The final atom is decoded via
        token_given_cat_rel(c_{n-1}, r_n, c_n).

        Examples:
            learner.predict_chain('Paris', ['capital_of', 'located_in'])
            # → 'Europe'  (Paris→France→Europe)

            learner.predict_chain(patch, ['H', 'V'])
            # → patch at (r, c+1) then (r+1, c+1)
        """
        if not relations:
            return None
        a_s = str(atom)
        c_curr = self.assignment.get(a_s)
        if c_curr is None:
            return None

        # Chain through relations, tracking (c_prev, c_curr) for final decode
        c_prev = c_curr
        for rel in relations:
            c_b_dist = self._get_nc_soft((str(c_curr), rel))
            if not c_b_dist:
                return None
            c_b_key = max(c_b_dist, key=c_b_dist.get)
            c_b_str = c_b_key[0] if isinstance(c_b_key, tuple) else str(c_b_key)
            c_prev = c_curr
            try:
                c_curr = int(c_b_str)
            except (ValueError, TypeError):
                c_curr = c_b_str  # type: ignore[assignment]

        # Decode final atom using last relation
        r_last = relations[-1]
        atom_dist = self._get_wgc_soft((str(c_prev), r_last, str(c_curr)))
        if not atom_dist:
            return None
        best = max(atom_dist, key=atom_dist.get)
        return best[0] if isinstance(best, tuple) else str(best)

    def infer_chain(self, atom: Any, relations: list[str],
                    topk: int = 5) -> list[tuple[Any, float]]:
        """R6: Distribution-preserving multi-hop relational inference.

        Propagates an atom-level probability distribution through each relation,
        using _atom_bigrams (empirical char-specific distributions) as the fast
        path and category-level E3 soft retrieval as the OOV fallback.

        Returns:
            [(atom, probability), ...] sorted descending, top-k entries.

        Example (Latin):
            learner.infer_chain('q', ['next', 'next'])
            # 'q' → 'u' (0.99) → then what follows 'u' most often

            learner.infer_chain('q', ['skip2f'])
            # Should give same result (R3 showed next∘next ≈ skip2f)
        """
        atom_dist: dict = {str(atom): 1.0}
        atom_bigs = getattr(self, '_atom_bigrams', {})

        for rel in relations:
            r_s = str(rel)
            next_dist: dict = {}
            for src_atom, p_src in atom_dist.items():
                if p_src < 1e-10:
                    continue
                # Fast path: empirical atom-level bigram
                tgt_dist = atom_bigs.get((src_atom, r_s))
                if not tgt_dist:
                    # OOV fallback: category-level E3 soft retrieval
                    tgt_dist = self.predict_dist(src_atom, r_s)
                for tgt, p_tgt in tgt_dist.items():
                    next_dist[tgt] = next_dist.get(tgt, 0.0) + p_src * p_tgt
            if not next_dist:
                return []
            total = sum(next_dist.values())
            atom_dist = {k: v / total for k, v in next_dist.items()}

        ranked = sorted(atom_dist.items(), key=lambda kv: -kv[1])
        return ranked[:topk]

    def predict(self, atom: Any, relation: str) -> Any | None:
        """Predict most likely target atom given source atom and relation name.

        Uses atom-level bigrams (fast path) for seen atoms, falls back to
        category-level E3 soft retrieval for OOV generalisation.
        """
        a_s = str(atom)
        r_s = str(relation)
        # Fast path: atom-level bigram
        bg = getattr(self, '_atom_bigrams', {}).get((a_s, r_s))
        if bg:
            return max(bg, key=bg.get)
        # Fallback: category-level E3 soft retrieval (OOV)
        c_a = self.assignment.get(a_s)
        if c_a is None:
            return None
        c_b_dist = self._get_nc_soft((str(c_a), r_s))
        if c_b_dist:
            c_b_tup = max(c_b_dist, key=c_b_dist.get)
            c_b = c_b_tup[0] if isinstance(c_b_tup, tuple) else str(c_b_tup)
            atom_dist = self._get_wgc_soft((str(c_a), r_s, c_b))
            if atom_dist:
                best = max(atom_dist, key=atom_dist.get)
                return best[0] if isinstance(best, tuple) else str(best)
        # E1 fallback
        c_b_t = self.ai.ask('next_cat_rel', (str(c_a), r_s))
        if c_b_t is None:
            return None
        c_b = c_b_t[0] if isinstance(c_b_t, tuple) else str(c_b_t)
        atom_t = self.ai.ask('token_given_cat_rel', (str(c_a), r_s, c_b))
        if atom_t is None:
            return None
        return atom_t[0] if isinstance(atom_t, tuple) else str(atom_t)

    def predict_dist(self, atom: Any, relation: str) -> dict[Any, float]:
        """Full distribution over target atoms: P(target | atom, relation).

        Fast path (seen atoms): returns the atom-level bigram distribution directly
        — V-level precision, no category abstraction overhead.

        Fallback (OOV atoms): marginalises over c_tgt via soft category retrieval,
        providing generalisation to atoms not seen in training.
        """
        a_s = str(atom)
        r_s = str(relation)
        # Fast path: atom-level bigram (precise, char-specific)
        bg = getattr(self, '_atom_bigrams', {}).get((a_s, r_s))
        if bg is not None:
            return dict(bg)
        # Fallback: category-level soft retrieval (OOV in small-V models)
        c_a = self.assignment.get(a_s)
        if c_a is not None:
            c_b_dist = self._get_nc_soft((str(c_a), r_s))
            if c_b_dist:
                result: dict = {}
                for c_b_key, p_cb in c_b_dist.items():
                    c_b = c_b_key[0] if isinstance(c_b_key, tuple) else str(c_b_key)
                    t_dist = self._get_wgc_soft((str(c_a), r_s, c_b))
                    if not t_dist:
                        continue
                    for tok_key, p_t in t_dist.items():
                        tok = tok_key[0] if isinstance(tok_key, tuple) else str(tok_key)
                        result[tok] = result.get(tok, 0.0) + p_cb * p_t
                if result:
                    return result
        # Final fallback: per-relation unigram P(t|r) — works in large-KG mode
        return dict(self._rel_unigram.get(r_s, {}))

    def update_online(self, atom: Any, rel: Any, target: Any) -> None:
        """Incorporate one new (atom, rel, target) observation into the model.

        Updates ``_atom_bigrams`` and ``_rel_unigram`` in O(1) without
        rerunning the E0-E3 cluster pipeline.  Intended for in-episode
        online learning: the agent observes a transition and immediately
        strengthens the corresponding prediction.

        Requires ``_atom_counts`` / ``_rel_counts`` to be present (populated
        by both ``fit()`` and ``_fit_large_kg()`` automatically).  If the
        learner has never been fitted, the dicts are created on first call.

        Args:
            atom    Source atom (will be str-coerced).
            rel     Relation name (will be str-coerced).
            target  Target atom observed (will be str-coerced).
        """
        from collections import Counter as _Ctr
        a_s, r_s, b_s = str(atom), str(rel), str(target)
        key = (a_s, r_s)

        # Ensure count dicts exist (tolerate un-fitted learner).
        if not hasattr(self, '_atom_counts') or self._atom_counts is None:
            self._atom_counts = {}
        if not hasattr(self, '_rel_counts') or self._rel_counts is None:
            self._rel_counts = {}

        # --- atom-level bigram update ---------------------------------------
        if key not in self._atom_counts:
            # Seed from existing normalised distribution (×10 pseudo-count)
            # so new observations don't dominate immediately.
            existing = getattr(self, '_atom_bigrams', {}).get(key, {})
            _pc = 10
            self._atom_counts[key] = _Ctr(
                {t: max(1, round(p * _pc)) for t, p in existing.items()}
            ) if existing else _Ctr()
        self._atom_counts[key][b_s] += 1
        total = sum(self._atom_counts[key].values())
        self._atom_bigrams[key] = {t: c / total
                                   for t, c in self._atom_counts[key].items()}

        # --- per-relation unigram update ------------------------------------
        if r_s not in self._rel_counts:
            existing_r = getattr(self, '_rel_unigram', {}).get(r_s, {})
            _pc_r = 50
            self._rel_counts[r_s] = _Ctr(
                {t: max(1, round(p * _pc_r)) for t, p in existing_r.items()}
            ) if existing_r else _Ctr()
        self._rel_counts[r_s][b_s] += 1
        total_r = sum(self._rel_counts[r_s].values())
        if not hasattr(self, '_rel_unigram') or self._rel_unigram is None:
            self._rel_unigram = {}
        self._rel_unigram[r_s] = {t: c / total_r
                                  for t, c in self._rel_counts[r_s].items()}

    def atom_neighbors(self, atom: Any, topn: int = 8) -> list[tuple]:
        """Atoms most similar in relational role (by category similarity).

        Uses the E3 sim_matrix (JSD on compound bigram successor distributions).
        Returns [(atom, similarity), ...] sorted descending.
        """
        a_s = str(atom)
        c_a = self.assignment.get(a_s)
        if c_a is None or not self._sim_matrix:
            return []
        results = []
        for other_tok, c_other in self.assignment.items():
            if other_tok == a_s:
                continue
            if 0 <= c_a < self._K and 0 <= c_other < self._K:
                sim = self._sim_matrix[c_a][c_other]
            else:
                sim = 1.0 if c_a == c_other else 0.0
            if sim > 1e-6:
                results.append((other_tok, sim))
        return sorted(results, key=lambda x: -x[1])[:topn]

    def cluster_summary(self, topn_per_cluster: int = 5) -> dict[int, list[str]]:
        """Return top-n most frequent atoms per cluster."""
        # Count atom frequencies from assignment
        freq: dict[str, int] = collections.Counter()
        # We don't track raw frequencies; just return the members list
        summary = {}
        for cid, members in sorted(self.clusters.items()):
            summary[cid] = members[:topn_per_cluster]
        return summary

    # ---- Private helpers ----------------------------------------------------

    def _get_nc_soft(self, key: tuple) -> dict | None:
        if key not in self._nc_soft:
            self._nc_soft[key] = _ask_soft(
                key, self._nc_cache, self._sim_matrix, self._K)
        return self._nc_soft[key]

    def _get_wgc_soft(self, key: tuple) -> dict | None:
        if key not in self._wgc_soft:
            self._wgc_soft[key] = _ask_soft(
                key, self._wgc_cache, self._sim_matrix, self._K)
        return self._wgc_soft[key]

    # ---- R4: geometry-adapted metric ----------------------------------------

    def adapt_metric(self, topology: str,
                     temperature: float = 2.0) -> None:
        """R4: Rebuild E3 sim matrix using the geometry-appropriate metric.

        Replaces the default JSD-based sim matrix with one tuned to the
        detected data topology (from GeometryDetector).

        Topology → metric:
            directed_linear / undirected_linear → 1D MDS position distance
            directed_2d / undirected_2d         → 2D classical MDS L2 distance
            hyperbolic                           → BFS category-graph hops
            dense / general_graph / *            → JSD (default, no change)

        Clears the lazy E3 caches so next predictions use the new metric.
        """
        succ_dists = getattr(self, '_succ_dists', {})
        if not succ_dists or self._K == 0:
            return

        self._sim_matrix = _build_sim_matrix_adapted(
            succ_dists, self._K, topology, temperature,
            trans=getattr(self, '_trans', {}))
        # Clear lazy caches
        self._nc_soft  = {}
        self._wgc_soft = {}


# ---------------------------------------------------------------------------
# ContextBeliefState  — Bayesian running context for RelationalLearner
# ---------------------------------------------------------------------------

class ContextBeliefState:
    """Bayes filter over RelationalLearner's K atom categories.

    Maintains a probability distribution P(c_t) over category IDs, updated by:
      - ``observe(atom)``: sharp concentration on atom's known category.
      - ``transition(relation)``: propagate belief through the category
        transition matrix T[relation][c_src → c_tgt].
      - ``decay()``: entropy increase between observations.

    This provides context-conditioned prediction: ``predict_target_dist()``
    returns P(target | current category belief, relation), which marginalises
    over the belief state rather than conditioning on a single point estimate.

    This is strictly more expressive than querying ``learner.predict_dist()``
    directly: a point-query ignores distributional uncertainty, while the Bayes
    filter propagates and updates that uncertainty over time.

    Design principle: general (domain-agnostic).  Plugs into AIFEngine via::

        engine._context_belief = ContextBeliefState(learner)
        # In feedback():
        engine._context_belief.observe(state['location'])
        engine._context_belief.transition(action)

    Parameters
    ----------
    learner
        A fitted ``RelationalLearner``.  Must have ``.assignment``, ``._K``,
        ``._trans`` (category transition matrices per relation), ``._nc_cache``,
        ``._wgc_cache``, ``._sim_matrix``.
    decay_rate
        Per-step decay toward uniform prior.  0.9 = slow decay (persists ~9
        steps).  0.5 = fast decay (half-life 1 step).
    obs_sharpness
        Concentration on observed category.  0.97 = 97% mass on category.
    """

    def __init__(
        self,
        learner:        'RelationalLearner',
        decay_rate:     float = 0.90,
        obs_sharpness:  float = 0.97,
    ) -> None:
        self._learner       = learner
        self._decay_rate    = decay_rate
        self._obs_sharpness = obs_sharpness
        self._belief: dict[int, float] = {}
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle

    def reset(self) -> None:
        """Reset to uniform prior over all K categories."""
        K = getattr(self._learner, '_K', 0) or 1
        self._belief = {c: 1.0 / K for c in range(K)}

    # ------------------------------------------------------------------
    # Update

    def observe(self, atom: Any, certainty: float | None = None) -> None:
        """Sharp Bayesian update: concentrate belief on atom's category.

        If atom is OOV (not in learner.assignment), the belief is unchanged.
        """
        a_s = str(atom)
        c = self._learner.assignment.get(a_s)
        if c is None or c not in self._belief:
            return  # OOV — cannot update
        sharp = certainty if certainty is not None else self._obs_sharpness
        # Concentrate mass on observed category; redistribute residual.
        other_total = max(
            sum(p for k, p in self._belief.items() if k != c), 1e-12)
        for k in self._belief:
            if k == c:
                self._belief[k] = sharp
            else:
                self._belief[k] = (1.0 - sharp) * (self._belief[k] / other_total)
        self._normalize()

    def transition(self, relation: str) -> None:
        """Propagate belief through the category transition matrix for relation.

        P(c_next) = Σ_{c_prev} P(c_prev) · T[relation][c_prev → c_next]

        If no transition data exists for relation, belief is unchanged (the
        relation is unknown; uncertainty cannot decrease).
        """
        trans = getattr(self._learner, '_trans', {}).get(str(relation), {})
        if not trans:
            return
        next_belief: dict[int, float] = {}
        for c_prev, p_prev in self._belief.items():
            if p_prev < 1e-12:
                continue
            c_prev_s = str(c_prev)
            c_next_dist = trans.get(c_prev_s, {})
            for c_next_s, p_t in c_next_dist.items():
                try:
                    c_next = int(c_next_s)
                except (ValueError, TypeError):
                    continue
                next_belief[c_next] = next_belief.get(c_next, 0.0) + p_prev * p_t
        if next_belief:
            total = sum(next_belief.values())
            if total > 0:
                self._belief = {k: v / total for k, v in next_belief.items()}

    def decay(self, rate: float | None = None) -> None:
        """Decay toward uniform prior (one step of unobserved time passes)."""
        d = rate if rate is not None else self._decay_rate
        K = max(len(self._belief), 1)
        uniform = 1.0 / K
        for k in self._belief:
            self._belief[k] = d * self._belief[k] + (1.0 - d) * uniform
        self._normalize()

    # ------------------------------------------------------------------
    # Query

    def predict_target_dist(self, relation: str) -> dict[Any, float]:
        """Context-conditioned P(target | belief_state, relation).

        Marginalises over the current category belief:
          P(t | rel) = Σ_c P(c) · Σ_{c'} P(c' | c, rel) · P(t | c, rel, c')

        Returns an empty dict if no transition/category data is available.
        """
        learner = self._learner
        r_s     = str(relation)
        result: dict = {}

        for c_src, p_src in self._belief.items():
            if p_src < 1e-12:
                continue
            c_src_s  = str(c_src)
            c_b_dist = learner._get_nc_soft((c_src_s, r_s))
            if not c_b_dist:
                continue
            for c_b_key, p_cb in c_b_dist.items():
                c_b = c_b_key[0] if isinstance(c_b_key, tuple) else str(c_b_key)
                t_dist = learner._get_wgc_soft((c_src_s, r_s, c_b))
                if not t_dist:
                    continue
                for tok_key, p_t in t_dist.items():
                    tok = tok_key[0] if isinstance(tok_key, tuple) else str(tok_key)
                    result[tok] = result.get(tok, 0.0) + p_src * p_cb * p_t

        total = sum(result.values())
        if total > 0:
            return {k: v / total for k, v in result.items()}
        return {}

    def most_likely_category(self) -> tuple[int, float]:
        """(category_id, probability) of the most probable category."""
        if not self._belief:
            return -1, 0.0
        best = max(self._belief, key=self._belief.__getitem__)
        return best, self._belief[best]

    def entropy(self) -> float:
        """Shannon entropy of the category belief in bits."""
        import math
        return -sum(
            p * math.log2(p + 1e-12)
            for p in self._belief.values() if p > 0
        )

    def __repr__(self) -> str:
        best, prob = self.most_likely_category()
        return (f'ContextBeliefState(K={len(self._belief)}, '
                f'best={best}, P={prob:.2f}, H={self.entropy():.2f} bits)')

    # ------------------------------------------------------------------
    # Private

    def _normalize(self) -> None:
        total = sum(self._belief.values()) or 1e-12
        for k in self._belief:
            self._belief[k] /= total


# ---------------------------------------------------------------------------
# Image2DRelationalLearner
# ---------------------------------------------------------------------------

class Image2DRelationalLearner:
    """RelationalLearner for 2D image patch grids.

    Generates H/V/D1/D2 neighborhood triples natively — no linearization.
    The 2D structure is preserved in the relation names, not destroyed by
    row-major ordering.

    Spatial relations:
        H  — horizontal: (r,c) → (r, c+1)
        V  — vertical:   (r,c) → (r+1, c)
        D1 — diagonal SE: (r,c) → (r+1, c+1)
        D2 — diagonal SW: (r,c) → (r+1, c-1)

    Patch vocabulary:
        Raw MD5 hashes (quantize_bits=0) make almost every natural-image patch
        unique → singleton bigram distributions → clustering has no signal.
        Perceptual quantization (quantize_bits=2 or 3) collapses visually similar
        patches to the same token, forcing the vocabulary repetition that
        distributional clustering needs.  Default is quantize_bits=2 (4 grey
        levels, very coarse) which gives enough recurrence even on small datasets.
        Use quantize_bits=3 (8 levels) for larger datasets.
    """

    ALL_RELATIONS = ('H', 'V', 'D1', 'D2')
    _OFFSETS = {'H': (0, 1), 'V': (1, 0), 'D1': (1, 1), 'D2': (1, -1)}

    def __init__(self, patch_size: int = 8, n_clusters: int = 12,
                 relations: tuple[str, ...] | None = None,
                 quantize_bits: int = 2,
                 codebook_size: int = 64):
        self.patch_size = patch_size
        self.relations = relations or self.ALL_RELATIONS
        self.quantize_bits = quantize_bits
        self.codebook_size = codebook_size
        self.learner = RelationalLearner(n_clusters=n_clusters)
        self._codebook: 'np.ndarray | None' = None  # (codebook_size, D)

    def image_to_patches(self, image) -> list[list[str]]:
        """Convert image (H×W×C or H×W numpy array) to 2D grid of patch token strings.

        Token assignment priority (in order):
          1. Codebook (k-means BoVW): if self._codebook is not None, assigns each
             patch to its nearest centroid → token = f'c{centroid_id}'.
             Enforces exactly codebook_size distinct tokens, guaranteeing recurrence.
          2. Perceptual quantization: if quantize_bits > 0, applies grayscale +
             local contrast normalization + bit quantization. Still mostly unique
             for natural images.
          3. Raw MD5 hash (quantize_bits == 0): always unique per patch.

        Codebook must be built first via fit_images(). Calling image_to_patches()
        before fit_images() falls back to quantization / MD5.
        """
        import numpy as np
        arr = np.asarray(image, dtype=np.uint8)
        ps = self.patch_size
        h, w = arr.shape[:2]
        rows, cols = h // ps, w // ps
        grid = []
        for r in range(rows):
            row = []
            for c in range(cols):
                patch_raw = arr[r*ps:(r+1)*ps, c*ps:(c+1)*ps]
                if self._codebook is not None:
                    vec = _preprocess_patch(patch_raw)
                    dists = np.sum((self._codebook - vec) ** 2, axis=1)
                    cid = int(np.argmin(dists))
                    token = f'c{cid}'
                elif self.quantize_bits > 0:
                    token = _quantize_patch(patch_raw, self.quantize_bits)
                else:
                    token = _patch_hash(patch_raw)
                row.append(token)
            grid.append(row)
        return grid

    def patches_to_triples(
            self,
            grid: list[list[str]],
    ) -> list[tuple[str, str, str]]:
        """Generate (patch_a, relation, patch_b) triples from a 2D patch grid."""
        triples = []
        rows = len(grid)
        if not rows:
            return triples
        cols = len(grid[0])
        for r in range(rows):
            for c in range(cols):
                a = grid[r][c]
                for rel in self.relations:
                    dr, dc = self._OFFSETS[rel]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        triples.append((a, rel, grid[nr][nc]))
        return triples

    def fit_images(self, images: list, verbose: bool = True) -> None:
        """Build patch codebook, then extract triples and run E0-E3.

        Step 1: Extract all raw patches from all images.
        Step 2: K-means codebook (codebook_size centroids) on grayscale patch vectors.
                This collapses visually similar patches to the same token, guaranteeing
                that the vocabulary (exactly codebook_size tokens) recurs across images.
                Without this, natural-image patches are nearly all unique → singleton
                bigram distributions → distributional clustering has no signal.
        Step 3: Re-tokenize all patches via nearest centroid.
        Step 4: Build H/V/D1/D2 triples and fit RelationalLearner E0-E3.
        """
        import numpy as np
        n_images = len(images)
        report_every = max(1, n_images // 5)

        # --- Step 1: collect all raw patches ---
        all_raw_patches: list = []
        all_grids_raw: list = []   # (rows, cols) arrays of raw patch arrays
        for i, img in enumerate(images):
            if verbose and (i == 0 or (i + 1) % report_every == 0):
                print(f'  Extracting patches: {i+1}/{n_images}')
            arr = np.asarray(img, dtype=np.uint8)
            ps = self.patch_size
            h, w = arr.shape[:2]
            rows, cols = h // ps, w // ps
            grid_raw = []
            for r in range(rows):
                row_raw = []
                for c in range(cols):
                    patch = arr[r*ps:(r+1)*ps, c*ps:(c+1)*ps]
                    row_raw.append(patch)
                    all_raw_patches.append(patch)
                grid_raw.append(row_raw)
            all_grids_raw.append(grid_raw)

        if verbose:
            print(f'  Raw patches: {len(all_raw_patches):,}  '
                  f'Building codebook (k={self.codebook_size})...')

        # --- Step 2: K-means codebook ---
        self._codebook = _build_patch_codebook(all_raw_patches, self.codebook_size,
                                               verbose=verbose)

        # --- Step 3+4: tokenize via codebook, build triples, fit ---
        all_triples: list[tuple[str, str, str]] = []
        for grid_raw in all_grids_raw:
            rows = len(grid_raw)
            cols = len(grid_raw[0]) if rows else 0
            # Build token grid via nearest centroid
            token_grid = []
            for r in range(rows):
                row_tok = []
                for c in range(cols):
                    vec = _preprocess_patch(grid_raw[r][c])
                    dists = np.sum((self._codebook - vec) ** 2, axis=1)
                    cid = int(np.argmin(dists))
                    row_tok.append(f'c{cid}')
                token_grid.append(row_tok)
            all_triples.extend(self.patches_to_triples(token_grid))

        if verbose:
            unique = len({a for a, _, _ in all_triples} |
                         {b for _, _, b in all_triples})
            print(f'  Total triples: {len(all_triples):,}  '
                  f'Unique visual words: {unique:,} / {self.codebook_size}')
        self.learner.fit(all_triples, verbose=verbose)

    @property
    def assignment(self) -> dict:
        return self.learner.assignment

    @property
    def clusters(self) -> dict:
        return self.learner.clusters

    def predict(self, patch: str, relation: str) -> str | None:
        return self.learner.predict(patch, relation)

    def predict_dist(self, patch: str, relation: str) -> dict:
        return self.learner.predict_dist(patch, relation)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_rel_succ_dists(ai: SymbolicAI, clusters: dict) -> dict:
    """Successor distributions per cluster for E3 similarity.

    Uses rel_next distributions for each atom in the cluster.
    MUST be called before examples.clear() — we scan examples here.

    One O(N) pass via build_full_freq_table() instead of V×O(N) scans
    through freq_dist() (the naive approach).
    """
    store = ai.stores.get('rel_next')
    if store is None or len(store) == 0:
        return {cid: {} for cid in clusters}

    # Single O(N) pass: build full conditional distribution for all atoms.
    full_table = store.build_full_freq_table()   # {(atom,): {(compound,): prob}}

    succ: dict = {}
    for cid, members in clusters.items():
        merged: dict = collections.defaultdict(float)
        n = 0
        for tok in members:
            d = full_table.get((str(tok),))
            if d is None:
                continue
            for out_tup, prob in d.items():
                k = out_tup[0] if isinstance(out_tup, tuple) else str(out_tup)
                merged[k] += prob
            n += 1
        succ[cid] = ({k: v / n for k, v in merged.items()} if n else {})
    return succ


def _build_sim_matrix_adapted(succ_dists: dict, K: int,
                               topology: str, temperature: float = 2.0,
                               trans: dict | None = None) -> list:
    """R4: Build K×K similarity matrix tuned to the detected geometry.

    Metric selection by topology:
        directed_linear / undirected_linear → 1D MDS embedding, |pos_i - pos_j|
        directed_2d / undirected_2d         → 2D classical MDS, L2 distance
        hyperbolic                           → BFS hop-count on category graph
        dense / general_graph / *            → JSD (default)

    Returns:
        K×K float matrix where m[i][j] = exp(-T * d(i, j)).
    """
    import numpy as np

    # Build K×K JSD pairwise distance matrix (always computed as base)
    D = np.zeros((K, K))
    for i in range(K):
        di = succ_dists.get(i, {})
        for j in range(i + 1, K):
            dj = succ_dists.get(j, {})
            d = _jsd(di, dj)
            D[i, j] = D[j, i] = d

    topo = topology.lower()

    # ---- 1D MDS (linear topology) -----------------------------------------
    if topo in ('directed_linear', 'undirected_linear', 'linear'):
        dim = 1
    # ---- 2D MDS (planar topology) ------------------------------------------
    elif topo in ('directed_2d', 'undirected_2d', 'grid'):
        dim = 2
    # ---- BFS hop-count (hyperbolic / tree topology) ------------------------
    elif topo in ('hyperbolic', 'directed_dag', 'undirected_tree', 'tree'):
        # Build category adjacency from _trans (if available)
        adj: list = [set() for _ in range(K)]
        if trans:
            for rel, src_map in trans.items():
                for c_src, tgt_map in src_map.items():
                    try:
                        i = int(c_src)
                    except (ValueError, TypeError):
                        continue
                    for c_tgt in tgt_map:
                        try:
                            j = int(c_tgt)
                        except (ValueError, TypeError):
                            continue
                        if 0 <= i < K and 0 <= j < K:
                            adj[i].add(j)
                            adj[j].add(i)
        # BFS hop distances
        hop = np.full((K, K), K, dtype=float)
        for src in range(K):
            hop[src, src] = 0
            queue = [src]
            visited = {src}
            dist = 0
            while queue:
                nxt = []
                dist += 1
                for node in queue:
                    for nb in adj[node]:
                        if nb not in visited:
                            visited.add(nb)
                            hop[src, nb] = dist
                            nxt.append(nb)
                queue = nxt
        max_hop = hop[hop < K].max() if (hop < K).any() else 1.0
        dist_mat = hop / max(max_hop, 1.0)
        m = [[math.exp(-temperature * dist_mat[i][j]) if i != j else 1.0
              for j in range(K)] for i in range(K)]
        return m

    # ---- dense / general_graph / unknown → JSD (no change) ----------------
    else:
        m = [[0.0] * K for _ in range(K)]
        for i in range(K):
            di = succ_dists.get(i, {})
            for j in range(K):
                m[i][j] = (1.0 if i == j
                           else math.exp(-temperature * D[i, j]))
        return m

    # ---- Classical MDS for linear or 2D topologies ------------------------
    n = K
    D2 = D ** 2
    # Double-centering
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    grand_mean = D2.mean()
    B = -0.5 * (D2 - row_mean - col_mean + grand_mean)

    # Eigendecompose (B is symmetric PSD)
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Take top 'dim' components (clamp negative eigenvalues to 0)
    lam = np.maximum(eigvals[:dim], 0.0)
    coords = eigvecs[:, :dim] * np.sqrt(lam)  # K × dim

    # L2 pairwise distances in embedding space
    m = [[0.0] * K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            if i == j:
                m[i][j] = 1.0
            else:
                d = float(np.linalg.norm(coords[i] - coords[j]))
                m[i][j] = math.exp(-temperature * d)
    return m


def _preprocess_patch(patch_uint8) -> 'np.ndarray':
    """Grayscale + local contrast normalization → flattened float32 vector.

    Same preprocessing as VisionLearner's _to_gray_f32, without quantization.
    Used as input to k-means codebook construction and nearest-centroid assignment.
    """
    import numpy as np
    arr = np.asarray(patch_uint8, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        gray = (0.299 * arr[:, :, 0] +
                0.587 * arr[:, :, 1] +
                0.114 * arr[:, :, 2])
    elif arr.ndim == 3:
        gray = arr[:, :, 0]
    else:
        gray = arr
    mn, mx = float(gray.min()), float(gray.max())
    if mx - mn > 1e-6:
        gray = (gray - mn) / (mx - mn)
    else:
        gray = np.zeros_like(gray)
    return gray.ravel()


def _build_patch_codebook(patches: list, k: int, max_iter: int = 30,
                           verbose: bool = False) -> 'np.ndarray':
    """K-means codebook on preprocessed patch vectors.

    Returns (k, D) float32 centroid array where D = patch_size².
    Guarantees exactly k visual-word tokens, forcing vocabulary repetition
    that distributional clustering requires.
    """
    import numpy as np
    X = np.array([_preprocess_patch(p) for p in patches], dtype=np.float32)
    n, D = X.shape
    k = min(k, n)
    rng = np.random.default_rng(42)
    centres = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int32)
    for it in range(max_iter):
        # Assignment: squared L2
        dists = np.sum((X[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels) and it > 0:
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centres[c] = X[mask].mean(axis=0)
    if verbose:
        counts = np.bincount(labels, minlength=k)
        print(f'  Codebook k={k}: '
              f'min={counts.min()} max={counts.max()} '
              f'mean={counts.mean():.1f} patches/word')
    return centres


def _quantize_patch(patch_uint8, bits: int) -> str:
    """Perceptually quantize a uint8 patch to a compact token string.

    Applies the same preprocessing as VisionLearner:
      1. Grayscale (ITU-R BT.601 luminance)
      2. Local contrast normalization (min-max per patch → [0, 1])
      3. n-bit quantization (2 = 4 levels, 3 = 8 levels)
      4. Bit packing → hex string

    Patches that are visually similar collapse to the same token, forcing
    vocabulary repetition necessary for distributional clustering.
    """
    import numpy as np
    arr = np.asarray(patch_uint8, dtype=np.float32)
    # Grayscale
    if arr.ndim == 3 and arr.shape[2] >= 3:
        gray = (0.299 * arr[:, :, 0] +
                0.587 * arr[:, :, 1] +
                0.114 * arr[:, :, 2])
    elif arr.ndim == 3:
        gray = arr[:, :, 0]
    else:
        gray = arr
    # Local contrast normalization
    mn, mx = float(gray.min()), float(gray.max())
    if mx - mn > 1e-6:
        gray = (gray - mn) / (mx - mn)
    else:
        gray = np.zeros_like(gray)
    # Quantize
    levels = (1 << bits) - 1
    q = np.clip(np.round(gray.ravel() * levels), 0, levels).astype(np.uint8)
    if bits <= 4:
        if len(q) % 2 != 0:
            q = np.concatenate([q, np.zeros(1, dtype=np.uint8)])
        packed = (q[0::2] << 4) | q[1::2]
        return 'q' + packed.tobytes().hex()
    return 'q' + q.tobytes().hex()


def _patch_hash(patch) -> str:
    """Raw MD5 hash of uint8 patch pixels (unique per distinct pixel pattern)."""
    import hashlib
    import numpy as np
    arr = np.asarray(patch, dtype=np.uint8)
    try:
        h = hashlib.md5(arr.tobytes(), usedforsecurity=False).hexdigest()[:8]
    except TypeError:
        h = hashlib.md5(arr.tobytes()).hexdigest()[:8]
    return f'p{h}'


# ---------------------------------------------------------------------------
# Level 2: RelationClusterer
# ---------------------------------------------------------------------------


def _jsd_cluster(sigs: dict[str, dict],
                 n_clusters: int | None = None,
                 jsd_threshold: float | None = None,
                 verbose: bool = False) -> tuple[dict, dict]:
    """Cluster named distributions by pairwise Jensen-Shannon divergence.

    Args:
        sigs:          {name: {key: probability}} — normalized distributions.
        n_clusters:    Target cluster count.  None → merge while JSD < threshold.
        jsd_threshold: Merge threshold when n_clusters is None.
                       None (default) → auto-detect via _gap_threshold() (Kneedle).
        verbose:       Print cluster assignments.

    Returns:
        (assignment {name → cluster_id},
         clusters   {cluster_id → [names]})
    """
    names = sorted(sigs.keys())
    n = len(names)
    if n == 0:
        return {}, {}
    if n == 1:
        return {names[0]: 0}, {0: names[:]}

    # Unified key space → float matrix (n × D)
    all_keys = sorted({k for sig in sigs.values() for k in sig})
    key_idx = {k: i for i, k in enumerate(all_keys)}
    D = len(all_keys)

    try:
        import numpy as np
        mat = np.zeros((n, D), dtype=np.float64)
        for i, name in enumerate(names):
            for k, v in sigs[name].items():
                mat[i, key_idx[k]] = v

        jsd_arr = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                m = (mat[i] + mat[j]) / 2
                def _h(p: 'np.ndarray') -> float:
                    mask = p > 1e-15
                    return float(-np.sum(p[mask] * np.log2(p[mask])))
                d = max(0.0, _h(m) - (_h(mat[i]) + _h(mat[j])) / 2)
                jsd_arr[i, j] = jsd_arr[j, i] = d

        def _get(i: int, j: int) -> float:
            return float(jsd_arr[i, j])
    except ImportError:
        jsd_list = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                p, q = sigs[names[i]], sigs[names[j]]
                all_k = set(p) | set(q)
                m = {k: (p.get(k, 0.0) + q.get(k, 0.0)) / 2 for k in all_k}
                def _h2(d: dict) -> float:
                    return -sum(v * math.log2(v) for v in d.values() if v > 1e-15)
                d = max(0.0, _h2(m) - (_h2(p) + _h2(q)) / 2)
                jsd_list[i][j] = jsd_list[j][i] = d

        def _get(i: int, j: int) -> float:  # type: ignore[misc]
            return jsd_list[i][j]

    # Greedy agglomerative clustering (single-link, ascending JSD)
    assignment = list(range(n))
    n_active = n
    pairs = sorted((_get(i, j), i, j) for i in range(n) for j in range(i + 1, n))

    # Auto-detect threshold from pairwise JSD distribution (Kneedle algorithm)
    if n_clusters is None and jsd_threshold is None:
        all_jsds = [d for d, _, _ in pairs]
        jsd_threshold = _gap_threshold(all_jsds)

    for d, i, j in pairs:
        if n_clusters is not None and n_active <= n_clusters:
            break
        if n_clusters is None and d > jsd_threshold:
            break
        ci, cj = assignment[i], assignment[j]
        if ci == cj:
            continue
        for idx in range(n):
            if assignment[idx] == cj:
                assignment[idx] = ci
        n_active -= 1

    # Remap to 0..K-1
    old_ids = sorted(set(assignment))
    remap = {old: new for new, old in enumerate(old_ids)}
    final = [remap[a] for a in assignment]

    asgn = {names[i]: final[i] for i in range(n)}
    clus: dict[int, list[str]] = collections.defaultdict(list)
    for name, cid in asgn.items():
        clus[cid].append(name)
    clus = dict(clus)

    if verbose:
        for cid, members in sorted(clus.items()):
            print(f'    cluster {cid}: {members}')
    return asgn, clus


class RelationClusterer:
    """Level 2: Cluster relation types by distributional behavior.

    For each relation r, builds its signature:
        rel_sig(r) = distribution over (c_src, c_tgt) category pairs connected by r

    Two relations are equivalent if they connect the same atom-category pairs.
    Discovers:
    - H and its reverse 'rev_H' are "the same relation, backwards"
    - Diagonal relations D1/D2 form a separate cluster from axial H/V
    - Relations equivalent by symmetry of the data merge automatically

    Implementation: build (relation,) → (f'{c_src}:{c_tgt}',) distributional
    signatures, compute pairwise JSD, greedy agglomerative clustering.
    """

    def __init__(self, n_clusters: int | None = None):
        self.n_clusters = n_clusters
        self.assignment: dict[str, int] = {}    # relation_name → cluster_id
        self.clusters:   dict[int, list[str]] = {}  # cluster_id → [rel_names]
        self._rel_sigs:  dict[str, dict] = {}   # rel → {(c_src, c_tgt): prob}
        self._jsd:       dict[tuple, float] = {}  # {(r1, r2): jsd_value}

    def fit(self, learner: 'RelationalLearner',
            triples: list[tuple[Any, str, Any]],
            jsd_threshold: float | None = None,
            verbose: bool = True) -> None:
        """Cluster relation types using atom assignments from a fitted RelationalLearner.

        Args:
            learner:       A fitted RelationalLearner (learner.assignment populated).
            triples:       The same (atom, relation, atom) triples used to fit learner.
            jsd_threshold: JSD below which two relations are merged (n_clusters=None).
                           None (default) → auto-detect via Kneedle algorithm.
            verbose:       Print cluster assignments.
        """
        rel_counts: dict[str, dict] = collections.defaultdict(
            lambda: collections.defaultdict(float))
        n_missing = 0
        for a, r, b in triples:
            a_s, r_s, b_s = str(a), str(r), str(b)
            c_a = learner.assignment.get(a_s)
            c_b = learner.assignment.get(b_s)
            if c_a is None or c_b is None:
                n_missing += 1
                continue
            rel_counts[r_s][(c_a, c_b)] += 1.0

        if not rel_counts:
            return

        for r, counts in rel_counts.items():
            total = sum(counts.values())
            self._rel_sigs[r] = {k: v / total for k, v in counts.items()}

        if verbose:
            print(f'  RelationClusterer: {len(rel_counts)} relations, '
                  f'{n_missing} triples with OOV atoms')

        # Convert tuple keys → string for _jsd_cluster
        str_sigs = {
            r: {f'{k[0]}:{k[1]}': v for k, v in sig.items()}
            for r, sig in self._rel_sigs.items()
        }
        self.assignment, self.clusters = _jsd_cluster(
            str_sigs, n_clusters=self.n_clusters,
            jsd_threshold=jsd_threshold, verbose=verbose)

        if verbose:
            for cid, members in sorted(self.clusters.items()):
                print(f'  Rel-cluster {cid}: {members}')

    def jsd_between(self, r1: str, r2: str) -> float:
        """JSD between the (c_src, c_tgt) distributions of two relations."""
        p = self._rel_sigs.get(r1, {})
        q = self._rel_sigs.get(r2, {})
        if not p or not q:
            return 1.0
        all_k = set(p) | set(q)
        m = {k: (p.get(k, 0.0) + q.get(k, 0.0)) / 2 for k in all_k}
        def _h(d: dict) -> float:
            return -sum(v * math.log2(v) for v in d.values() if v > 1e-15)
        return max(0.0, _h(m) - (_h(p) + _h(q)) / 2)


# ---------------------------------------------------------------------------
# Level 4: SecondOrderGrammar
# ---------------------------------------------------------------------------

class SecondOrderGrammar:
    """Level 4: Learn next_rel(r1) → r2 — which relations follow other relations.

    This is E1 applied to relations rather than atoms: finds the distributional
    structure *of* the relational structure itself.

    Discovers:
    - Grammatical sequences in text:  SUBJ → VERB, DET → NOUN
    - Spatial patterns in 2D images:  H after H = horizontal run; V after H = corner
    - Causal chains in action sequences: PUSH → OPEN → ENTER

    Implementation:
    1. Find all chains a -r1→ b -r2→ c in the triple set.
    2. Accumulate P(r2 | r1) from chain counts.
    3. Cluster relations by JSD of their P(next_rel | r) distributions.
    """

    def __init__(self):
        self.next_rel_dist: dict[str, dict[str, float]] = {}  # r1 → {r2: prob}
        self.assignment:    dict[str, int] = {}   # relation → cluster_id
        self.clusters:      dict[int, list[str]] = {}  # cluster_id → [rel_names]
        self._n_chains: int = 0

    def fit(self, triples: list[tuple[Any, str, Any]],
            n_clusters: int | None = None,
            jsd_threshold: float | None = None,
            verbose: bool = True) -> None:
        """Learn next_rel distributions from chains in the triple set.

        Finds all (a -r1→ b) AND (b -r2→ c) chains, accumulates (r1, r2) pairs,
        then clusters relations by JSD of their P(next_rel | r) distributions.

        Args:
            triples:       (atom, relation, atom) triples.
            n_clusters:    Target cluster count.  None → JSD-threshold merging.
            jsd_threshold: Merge threshold when n_clusters is None.
                           None (default) → auto-detect via Kneedle algorithm.
            verbose:       Print distributions and clusters.
        """
        # O(N) chain count via in/out aggregation.
        # Naïve triple-loop is O(N × avg_out_degree) = O(N²/V), hanging on
        # large corpora (180B iterations for 2.2M triples over 27 chars).
        #
        # chain_count[r1][r2] = Σ_b  in_count[b][r1] × out_count[b][r2]
        # where:
        #   in_count[b][r1]  = # triples (?, r1, b)   [edges arriving at b]
        #   out_count[b][r2] = # triples (b, r2, ?)   [edges leaving b]
        #
        # O(N) to build counts, O(V × R²) to aggregate.
        in_count:  dict[str, dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int))
        out_count: dict[str, dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int))
        for a, r, b in triples:
            r_s, b_s, a_s = str(r), str(b), str(a)
            in_count[b_s][r_s]  += 1
            out_count[a_s][r_s] += 1

        seq_counts: dict[str, dict[str, float]] = collections.defaultdict(
            lambda: collections.defaultdict(float))
        for b_s in set(in_count) & set(out_count):
            for r1, cnt_in in in_count[b_s].items():
                for r2, cnt_out in out_count[b_s].items():
                    seq_counts[r1][r2] += float(cnt_in * cnt_out)
        self._n_chains = sum(
            int(cnt) for d in seq_counts.values() for cnt in d.values())

        if not seq_counts:
            if verbose:
                print('  SecondOrderGrammar: no chains found in triples')
            return

        for r, counts in seq_counts.items():
            total = sum(counts.values())
            self.next_rel_dist[r] = {k: v / total for k, v in counts.items()}

        if verbose:
            print(f'  SecondOrderGrammar: {len(seq_counts)} relations, '
                  f'{self._n_chains} chains')
            for r in sorted(seq_counts):
                top = sorted(self.next_rel_dist[r].items(), key=lambda x: -x[1])[:4]
                top_str = ', '.join(f'{r2}({p:.2f})' for r2, p in top)
                print(f'    P(next | {r}): [{top_str}]')

        if len(seq_counts) < 2:
            rels = sorted(seq_counts)
            self.assignment = {rels[0]: 0}
            self.clusters = {0: rels}
            return

        self.assignment, self.clusters = _jsd_cluster(
            self.next_rel_dist,
            n_clusters=n_clusters,
            jsd_threshold=jsd_threshold,
            verbose=verbose)

        if verbose:
            for cid, members in sorted(self.clusters.items()):
                print(f'  Rel-cluster {cid}: {members}')

    def predict_next_rel(self, relation: str) -> str | None:
        """Most likely next relation after the given relation."""
        dist = self.next_rel_dist.get(str(relation), {})
        if not dist:
            return None
        return max(dist, key=dist.get)

    def next_rel_distribution(self, relation: str) -> dict[str, float]:
        """Full distribution over next relations."""
        return dict(self.next_rel_dist.get(str(relation), {}))


# ---------------------------------------------------------------------------
# R0: GeometryDetector — What shape is the data?
# ---------------------------------------------------------------------------

class GeometryDetector:
    """R0: Infer the underlying geometry of a relational dataset.

    Given a fitted RelationalLearner + RelationClusterer, computes four
    structure metrics and classifies the topology without any prior assumption
    about dimensionality, curvature, or direction.

    Metrics
    -------
    symmetry_score : float in [0, 1]
        1 = fully undirected (every relation has a near-identical reverse).
        0 = fully directed (no relation pair is mutually inverse).
        Computed by testing whether T[r] ≈ T[r']ᵀ for all relation pairs.

    effective_rank : int
        Approximate rank of the aggregate K×K category transition matrix.
        1 → linear structure.  2 → planar.  K → full graph.
        Estimated via the fraction of cumulative singular-value mass.

    growth_profile : List[int]
        |reachable categories| at hop distance d = 0, 1, 2, ...
        Exponential growth → hyperbolic (tree-like).
        Polynomial d → Euclidean d-dimensional.
        Plateau after D steps → bounded/spherical or dense graph.

    has_cycles : bool
        Whether the category transition graph has any directed cycle.
        False → DAG / tree.  True → cyclic / circular / general graph.

    Topology labels
    ---------------
    directed_linear      symmetry≈0, rank=1, growth~d, no broad cycles
    undirected_linear    symmetry≈1, rank=1, growth~d
    directed_2d          symmetry≈0, rank=2, growth~d²
    undirected_2d        symmetry≈1, rank=2, growth~d²
    hyperbolic           growth exponential (tree-like)
    spherical            growth peaks then shrinks (bounded)
    directed_graph       symmetry≈0, high rank, general
    undirected_graph     symmetry≈1, high rank, general
    """

    def __init__(self) -> None:
        self.symmetry_score:  float      = 0.0
        self.effective_rank:  int        = 0
        self.growth_profile:  list       = []
        self.has_cycles:      bool       = False
        self.curvature:       str        = 'unknown'
        self.topology:        str        = 'unknown'
        self._trans:          dict       = {}   # {rel: {c_src: {c_tgt: prob}}}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self,
            learner:      'RelationalLearner',
            rel_clusterer: 'RelationClusterer',
            ) -> 'GeometryDetector':
        """Detect geometry from a fitted RelationalLearner + RelationClusterer."""
        self._trans  = self._build_trans(learner)
        K            = learner._K
        relations    = list(self._trans.keys())

        self.symmetry_score  = self._compute_symmetry(relations, K)
        self.effective_rank  = self._compute_rank(relations, K)
        self.growth_profile  = self._compute_growth(relations, K)
        self.has_cycles      = self._compute_cycles(relations, K)
        self.curvature       = self._classify_curvature()
        self.topology        = self._classify_topology()
        return self

    def report(self) -> str:
        lines = [
            '=' * 65,
            'R0  GEOMETRY DETECTION',
            '=' * 65,
            f'  Symmetry score : {self.symmetry_score:.3f}  '
            f'(1=undirected, 0=directed)',
            f'  Effective rank : {self.effective_rank}  '
            f'(proxy for embedding dimension)',
            f'  Has cycles     : {self.has_cycles}',
            f'  Curvature      : {self.curvature}',
            f'  Topology       : {self.topology}',
            '',
            f'  Growth profile (|reachable categories| at hop d):',
        ]
        for d, n in enumerate(self.growth_profile):
            bar = '█' * min(40, n)
            lines.append(f'    hop {d}: {n:3d}  {bar}')
        lines.append('=' * 65)
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_trans(self, learner: 'RelationalLearner') -> dict:
        """Return the T[rel][c_src][c_tgt] matrix built during learner.fit()."""
        return getattr(learner, '_trans', {})

    def _compute_symmetry(self, relations: list, K: int) -> float:
        """Symmetry score: mean max-transpose-similarity over all relation pairs.

        For each ordered pair (r, r'), compute the JSD between the flattened
        transition matrix T[r] and the transpose T[r']ᵀ.  A low JSD means r'
        is the reverse of r → symmetric (undirected) relation pair.
        The symmetry score is 1 - mean(min_JSD over partner) over all relations.
        """
        import math as _math

        def _flat(r: str, transpose: bool) -> list:
            mat = self._trans.get(r, {})
            cats = sorted({c for m in mat.values() for c in m} | set(mat))
            n = len(cats)
            if n == 0:
                return []
            idx = {c: i for i, c in enumerate(cats)}
            vec = [0.0] * (n * n)
            for c_src, row in mat.items():
                i = idx.get(c_src)
                if i is None:
                    continue
                for c_tgt, p in row.items():
                    j = idx.get(c_tgt)
                    if j is None:
                        continue
                    pos = (j * n + i) if transpose else (i * n + j)
                    vec[pos] = p
            total = sum(vec)
            if total < 1e-12:
                return vec
            return [v / total for v in vec]

        def _jsd_vecs(p: list, q: list) -> float:
            if len(p) != len(q) or not p:
                return 1.0
            result = 0.0
            for pi, qi in zip(p, q):
                mi = 0.5 * (pi + qi)
                if pi > 1e-12 and mi > 1e-12:
                    result += 0.5 * pi * _math.log2(pi / mi)
                if qi > 1e-12 and mi > 1e-12:
                    result += 0.5 * qi * _math.log2(qi / mi)
            return max(0.0, min(1.0, result))

        if len(relations) < 2:
            return 1.0

        scores = []
        for r in relations:
            p = _flat(r, transpose=False)
            if not p:
                continue
            min_jsd = min(
                _jsd_vecs(p, _flat(r2, transpose=True))
                for r2 in relations
            )
            scores.append(1.0 - min_jsd)
        return sum(scores) / len(scores) if scores else 0.0

    def _compute_rank(self, relations: list, K: int) -> int:
        """Effective rank of the aggregate K×K transition matrix via SVD."""
        try:
            import numpy as np
        except ImportError:
            return -1   # numpy not available

        all_cats = sorted({
            c
            for mat in self._trans.values()
            for c_src, row in mat.items()
            for c in [c_src] + list(row)
        })
        n = len(all_cats)
        if n == 0:
            return 0
        idx = {c: i for i, c in enumerate(all_cats)}
        T = np.zeros((n, n), dtype=np.float64)
        for mat in self._trans.values():
            for c_src, row in mat.items():
                i = idx.get(c_src)
                if i is None:
                    continue
                for c_tgt, p in row.items():
                    j = idx.get(c_tgt)
                    if j is None:
                        continue
                    T[i, j] += p
        total = T.sum()
        if total > 0:
            T /= total
        sv = np.linalg.svd(T, compute_uv=False)
        cum = sv.cumsum() / (sv.sum() + 1e-12)
        # Effective rank = number of singular values needed for 90% of mass.
        rank = int(np.searchsorted(cum, 0.90)) + 1
        return rank

    def _compute_growth(self, relations: list, K: int,
                        max_hops: int = 10) -> list:
        """BFS on the category graph to measure |reachable| at each hop."""
        # Build adjacency: c_src → set of c_tgt (across all relations).
        adj: dict = {}
        for mat in self._trans.values():
            for c_src, row in mat.items():
                adj.setdefault(c_src, set()).update(row.keys())

        all_cats = list(adj.keys())
        if not all_cats:
            return [0]

        # Pick the most-connected seed.
        seed = max(all_cats, key=lambda c: len(adj.get(c, set())))
        visited = {seed}
        frontier = {seed}
        profile = [1]
        for _ in range(max_hops):
            next_f = set()
            for c in frontier:
                next_f.update(adj.get(c, set()) - visited)
            if not next_f:
                break
            visited.update(next_f)
            profile.append(len(next_f))
            frontier = next_f
        return profile

    def _compute_cycles(self, relations: list, K: int) -> bool:
        """Detect directed cycles in the category graph via DFS."""
        adj: dict = {}
        for mat in self._trans.values():
            for c_src, row in mat.items():
                adj.setdefault(c_src, set()).update(row.keys())

        visited: set = set()
        in_stack: set = set()

        def _dfs(node: str) -> bool:
            visited.add(node)
            in_stack.add(node)
            for nb in adj.get(node, set()):
                if nb not in visited:
                    if _dfs(nb):
                        return True
                elif nb in in_stack:
                    return True
            in_stack.discard(node)
            return False

        for node in list(adj.keys()):
            if node not in visited:
                if _dfs(node):
                    return True
        return False

    def _classify_curvature(self) -> str:
        """Infer curvature from growth profile shape."""
        p = self.growth_profile
        if len(p) < 2:
            return 'unknown'
        # If all nodes reachable in 1 hop → dense / complete graph.
        if len(p) == 2 and p[1] > 0:
            return 'dense (all nodes reachable in 1 hop)'
        if len(p) < 3:
            return 'unknown'
        diffs = [p[i+1] - p[i] for i in range(len(p)-1)]
        if not diffs:
            return 'unknown'
        increasing = sum(1 for i in range(len(diffs)-1) if diffs[i+1] > diffs[i])
        decreasing = sum(1 for i in range(len(diffs)-1) if diffs[i+1] < diffs[i])
        total = len(diffs) - 1
        if total == 0:
            return 'zero (Euclidean/flat)'
        if increasing / total > 0.6:
            return 'negative (hyperbolic/tree)'
        if decreasing / total > 0.6:
            return 'positive (spherical/bounded)'
        return 'zero (Euclidean/flat)'

    def _classify_topology(self) -> str:
        s = self.symmetry_score
        r = self.effective_rank
        cyc = self.has_cycles
        curv = self.curvature

        if 'hyperbolic' in curv:
            return 'hyperbolic (tree/DAG)'
        if r <= 1:
            if s > 0.7:
                return 'undirected_linear'
            return 'directed_linear'
        if r == 2:
            if s > 0.7:
                return 'undirected_2d'
            return 'directed_2d'
        # High rank.
        if s > 0.7:
            return 'undirected_graph'
        return 'directed_graph'


# ---------------------------------------------------------------------------
# R1: RelationalParadigmDiscoverer — Relational E4
# ---------------------------------------------------------------------------

class RelationalParadigmDiscoverer:
    """R1/E4: Cluster atoms by the relational *role* they play in the graph.

    Orthogonal to E0 clustering:

    - **E0** clusters atoms by *what neighbors they have* (distributional context).
      Two atoms cluster together if similar tokens tend to follow/precede them.

    - **E4** clusters atoms by *what role they play* — which relation types they
      participate in, and what *category* of atom they connect to as source or target.
      Two atoms cluster together if they are interchangeable without changing the
      relational skeleton.

    Role signature of atom *a*:
        ``role_sig(a) = P(direction, relation, partner_category)``

        - ``'>rel:c_tgt'`` — *a* is the SOURCE of a *rel*-relation to a *c_tgt*-atom.
        - ``'<rel:c_src'`` — *a* is the TARGET of a *rel*-relation from a *c_src*-atom.

    This is built at **category level** (partner identified by E0 cluster id, not
    surface form), so it generalises across surface variation.

    Example (knowledge graph):
        E0 might cluster {Paris, London, Berlin} together (similar neighbours).
        E4 clusters them into *roles*: all three are ``capital_of`` **sources**
        pointing to ``country``-category targets → same role, substitutable.

    Example (Latin chars):
        Space has a unique role: it is target of word-final chars and source of
        word-initial chars.  'q' has a unique role: always source of a next-relation
        to 'u'.  Common vowels may share a role distinct from common consonants if
        their category-level skip patterns differ.
    """

    def __init__(self) -> None:
        self.role_assignment: dict = {}   # atom_str → role_id
        self.role_clusters:   dict = {}   # role_id → [atom_str]
        self._K_roles:        int  = 0
        self._role_sigs:      dict = {}   # atom_str → {key: prob}  (for inspection)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self,
            learner:           'RelationalLearner',
            triples:           list,
            verbose:           bool = False,
            use_atom_partners: bool = True,
            ) -> 'RelationalParadigmDiscoverer':
        """Fit role signatures from (atom, relation, atom) triples.

        Args:
            use_atom_partners: If True (default), partner atoms are identified by
                their surface form.  This gives a V-dimensional role signature and
                can separate fine-grained roles even when E0 has a mega-cluster.
                If False, use E0 category id as partner label (K-dimensional),
                which is more abstract but loses sub-category distinctions.

        One O(N) pass to build raw counts, then O(V²×D) clustering where V
        is the number of unique atoms and D is the role-signature dimension.
        For character-level data: V=27, D≤R×V≈108 → instant.
        """
        from collections import defaultdict, Counter as _Ctr

        src_raw: dict = defaultdict(_Ctr)   # atom → {'>rel:partner': count}
        tgt_raw: dict = defaultdict(_Ctr)   # atom → {'<rel:partner': count}

        for a, r, b in triples:
            a_s, r_s, b_s = str(a), str(r), str(b)
            if use_atom_partners:
                partner_a = b_s   # partner of a is b (surface form)
                partner_b = a_s   # partner of b is a (surface form)
            else:
                c_a = learner.assignment.get(a_s)
                c_b = learner.assignment.get(b_s)
                if c_a is None or c_b is None:
                    continue
                partner_a = str(c_b)
                partner_b = str(c_a)
            src_raw[a_s][f'>{r_s}:{partner_a}'] += 1   # a is source
            tgt_raw[b_s][f'<{r_s}:{partner_b}'] += 1   # b is target

        # Merge source and target counts into a single normalised signature.
        all_atoms = set(src_raw) | set(tgt_raw)
        sigs: dict = {}
        for atom in all_atoms:
            combined: dict = {}
            total = 0
            for k, v in src_raw.get(atom, {}).items():
                combined[k] = combined.get(k, 0) + v
                total += v
            for k, v in tgt_raw.get(atom, {}).items():
                combined[k] = combined.get(k, 0) + v
                total += v
            if total > 0:
                sigs[atom] = {k: v / total for k, v in combined.items()}

        self._role_sigs = sigs

        if not sigs:
            if verbose:
                print('  E4: no data — skipping')
            return self

        assignment, clusters = _jsd_cluster(sigs, verbose=False)
        self.role_assignment = assignment
        self.role_clusters   = clusters
        self._K_roles        = len(clusters)

        if verbose:
            print(f'  E4: {len(sigs)} atoms → {self._K_roles} role categories')

        return self

    def role_occupants(self, role_id: int) -> list:
        """Atoms sharing role *role_id* (substitutable in the same slot)."""
        return self.role_clusters.get(role_id, [])

    def report(self, learner: 'RelationalLearner | None' = None) -> str:
        """Human-readable role cluster summary with cross-reference to E0 clusters."""
        lines = [
            '=' * 65,
            'R1  PARADIGMATIC ROLES  (atoms by relational role, E4)',
            '=' * 65,
        ]
        for rid in sorted(self.role_clusters):
            members = sorted(self.role_clusters[rid])
            # Show E0 category membership alongside, for comparison.
            if learner is not None:
                cats = sorted({learner.assignment.get(m, '?') for m in members})
                cat_str = f'  [E0 cats: {cats}]'
            else:
                cat_str = ''
            # Show top role-signature keys for the first member.
            top_sig = ''
            if members and self._role_sigs:
                sig = self._role_sigs.get(members[0], {})
                top = sorted(sig.items(), key=lambda kv: -kv[1])[:3]
                top_sig = '  top: ' + ', '.join(f'{k}({v:.2f})' for k, v in top)
            label = ' '.join(members) if len(members) <= 12 else (
                ' '.join(members[:12]) + f' +{len(members)-12}')
            lines.append(f'  Role {rid:2d}: [{label}]{cat_str}{top_sig}')
        lines.append('')
        lines.append(f'  {self._K_roles} role categories  '
                     f'({len(self.role_assignment)} atoms)')
        lines.append('=' * 65)
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# R2: RelationalSenseSplitter — Sense Disambiguation (E5)
# ---------------------------------------------------------------------------


def _jsd_dists(p: dict, q: dict) -> float:
    """Jensen-Shannon divergence between two dict-based probability distributions."""
    all_keys = set(p) | set(q)
    m = {k: (p.get(k, 0.0) + q.get(k, 0.0)) / 2.0 for k in all_keys}

    def _h(d: dict) -> float:
        return -sum(v * math.log2(v) for v in d.values() if v > 1e-15)

    return max(0.0, _h(m) - (_h(p) + _h(q)) / 2.0)


class RelationalSenseSplitter:
    """R2/E5: Detect atoms with polysemous relational behavior.

    An atom is polysemous if its *forward* distribution (what follows it) changes
    significantly depending on its *backward* context (what precedes it).

    Algorithm:
    1. Scan sequences.  At each position i, record (prev_atom, atom, next_atom).
    2. For each atom 'a', build conditional forward distributions
       P(next | prev=x) for every observed prev_atom x.
    3. Cluster these conditional distributions by JSD (auto-K).
    4. If ≥2 clusters survive with inter-cluster JSD ≥ min_sense_jsd → polysemy.

    Example (Latin):
        'v' sometimes acts as a vowel (u) and sometimes as a consonant.
        After vowels, P(next) ≈ consonant-heavy (consonant onset).
        After consonants, P(next) ≈ vowel-heavy (vowel continuation).
        → two sense clusters → 'v' is polysemous.
    """

    def __init__(self) -> None:
        self.polysemous: dict = {}  # atom → {n_senses, max_jsd, ...}

    def fit(self,
            sequences:         list,
            verbose:           bool  = False,
            min_sense_jsd:     float = 0.3,
            min_context_count: int   = 5,
            ) -> 'RelationalSenseSplitter':
        """
        Args:
            sequences:         Original token sequences (list of list of str/char).
            min_sense_jsd:     Minimum inter-cluster JSD to confirm polysemy.
            min_context_count: Minimum occurrences of a (prev, atom) pair to include.
        """
        # tri_counts[atom][prev_atom] → Counter(next_atom → count)
        tri_counts: dict = collections.defaultdict(
            lambda: collections.defaultdict(collections.Counter))

        for seq in sequences:
            n = len(seq)
            for i in range(1, n - 1):
                atom = str(seq[i])
                prev = str(seq[i - 1])
                nxt  = str(seq[i + 1])
                tri_counts[atom][prev][nxt] += 1

        polysemous: dict = {}

        for atom in sorted(tri_counts):
            # Build conditional distributions P(next | prev=x)
            cond: dict = {}
            for prev_atom, next_ctr in tri_counts[atom].items():
                total = sum(next_ctr.values())
                if total >= min_context_count:
                    cond[prev_atom] = {k: v / total for k, v in next_ctr.items()}

            if len(cond) < 2:
                continue  # not enough context diversity

            # Cluster conditional distributions by JSD (auto-K via Kneedle)
            assignment, clusters = _jsd_cluster(cond, verbose=False)

            if len(clusters) < 2:
                continue

            # Merge per-cluster: average of member distributions
            cluster_profiles: dict = {}
            for rid, members in clusters.items():
                merged: dict = {}
                for m in members:
                    for k, v in cond[m].items():
                        merged[k] = merged.get(k, 0.0) + v / len(members)
                cluster_profiles[rid] = merged

            # Max pairwise JSD between cluster profiles
            rids = sorted(cluster_profiles)
            max_jsd = 0.0
            for a in range(len(rids)):
                for b in range(a + 1, len(rids)):
                    max_jsd = max(max_jsd,
                                  _jsd_dists(cluster_profiles[rids[a]],
                                             cluster_profiles[rids[b]]))

            if max_jsd < min_sense_jsd:
                continue

            polysemous[atom] = {
                'n_senses':          len(clusters),
                'max_jsd':           max_jsd,
                'cluster_assign':    assignment,   # prev_atom → sense_id
                'sense_profiles':    cluster_profiles,  # sense_id → P(next)
            }

            if verbose:
                for sid in sorted(cluster_profiles):
                    top_prevs = [p for p, s in assignment.items() if s == sid][:5]
                    top_next  = sorted(cluster_profiles[sid].items(),
                                       key=lambda kv: -kv[1])[:4]
                    top_str   = ', '.join(f'{k}({v:.2f})' for k, v in top_next)
                    print(f'  E5: {atom!r} sense {sid} '
                          f'(prev ∈ {top_prevs}): next → {top_str}')

        self.polysemous = polysemous

        if verbose:
            print(f'  E5: {len(polysemous)} polysemous atoms '
                  f'(of {len(tri_counts)} total atoms)')

        return self

    def report(self) -> str:
        lines = [
            '=' * 65,
            'R2  SENSE DISAMBIGUATION  (polysemous atoms, E5)',
            '=' * 65,
        ]
        if not self.polysemous:
            lines.append('  No polysemous atoms detected.')
        else:
            for atom, info in sorted(self.polysemous.items()):
                lines.append(
                    f'\n  Atom {atom!r}: {info["n_senses"]} senses  '
                    f'(max_JSD={info["max_jsd"]:.3f})')
                for sid in sorted(info['sense_profiles']):
                    profile = info['sense_profiles'][sid]
                    top_next = sorted(profile.items(),
                                      key=lambda kv: -kv[1])[:5]
                    top_str  = ', '.join(f'{k}({v:.2f})' for k, v in top_next)
                    prevs = [p for p, s in info['cluster_assign'].items()
                             if s == sid]
                    lines.append(f'    Sense {sid}  '
                                 f'(prev ∈ {prevs[:8]}): '
                                 f'next → {top_str}')
        lines.append('')
        lines.append('=' * 65)
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# R3: RelationalAlgebra — Structural Meta-Synthesis (E6)
# ---------------------------------------------------------------------------


class RelationalAlgebra:
    """R3/E6: Discover composition rules R_i ∘ R_j = R_k.

    Tests whether the 2-hop transition matrix for every (R_i, R_j) pair
    approximates any single-hop relation R_k.  If yes, records the rule.
    Builds the full relational composition table.

    Example (Latin chars):
        next ∘ next ≈ skip2f    (two forward steps = skip-by-2)
        prev ∘ prev ≈ skip2b
        next ∘ prev ≈ identity  (forward then back = stay in place)

    Example (knowledge graph):
        capital_of ∘ in_continent = cities_in_continent
        PARENT ∘ PARENT = GRANDPARENT
    """

    def __init__(self) -> None:
        self.composition_table: dict = {}  # (r_i, r_j) → (r_k | None, jsd)
        self.relations:          list = []

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self,
            learner:       'RelationalLearner',
            triples:       list | None = None,
            verbose:       bool        = False,
            max_jsd:       float       = 0.3,
            ) -> 'RelationalAlgebra':
        """
        Args:
            learner:  Fitted RelationalLearner (provides ``assignment``).
            triples:  Raw (atom, rel, atom) triples.  When provided, builds
                      atom-level V×V matrices (higher resolution than K×K).
                      For V≤300, this is fast enough in pure Python.
                      For V>300, falls back to K×K category-level matrices.
            max_jsd:  Maximum row-weighted average JSD to accept a rule.
        """
        if triples is not None:
            V = len(learner.assignment)
            use_atom_level = (V <= 300)
        else:
            use_atom_level = False

        if use_atom_level:
            return self._fit_atom_level(learner, triples, verbose, max_jsd)
        else:
            return self._fit_cat_level(learner, verbose, max_jsd)

    def _fit_atom_level(self,
                        learner: 'RelationalLearner',
                        triples: list,
                        verbose: bool,
                        max_jsd: float,
                        ) -> 'RelationalAlgebra':
        """Atom-level V×V matrices — high resolution, O(V³) composition."""
        atoms   = sorted(learner.assignment.keys())
        V       = len(atoms)
        idx     = {a: i for i, a in enumerate(atoms)}

        # Count raw triples
        raw: dict = collections.defaultdict(int)             # (rel, src_i, tgt_j) → count
        rel_src_total: dict = collections.defaultdict(       # rel → {src_i → total}
            lambda: collections.defaultdict(int))

        relations_seen: set = set()
        for a, r, b in triples:
            a_s, r_s, b_s = str(a), str(r), str(b)
            if a_s not in idx or b_s not in idx:
                continue
            i, j = idx[a_s], idx[b_s]
            raw[(r_s, i, j)]     += 1
            rel_src_total[r_s][i] += 1
            relations_seen.add(r_s)

        relations = sorted(relations_seen)
        self.relations = relations

        # Build V×V probability matrices
        def _make_mat(rel: str) -> list:
            mat = [[0.0] * V for _ in range(V)]
            for i, tot in rel_src_total[rel].items():
                if tot <= 0:
                    continue
                for j in range(V):
                    c = raw.get((rel, i, j), 0)
                    if c:
                        mat[i][j] = c / tot
            return mat

        mats = {r: _make_mat(r) for r in relations}

        # Atom frequencies for weighted JSD
        atom_freq = [1] * V  # uniform weight — all atoms equally important

        self._finish(mats, relations, V, atom_freq, max_jsd, verbose)
        return self

    def _fit_cat_level(self,
                       learner: 'RelationalLearner',
                       verbose: bool,
                       max_jsd: float,
                       ) -> 'RelationalAlgebra':
        """Category-level K×K matrices — lower resolution, works without triples."""
        trans = getattr(learner, '_trans', {})
        K     = learner._K or 0

        if not trans or K == 0:
            if verbose:
                print('  R3: _trans empty — skipping')
            return self

        relations = sorted(trans.keys())
        self.relations = relations

        cat_freq = [0] * K
        for cat in learner.assignment.values():
            if 0 <= cat < K:
                cat_freq[cat] += 1

        def _make_mat(rel: str) -> list:
            mat = [[0.0] * K for _ in range(K)]
            for c_src, dist in trans.get(rel, {}).items():
                try:
                    i = int(c_src)
                except (ValueError, TypeError):
                    continue
                if not (0 <= i < K):
                    continue
                tot = sum(dist.values())
                if tot <= 0:
                    continue
                for c_tgt, p in dist.items():
                    try:
                        j = int(c_tgt)
                    except (ValueError, TypeError):
                        continue
                    if 0 <= j < K:
                        mat[i][j] = p / tot
            return mat

        mats = {r: _make_mat(r) for r in relations}
        self._finish(mats, relations, K, cat_freq, max_jsd, verbose)
        return self

    def _finish(self,
                mats:      dict,
                relations: list,
                N:         int,
                weights:   list,
                max_jsd:   float,
                verbose:   bool,
                ) -> None:
        """Matrix multiply, compare, build composition table."""

        def _compose(m1: list, m2: list) -> list:
            res = [[0.0] * N for _ in range(N)]
            for i in range(N):
                for k in range(N):
                    if m1[i][k] > 1e-10:
                        for j in range(N):
                            res[i][j] += m1[i][k] * m2[k][j]
            return res

        def _row_jsd(a: list, b: list) -> float:
            m = [(a[x] + b[x]) / 2.0 for x in range(N)]
            def _h(v: list) -> float:
                return -sum(x * math.log2(x) for x in v if x > 1e-15)
            return max(0.0, _h(m) - (_h(a) + _h(b)) / 2.0)

        def _mat_jsd(m1: list, m2: list) -> float:
            w_sum = j_sum = 0.0
            for i in range(N):
                w = weights[i]
                if w > 0:
                    j_sum += w * _row_jsd(m1[i], m2[i])
                    w_sum += w
            return j_sum / w_sum if w_sum > 0 else 1.0

        # First pass: find best-match JSD for every pair
        best_matches: list = []  # (r_i, r_j, best_r, best_jsd)
        for r_i in relations:
            for r_j in relations:
                m_comp   = _compose(mats[r_i], mats[r_j])
                best_r:   str | None = None
                best_jsd: float      = float('inf')
                for r_k in relations:
                    d = _mat_jsd(m_comp, mats[r_k])
                    if d < best_jsd:
                        best_jsd = d
                        best_r   = r_k
                best_matches.append((r_i, r_j, best_r, best_jsd))

        # Auto-detect threshold from gap in best-JSD values (Kneedle)
        all_jsds = sorted(jsd for _, _, _, jsd in best_matches)
        auto_thr = _gap_threshold(all_jsds)
        threshold = min(max_jsd, auto_thr) if auto_thr < float('inf') else max_jsd

        if verbose:
            print(f'  R3: auto-threshold={threshold:.3f}  '
                  f'(from gap in {len(all_jsds)} JSD values)')

        table: dict = {}
        for r_i, r_j, best_r, best_jsd in best_matches:
            confirmed = (best_jsd <= threshold)
            table[(r_i, r_j)] = (best_r if confirmed else None, best_jsd)

        self.composition_table = table

        if verbose:
            for (r_i, r_j), (r_k, jsd) in sorted(table.items()):
                arrow = f'= {r_k}' if r_k else '= ? (novel)'
                print(f'  R3: {r_i} ∘ {r_j} {arrow}  (JSD={jsd:.3f})')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compose(self, rel1: str, rel2: str) -> str | None:
        """Return the equivalent single relation for rel1 ∘ rel2, or None."""
        entry = self.composition_table.get((rel1, rel2))
        return entry[0] if entry else None

    def report(self) -> str:
        lines = [
            '=' * 65,
            'R3  RELATIONAL ALGEBRA  (composition rules R_i ∘ R_j = R_k)',
            '=' * 65,
        ]
        if not self.composition_table:
            lines.append('  No composition rules computed.')
        else:
            confirmed = [(ri, rj, rk, jsd)
                         for (ri, rj), (rk, jsd) in sorted(self.composition_table.items())
                         if rk is not None]
            novel     = [(ri, rj, jsd)
                         for (ri, rj), (rk, jsd) in sorted(self.composition_table.items())
                         if rk is None]
            if confirmed:
                lines.append('  Confirmed rules:')
                for ri, rj, rk, jsd in confirmed:
                    lines.append(f'    {ri} ∘ {rj} = {rk}  (JSD={jsd:.3f})')
            if novel:
                lines.append('  Novel (no matching single relation):')
                for ri, rj, jsd in novel:
                    lines.append(f'    {ri} ∘ {rj} = ?  (min_JSD={jsd:.3f})')
        lines.append('')
        lines.append('=' * 65)
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Level 5a: RawOffsetLearner — Symmetry Discovery
# ---------------------------------------------------------------------------

class RawOffsetLearner:
    """Level 5a: Discover 2D spatial symmetry with zero built-in structure.

    Standard Image2DRelationalLearner hardcodes H/V/D1/D2 as the four spatial
    relations.  RawOffsetLearner instead uses raw pixel offsets (dr, dc) as
    relation names — every distinct offset is a separate 'relation' initially.

    After fitting, RelationClusterer clusters these raw offsets by their
    (c_src, c_tgt) distributional profiles.  Two offsets are equivalent iff
    they connect the same category distributions → they are symmetric.

    The discovered clusters ARE the discovered symmetry orbits:
        - In natural images: horizontal offsets (0,1),(0,-1) cluster together
          (left/right symmetry); vertical (1,0),(-1,0) form a separate cluster
          (up/down asymmetric due to sky/ground); diagonals form a third.
        - In a rotationally symmetric dataset: all 8 directions collapse to 1.
        - In a purely horizontal dataset: only (0,±1) and all others collapse.

    This inverts Klein's Erlangen Programme:
        Standard: given symmetry group G, define geometry as G-invariant properties.
        Ours:     given distributional data, discover G from distributional invariance.

    Connection to SymmetryLens (Efe & Ozakin, arXiv:2410.05232):
        Their loss = locality + distribution-preservation.
        Our criterion = JSD between (c_src, c_tgt) distributions of offsets.
        Same principle, discrete implementation.

    Usage:
        learner = RawOffsetLearner(patch_size=16, n_atom_clusters=32,
                                   max_offset=2, codebook_size=64)
        learner.fit_images(images, verbose=True)
        # → discovered_symmetry_clusters: dict[int, list[str]]
        # Each cluster is a set of offsets that play the same structural role.

    Args:
        patch_size:      Pixel size of each patch.
        n_atom_clusters: K for atom (patch type) clustering.
        max_offset:      Maximum |dr| and |dc| to include.  max_offset=1 gives
                         8-neighbourhood; max_offset=2 gives 24-neighbourhood.
        codebook_size:   K-means codebook for patch tokenization.
        n_rel_clusters:  Target number of offset clusters (None = JSD-threshold).
        jsd_threshold:   Merge threshold when n_rel_clusters is None.
                         None (default) → auto-detect via Kneedle algorithm.
    """

    def __init__(self, patch_size: int = 16, n_atom_clusters: int = 32,
                 max_offset: int = 1, codebook_size: int = 64,
                 n_rel_clusters: int | None = None,
                 jsd_threshold: float | None = None):
        self.patch_size     = patch_size
        self.n_atom_clusters = n_atom_clusters
        self.max_offset     = max_offset
        self.codebook_size  = codebook_size
        self.n_rel_clusters = n_rel_clusters
        self.jsd_threshold  = jsd_threshold

        # Generate all (dr, dc) offsets in [-max_offset, max_offset] × [-max_offset, max_offset]
        # excluding (0, 0)
        self.all_offsets: list[tuple[int, int]] = [
            (dr, dc)
            for dr in range(-max_offset, max_offset + 1)
            for dc in range(-max_offset, max_offset + 1)
            if not (dr == 0 and dc == 0)
        ]
        self.relations: list[str] = [f'{dr:+d},{dc:+d}' for dr, dc in self.all_offsets]
        self._offset_map: dict[str, tuple[int, int]] = {
            r: off for r, off in zip(self.relations, self.all_offsets)
        }

        # Sub-learners
        self._atom_learner: RelationalLearner = RelationalLearner(
            n_clusters=n_atom_clusters)
        self._codebook: 'np.ndarray | None' = None

        # Output
        self.rel_clusterer: RelationClusterer | None = None
        self.discovered_symmetry: dict[int, list[str]] = {}  # cluster_id → [offsets]

    def _offset_name(self, dr: int, dc: int) -> str:
        return f'{dr:+d},{dc:+d}'

    def fit_images(self, images: list, verbose: bool = True) -> None:
        """Fit RawOffsetLearner on a list of images.

        Step 1: Build k-means codebook from all patches (same as Image2DRelationalLearner).
        Step 2: Tokenize all patches via nearest centroid.
        Step 3: Build (patch_a, 'dr,dc', patch_b) triples for ALL offsets in neighbourhood.
        Step 4: Fit RelationalLearner E0-E3 to get atom categories.
        Step 5: Run RelationClusterer to cluster offsets by (c_src,c_tgt) distributions.
                → discovered_symmetry = the symmetry group orbits.
        """
        import numpy as np
        n_images = len(images)
        report_every = max(1, n_images // 5)

        # Step 1: collect raw patches for codebook
        all_raw: list = []
        all_grids_raw: list = []
        for i, img in enumerate(images):
            if verbose and (i == 0 or (i + 1) % report_every == 0):
                print(f'  Extracting patches: {i+1}/{n_images}')
            arr = np.asarray(img, dtype=np.uint8)
            ps = self.patch_size
            h, w = arr.shape[:2]
            rows, cols = h // ps, w // ps
            grid_raw = []
            for r in range(rows):
                row_raw = []
                for c in range(cols):
                    patch = arr[r*ps:(r+1)*ps, c*ps:(c+1)*ps]
                    row_raw.append(patch)
                    all_raw.append(patch)
                grid_raw.append(row_raw)
            all_grids_raw.append(grid_raw)

        if verbose:
            print(f'  Raw patches: {len(all_raw):,}  '
                  f'Building codebook (k={self.codebook_size})...')
        self._codebook = _build_patch_codebook(all_raw, self.codebook_size,
                                               verbose=verbose)

        # Step 2+3: tokenize + build raw-offset triples
        all_triples: list[tuple[str, str, str]] = []
        for grid_raw in all_grids_raw:
            rows = len(grid_raw)
            cols = len(grid_raw[0]) if rows else 0
            # Tokenize via codebook
            token_grid = []
            for r in range(rows):
                row_tok = []
                for c in range(cols):
                    vec = _preprocess_patch(grid_raw[r][c])
                    dists = np.sum((self._codebook - vec) ** 2, axis=1)
                    row_tok.append(f'c{int(np.argmin(dists))}')
                token_grid.append(row_tok)
            # Build raw-offset triples
            for r in range(rows):
                for c in range(cols):
                    a = token_grid[r][c]
                    for (dr, dc), rel in zip(self.all_offsets, self.relations):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            all_triples.append((a, rel, token_grid[nr][nc]))

        if verbose:
            unique_toks = len({a for a, _, _ in all_triples} |
                              {b for _, _, b in all_triples})
            print(f'  Total triples: {len(all_triples):,}  '
                  f'Unique tokens: {unique_toks}  '
                  f'Offsets (relations): {len(self.relations)}')

        # Step 4: fit atom learner
        self._atom_learner.fit(all_triples, verbose=verbose)

        # Step 5: cluster offsets by distributional behaviour
        if verbose:
            print(f'\n  Level 5a — clustering {len(self.relations)} raw offsets '
                  f'by (c_src, c_tgt) distributions...')
        self.rel_clusterer = RelationClusterer(n_clusters=self.n_rel_clusters)
        self.rel_clusterer.fit(self._atom_learner, all_triples,
                               jsd_threshold=self.jsd_threshold,
                               verbose=verbose)
        self.discovered_symmetry = dict(self.rel_clusterer.clusters)

        if verbose:
            self._print_symmetry_summary()

    def _print_symmetry_summary(self) -> None:
        """Print a 2D grid showing which offsets belong to which cluster."""
        if not self.rel_clusterer:
            return
        asgn = self.rel_clusterer.assignment
        mo = self.max_offset
        symbols = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        print(f'\n  Discovered symmetry orbits (max_offset={mo}):')
        print(f'  {len(self.discovered_symmetry)} clusters  '
              f'(offsets with same symbol → same symmetry orbit)')
        print()

        # Print as a (2*mo+1) × (2*mo+1) grid
        for dr in range(-mo, mo + 1):
            row_str = '    '
            for dc in range(-mo, mo + 1):
                if dr == 0 and dc == 0:
                    row_str += '·  '
                else:
                    rel = self._offset_name(dr, dc)
                    cid = asgn.get(rel)
                    sym = symbols[cid % len(symbols)] if cid is not None else '?'
                    row_str += f'{sym}  '
            print(row_str)
        print()

        for cid, members in sorted(self.discovered_symmetry.items()):
            print(f'  Orbit {cid}: {members}')

    def jsd_between_offsets(self, offset1: tuple[int, int],
                            offset2: tuple[int, int]) -> float:
        """JSD between the distributional profiles of two raw offsets."""
        if self.rel_clusterer is None:
            return 1.0
        r1 = self._offset_name(*offset1)
        r2 = self._offset_name(*offset2)
        return self.rel_clusterer.jsd_between(r1, r2)


def generate_stripe_grid(n_rows: int = 16, n_cols: int = 16,
                         n_types: int = 4, horizontal: bool = True) -> 'np.ndarray':
    """Generate a grid with pure horizontal or vertical stripes.

    Horizontal stripes: row r has type r % n_types.
      - H transitions: c_src == c_tgt  (perfectly autocorrelated)
      - V transitions: c_tgt = (c_src ± 1) % n_types  (perfectly type-changing)
      → JSD(H_dist, V_dist) = 1.0

    Vertical stripes: column c has type c % n_types.
      - V transitions: c_src == c_tgt
      - H transitions: perfectly type-changing
      → JSD(H_dist, V_dist) = 1.0

    Both cases: RawOffsetLearner should separate H-family from V-family.
    """
    import numpy as np
    grid = np.zeros((n_rows, n_cols), dtype=np.int32)
    for r in range(n_rows):
        for c in range(n_cols):
            grid[r, c] = r % n_types if horizontal else c % n_types
    return grid.astype(np.uint8)


def generate_euclidean_grid(n_rows: int = 20, n_cols: int = 20,
                            n_types: int = 8, seed: int = 42) -> 'np.ndarray':
    """Generate a synthetic 2D grid with spatial structure for symmetry testing.

    Creates a grid where cell values are drawn from a Markov random field:
    - Horizontal neighbors tend to be similar (correlation parameter)
    - Vertical neighbors have different statistics (anisotropic)
    - Diagonal neighbors have intermediate statistics

    Returns a (n_rows × n_cols) uint8 array of 'patch type' values [0, n_types-1].
    This can be fed to RawOffsetLearner to test whether it discovers the
    spatial symmetry structure (H/rev_H equivalent, V/rev_V equivalent, etc.)
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    grid = np.zeros((n_rows, n_cols), dtype=np.int32)

    # Fill with spatially correlated values using simple MRF-like sampling
    # Left-to-right, top-to-bottom sequential sampling with neighbor influence
    for r in range(n_rows):
        for c in range(n_cols):
            probs = np.ones(n_types, dtype=np.float64)
            # Horizontal correlation (strong)
            if c > 0:
                probs[grid[r, c-1]] += 4.0
            # Vertical correlation (weaker, anisotropic)
            if r > 0:
                probs[grid[r-1, c]] += 2.0
            # Diagonal correlation (weak)
            if r > 0 and c > 0:
                probs[grid[r-1, c-1]] += 1.0
            probs /= probs.sum()
            grid[r, c] = rng.choice(n_types, p=probs)

    return grid.astype(np.uint8)


def generate_rotationally_symmetric_grid(n_rows: int = 20, n_cols: int = 20,
                                          n_types: int = 8,
                                          seed: int = 42) -> 'np.ndarray':
    """Rotationally symmetric grid: all 8 neighbours have the same correlation.

    If RawOffsetLearner is working correctly, it should discover that all 8
    offset directions belong to the same symmetry orbit (1 cluster for 8 offsets).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    grid = np.zeros((n_rows, n_cols), dtype=np.int32)

    neighbours = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for r in range(n_rows):
        for c in range(n_cols):
            probs = np.ones(n_types, dtype=np.float64)
            for dr, dc in neighbours:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    probs[grid[nr, nc]] += 2.0  # equal weight for all directions
            probs /= probs.sum()
            grid[r, c] = rng.choice(n_types, p=probs)

    return grid.astype(np.uint8)


def grid_to_patch_image(grid: 'np.ndarray', patch_size: int = 8) -> 'np.ndarray':
    """Convert a type-grid to a uint8 image with TEXTURED patches.

    Each cell type t → a distinct structured noise pattern (fixed per type).
    Using solid-colour patches fails because _preprocess_patch() applies local
    contrast normalization per patch: a uniform patch → zero vector → every type
    maps to the same codebook centroid.

    Fix: give each type a fixed random texture.  After LCN, the spatial PATTERN
    (which pixels are high vs. low) is preserved.  Same type → same pattern →
    same centroid.  Different types → different patterns → different centroids.

    The random seed is fixed per type, so the same type always produces the same
    texture regardless of grid position — this is the critical property.
    """
    import numpy as np
    n_types = int(grid.max()) + 1
    ps = patch_size
    # Build fixed texture per type (random noise, seeded by type)
    textures = []
    for t in range(n_types):
        rng_t = np.random.default_rng(t * 1000 + 7)
        tex = rng_t.integers(30, 226, (ps, ps), dtype=np.int32)
        textures.append(tex.astype(np.uint8))

    n_rows, n_cols = grid.shape
    img = np.zeros((n_rows * ps, n_cols * ps), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            t = int(grid[r, c]) % n_types
            img[r*ps:(r+1)*ps, c*ps:(c+1)*ps] = textures[t]
    return img


# ---------------------------------------------------------------------------
# Level 5b: Non-Euclidean Geometry Tests
# ---------------------------------------------------------------------------

def generate_tree_triples(depth: int = 4, branching: int = 3,
                           n_types: int = 4) -> list[tuple[str, str, str]]:
    """Generate (atom, relation, atom) triples from a rooted k-ary tree.

    Atom types are based on depth mod n_types.  This makes each depth level
    a distinct 'atom type', and the relations are:
        'to_child'  — moving from depth d to depth d+1
        'to_parent' — moving from depth d to depth d-1

    Distributional signatures:
        to_child:  (type_d → type_{d+1 mod n}) for all depths d  [cyclic]
        to_parent: (type_d → type_{d-1 mod n}) for all depths d  [reverse cyclic]
    JSD(to_child, to_parent) = 1.0 since no (c_src,c_tgt) pair appears in both.

    RelationClusterer should discover 2 orbits: {to_child} vs {to_parent}.
    This is the hyperbolic/tree analog of H-family vs V-family in 2D grids.

    Returns:
        List of (src_type, relation, tgt_type) triples.
    """
    triples = []
    branching_count = branching

    def recurse(cur_depth: int, count: int) -> None:
        if cur_depth >= depth:
            return
        src_type = f't{cur_depth % n_types}'
        tgt_type = f't{(cur_depth + 1) % n_types}'
        for _ in range(count):
            triples.append((src_type, 'to_child', tgt_type))
            triples.append((tgt_type, 'to_parent', src_type))
        recurse(cur_depth + 1, count * branching_count)

    recurse(0, 1)
    return triples


def generate_cycle_triples(n_nodes: int = 8, n_types: int = 4) -> list[tuple]:
    """Generate triples from a directed cycle graph.

    Atoms are types based on node_id mod n_types.  Relations:
        'forward'  — node i → node (i+1) mod n
        'backward' — node i → node (i-1) mod n

    In a symmetric cycle, forward and backward have the same distributional
    profile (same transition statistics, just reversed) → 1 orbit cluster
    if n_types divides n_nodes cleanly, else the asymmetry of the starting
    point may introduce 2 clusters.

    For RelationClusterer testing: compare with tree (asymmetric) case.
    """
    triples = []
    for i in range(n_nodes):
        src = f't{i % n_types}'
        fwd = f't{(i + 1) % n_types}'
        bwd = f't{(i - 1) % n_types}'
        triples.append((src, 'forward',  fwd))
        triples.append((src, 'backward', bwd))
    return triples


def image_to_patches(image, patch_size: int = 8) -> list[list[str]]:
    """Standalone helper: image → 2D grid of patch hash strings."""
    return Image2DRelationalLearner(patch_size=patch_size).image_to_patches(image)


def patches_to_triples(
        grid: list[list[str]],
        relations: tuple[str, ...] = ('H', 'V', 'D1', 'D2'),
) -> list[tuple[str, str, str]]:
    """Standalone helper: 2D patch grid → (a, relation, b) triples."""
    return Image2DRelationalLearner(relations=relations).patches_to_triples(grid)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== RelationalLearner smoke test ===')

    # --- Simple knowledge graph ---
    print('\n-- Knowledge graph (IS_A / HAS) --')
    triples = [
        ('dog',   'IS_A',  'animal'),
        ('cat',   'IS_A',  'animal'),
        ('eagle', 'IS_A',  'animal'),
        ('rose',  'IS_A',  'plant'),
        ('oak',   'IS_A',  'plant'),
        ('dog',   'HAS',   'legs'),
        ('cat',   'HAS',   'legs'),
        ('eagle', 'HAS',   'wings'),
        ('rose',  'HAS',   'petals'),
        ('oak',   'HAS',   'leaves'),
        ('lion',  'IS_A',  'animal'),
        ('lion',  'HAS',   'legs'),
        ('hawk',  'IS_A',  'animal'),
        ('hawk',  'HAS',   'wings'),
        ('tulip', 'IS_A',  'plant'),
        ('tulip', 'HAS',   'petals'),
    ]
    learner = RelationalLearner(n_clusters=4)
    learner.fit(triples, verbose=True)
    print(f'\nAssignment: {learner.assignment}')
    print(f'dog → IS_A: {learner.predict("dog", "IS_A")}')
    print(f'eagle → HAS: {learner.predict("eagle", "HAS")}')
    print(f'Neighbors of dog: {learner.atom_neighbors("dog", topn=3)}')

    # --- Level 2: RelationClusterer ---
    print('\n-- Level 2: RelationClusterer --')
    rc = RelationClusterer()
    rc.fit(learner, triples, verbose=True)
    print(f'Relation clusters: {rc.clusters}')
    print(f'IS_A ↔ HAS JSD: {rc.jsd_between("IS_A", "HAS"):.3f}')

    # --- Level 3: predict_chain ---
    print('\n-- Level 3: predict_chain --')
    # Build a mini KB with chains
    chain_triples = triples + [
        ('animal', 'FOUND_IN', 'nature'),
        ('plant',  'FOUND_IN', 'nature'),
    ]
    lc = RelationalLearner(n_clusters=4)
    lc.fit(chain_triples, verbose=False)
    # dog -IS_A→ animal -FOUND_IN→ nature
    result = lc.predict_chain('dog', ['IS_A', 'FOUND_IN'])
    print(f'dog -IS_A→ ? -FOUND_IN→ ? : {result}')

    # --- Level 4: SecondOrderGrammar ---
    print('\n-- Level 4: SecondOrderGrammar --')
    sg = SecondOrderGrammar()
    sg.fit(chain_triples, verbose=True)
    print(f'next_rel after IS_A: {sg.predict_next_rel("IS_A")}')

    # --- 2D image patch test ---
    print('\n-- Image2DRelationalLearner (synthetic stripes) --')
    try:
        import numpy as np
        # Create 2 distinct patch types: bright (A) and dark (B)
        # in alternating horizontal stripes
        imgs = []
        for _ in range(10):
            img = np.zeros((32, 32), dtype=np.uint8)
            img[0:8, :] = 220    # top stripe: bright
            img[8:16, :] = 30    # second: dark
            img[16:24, :] = 220  # third: bright
            img[24:32, :] = 30   # fourth: dark
            imgs.append(img)

        rl2d = Image2DRelationalLearner(patch_size=8, n_clusters=4)
        rl2d.fit_images(imgs, verbose=True)
        print(f'\nUnique clusters: {len(rl2d.clusters)}')
        print(f'Sample assignment (first 6 patches):')
        for tok, cid in list(rl2d.assignment.items())[:6]:
            print(f'  {tok} → cluster {cid}')

        # Level 2 on image learner
        print('\n-- Level 2 on image learner --')
        img_triples = []
        for grid_tok in [rl2d.image_to_patches(img) for img in imgs]:
            img_triples.extend(rl2d.patches_to_triples(grid_tok))
        img_rc = RelationClusterer()
        img_rc.fit(rl2d.learner, img_triples, verbose=True)
        print(f'Image relation clusters: {img_rc.clusters}')

        # Level 4 on image learner
        print('\n-- Level 4 on image learner --')
        img_sg = SecondOrderGrammar()
        img_sg.fit(img_triples, verbose=True)

        print('Smoke test PASSED')
    except ImportError:
        print('numpy not available — skipping image test')

    # --- Level 5a: RawOffsetLearner ---
    print('\n' + '=' * 50)
    print('Level 5a: RawOffsetLearner — Symmetry Discovery')
    print('=' * 50)
    try:
        import numpy as np

        # Test 1: Horizontal stripes — maximum H/V anisotropy
        # H transitions: same type (JSD=0 among H offsets)
        # V transitions: type changes by 1 (JSD=1 vs H)
        # Expected: H-family {(0,+1),(0,-1)} and V-family {(+1,0),(-1,0)} are separate orbits
        # Diagonals {(±1,±1)} are mixed → third orbit (or lumped with V)
        print('\n-- Horizontal stripes (H = same type, V = type changes) --')
        print('Expected: ≥2 orbit clusters  (H-family ≠ V-family)')
        h_grid = generate_stripe_grid(n_rows=16, n_cols=16, n_types=4, horizontal=True)
        h_img  = grid_to_patch_image(h_grid, patch_size=8)
        # Tile to get more training signal
        h_img_big = np.tile(h_img, (4, 4))
        raw_h = RawOffsetLearner(patch_size=8, n_atom_clusters=4,
                                 max_offset=1, codebook_size=4,
                                 jsd_threshold=0.01)
        raw_h.fit_images([h_img_big], verbose=True)

        # Test 2: Rotationally symmetric grid → all 8 offsets collapse to 1
        print('\n-- Rotationally symmetric MRF grid (all 8 directions equal) --')
        print('Expected: 1 orbit cluster (all offsets equivalent)')
        sym_grid = generate_rotationally_symmetric_grid(n_rows=40, n_cols=40,
                                                        n_types=4, seed=2)
        sym_img  = grid_to_patch_image(sym_grid, patch_size=8)
        raw_sym = RawOffsetLearner(patch_size=8, n_atom_clusters=4,
                                   max_offset=1, codebook_size=4,
                                   jsd_threshold=0.05)
        raw_sym.fit_images([sym_img], verbose=True)

        n_h   = len(raw_h.discovered_symmetry)
        n_sym = len(raw_sym.discovered_symmetry)
        print(f'\nLevel 5a result:')
        print(f'  Stripe grid:    {n_h} orbit cluster(s)  '
              f'(≥2 = H/V anisotropy discovered ✓)')
        print(f'  Symmetric grid: {n_sym} orbit cluster(s)  '
              f'(≤2 = near-symmetry discovered ✓)')
        passed = n_h >= 2
        print(f'Level 5a smoke test {"PASSED" if passed else "FAILED — check threshold"}')
    except ImportError:
        print('numpy not available — skipping Level 5a test')

    # --- Level 5b: Non-Euclidean Geometry ---
    print('\n' + '=' * 50)
    print('Level 5b: Non-Euclidean Geometry (Tree / Graph)')
    print('=' * 50)

    # Test 1: k-ary tree
    # to_child:  (type_d → type_{d+1}) — cyclic transitions
    # to_parent: (type_d → type_{d-1}) — reverse cyclic transitions
    # JSD(to_child, to_parent) = 1.0 → must be in different orbits
    print('\n-- k-ary tree (branching=3, depth=4, n_types=4) --')
    print('Expected: 2 orbit clusters (to_child ≠ to_parent)')
    tree_triples = generate_tree_triples(depth=4, branching=3, n_types=4)
    tree_learner = RelationalLearner(n_clusters=4)
    tree_learner.fit(tree_triples, verbose=False)
    tree_rc = RelationClusterer()
    tree_rc.fit(tree_learner, tree_triples, verbose=True)

    # Level 3: multi-hop chain on tree
    chain_tree = lc if False else tree_learner  # use tree_learner
    hop1 = tree_learner.predict('t0', 'to_child')
    hop2 = tree_learner.predict_chain('t0', ['to_child', 'to_child'])
    print(f'\nTree chain:  t0 -to_child→ {hop1}  '
          f'-to_child→ {hop2}  (expected: t2)')

    # Test 2: Directed cycle graph
    print('\n-- Directed cycle (n_nodes=8, n_types=4) --')
    print('Expected: 1-2 orbit clusters (forward/backward may merge if symmetric)')
    cycle_triples = generate_cycle_triples(n_nodes=8, n_types=4)
    cycle_learner = RelationalLearner(n_clusters=4)
    cycle_learner.fit(cycle_triples, verbose=False)
    cycle_rc = RelationClusterer()
    cycle_rc.fit(cycle_learner, cycle_triples, verbose=True)
    print(f'  Cycle JSD(forward, backward): {cycle_rc.jsd_between("forward","backward"):.3f}')

    # Level 4 on tree: what relation follows to_child?
    print('\n-- Level 4 on tree: second-order grammar --')
    tree_sg = SecondOrderGrammar()
    tree_sg.fit(tree_triples, verbose=True)
    print(f'  After to_child:  {tree_sg.predict_next_rel("to_child")}')
    print(f'  After to_parent: {tree_sg.predict_next_rel("to_parent")}')

    n_tree  = len(tree_rc.clusters)
    n_cycle = len(cycle_rc.clusters)
    print(f'\nLevel 5b result:')
    print(f'  Tree (asymmetric):  {n_tree} orbit cluster(s)  '
          f'(expected: 2 for to_child vs to_parent)')
    print(f'  Cycle (symmetric):  {n_cycle} orbit cluster(s)  '
          f'(expected: 1-2)')
    passed_5b = n_tree >= 2
    print(f'Level 5b smoke test {"PASSED" if passed_5b else "FAILED"}')


# ===========================================================================
# M1-M4+: Hierarchical Merge + Segment — all-scale pattern learning
#
# Broca's area principle: one algorithm (RelationalLearner), applied
# recursively via Merge (categorical composition) at every scale present
# in the data.  The discovered hierarchy converges toward the CTKG structure.
#
# Two dual operations on the PMI landscape:
#   HIGH PMI → MERGE:   tight bigrams are unit-internal → join them
#   LOW PMI  → SEGMENT: loose bigrams are boundaries → split here
#
# Together they climb the full hierarchy: chars → morphemes → words →
# collocations → phrases → clauses → sentences → paragraphs.
#
# Design principle (user): the surface string of any atom is a read-off,
# not the definition.  The definition IS the composition chain back to base
# sensory atoms.  Both MergedAtom (binary) and SegmentedAtom (n-ary) expose
# leaves() to recover that chain.  In the CTKG, the concept exists without
# a name; the string is kept only for human interpretability.
#
# Classes:
#   MergedAtom              — binary composition (Merge); left + right
#   SegmentedAtom           — n-ary sequence composition (Segment); constituents
#   AtomVocabulary          — growing vocabulary; handles both atom types
#   MergeDetector           — PMI landscape: merge candidates + boundary PMIs
#   HierarchicalRelationalLearner — iterative Merge+Segment stack (M2+M3)
#   MultiLevelContextBelief — top-down prediction (M4)
# ===========================================================================


# ---------------------------------------------------------------------------
# MergedAtom
# ---------------------------------------------------------------------------

class MergedAtom:
    """A higher-level atom created by merging two adjacent atoms.

    Preserves internal structure (left, right constituents) so that the merge
    tree is always recoverable.  Acts as a string for downstream processing —
    all RelationalLearner operations receive ``str(atom)`` so MergedAtom
    participates transparently.

    Attributes
    ----------
    surface  Canonical string key (e.g. '[a+b]').
    left     Left constituent: str (base) or MergedAtom (composite).
    right    Right constituent: str (base) or MergedAtom (composite).
    level    Merge depth: 0 = base token, N = Nth composition.
    label    Distributional label discovered by E3 at level+1 (filled in later).
    """

    __slots__ = ('surface', 'left', 'right', 'level', 'label')

    def __init__(self, surface: str, left: Any, right: Any,
                 level: int, label: str = '') -> None:
        self.surface = surface
        self.left    = left
        self.right   = right
        self.level   = level
        self.label   = label

    def __str__(self)  -> str:  return self.surface
    def __repr__(self) -> str:  return f'MergedAtom({self.surface!r}, lvl={self.level})'
    def __hash__(self) -> int:  return hash(self.surface)
    def __eq__(self, other) -> bool: return self.surface == str(other)

    def leaves(self) -> list[str]:
        """Return all base-level atoms (depth-first left-to-right)."""
        left_leaves  = (self.left.leaves()  if isinstance(self.left,  MergedAtom)
                        else [str(self.left)])
        right_leaves = (self.right.leaves() if isinstance(self.right, MergedAtom)
                        else [str(self.right)])
        return left_leaves + right_leaves


# ---------------------------------------------------------------------------
# SegmentedAtom
# ---------------------------------------------------------------------------

class SegmentedAtom:
    """A higher-level atom created by segmenting a contiguous run of atoms.

    Produced when the PMI between the last atom of the run and the first atom
    of the next run drops below a boundary threshold — the run is a unit.

    Design principle: the surface string is a read-off for human
    interpretability, NOT the definition.  The definition is the ordered
    sequence of constituents, which chains back to base sensory atoms via
    ``leaves()``.  In the CTKG this concept is represented purely by its
    ordered ``requires`` edges — no name is strictly necessary.

    Attributes
    ----------
    surface       Derived string: join of constituent surface strings.
                  (e.g. 'principio' from ['p','r','i','n','c','i','p','i','o'])
    constituents  Ordered list of atoms making up this unit.  Each element is
                  either a str (base atom), MergedAtom, or SegmentedAtom.
    level         Level at which this segmentation was performed.
    label         Distributional label from E3 clustering (filled in later).
    """

    __slots__ = ('surface', 'constituents', 'level', 'label')

    def __init__(self, constituents: list, level: int,
                 label: str = '') -> None:
        self.constituents = constituents
        self.level        = level
        self.label        = label
        # Surface derived from constituents — string is a read-off, not definition
        self.surface      = ''.join(str(c) for c in constituents)

    def __str__(self)  -> str:  return self.surface
    def __repr__(self) -> str:
        return f'SegmentedAtom({self.surface!r}, n={len(self.constituents)}, lvl={self.level})'
    def __hash__(self) -> int:  return hash(self.surface)
    def __eq__(self, other) -> bool: return self.surface == str(other)

    def leaves(self) -> list[str]:
        """Ordered base-level atoms, recovering the full composition chain."""
        result: list[str] = []
        for c in self.constituents:
            if hasattr(c, 'leaves'):
                result.extend(c.leaves())
            else:
                result.append(str(c))
        return result

    def structure(self) -> list:
        """Return constituents as a nested list for inspection."""
        parts = []
        for c in self.constituents:
            if isinstance(c, (MergedAtom, SegmentedAtom)):
                parts.append(c.structure())
            else:
                parts.append(str(c))
        return parts


# ---------------------------------------------------------------------------
# AtomVocabulary
# ---------------------------------------------------------------------------

class AtomVocabulary:
    """Dynamic vocabulary that grows as Merge operations are applied.

    Tokenization strategy: BPE-style greedy left-to-right application of
    merge rules in discovery order.  Each merge rule replaces all occurrences
    of an adjacent pair with the merged surface string in a single left-to-right
    pass.  Rules are applied in the order they were added.

    Example::

        vocab = AtomVocabulary()
        vocab.add_merge('q', 'u')   # → '[q+u]'
        vocab.tokenize(['q', 'u', 'i', 's'])
        # → ['[q+u]', 'i', 's']
    """

    def __init__(self) -> None:
        # Ordered list of (left_surface, right_surface, MergedAtom)
        self._merges: list[tuple[str, str, MergedAtom]] = []
        # Ordered list of SegmentedAtoms
        self._segments: list[SegmentedAtom] = []
        # Surface → MergedAtom | SegmentedAtom (for lookup)
        self._by_surface: dict[str, Any] = {}
        # Fast merge-pair lookup: (left_surf, right_surf) → merged_surf
        self._pair_to_surface: dict[tuple[str, str], str] = {}

    def add_merge(self, left: Any, right: Any) -> MergedAtom:
        """Create and register a new MergedAtom for the (left, right) pair."""
        l_s = str(left)
        r_s = str(right)
        surface = f'[{l_s}+{r_s}]'
        if surface in self._by_surface:
            return self._by_surface[surface]   # idempotent
        level_l = left.level  if isinstance(left,  MergedAtom) else 0
        level_r = right.level if isinstance(right, MergedAtom) else 0
        merged = MergedAtom(surface=surface, left=left, right=right,
                            level=max(level_l, level_r) + 1)
        self._merges.append((l_s, r_s, merged))
        self._by_surface[surface] = merged
        self._pair_to_surface[(l_s, r_s)] = surface
        return merged

    def tokenize(self, sequence: list) -> list[str]:
        """Apply all known merge rules to a sequence, returning merged tokens.

        Applies rules left-to-right in discovery order (BPE convention).
        Each rule is applied in a single pass over the current token list.
        """
        tokens: list[str] = [str(t) for t in sequence]
        for l_s, r_s, merged in self._merges:
            surf = merged.surface
            if l_s not in tokens:          # fast skip
                continue
            i = 0
            new_tokens: list[str] = []
            while i < len(tokens):
                if (i + 1 < len(tokens)
                        and tokens[i] == l_s
                        and tokens[i + 1] == r_s):
                    new_tokens.append(surf)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def add_segment(self, constituents: list, level: int = 1) -> 'SegmentedAtom':
        """Create and register a SegmentedAtom for the given constituents.

        The surface string is derived from the constituents (read-off only).
        The definition IS the ordered constituent list — recoverable via
        ``leaves()`` all the way back to base sensory atoms.
        """
        seg = SegmentedAtom(constituents=list(constituents), level=level)
        surface = seg.surface
        if surface in self._by_surface:
            existing = self._by_surface[surface]
            if isinstance(existing, SegmentedAtom):
                return existing
        self._by_surface[surface] = seg
        self._segments.append(seg)
        return seg

    def lookup(self, surface: str) -> 'MergedAtom | SegmentedAtom | None':
        return self._by_surface.get(surface)

    def n_merges(self) -> int:
        return len(self._merges)

    def n_segments(self) -> int:
        return len(self._segments)

    def summary(self, top_n: int = 20) -> list[str]:
        return [f'[{l}+{r}] (lvl {m.level})'
                for l, r, m in self._merges[:top_n]]


# ---------------------------------------------------------------------------
# MergeDetector  (M1)
# ---------------------------------------------------------------------------

class MergeDetector:
    """M1: PMI-based merge candidate detection from a fitted RelationalLearner.

    Pointwise Mutual Information for an adjacent pair (A, B):

        PMI(A→B) = log₂ P(B | A, rel) − log₂ P(B | rel)

    where P(B | rel) is the marginal (relation unigram) and
    P(B | A, rel) is the atom-level bigram.

    High PMI: A and B co-occur far more than chance → they form a unit.
    Low PMI:  boundary — the transition is no more predictable than random.

    This is exactly the transitional-probability signal that Saffran et al.
    showed infants use to segment the speech stream into word-like units.
    """

    def __init__(self, learner: 'RelationalLearner',
                 next_rel: str = 'next') -> None:
        self._learner  = learner
        self._next_rel = next_rel

    def detect(self,
               threshold: float | None = None,
               top_k:     int   | None = None,
               min_count: int          = 2,
               ) -> list[tuple[str, str, float]]:
        """Return ranked merge candidates as (left_atom, right_atom, pmi).

        Args
        ----
        threshold  Minimum PMI in bits.  None → include all positive-PMI pairs.
        top_k      Return at most this many.  None → return all above threshold.
        min_count  Minimum raw co-occurrence count (filters rare pairs).

        Returns
        -------
        [(left, right, pmi), ...] sorted by PMI descending.
        """
        r_s      = self._next_rel
        learner  = self._learner
        marginal = getattr(learner, '_rel_unigram',  {}).get(r_s, {})
        atom_bgs = getattr(learner, '_atom_bigrams', {})
        counts   = getattr(learner, '_atom_counts',  {})

        if not marginal or not atom_bgs:
            return []

        candidates: list[tuple[str, str, float]] = []

        for (atom, rel), dist in atom_bgs.items():
            if rel != r_s:
                continue
            raw = counts.get((atom, rel), {})
            for target, p_cond in dist.items():
                # Count filter
                cnt = raw.get(target, 0) if hasattr(raw, 'get') else 0
                if cnt < min_count:
                    continue
                p_marg = marginal.get(target, 1e-12)
                if p_marg < 1e-12:
                    continue
                pmi = math.log2(max(p_cond, 1e-12) / p_marg)
                if threshold is not None and pmi < threshold:
                    continue
                candidates.append((atom, target, pmi))

        candidates.sort(key=lambda x: -x[2])
        if top_k is not None:
            candidates = candidates[:top_k]
        return candidates

    def boundary_pmi(self, seq: list[str]) -> list[float]:
        """Return PMI at each boundary position in a sequence.

        Returns a list of length len(seq)-1: boundary_pmi[i] is the PMI
        between seq[i] and seq[i+1].  Low PMI = likely word/unit boundary.
        """
        r_s      = self._next_rel
        learner  = self._learner
        marginal = getattr(learner, '_rel_unigram',  {}).get(r_s, {})
        atom_bgs = getattr(learner, '_atom_bigrams', {})

        result = []
        for i in range(len(seq) - 1):
            a, b = str(seq[i]), str(seq[i + 1])
            dist     = atom_bgs.get((a, r_s), {})
            p_cond   = dist.get(b, 1e-12)
            p_marg   = marginal.get(b, 1e-12)
            pmi      = math.log2(max(p_cond, 1e-12) / max(p_marg, 1e-12))
            result.append(pmi)
        return result

    def segment_by_boundary(self,
                            sequence:       list[str],
                            threshold:      float | None    = None,
                            level:          int             = 1,
                            boundary_atoms: set[str] | None = None,
                            ) -> list:
        """Split a sequence into SegmentedAtoms at low-PMI or explicit boundaries.

        Two complementary boundary mechanisms:

        1. **Explicit boundary atoms** (``boundary_atoms`` parameter):
           Tokens that unconditionally create a segment break.  For written
           text, pass ``boundary_atoms={' '}`` — the space character is an
           unambiguous word delimiter.  Each boundary atom forms its own
           single-element output; PMI segmentation is applied recursively
           within the chunks between them.

        2. **PMI-based boundaries** (``threshold`` parameter):
           Position i is a boundary if PMI(seq[i], seq[i+1]) < threshold.
           If threshold is None (default), the mean PMI of the chunk is used.

        Rationale: at the character level, written Latin space-boundary PMI
        (mean≈0.48) and intra-word PMI (mean≈0.68) are too close for a single
        threshold to cleanly separate them.  Explicit ``boundary_atoms={' '}``
        handles level 0; pure PMI handles higher levels where no delimiter exists.

        Single-atom outputs are returned as plain strings (no SegmentedAtom wrap).
        Multi-atom segments become SegmentedAtom, preserving the full composition
        chain via ``leaves()``.

        Parameters
        ----------
        sequence        Atom strings at the current level.
        threshold       PMI cutoff.  None = adaptive mean of each chunk.
        level           Level label assigned to resulting SegmentedAtoms.
        boundary_atoms  Tokens that always create a segment break.
                        Each such token forms its own single-element output.
                        For text data use ``{' '}``.
        """
        if len(sequence) < 2:
            return list(sequence)

        # --- Stage 1: split at explicit boundary atoms ---
        if boundary_atoms and any(t in boundary_atoms for t in sequence):
            chunks: list[list[str]] = []
            current: list[str] = []
            for t in sequence:
                if t in boundary_atoms:
                    if current:
                        chunks.append(current)
                        current = []
                    chunks.append([t])       # boundary atom = its own chunk
                else:
                    current.append(t)
            if current:
                chunks.append(current)

            # Apply PMI segmentation within each non-boundary chunk, then flatten
            result: list = []
            for chunk in chunks:
                if len(chunk) == 1:
                    result.append(chunk[0])
                else:
                    result.extend(self.segment_by_boundary(
                        chunk, threshold=threshold, level=level,
                        boundary_atoms=None,   # already split; avoid double-pass
                    ))
            return result

        # --- Stage 2: PMI-based boundary detection ---
        pmis = self.boundary_pmi(sequence)

        if threshold is None:
            threshold = sum(pmis) / len(pmis) if pmis else 0.0

        segments: list = []
        start = 0
        for i, p in enumerate(pmis):
            if p < threshold:
                # Boundary between position i and i+1 — close current segment
                chunk = sequence[start: i + 1]
                if len(chunk) > 1:
                    segments.append(SegmentedAtom(list(chunk), level=level))
                elif chunk:
                    segments.append(chunk[0])
                start = i + 1

        # Remaining atoms after last boundary
        if start < len(sequence):
            chunk = sequence[start:]
            if len(chunk) > 1:
                segments.append(SegmentedAtom(list(chunk), level=level))
            elif chunk:
                segments.append(chunk[0])

        return segments


# ---------------------------------------------------------------------------
# Helper: sequences → triples
# ---------------------------------------------------------------------------

def _sequences_to_next_triples(sequences: list[list],
                                next_rel: str = 'next',
                                ) -> list[tuple[str, str, str]]:
    """Convert sequences to (atom_i, next_rel, atom_{i+1}) triples."""
    triples: list[tuple[str, str, str]] = []
    for seq in sequences:
        for i in range(len(seq) - 1):
            triples.append((str(seq[i]), next_rel, str(seq[i + 1])))
    return triples


# ---------------------------------------------------------------------------
# HierarchicalRelationalLearner  (M2 + M3)
# ---------------------------------------------------------------------------

class HierarchicalRelationalLearner:
    """M2/M3: Multi-level RelationalLearner with iterative Merge.

    Implements Broca's area / categorical composition:

        Level 0: base atoms  (characters, words, patches, …)
        Level N: merged atoms discovered from level N-1 statistics

    At each level:
      1. Build (atom_i, 'next', atom_{i+1}) triples from current sequences.
      2. Fit a RelationalLearner — learns P(target | atom, rel) at this scale.
      3. Run MergeDetector — find high-PMI adjacent pairs.
      4. Add those pairs to AtomVocabulary and re-tokenize all sequences.
      5. Repeat with merged sequences at the next level.

    All levels remain active.  ``predict_dist(atom, rel, level=N)`` queries
    level N.  ``infer_chain`` can chain across a single level or be called
    at any level independently.

    The discovered type structure at each level feeds into CTKG grounding (M6).

    Parameters
    ----------
    n_clusters             K for each RelationalLearner (None = auto-detect).
    pmi_threshold          Minimum PMI (bits) to accept a merge candidate.
    max_merges_per_level   Hard cap on merges per level (avoids over-merging).
    max_levels             Maximum recursion depth.
    next_rel               Name of the sequential relation (default 'next').
    min_merge_count        Minimum co-occurrence count for merge candidates.
    use_segment            If True (default), after each Merge step, segment
                           the merged sequences at low-PMI boundaries to form
                           the next level's atoms.  This allows the hierarchy
                           to climb all the way from characters to words to
                           phrases to clauses in a single fit() call.
    boundary_threshold     PMI cutoff for segmentation.  None (default) =
                           adaptive: use the mean PMI of each sequence so
                           boundaries are below-average-predictability joins.
    boundary_atoms         Tokens that always create an unconditional segment
                           break, regardless of PMI.  For written text, pass
                           ``{' '}`` so the space character is always a word
                           delimiter.  Only applied at the first level where
                           the atoms are base characters; at higher levels
                           (word tokens, phrase tokens) boundaries are PMI-only.
    """

    def __init__(self,
                 n_clusters:           int | None       = None,
                 pmi_threshold:        float            = 0.5,
                 max_merges_per_level: int              = 50,
                 max_levels:           int              = 6,
                 next_rel:             str              = 'next',
                 min_merge_count:      int              = 3,
                 use_segment:          bool             = True,
                 boundary_threshold:   float | None     = None,
                 boundary_atoms:       set[str] | None  = None,
                 ) -> None:
        self.n_clusters           = n_clusters
        self.pmi_threshold        = pmi_threshold
        self.max_merges_per_level = max_merges_per_level
        self.max_levels           = max_levels
        self.next_rel             = next_rel
        self.min_merge_count      = min_merge_count
        self.use_segment          = use_segment
        self.boundary_threshold   = boundary_threshold
        self.boundary_atoms       = boundary_atoms

        self.levels:   list[RelationalLearner]  = []
        self.vocab:    AtomVocabulary           = AtomVocabulary()
        # Sequences as seen at each level (for diagnostics / re-use)
        self._seqs_at: list[list[list[str]]]   = []

    # ------------------------------------------------------------------
    # Fit

    def fit(self, sequences: list[list],
            verbose: bool = True) -> None:
        """Fit hierarchically from base sequences.

        Args
        ----
        sequences  List of token sequences.  Each token can be any str-able.
        verbose    Print level-by-level progress.
        """
        current_seqs: list[list[str]] = [
            [str(t) for t in seq] for seq in sequences
        ]

        for level in range(self.max_levels):
            n_tok   = sum(len(s) for s in current_seqs)
            vocab_s = {t for s in current_seqs for t in s}
            if verbose:
                print(f'\n  [Merge L{level}] '
                      f'{len(vocab_s)} unique atoms, {n_tok:,} tokens')

            if not vocab_s or n_tok < 2:
                if verbose:
                    print(f'  [Merge L{level}] Too few tokens — stopping.')
                break

            # Step 1: build triples
            triples = _sequences_to_next_triples(current_seqs, self.next_rel)
            if not triples:
                if verbose:
                    print(f'  [Merge L{level}] No triples — stopping.')
                break

            # Step 2: fit RelationalLearner
            learner = RelationalLearner(n_clusters=self.n_clusters)
            learner.fit(triples, verbose=verbose)
            self.levels.append(learner)
            self._seqs_at.append(current_seqs)

            # Step 3: detect merge candidates
            detector   = MergeDetector(learner, next_rel=self.next_rel)
            candidates = detector.detect(
                threshold=self.pmi_threshold,
                top_k=self.max_merges_per_level,
                min_count=self.min_merge_count,
            )

            if not candidates:
                if verbose:
                    print(f'  [Merge L{level}] No candidates above '
                          f'PMI={self.pmi_threshold:.2f} — stopping.')
                break

            if verbose:
                top5 = [(l, r, f'{p:.2f}') for l, r, p in candidates[:5]]
                print(f'  [Merge L{level}] {len(candidates)} candidates. '
                      f'Top-5: {top5}')

            # Step 4: register merges and re-tokenize
            for left, right, _pmi in candidates:
                self.vocab.add_merge(left, right)

            new_seqs = [self.vocab.tokenize(seq) for seq in current_seqs]

            # Step 5 (optional): Segment at low-PMI boundaries.
            # This is the dual of Merge: Merge joins high-PMI adjacent pairs
            # (sub-word units), Segment splits at low-PMI positions (word/
            # phrase boundaries).  Together they climb the full hierarchy:
            #   chars → [Merge] → morpheme tokens → [Segment] → words
            #   words → [Merge] → word-bigram tokens → [Segment] → phrases
            if self.use_segment:
                seg_detector  = MergeDetector(learner, next_rel=self.next_rel)
                segmented: list[list[str]] = []
                for seq in new_seqs:
                    # boundary_atoms only used at level 0 (char→word);
                    # at higher levels boundaries are discovered from PMI.
                    b_atoms = self.boundary_atoms if level == 0 else None
                    seg_atoms = seg_detector.segment_by_boundary(
                        seq,
                        threshold=self.boundary_threshold,
                        level=level + 1,
                        boundary_atoms=b_atoms,
                    )
                    # Register SegmentedAtoms in vocabulary
                    for sa in seg_atoms:
                        if isinstance(sa, SegmentedAtom):
                            self.vocab.add_segment(sa.constituents,
                                                   level=level + 1)
                    segmented.append([str(a) for a in seg_atoms])

                new_seqs = segmented
                if verbose:
                    n_seg   = self.vocab.n_segments()
                    example = new_seqs[0][:8] if new_seqs else []
                    print(f'  [Seg   L{level}] {n_seg} segments registered. '
                          f'First tokens: {example}')

            # Step 6: check for progress
            new_vocab = {t for s in new_seqs for t in s}
            if new_vocab == vocab_s:
                if verbose:
                    print(f'  [Merge L{level}] Vocabulary unchanged — stopping.')
                break

            current_seqs = new_seqs

        if verbose:
            print(f'\n  HierarchicalRelationalLearner ready: '
                  f'{len(self.levels)} levels, {self.vocab.n_merges()} merges, '
                  f'{self.vocab.n_segments()} segments.')
            if self.vocab.n_merges():
                print(f'  Top merges: {self.vocab.summary(10)}')

    # ------------------------------------------------------------------
    # Prediction

    def predict_dist(self, atom: Any, relation: str,
                     level: int = 0) -> dict[str, float]:
        """P(target | atom, relation) at the requested level."""
        if level >= len(self.levels):
            return {}
        return self.levels[level].predict_dist(atom, relation)

    def infer_chain(self, atom: Any, relations: list[str],
                    level: int = 0, topk: int = 5,
                    ) -> list[tuple[str, float]]:
        """Multi-hop inference at the requested level."""
        if level >= len(self.levels):
            return []
        return self.levels[level].infer_chain(atom, relations, topk=topk)

    # ------------------------------------------------------------------
    # Merge-level query helpers

    def merged_form(self, base_atom: str) -> str:
        """Return the highest-level surface form that contains base_atom."""
        tokens = self.vocab.tokenize([base_atom])
        return tokens[0] if tokens else base_atom

    def boundary_pmis(self, sequence: list[str]) -> list[float]:
        """PMI at each boundary in the raw (level-0) sequence.

        Wraps MergeDetector.boundary_pmi() on the level-0 learner.
        Low values mark likely unit boundaries.
        """
        if not self.levels:
            return []
        return MergeDetector(self.levels[0], self.next_rel).boundary_pmi(sequence)

    def level_summary(self) -> None:
        """Print a summary of vocabulary and merge structure at each level."""
        for i, (learner, seqs) in enumerate(
                zip(self.levels, self._seqs_at)):
            vocab = {t for s in seqs for t in s}
            print(f'  Level {i}: {len(vocab)} unique atoms, '
                  f'{sum(len(s) for s in seqs):,} tokens, '
                  f'K={learner._K} clusters')


# ---------------------------------------------------------------------------
# MultiLevelContextBelief  (M4)
# ---------------------------------------------------------------------------

class MultiLevelContextBelief:
    """M4: Top-down context prediction via stacked ContextBeliefStates.

    Maintains one ContextBeliefState per fitted level.  Observations update
    all levels simultaneously (using the merge vocabulary to map base atoms
    to their merged forms).  Higher-level beliefs constrain lower-level
    predictions via a weighted blend.

    Top-down blend at level 0::

        P_conditioned(t | rel) =
            (1 − α) · P_level0(t | belief, rel)
          +       α · P_level1_projected(t | belief, rel)

    The level-1 prediction is marginalised back to level-0 atoms via the
    constituent structure of each MergedAtom (``leaves()``).

    Parameters
    ----------
    hier_learner      A fitted HierarchicalRelationalLearner.
    top_down_weight   α — blending weight for upper-level signal (0 = no blend).
    decay_rate        Per-step belief decay toward uniform (passed to each CBS).
    obs_sharpness     Bayesian update sharpness (passed to each CBS).
    """

    def __init__(self,
                 hier_learner:    HierarchicalRelationalLearner,
                 top_down_weight: float = 0.3,
                 decay_rate:      float = 0.90,
                 obs_sharpness:   float = 0.97,
                 ) -> None:
        self._hier  = hier_learner
        self._alpha = top_down_weight
        self._beliefs: list[ContextBeliefState] = [
            ContextBeliefState(learner,
                               decay_rate=decay_rate,
                               obs_sharpness=obs_sharpness)
            for learner in hier_learner.levels
        ]

    # ------------------------------------------------------------------
    # Update

    def observe(self, atom: Any) -> None:
        """Update all levels given a new base atom.

        Level 0 observes the raw atom.
        Level N observes the merged form of the atom at that level.
        """
        a_s = str(atom)
        if not self._beliefs:
            return
        self._beliefs[0].observe(a_s)

        if len(self._beliefs) > 1:
            tokens = self._hier.vocab.tokenize([a_s])
            merged = tokens[0] if tokens else a_s
            for lvl in range(1, len(self._beliefs)):
                # At level N, observe the merged surface that was active
                self._beliefs[lvl].observe(merged)

    def transition(self, relation: str) -> None:
        """Propagate all level beliefs through their transition matrices."""
        for belief in self._beliefs:
            belief.transition(relation)

    def decay(self) -> None:
        """Decay all levels toward uniform prior."""
        for belief in self._beliefs:
            belief.decay()

    # ------------------------------------------------------------------
    # Prediction

    def predict_dist_conditioned(self, relation: str,
                                 level: int = 0,
                                 ) -> dict[str, float]:
        """Top-down conditioned P(target | belief, relation) at given level.

        Blends level-N prediction with a projection from level N+1.
        The level N+1 prediction is distributed uniformly over the constituent
        base atoms of each MergedAtom (structural top-down signal).
        """
        if level >= len(self._beliefs):
            return {}

        base_dist = self._beliefs[level].predict_target_dist(relation)
        alpha = self._alpha

        if alpha == 0.0 or level + 1 >= len(self._beliefs):
            return base_dist

        upper_dist_merged = self._beliefs[level + 1].predict_target_dist(relation)
        if not upper_dist_merged:
            return base_dist

        # Project merged predictions down to base atoms via constituent leaves
        projected: dict[str, float] = {}
        for merged_surf, p in upper_dist_merged.items():
            ma = self._hier.vocab.lookup(merged_surf)
            if ma is not None:
                leaves = ma.leaves()
                share  = p / max(len(leaves), 1)
                for leaf in leaves:
                    projected[leaf] = projected.get(leaf, 0.0) + share
            else:
                # Surface is a base atom at the upper level
                projected[merged_surf] = projected.get(merged_surf, 0.0) + p

        if not projected:
            return base_dist

        # Normalise
        total = sum(projected.values())
        if total > 0:
            projected = {k: v / total for k, v in projected.items()}

        # Blend
        all_keys = set(base_dist) | set(projected)
        combined = {
            k: (1.0 - alpha) * base_dist.get(k, 0.0)
               +       alpha  * projected.get(k, 0.0)
            for k in all_keys
        }
        total2 = sum(combined.values())
        if total2 > 0:
            return {k: v / total2 for k, v in combined.items()}
        return base_dist

    # ------------------------------------------------------------------
    # Diagnostics

    def entropy(self, level: int = 0) -> float:
        """Shannon entropy of the belief at the given level (bits)."""
        if level >= len(self._beliefs):
            return 0.0
        return self._beliefs[level].entropy()

    def reset(self) -> None:
        """Reset all beliefs to uniform prior."""
        for b in self._beliefs:
            b.reset()

    def __repr__(self) -> str:
        levels_str = ', '.join(
            f'L{i}:{b.entropy():.1f}b' for i, b in enumerate(self._beliefs)
        )
        return f'MultiLevelContextBelief({levels_str})'


# ---------------------------------------------------------------------------
# M5: Structural relation extraction
# ---------------------------------------------------------------------------

def extract_structural_relations(
        hier: HierarchicalRelationalLearner,
) -> dict[str, list[tuple[str, str, str]]]:
    """M5: Extract constituency and distributional relations from a fitted
    HierarchicalRelationalLearner.

    Two kinds of relations are returned:

    1. **Constituency** (exact, from merge tree):
       (merged_surface, 'left_of',  left_surface)
       (merged_surface, 'right_of', right_surface)
       (left_surface,  'part_of',   merged_surface)
       (right_surface, 'part_of',   merged_surface)

       These are always available and form the hierarchical parse tree.

    2. **Distributional** (from E3 clusters at each level):
       (atom_surface, 'has_type_L{N}', cluster_label)

       Where cluster_label = 'type_{cluster_id}' from the RelationalLearner
       at level N.  Atoms in the same cluster share a distributional category
       — these are the CTKG types discovered from data.

    Returns
    -------
    dict with keys:
        'constituency'    — [(subject, relation, object), ...]
        'distributional'  — [(atom, 'has_type_LN', type_label), ...]
    """
    constituency:   list[tuple[str, str, str]] = []
    distributional: list[tuple[str, str, str]] = []

    # --- Constituency relations from merge history ---
    for l_s, r_s, merged in hier.vocab._merges:
        m_s = merged.surface
        constituency.append((m_s,  'left_of',  l_s))
        constituency.append((m_s,  'right_of', r_s))
        constituency.append((l_s,  'part_of',  m_s))
        constituency.append((r_s,  'part_of',  m_s))

    # --- Distributional type labels from E3 clusters at each level ---
    for lvl, (learner, seqs) in enumerate(
            zip(hier.levels, hier._seqs_at)):
        assignment = getattr(learner, 'assignment', {})
        if not assignment:
            continue
        label_prefix = f'type_L{lvl}'
        for atom_s, cluster_id in assignment.items():
            type_label = f'{label_prefix}_{cluster_id}'
            distributional.append((atom_s, f'has_type_L{lvl}', type_label))

    return {
        'constituency':   constituency,
        'distributional': distributional,
    }


def label_merged_atoms(hier: HierarchicalRelationalLearner) -> None:
    """M5: Fill in the ``label`` field of each MergedAtom using the E3 cluster
    assignment at the level where the atom first appears as a token.

    After this call, ``MergedAtom.label`` holds e.g. 'type_L1_2' (level 1,
    cluster 2), which is the distributional category discovered at that level.
    This label IS the CTKG type for this merge unit.
    """
    for l_s, r_s, merged in hier.vocab._merges:
        # The merged atom is a token at level merged.level
        target_level = merged.level
        if target_level >= len(hier.levels):
            continue
        learner    = hier.levels[target_level]
        assignment = getattr(learner, 'assignment', {})
        cluster_id = assignment.get(merged.surface)
        if cluster_id is not None:
            merged.label = f'type_L{target_level}_{cluster_id}'


# ---------------------------------------------------------------------------
# M6: CTKG grounding — export discovered structure as .ctkg
# ---------------------------------------------------------------------------

def export_ctkg(
        hier:        HierarchicalRelationalLearner,
        domain_name: str  = 'discovered',
        out_path:    str | None = None,
) -> str:
    """M6: Export the structure discovered by HierarchicalRelationalLearner
    as a CTKG domain file (.ctkg DSL format).

    The export encodes:

    - **Types**: one per distributional cluster per level.
      Level-0 clusters → ``symbol`` base type.
      Level-N clusters (N > 0) → ``seq(type_L{N-1}_*)`` (sequence of lower type).

    - **Concepts**: one per MergedAtom (named merge unit).
      Each concept has a ``requires`` edge to its left and right constituent
      concepts (categorical composition = prerequisite structure).

    - **Interface**: exports all type names for cross-domain use.

    This file is the bridge between statistical learning (RelationalLearner)
    and the formal cognitive architecture (CTKG).  A hand-authored domain file
    for the same domain should overlap with this generated one — the degree of
    overlap is the validation metric for M6.

    Parameters
    ----------
    hier         A fitted HierarchicalRelationalLearner (call label_merged_atoms
                 first to populate MergedAtom.label fields).
    domain_name  Name for the generated domain block.
    out_path     If given, write to this file.  Otherwise return the string.

    Returns
    -------
    The .ctkg file content as a string.
    """
    label_merged_atoms(hier)
    lines: list[str] = []

    lines.append(f'# Auto-generated CTKG domain: {domain_name}')
    lines.append(f'# Source: HierarchicalRelationalLearner '
                 f'({len(hier.levels)} levels, {hier.vocab.n_merges()} merges)')
    lines.append('')

    # --- Collect all type labels ---
    type_labels: set[str] = set()
    for lvl, learner in enumerate(hier.levels):
        assignment = getattr(learner, 'assignment', {})
        for cluster_id in set(assignment.values()):
            type_labels.add(f'type_L{lvl}_{cluster_id}')

    # --- Type blocks ---
    lines.append('# Types discovered by E3 clustering at each level')
    for label in sorted(type_labels):
        lvl_str = label.split('_')[1]   # 'L0', 'L1', …
        lvl     = int(lvl_str[1:])
        if lvl == 0:
            lines.append(f'type {label} = symbol')
        else:
            lines.append(f'type {label} = seq(type_L{lvl - 1}_*)')
    lines.append('')

    # --- Concept blocks for MergedAtoms and SegmentedAtoms ---
    lines.append('# Concepts: discovered compositional units')
    lines.append('#   MergedAtom  → left_constituent + right_constituent (binary Merge)')
    lines.append('#   SegmentedAtom → position_0, position_1, … (ordered Segment)')
    lines.append('#   In the CTKG, the surface string is a read-off only.')
    lines.append('#   The definition IS the ordered requires chain back to base atoms.')
    seen_concepts: set[str] = set()

    def _safe_name(s: str) -> str:
        """Convert a surface string to a valid CTKG identifier."""
        return (s.replace('[', 'M').replace(']', '').replace('+', '_')
                 .replace(' ', '_SP_').replace("'", '_AP_'))

    # MergedAtoms: binary composition (left_constituent + right_constituent)
    for l_s, r_s, merged in hier.vocab._merges:
        concept_name = _safe_name(merged.surface)
        left_name    = _safe_name(l_s)
        right_name   = _safe_name(r_s)
        type_label   = merged.label or f'type_L{merged.level}_unknown'

        if concept_name in seen_concepts:
            continue
        seen_concepts.add(concept_name)

        lines.append(f'concept {concept_name}')
        lines.append(f'    type {type_label}')
        lines.append(f'    requires {left_name} via "left_constituent"')
        lines.append(f'    requires {right_name} via "right_constituent"')
        lines.append('')

    # SegmentedAtoms: ordered sequence composition (position_0, position_1, …)
    # The ordered requires edges ARE the definition — no name is strictly needed.
    lines.append('# Segmented units: n-ary ordered composition')
    for seg in hier.vocab._segments:
        concept_name = _safe_name(seg.surface)
        type_label   = seg.label or f'type_L{seg.level}_unknown'

        if concept_name in seen_concepts:
            continue
        seen_concepts.add(concept_name)

        lines.append(f'concept {concept_name}')
        lines.append(f'    type {type_label}')
        for pos, constituent in enumerate(seg.constituents):
            part_name = _safe_name(str(constituent))
            lines.append(f'    requires {part_name} via "position_{pos}"')
        lines.append('')

    # --- Interface block ---
    lines.append('interface')
    for label in sorted(type_labels):
        lines.append(f'    exports type {label}')
    lines.append('')

    content = '\n'.join(lines)

    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)

    return content
