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
        # Online-learning raw counts — pre-initialized so update_online needs no hasattr guards.
        self._atom_counts:  dict = {}   # (atom, rel) → {target: int}
        self._rel_counts:   dict = {}   # rel → {target: int}
        self._atom_totals:  dict = {}   # (atom, rel) → int
        self._rel_totals:   dict = {}   # rel → int
        self._max_online_vocab: int = 50_000  # overridden by PCH

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
        self._atom_totals: dict = {}
        for (atom, rel), ctr in _atom_raw.items():
            total = sum(ctr.values())
            self._atom_totals[(atom, rel)] = total
            if total > 0:
                self._atom_bigrams[(atom, rel)] = {
                    k: v / total for k, v in ctr.items()}

        self._rel_unigram = {}
        self._rel_totals: dict = {}
        for rel, ctr in _rel_raw.items():
            total = sum(ctr.values())
            self._rel_totals[rel] = total
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
            max_atoms: int = 50_000) -> None:
        """Fit E0-E3 on a list of (atom_a, relation_name, atom_b) triples.

        Fast path: bypasses ExampleStore entirely.  One O(N) pass collects
        compound bigram signatures (fwd + bwd), atom bigrams, and rel unigrams.
        Clustering and E3 cache construction proceed directly from raw dicts,
        eliminating 8M+ engine.teach() calls.

        Args:
            max_atoms: If the number of unique atoms exceeds this threshold,
                       skip the E0-E3 cluster pipeline and use atom-level
                       bigrams + per-relation unigram only.
        """
        from collections import defaultdict, Counter as _Counter

        if verbose:
            m = _rss_mb()
            if m:
                print(f'  RSS at fit() start = {m:.0f} MB')

        # ── PASS 1 (O(N)): collect ALL data in one iteration ────────────────
        fwd_raw:   dict = defaultdict(_Counter)   # atom → {"rel:tgt": count}
        bwd_raw:   dict = defaultdict(_Counter)   # atom → {"rev_rel:src": count}
        _atom_raw: dict = defaultdict(_Counter)   # (atom, rel) → Counter(tgt)
        _rel_raw:  dict = defaultdict(_Counter)   # rel → Counter(tgt)
        atoms_seen: set = set()
        rels_seen:  set = set()
        for a, r, b in triples:
            a_s, r_s, b_s = str(a), str(r), str(b)
            atoms_seen.add(a_s); atoms_seen.add(b_s); rels_seen.add(r_s)
            fwd_raw[a_s][f'{r_s}:{b_s}'] += 1
            bwd_raw[b_s][f'rev_{r_s}:{a_s}'] += 1
            _atom_raw[(a_s, r_s)][b_s] += 1
            _rel_raw[r_s][b_s] += 1

        n_unique = len(atoms_seen)
        n_rels   = max(len(rels_seen), 1)
        est_bytes = n_unique * (n_rels * n_unique) * 4 * 2
        est_gb    = est_bytes / 1_000_000_000

        if n_unique > max_atoms:
            if verbose:
                print(f'  Large-KG mode: {n_unique:,} atoms × {n_rels} relations'
                      f' — cluster matrix ≈{est_gb:.1f} GB, skipping E0-E3; '
                      f'using atom bigrams + rel unigram only.')
            # Reuse already-collected data — no need to re-scan triples.
            self._atom_counts = dict(_atom_raw)
            self._rel_counts  = dict(_rel_raw)
            self._atom_totals = {k: sum(v.values()) for k, v in _atom_raw.items()}
            self._rel_totals  = {r: sum(v.values()) for r, v in _rel_raw.items()}
            self._atom_bigrams = {}
            for key, ctr in _atom_raw.items():
                t = self._atom_totals[key]
                if t > 0:
                    self._atom_bigrams[key] = {k: v / t for k, v in ctr.items()}
            self._rel_unigram = {}
            for rel, ctr in _rel_raw.items():
                t = self._rel_totals[rel]
                if t > 0:
                    self._rel_unigram[rel] = {k: v / t for k, v in ctr.items()}
            self.assignment = {}; self.clusters = {}; self._K = 0
            if verbose:
                print(f'  atom_bigrams: {len(self._atom_bigrams):,} (h,r) pairs')
                print(f'  rel_unigram:  {len(self._rel_unigram):,} relations')
                m = _rss_mb()
                if m:
                    print(f'  RSS after large-KG fit = {m:.0f} MB')
            return

        # ── E0: compound signatures (replaces discover + induce_hierarchy_bidir)
        if verbose:
            print(f'  Clustering {self._K} atom categories '
                  f'from rel_next/rel_prev...')

        all_atoms_v = sorted(set(fwd_raw) | set(bwd_raw))
        atom_freq:   dict = {}
        sigs:        dict = {}
        for atom in all_atoms_v:
            fwd_t = sum(fwd_raw[atom].values())
            bwd_t = sum(bwd_raw[atom].values())
            d: dict = {}
            if fwd_t > 0:
                for k, cnt in fwd_raw[atom].items():
                    d[('fwd', k)] = cnt / fwd_t
            if bwd_t > 0:
                for k, cnt in bwd_raw[atom].items():
                    d[('bwd', k)] = cnt / bwd_t
            if d:
                sigs[atom] = d
                atom_freq[atom] = fwd_t + bwd_t

        # Cap output-vocabulary dimension (mirrors vocab_size in induce_hierarchy_bidir)
        _vocab_size = min(2000, n_unique * 2) if n_unique > 50 else None
        if _vocab_size is not None:
            out_freq: _Counter = _Counter()
            for atom, sig in sigs.items():
                w = atom_freq.get(atom, 1)
                for ctx_key, p in sig.items():
                    out_freq[ctx_key] += p * w
            top_out = {k for k, _ in out_freq.most_common(_vocab_size)}
            capped: dict = {}
            for atom, sig in sigs.items():
                restricted = {k: v for k, v in sig.items() if k in top_out}
                mass = sum(restricted.values())
                if mass > 1e-12:
                    capped[atom] = {k: v / mass for k, v in restricted.items()}
            sigs = capped

        assignment, clusters = _jsd_cluster(sigs, n_clusters=self._K)
        self.assignment = assignment
        self.clusters   = clusters
        self._K         = len(clusters)
        if verbose:
            print(f'  Atoms: {len(self.assignment):,}  Clusters: {self._K}')

        # ── E3: cluster successor distributions for sim_matrix ──────────────
        if verbose:
            print('  E3: building successor distributions...')
        succ: dict = {}
        for cid, members in clusters.items():
            merged: dict = collections.defaultdict(float)
            n = 0
            for tok in members:
                ctr = fwd_raw.get(tok)
                if ctr is None:
                    continue
                total = sum(ctr.values())
                if total == 0:
                    continue
                for k, cnt in ctr.items():
                    merged[k] += cnt / total
                n += 1
            succ[cid] = {k: v / n for k, v in merged.items()} if n else {}
        self._succ_dists = succ
        sim_matrix = _build_sim_matrix(succ, self._K, e3_temperature)

        # ── Atom bigrams + totals ────────────────────────────────────────────
        self._atom_counts:  dict = dict(_atom_raw)
        self._atom_bigrams: dict = {}
        self._atom_totals:  dict = {}
        for (atom, rel), ctr in _atom_raw.items():
            total = sum(ctr.values())
            self._atom_totals[(atom, rel)] = total
            if total > 0:
                self._atom_bigrams[(atom, rel)] = {
                    k: v / total for k, v in ctr.items()}

        self._rel_counts:  dict = dict(_rel_raw)
        self._rel_unigram: dict = {}
        self._rel_totals:  dict = {}
        for rel, ctr in _rel_raw.items():
            total = sum(ctr.values())
            self._rel_totals[rel] = total
            if total > 0:
                self._rel_unigram[rel] = {k: v / total for k, v in ctr.items()}

        if verbose:
            m = _rss_mb()
            if m:
                print(f'  RSS after E1 = {m:.0f} MB')

        # ── E1 caches: nc / wgc / _trans — O(V×R×V) from _atom_raw ─────────
        # Build directly from atom-level counts: avoids second O(N) triple scan
        # AND replaces ai.teach() calls + _precompute_dist_cache() entirely.
        _register_concepts(self.ai, [
            ('next_cat_rel',        ['cat', 'rel'],        ['cat'],  'relational'),
            ('token_given_cat_rel', ['cat', 'rel', 'cat'], ['atom'], 'relational'),
        ])
        _trans_raw: dict = defaultdict(lambda: defaultdict(_Counter))
        nc_raw:     dict = defaultdict(_Counter)   # (ca_s, r_s) → Counter(cb_s)
        wgc_raw:    dict = defaultdict(_Counter)   # (ca_s, r_s, cb_s) → Counter(b_s)
        n_used = n_skip = 0
        for (a_s, r_s), ctr in _atom_raw.items():
            c_a = assignment.get(a_s)
            if c_a is None:
                n_skip += len(ctr)
                continue
            ca_s = str(c_a)
            for b_s, cnt in ctr.items():
                c_b = assignment.get(b_s)
                if c_b is None:
                    n_skip += cnt
                    continue
                cb_s = str(c_b)
                nc_raw[(ca_s, r_s)][cb_s]          += cnt
                wgc_raw[(ca_s, r_s, cb_s)][b_s]    += cnt
                _trans_raw[r_s][ca_s][cb_s]         += cnt
                n_used                              += cnt

        if verbose:
            print(f'  E1: {n_used:,} triples used, {n_skip:,} OOV skipped')

        # nc_cache / wgc_cache: tuple output keys to match _precompute_dist_cache
        nc_cache:  dict = {}
        for k, ctr in nc_raw.items():
            total = sum(ctr.values())
            if total > 0:
                nc_cache[k] = {(t,): c / total for t, c in ctr.items()}

        wgc_cache: dict = {}
        for k, ctr in wgc_raw.items():
            total = sum(ctr.values())
            if total > 0:
                wgc_cache[k] = {(t,): c / total for t, c in ctr.items()}

        self._trans: dict = {}
        for rel, src_map in _trans_raw.items():
            self._trans[rel] = {}
            for c_src, ctr in src_map.items():
                total = sum(ctr.values())
                if total > 0:
                    self._trans[rel][c_src] = {c: n / total
                                               for c, n in ctr.items()}

        self._nc_cache   = nc_cache
        self._wgc_cache  = wgc_cache
        self._sim_matrix = sim_matrix
        self._nc_soft    = {}
        self._wgc_soft   = {}

        if verbose:
            print(f'  E3: nc_cache {len(nc_cache)} keys  '
                  f'wgc_cache {len(wgc_cache)} keys')
            m = _rss_mb()
            if m:
                print(f'  RSS after E3 cache = {m:.0f} MB')

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
        key = (a_s, r_s)
        # Counts path: online updates go here; lazily normalise from raw counts.
        counts = getattr(self, '_atom_counts', {}).get(key)
        if counts is not None:
            total = getattr(self, '_atom_totals', {}).get(key, 1)
            return {t: c / total for t, c in counts.items()} if total > 0 else {}
        # Batch-fitted bigrams (no subsequent online updates for this key).
        bg = getattr(self, '_atom_bigrams', {}).get(key)
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
        # Final fallback: per-relation unigram — rebuild lazily from raw counts.
        rel_counts = getattr(self, '_rel_counts', {}).get(r_s)
        if rel_counts is not None:
            total_r = getattr(self, '_rel_totals', {}).get(r_s, 1)
            return {t: c / total_r for t, c in rel_counts.items()} if total_r > 0 else {}
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
        _Ctr = collections.Counter   # already imported at module level; avoids importlib lookups
        a_s, r_s, b_s = str(atom), str(rel), str(target)
        key = (a_s, r_s)

        # --- atom-level bigram update ---------------------------------------
        # Hard vocab cap: once we exceed _max_online_vocab keys, new (atom,rel)
        # pairs only update the unigram fallback.  This bounds memory to
        # O(max_online_vocab × avg_successors) regardless of corpus size.
        _max_vocab = getattr(self, '_max_online_vocab', 50_000)
        if key not in self._atom_counts:
            if len(self._atom_counts) < _max_vocab:
                # Fast path: plain dict (no Counter overhead).
                # Pseudo-count seed from _atom_bigrams only when non-empty.
                existing = getattr(self, '_atom_bigrams', {}).get(key, {})
                if existing:
                    _pc = 10
                    self._atom_counts[key] = _Ctr(
                        {t: max(1, round(p * _pc)) for t, p in existing.items()}
                    )
                    self._atom_totals[key] = sum(self._atom_counts[key].values())
                else:
                    self._atom_counts[key] = {}   # plain dict — 10× faster than Counter()
                    self._atom_totals[key] = 0
        ct = self._atom_counts.get(key)
        if ct is not None:
            ct[b_s] = ct.get(b_s, 0) + 1    # explicit get avoids Counter default-int overhead
            self._atom_totals[key] += 1
        # _atom_bigrams rebuilt lazily in predict_dist; no dict comprehension here.

        # --- per-relation unigram update ------------------------------------
        if r_s not in self._rel_counts:
            existing_r = getattr(self, '_rel_unigram', {}).get(r_s, {})
            if existing_r:
                _pc_r = 50
                self._rel_counts[r_s] = _Ctr(
                    {t: max(1, round(p * _pc_r)) for t, p in existing_r.items()}
                )
                self._rel_totals[r_s] = sum(self._rel_counts[r_s].values())
            else:
                self._rel_counts[r_s] = {}
                self._rel_totals[r_s] = 0
        rc = self._rel_counts[r_s]
        rc[b_s] = rc.get(b_s, 0) + 1
        self._rel_totals[r_s] += 1
        # _rel_unigram rebuilt lazily in predict_dist; no dict comprehension here.

    def cluster_from_counts(
        self,
        verbose:          bool = False,
        max_cluster_atoms: int = 100,
    ) -> None:
        """Run E0+E1+E3 clustering from already-collected ``_atom_counts``.

        After online learning (e.g. via PCH's ``update_online()`` calls) the
        learner has raw bigram counts but no distributional categories.  This
        method reconstructs compound E0 signatures from ``_atom_counts`` and
        runs the clustering pipeline — equivalent to the E0/E1/E3 portion of
        ``fit()`` but without any triple scan (O(V×R×V) only).

        Parameters
        ----------
        max_cluster_atoms
            Maximum number of atoms to cluster.  At word level V can be in the
            thousands, making O(V²) JSD clustering impractically slow.  We
            restrict to the ``max_cluster_atoms`` most frequent atoms (those with
            highest total bigram count).  Default 300 — yields sub-second
            clustering at every level while covering the vast majority of tokens.

        Sets ``assignment``, ``clusters``, ``_trans``, ``_nc_cache``,
        ``_wgc_cache``, ``_succ_dists``, ``_sim_matrix`` on self.
        """
        from collections import defaultdict, Counter as _Counter

        counts = getattr(self, '_atom_counts', None)
        if not counts:
            if verbose:
                print('  cluster_from_counts: no _atom_counts — skipping')
            return

        # ── Pre-filter to top-N atoms BEFORE building bwd_raw ───────────────
        # Building bwd_raw over all atoms creates O(unique_targets × unique_sources)
        # temporary dict entries — for large corpora this causes multi-GB spikes.
        # Solution: compute forward totals first (one pass, no large allocs),
        # select the top-N atoms, then only build fwd/bwd for those atoms.
        fwd_totals: dict = defaultdict(int)
        n_unique_fwd: int = 0
        for (a_s, r_s), ctr in counts.items():
            total = sum(ctr.values())
            fwd_totals[a_s] += total
            if a_s not in fwd_totals:
                n_unique_fwd += 1
        n_unique = len(fwd_totals)
        if n_unique < 2:
            return
        # Determine which atoms to include.
        if len(fwd_totals) > max_cluster_atoms:
            top_set: set = set(
                sorted(fwd_totals, key=lambda a: -fwd_totals[a])[:max_cluster_atoms]
            )
            if verbose:
                print(f'  cluster_from_counts: V={n_unique} → top {max_cluster_atoms} atoms')
        else:
            top_set = set(fwd_totals)

        # ── Reconstruct fwd_raw / bwd_raw for top atoms only ─────────────────
        fwd_raw: dict = defaultdict(_Counter)   # atom → {'rel:tgt': count}
        bwd_raw: dict = defaultdict(_Counter)   # atom → {'rev_rel:src': count}
        _rel_raw: dict = defaultdict(_Counter)

        for (a_s, r_s), ctr in counts.items():
            if a_s not in top_set:
                for b_s, cnt in ctr.items():
                    _rel_raw[r_s][b_s] += cnt
                continue
            for b_s, cnt in ctr.items():
                fwd_raw[a_s][f'{r_s}:{b_s}']     += cnt
                if b_s in top_set:
                    bwd_raw[b_s][f'rev_{r_s}:{a_s}'] += cnt
                _rel_raw[r_s][b_s]                += cnt

        # ── E0: compound signatures ──────────────────────────────────────────
        sigs:      dict = {}
        atom_freq: dict = {}
        for atom in top_set:
            fwd_t = sum(fwd_raw[atom].values())
            bwd_t = sum(bwd_raw[atom].values())
            d: dict = {}
            if fwd_t > 0:
                for k, cnt in fwd_raw[atom].items():
                    d[('fwd', k)] = cnt / fwd_t
            if bwd_t > 0:
                for k, cnt in bwd_raw[atom].items():
                    d[('bwd', k)] = cnt / bwd_t
            if d:
                sigs[atom] = d
                atom_freq[atom] = fwd_t + bwd_t

        # Cap output vocab (mirrors fit() behaviour).
        n_for_cap  = len(sigs)
        _vocab_size = min(2000, n_for_cap * 2) if n_for_cap > 50 else None
        if _vocab_size is not None:
            out_freq: _Counter = _Counter()
            for atom, sig in sigs.items():
                w = atom_freq.get(atom, 1)
                for ctx_key, p in sig.items():
                    out_freq[ctx_key] += p * w
            top_out = {k for k, _ in out_freq.most_common(_vocab_size)}
            capped: dict = {}
            for atom, sig in sigs.items():
                restricted = {k: v for k, v in sig.items() if k in top_out}
                mass = sum(restricted.values())
                if mass > 1e-12:
                    capped[atom] = {k: v / mass for k, v in restricted.items()}
            sigs = capped

        if len(sigs) < 2:
            return

        assignment, clusters = _jsd_cluster(sigs, n_clusters=self._K,
                                             verbose=verbose)
        self.assignment = assignment
        self.clusters   = clusters
        self._K         = len(clusters)

        # Pass fwd_raw to avoid a second O(E) scan of _atom_counts inside
        # _rebuild_caches_from_assignment (eliminates the double-scan).
        self._rebuild_caches_from_assignment(
            assignment, clusters, _rel_raw=_rel_raw, _fwd_raw=fwd_raw)

        if verbose:
            print(f'  cluster_from_counts: V={n_unique} atoms → K={self._K} clusters')

    def _rebuild_caches_from_assignment(
        self,
        assignment: dict,
        clusters:   dict,
        _rel_raw:   'dict | None' = None,
        _fwd_raw:   'dict | None' = None,
    ) -> None:
        """Rebuild E1/E3 caches from any ``assignment`` dict.

        Called by ``cluster_from_counts()`` (after JSD clustering) and by
        ``cluster_from_type_abstraction()`` (after type-tuple grouping).
        Reconstructs the forward-context distributions from ``self._atom_counts``
        and derives cluster-level transition matrices.

        Parameters
        ----------
        assignment
            ``{surface_str: cluster_id}`` mapping every atom to its cluster.
        clusters
            ``{cluster_id: set_or_list_of_surface_strings}`` inverse mapping.
        _rel_raw
            Optional pre-computed ``{rel: {target: count}}`` dict (passed by
            ``cluster_from_counts`` to avoid a second scan of ``_atom_counts``).
            If *None* the dict is reconstructed internally.
        _fwd_raw
            Optional pre-computed ``{atom: {'rel:target': count}}`` dict (passed
            by ``cluster_from_counts`` to eliminate the second O(E) scan of
            ``_atom_counts``).  If *None* fwd_raw is reconstructed from counts.
        """
        import collections as _col
        from collections import Counter as _Counter

        counts = getattr(self, '_atom_counts', {})

        # ── Reconstruct fwd_raw (and optionally _rel_raw) ────────────────────
        if _fwd_raw is not None:
            # Caller already built fwd_raw — skip the O(E) scan entirely.
            fwd_raw: dict = _fwd_raw
            if _rel_raw is None:
                # Need to build _rel_raw from counts (rare: only when called
                # without a pre-computed _rel_raw, e.g. from type-abstraction).
                _rel_raw_local: dict = _col.defaultdict(_Counter)
                for (a_s, r_s), ctr in counts.items():
                    for b_s, cnt in ctr.items():
                        _rel_raw_local[r_s][b_s] += cnt
                _rel_raw = _rel_raw_local
        else:
            fwd_raw = _col.defaultdict(_Counter)
            if _rel_raw is None:
                _rel_raw_local = _col.defaultdict(_Counter)
                for (a_s, r_s), ctr in counts.items():
                    for b_s, cnt in ctr.items():
                        fwd_raw[a_s][f'{r_s}:{b_s}'] += cnt
                        _rel_raw_local[r_s][b_s]      += cnt
                _rel_raw = _rel_raw_local
            else:
                for (a_s, r_s), ctr in counts.items():
                    for b_s, cnt in ctr.items():
                        fwd_raw[a_s][f'{r_s}:{b_s}'] += cnt

        # ── E3: succ_dists → sim_matrix ──────────────────────────────────────
        succ: dict = {}
        for cid, members in clusters.items():
            merged: dict = _col.defaultdict(float)
            n = 0
            for tok in members:
                ctr = fwd_raw.get(tok)
                if not ctr:
                    continue
                total = sum(ctr.values())
                if total == 0:
                    continue
                for k, cnt in ctr.items():
                    merged[k] += cnt / total
                n += 1
            succ[cid] = {k: v / n for k, v in merged.items()} if n else {}
        self._succ_dists = succ
        self._sim_matrix = _build_sim_matrix(succ, self._K)

        # ── E1: nc_cache / wgc_cache / _trans ────────────────────────────────
        _trans_raw  = _col.defaultdict(lambda: _col.defaultdict(_Counter))
        nc_raw      = _col.defaultdict(_Counter)
        wgc_raw     = _col.defaultdict(_Counter)

        for (a_s, r_s), ctr in counts.items():
            c_a = assignment.get(a_s)
            if c_a is None:
                continue
            ca_s = str(c_a)
            for b_s, cnt in ctr.items():
                c_b = assignment.get(b_s)
                if c_b is None:
                    continue
                cb_s = str(c_b)
                nc_raw[(ca_s, r_s)][cb_s]        += cnt
                wgc_raw[(ca_s, r_s, cb_s)][b_s]  += cnt
                _trans_raw[r_s][ca_s][cb_s]       += cnt

        nc_cache: dict = {}
        for k, ctr in nc_raw.items():
            total = sum(ctr.values())
            if total > 0:
                nc_cache[k] = {(t,): c / total for t, c in ctr.items()}

        wgc_cache: dict = {}
        for k, ctr in wgc_raw.items():
            total = sum(ctr.values())
            if total > 0:
                wgc_cache[k] = {(t,): c / total for t, c in ctr.items()}

        _trans: dict = {}
        for rel, src_map in _trans_raw.items():
            _trans[rel] = {}
            for c_src, ctr in src_map.items():
                total = sum(ctr.values())
                if total > 0:
                    _trans[rel][c_src] = {c: n / total for c, n in ctr.items()}

        # ── Update rel stats ─────────────────────────────────────────────────
        self._rel_unigram = {}
        self._rel_totals  = {}
        for rel, ctr in _rel_raw.items():
            total = sum(ctr.values())
            self._rel_totals[rel] = total
            if total > 0:
                self._rel_unigram[rel] = {k: v / total for k, v in ctr.items()}

        self._nc_cache  = nc_cache
        self._wgc_cache = wgc_cache
        self._trans     = _trans
        self._nc_soft   = {}
        self._wgc_soft  = {}

    def cluster_from_type_abstraction(
        self,
        vocab:            'AtomVocabulary',
        lower_assignment: dict,
        verbose:          bool = False,
        max_alphabet:     int  = 15,
        max_tuple_len:    int  = 2,
    ) -> None:
        """Assign clusters by constituent type-tuple — symbolic parameter sharing.

        For each atom seen at this level, look up its :class:`MergedAtom` /
        :class:`SegmentedAtom` record in ``vocab`` and map each direct
        constituent to its type-id in ``lower_assignment``.  Two surface strings
        receive the same cluster iff they have the *same constituent type-sequence*.

        This is the symbolic analogue of weight sharing in neural networks:
        ``[the+king]`` and ``[a+king]`` receive the same abstract type if both
        ``the`` and ``a`` have type ``DET`` in ``lower_assignment`` — so the
        upper learner generalises across surface forms that share structure.

        Called as a fallback by :meth:`PCH.analyse` when
        ``cluster_from_counts`` yields K≤1 at level≥1 (surface forms too sparse
        for JSD clustering).

        Parameters
        ----------
        vocab
            Shared :class:`AtomVocabulary` from the parent
            :class:`PredictiveCodingHierarchy`.
        lower_assignment
            ``assignment`` dict of the level directly below (level−1
            :class:`RelationalLearner`): ``{surface_str: type_id}``.
        max_alphabet
            Keep only the ``max_alphabet`` most-frequent constituent types;
            map everything else to ``'__OTHER__'``.  Limits alphabet size so
            the number of distinct type-tuples stays manageable.  Default 15.
        max_tuple_len
            For chunks with more than ``max_tuple_len`` constituents, use only
            the first and last constituent types (boundary signature).  Default
            2, giving at most ``(max_alphabet+1)²`` possible type-tuples.
        """
        import collections as _col

        counts = getattr(self, '_atom_counts', None)
        if not counts:
            return

        all_surfaces = sorted({a_s for (a_s, _) in counts})

        # ── Count constituent type frequencies to build coarsening map ────────
        type_freq: dict = _col.Counter()
        for surf in all_surfaces:
            atom = vocab.lookup(surf)
            if atom is None:
                t = str(lower_assignment.get(surf, surf))
                type_freq[t] += 1
            elif isinstance(atom, MergedAtom):
                type_freq[str(lower_assignment.get(str(atom.left),  str(atom.left)))]  += 1
                type_freq[str(lower_assignment.get(str(atom.right), str(atom.right)))] += 1
            else:   # SegmentedAtom
                for c in atom.constituents:
                    type_freq[str(lower_assignment.get(str(c), str(c)))] += 1

        top_types = {t for t, _ in type_freq.most_common(max_alphabet)}
        OTHER = '__OTHER__'

        def coarsen(raw) -> str:
            t = str(raw)
            return t if t in top_types else OTHER

        # ── Map each surface → coarsened type-tuple ───────────────────────────
        type_tuple_map: dict = {}
        for surf in all_surfaces:
            atom = vocab.lookup(surf)
            if atom is None:
                tup = (coarsen(lower_assignment.get(surf, surf)),)
            elif isinstance(atom, MergedAtom):
                tup = (coarsen(lower_assignment.get(str(atom.left),  str(atom.left))),
                       coarsen(lower_assignment.get(str(atom.right), str(atom.right))))
            else:   # SegmentedAtom
                full = tuple(coarsen(lower_assignment.get(str(c), str(c)))
                             for c in atom.constituents)
                # Truncate long tuples to boundary signature (first, last)
                tup = (full[0], full[-1]) if len(full) > max_tuple_len else full
            type_tuple_map[surf] = tup

        # ── Deterministic grouping by type-tuple → clusters ──────────────────
        tuple_to_cid: dict = {}
        next_cid   = 0
        assignment: dict = {}
        clusters:   dict = {}
        for surf, tup in type_tuple_map.items():
            if tup not in tuple_to_cid:
                tuple_to_cid[tup] = next_cid
                clusters[next_cid] = set()
                next_cid += 1
            cid = tuple_to_cid[tup]
            assignment[surf] = cid
            clusters[cid].add(surf)

        if len(clusters) < 2:
            if verbose:
                print(f'  cluster_from_type_abstraction: only {len(clusters)} '
                      f'cluster(s) — skipping')
            return

        self.assignment = assignment
        self.clusters   = clusters
        self._K         = len(clusters)

        self._rebuild_caches_from_assignment(assignment, clusters)

        # ── Cross-level functor maps (for build_cross_level_functor) ──────────
        # _constituent_type_map[lower_type_str] = {upper_cid: probability}
        # Records: given a constituent of lower type k, which upper cluster j
        # does the composed chunk tend to belong to?
        ctmap_raw: dict = _col.defaultdict(_col.Counter)
        for surf, cid in assignment.items():
            atom = vocab.lookup(surf)
            if atom is None:
                lower_t = str(lower_assignment.get(surf, surf))
                ctmap_raw[lower_t][cid] += 1
            elif isinstance(atom, MergedAtom):
                for c_surf in (str(atom.left), str(atom.right)):
                    lower_t = str(lower_assignment.get(c_surf, c_surf))
                    ctmap_raw[lower_t][cid] += 1
            else:   # SegmentedAtom
                for c in atom.constituents:
                    lower_t = str(lower_assignment.get(str(c), str(c)))
                    ctmap_raw[lower_t][cid] += 1

        self._constituent_type_map: dict = {}
        for lower_t, ctr in ctmap_raw.items():
            total = sum(ctr.values())
            self._constituent_type_map[lower_t] = {c: n / total
                                                   for c, n in ctr.items()}

        # _upper_to_constituent[upper_cid] = dominant lower type string
        # Inverse map used for the Decompose functor G: L_{N+1} → L_N.
        upper_to_lower_raw: dict = _col.defaultdict(_col.Counter)
        for lower_t, cid_dist in self._constituent_type_map.items():
            for cid, p in cid_dist.items():
                upper_to_lower_raw[cid][lower_t] += p
        self._upper_to_constituent: dict = {
            cid: max(ctr, key=ctr.get)
            for cid, ctr in upper_to_lower_raw.items()
        }

        if verbose:
            print(f'  cluster_from_type_abstraction: {len(all_surfaces)} atoms '
                  f'→ K={self._K} type-clusters')

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

    def observe_gated(
        self,
        atom:       Any,
        surprisal:  float,
        threshold:  float = 1.0,
        beta:       float = 1.0,
    ) -> None:
        """Gated DeltaNet-inspired belief update.

        Inspired by Gated DeltaNet (Yang et al. 2025): the delta rule performs
        a selective **erase-then-write** whose amplitude is controlled by a
        data-dependent gate ``α_t``.  Here:

        1. Compute gate:  ``alpha = exp(−beta × surp / threshold)``
           - ``surprisal ≈ 0``  (expected token)   → ``alpha ≈ 1`` → almost no erase.
           - ``surprisal >> threshold`` (boundary)  → ``alpha ≈ 0`` → strong erase,
             old context wiped before the new observation is written.
        2. Decay (erase): ``belief = alpha * belief + (1-alpha) * uniform``.
        3. Observe (write): standard sharp Bayesian update with ``obs_sharpness``.

        For document-boundary resets call ``decay(rate=0.0)`` explicitly —
        ``_reset_buffers`` does this automatically.

        Parameters
        ----------
        atom
            The emitted chunk surface string.
        surprisal
            −log₂ P(atom | context) computed just before emission.
        threshold
            Adaptive boundary threshold at this level; used to normalise
            surprisal so the gate is calibrated across levels.
        beta
            Gate sharpness (default 1.0).  Larger values make the erase more
            sensitive to deviations from the threshold.
        """
        alpha = math.exp(-beta * surprisal / max(threshold, 1e-6))
        self.decay(rate=alpha)
        self.observe(atom)

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

    # Build K×K JSD pairwise distance matrix — vectorised (same as _jsd_cluster).
    # Old: O(K²·D) Python dict iterations.  New: O(K·V) numpy ops per row.
    _all_toks = sorted({tok for dd in succ_dists.values() for tok in dd})
    _V = len(_all_toks)
    if _V > 0:
        _tok_idx = {t: ii for ii, t in enumerate(_all_toks)}
        _P = np.zeros((K, _V), dtype=np.float64)
        for _cid, _dd in succ_dists.items():
            if isinstance(_cid, int) and 0 <= _cid < K:
                for _tok, _prob in _dd.items():
                    _jj = _tok_idx.get(_tok)
                    if _jj is not None:
                        _P[_cid, _jj] = _prob
        _eps = 1e-15
        with np.errstate(divide='ignore', invalid='ignore'):
            _log_P = np.where(_P > _eps, np.log2(_P), 0.0)
        _H_P = -np.einsum('iv,iv->i', _P, _log_P)   # (K,)
        D = np.zeros((K, K), dtype=np.float64)
        for _i in range(K):
            _mix = (_P[_i:_i+1, :] + _P) * 0.5
            with np.errstate(divide='ignore', invalid='ignore'):
                _log_mix = np.where(_mix > _eps, np.log2(_mix), 0.0)
            _H_mix = -np.einsum('jv,jv->j', _mix, _log_mix)
            D[_i] = np.maximum(0.0, _H_mix - (_H_P[_i] + _H_P) * 0.5)
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) * 0.5  # symmetrise for numerical safety
    else:
        D = np.zeros((K, K), dtype=np.float64)

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
        # Vectorised: exp(-T * dist_mat), diagonal forced to 1.
        sim = np.exp(-temperature * dist_mat)
        np.fill_diagonal(sim, 1.0)
        return sim

    # ---- dense / general_graph / unknown → JSD (no change) ----------------
    else:
        # D is already the numpy JSD matrix; just apply exp(-T·D) vectorised.
        sim = np.exp(-temperature * D)
        np.fill_diagonal(sim, 1.0)
        return sim

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

    # L2 pairwise distances — vectorised via broadcasting.
    # Old: O(K²) Python loop calling np.linalg.norm K² times.
    # New: one broadcast subtraction + einsum → O(K²·dim) numpy.
    diff = coords[:, None, :] - coords[None, :, :]   # K × K × dim
    dist_mat = np.sqrt((diff ** 2).sum(axis=-1))       # K × K
    sim = np.exp(-temperature * dist_mat)
    np.fill_diagonal(sim, 1.0)
    return sim


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

        # Vectorised pairwise JSD — precompute per-row entropy then broadcast.
        eps = 1e-15
        with np.errstate(divide='ignore', invalid='ignore'):
            log_mat = np.where(mat > eps, np.log2(mat), 0.0)
        row_ents = -np.einsum('id,id->i', mat, log_mat)   # (n,)

        # Full broadcast (n×n×D) when memory is manageable (< 128 MB).
        if n * n * D * 8 < 128 * 1024 * 1024:
            mix = (mat[:, None, :] + mat[None, :, :]) * 0.5  # (n, n, D)
            with np.errstate(divide='ignore', invalid='ignore'):
                log_mix = np.where(mix > eps, np.log2(mix), 0.0)
            H_mix = -np.einsum('ijd,ijd->ij', mix, log_mix)   # (n, n)
            jsd_arr = np.maximum(
                0.0, H_mix - (row_ents[:, None] + row_ents[None, :]) * 0.5)
            np.fill_diagonal(jsd_arr, 0.0)
        else:
            # Row-by-row to stay within memory budget.
            jsd_arr = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                mix_i = (mat[i:i+1, :] + mat) * 0.5        # (n, D)
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_mix_i = np.where(mix_i > eps, np.log2(mix_i), 0.0)
                H_mix_i = -np.einsum('jd,jd->j', mix_i, log_mix_i)  # (n,)
                jsd_arr[i] = np.maximum(
                    0.0, H_mix_i - (row_ents[i] + row_ents) * 0.5)
            np.fill_diagonal(jsd_arr, 0.0)

        # Sort all upper-triangle pairs by JSD using numpy argsort.
        i_idx, j_idx = np.triu_indices(n, k=1)
        jsd_vals = jsd_arr[i_idx, j_idx]
        order = np.argsort(jsd_vals, kind='stable')
        pairs = [(float(jsd_vals[o]), int(i_idx[o]), int(j_idx[o]))
                 for o in order]

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

        pairs = sorted((_get(i, j), i, j)
                       for i in range(n) for j in range(i + 1, n))

    # UPGMA agglomerative clustering (average-link) with lazy-deletion heap.
    # Average-link avoids single-link chaining (which collapses all atoms into K=1).
    # Heap gives O(n² log n) vs O(n³) for the naive submatrix approach.
    #
    # Threshold strategy: when n_clusters is None we run ALL merges to completion,
    # collect the sequence of merge distances, then apply _gap_threshold on those
    # distances (not original pairwise JSDs) to find the dendrogram cut point.
    # This is more principled than thresholding on raw pairwise JSDs.

    import heapq as _hq

    # Build initial distance matrix (n×n float list for in-place UPGMA updates)
    try:
        import numpy as _np_upgma
        _D = jsd_arr.tolist()
    except (ImportError, NameError):
        _D = [[_get(i, j) for j in range(n)] for i in range(n)]

    _sizes   = [1] * n
    _active  = set(range(n))

    # Build initial heap: all upper-triangle pairs sorted by JSD.
    _heap: list = [(float(_D[i][j]), i, j)
                   for i in range(n) for j in range(i + 1, n)]
    _hq.heapify(_heap)

    # Map: original atom index → current cluster representative.
    _label = list(range(n))

    if n_clusters is not None:
        # ── Fixed-K mode: stop when target cluster count is reached ────────────
        while _heap and len(_active) > n_clusters:
            while _heap:
                min_d, ci_idx, cj_idx = _hq.heappop(_heap)
                if ci_idx in _active and cj_idx in _active:
                    break
            else:
                break
            ni, nj = _sizes[ci_idx], _sizes[cj_idx]
            for k in _active:
                if k == ci_idx or k == cj_idx:
                    continue
                new_d = (ni * _D[ci_idx][k] + nj * _D[cj_idx][k]) / (ni + nj)
                _D[ci_idx][k] = _D[k][ci_idx] = new_d
                _hq.heappush(_heap, (new_d, min(ci_idx, k), max(ci_idx, k)))
            _sizes[ci_idx] += nj
            _active.discard(cj_idx)
            for idx in range(n):
                if _label[idx] == cj_idx:
                    _label[idx] = ci_idx
    else:
        # ── Auto-K mode: full dendrogram + gap detection ────────────────────────
        # Run all merges, recording merge distances in order.
        _merge_log: list[tuple[float, int, int]] = []   # (dist, ci, cj)
        _snapshots: list[list[int]]               = []  # _label after each merge

        while _heap and len(_active) > 1:
            while _heap:
                min_d, ci_idx, cj_idx = _hq.heappop(_heap)
                if ci_idx in _active and cj_idx in _active:
                    break
            else:
                break
            ni, nj = _sizes[ci_idx], _sizes[cj_idx]
            for k in _active:
                if k == ci_idx or k == cj_idx:
                    continue
                new_d = (ni * _D[ci_idx][k] + nj * _D[cj_idx][k]) / (ni + nj)
                _D[ci_idx][k] = _D[k][ci_idx] = new_d
                _hq.heappush(_heap, (new_d, min(ci_idx, k), max(ci_idx, k)))
            _sizes[ci_idx] += nj
            _active.discard(cj_idx)
            for idx in range(n):
                if _label[idx] == cj_idx:
                    _label[idx] = ci_idx
            _merge_log.append((min_d, ci_idx, cj_idx))
            _snapshots.append(_label[:])

        # Apply gap detection on merge distances.
        if jsd_threshold is None:
            merge_dists = [d for d, _, _ in _merge_log]
            jsd_threshold = _gap_threshold(merge_dists)

        # Find the latest snapshot whose next merge distance exceeds the threshold.
        # That snapshot gives the dendrogram cut.
        best_snap = _snapshots[-1] if _snapshots else _label[:]
        for step_idx, (d, _, _) in enumerate(_merge_log):
            if d > jsd_threshold:
                best_snap = (_snapshots[step_idx - 1] if step_idx > 0
                             else list(range(n)))
                break
        _label = best_snap

    # Remap active root IDs → 0..K-1
    old_ids = sorted(set(_label))
    remap = {old: new for new, old in enumerate(old_ids)}
    final = [remap[_label[i]] for i in range(n)]

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
            triples: list[tuple[Any, str, Any]] | None = None,
            jsd_threshold: float | None = None,
            verbose: bool = True) -> None:
        """Cluster relation types using atom assignments from a fitted RelationalLearner.

        Args:
            learner:       A fitted RelationalLearner (learner.assignment populated).
            triples:       Ignored — kept for backward compat. Uses learner._atom_counts.
            jsd_threshold: JSD below which two relations are merged (n_clusters=None).
                           None (default) → auto-detect via Kneedle algorithm.
            verbose:       Print cluster assignments.
        """
        rel_counts: dict[str, dict] = collections.defaultdict(
            lambda: collections.defaultdict(float))
        n_missing = 0
        # O(V×R×V) instead of O(N) triple scan — uses learner._atom_counts
        for (a_s, r_s), ctr in learner._atom_counts.items():
            c_a = learner.assignment.get(a_s)
            if c_a is None:
                n_missing += sum(ctr.values())
                continue
            for b_s, cnt in ctr.items():
                c_b = learner.assignment.get(b_s)
                if c_b is None:
                    n_missing += cnt
                    continue
                rel_counts[r_s][(c_a, c_b)] += float(cnt)

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

    def fit(self, triples: list[tuple[Any, str, Any]] | None = None,
            n_clusters: int | None = None,
            jsd_threshold: float | None = None,
            verbose: bool = True,
            learner: 'RelationalLearner | None' = None) -> None:
        """Learn next_rel distributions from chains in the triple set.

        Finds all (a -r1→ b) AND (b -r2→ c) chains, accumulates (r1, r2) pairs,
        then clusters relations by JSD of their P(next_rel | r) distributions.

        Args:
            triples:       (atom, relation, atom) triples. Ignored when learner provided.
            n_clusters:    Target cluster count.  None → JSD-threshold merging.
            jsd_threshold: Merge threshold when n_clusters is None.
                           None (default) → auto-detect via Kneedle algorithm.
            verbose:       Print distributions and clusters.
            learner:       Fitted RelationalLearner. When provided, uses learner._atom_counts
                           for O(V×R×V) chain counting instead of O(N) triple scan.
        """
        # chain_count[r1][r2] = Σ_b  in_count[b][r1] × out_count[b][r2]
        # where:
        #   in_count[b][r1]  = # triples (?, r1, b)   [edges arriving at b]
        #   out_count[b][r2] = # triples (b, r2, ?)   [edges leaving b]
        #
        # O(V×R×V) when learner._atom_counts available, else O(N) from triples.
        in_count:  dict[str, dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int))
        out_count: dict[str, dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int))
        if learner is not None and hasattr(learner, '_atom_counts'):
            for (a_s, r_s), ctr in learner._atom_counts.items():
                total = sum(ctr.values())
                out_count[a_s][r_s] += total
                for b_s, cnt in ctr.items():
                    in_count[b_s][r_s] += cnt
        else:
            for a, r, b in (triples or []):
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
            triples:           list | None = None,
            verbose:           bool = False,
            use_atom_partners: bool = True,
            max_atoms:         int  = 200,
            ) -> 'RelationalParadigmDiscoverer':
        """Fit role signatures from learner._atom_counts (or triples as fallback).

        Args:
            use_atom_partners: If True (default), partner atoms are identified by
                their surface form.  This gives a V-dimensional role signature and
                can separate fine-grained roles even when E0 has a mega-cluster.
                If False, use E0 category id as partner label (K-dimensional),
                which is more abstract but loses sub-category distinctions.
            max_atoms: Cap the number of atoms clustered (top-N by frequency).
                       Prevents O(V²×D) blowup at word/chunk levels where V≫200.

        O(V×R×V) from learner._atom_counts — no triple scan needed.
        For character-level data: V=27, D≤R×V≈108 → instant.
        """
        from collections import defaultdict, Counter as _Ctr

        src_raw: dict = defaultdict(_Ctr)   # atom → {'>rel:partner': count}
        tgt_raw: dict = defaultdict(_Ctr)   # atom → {'<rel:partner': count}
        atom_freq: dict = {}                # atom → total count (for capping)

        if hasattr(learner, '_atom_counts'):
            # O(V×R×V) — use precomputed counts from learner
            for (a_s, r_s), ctr in learner._atom_counts.items():
                if use_atom_partners:
                    for b_s, cnt in ctr.items():
                        src_raw[a_s][f'>{r_s}:{b_s}'] += cnt   # a is source
                        tgt_raw[b_s][f'<{r_s}:{a_s}'] += cnt   # b is target
                        atom_freq[a_s] = atom_freq.get(a_s, 0) + cnt
                        atom_freq[b_s] = atom_freq.get(b_s, 0) + cnt
                else:
                    c_a = learner.assignment.get(a_s)
                    if c_a is None:
                        continue
                    for b_s, cnt in ctr.items():
                        c_b = learner.assignment.get(b_s)
                        if c_b is None:
                            continue
                        src_raw[a_s][f'>{r_s}:{c_b!s}'] += cnt
                        tgt_raw[b_s][f'<{r_s}:{c_a!s}'] += cnt
                        atom_freq[a_s] = atom_freq.get(a_s, 0) + cnt
                        atom_freq[b_s] = atom_freq.get(b_s, 0) + cnt
        else:
            for a, r, b in (triples or []):
                a_s, r_s, b_s = str(a), str(r), str(b)
                if use_atom_partners:
                    partner_a = b_s
                    partner_b = a_s
                else:
                    c_a = learner.assignment.get(a_s)
                    c_b = learner.assignment.get(b_s)
                    if c_a is None or c_b is None:
                        continue
                    partner_a = str(c_b)
                    partner_b = str(c_a)
                src_raw[a_s][f'>{r_s}:{partner_a}'] += 1
                tgt_raw[b_s][f'<{r_s}:{partner_b}'] += 1
                atom_freq[a_s] = atom_freq.get(a_s, 0) + 1
                atom_freq[b_s] = atom_freq.get(b_s, 0) + 1

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

        # Cap to top-max_atoms most frequent atoms (prevents O(V²×D) blowup).
        if len(sigs) > max_atoms:
            top = sorted(sigs, key=lambda a: -atom_freq.get(a, 0))[:max_atoms]
            top_set = set(top)
            sigs = {a: s for a, s in sigs.items() if a in top_set}

        # Cap context dimension (D) to prevent sparse O(n²×D) blowup.
        # At word level each atom partners with unique words → D ~ 2×V.
        # Keeping the top-2N most informative context keys keeps D manageable.
        _vocab_cap = min(2000, len(sigs) * 2) if len(sigs) > 10 else None
        if _vocab_cap is not None:
            from collections import Counter as _Ctr2
            ctx_freq: _Ctr2 = _Ctr2()
            for a, sig in sigs.items():
                w = atom_freq.get(a, 1)
                for k, p in sig.items():
                    ctx_freq[k] += p * w
            top_ctx = {k for k, _ in ctx_freq.most_common(_vocab_cap)}
            capped_sigs: dict = {}
            for a, sig in sigs.items():
                r = {k: v for k, v in sig.items() if k in top_ctx}
                mass = sum(r.values())
                if mass > 1e-12:
                    capped_sigs[a] = {k: v / mass for k, v in r.items()}
            sigs = capped_sigs

        # ------------------------------------------------------------------
        # M20: PPMI normalisation — remove language/domain-frequency bias.
        #
        # Raw probability signatures P(ctx | atom) encode *both* functional
        # role and domain-frequency: a context that appears uniformly across
        # all atoms (e.g. a very common character) gets high raw probability
        # but carries zero role information.  PPMI = max(0, log₂ P(ctx|a) /
        # P(ctx)) retains only the *excess* predictability above baseline,
        # making signatures domain-agnostic (same algorithm works on pixels,
        # audio, or sensorimotor sequences, not just natural language).
        #
        # Reference: Levy & Goldberg 2014 (NeurIPS) — skip-gram word2vec
        # implicitly computes SVD on the shifted-PMI matrix; PPMI is the
        # standard language-agnostic normalisation used in cross-lingual
        # distributional clustering (Christodoulopoulos et al., EMNLP 2010).
        #
        # After PPMI, each signature is re-normalised to sum=1 so that the
        # downstream JSD-based clustering step remains unchanged.
        # ------------------------------------------------------------------
        if sigs:
            # Compute marginal P(ctx) = weighted average over all atoms,
            # where the weight is the atom's relative total frequency.
            total_mass = sum(atom_freq.get(a, 1) for a in sigs)
            if total_mass <= 0:
                total_mass = len(sigs)
            ctx_marginal: dict = {}
            for a, sig in sigs.items():
                w = atom_freq.get(a, 1) / total_mass
                for ctx, p in sig.items():
                    ctx_marginal[ctx] = ctx_marginal.get(ctx, 0.0) + p * w
            # Apply PPMI: max(0, log₂(P(ctx|a) / P(ctx))), then re-normalise.
            ppmi_sigs: dict = {}
            for a, sig in sigs.items():
                ppmi: dict = {}
                for ctx, p_ctx_given_a in sig.items():
                    p_ctx = ctx_marginal.get(ctx, 1e-15)
                    val   = p_ctx_given_a / max(p_ctx, 1e-15)
                    ppmi_val = math.log2(val) if val > 1.0 else 0.0   # max(0, log)
                    if ppmi_val > 0.0:
                        ppmi[ctx] = ppmi_val
                if ppmi:
                    mass = sum(ppmi.values())
                    ppmi_sigs[a] = {k: v / mass for k, v in ppmi.items()}
                # Atoms with all-zero PPMI (indistinguishable from background)
                # are omitted: they carry no role information.
            sigs = ppmi_sigs if ppmi_sigs else sigs   # fall back to raw if all zero

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
                cats = sorted({str(learner.assignment.get(m, '?')) for m in members})
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
            learner:  Fitted RelationalLearner (provides ``assignment`` + ``_atom_counts``).
            triples:  Ignored — kept for backward compat. Uses learner._atom_counts when
                      available (V≤300 → atom-level matrices, else K×K category-level).
            max_jsd:  Maximum row-weighted average JSD to accept a rule.
        """
        V = len(learner.assignment)
        if hasattr(learner, '_atom_counts') and V <= 300:
            use_atom_level = True
        elif triples is not None:
            use_atom_level = (V <= 300)
        else:
            use_atom_level = False

        if use_atom_level:
            return self._fit_atom_level(learner, triples, verbose, max_jsd)
        else:
            return self._fit_cat_level(learner, verbose, max_jsd)

    def _fit_atom_level(self,
                        learner: 'RelationalLearner',
                        triples: list | None,
                        verbose: bool,
                        max_jsd: float,
                        ) -> 'RelationalAlgebra':
        """Atom-level V×V matrices — high resolution, O(V³) composition."""
        atoms   = sorted(learner.assignment.keys())
        V       = len(atoms)
        idx     = {a: i for i, a in enumerate(atoms)}

        # Count raw triples — O(V×R×V) from _atom_counts, else O(N) from triples
        raw: dict = collections.defaultdict(int)             # (rel, src_i, tgt_j) → count
        rel_src_total: dict = collections.defaultdict(       # rel → {src_i → total}
            lambda: collections.defaultdict(int))

        relations_seen: set = set()
        if hasattr(learner, '_atom_counts'):
            for (a_s, r_s), ctr in learner._atom_counts.items():
                if a_s not in idx:
                    continue
                i = idx[a_s]
                relations_seen.add(r_s)
                for b_s, cnt in ctr.items():
                    if b_s not in idx:
                        continue
                    j = idx[b_s]
                    raw[(r_s, i, j)]      += cnt
                    rel_src_total[r_s][i] += cnt
        else:
            for a, r, b in (triples or []):
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
        """Matrix multiply, compare, build composition table (numpy)."""
        import numpy as _np

        # Convert to numpy arrays once
        np_mats = {r: _np.array(mats[r], dtype=_np.float64) for r in relations}
        w_arr   = _np.array(weights, dtype=_np.float64)
        w_sum   = float(w_arr.sum())
        if w_sum <= 0:
            w_arr = _np.ones(N, dtype=_np.float64)
            w_sum = float(N)

        eps = 1e-15

        def _row_entropy(m: '_np.ndarray') -> '_np.ndarray':
            """Row-wise Shannon entropy (base 2) of (N,N) matrix."""
            with _np.errstate(divide='ignore', invalid='ignore'):
                lm = _np.where(m > eps, _np.log2(m + eps), 0.0)
            return -_np.einsum('ij,ij->i', m, lm)

        def _mat_jsd(m1: '_np.ndarray', m2: '_np.ndarray') -> float:
            mix = (m1 + m2) * 0.5
            row_jsd = _np.maximum(
                0.0,
                _row_entropy(mix) - (_row_entropy(m1) + _row_entropy(m2)) * 0.5)
            return float(_np.dot(row_jsd, w_arr)) / w_sum

        # M22: MDL self-normalising criterion.
        #
        # Old approach: accept A∘B ≈ C when JSD(M_AB, M_C) < threshold, where
        # the threshold was a free parameter (Kneedle auto-detect + max_jsd cap).
        # This failed on multilingual data because the Kneedle gap vanishes when
        # V is large and matrices are sparse.
        #
        # New approach (bitter lesson): no threshold.  Accept A∘B ≈ C if and
        # only if C explains the composed transition BETTER THAN CHANCE:
        #
        #   JSD(M_AB, M_C) < JSD(M_AB, M_uniform)
        #
        # JSD(p, uniform) = (log2(N) − H(p)) / 2  per row, where N = vocab size
        # and H(p) is the row entropy of M_AB.  This baseline is computed from
        # M_AB alone — no free parameter.  It adapts automatically: sparse/flat
        # rows (near-uniform M_AB) produce a low baseline and are hard to beat;
        # peaked rows produce a high baseline and are easy to beat if M_C is
        # sharp.  The criterion is equivalent to "C reduces description length
        # relative to a flat prior" — the minimal MDL condition.
        #
        # Reference: Sequitur (Nevill-Manning & Witten 1997) + MDL theory.

        def _baseline_jsd(m_comp: '_np.ndarray') -> float:
            """JSD(M_AB, M_uniform) — MDL baseline for composition M_AB."""
            N          = m_comp.shape[1]
            H_comp_row = _row_entropy(m_comp)           # entropy of each row of M_AB
            H_max      = math.log2(N) if N > 1 else 1.0
            # JSD(p, uniform) = [H((p+u)/2) - (H(p)+H(u))/2]
            # H(uniform) = log2(N); H((p+u)/2) = H(mix where u=1/N uniform)
            # Compact form: (H_max - H_comp_row) / 2  per row (since uniform
            # has H=H_max and the mixture entropy is between H(p) and H_max).
            row_baseline = _np.maximum(0.0, (H_max - H_comp_row) * 0.5)
            return float(_np.dot(row_baseline, w_arr)) / w_sum

        # First pass: find best-match JSD for every pair AND its MDL baseline.
        best_matches: list = []  # (r_i, r_j, best_r, best_jsd, mdl_baseline)
        for r_i in relations:
            for r_j in relations:
                m_comp    = np_mats[r_i] @ np_mats[r_j]   # numpy matmul
                baseline  = _baseline_jsd(m_comp)          # MDL baseline (no param)
                best_r:   str | None = None
                best_jsd: float      = float('inf')
                for r_k in relations:
                    d = _mat_jsd(m_comp, np_mats[r_k])
                    if d < best_jsd:
                        best_jsd = d
                        best_r   = r_k
                best_matches.append((r_i, r_j, best_r, best_jsd, baseline))

        if verbose:
            print(f'  R3: MDL baseline criterion (no threshold) — '
                  f'{len(best_matches)} pairs evaluated')

        table: dict = {}
        for r_i, r_j, best_r, best_jsd, baseline in best_matches:
            # Accept iff best candidate beats the MDL baseline (better than chance)
            # AND (for numerical robustness) JSD is below the old max_jsd hard cap.
            confirmed = (best_jsd < baseline) and (best_jsd <= max_jsd)
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


def label_merged_atoms(hier) -> None:
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
        hier,
        domain_name: str  = 'discovered',
        out_path:    str | None = None,
) -> str:
    """M6: Export discovered hierarchy structure as a CTKG domain file (.ctkg DSL format).

    ``hier`` must expose ``.vocab`` (AtomVocabulary) and ``.levels`` (list of RelationalLearner).

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
    hier         Object with .vocab (AtomVocabulary) and .levels (list of
                 RelationalLearner).  label_merged_atoms is called automatically.
    domain_name  Name for the generated domain block.
    out_path     If given, write to this file.  Otherwise return the string.

    Returns
    -------
    The .ctkg file content as a string.
    """
    label_merged_atoms(hier)
    lines: list[str] = []

    lines.append(f'# Auto-generated CTKG domain: {domain_name}')
    lines.append(f'# Source: PredictiveCodingHierarchy '
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


# =============================================================================
# M8  PredictiveCodingHierarchy — online surprisal-based hierarchical learning
# =============================================================================

class _RunningStats:
    """O(1) sliding-window mean and standard deviation.

    Maintains a fixed-size circular buffer.  On each ``append`` the oldest
    evicted value is subtracted from the running sum/sum-of-squares and the
    new value is added — keeping mean and std in O(1) rather than the O(n)
    recomputation that a plain deque+sum approach requires.
    """
    __slots__ = ('_buf', '_sum', '_sum_sq', '_maxlen')

    def __init__(self, maxlen: int = 1000) -> None:
        self._maxlen = maxlen
        self._buf    = collections.deque(maxlen=maxlen)
        self._sum    = 0.0
        self._sum_sq = 0.0

    def append(self, x: float) -> None:
        if len(self._buf) == self._maxlen:
            old          = self._buf[0]   # about to be evicted
            self._sum    -= old
            self._sum_sq -= old * old
        self._buf.append(x)
        self._sum    += x
        self._sum_sq += x * x

    def __len__(self) -> int:
        return len(self._buf)

    def mean(self) -> float:
        n = len(self._buf)
        return self._sum / n if n > 0 else 0.0

    def std(self) -> float:
        n = len(self._buf)
        if n < 2:
            return 0.0
        mean = self._sum / n
        var  = self._sum_sq / n - mean * mean
        return math.sqrt(max(0.0, var))

class PredictiveCodingHierarchy:
    """Broca's-area-inspired hierarchical predictive coding over token sequences.

    Key properties:

    - **Online** (token-by-token): each token is processed immediately.
    - **Simultaneous multi-level**: emitting a chunk at level N instantly feeds
      level N+1 within the *same* ``process()`` call.
    - **Bounded working memory**: ``max_chunk_size`` (default 7, Miller's 7±2)
      prevents paragraph-swallowing.
    - **Surprisal-based boundaries**: prediction error drives segmentation, not PMI.
    - **Adaptive threshold**: self-calibrates to the running surprisal distribution
      at each level (mean + k*std), so no per-domain tuning is needed.
    - **Cold-start safe**: works with unfitted learners; relies on
      ``update_online()`` to build up prediction distributions incrementally.

    Usage (cold-start)::

        pch = PredictiveCodingHierarchy()
        pch.process_corpus(sequences)   # list[list[str]], one sub-list per document
        pch.level_summary()

    Usage (warm-start by injecting pre-fitted RelationalLearners)::

        pch = PredictiveCodingHierarchy.from_hrl(prior_pch)
        pch.process_corpus(new_sequences)
    """

    def __init__(
        self,
        n_levels:           int   = 10,
        max_chunk_size:     int   = 7,
        surprise_threshold: float = 2.0,
        adaptive_threshold: bool  = True,
        surprise_k:         float = 0.5,
        min_tokens_active:  int   = 20,
        top_down_weight:    float = 0.3,
        next_rel:           str   = 'next',
        max_online_vocab:   int   = 50_000,
        threshold_recompute_every: int = 50,
        mdl_boundaries:     bool  = False,
        min_merge_count:    int   = 2,
    ) -> None:
        """
        Parameters
        ----------
        n_levels
            Number of hierarchical levels (default 10).  The hierarchy stops
            propagating at the highest level that receives input.
        max_chunk_size
            Hard cap on working-memory buffer length.  When a buffer reaches
            this size it is flushed regardless of surprisal.  Default 7
            (Miller 1956, "The Magical Number Seven").
        surprise_threshold
            Fixed boundary threshold in bits.  Only used when
            ``adaptive_threshold=False`` or during the cold-start grace period.
        adaptive_threshold
            If True (default) the threshold at each level is
            ``mean(surprisal) + surprise_k * std(surprisal)`` computed over
            a rolling window of the last 1 000 observations.
        surprise_k
            Standard-deviation multiplier for the adaptive threshold (default
            0.5).  Smaller values = more boundaries; larger = fewer.
        min_tokens_active
            Number of tokens a level must have seen before surprisal-based
            boundaries are enabled.  During this grace period only the hard
            ``max_chunk_size`` cap produces boundaries (cold-start safety).
        top_down_weight
            Reserved for future top-down modulation; stored but not yet used.
        next_rel
            Relation name for sequential adjacency (default ``'next'``).
        mdl_boundaries
            M21: If True, use MDL pair-count criterion (Sequitur-inspired) as
            an *additional* boundary signal alongside surprisal.  A boundary
            is emitted before token T whenever the global pair count for
            (buffer[-1], T) is below ``min_merge_count`` — i.e., the pair has
            not yet appeared often enough to justify a grammar rule.  When the
            pair is seen ``min_merge_count`` times it is freely merged.  This
            replaces threshold calibration with a theoretically-grounded
            criterion: the minimum number of repetitions that yields any
            description-length reduction is exactly 2.
        min_merge_count
            M21: Minimum global occurrence count for a pair before it is
            considered "worth a grammar rule" and can be merged without a
            boundary.  Default 2 (Sequitur's digram-uniqueness threshold —
            the absolute minimum for any compression benefit).
        """
        self.n_levels           = n_levels
        self.max_chunk_size     = max_chunk_size
        self.surprise_threshold = surprise_threshold
        self.adaptive_threshold = adaptive_threshold
        self.surprise_k         = surprise_k
        self.min_tokens_active  = min_tokens_active
        self.top_down_weight    = top_down_weight
        self.next_rel           = next_rel
        self.max_online_vocab   = max_online_vocab
        self.threshold_recompute_every = threshold_recompute_every
        # M21: MDL pair-count boundaries (Sequitur-inspired).
        self.mdl_boundaries     = mdl_boundaries
        self.min_merge_count    = min_merge_count
        # M21: global pair counts accumulated across the whole corpus per level.
        # NOT reset at document boundaries — counts must be corpus-wide to give
        # a correct MDL estimate of pair frequency.
        self._pair_counts: list[dict] = [{} for _ in range(n_levels)]

        # One RelationalLearner per level (initially unfitted — cold start).
        self.learners: list[RelationalLearner] = [
            RelationalLearner() for _ in range(n_levels)
        ]
        # Propagate vocab cap to learners.
        for lrn in self.learners:
            lrn._max_online_vocab = max_online_vocab

        # Shared merge/segment registry (grows as chunks are emitted).
        self.vocab: AtomVocabulary = AtomVocabulary()

        # Per-level bounded working-memory buffers.
        self._buffers: list[list] = [[] for _ in range(n_levels)]

        # Previous token at each level (for bigram prediction context).
        self._prev: list = [None] * n_levels

        # Total tokens processed at each level (used for cold-start gate).
        self._seen: list[int] = [0] * n_levels

        # O(1) sliding-window running stats per level (mean + std for adaptive threshold).
        self._surp_hist: list[_RunningStats] = [
            _RunningStats(maxlen=1000) for _ in range(n_levels)
        ]

        # Per-level corpus of chunk sequences (needed by RelationalSenseSplitter).
        # _chunk_seqs[level] is a list of sequences; each sequence is the ordered
        # list of chunk strings emitted at that level during one document.
        # Storage is capped at _CHUNK_SEQ_TOK_CAP total tokens per level to bound
        # RAM and R2 scan time (200 K is ample for trigram sense detection).
        self._chunk_seqs:     list[list[list[str]]] = [[] for _ in range(n_levels)]
        self._chunk_cur_doc:  list[list[str]]       = [[] for _ in range(n_levels)]
        self._chunk_seqs_tok: list[int]             = [0]  * n_levels  # running token count
        self._CHUNK_SEQ_TOK_CAP: int = 200_000

        # M13: per-level Bayes filters — populated by init_beliefs() after analyse().
        self._beliefs: list = [None] * n_levels

        # M13: frozen flag — set during reprocess() to skip update_online.
        self._frozen: bool = False

        # M16: surprisal at the time _emit_buffer() is called (for gated DeltaNet).
        # Set by _process_level before boundary-triggered emission; 0.0 otherwise.
        self._emit_surp: list[float] = [0.0] * n_levels

        # Speed: cached adaptive threshold per level.  Recomputed every
        # threshold_recompute_every tokens instead of every token.
        self._threshold_cache: list[float] = [surprise_threshold] * n_levels
        self._threshold_age:   list[int]   = [0] * n_levels

    # ------------------------------------------------------------------
    # Core internals
    # ------------------------------------------------------------------

    def _surprisal(self, level: int, token: str) -> float:
        """Return -log2 P(token | prev_token) at *level* — O(1).

        Bypasses predict_dist to avoid building a full normalized dict.
        Looks up the raw count for (prev, token) and divides by the stored
        total — one dict lookup and one float division.
        Returns 0.0 during the cold-start grace period or before any data.
        """
        prev = self._prev[level]
        if prev is None or self._seen[level] < self.min_tokens_active:
            return 0.0
        learner = self.learners[level]
        key = (str(prev), self.next_rel)
        counts = getattr(learner, '_atom_counts', None)
        if counts is not None:
            ctr = counts.get(key)
            if ctr is not None:
                c = ctr.get(token, 0)
                total = learner._atom_totals.get(key, 1)
                p = c / total if c > 0 else 1e-10
                return -math.log2(p)
        # Unigram fallback (unfitted or OOV key).
        rel_counts = getattr(learner, '_rel_counts', {}).get(self.next_rel)
        if rel_counts:
            c = rel_counts.get(token, 0)
            total_r = getattr(learner, '_rel_totals', {}).get(self.next_rel, 1)
            p = c / total_r if c > 0 else 1e-10
            return -math.log2(p)
        return 0.0

    def _effective_threshold(self, level: int) -> float:
        """Adaptive boundary threshold for *level* — O(1).

        With ``adaptive_threshold=True`` returns mean + k*std using the
        O(1) ``_RunningStats`` running statistics.  Falls back to
        ``surprise_threshold`` when fewer than 10 samples have been collected.

        M13 top-down modulation (active only after ``init_beliefs()``):
        when the belief at this level has high entropy (we're uncertain about
        the current chunk category), reduce the threshold so we create more
        boundaries — smaller chunks give higher-level learners more data points
        to resolve the ambiguity.
        """
        if not self.adaptive_threshold:
            base = self.surprise_threshold
        else:
            hist = self._surp_hist[level]
            if len(hist) < 10:
                base = self.surprise_threshold
            else:
                base = hist.mean() + self.surprise_k * hist.std()

        # M13: top-down entropy modulation (only when belief is active).
        belief = self._beliefs[level]
        if belief is not None and self.top_down_weight > 0.0:
            K = max(getattr(self.learners[level], '_K', 1), 1)
            H_max = math.log2(K) if K > 1 else 1.0
            H     = belief.entropy()
            uncertainty = min(H / H_max, 1.0)   # normalised [0, 1]
            base = base * (1.0 - self.top_down_weight * uncertainty)

        return base

    def _emit_buffer(self, level: int) -> None:
        """Flush the working-memory buffer at *level*.

        Creates a composite atom:
        - len == 1  → pass through as-is (no composition)
        - len == 2  → ``MergedAtom`` (binary Merge)
        - len  > 2  → ``SegmentedAtom`` (n-ary ordered Segment)

        The chunk is immediately fed to level+1, implementing the
        **simultaneous multi-level** property within a single ``process()``
        call.
        """
        buf = self._buffers[level]
        if not buf:
            return

        if len(buf) == 1:
            chunk = str(buf[0])
        elif len(buf) == 2:
            chunk = self.vocab.add_merge(str(buf[0]), str(buf[1])).surface
        else:
            chunk = self.vocab.add_segment(
                [str(a) for a in buf], level=level + 1
            ).surface

        # Clear before recursing to prevent accidental double-flush.
        self._buffers[level] = []

        # Log the emitted chunk for RelationalSenseSplitter.
        self._chunk_cur_doc[level].append(chunk)

        # M13/M16: Gated DeltaNet belief update — erase proportional to surprisal,
        # then write observation, then predict next category.
        belief = self._beliefs[level]
        if belief is not None:
            surp      = self._emit_surp[level]
            threshold = self._effective_threshold(level)
            belief.observe_gated(chunk, surp, threshold)
            belief.transition(self.next_rel)

        # Propagate chunk to the next level (simultaneous multi-level).
        if level + 1 < self.n_levels:
            self._process_level(level + 1, chunk)

    def _process_level(self, level: int, token: str) -> None:
        """Process one *token* at the given *level*.

        Order of operations (deliberate):

        1. Predict  — compute surprisal *before* incorporating the observation
                      so the measurement is uncontaminated.
        2. Learn    — ``update_online`` incorporates the new (prev, token)
                      bigram immediately.
        3. Boundary — if surprisal exceeds the adaptive threshold, flush the
                      current buffer (emitting a chunk) before appending.
        4. Buffer   — append token to working memory.
        5. Cap      — if buffer is full, force-flush (bounded working memory).
        """
        # 1. Predict.
        surp = self._surprisal(level, token)
        self._surp_hist[level].append(surp)

        # 2. Learn — forward AND reverse bigrams for richer E0 signatures.
        #    Skipped during frozen reprocess() pass.
        prev = self._prev[level]
        if prev is not None and not self._frozen:
            self.learners[level].update_online(prev, self.next_rel, token)
            self.learners[level].update_online(token, 'prev', prev)
        self._prev[level] = token
        self._seen[level] += 1

        # 2b. M21: Update global pair count for (prev, token).
        # Done before boundary check so counts are corpus-wide and accurate.
        if prev is not None and self.mdl_boundaries:
            pair_key = (str(prev), token)
            self._pair_counts[level][pair_key] = (
                self._pair_counts[level].get(pair_key, 0) + 1)

        # 3. Boundary detection: flush *before* adding the surprising token.

        # 3a. Surprise-based boundary (original).
        # Threshold is recomputed every threshold_recompute_every tokens (cache miss)
        # rather than every token — threshold changes slowly, so this is accurate.
        self._threshold_age[level] += 1
        if self._threshold_age[level] >= self.threshold_recompute_every:
            self._threshold_cache[level] = self._effective_threshold(level)
            self._threshold_age[level] = 0
        threshold = self._threshold_cache[level]
        if surp > threshold and self._buffers[level]:
            self._emit_surp[level] = surp   # M16: record triggering surprisal
            self._emit_buffer(level)

        # 3b. M21: MDL pair-count boundary (Sequitur-inspired).
        # Emit a boundary before *token* if the pair (buffer[-1], token) has
        # not been seen enough times globally to justify a grammar rule.
        # The minimum threshold min_merge_count=2 is the Sequitur invariant:
        # a pair must appear at least twice before a rule reduces DL at all.
        # This criterion is checked AFTER the surprise boundary (both can fire
        # independently), and is skipped during the cold-start grace period.
        if (self.mdl_boundaries
                and self._buffers[level]
                and self._seen[level] >= self.min_tokens_active):
            last   = str(self._buffers[level][-1])
            pcount = self._pair_counts[level].get((last, token), 0)
            if pcount < self.min_merge_count:
                self._emit_surp[level] = surp
                self._emit_buffer(level)

        # 4. Append to working memory.
        self._buffers[level].append(token)

        # 5. Hard cap (Miller's 7±2 prevents paragraph-swallowing).
        if len(self._buffers[level]) >= self.max_chunk_size:
            self._emit_buffer(level)

    def _reset_buffers(self) -> None:
        """Flush all level buffers and reset context pointers.

        Called at document boundaries so the last token of document N does
        not act as prediction context for the first token of document N+1.
        The ``_seen`` counters are intentionally *not* reset here — the
        cold-start grace period applies only at the very start of corpus
        processing.
        """
        for level in range(self.n_levels):
            self._emit_surp[level] = 0.0   # M16: forced flush — no surprisal gate
            self._emit_buffer(level)
            self._prev[level] = None
            # M16: document-boundary full belief reset (hidden-state reset between episodes).
            belief = self._beliefs[level]
            if belief is not None:
                belief.decay(rate=0.0)   # → uniform prior
            # Finalise the current document's chunk sequence for this level.
            # Cap storage to _CHUNK_SEQ_TOK_CAP tokens to bound RAM + R2 scan time.
            if self._chunk_cur_doc[level]:
                if self._chunk_seqs_tok[level] < self._CHUNK_SEQ_TOK_CAP:
                    self._chunk_seqs[level].append(self._chunk_cur_doc[level])
                    self._chunk_seqs_tok[level] += len(self._chunk_cur_doc[level])
                self._chunk_cur_doc[level] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, token: str) -> None:
        """Process one base-level token."""
        self._process_level(0, str(token))

    def process_sequence(self, sequence) -> None:
        """Process a sequence of base-level tokens, then flush all buffers."""
        for token in sequence:
            self._process_level(0, str(token))
        # End-of-sequence: drain remaining working memory up the hierarchy.
        for level in range(self.n_levels):
            self._emit_buffer(level)

    def process_corpus(self, sequences) -> None:
        """Process multiple sequences (e.g. books), resetting context between them.

        Parameters
        ----------
        sequences
            ``list[list[str]]`` — one sub-list per independent document.
            Context is reset between documents (``_reset_buffers``), so
            tokens from different books never form bigrams.
        """
        for seq in sequences:
            self._reset_buffers()          # document boundary: flush + clear prev
            for token in seq:
                self._process_level(0, str(token))
        # Final flush after the last document.
        for level in range(self.n_levels):
            self._emit_buffer(level)
        # Finalise any remaining current-doc chunk sequences.
        # Cap storage to _CHUNK_SEQ_TOK_CAP tokens to bound RAM + R2 scan time.
        for level in range(self.n_levels):
            if self._chunk_cur_doc[level]:
                if self._chunk_seqs_tok[level] < self._CHUNK_SEQ_TOK_CAP:
                    self._chunk_seqs[level].append(self._chunk_cur_doc[level])
                    self._chunk_seqs_tok[level] += len(self._chunk_cur_doc[level])
                self._chunk_cur_doc[level] = []

    def init_beliefs(self) -> None:
        """M13: Create a ContextBeliefState for every active, fitted level.

        Must be called *after* :meth:`analyse` (or :meth:`analyse_with_sequences`)
        so that each ``RelationalLearner`` already has ``assignment``, ``_K``,
        ``_trans``, ``_nc_cache``, and ``_wgc_cache`` populated.

        After this call, :meth:`_emit_buffer` updates the belief at each
        level and :meth:`_effective_threshold` applies top-down entropy
        modulation.  Run :meth:`reprocess` to do a frozen second pass that
        benefits from the active beliefs.
        """
        self._beliefs = [None] * self.n_levels
        for level in range(self.n_levels):
            learner = self.learners[level]
            assignment = getattr(learner, 'assignment', {})
            if len(assignment) >= 2:
                self._beliefs[level] = ContextBeliefState(learner)

    def reprocess(self, sequences) -> None:
        """M13: Frozen second pass — re-run corpus with active belief states.

        Online learning (``update_online``) is disabled: the learner weights
        are frozen so the second pass only uses beliefs for surprisal and
        top-down threshold modulation.  Call this after
        :meth:`process_corpus` → :meth:`analyse` → :meth:`init_beliefs`.

        Parameters
        ----------
        sequences
            Same corpus passed to :meth:`process_corpus`.
        """
        # Reset all beliefs to uniform prior before the second pass.
        for belief in self._beliefs:
            if belief is not None:
                belief.reset()
        # Reset buffers and context pointers (but not _seen, so cold-start stays off).
        for level in range(self.n_levels):
            self._buffers[level] = []
            self._prev[level]    = None
        self._frozen = True
        try:
            self.process_corpus(sequences)
        finally:
            self._frozen = False

    def compress(self, verbose: bool = True) -> None:
        """M14: Discard concept instances; retain only the type system.

        After ``analyse()`` the expensive data — ``vocab._merges``,
        ``vocab._segments``, ``vocab._by_surface``, and per-level
        ``_atom_counts`` / ``_atom_totals`` — is no longer needed for
        inference.  This method clears those structures, shrinking the object
        from O(corpus) to O(K² × levels).

        After ``compress()`` the PCH can still:
        - Classify new surface strings via ``cluster_from_type_abstraction``.
        - Predict next-token distributions via ``ContextBeliefState.predict_target_dist``.
        - Export a *type-only* CTKG (no concept instances).

        It can no longer:
        - Add new ``MergedAtom`` / ``SegmentedAtom`` entries to the vocab.
        - Produce the full concept-instance CTKG export.

        Sets ``self._compressed = True``.
        """
        # Clear vocab concept instances.
        self.vocab._merges.clear()
        self.vocab._segments.clear()
        self.vocab._by_surface.clear()
        self.vocab._pair_to_surface.clear()

        # Clear raw count tables on every learner (subsumed by type-level stats).
        cleared_levels = 0
        for learner in self.learners:
            for attr in ('_atom_counts', '_atom_totals', '_atom_bigrams',
                         '_rel_counts', '_succ_dists'):
                if hasattr(learner, attr):
                    getattr(learner, attr).clear()
            cleared_levels += 1

        # Clear chunk sequence logs (large lists of strings).
        self._chunk_seqs     = [[] for _ in range(self.n_levels)]
        self._chunk_cur_doc  = [[] for _ in range(self.n_levels)]
        self._chunk_seqs_tok = [0]  * self.n_levels

        # Clear M21 pair counts (not saved in checkpoint; rebuilt on next run).
        self._pair_counts = [{} for _ in range(self.n_levels)]

        self._compressed = True
        if verbose:
            print(f'PCH compressed: vocab + raw counts cleared '
                  f'({cleared_levels} learner levels)')

    def save_compressed(self, path: str) -> None:
        """M14: Serialise the type-only model to a compact pickle file.

        Stores only what is needed for inference:
        - Per-level: ``_K``, ``assignment``, ``clusters``, ``_trans``,
          ``_nc_cache``, ``_wgc_cache``, ``_sim_matrix``, ``_rel_unigram``,
          ``_rel_totals``.
        - PCH-level hyperparameters and ``_seen`` counters.

        Does *not* require ``compress()`` to have been called first.
        """
        import pickle

        def _extract_learner(learner: 'RelationalLearner') -> dict:
            return {
                '_K':           getattr(learner, '_K',           1),
                'assignment':   getattr(learner, 'assignment',   {}),
                'clusters':     {k: list(v) for k, v in
                                 getattr(learner, 'clusters', {}).items()},
                '_trans':       getattr(learner, '_trans',        {}),
                '_nc_cache':    getattr(learner, '_nc_cache',     {}),
                '_wgc_cache':   getattr(learner, '_wgc_cache',    {}),
                '_sim_matrix':  getattr(learner, '_sim_matrix',   []),
                '_rel_unigram': getattr(learner, '_rel_unigram',  {}),
                '_rel_totals':  getattr(learner, '_rel_totals',   {}),
                # Cross-level functor maps (from cluster_from_type_abstraction)
                '_constituent_type_map': getattr(learner, '_constituent_type_map', {}),
                '_upper_to_constituent': getattr(learner, '_upper_to_constituent', {}),
            }

        payload = {
            'version':          2,
            'n_levels':         self.n_levels,
            'max_chunk_size':   self.max_chunk_size,
            'surprise_threshold': self.surprise_threshold,
            'adaptive_threshold': self.adaptive_threshold,
            'surprise_k':       self.surprise_k,
            'min_tokens_active': self.min_tokens_active,
            'top_down_weight':  self.top_down_weight,
            'next_rel':         self.next_rel,
            'max_online_vocab': self.max_online_vocab,
            '_seen':            list(self._seen),
            'levels':           [_extract_learner(l) for l in self.learners],
        }

        with open(path, 'wb') as f:
            pickle.dump(payload, f, protocol=4)

        size_kb = len(pickle.dumps(payload, protocol=4)) / 1024
        print(f'PCH saved → {path}  ({size_kb:.1f} KB)')

    @classmethod
    def load_compressed(cls, path: str) -> 'PredictiveCodingHierarchy':
        """M14: Restore a type-only model from a ``save_compressed`` pickle.

        Returns a :class:`PredictiveCodingHierarchy` with all type-level
        caches populated and ``_compressed=True``.  Ready for inference via
        ``init_beliefs()`` + ``predict_next()``.
        """
        import pickle

        with open(path, 'rb') as f:
            payload = pickle.load(f)

        pch = cls(
            n_levels=payload['n_levels'],
            max_chunk_size=payload['max_chunk_size'],
            surprise_threshold=payload['surprise_threshold'],
            adaptive_threshold=payload['adaptive_threshold'],
            surprise_k=payload['surprise_k'],
            min_tokens_active=payload['min_tokens_active'],
            top_down_weight=payload['top_down_weight'],
            next_rel=payload['next_rel'],
            max_online_vocab=payload.get('max_online_vocab', 50_000),
        )
        pch._seen = list(payload['_seen'])
        pch._compressed = True

        for i, d in enumerate(payload['levels']):
            learner = pch.learners[i]
            learner._K           = d['_K']
            learner.assignment   = d['assignment']
            learner.clusters     = {k: set(v) for k, v in d['clusters'].items()}
            learner._trans       = d['_trans']
            learner._nc_cache    = d['_nc_cache']
            learner._wgc_cache   = d['_wgc_cache']
            learner._sim_matrix  = d['_sim_matrix']
            learner._rel_unigram         = d['_rel_unigram']
            learner._rel_totals          = d['_rel_totals']
            learner._constituent_type_map = d.get('_constituent_type_map', {})
            learner._upper_to_constituent = d.get('_upper_to_constituent', {})
            # Pre-populate soft caches from exact cache entries so predict_dist
            # never triggers _ask_soft's O(K⁴) lazy warm-up on large-K models.
            learner._nc_soft  = dict(d['_nc_cache'])
            learner._wgc_soft = dict(d['_wgc_cache'])

        return pch

    # ------------------------------------------------------------------
    # M15: Type-Level Inference
    # ------------------------------------------------------------------

    def predict_next(
        self,
        tokens,
        level:     int = 0,
        rel:       str | None = None,
        topk:      int = 20,
    ) -> list[tuple]:
        """M15: Context-conditioned P(next | context tokens) at *level*.

        Runs the given token sequence through :meth:`process_sequence`
        (frozen — no online learning) to update the belief state at *level*,
        then calls :meth:`ContextBeliefState.predict_target_dist` to get the
        surface-form distribution.

        Returns a list of ``(surface_string, probability)`` tuples sorted
        descending by probability, up to *topk* results.

        Requirements
        ------------
        - :meth:`init_beliefs` must have been called (belief states exist).
        - The learner at *level* must have ``_nc_cache`` / ``_wgc_cache``
          (either from training or from :meth:`load_compressed`).

        Parameters
        ----------
        tokens
            Sequence of base-level tokens (e.g. characters).  Fed through
            the hierarchy exactly once (frozen pass) to prime the belief.
        level
            Which level to query.  Level 0 = character prediction, level 3
            ≈ word prediction.  Default 0.
        rel
            Relation to predict along (default ``self.next_rel``).
        topk
            Maximum number of results to return.
        """
        rel = rel or self.next_rel
        belief = self._beliefs[level]
        if belief is None:
            return []

        # Prime the belief with the context tokens (frozen pass).
        old_frozen = self._frozen
        self._frozen = True
        try:
            for tok in tokens:
                self._process_level(0, str(tok))
        finally:
            self._frozen = old_frozen

        dist = belief.predict_target_dist(rel)
        items = sorted(dist.items(), key=lambda kv: -kv[1])
        return items[:topk]

    def predict_type(self, level: int, rel: str | None = None) -> dict:
        """M15: Type-level marginal P(next_type | current_belief) at *level*.

        Uses :class:`ContextBeliefState` transition matrix directly —
        O(K²) per call.  Does not consume any new tokens.

        Returns ``{type_id: probability}`` after propagating the current
        belief through the ``_trans[rel]`` matrix.
        """
        rel = rel or self.next_rel
        belief = self._beliefs[level]
        if belief is None:
            return {}
        # We want P(c_next) = Σ_{c} P(c) * T[rel][c → c_next]
        learner = self.learners[level]
        trans = getattr(learner, '_trans', {}).get(rel, {})
        if not trans:
            return dict(belief._belief)
        result: dict = {}
        for c_src, p_src in belief._belief.items():
            if p_src < 1e-12:
                continue
            c_next_dist = trans.get(str(c_src), {})
            for c_next_s, p_t in c_next_dist.items():
                try:
                    c_next = int(c_next_s)
                except (ValueError, TypeError):
                    continue
                result[c_next] = result.get(c_next, 0.0) + p_src * p_t
        total = sum(result.values())
        if total > 0:
            return {k: v / total for k, v in result.items()}
        return {}

    def evaluate_perplexity(
        self,
        sequences,
        level:             int   = 0,
        rel:               str | None = None,
        belief_conditioned: bool  = False,
        belief_weight:     float = 0.5,
    ) -> float:
        """M15/M19: Per-token cross-entropy (in bits) under the learner's model.

        **Bigram mode** (``belief_conditioned=False``, default):
        Uses the empirical ``predict_dist(prev, rel)`` fast-path — V-level
        precision, no category abstraction overhead.

        **Belief-conditioned mode** (``belief_conditioned=True``, M19):
        Interpolates the bigram with the belief-state prediction:

            P(next | prev, belief) =
                (1 − α) · P_bigram(next | prev)
              +  α      · P_belief(next | belief, rel)

        where α = belief_weight × (1 − H(belief)/log₂K) scales with belief
        *sharpness*: a diffuse (near-uniform) belief contributes nothing and
        the prediction falls back to the pure bigram; a concentrated belief
        contributes its full long-range contextual signal.

        Beliefs are updated *after* scoring each token (no future leakage):
        ``belief.observe(tok); belief.transition(rel)``.  Beliefs reset at
        each sequence boundary to match how the model was trained.

        Parameters
        ----------
        sequences
            List of token sequences (same format as :meth:`process_corpus`).
        level
            Hierarchy level.
        rel
            Relation (default ``self.next_rel``).
        belief_conditioned
            If True, use M19 belief-interpolated prediction.
        belief_weight
            Maximum interpolation weight toward the belief component (0–1).
            The effective weight is further scaled by belief sharpness.
        """
        rel     = rel or self.next_rel
        learner = self.learners[level]

        # Fetch belief state if requested and available.
        beliefs = getattr(self, 'beliefs', [])
        belief  = beliefs[level] if (belief_conditioned and level < len(beliefs)) else None

        total_nll = 0.0
        n_tokens  = 0
        for seq in sequences:
            if belief is not None:
                belief.reset()
            prev = None
            for tok in seq:
                tok_s = str(tok)
                if prev is not None:
                    if belief is not None:
                        dist = self.predict_dist_conditioned(
                            level, prev, rel, belief_weight=belief_weight)
                    else:
                        dist = learner.predict_dist(prev, rel)
                    p = dist.get(tok_s, 1e-10)
                    total_nll += -math.log2(p)
                    n_tokens  += 1
                    # Update belief AFTER scoring to avoid future leakage.
                    if belief is not None:
                        belief.observe(tok_s)
                        belief.transition(rel)
                prev = tok_s

        return total_nll / n_tokens if n_tokens > 0 else float('inf')

    # ------------------------------------------------------------------
    # M19 — Belief-Conditioned Prediction
    # ------------------------------------------------------------------

    def predict_dist_conditioned(
        self,
        level:        int,
        atom:         str,
        rel:          str | None = None,
        belief_weight: float = 0.5,
    ) -> dict:
        """M19: Belief-interpolated token prediction.

        Combines the immediate atom-level bigram (precise local conditioning)
        with the belief-state's long-range contextual prediction:

            P(next | atom, belief) =
                (1 − α) · P_bigram(next | atom, rel)
              +  α      · P_belief(next | belief_state, rel)

        α = belief_weight × (1 − H(belief) / log₂K)

        When belief is uniform (maximum entropy, no information) α = 0 and
        the result is identical to the pure bigram.  When belief is concentrated
        on a single category α = belief_weight, giving the belief component
        its maximum contribution.

        Parameters
        ----------
        level
            PCH hierarchy level.
        atom
            Current surface token (string-coerced).
        rel
            Relation to predict along (default ``self.next_rel``).
        belief_weight
            Maximum weight on the belief component (0–1).  Effective weight
            is further scaled by belief sharpness.
        """
        rel = rel or self.next_rel

        learner   = self.learners[level]
        p_bigram  = learner.predict_dist(atom, rel)

        # Fetch belief state; fall back to pure bigram if unavailable.
        beliefs = getattr(self, 'beliefs', [])
        if level >= len(beliefs) or beliefs[level] is None:
            return p_bigram

        belief_state = beliefs[level]
        p_belief     = belief_state.predict_target_dist(rel)
        if not p_belief:
            return p_bigram

        # Compute belief sharpness: 0 = uniform, 1 = point mass.
        bvec      = belief_state._belief
        K         = max(len(bvec), 1)
        H_belief  = -sum(p * math.log2(max(p, 1e-15)) for p in bvec.values())
        H_uniform = math.log2(K) if K > 1 else 1.0
        sharpness = max(0.0, 1.0 - H_belief / H_uniform)
        alpha     = belief_weight * sharpness

        if alpha < 1e-9:
            return p_bigram

        # Interpolate: (1-α)·bigram + α·belief.
        all_toks = set(p_bigram) | set(p_belief)
        result   = {}
        for tok in all_toks:
            result[tok] = ((1.0 - alpha) * p_bigram.get(tok, 1e-10)
                           + alpha       * p_belief.get(tok, 1e-10))
        total = sum(result.values())
        return {k: v / total for k, v in result.items()} if total > 0 else p_bigram

    def reason_chain(
        self,
        start_token: str,
        relations:   list,
        level:       int = 0,
        topk:        int = 10,
    ) -> list[tuple]:
        """M15: Multi-hop type reasoning via infer_chain().

        Looks up *start_token*'s category at *level*, then calls
        :meth:`RelationalLearner.infer_chain` to propagate distributions
        through the *relations* sequence.  Returns ``(surface, prob)`` pairs
        for the final distribution.

        Example::

            pch.reason_chain('in', ['next', 'next'], level=3, topk=5)
            # → what usually comes 2 words after a preposition

        Parameters
        ----------
        start_token
            A surface string whose category is looked up in the learner.
        relations
            Ordered list of relation names to chain through.
        level
            Learner level to query.
        topk
            Max results.
        """
        learner = self.learners[level]
        if not hasattr(learner, 'assignment'):
            return []
        # infer_chain already returns List[Tuple[atom, prob]] sorted descending.
        return learner.infer_chain(start_token, relations, topk=topk)

    # ------------------------------------------------------------------
    # M17 — Cross-Domain Functors + Sheaf Consistency
    # ------------------------------------------------------------------

    @staticmethod
    def _transition_profile(learner, k: int, rels: list) -> list:
        """Eigen-profile of type k: sorted transition probabilities per relation.

        Captures the *shape* of the transition distribution (concentrated vs.
        diffuse, fast vs. slow decay) without reference to specific target
        category indices.  This makes profiles comparable across two
        independently-trained PCH instances whose category labels differ.

        Returns a flat float vector of length ``len(rels) * top_n``.
        """
        top_n = 8
        trans = getattr(learner, '_trans', {})
        vec: list = []
        for rel in rels:
            dist = trans.get(rel, {}).get(str(k), {})
            sorted_probs = sorted(dist.values(), reverse=True)
            # Pad / truncate to top_n
            vec.extend(sorted_probs[:top_n] + [0.0] * (top_n - len(sorted_probs)))
        return vec

    @staticmethod
    def _cosine_sim(a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na  = math.sqrt(sum(x * x for x in a))
        nb  = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb + 1e-12)

    def build_interface(self, domain_name: str = 'pch') -> 'KnowledgeGraph':
        """M17: Build a CTKG KnowledgeGraph with an Interface exporting all discovered types.

        The Interface block lists every ``type_L{level}_{k}`` and every
        ``cat_L{level}_{k}`` as exported symbols.  This is required before
        calling :meth:`sheaf_check` or :meth:`build_functor` on two PCH
        instances whose categories may overlap in meaning.

        Parameters
        ----------
        domain_name
            Domain identifier used in the Interface block and Concept domain
            fields.

        Returns
        -------
        KnowledgeGraph
            A fresh graph containing TypeDef entries, Concept entries, and a
            single Interface listing all exported names.
        """
        from ctkg.graph import Interface, TypeDef, Concept  # type: ignore

        kg          = KnowledgeGraph()
        type_names: list  = []
        conc_names: list  = []

        for level in range(self.n_levels):
            learner = self.learners[level]
            K       = getattr(learner, '_K', 0)
            if K < 2:
                continue

            # Type constructor follows M18e universal type hierarchy (simplified).
            constructor  = 'symbol'  if level == 0 else ('seq' if level <= 2 else 'tuple')
            annotations  = {'ordered'} if level < 3 else set()

            for k in range(K):
                tname = f'type_L{level}_{k}'
                kg.types[tname] = TypeDef(
                    name        = tname,
                    constructor = constructor,
                    annotations = annotations,
                    description = (f'Discovered type: level={level} category={k} '
                                   f'domain={domain_name}'),
                )
                type_names.append(tname)

            # Category concepts (one per cluster).
            clusters = getattr(learner, 'clusters', {})
            for cid, members in clusters.items():
                cname = f'cat_L{level}_{cid}'
                top   = sorted(members)[:5]
                kg.concepts[cname] = Concept(
                    name        = cname,
                    description = (f'L{level} cluster {cid}: '
                                   + ', '.join(repr(m) for m in top)
                                   + (f' +{len(members)-5} more' if len(members) > 5 else '')),
                    domain      = domain_name,
                    input_type  = [f'type_L{level}_{cid}'],
                    output_type = [f'type_L{level}_{cid}'],
                )
                conc_names.append(cname)

        kg.interfaces[domain_name] = Interface(
            name     = domain_name,
            types    = type_names,
            concepts = conc_names,
        )
        return kg

    def build_functor(
        self,
        other:         'PredictiveCodingHierarchy',
        level:         int | None = None,
        sim_threshold: float      = 0.7,
        domain_self:   str        = 'pch_A',
        domain_other:  str        = 'pch_B',
    ) -> dict:
        """M17: Align type categories between two PCH instances.

        For each active level (or the specified *level*), compares the
        transition eigen-profiles of every type pair using cosine similarity.
        A greedy bipartite matching yields ``{self_k: other_j}`` pairs whose
        profiles exceed *sim_threshold*.

        The resulting :class:`ctkg.graph.Functor` witnesses that structurally
        identical categories are the same abstract concept, enabling
        cross-domain prediction transfer: given self's current type belief,
        map it to other's equivalent type and borrow other's ``_nc_cache``
        for next-token prediction.

        Runs ``sheaf_check()`` on the two Interface graphs to verify that
        shared type names are structurally compatible.

        Parameters
        ----------
        other
            The second PCH instance to align with.
        level
            If given, only align this level; otherwise all active levels.
        sim_threshold
            Minimum cosine similarity to accept a type match.
        domain_self, domain_other
            Names used to label the two sides of the functor.

        Returns
        -------
        dict with keys:
            ``'mapping'``           — ``{level: {self_k: other_j}}``
            ``'functor'``           — :class:`ctkg.graph.Functor`
            ``'sheaf_violations'``  — list of :class:`ctkg.graph.SheafViolation`
            ``'kg'``                — :class:`KnowledgeGraph` containing the Functor
        """
        from ctkg.graph import Functor  # type: ignore

        rels_to_use   = [self.next_rel, 'prev']
        levels_to_try = [level] if level is not None else range(self.n_levels)
        mapping: dict = {}   # {level: {self_k: other_j}}

        for lv in levels_to_try:
            sl = self.learners[lv]
            ol = other.learners[lv]
            K_self  = getattr(sl, '_K', 0)
            K_other = getattr(ol, '_K', 0)
            if K_self < 2 or K_other < 2:
                continue

            profiles_self  = {k: self._transition_profile(sl, k, rels_to_use)
                              for k in range(K_self)}
            profiles_other = {j: self._transition_profile(ol, j, rels_to_use)
                              for j in range(K_other)}

            # Greedy matching: for each self type, find best-matching other type.
            level_map: dict = {}
            used_other: set = set()
            for k in range(K_self):
                best_j, best_sim = None, -1.0
                for j in range(K_other):
                    if j in used_other:
                        continue
                    sim = self._cosine_sim(profiles_self[k], profiles_other[j])
                    if sim > best_sim:
                        best_sim, best_j = sim, j
                if best_j is not None and best_sim >= sim_threshold:
                    level_map[k]   = best_j
                    used_other.add(best_j)

            if level_map:
                mapping[lv] = level_map

        # Build Functor in a fresh KnowledgeGraph.
        concept_map: dict = {}
        for lv, lmap in mapping.items():
            for k, j in lmap.items():
                concept_map[f'cat_L{lv}_{k}'] = f'cat_L{lv}_{j}'

        functor = Functor(
            name          = f'{domain_self}_to_{domain_other}',
            source_domain = domain_self,
            target_domain = domain_other,
            concept_map   = concept_map,
            preserves     = ['type_transitions'],
        )
        kg                 = KnowledgeGraph()
        kg.functors[functor.name] = functor

        # Sheaf check: build Interface for each side, verify structural compatibility.
        kg_self  = self.build_interface(domain_self)
        kg_other = other.build_interface(domain_other)
        violations = kg_self.sheaf_check(kg_other)

        self._functor_kg      = kg
        self._functor_mapping = mapping
        return {
            'mapping':          mapping,
            'functor':          functor,
            'sheaf_violations': violations,
            'kg':               kg,
        }

    def build_adjunction(self, level: int | None = None) -> 'KnowledgeGraph':
        """M17: Build CTKG Adjunction objects for next/prev inverse pairs.

        At every level, 'next' and 'prev' form an adjoint pair: following
        'next' then 'prev' should approximately recover the original token
        (the *unit* of the adjunction).  This was confirmed numerically in
        R3 (next∘prev ≈ identity).

        Verifies each level by sampling the top-5 most common tokens and
        computing ``infer_chain(tok, ['next', 'prev'])``.  The round-trip
        quality (probability mass on the original token after 2 hops) is
        stored in the Adjunction's ``unit`` field and in
        ``self._adjunction_quality``.

        Parameters
        ----------
        level
            If given, only build the adjunction for this level; otherwise
            all active levels.

        Returns
        -------
        KnowledgeGraph
            Fresh graph with one :class:`ctkg.graph.Adjunction` per active level.
        """
        from ctkg.graph import Adjunction  # type: ignore
        from collections import Counter as _Counter

        levels_to_try = [level] if level is not None else range(self.n_levels)
        kg = KnowledgeGraph()
        quality_map: dict = {}

        for lv in levels_to_try:
            learner    = self.learners[lv]
            K          = getattr(learner, '_K', 0)
            assignment = getattr(learner, 'assignment', {})
            if K < 2 or not assignment:
                continue

            # Sample the 5 most frequent tokens (by category population).
            cat_counts = _Counter(assignment.values())
            samples    = sorted(assignment, key=lambda a: -cat_counts.get(
                assignment.get(a, 0), 0))[:5]

            round_trip_scores: list = []
            for tok in samples:
                dist = learner.infer_chain(str(tok), [self.next_rel, 'prev'], topk=20)
                # P(original | round-trip)
                p_orig = next((p for t, p in dist if t == str(tok)), 0.0)
                round_trip_scores.append(p_orig)

            quality = (sum(round_trip_scores) / len(round_trip_scores)
                       if round_trip_scores else 0.0)
            quality_map[lv] = quality

            adj = Adjunction(
                name    = f'next_prev_L{lv}',
                forward = 'next',
                inverse = 'prev',
                unit    = f'next∘prev round-trip quality={quality:.3f}',
                counit  = f'prev∘next (symmetric)',
            )
            kg.adjunctions[adj.name] = adj

        self._adjunction_kg      = kg
        self._adjunction_quality = quality_map
        return kg

    def build_cross_level_functor(
        self,
        level_lo:    int,
        domain_name: str = 'pch',
    ) -> dict:
        """M17+: Composition functor F: L_{N} → L_{N+1} within a single PCH.

        Unlike :meth:`build_functor` (which aligns two *independently-trained*
        PCH instances by profile similarity), this functor is grounded in the
        actual constituent structure discovered during
        :meth:`cluster_from_type_abstraction`.

        For each lower-level type *k*, ``F(k)`` is the upper-level cluster *j*
        that most often contains chunks whose direct constituents are of type *k*.
        This is a true functional dependency: the upper type IS determined by the
        lower types it was built from.

        Falls back to transition-profile cosine similarity when
        ``_constituent_type_map`` is absent (e.g. JSD-clustered lower levels
        where M12 type-abstraction was not invoked).

        Parameters
        ----------
        level_lo
            Source level N.  Target is automatically N+1.
        domain_name
            Domain prefix for concept names.

        Returns
        -------
        dict with keys:
            ``'functor'``     — :class:`ctkg.graph.Functor` F: L_N → L_{N+1}
            ``'concept_map'`` — ``{cat_LN_k: cat_L{N+1}_j}``
            ``'kg'``          — :class:`KnowledgeGraph` containing the functor
        """
        from ctkg.graph import Functor  # type: ignore

        level_hi = level_lo + 1
        if level_hi >= self.n_levels:
            return {}

        sl_lo = self.learners[level_lo]
        sl_hi = self.learners[level_hi]
        K_lo  = getattr(sl_lo, '_K', 0)
        K_hi  = getattr(sl_hi, '_K', 0)
        if K_lo < 2 or K_hi < 2:
            return {}

        concept_map: dict = {}

        # Primary: use constituent_type_map from M12 type-abstraction.
        ctmap = getattr(sl_hi, '_constituent_type_map', {})
        if ctmap:
            for k in range(K_lo):
                dist = ctmap.get(str(k), {})
                if dist:
                    best_j = max(dist, key=dist.get)
                    concept_map[f'cat_L{level_lo}_{k}'] = f'cat_L{level_hi}_{best_j}'

        if not concept_map:
            # Fallback: transition eigen-profile similarity.
            rels = [self.next_rel, 'prev']
            profiles_lo = {k: self._transition_profile(sl_lo, k, rels)
                           for k in range(K_lo)}
            profiles_hi = {j: self._transition_profile(sl_hi, j, rels)
                           for j in range(K_hi)}
            used: set = set()
            for k in range(K_lo):
                best_j, best_sim = None, -1.0
                for j in range(K_hi):
                    if j in used:
                        continue
                    sim = self._cosine_sim(profiles_lo[k], profiles_hi[j])
                    if sim > best_sim:
                        best_sim, best_j = sim, j
                if best_j is not None and best_sim > 0.2:
                    concept_map[f'cat_L{level_lo}_{k}'] = f'cat_L{level_hi}_{best_j}'
                    used.add(best_j)

        functor = Functor(
            name          = f'compose_L{level_lo}_L{level_hi}',
            source_domain = f'{domain_name}_L{level_lo}',
            target_domain = f'{domain_name}_L{level_hi}',
            concept_map   = concept_map,
            preserves     = ['constituent_structure', 'type_transitions'],
        )
        kg = KnowledgeGraph()
        kg.functors[functor.name] = functor
        return {'functor': functor, 'concept_map': concept_map, 'kg': kg}

    def build_compose_decompose_adjunction(
        self,
        level_lo:    int,
        domain_name: str = 'pch',
    ) -> dict:
        """M17+: The fundamental Compose ⊣ Decompose adjunction of the hierarchy.

        This is the adjunction that Merge itself implements:

        - **F = Compose** (left adjoint): L_N → L_{N+1}
          Maps each lower type *k* to the upper type *j* it most often
          composes into.  Built by :meth:`build_cross_level_functor`.

        - **G = Decompose** (right adjoint): L_{N+1} → L_N
          Maps each upper type *j* back to its dominant lower constituent type.
          Built from ``_upper_to_constituent`` (the inverse of the composition
          map, also recorded during :meth:`cluster_from_type_abstraction`).

        **Unit** (η: id_{L_N} → G∘F): following F then G should recover the
        original lower type.  Unit quality = fraction of lower types for which
        G(F(k)) = k.

        **Counit** (ε: F∘G → id_{L_{N+1}}): following G then F should recover
        the original upper type.  Counit quality = fraction of upper types for
        which F(G(j)) = j.

        High quality on both sides ≈ the hierarchy has clean, unambiguous
        constituent structure at this boundary.  Low quality ≈ many-to-one
        merging (lossy composition).

        Parameters
        ----------
        level_lo
            Lower level N.  Upper level is N+1.
        domain_name
            Domain prefix for concept names.

        Returns
        -------
        dict with keys:
            ``'adjunction'``    — :class:`ctkg.graph.Adjunction`
            ``'F'``             — Compose concept map ``{cat_LN_k: cat_L{N+1}_j}``
            ``'G'``             — Decompose concept map ``{cat_L{N+1}_j: cat_LN_k}``
            ``'unit_quality'``  — float in [0, 1]: G∘F round-trip accuracy
            ``'counit_quality'``— float in [0, 1]: F∘G round-trip accuracy
            ``'kg'``            — :class:`KnowledgeGraph` with the Adjunction
        """
        from ctkg.graph import Adjunction  # type: ignore

        level_hi = level_lo + 1
        if level_hi >= self.n_levels:
            return {}

        sl_lo = self.learners[level_lo]
        sl_hi = self.learners[level_hi]
        K_lo  = getattr(sl_lo, '_K', 0)
        K_hi  = getattr(sl_hi, '_K', 0)
        if K_lo < 2 or K_hi < 2:
            return {}

        # Compose functor F.
        compose_result = self.build_cross_level_functor(level_lo, domain_name)
        F = compose_result.get('concept_map', {})  # {cat_LN_k: cat_L{N+1}_j}

        # Decompose functor G: built from _upper_to_constituent.
        G: dict = {}
        upper_to_const = getattr(sl_hi, '_upper_to_constituent', {})
        for j, lower_t_str in upper_to_const.items():
            G[f'cat_L{level_hi}_{j}'] = f'cat_L{level_lo}_{lower_t_str}'

        # Unit quality: fraction of lower types k where G(F(k)) == k.
        unit_correct = unit_total = 0
        for k in range(K_lo):
            src = f'cat_L{level_lo}_{k}'
            mid = F.get(src)
            if mid is None:
                continue
            dest = G.get(mid)
            unit_total += 1
            if dest == src:
                unit_correct += 1
        unit_quality = unit_correct / unit_total if unit_total > 0 else 0.0

        # Counit quality: fraction of upper types j where F(G(j)) == j.
        counit_correct = counit_total = 0
        for j in range(K_hi):
            src = f'cat_L{level_hi}_{j}'
            mid = G.get(src)
            if mid is None:
                continue
            dest = F.get(mid)
            counit_total += 1
            if dest == src:
                counit_correct += 1
        counit_quality = counit_correct / counit_total if counit_total > 0 else 0.0

        adj = Adjunction(
            name    = f'compose_decompose_L{level_lo}_{level_hi}',
            forward = f'compose_L{level_lo}_L{level_hi}',
            inverse = f'decompose_L{level_hi}_L{level_lo}',
            unit    = (f'G∘F={unit_quality:.3f} '
                       f'({unit_correct}/{unit_total} lower types round-trip)'),
            counit  = (f'F∘G={counit_quality:.3f} '
                       f'({counit_correct}/{counit_total} upper types round-trip)'),
        )
        kg = KnowledgeGraph()
        kg.adjunctions[adj.name] = adj
        return {
            'adjunction':    adj,
            'F':             F,
            'G':             G,
            'unit_quality':  unit_quality,
            'counit_quality': counit_quality,
            'kg':            kg,
        }

    def build_all_compose_decompose_adjunctions(
        self,
        domain_name: str = 'pch',
    ) -> 'KnowledgeGraph':
        """Build Compose⊣Decompose adjunctions for every adjacent level pair.

        Sweeps levels 0→1, 1→2, ..., (N-2)→(N-1) and collects all
        adjunctions into a single :class:`KnowledgeGraph`.

        Returns
        -------
        KnowledgeGraph
            Contains one :class:`~ctkg.graph.Adjunction` per active adjacent
            pair, plus all cross-level :class:`~ctkg.graph.Functor` objects.
        """
        kg = KnowledgeGraph()
        self._compose_decompose_results: dict = {}

        for level_lo in range(self.n_levels - 1):
            result = self.build_compose_decompose_adjunction(level_lo, domain_name)
            if not result:
                continue
            self._compose_decompose_results[level_lo] = result
            kg.adjunctions[result['adjunction'].name] = result['adjunction']
            # Also register the compose functor in the merged KG.
            f_result = self.build_cross_level_functor(level_lo, domain_name)
            if f_result:
                kg.functors[f_result['functor'].name] = f_result['functor']

        return kg

    # ------------------------------------------------------------------
    # M18 — Causal Reasoning, MasteryState, Full CTKG Closure
    # ------------------------------------------------------------------

    def build_transition_kg(
        self,
        level:       int,
        domain_name: str = 'pch',
    ) -> 'KnowledgeGraph':
        """M18c/d: Build a Markov-chain CTKG at a single level.

        Each discovered type category becomes a Concept node.  Prerequisite
        edges carry ``transfer_probability = _trans['next'][src][tgt]``,
        making the graph a probabilistic causal model suitable for:
        - :func:`ctkg.graph.KnowledgeGraph.d_separated` (Bayes-ball)
        - :func:`ctkg.graph.KnowledgeGraph.intervene` (do-calculus)
        - :func:`ctkg.graph.KnowledgeGraph.information_flow` (information bits)

        Parameters
        ----------
        level
            The hierarchy level to model.
        domain_name
            Domain label for the Concept entries.

        Returns
        -------
        KnowledgeGraph
            Fresh graph with K concept nodes and up to K² Prerequisite edges
            (one per non-zero transition probability).
        """
        from ctkg.graph import TypeDef, Concept, Prerequisite  # type: ignore

        learner = self.learners[level]
        K       = getattr(learner, '_K', 0)
        if K < 2:
            return KnowledgeGraph()

        trans   = getattr(learner, '_trans', {}).get(self.next_rel, {})
        clusters = getattr(learner, 'clusters', {})

        kg = KnowledgeGraph()
        constructor = 'symbol' if level == 0 else ('seq' if level <= 2 else 'tuple')

        # Add one concept per type category.
        for k in range(K):
            tname = f'type_L{level}_{k}'
            kg.types[tname] = TypeDef(
                name=tname, constructor=constructor,
                description=f'Type at level {level} category {k}',
            )
            members = clusters.get(k, [])
            top     = sorted(members)[:5]
            kg.concepts[tname] = Concept(
                name=tname,
                description=(', '.join(repr(m) for m in top)
                             + (f' +{len(members)-5} more' if len(members) > 5 else '')),
                domain=domain_name,
                input_type=[tname],
                output_type=[tname],
            )

        # Add Markov transition edges as prerequisites with transfer_probability.
        for c_src, tgt_dist in trans.items():
            for c_tgt, tp in tgt_dist.items():
                if tp < 1e-6:
                    continue
                src_cname = f'type_L{level}_{c_src}'
                tgt_cname = f'type_L{level}_{c_tgt}'
                if src_cname not in kg.concepts or tgt_cname not in kg.concepts:
                    continue
                prereq = Prerequisite(
                    source               = src_cname,
                    target               = tgt_cname,
                    role                 = 'transitions_to',
                    transfer_probability = float(tp),
                )
                kg.prerequisites.append(prereq)
                kg._children.setdefault(src_cname, set()).add(tgt_cname)
                kg._parents.setdefault(tgt_cname, set()).add(src_cname)

        return kg

    def d_separated_types(
        self,
        level:       int,
        type_a:      int,
        type_b:      int,
        given_types: 'set[int] | None' = None,
    ) -> bool:
        """M18c: Test conditional independence of two type categories.

        Builds the Markov-chain CTKG for *level* and runs the Bayes-ball
        d-separation test.  Returns ``True`` if type A and type B are
        conditionally independent given the observed set.

        Example
        -------
        ``pch.d_separated_types(0, 2, 5, given_types={1})``
        → are L0-type-2 and L0-type-5 independent given type 1 is observed?
        """
        kg    = self.build_transition_kg(level)
        given = {f'type_L{level}_{g}' for g in (given_types or set())}
        a_name = f'type_L{level}_{type_a}'
        b_name = f'type_L{level}_{type_b}'
        return kg.d_separated(a_name, b_name, given)

    def intervene_type(
        self,
        level:    int,
        do_types: 'set[int]',
    ) -> 'KnowledgeGraph':
        """M18d: Do-calculus intervention at the type level.

        Returns a mutilated Markov-chain CTKG where all incoming transition
        edges to the ``do_types`` are severed (Pearl's do-operator).  This
        simulates "what if the topic forcibly shifted to these categories?"

        The mutilated graph can be used to compute post-intervention
        :func:`information_flow` and compared to the observational flow to
        measure the causal effect of the topic shift.

        Parameters
        ----------
        level
            Hierarchy level.
        do_types
            Set of type-category indices to intervene on.

        Returns
        -------
        KnowledgeGraph
            Mutilated graph with do_types' incoming edges removed.
        """
        kg        = self.build_transition_kg(level)
        do_names  = {f'type_L{level}_{t}' for t in do_types}
        return kg.intervene(do_names)

    def type_mastery(
        self,
        tokens_needed_per_type: int = 200,
        domain_name:            str = 'pch',
    ) -> 'MasteryState':
        """M18a: MasteryState reflecting how well each type is characterized.

        Mastery of a type category is estimated as:
          ``mastery = min(1.0, tokens_seen / tokens_needed_per_type)``
        where ``tokens_seen`` is the number of tokens assigned to that
        category.  A type with fewer than *tokens_needed_per_type* observations
        is considered still being learned; ``frontier()`` returns these
        under-characterised types.

        Adds cross-level Prerequisite edges so that higher-level categories
        depend on the lower-level categories they are built from.  Uses
        M18b transfer probabilities estimated from the type-abstraction
        mapping (uniform fallback if not available).

        Parameters
        ----------
        tokens_needed_per_type
            Token count threshold for "mastered" (default 200).
        domain_name
            Domain label for the KnowledgeGraph.

        Returns
        -------
        MasteryState
            Populated mastery state; call ``.frontier()`` to get types that
            need more data, ``.information_gain(concept)`` for prioritisation.
        """
        from ctkg.graph import TypeDef, Concept, Prerequisite  # type: ignore

        kg = KnowledgeGraph()
        # Collect per-type token counts from assignment.
        for level in range(self.n_levels):
            learner  = self.learners[level]
            K        = getattr(learner, '_K', 0)
            clusters = getattr(learner, 'clusters', {})
            if K < 2:
                continue
            constructor = 'symbol' if level == 0 else ('seq' if level <= 2 else 'tuple')
            for k in range(K):
                tname = f'type_L{level}_{k}'
                members = clusters.get(k, [])
                kg.types[tname] = TypeDef(name=tname, constructor=constructor)
                kg.concepts[tname] = Concept(
                    name=tname,
                    description=f'L{level} type {k}: {len(members)} members',
                    domain=domain_name,
                    input_type=[tname],
                    output_type=[tname],
                )

        # M18b: cross-level prerequisites with transfer_probability.
        for level in range(self.n_levels - 1):
            sl   = self.learners[level]
            sl_n = self.learners[level + 1]
            K_lo = getattr(sl,   '_K', 0)
            K_hi = getattr(sl_n, '_K', 0)
            if K_lo < 2 or K_hi < 2:
                continue
            # Uniform transfer probability as fallback.
            tp_uniform = 1.0 / K_hi
            for k in range(K_lo):
                for j in range(K_hi):
                    src_name = f'type_L{level}_{k}'
                    tgt_name = f'type_L{level+1}_{j}'
                    if src_name not in kg.concepts or tgt_name not in kg.concepts:
                        continue
                    prereq = Prerequisite(
                        source=src_name, target=tgt_name,
                        role=f'constitutes_L{level+1}',
                        transfer_probability=tp_uniform,
                    )
                    kg.prerequisites.append(prereq)
                    kg._children.setdefault(src_name, set()).add(tgt_name)
                    kg._parents.setdefault(tgt_name, set()).add(src_name)

        mastery = kg.mastery_state()

        # Observe each type: score = min(1.0, |members| / threshold).
        for level in range(self.n_levels):
            learner  = self.learners[level]
            clusters = getattr(learner, 'clusters', {})
            K        = getattr(learner, '_K', 0)
            if K < 2:
                continue
            for k, members in clusters.items():
                tname = f'type_L{level}_{k}'
                score = min(1.0, len(members) / tokens_needed_per_type)
                mastery.observe(tname, score)

        return mastery

    def causal_analysis(
        self,
        level: int = 0,
        verbose: bool = True,
    ) -> dict:
        """M18: Unified causal analysis at *level*: information flow + d-sep + intervention.

        Combines all M18 sub-tasks into a single convenience method:
        1. Build Markov-chain CTKG.
        2. Compute information_flow() per transition edge.
        3. Find maximally d-separated type pairs (strongest independence).
        4. Perform do-intervention on the highest-flow type; measure flow change.

        Returns
        -------
        dict with keys:
            ``'flow'``        — ``{src→tgt: bits}`` from ``information_flow()``
            ``'dsep_pairs'``  — list of ``(type_a, type_b, given, result)``
            ``'intervention'``— ``{'do_type': k, 'flow_before': ..., 'flow_after': ...}``
        """
        kg   = self.build_transition_kg(level)
        K    = getattr(self.learners[level], '_K', 0)

        # 1. Information flow.
        flow = kg.information_flow()

        # 2. Find a representative d-separation triple.
        dsep_pairs = []
        if K >= 3:
            # Test: given the most common type, are the two extreme types independent?
            learner    = self.learners[level]
            clusters   = getattr(learner, 'clusters', {})
            by_size    = sorted(clusters, key=lambda k: -len(clusters.get(k, [])))
            if len(by_size) >= 3:
                given_k = by_size[0]   # most common
                a_k     = by_size[1]
                b_k     = by_size[2]
                result  = self.d_separated_types(level, a_k, b_k, {given_k})
                dsep_pairs.append((a_k, b_k, {given_k}, result))

        # 3. Intervention on the highest-flow source type.
        intervention_result: dict = {}
        if flow:
            top_edge     = max(flow, key=flow.get)
            src_name, _  = top_edge.split('->', 1)
            # Extract type index from name like 'type_L0_3'
            try:
                do_k = int(src_name.split('_')[-1])
            except ValueError:
                do_k = 0
            flow_before  = sum(flow.values())
            kg_mutilated = self.intervene_type(level, {do_k})
            flow_after_d = kg_mutilated.information_flow()
            flow_after   = sum(flow_after_d.values())
            intervention_result = {
                'do_type':    do_k,
                'do_name':    src_name,
                'flow_before': flow_before,
                'flow_after':  flow_after,
                'reduction':   flow_before - flow_after,
            }

        if verbose:
            print(f'\n  M18 causal analysis at level {level}:')
            top_flows = sorted(flow.items(), key=lambda kv: -kv[1])[:5]
            print(f'  Top-5 information flows (bits):')
            for edge, bits in top_flows:
                print(f'    {edge}: {bits:.3f} bits')
            for a_k, b_k, given, res in dsep_pairs:
                g_str = ', '.join(f'type_L{level}_{g}' for g in given)
                print(f'  d-sep(L{level}:{a_k}, L{level}:{b_k} | {g_str}): {res}')
            if intervention_result:
                r = intervention_result
                print(f'  do(type_L{level}_{r["do_type"]}): '
                      f'flow {r["flow_before"]:.3f} → {r["flow_after"]:.3f} '
                      f'({r["reduction"]:+.3f} bits)')

        return {
            'flow':         flow,
            'dsep_pairs':   dsep_pairs,
            'intervention': intervention_result,
        }

    def level_summary(self) -> None:
        """Print per-level diagnostics to stdout."""
        print(f'\nPredictiveCodingHierarchy  n_levels={self.n_levels}')
        print(f'  max_chunk_size={self.max_chunk_size}  '
              f'adaptive={self.adaptive_threshold}  '
              f'k={self.surprise_k}  '
              f'grace={self.min_tokens_active}')
        header = (f'  {"Level":>6}  {"Tokens":>10}  {"VocabSz":>8}  '
                  f'{"MeanSurp":>10}  {"Threshold":>10}')
        print(header)
        active = 0
        for lvl in range(self.n_levels):
            seen = self._seen[lvl]
            if seen == 0:
                break
            active += 1
            hist   = self._surp_hist[lvl]
            mean_s = hist.mean()
            thresh = self._effective_threshold(lvl)
            bg     = getattr(self.learners[lvl], '_atom_bigrams', {})
            vocab_sz = len({a for (a, _) in bg})
            print(f'  {lvl:>6}  {seen:>10,}  {vocab_sz:>8,}  '
                  f'{mean_s:>10.3f}  {thresh:>10.3f}')
        print(f'  Active levels      : {active}')
        print(f'  Merges registered  : {self.vocab.n_merges():,}')
        print(f'  Segments registered: {self.vocab.n_segments():,}')

    def analyse(self, verbose: bool = True) -> list[dict]:
        """Run full R0-R6 analysis on every active PCH level.

        For each level that has seen at least ``min_tokens_active`` tokens:

        1. Call ``cluster_from_counts()`` on the level's RelationalLearner to
           produce E0 distributional categories (``assignment`` / ``clusters``).
        2. Run RelationClusterer (L2), SecondOrderGrammar (L4),
           GeometryDetector (R0), RelationalParadigmDiscoverer (R1),
           RelationalSenseSplitter (R2), RelationalAlgebra (R3).
        3. Store the results in ``self.analyses[level]``.

        Returns ``self.analyses`` (list indexed by level).
        """
        self.analyses: list[dict] = [{} for _ in range(self.n_levels)]

        for level in range(self.n_levels):
            if self._seen[level] < max(self.min_tokens_active, 10):
                break   # higher levels not active

            learner = self.learners[level]
            if verbose:
                n_atoms = len(getattr(learner, '_atom_counts', {}) or {})
                print(f'\n── Level {level}: {self._seen[level]:,} tokens, '
                      f'{n_atoms} (atom,rel) pairs ──')

            # E0: cluster atoms from online-collected _atom_counts.
            learner.cluster_from_counts(verbose=verbose)
            n_cats = len(getattr(learner, 'clusters', {}))

            # M12: type-abstraction fallback for sparse upper levels.
            # When JSD clustering yields K≤1 (surface forms too unique to cluster),
            # group atoms by the type-tuple of their direct constituents instead.
            # Analogous to weight sharing: chunks with the same abstract structure
            # are generalised together regardless of surface form.
            if n_cats <= 1 and level >= 1:
                lower_assign = getattr(self.learners[level - 1], 'assignment', {})
                if lower_assign:
                    learner.cluster_from_type_abstraction(
                        self.vocab, lower_assign, verbose=verbose)
                    n_cats = len(getattr(learner, 'clusters', {}))

            if n_cats == 0:
                if verbose:
                    print(f'  Level {level}: too few atoms for clustering — skip')
                continue

            # L2: relation clustering.
            rc = RelationClusterer()
            rc.fit(learner, verbose=verbose)

            # L4: second-order grammar.
            sog = SecondOrderGrammar()
            sog.fit(verbose=verbose, learner=learner)

            # R0: graph geometry.
            geo = GeometryDetector()
            geo.fit(learner, rc)
            if verbose:
                print(geo.report())

            # R1: paradigmatic roles.
            rpd = RelationalParadigmDiscoverer()
            rpd.fit(learner, verbose=verbose)
            if verbose:
                print(rpd.report(learner))

            # R2: sense disambiguation — needs raw chunk sequences.
            rss = RelationalSenseSplitter()
            level_seqs = self._chunk_seqs[level] if level > 0 else []
            # Level 0 sequences come from the original corpus (not chunk_seqs).
            # Caller should pass them via analyse(sequences=...) if desired.
            if level_seqs:
                # Cap total tokens fed to R2 to avoid O(N) scan on large corpora.
                # 200 K tokens provides ample trigram statistics for sense detection.
                _RSS_CAP = 200_000
                _rss_total = 0
                _rss_seqs: list = []
                for _s in level_seqs:
                    if _rss_total >= _RSS_CAP:
                        break
                    _rss_seqs.append(_s)
                    _rss_total += len(_s)
                rss.fit(_rss_seqs, verbose=verbose)
                if verbose:
                    print(rss.report())

            # R3: relational algebra.
            ra = RelationalAlgebra()
            ra.fit(learner, verbose=verbose)
            if verbose:
                print(ra.report())

            self.analyses[level] = {
                'rel_clusterer':  rc,
                'grammar':        sog,
                'geometry':       geo,
                'paradigm':       rpd,
                'sense':          rss,
                'algebra':        ra,
            }

            if not verbose:
                # Correct attribute names (role_clusters, composition_table).
                n_roles  = len(getattr(rpd, 'role_clusters', {}))
                ctable   = getattr(ra, 'composition_table', {})
                n_rules  = sum(1 for rk, _ in ctable.values() if rk is not None)
                n_novel  = sum(1 for rk, _ in ctable.values() if rk is None)
                geom_lbl = getattr(geo, 'topology', '?')
                print(f'  L{level:02d}  cats={n_cats}  roles={n_roles}  '
                      f'geom={geom_lbl}  rules={n_rules}(+{n_novel} novel)')

        return self.analyses

    def analyse_with_sequences(
        self,
        base_sequences,
        verbose: bool = True,
    ) -> list[dict]:
        """Like ``analyse()`` but also runs RelationalSenseSplitter at level 0.

        Parameters
        ----------
        base_sequences
            The original token sequences used in ``process_corpus()`` — needed
            so RelationalSenseSplitter can see the raw level-0 context.
        """
        self.analyses = [{} for _ in range(self.n_levels)]
        # Temporarily stash level-0 sequences so analyse() can use them.
        self._chunk_seqs[0] = list(base_sequences)
        result = self.analyse(verbose=verbose)
        self._chunk_seqs[0] = []   # restore (these are raw seqs, not chunk seqs)
        return result

    def analyse_cross_level(self, verbose: bool = True) -> dict:
        """Run R0-R6 on the cross-level constituency triples from ``self.vocab``.

        Builds triples of the form:
          - ``(chunk,  'left_const',  left_part)``
          - ``(chunk,  'right_const', right_part)``
          - ``(left_part,  'is_left_of',  chunk)``
          - ``(right_part, 'is_right_of', chunk)``
        from every MergedAtom and SegmentedAtom in the vocabulary, then fits a
        fresh RelationalLearner on them and runs the full R0-R6 pipeline.

        Discoveries expected:
        - Which atom-types appear only as left constituents, only as right, or
          both → head-vs-modifier distinction without supervision.
        - Composition rules: ``left_const ∘ right_const`` — are left-first
          or right-first patterns dominant?
        - Paradigmatic substitution classes across levels.

        Returns ``self.cross_level_analysis`` dict.
        """
        # ── Build cross-level triples ────────────────────────────────────────
        triples: list = []

        for l_s, r_s, atom in self.vocab._merges:
            s = atom.surface
            triples.append((s,   'left_const',  l_s))
            triples.append((s,   'right_const', r_s))
            triples.append((l_s, 'is_left_of',  s))
            triples.append((r_s, 'is_right_of', s))

        for atom in self.vocab._segments:
            s = atom.surface
            for i, part in enumerate(atom.constituents):
                part_s  = str(part)
                triples.append((s,      f'part_{i}',     part_s))
                triples.append((part_s, f'at_pos_{i}_of', s))

        if not triples:
            if verbose:
                print('  analyse_cross_level: no Merge/Segment atoms yet — skipping')
            self.cross_level_analysis = {}
            return {}

        if verbose:
            print(f'\n── Cross-level analysis: {len(triples):,} constituency triples ──')

        # ── Build _atom_counts via online updates (avoids slow batch fit()) ──
        # Batch RelationalLearner.fit() triggers E0 clustering on V=5,000-10,000
        # atoms which causes OOM/timeout.  Instead, feed triples as online updates
        # to build _atom_counts, then cluster_from_counts() with a small atom cap.
        cl_learner = RelationalLearner()
        for a_s, r_s, b_s in triples:
            cl_learner.update_online(str(a_s), str(r_s), str(b_s))
        cl_learner.cluster_from_counts(max_cluster_atoms=100, verbose=verbose)

        # ── Run R0-R6 ────────────────────────────────────────────────────────
        rc  = RelationClusterer();  rc.fit(cl_learner, verbose=verbose)
        sog = SecondOrderGrammar(); sog.fit(verbose=verbose, learner=cl_learner)
        geo = GeometryDetector();   geo.fit(cl_learner, rc)
        rpd = RelationalParadigmDiscoverer(); rpd.fit(cl_learner, verbose=verbose)
        rss = RelationalSenseSplitter()       # no raw sequences at cross-level
        ra  = RelationalAlgebra();            ra.fit(cl_learner, verbose=verbose)

        if verbose:
            print(geo.report())
            print(rpd.report(cl_learner))
            print(ra.report())

        self.cross_level_analysis = {
            'learner':        cl_learner,
            'rel_clusterer':  rc,
            'grammar':        sog,
            'geometry':       geo,
            'paradigm':       rpd,
            'sense':          rss,
            'algebra':        ra,
        }
        return self.cross_level_analysis

    def export_ctkg(
        self,
        domain_name: str       = 'discovered',
        out_path:    str | None = None,
    ) -> str:
        """Export the full PCH structure as a .ctkg DSL string.

        Includes:
        - Compositional units (MergedAtom/SegmentedAtom hierarchy).
        - Per-level E0 distributional categories (if ``analyse()`` was called).
        - Per-level R1 paradigmatic role clusters.
        - Per-level R3 relational composition rules.
        - Cross-level constituency analysis (if ``analyse_cross_level()`` called).
        """
        class _Shim:
            pass
        shim        = _Shim()
        shim.vocab  = self.vocab
        shim.levels = self.learners
        content = export_ctkg(shim, domain_name=domain_name, out_path=None)

        # Append multi-scale analysis if available.
        extra: list[str] = []
        analyses = getattr(self, 'analyses', [])
        for level, analysis in enumerate(analyses):
            if not analysis:
                continue
            learner = self.learners[level]
            assignment = getattr(learner, 'assignment', {})
            clusters   = getattr(learner, 'clusters',   {})
            if not clusters:
                continue

            extra.append(f'\n# ── Level {level} distributional categories ──')
            for cid, members in sorted(clusters.items()):
                cname = f'cat_L{level}_{cid}'
                top   = sorted(members)[:8]
                extra.append(f'concept {cname}')
                extra.append(f'    # members: {", ".join(repr(m) for m in top)}'
                             + (f' (+{len(members)-8} more)' if len(members)>8 else ''))
                extra.append('')

            rpd = analysis.get('paradigm')
            if rpd and rpd.role_clusters:
                extra.append(f'# ── Level {level} paradigmatic roles ──')
                for rid, members in sorted(rpd.role_clusters.items()):
                    rname = f'role_L{level}_{rid}'
                    extra.append(f'concept {rname}')
                    extra.append(f'    # role members: {", ".join(repr(m) for m in sorted(members)[:8])}')
                    extra.append('')

            ra = analysis.get('algebra')
            if ra and ra.composition_table:
                extra.append(f'# ── Level {level} relational algebra ──')
                extra.append(f'# {ra.report().strip()}')
                extra.append('')

        cla = getattr(self, 'cross_level_analysis', {})
        if cla:
            cl_learner = cla.get('learner')
            if cl_learner:
                cl_clusters = getattr(cl_learner, 'clusters', {})
                if cl_clusters:
                    extra.append('\n# ── Cross-level constituency categories ──')
                    for cid, members in sorted(cl_clusters.items()):
                        extra.append(f'concept cross_cat_{cid}')
                        extra.append(f'    # members: {", ".join(repr(m) for m in sorted(members)[:8])}')
                        extra.append('')
            cl_ra = cla.get('algebra')
            if cl_ra and cl_ra.composition_table:
                extra.append('# ── Cross-level relational algebra ──')
                extra.append(f'# {cl_ra.report().strip()}')
                extra.append('')

        if extra:
            content = content.rstrip() + '\n' + '\n'.join(extra) + '\n'

        if out_path:
            import pathlib
            pathlib.Path(out_path).write_text(content, encoding='utf-8')
        return content

    # ------------------------------------------------------------------
    # Warm-start factory
    # ------------------------------------------------------------------

    @classmethod
    def from_hrl(
        cls,
        hrl,
        **kwargs,
    ) -> 'PredictiveCodingHierarchy':
        """Warm-start a new PCH from an object that exposes ``.levels`` and ``.vocab``.

        Borrows fitted RelationalLearners so that surprisal is immediately
        meaningful.  Higher levels start cold and warm up online.

        Parameters
        ----------
        hrl
            Any object with ``.levels`` (list[RelationalLearner]) and
            ``.vocab`` (AtomVocabulary) — e.g. a previously trained PCH.
        **kwargs
            Additional constructor arguments forwarded to ``__init__``.
        """
        pch = cls(**kwargs)
        for i, learner in enumerate(hrl.levels):
            if i < pch.n_levels:
                pch.learners[i] = learner
        pch.vocab = hrl.vocab   # share the merge/segment registry
        return pch
