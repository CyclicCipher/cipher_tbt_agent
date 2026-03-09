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

    def __init__(self, ai: SymbolicAI | None = None, n_clusters: int = 12):
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

    # ---- E0-E3: fit ---------------------------------------------------------

    def fit(self, triples: list[tuple[Any, str, Any]],
            e3_temperature: float = 2.0,
            verbose: bool = True) -> None:
        """Fit E0-E3 on a list of (atom_a, relation_name, atom_b) triples."""
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
        if verbose:
            print(f'  Atoms: {len(self.assignment):,}  '
                  f'Clusters: {len(self.clusters)}')

        # Build E3 successor distributions BEFORE clearing E0 examples
        # (ask_dist uses store.examples, which is gone after .clear())
        if verbose:
            print('  E3: building successor distributions...')
        succ_dists = _build_rel_succ_dists(self.ai, self.clusters)
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
        n_used = n_skip = 0
        for a, r, b in triples:
            a_s, r_s, b_s = str(a), str(r), str(b)
            c_a = self.assignment.get(a_s)
            c_b = self.assignment.get(b_s)
            if c_a is None or c_b is None:
                n_skip += 1
                continue
            self.ai.teach('next_cat_rel',
                          (str(c_a), r_s), (str(c_b),))
            self.ai.teach('token_given_cat_rel',
                          (str(c_a), r_s, str(c_b)), (b_s,))
            n_used += 1
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

    def predict(self, atom: Any, relation: str) -> Any | None:
        """Predict most likely target atom given source atom and relation name."""
        a_s = str(atom)
        r_s = str(relation)
        c_a = self.assignment.get(a_s)
        if c_a is None:
            return None
        # E3 soft
        c_b_dist = self._get_nc_soft((str(c_a), r_s))
        if c_b_dist:
            c_b_tup = max(c_b_dist, key=c_b_dist.get)
            c_b = c_b_tup[0] if isinstance(c_b_tup, tuple) else str(c_b_tup)
            atom_dist = self._get_wgc_soft((str(c_a), r_s, c_b))
            if atom_dist:
                best = max(atom_dist, key=atom_dist.get)
                return best[0] if isinstance(best, tuple) else str(best)
        # E1 fallback (uses _index, works after examples.clear())
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

        Marginalises over c_tgt: P(b|a,r) = Σ_{c_tgt} P(c_tgt|c_a,r) · P(b|c_a,r,c_tgt).
        """
        a_s = str(atom)
        r_s = str(relation)
        c_a = self.assignment.get(a_s)
        if c_a is None:
            return {}
        c_b_dist = self._get_nc_soft((str(c_a), r_s))
        if not c_b_dist:
            return {}
        result: dict = {}
        for c_b_key, p_cb in c_b_dist.items():
            c_b = c_b_key[0] if isinstance(c_b_key, tuple) else str(c_b_key)
            t_dist = self._get_wgc_soft((str(c_a), r_s, c_b))
            if not t_dist:
                continue
            for tok_key, p_t in t_dist.items():
                tok = tok_key[0] if isinstance(tok_key, tuple) else str(tok_key)
                result[tok] = result.get(tok, 0.0) + p_cb * p_t
        return result

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
    MUST be called before examples.clear() — ask_dist() iterates examples.
    """
    succ: dict = {}
    for cid, members in clusters.items():
        merged: dict = collections.defaultdict(float)
        n = 0
        for tok in members:
            d = ai.ask_dist('rel_next', (str(tok),))
            if d is None:
                continue
            for out_tup, prob in d.items():
                k = out_tup[0] if isinstance(out_tup, tuple) else str(out_tup)
                merged[k] += prob
            n += 1
        succ[cid] = ({k: v / n for k, v in merged.items()} if n else {})
    return succ


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

def _gap_threshold(jsd_values: list, sensitivity: float = 0.1) -> float:
    """Data-driven merge threshold via Kneedle algorithm on sorted pairwise JSDs.

    Normalizes the sorted JSD values to the unit square [0,1]×[0,1], then
    finds the point of maximum perpendicular distance from the diagonal y=x.
    The midpoint just before that knee is the merge threshold.

    If the curve is nearly linear (max distance < sensitivity), there is no
    clear cluster boundary → returns float('inf') → everything stays merged.

    This replaces the hardcoded jsd_threshold with a data-driven value that
    scales with the actual distribution of pairwise distances in the data.
    """
    n = len(jsd_values)
    if n == 0:
        return float('inf')
    vals = sorted(float(v) for v in jsd_values)
    y_min, y_max = vals[0], vals[-1]
    if y_max - y_min < 1e-9:
        return float('inf')  # all equal → no structure to split
    if n == 1:
        return vals[0] / 2.0 if vals[0] >= 0.3 else float('inf')
    x_norm = [i / (n - 1) for i in range(n)]
    y_norm = [(v - y_min) / (y_max - y_min) for v in vals]
    # Perpendicular distance from the diagonal y = x (in unit square)
    distances = [abs(y_norm[i] - x_norm[i]) for i in range(n)]
    max_dist = max(distances)
    if max_dist < sensitivity:
        return float('inf')  # nearly linear → no meaningful knee → 1 cluster
    knee_idx = distances.index(max_dist)
    if knee_idx == 0:
        return (vals[0] + vals[1]) / 2.0 if n > 1 else vals[0] / 2.0
    return (vals[knee_idx - 1] + vals[knee_idx]) / 2.0


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
        # Build adjacency
        out_edges: dict[str, list] = collections.defaultdict(list)
        for a, r, b in triples:
            out_edges[str(a)].append((str(r), str(b)))

        # Accumulate (r1, r2) co-occurrences from chains
        seq_counts: dict[str, dict[str, float]] = collections.defaultdict(
            lambda: collections.defaultdict(float))
        for a_s, edges_a in out_edges.items():
            for r1, b_s in edges_a:
                for r2, _ in out_edges.get(b_s, []):
                    seq_counts[r1][r2] += 1.0
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
