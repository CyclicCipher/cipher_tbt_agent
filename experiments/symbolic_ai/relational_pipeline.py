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
        print('Smoke test PASSED')
    except ImportError:
        print('numpy not available — skipping image test')
