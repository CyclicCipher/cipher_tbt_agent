"""Hierarchical retinotopic visual cortex with category discovery.

V1: pixel patches → VQ codes → category discovery → object classes
V2+: receives lower level's categorical outputs as features,
     accumulates triples, discovers higher-level categories.

Each level uses the SAME protocol:
  explore → accumulate triples → discover categories → learn → recognize

Higher levels see COMPOSITIONS of lower-level features. V2's feature
is the pattern of V1 categories across its receptive field.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict

from symbolic_column import SymbolicColumn, MAX_MEMORY
from eye import Eye, SalienceMap
from codebook import PatchCodebook
from category_discovery import discover_category, DiscoveredCategory


def parietal_transform(retinal_pos, eye_fixation, object_origin):
    return (round(retinal_pos[0] + eye_fixation[0] - object_origin[0]),
            round(retinal_pos[1] + eye_fixation[1] - object_origin[1]))


class CorticalLevel:
    """One level of the visual hierarchy.

    Has a grid of columns, each with a receptive field over the level
    below (or over the retina for V1). Accumulates triples, discovers
    categories, learns and recognizes with categorical features.
    """

    def __init__(self, name: str, grid_h: int, grid_w: int):
        self.name = name
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_cols = grid_h * grid_w

        # Triples accumulated during exploration.
        self._triples: list[tuple[str, tuple, str]] = []

        # Discovered category.
        self.category: DiscoveredCategory | None = None

        # Object models: {col_idx: {object_id: {(cat_loc, cat_feat): count}}}
        self._models: list[dict[str, dict]] = [{} for _ in range(self.n_cols)]

        # Current features (set during observe).
        self._current_features: list[str] = [""] * self.n_cols

    def categorize(self, raw_feature: str) -> str:
        """Map raw feature to discovered category class."""
        if self.category is None:
            return raw_feature
        cls = self.category.feature_to_object.get(raw_feature)
        return f"{self.name}_c{cls}" if cls is not None else raw_feature

    def observe(self, features: list[str]):
        """Set current features for all columns."""
        self._current_features = [self.categorize(f) for f in features]

    def accumulate_triple(self, prev_features: list[str], displacement: tuple):
        """Record (prev_feature, displacement, curr_feature) for each column."""
        for i in range(self.n_cols):
            self._triples.append((prev_features[i], displacement, self._current_features[i]))

    def discover(self, n_classes: int = 32, verbose: bool = False):
        """Run category discovery on accumulated triples."""
        if not self._triples:
            return
        if verbose:
            print(f"  {self.name}: discovering from {len(self._triples)} triples...")
        self.category = discover_category(self._triples, n_object_classes=n_classes,
                                          verbose=verbose)

    def learn(self, object_id: str, locations: list[tuple]):
        """Learn: bind (categorical_feature, location) → object for each column."""
        for i in range(self.n_cols):
            feat = self._current_features[i]
            loc = f"L{locations[i][0]},{locations[i][1]}"
            if object_id not in self._models[i]:
                self._models[i][object_id] = {}
            model = self._models[i][object_id]
            key = (loc, feat)
            model[key] = model.get(key, 0) + 1

    def vote(self, locations: list[tuple]) -> dict[str, float]:
        """Recognition: score each object by matching current observations."""
        votes: dict[str, float] = {}
        for i in range(self.n_cols):
            feat = self._current_features[i]
            loc = f"L{locations[i][0]},{locations[i][1]}"
            key = (loc, feat)
            for obj_id, model in self._models[i].items():
                if key in model:
                    votes[obj_id] = votes.get(obj_id, 0.0) + model[key]
        return votes

    def total_triples(self) -> int:
        return len(self._triples)


class HierarchicalV1:
    """Multi-level visual cortex with category discovery at each level."""

    def __init__(self, eye: Eye, codebook: PatchCodebook,
                 patch_size: int = 5, stride: int = 3,
                 n_levels: int = 2, pool: int = 2,
                 n_categories: int = 32):
        self.eye = eye
        self.codebook = codebook
        self.patch_size = patch_size
        self.stride = stride
        self.n_categories = n_categories

        rs = eye.retina_size
        grid_h = (rs - patch_size) // stride + 1
        grid_w = (rs - patch_size) // stride + 1

        # Build levels.
        self.levels: list[CorticalLevel] = []
        self._pool = pool

        # Level 0 (V1): one column per retinal patch.
        self.levels.append(CorticalLevel("V1", grid_h, grid_w))

        # Higher levels: pool from level below.
        cur_h, cur_w = grid_h, grid_w
        for lev in range(1, n_levels):
            next_h = max(1, cur_h // pool)
            next_w = max(1, cur_w // pool)
            self.levels.append(CorticalLevel(f"V{lev+1}", next_h, next_w))
            cur_h, cur_w = next_h, next_w

        # Retinal positions for parietal transform (V1 only).
        half = rs / 2.0
        self._retinal_pos = []
        for gy in range(grid_h):
            for gx in range(grid_w):
                ry = gy * stride + patch_size / 2.0 - half
                rx = gx * stride + patch_size / 2.0 - half
                self._retinal_pos.append((rx, ry))

        self._object_origin = (0.0, 0.0)

    def set_object_origin(self, origin):
        self._object_origin = origin

    def _v1_encode(self, image: np.ndarray) -> list[str]:
        """Encode V1 patches from retinal image via VQ codebook."""
        retina = self.eye.sample(image)
        ps, st = self.patch_size, self.stride
        level0 = self.levels[0]
        patches = np.empty((level0.n_cols, ps, ps), dtype=np.float32)
        idx = 0
        for gy in range(level0.grid_h):
            y0 = gy * st
            for gx in range(level0.grid_w):
                x0 = gx * st
                patches[idx] = retina[y0:y0+ps, x0:x0+ps]
                idx += 1
        codes = self.codebook.encode_batch(patches)
        return [f"v{c}" for c in codes]

    def _propagate_up(self):
        """Propagate features from V1 upward. Higher levels compose lower."""
        for lev_idx in range(1, len(self.levels)):
            lower = self.levels[lev_idx - 1]
            upper = self.levels[lev_idx]
            pool = self._pool

            # Each upper column covers a pool×pool patch of lower columns.
            features = []
            for gy in range(upper.grid_h):
                for gx in range(upper.grid_w):
                    # Gather lower-level features in this patch.
                    parts = []
                    for dy in range(pool):
                        for dx in range(pool):
                            ly = gy * pool + dy
                            lx = gx * pool + dx
                            li = ly * lower.grid_w + lx
                            if li < lower.n_cols:
                                parts.append(lower._current_features[li])
                            else:
                                parts.append("_")
                    # Compose: ORDERED spatial tuple (position-specific).
                    # NOT sorted — spatial arrangement IS the feature.
                    # Different spatial arrangements of the same V1 features
                    # produce DIFFERENT V2 features (like V2 corner detectors
                    # have spatially distinct subunits wired to specific V1 positions).
                    composed = "|".join(parts)
                    features.append(composed)

            upper.observe(features)

    def _get_locations(self) -> list[tuple]:
        """Object-centered locations for V1 columns."""
        eye_fix = self.eye.fixation
        return [parietal_transform(rp, eye_fix, self._object_origin)
                for rp in self._retinal_pos]

    def _get_level_locations(self, lev_idx: int) -> list[tuple]:
        """Locations for a specific level (pooled from V1 locations)."""
        if lev_idx == 0:
            return self._get_locations()
        # Higher levels: average of V1 locations in receptive field.
        lower_locs = self._get_level_locations(lev_idx - 1)
        lower = self.levels[lev_idx - 1]
        upper = self.levels[lev_idx]
        pool = self._pool
        locs = []
        for gy in range(upper.grid_h):
            for gx in range(upper.grid_w):
                sum_x, sum_y, count = 0.0, 0.0, 0
                for dy in range(pool):
                    for dx in range(pool):
                        ly = gy * pool + dy
                        lx = gx * pool + dx
                        li = ly * lower.grid_w + lx
                        if li < len(lower_locs):
                            sum_x += lower_locs[li][0]
                            sum_y += lower_locs[li][1]
                            count += 1
                if count > 0:
                    locs.append((round(sum_x / count), round(sum_y / count)))
                else:
                    locs.append((0, 0))
        return locs

    # --- Phase 1: Explore unlabeled ---

    def explore_unlabeled(self, image: np.ndarray, displacement: tuple | None,
                          prev_v1_features: list[str] | None) -> list[str]:
        """One fixation: encode, observe at all levels, accumulate triples."""
        v1_features = self._v1_encode(image)
        self.levels[0].observe(v1_features)
        self._propagate_up()

        if prev_v1_features is not None and displacement is not None:
            # Accumulate triples at V1.
            qd = (round(displacement[0]), round(displacement[1]))
            prev_cat = [self.levels[0].categorize(f) for f in prev_v1_features]
            self.levels[0].accumulate_triple(prev_cat, qd)

            # Higher levels: same displacement, composed features.
            for lev_idx in range(1, len(self.levels)):
                level = self.levels[lev_idx]
                # prev features at this level were set during previous fixation.
                # We need to store them — use a simple buffer.
                if hasattr(level, '_prev_features') and level._prev_features:
                    level.accumulate_triple(level._prev_features, qd)

        # Buffer current features for next triple.
        for level in self.levels:
            level._prev_features = list(level._current_features)

        return v1_features

    # --- Phase 2: Discover categories ---

    def discover_all(self, verbose: bool = False):
        """Run category discovery at each level."""
        for level in self.levels:
            level.discover(n_classes=self.n_categories, verbose=verbose)

    # --- Phase 3: Learn ---

    def learn(self, image: np.ndarray, object_id: str):
        """Learn at one fixation: all levels bind (feature, location) → object."""
        v1_features = self._v1_encode(image)
        self.levels[0].observe(v1_features)
        self._propagate_up()

        for lev_idx, level in enumerate(self.levels):
            locs = self._get_level_locations(lev_idx)
            level.learn(object_id, locs)

    # --- Phase 3: Recognize ---

    def recognize(self, image: np.ndarray) -> tuple[str | None, float]:
        """Recognize at one fixation: all levels vote."""
        v1_features = self._v1_encode(image)
        self.levels[0].observe(v1_features)
        self._propagate_up()

        # Weighted votes from all levels (higher = more weight).
        all_votes: dict[str, float] = {}
        for lev_idx, level in enumerate(self.levels):
            weight = 2.0 ** lev_idx
            locs = self._get_level_locations(lev_idx)
            level_votes = level.vote(locs)
            for obj, score in level_votes.items():
                all_votes[obj] = all_votes.get(obj, 0.0) + score * weight

        if not all_votes:
            return None, 0.0
        winner = max(all_votes, key=all_votes.get)
        total = sum(all_votes.values())
        return winner, all_votes[winner] / total


class FovealExplorer:
    """Three-phase foveal exploration with hierarchy."""

    def __init__(self, eye: Eye, cortex: HierarchicalV1, n_fixations: int = 9):
        self.eye = eye
        self.cortex = cortex
        self.n_fixations = n_fixations

    def _get_fixations(self, image: np.ndarray) -> list[tuple]:
        return self.eye.cardinal_scan(image, step=5, n_fixations=self.n_fixations)

    def explore_unlabeled(self, image: np.ndarray):
        h, w = image.shape[:2]
        self.cortex.set_object_origin((w / 2.0, h / 2.0))
        fixations = self._get_fixations(image)

        fx, fy = fixations[0]
        self.eye.fixate(float(fx), float(fy))
        prev_v1 = self.cortex.explore_unlabeled(image, None, None)

        for fx, fy in fixations[1:]:
            self.eye.saccade_to(float(fx), float(fy))
            disp = self.eye.last_displacement
            prev_v1 = self.cortex.explore_unlabeled(image, disp, prev_v1)

    def learn(self, image: np.ndarray, object_id: str):
        h, w = image.shape[:2]
        self.cortex.set_object_origin((w / 2.0, h / 2.0))
        for fx, fy in self._get_fixations(image):
            self.eye.fixate(float(fx), float(fy))
            self.cortex.learn(image, object_id)

    def recognize(self, image: np.ndarray) -> tuple[str | None, float]:
        h, w = image.shape[:2]
        self.cortex.set_object_origin((w / 2.0, h / 2.0))
        all_votes: dict[str, float] = {}
        for fx, fy in self._get_fixations(image):
            self.eye.fixate(float(fx), float(fy))
            pred, conf = self.cortex.recognize(image)
            if pred:
                all_votes[pred] = all_votes.get(pred, 0.0) + conf
        if not all_votes:
            return None, 0.0
        winner = max(all_votes, key=all_votes.get)
        total = sum(all_votes.values())
        return winner, all_votes[winner] / total
