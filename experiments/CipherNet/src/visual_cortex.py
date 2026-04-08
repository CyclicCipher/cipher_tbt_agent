"""Retinotopic visual cortex with category-based recognition.

Phase 1: Explore images (unlabeled), accumulate sensorimotor triples.
Phase 2: Run category discovery → feature + morphism equivalence classes.
Phase 3: Learn/recognize using categorical features (generalizable).

The category discovery transforms raw observations into abstract
categorical tokens that generalize across images.
"""
from __future__ import annotations

import numpy as np

from symbolic_column import SymbolicColumn, CorticalMessage
from sdr import SDR, SDREncoder
from eye import Eye, SalienceMap
from codebook import PatchCodebook
from category_discovery import discover_category, DiscoveredCategory


def parietal_transform(retinal_pos, eye_fixation, object_origin):
    """Symbolic parietal cortex: object-centered coordinates."""
    return (round(retinal_pos[0] + eye_fixation[0] - object_origin[0]),
            round(retinal_pos[1] + eye_fixation[1] - object_origin[1]))


class RetinotopicV1:
    """V1 cortex with category discovery for reference frame learning."""

    def __init__(self, eye: Eye, codebook: PatchCodebook,
                 patch_size: int = 5, stride: int = 3):
        self.eye = eye
        self.codebook = codebook
        self.patch_size = patch_size
        self.stride = stride

        rs = eye.retina_size
        self.grid_h = (rs - patch_size) // stride + 1
        self.grid_w = (rs - patch_size) // stride + 1
        self.n_cols = self.grid_h * self.grid_w

        # Retinal positions (center of each column's RF).
        half = rs / 2.0
        self._retinal_pos = []
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                ry = gy * stride + patch_size / 2.0 - half
                rx = gx * stride + patch_size / 2.0 - half
                self._retinal_pos.append((rx, ry))

        self._object_origin = (0.0, 0.0)

        # Phase 1 storage: triples per column.
        # Each column accumulates (feature_code, displacement, feature_code).
        self._triples: list[list[tuple[str, tuple, str]]] = [[] for _ in range(self.n_cols)]

        # Phase 2 result: discovered category.
        self.category: DiscoveredCategory | None = None

        # Phase 3: object models per column.
        # {column_idx: {object_id: {categorical_location: categorical_feature}}}
        self._models: list[dict[str, dict]] = [{} for _ in range(self.n_cols)]

    def set_object_origin(self, origin):
        self._object_origin = origin

    def _encode_patches(self, image: np.ndarray) -> list[str]:
        """Extract patches from retinal image, VQ encode, return code strings."""
        retina = self.eye.sample(image)
        ps, st = self.patch_size, self.stride
        patches = np.empty((self.n_cols, ps, ps), dtype=np.float32)
        idx = 0
        for gy in range(self.grid_h):
            y0 = gy * st
            for gx in range(self.grid_w):
                x0 = gx * st
                patches[idx] = retina[y0:y0+ps, x0:x0+ps]
                idx += 1
        codes = self.codebook.encode_batch(patches)
        return [f"v{c}" for c in codes]

    def _get_locations(self) -> list[tuple]:
        """Compute object-centered locations for all columns."""
        eye_fix = self.eye.fixation
        locations = []
        for rp in self._retinal_pos:
            loc = parietal_transform(rp, eye_fix, self._object_origin)
            locations.append(loc)
        return locations

    # --- Phase 1: Explore and accumulate triples ---

    def observe_and_accumulate(self, image: np.ndarray,
                               prev_codes: list[str] | None,
                               displacement: tuple | None) -> list[str]:
        """Observe features. If prev_codes + displacement given, record triples."""
        codes = self._encode_patches(image)
        if prev_codes is not None and displacement is not None:
            qd = (round(displacement[0]), round(displacement[1]))
            for i in range(self.n_cols):
                self._triples[i].append((prev_codes[i], qd, codes[i]))
        return codes

    def total_triples(self) -> int:
        return sum(len(t) for t in self._triples)

    # --- Phase 2: Discover categories ---

    def discover(self, verbose: bool = False):
        """Run category discovery on accumulated triples (pooled across columns).

        All columns see the same displacement structure (same eye movements),
        so we pool triples to get better statistics.
        """
        all_triples = []
        for col_triples in self._triples:
            all_triples.extend(col_triples)

        if verbose:
            print(f"  Discovering category from {len(all_triples)} triples...")

        self.category = discover_category(all_triples, verbose=verbose)

    def _categorize_feature(self, code: str) -> str:
        """Map a raw VQ code to its discovered object class."""
        if self.category is None:
            return code
        cls = self.category.feature_to_object.get(code)
        if cls is not None:
            return f"c{cls}"
        return code  # unknown feature: keep raw

    def _categorize_location(self, loc: tuple) -> str:
        """Map a raw location to a canonical string key."""
        # Location is already quantized integers from parietal transform.
        return f"L{loc[0]},{loc[1]}"

    # --- Phase 3: Learn and recognize with categorical features ---

    def learn_object(self, image: np.ndarray, object_id: str):
        """Learn: bind categorical features to categorical locations for this object."""
        codes = self._encode_patches(image)
        locations = self._get_locations()
        for i in range(self.n_cols):
            cat_feat = self._categorize_feature(codes[i])
            cat_loc = self._categorize_location(locations[i])
            if object_id not in self._models[i]:
                self._models[i][object_id] = {}
            model = self._models[i][object_id]
            # Store: at this categorical location, this categorical feature
            # was seen for this object. Count occurrences.
            key = (cat_loc, cat_feat)
            model[key] = model.get(key, 0) + 1

    def recognize(self, image: np.ndarray) -> tuple[str | None, float]:
        """Recognize: match current observations against stored models."""
        codes = self._encode_patches(image)
        locations = self._get_locations()

        # Score each object across all columns.
        votes: dict[str, float] = {}
        for i in range(self.n_cols):
            cat_feat = self._categorize_feature(codes[i])
            cat_loc = self._categorize_location(locations[i])
            key = (cat_loc, cat_feat)

            for obj_id, model in self._models[i].items():
                if key in model:
                    # This (location, feature) pair was seen for this object.
                    count = model[key]
                    votes[obj_id] = votes.get(obj_id, 0.0) + count

        if not votes:
            return None, 0.0
        winner = max(votes, key=votes.get)
        total = sum(votes.values())
        return winner, votes[winner] / total


class FovealExplorer:
    """Three-phase foveal exploration: collect → discover → recognize."""

    def __init__(self, eye: Eye, v1: RetinotopicV1, n_fixations: int = 3):
        self.eye = eye
        self.v1 = v1
        self.n_fixations = n_fixations

    def _get_fixations(self, image: np.ndarray) -> list[tuple]:
        """Cardinal scan: regular saccade vocabulary for structured exploration."""
        return self.eye.cardinal_scan(image, step=5, n_fixations=self.n_fixations)

    def explore_unlabeled(self, image: np.ndarray):
        """Phase 1: Explore an image, accumulate triples. No labels."""
        h, w = image.shape[:2]
        self.v1.set_object_origin((w / 2.0, h / 2.0))
        fixations = self._get_fixations(image)

        fx, fy = fixations[0]
        self.eye.fixate(float(fx), float(fy))
        prev_codes = self.v1.observe_and_accumulate(image, None, None)

        for fx, fy in fixations[1:]:
            old_fix = self.eye.fixation
            self.eye.saccade_to(float(fx), float(fy))
            disp = self.eye.last_displacement
            prev_codes = self.v1.observe_and_accumulate(image, prev_codes, disp)

    def learn(self, image: np.ndarray, object_id: str):
        """Phase 3: Learn object model with categorical features."""
        h, w = image.shape[:2]
        self.v1.set_object_origin((w / 2.0, h / 2.0))
        fixations = self._get_fixations(image)

        for fx, fy in fixations:
            self.eye.fixate(float(fx), float(fy))
            self.v1.learn_object(image, object_id)

    def recognize(self, image: np.ndarray) -> tuple[str | None, float]:
        """Phase 3: Recognize using categorical features."""
        h, w = image.shape[:2]
        self.v1.set_object_origin((w / 2.0, h / 2.0))
        fixations = self._get_fixations(image)

        # Recognize at each fixation, accumulate votes.
        all_votes: dict[str, float] = {}
        for fx, fy in fixations:
            self.eye.fixate(float(fx), float(fy))
            pred, conf = self.v1.recognize(image)
            if pred is not None:
                all_votes[pred] = all_votes.get(pred, 0.0) + conf

        if not all_votes:
            return None, 0.0
        winner = max(all_votes, key=all_votes.get)
        total = sum(all_votes.values())
        return winner, all_votes[winner] / total
