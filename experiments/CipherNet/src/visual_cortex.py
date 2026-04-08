"""Ventral visual stream: V1 → V2 → V4 → IT → linear decision.

The hierarchy progressively untangles object representations.
V1 detects edges. V2 detects corners. V4 detects shapes.
IT produces position-invariant representations where different
object categories are linearly separable.

Classification is a SIMPLE LINEAR READOUT of the IT representation.
Not feature-location matching. Not nearest-neighbor. Just a dot product.

Each level uses category discovery to compress its feature space.
Higher levels compose lower levels' categorical outputs into
progressively more abstract, invariant representations.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict

from eye import Eye, SalienceMap
from codebook import GaborFilterBank
from category_discovery import discover_category


def parietal_transform(retinal_pos, eye_fixation, object_origin):
    return (round(retinal_pos[0] + eye_fixation[0] - object_origin[0]),
            round(retinal_pos[1] + eye_fixation[1] - object_origin[1]))


class VentralStream:
    """Complete ventral visual stream with linear decision.

    V1 (5×5) → V2 (2×2) → V4 (1×1) → IT (1×1, invariant)
    Each level: encode → categorize → compose → propagate up.
    IT output → linear classifier → digit label.
    """

    def __init__(self, eye: Eye, gabor: GaborFilterBank,
                 patch_size: int = 5, stride: int = 3,
                 n_categories: int = 32):
        self.eye = eye
        self.gabor = gabor
        self.patch_size = patch_size
        self.stride = stride
        self.n_categories = n_categories

        rs = eye.retina_size
        self.v1_h = (rs - patch_size) // stride + 1
        self.v1_w = (rs - patch_size) // stride + 1

        # Retinal positions for V1.
        half = rs / 2.0
        self._retinal_pos = []
        for gy in range(self.v1_h):
            for gx in range(self.v1_w):
                ry = gy * stride + patch_size / 2.0 - half
                rx = gx * stride + patch_size / 2.0 - half
                self._retinal_pos.append((rx, ry))

        self._object_origin = (0.0, 0.0)

        # Category discovery state per level.
        self._triples = {'V1': [], 'V2': [], 'V4': []}
        self._categories = {}  # level_name → DiscoveredCategory
        self._prev_features = {}  # level_name → previous feature list

        # IT representation accumulator.
        # For each image: collect all (level, category_id) across fixations.
        self._it_vector_size = 0  # set after discovery
        self._it_index = {}  # (level, category_id) → index in IT vector

        # Linear classifier (vlPFC): weight matrix + bias.
        # Maps IT vector → class scores. Learned from labeled examples.
        self._weights = None  # (n_classes, it_vector_size)
        self._bias = None     # (n_classes,)

    def set_object_origin(self, origin):
        self._object_origin = origin

    # --- V1 encoding ---

    def _v1_encode(self, image: np.ndarray) -> list[str]:
        """Gabor encode retinal patches."""
        retina = self.eye.sample(image)
        ps, st = self.patch_size, self.stride
        n = self.v1_h * self.v1_w
        patches = np.empty((n, ps, ps), dtype=np.float32)
        idx = 0
        for gy in range(self.v1_h):
            y0 = gy * st
            for gx in range(self.v1_w):
                x0 = gx * st
                patches[idx] = retina[y0:y0+ps, x0:x0+ps]
                idx += 1
        return self.gabor.encode_batch(patches)

    # --- Hierarchical composition ---

    def _compose(self, features: list[str], grid_h: int, grid_w: int,
                 pool: int = 2) -> list[str]:
        """Compose features from a grid into higher-level features.
        Pool×pool lower features → 1 higher feature (ordered spatial tuple).
        """
        out_h = max(1, grid_h // pool)
        out_w = max(1, grid_w // pool)
        composed = []
        for gy in range(out_h):
            for gx in range(out_w):
                parts = []
                for dy in range(pool):
                    for dx in range(pool):
                        ly = min(gy * pool + dy, grid_h - 1)
                        lx = min(gx * pool + dx, grid_w - 1)
                        li = ly * grid_w + lx
                        parts.append(features[li] if li < len(features) else "_")
                composed.append("|".join(parts))
        return composed

    def _categorize(self, features: list[str], level: str) -> list[str]:
        """Map raw features to discovered category IDs."""
        cat = self._categories.get(level)
        if cat is None:
            return features
        result = []
        for f in features:
            cls = cat.feature_to_object.get(f)
            result.append(f"{level}_c{cls}" if cls is not None else f)
        return result

    # --- Full forward pass ---

    def _forward(self, image: np.ndarray) -> dict[str, list[str]]:
        """Forward pass: V1 → V2 → V4 → IT categorized features per level."""
        v1_raw = self._v1_encode(image)
        v1_cat = self._categorize(v1_raw, 'V1')

        v2_raw = self._compose(v1_cat, self.v1_h, self.v1_w, pool=2)
        v2_h = max(1, self.v1_h // 2)
        v2_w = max(1, self.v1_w // 2)
        v2_cat = self._categorize(v2_raw, 'V2')

        v4_raw = self._compose(v2_cat, v2_h, v2_w, pool=max(v2_h, v2_w))
        v4_cat = self._categorize(v4_raw, 'V4')

        # IT = V4 output (position-invariant at this point: single column
        # covering the entire visual field through progressive pooling).
        return {'V1': v1_cat, 'V2': v2_cat, 'V4': v4_cat, 'IT': v4_cat}

    # --- Phase 1: Explore and accumulate triples ---

    def explore_fixation(self, image: np.ndarray, displacement: tuple | None):
        """One fixation during exploration. Accumulate triples at each level."""
        features = self._forward(image)

        for level in ['V1', 'V2', 'V4']:
            prev = self._prev_features.get(level)
            if prev is not None and displacement is not None:
                qd = (round(displacement[0]), round(displacement[1]))
                for i in range(len(features[level])):
                    if i < len(prev):
                        self._triples[level].append((prev[i], qd, features[level][i]))

        for level in ['V1', 'V2', 'V4']:
            self._prev_features[level] = list(features[level])

    def total_triples(self) -> int:
        return sum(len(t) for t in self._triples.values())

    # --- Phase 2: Discover categories ---

    def discover(self, verbose: bool = False):
        """Run category discovery at V1, V2, V4."""
        for level in ['V1', 'V2', 'V4']:
            triples = self._triples[level]
            if not triples:
                continue
            if verbose:
                print(f"  {level}: {len(triples)} triples...")
            self._categories[level] = discover_category(
                triples, n_object_classes=self.n_categories, verbose=verbose)

        # Build IT vector index: one dimension per (level, category_id).
        self._it_index = {}
        idx = 0
        for level in ['V1', 'V2', 'V4']:
            cat = self._categories.get(level)
            if cat is None:
                continue
            for cid in sorted(cat.objects.keys()):
                self._it_index[(level, cid)] = idx
                idx += 1
        self._it_vector_size = idx
        if verbose:
            print(f"  IT vector size: {self._it_vector_size} dimensions")

    # --- IT representation ---

    def _it_vector(self, image: np.ndarray) -> np.ndarray:
        """Compute the IT representation vector for one fixation.

        Counts how many columns at each level activate each category.
        This is the population code: a histogram over categories.
        """
        features = self._forward(image)
        vec = np.zeros(self._it_vector_size, dtype=np.float32)
        for level in ['V1', 'V2', 'V4']:
            cat = self._categories.get(level)
            if cat is None:
                continue
            for f in features[level]:
                # Extract category ID from categorized feature string.
                if f.startswith(f"{level}_c"):
                    try:
                        cid = int(f[len(level) + 2:])
                        key = (level, cid)
                        if key in self._it_index:
                            vec[self._it_index[key]] += 1.0
                    except ValueError:
                        pass
        return vec

    def compute_it(self, image: np.ndarray,
                    fixations: list[tuple]) -> np.ndarray:
        """Compute IT vector accumulated across multiple fixations.

        This is the position-invariant representation: summing
        category activations across all fixation points gives a
        representation that doesn't depend on saccade order.
        """
        vec = np.zeros(self._it_vector_size, dtype=np.float32)
        for fx, fy in fixations:
            self.eye.fixate(float(fx), float(fy))
            vec += self._it_vector(image)
        # Normalize.
        total = vec.sum()
        if total > 0:
            vec /= total
        return vec

    # --- Phase 3: Train linear classifier (vlPFC) ---

    def train_classifier(self, images: np.ndarray, labels: np.ndarray,
                         fixation_fn, n_classes: int = 10,
                         verbose: bool = True):
        """Train linear classifier on IT representations.

        This is the vlPFC: maps IT vector → class scores.
        Uses simple least-squares (no gradient descent, no epochs).
        One-shot: solve X @ W = Y directly.
        """
        n = len(images)
        if self._it_vector_size == 0:
            print("  WARNING: IT vector size is 0. Run discover() first.")
            return

        if verbose:
            print(f"  Computing IT vectors for {n} images...")

        # Build IT matrix.
        X = np.zeros((n, self._it_vector_size), dtype=np.float32)
        for i in range(n):
            h, w = images[i].shape[:2]
            self.set_object_origin((w / 2.0, h / 2.0))
            fixations = fixation_fn(images[i])
            X[i] = self.compute_it(images[i], fixations)
            if verbose and (i + 1) % 1000 == 0:
                print(f"    {i+1}/{n}")

        # One-hot labels.
        Y = np.zeros((n, n_classes), dtype=np.float32)
        for i in range(n):
            Y[i, int(labels[i])] = 1.0

        # Least-squares: W = (X^T X + λI)^{-1} X^T Y (ridge regression).
        # This is the "one-shot learning" — no epochs, no gradient descent.
        lam = 0.01  # small regularization
        XtX = X.T @ X + lam * np.eye(self._it_vector_size)
        XtY = X.T @ Y
        self._weights = np.linalg.solve(XtX, XtY).T  # (n_classes, it_dim)
        self._bias = np.zeros(n_classes, dtype=np.float32)

        # Training accuracy.
        scores = X @ self._weights.T + self._bias
        preds = scores.argmax(axis=1)
        acc = (preds == labels).mean() * 100
        if verbose:
            print(f"  Classifier trained: {acc:.1f}% training accuracy")

    # --- Phase 3: Classify ---

    def classify(self, image: np.ndarray,
                 fixations: list[tuple]) -> tuple[int, np.ndarray]:
        """Classify an image: IT vector → linear readout → predicted class."""
        h, w = image.shape[:2]
        self.set_object_origin((w / 2.0, h / 2.0))
        vec = self.compute_it(image, fixations)
        scores = self._weights @ vec + self._bias
        return int(scores.argmax()), scores


class FovealExplorer:
    """Drives the ventral stream through saccadic exploration."""

    def __init__(self, eye: Eye, stream: VentralStream, n_fixations: int = 9):
        self.eye = eye
        self.stream = stream
        self.n_fixations = n_fixations

    def get_fixations(self, image: np.ndarray) -> list[tuple]:
        return self.eye.cardinal_scan(image, step=5, n_fixations=self.n_fixations)

    def explore_unlabeled(self, image: np.ndarray):
        """Phase 1: explore image, accumulate triples."""
        h, w = image.shape[:2]
        self.stream.set_object_origin((w / 2.0, h / 2.0))
        fixations = self.get_fixations(image)

        fx, fy = fixations[0]
        self.eye.fixate(float(fx), float(fy))
        self.stream.explore_fixation(image, None)

        for fx, fy in fixations[1:]:
            self.eye.saccade_to(float(fx), float(fy))
            disp = self.eye.last_displacement
            self.stream.explore_fixation(image, disp)
