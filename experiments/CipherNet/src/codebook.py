"""Vector Quantization codebook for patch encoding.

Uses scikit-learn KMeans for fast codebook training (~2s for 50K patches).
Converts continuous pixel patches into discrete feature tokens for
symbolic columns.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans


class PatchCodebook:
    """VQ codebook: k-means centroids for patch encoding."""

    def __init__(self, n_codes: int = 256, seed: int = 42):
        self.n_codes = n_codes
        self.seed = seed
        self.centroids: np.ndarray | None = None
        self._kmeans: MiniBatchKMeans | None = None

    def fit(self, patches: np.ndarray, verbose: bool = True):
        """K-means on flattened patch vectors. Fast (~2s for 50K patches)."""
        if patches.ndim == 3:
            flat = patches.reshape(patches.shape[0], -1).astype(np.float32)
        else:
            flat = patches.astype(np.float32)

        if verbose:
            print(f"  Codebook: fitting {self.n_codes} codes on "
                  f"{flat.shape[0]} patches (dim={flat.shape[1]})...")

        self._kmeans = MiniBatchKMeans(
            n_clusters=self.n_codes,
            random_state=self.seed,
            batch_size=1024,
            n_init=3,
        )
        self._kmeans.fit(flat)
        self.centroids = self._kmeans.cluster_centers_

        if verbose:
            print(f"  Codebook ready: {self.n_codes} codes")

    def encode(self, patch: np.ndarray) -> str:
        """Encode a single patch → nearest centroid ID as string."""
        flat = patch.flatten().astype(np.float32).reshape(1, -1)
        idx = self._kmeans.predict(flat)[0]
        return f"v{idx}"

    def encode_batch(self, patches: np.ndarray) -> np.ndarray:
        """Encode a batch of patches → centroid ID array (int)."""
        if patches.ndim == 3:
            flat = patches.reshape(patches.shape[0], -1).astype(np.float32)
        else:
            flat = patches.astype(np.float32)
        return self._kmeans.predict(flat)
