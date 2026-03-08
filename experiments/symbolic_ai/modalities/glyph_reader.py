"""glyph_reader.py -- Phase S.0a: Learned character recognition.

Architecture: GlyphReader trains on rendered character images using the same
distributional clustering machinery as Phase O (text) and Phase R6 (pixel patches).
No external OCR library, no hardcoded font templates.

Design principle: THE MODEL MUST NEVER BE DESIGNED AROUND A SPECIFIC TASK.
THE MODEL MUST BE GENERAL. No font-specific logic. Structure from distributions.
"""
from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from modalities.visual_symbol import _to_gray_f32, _extract_patches, _quantize
from synthesis import discover_categories_from_dists  # type: ignore[import]


_DEFAULT_PATCH_SIZE: int = 16
_DEFAULT_QUANT_BITS: int = 3
_DEFAULT_N_CLUSTERS: int = 128

_DEFAULT_CHARSET: str = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " !#%()*+,-./:;<=>?@[]^_{}|~"
    "+-*/="
)


@dataclass
class GlyphResult:
    """Result of reading a single pixel patch."""
    char:        str
    confidence:  float
    x_center:    int
    y_center:    int
    patch_w:     int
    patch_h:     int

    @property
    def is_space(self) -> bool:
        return self.char in (" ", "")

    @property
    def scale_relative(self) -> float:
        """Relative scale vs expected patch_size. < 0.7 -> subscript candidate."""
        return (self.patch_w * self.patch_h) ** 0.5 / _DEFAULT_PATCH_SIZE


class GlyphReader:
    """Trained character recogniser built on distributional patch clustering."""

    def __init__(
        self,
        patch_size:  int = _DEFAULT_PATCH_SIZE,
        quant_bits:  int = _DEFAULT_QUANT_BITS,
        n_clusters:  int = _DEFAULT_N_CLUSTERS,
        model_path:  str = "glyph_reader.pkl",
    ) -> None:
        self.patch_size  = patch_size
        self.quant_bits  = quant_bits
        self.n_clusters  = n_clusters
        self.model_path  = model_path
        self._centroids:       Dict[int, np.ndarray] = {}
        self._labels:          Dict[int, str]        = {}
        self._label_conf:      Dict[int, float]      = {}
        self._hash_to_cluster: Dict[str, int]        = {}
        self._trained = False

    def train(
        self,
        chars:   str             = _DEFAULT_CHARSET,
        fonts:   Optional[List]  = None,
        sizes:   Tuple[int, ...] = (10, 12, 14, 16, 20),
        augment: bool            = True,
        verbose: bool            = True,
    ) -> "GlyphReader":
        """Train on PIL-rendered character images."""
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import]

        if fonts is None:
            fonts = [ImageFont.load_default()]

        if verbose:
            print(f"GlyphReader.train(): rendering {len(chars)} chars x "
                  f"{len(fonts)} fonts x {len(sizes)} sizes ...")

        samples: List[Tuple[str, np.ndarray]] = []
        for font in fonts:
            for size in sizes:
                try:
                    scaled_font = ImageFont.truetype(font.path, size=size)  # type: ignore[attr-defined]
                except Exception:
                    scaled_font = font
                for ch in chars:
                    patch = self._render_char(ch, scaled_font, augment=False)
                    if patch is not None:
                        samples.append((ch, patch))
                    if augment:
                        for _ in range(2):
                            a = self._render_char(ch, scaled_font, augment=True)
                            if a is not None:
                                samples.append((ch, a))

        if verbose:
            print(f"  {len(samples)} training patches")

        ctx_counts: Dict[str, Dict[str, int]] = defaultdict(Counter)
        patch_counts: Dict[str, int] = Counter()
        hash_to_chars: Dict[str, List[str]] = defaultdict(list)

        hashes = [_quantize(_to_gray_f32(p), self.quant_bits) for _, p in samples]
        labels_seq = [ch for ch, _ in samples]

        for i, (h, ch) in enumerate(zip(hashes, labels_seq)):
            patch_counts[h] += 1
            hash_to_chars[h].append(ch)
            if i > 0:
                prev = hashes[i - 1]
                ctx_counts[h][prev] += 1
                ctx_counts[prev][h] += 1

        dists:        Dict = {}
        input_counts: Dict = {}
        for h, ctx in ctx_counts.items():
            total = sum(ctx.values())
            if total < 1:
                continue
            dists[(h,)]        = {(c,): cnt / total for c, cnt in ctx.items()}
            input_counts[(h,)] = patch_counts[h]

        # --- Direct nearest-centroid classifier (labeled training data) ---
        # Build per-character pixel centroid from labeled samples.
        # Each unique hash is assigned to the character that rendered it most often.
        char_pixel_sums: Dict[str, np.ndarray] = {}
        char_pixel_cnts: Dict[str, int]        = {}

        for (ch, patch), h in zip(samples, hashes):
            flat = _to_gray_f32(patch).ravel()
            if ch not in char_pixel_sums:
                char_pixel_sums[ch] = np.zeros(len(flat), dtype=np.float64)
                char_pixel_cnts[ch] = 0
            char_pixel_sums[ch] += flat
            char_pixel_cnts[ch] += 1

        # Assign integer cluster IDs to each character
        char_list = sorted(char_pixel_sums.keys())
        char_to_cid: Dict[str, int] = {ch: i for i, ch in enumerate(char_list)}

        # Build centroids and labels
        self._centroids  = {}
        self._labels     = {}
        self._label_conf = {}
        for ch, cid in char_to_cid.items():
            n = char_pixel_cnts[ch]
            self._centroids[cid]  = (char_pixel_sums[ch] / n).astype(np.float32)
            self._labels[cid]     = ch
            self._label_conf[cid] = 1.0

        # Build hash -> cluster: assign each hash to the majority-vote character
        self._hash_to_cluster = {}
        for h in hash_to_chars:
            votes: Counter = Counter(hash_to_chars[h])
            best_ch = votes.most_common(1)[0][0]
            self._hash_to_cluster[h] = char_to_cid[best_ch]

        self._trained = True
        k = len(char_list)
        n_labelled = sum(1 for lbl in self._labels.values() if lbl)
        if verbose:
            print(f"  Training complete: {k} character centroids, "
                  f"{n_labelled} labelled ({n_labelled/max(1,k):.0%})")
        return self

    def read_patch(
        self,
        patch:    np.ndarray,
        x_center: int = 0,
        y_center: int = 0,
    ) -> GlyphResult:
        """Classify a pixel patch -> GlyphResult."""
        if not self._trained:
            return GlyphResult("", 0.0, x_center, y_center,
                               patch.shape[1] if patch.ndim > 1 else 1,
                               patch.shape[0])
        gray  = _to_gray_f32(patch)
        h_key = _quantize(gray, self.quant_bits)
        cid   = self._hash_to_cluster.get(h_key)
        if cid is None:
            cid = self._nearest_centroid(gray.ravel())
        char = self._labels.get(cid, "")
        conf = self._label_conf.get(cid, 0.0)
        ph, pw = patch.shape[:2]
        return GlyphResult(char, conf, x_center, y_center, pw, ph)

    def _nearest_centroid(self, flat_patch: np.ndarray) -> int:
        """Find the cluster with the nearest centroid (L2 distance)."""
        if not self._centroids:
            return 0
        plen = len(flat_patch)
        best_cid, best_d = 0, float("inf")
        for cid, centroid in self._centroids.items():
            c = centroid[:plen]
            if len(c) != plen:
                continue
            d = float(np.sum((flat_patch - c) ** 2))
            if d < best_d:
                best_d, best_cid = d, cid
        return best_cid

    def read_patch_topk(
        self,
        patch:    np.ndarray,
        k:        int = 8,
        x_center: int = 0,
        y_center: int = 0,
    ) -> List[Tuple[str, float]]:
        """Return top-K (char, score) pairs sorted by score descending.

        Scores are softmax-normalised over L2 distances to character centroids.
        Use this instead of read_patch() when you want a distribution over
        candidates (e.g. for Viterbi decoding with a language prior).

        Parameters
        ----------
        patch     Grayscale pixel patch (any dtype; will be converted).
        k         Number of top candidates to return.

        Returns
        -------
        List of (char, probability) tuples, length <= k, sorted best-first.
        """
        if not self._trained:
            return [("", 1.0)]

        gray = _to_gray_f32(patch)
        flat = gray.ravel()
        plen = len(flat)

        # Compute L2 distance to every centroid
        dists: List[Tuple[int, float]] = []
        for cid, centroid in self._centroids.items():
            c = centroid[:plen]
            if len(c) != plen:
                continue
            d = float(np.sum((flat - c) ** 2))
            dists.append((cid, d))

        if not dists:
            return [("", 1.0)]

        dists.sort(key=lambda x: x[1])
        top_k = dists[:k]

        # Softmax over negative distances (with numeric stability via shift)
        min_d = top_k[0][1]
        # Adaptive temperature: scale by mean distance so scores aren't
        # collapsed to a single spike when all distances are small.
        T_scale = max(1.0, sum(d for _, d in top_k) / len(top_k))
        scores = [(cid, math.exp(-(d - min_d) / T_scale)) for cid, d in top_k]
        total = sum(s for _, s in scores) or 1.0

        result = []
        for cid, s in scores:
            ch = self._labels.get(cid, "")
            if ch:
                result.append((ch, s / total))
        return result if result else [("", 1.0)]

    def calibrate(
        self,
        frame: np.ndarray,
        n_iters: int = 3,
        verbose: bool = False,
    ) -> "GlyphReader":
        """Fine-tune cluster centroids to the rendering style in frame."""
        if not self._trained:
            return self
        gray    = _to_gray_f32(frame)
        patches = _extract_patches(gray, self.patch_size, self.patch_size)
        if not patches:
            return self
        for _ in range(n_iters):
            cluster_sums: Dict[int, np.ndarray] = {}
            cluster_cnts: Dict[int, int]        = {}
            for p in patches:
                h_key = _quantize(_to_gray_f32(p), self.quant_bits)
                flat  = _to_gray_f32(p).ravel()
                cid   = self._hash_to_cluster.get(h_key, self._nearest_centroid(flat))
                if cid not in cluster_sums:
                    cluster_sums[cid] = np.zeros_like(self._centroids.get(
                        cid, np.zeros(len(flat), dtype=np.float32)))
                cluster_sums[cid] += flat
                cluster_cnts[cid]  = cluster_cnts.get(cid, 0) + 1
            for cid, s in cluster_sums.items():
                new_c = (s / cluster_cnts[cid]).astype(np.float32)
                if cid in self._centroids:
                    self._centroids[cid] = 0.8 * self._centroids[cid] + 0.2 * new_c
                else:
                    self._centroids[cid] = new_c
        if verbose:
            print(f"GlyphReader.calibrate(): updated {len(cluster_sums)} centroids")
        return self

    def save(self, path: Optional[str] = None) -> None:
        """Save trained model to a pickle file."""
        path = path or self.model_path
        with open(path, "wb") as f:
            pickle.dump({
                "patch_size":       self.patch_size,
                "quant_bits":       self.quant_bits,
                "n_clusters":       self.n_clusters,
                "centroids":        self._centroids,
                "labels":           self._labels,
                "label_conf":       self._label_conf,
                "hash_to_cluster":  self._hash_to_cluster,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str = "glyph_reader.pkl") -> "GlyphReader":
        """Load a previously trained GlyphReader from a pickle file."""
        with open(path, "rb") as f:
            d = pickle.load(f)
        reader = cls(
            patch_size = d["patch_size"],
            quant_bits = d["quant_bits"],
            n_clusters = d["n_clusters"],
            model_path = path,
        )
        reader._centroids       = d["centroids"]
        reader._labels          = d["labels"]
        reader._label_conf      = d["label_conf"]
        reader._hash_to_cluster = d["hash_to_cluster"]
        reader._trained         = True
        return reader

    def _render_char(
        self,
        char:    str,
        font,
        size:    int = 16,
        augment: bool = False,
    ) -> Optional[np.ndarray]:
        """Render a single character to a float32 greyscale patch."""
        try:
            from PIL import Image, ImageDraw, ImageFilter  # type: ignore[import]
            canvas_size = max(size * 3, self.patch_size * 2)
            img = Image.new("L", (canvas_size, canvas_size), color=255)
            draw = ImageDraw.Draw(img)
            draw.text((canvas_size // 4, canvas_size // 4), char, fill=0, font=font)
            if augment:
                import random
                if random.random() < 0.5:
                    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            arr = np.array(img, dtype=np.float32) / 255.0
            mask = arr < 0.9
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                return np.ones((self.patch_size, self.patch_size), dtype=np.float32)
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            crop = arr[r0:r1+1, c0:c1+1]
            from PIL import Image as _Im
            crop_img = _Im.fromarray((crop * 255).astype(np.uint8), mode="L")
            crop_img = crop_img.resize((self.patch_size, self.patch_size), _Im.NEAREST)
            patch = np.array(crop_img, dtype=np.float32) / 255.0
            if augment:
                patch += np.random.normal(0, 0.03, patch.shape).astype(np.float32)
                patch = np.clip(patch, 0.0, 1.0)
            return patch
        except Exception:
            return None

    @property
    def is_trained(self) -> bool:
        return self._trained

    def summary(self) -> str:
        if not self._trained:
            return "GlyphReader: untrained"
        n_lab = sum(1 for lbl in self._labels.values() if lbl)
        n_tot = len(self._labels)
        top5  = sorted(self._label_conf.items(), key=lambda x: -x[1])[:5]
        top_str = ", ".join(
            f"char={self._labels[cid]!r} ({conf:.0%})" for cid, conf in top5)
        return (
            f"GlyphReader: {n_tot} clusters, {n_lab} labelled ({n_lab/max(1,n_tot):.0%})\n"
            f"  Top-5 confident: {top_str}"
        )
