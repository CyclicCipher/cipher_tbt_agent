"""vision_pipeline.py — Sequence learning applied to image patches.

Broca's area principle: one algorithm for all domains.
VisionLearner wraps SequenceLearner with image→patch-sequence conversion.

    from vision_pipeline import VisionLearner
    import numpy as np

    learner = VisionLearner(patch_size=16, n_clusters=64)
    learner.fit_images(images)           # list of HxW numpy arrays
    patch_id = learner.predict_patch([h1, h2])   # next patch hash

Patch extraction reuses _to_gray_f32, _extract_patches, _quantize from
modalities/visual_symbol.py (same as discover_chars.phase2_glyphs).

Usage (demo):
    python vision_pipeline.py --demo          # synthetic random patches
    python vision_pipeline.py --corpus IAM    # real OCR line images

Architecture:
    image  → patch rows (scanline order)
    patches → patch hashes (quantized signature)
    hashes  → SequenceLearner.fit(sequences)
    ↓
    E0-E6: category chain, soft retrieval, paradigmatic axis
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, List

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if os.path.join(_HERE, '..') not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, '..'))

from sequence_pipeline import SequenceLearner

# Patch extraction constants (same as discover_chars.py)
_PATCH_SIZE = 16
_QUANT_BITS = 3


def _load_patch_fns():
    """Import patch helper functions from modalities.visual_symbol."""
    try:
        from modalities.visual_symbol import _to_gray_f32, _extract_patches, _quantize
        return _to_gray_f32, _extract_patches, _quantize
    except ImportError:
        return None, None, None


def image_to_sequence(
    image,
    patch_size: int = _PATCH_SIZE,
    quant_bits: int = _QUANT_BITS,
) -> list[int]:
    """Convert a single image to a scanline sequence of patch hashes.

    Args:
        image:      HxW or HxWxC numpy array (float32 or uint8).
        patch_size: Pixel size of each square patch.
        quant_bits: Quantization bits for patch hash (3 → 8 bins/channel).

    Returns:
        List of integer patch hashes in scanline order.
        Empty list if patch extraction fails or image is too small.
    """
    try:
        import numpy as np
    except ImportError:
        return []

    _to_gray_f32, _extract_patches, _quantize = _load_patch_fns()
    if _extract_patches is None:
        return []

    try:
        arr = np.array(image, dtype=np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        gray = _to_gray_f32(arr) if arr.ndim > 2 else arr
        patches = _extract_patches(gray, patch_size, patch_size)
        if not patches:
            return []
        return [_quantize(_to_gray_f32(p), quant_bits) for p in patches]
    except Exception:
        return []


class VisionLearner:
    """Sequence learning applied to image patch sequences.

    Broca's area principle: the same E0-E6 algorithm that discovers
    syntactic structure in text discovers spatial structure in images.

    Visual patches in scanline order form a sequence — exactly like words
    in a sentence.  The category chain learns which patch types follow which,
    discovering visual regularities (edges, textures, glyph boundaries)
    without any labels.

    Usage:
        import numpy as np
        learner = VisionLearner(patch_size=16, n_clusters=64)

        # Train on list of images
        learner.fit_images(images)

        # Predict next patch hash given context
        next_hash = learner.predict_patch([hash1, hash2])

        # Calibrate to a specific visual style (fine-tune)
        learner.calibrate(new_frame)
    """

    def __init__(self, patch_size: int = _PATCH_SIZE,
                 quant_bits: int = _QUANT_BITS,
                 n_clusters: int = 64):
        self.patch_size = patch_size
        self.quant_bits = quant_bits
        self.learner = SequenceLearner(n_clusters=n_clusters)
        self._n_images = 0

    def fit_images(self, images: list, verbose: bool = True) -> None:
        """Train E1-E3 on a list of images.

        Args:
            images:  List of HxW or HxWxC numpy arrays (float32 or uint8).
            verbose: Print progress.
        """
        sequences = []
        n_empty = 0
        for img in images:
            seq = image_to_sequence(img, self.patch_size, self.quant_bits)
            if seq:
                sequences.append(seq)
            else:
                n_empty += 1

        self._n_images = len(sequences)
        if verbose:
            print(f'  VisionLearner: {len(sequences)} images, '
                  f'{n_empty} empty/failed')
            total_patches = sum(len(s) for s in sequences)
            print(f'  Total patches: {total_patches:,}  '
                  f'Patch size: {self.patch_size}px  '
                  f'Quant bits: {self.quant_bits}')

        if not sequences:
            print('  ERROR: no patches extracted — check images and PIL.')
            return

        self.learner.fit(sequences, verbose=verbose)

    def predict_patch(self, context_hashes: list[int]) -> int | None:
        """Predict next patch hash given 2 previous patch hashes."""
        if len(context_hashes) < 2:
            return None
        return self.learner.predict(tuple(context_hashes[-2:]))

    def calibrate(self, frame, verbose: bool = False) -> None:
        """Fine-tune to a specific visual style (e.g. game frame, font).

        Teaches additional patch transitions from a single image,
        then re-runs E1-E3 fit to update the category chain.

        Args:
            frame:   HxW or HxWxC numpy array.
            verbose: Print progress.
        """
        seq = image_to_sequence(frame, self.patch_size, self.quant_bits)
        if not seq:
            if verbose:
                print('  calibrate: no patches extracted from frame.')
            return
        self.learner.fit([seq], verbose=verbose)

    def evaluate_images(self, test_images: list,
                        train_images: list | None = None,
                        verbose: bool = True) -> dict:
        """Evaluate patch prediction accuracy on test images.

        Args:
            test_images:  List of numpy arrays to evaluate on.
            train_images: If provided, marks (patch2, patch3) pairs seen
                          in training for the unseen-pair split.
            verbose:      Print results table.

        Returns:
            Same dict as SequenceLearner.evaluate().
        """
        test_seqs = [
            image_to_sequence(img, self.patch_size, self.quant_bits)
            for img in test_images
        ]
        test_seqs = [s for s in test_seqs if s]

        train_pairs: set[tuple] = set()
        if train_images:
            for img in train_images:
                seq = image_to_sequence(img, self.patch_size, self.quant_bits)
                for i in range(len(seq) - 1):
                    train_pairs.add((seq[i], seq[i + 1]))

        return self.learner.evaluate(test_seqs, train_pairs=train_pairs,
                                     verbose=verbose)


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

def _synthetic_images(n: int = 50, h: int = 64, w: int = 128, seed: int = 42):
    """Generate synthetic grayscale images with stripe patterns."""
    try:
        import numpy as np
    except ImportError:
        print('ERROR: numpy required for demo.')
        return []

    rng = np.random.default_rng(seed)
    images = []
    for i in range(n):
        img = rng.random((h, w), dtype=np.float32)
        # Add horizontal stripe pattern to create spatial regularity
        period = rng.integers(8, 32)
        stripe = (np.arange(h) % period < period // 2).astype(np.float32)
        img = 0.5 * img + 0.5 * stripe[:, None]
        images.append(img)
    return images


def main() -> None:
    p = argparse.ArgumentParser(
        description='VisionLearner: E0-E6 on image patch sequences.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--demo',     action='store_true',
                   help='Run on synthetic random-stripe images')
    p.add_argument('--corpus',   default=None,
                   help='Path to directory of .png images')
    p.add_argument('--n_images', type=int, default=100)
    p.add_argument('--n_clusters', type=int, default=32)
    p.add_argument('--patch_size', type=int, default=16)
    args = p.parse_args()

    if args.demo:
        print('=== VisionLearner Demo (synthetic stripe images) ===')
        images = _synthetic_images(n=args.n_images)
        if not images:
            return
        n_train = int(len(images) * 0.8)
        train_imgs = images[:n_train]
        test_imgs  = images[n_train:]

        learner = VisionLearner(patch_size=args.patch_size,
                                n_clusters=args.n_clusters)
        learner.fit_images(train_imgs, verbose=True)
        learner.evaluate_images(test_imgs, train_imgs, verbose=True)

    elif args.corpus:
        import glob
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            print('ERROR: PIL and numpy required for corpus mode.')
            return

        paths = glob.glob(os.path.join(args.corpus, '**', '*.png'), recursive=True)
        if not paths:
            print(f'ERROR: no .png files found in {args.corpus!r}')
            return

        paths = paths[:args.n_images]
        images = []
        for path in paths:
            try:
                img = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
                images.append(img)
            except Exception:
                pass

        print(f'Loaded {len(images)} images from {args.corpus!r}')
        n_train = int(len(images) * 0.8)
        train_imgs = images[:n_train]
        test_imgs  = images[n_train:]

        learner = VisionLearner(patch_size=args.patch_size,
                                n_clusters=args.n_clusters)
        learner.fit_images(train_imgs, verbose=True)
        learner.evaluate_images(test_imgs, train_imgs, verbose=True)

    else:
        p.print_help()
        print('\nRun with --demo for a quick sanity check.')


if __name__ == '__main__':
    main()
