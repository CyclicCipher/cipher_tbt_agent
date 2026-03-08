"""vision_pipeline.py — Sequence learning applied to image patches.

Broca's area principle: one algorithm for all domains.
VisionLearner wraps SequenceLearner with image→patch-sequence conversion.
FovealVisionLearner adds biologically-inspired attention: a coarse peripheral
scan identifies salient locations via prediction error, then a pixel-accurate
foveal scan reads each attended region in full detail.

    from vision_pipeline import VisionLearner, FovealVisionLearner
    import numpy as np

    # Coarse whole-image scan (original):
    learner = VisionLearner(patch_size=16, n_clusters=64)
    learner.fit_images(images)

    # Foveal attention (new — handles complex scenes with backgrounds):
    foveal = FovealVisionLearner()
    foveal.fit_images(images)
    fixations = foveal.fixate(img)   # [{center_px, saliency, sequence}]

Patch extraction reuses _to_gray_f32, _extract_patches, _quantize from
modalities/visual_symbol.py (same as discover_chars.phase2_glyphs).

Usage (demo):
    python vision_pipeline.py --demo           # coarse VisionLearner
    python vision_pipeline.py --demo --foveal  # FovealVisionLearner
    python vision_pipeline.py --corpus IAM     # real OCR line images

Architecture (FovealVisionLearner):
    Peripheral scan (full image, coarse patches)
        → prediction error per patch → saliency map
        → non-max suppression → N fixation points
    Foveal scan (crop around each fixation, fine_patch=1 pixel)
        → interleaved [pos_token, content_hash, pos_token, ...] sequence
        → SequenceLearner learns spatial grammar of attended objects
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
# Foveal attention — saliency, fixation selection, pixel-accurate crop
# ---------------------------------------------------------------------------

def saliency_map(image, learner: 'VisionLearner', patch_size: int) -> 'np.ndarray':
    """Compute prediction-error saliency at each coarse patch position.

    Runs a scanline scan at ``patch_size`` resolution, computing
    −log₂ P(patch_i | patch_{i-2}, patch_{i-1}) at every position.
    High values = unexpected patches = candidate fixation targets.

    Biological basis: V1 prediction errors propagate upward (Rao & Ballard 1999);
    attention amplifies those error signals, guiding the next saccade.

    Args:
        image:       HxW or HxWxC numpy array (float32 or uint8).
        learner:     Trained VisionLearner (must have called fit_images first).
        patch_size:  Coarse patch size in pixels (typically 16–32).

    Returns:
        2D float32 array of shape (H // patch_size, W // patch_size).
        None if numpy is unavailable or image is too small.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    _to_gray_f32, _extract_patches, _quantize = _load_patch_fns()
    if _to_gray_f32 is None:
        return None

    try:
        arr = np.array(image, dtype=np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        gray = _to_gray_f32(arr) if arr.ndim > 2 else arr
    except Exception:
        return None

    H, W = gray.shape
    rows = H // patch_size
    cols = W // patch_size
    if rows == 0 or cols == 0:
        return None

    # Extract all patch hashes in scanline order
    hashes = []
    for r in range(rows):
        for c in range(cols):
            patch = gray[r * patch_size:(r + 1) * patch_size,
                         c * patch_size:(c + 1) * patch_size]
            hashes.append(_quantize(_to_gray_f32(patch), learner.quant_bits))

    # Compute surprise at each position using trained SequenceLearner
    sal = np.full(rows * cols, 1.0, dtype=np.float32)  # default: medium
    for i in range(2, len(hashes)):
        lp = learner.learner.logprob(hashes[i - 2], hashes[i - 1], hashes[i])
        # None = completely unseen context → maximum surprise
        sal[i] = -lp if lp is not None else 5.0

    return sal.reshape(rows, cols)


def select_fixations(saliency: 'np.ndarray', patch_size: int,
                     n: int = 3, min_dist_px: int = 32) -> list:
    """Select top-N fixation centers via non-max suppression on the saliency map.

    Mimics the foveation policy of the brain's frontal eye fields (FEF):
    pick the most salient location, suppress its neighbourhood, repeat.

    Args:
        saliency:     2D float32 from saliency_map(), shape (rows, cols).
        patch_size:   Pixels per patch (used to convert grid → pixel coords).
        n:            Maximum number of fixation points to return.
        min_dist_px:  Minimum pixel distance between fixation centres.

    Returns:
        List of (px, py) tuples in pixel coordinates (centre of each selected
        patch). Ordered by descending saliency.
    """
    try:
        import numpy as np
    except ImportError:
        return []

    if saliency is None or saliency.size == 0:
        return []

    rows, cols = saliency.shape
    min_dist_patches = max(1, min_dist_px // patch_size)
    remaining = saliency.copy()
    fixations = []

    for _ in range(n):
        if remaining.max() <= 0:
            break
        r, c = np.unravel_index(remaining.argmax(), remaining.shape)
        # Convert patch-grid indices → pixel centre
        px = int(c * patch_size + patch_size // 2)
        py = int(r * patch_size + patch_size // 2)
        fixations.append((px, py))
        # Suppress neighbourhood (non-max suppression)
        r0 = max(0, r - min_dist_patches)
        r1 = min(rows, r + min_dist_patches + 1)
        c0 = max(0, c - min_dist_patches)
        c1 = min(cols, c + min_dist_patches + 1)
        remaining[r0:r1, c0:c1] = -1.0

    return fixations


def foveal_sequence(image,
                    px_cx: int, px_cy: int,
                    radius_px: int = 48,
                    fine_patch: int = 1,
                    n_pos_bins: int = 8,
                    quant_bits: int = _QUANT_BITS) -> list:
    """Extract a pixel-accurate sequence from a foveal crop with position tokens.

    Each sub-patch in the crop produces two tokens:
        [pos_token, content_hash, pos_token, content_hash, ...]

    where pos_token = 'P{row_bin},{col_bin}' (coarse spatial bin within the
    crop) and content_hash is the quantized grayscale value of the sub-patch.

    Position tokens are ordinary tokens in the SequenceLearner vocabulary — the
    category chain naturally learns which positions predict which content, encoding
    the spatial grammar of whatever object falls in the foveal crop.

    At fine_patch=1 (pixel-accurate): each content hash encodes a single pixel
    as a 3-bit grey level (hex string '00'..'07').  Enough to distinguish text
    strokes, fine edges, fur direction, pupil shape, etc.

    Args:
        image:       Full image (HxW or HxWxC, float32 [0,1] or uint8).
        px_cx, px_cy: Fixation centre in pixel coordinates.
        radius_px:   Half-width of the foveal crop in pixels.
        fine_patch:  Sub-patch size in pixels (1 = pixel-accurate, 2 = 2×2).
        n_pos_bins:  Position grid bins per axis (8 → 8×8 = 64 positions).
        quant_bits:  Quantization depth for content hash (3 → 8 grey levels).

    Returns:
        Flat list of alternating position strings and content hash strings.
        Empty if the crop falls outside the image or PIL/numpy is missing.
    """
    _to_gray_f32, _extract_patches, _quantize = _load_patch_fns()
    if _to_gray_f32 is None:
        return []

    try:
        import numpy as np
    except ImportError:
        return []

    try:
        arr = np.array(image, dtype=np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        gray = _to_gray_f32(arr) if arr.ndim > 2 else arr
    except Exception:
        return []

    H, W = gray.shape
    r0 = max(0, px_cy - radius_px)
    r1 = min(H, px_cy + radius_px)
    c0 = max(0, px_cx - radius_px)
    c1 = min(W, px_cx + radius_px)
    crop = gray[r0:r1, c0:c1]
    ch, cw = crop.shape

    if ch < fine_patch or cw < fine_patch:
        return []

    crop_rows = ch // fine_patch
    crop_cols = cw // fine_patch
    tokens: list = []

    for r in range(crop_rows):
        for c in range(crop_cols):
            # Coarse position bin (0..n_pos_bins-1 per axis)
            pos_r = int(r / crop_rows * n_pos_bins)
            pos_c = int(c / crop_cols * n_pos_bins)
            pos_token = f'P{pos_r},{pos_c}'

            # Sub-patch content hash
            patch = crop[r * fine_patch:(r + 1) * fine_patch,
                         c * fine_patch:(c + 1) * fine_patch]
            content_hash = _quantize(_to_gray_f32(patch), quant_bits)

            tokens.append(pos_token)
            tokens.append(content_hash)

    return tokens


class FovealVisionLearner:
    """Biologically-inspired dual-scale vision learner.

    Peripheral scan (full image, coarse patches):
        Identifies salient locations via prediction error — unexpected patches
        are candidates for fixation, just as the brain uses prediction error
        from V1 to guide saccades via the frontal eye fields.

    Foveal scan (crop around each fixation, pixel-accurate):
        Interleaves position tokens with 1-pixel content hashes.  The
        SequenceLearner learns the spatial grammar of each attended region.
        "Cat" = a repeating sequence of fur patches at 'P3,2', whisker patches
        at 'P4,3', etc. — the object's spatial signature.

    This architecture solves the "cat in a meadow" problem: grass is highly
    predictable (low prediction error, low saliency), so the fovea never
    fixates on it.  The cat boundary is maximally surprising — fixation target.

    Usage:
        learner = FovealVisionLearner()
        learner.fit_images(images)

        # Inspect fixations on a new image
        for fix in learner.fixate(img):
            print(fix['center_px'], fix['saliency'], len(fix['sequence']))

        # Evaluate foveal prediction accuracy
        learner.evaluate_images(test_imgs, train_imgs)
    """

    def __init__(self,
                 peripheral_patch: int = 32,
                 foveal_patch: int = 1,
                 foveal_radius_px: int = 48,
                 n_fixations: int = 3,
                 n_pos_bins: int = 8,
                 n_peripheral_clusters: int = 32,
                 n_foveal_clusters: int = 64,
                 quant_bits: int = _QUANT_BITS):
        """
        Args:
            peripheral_patch:      Coarse patch size for saliency map (pixels).
            foveal_patch:          Fine patch size for foveal crop (1 = pixel-accurate).
            foveal_radius_px:      Half-width of foveal crop in pixels.
            n_fixations:           Max fixation points per image.
            n_pos_bins:            Position grid bins per axis (8 → 8×8 = 64 tokens).
            n_peripheral_clusters: Cluster count for peripheral VisionLearner.
            n_foveal_clusters:     Cluster count for foveal SequenceLearner.
            quant_bits:            Quantization depth shared by both learners.
        """
        self.peripheral_patch  = peripheral_patch
        self.foveal_patch      = foveal_patch
        self.foveal_radius_px  = foveal_radius_px
        self.n_fixations       = n_fixations
        self.n_pos_bins        = n_pos_bins
        self.quant_bits        = quant_bits

        self.peripheral = VisionLearner(patch_size=peripheral_patch,
                                        quant_bits=quant_bits,
                                        n_clusters=n_peripheral_clusters)
        self.foveal = SequenceLearner(n_clusters=n_foveal_clusters)

    def fit_images(self, images: list, verbose: bool = True) -> None:
        """Train both learners on a list of images.

        Phase 1: Train peripheral VisionLearner on full coarse scans.
        Phase 2: For each image, extract foveal sequences at salient locations
                 and train the foveal SequenceLearner on those crops.
        """
        if verbose:
            print(f'  FovealVisionLearner: {len(images)} images  '
                  f'peripheral={self.peripheral_patch}px  '
                  f'foveal={self.foveal_patch}px  '
                  f'radius={self.foveal_radius_px}px  '
                  f'fixations={self.n_fixations}')

        # --- Phase 1: peripheral ---
        if verbose:
            print('  [Phase 1] Training peripheral learner...')
        self.peripheral.fit_images(images, verbose=verbose)

        # --- Phase 2: foveal ---
        if verbose:
            print('  [Phase 2] Extracting foveal sequences at salient locations...')
        foveal_seqs: list[list] = []
        n_empty = 0
        for img in images:
            sal = saliency_map(img, self.peripheral, self.peripheral_patch)
            if sal is None:
                n_empty += 1
                continue
            fixations = select_fixations(sal, self.peripheral_patch,
                                         self.n_fixations,
                                         min_dist_px=self.foveal_radius_px)
            for px, py in fixations:
                seq = foveal_sequence(img, px, py, self.foveal_radius_px,
                                      self.foveal_patch, self.n_pos_bins,
                                      self.quant_bits)
                if seq:
                    foveal_seqs.append(seq)
            if not fixations:
                n_empty += 1

        if verbose:
            total_tok = sum(len(s) for s in foveal_seqs)
            print(f'  Foveal crops: {len(foveal_seqs)}  '
                  f'({n_empty} images had no fixations)  '
                  f'total tokens: {total_tok:,}')

        if foveal_seqs:
            if verbose:
                print('  [Phase 2] Training foveal SequenceLearner...')
            self.foveal.fit(foveal_seqs, verbose=verbose)
        else:
            if verbose:
                print('  WARNING: no foveal sequences extracted — '
                      'train peripheral learner first.')

    def fixate(self, image) -> list:
        """Run peripheral saliency + foveal extraction on a single image.

        Returns:
            List of dicts, one per fixation:
                'center_px':  (px, py) pixel coordinates of fixation centre.
                'saliency':   Peak saliency value at this location.
                'sequence':   Interleaved [pos_token, content_hash, ...] list.
        """
        sal = saliency_map(image, self.peripheral, self.peripheral_patch)
        if sal is None:
            return []
        fixations = select_fixations(sal, self.peripheral_patch,
                                     self.n_fixations,
                                     min_dist_px=self.foveal_radius_px)
        result = []
        for px, py in fixations:
            r_idx = py // self.peripheral_patch
            c_idx = px // self.peripheral_patch
            r_idx = min(r_idx, sal.shape[0] - 1)
            c_idx = min(c_idx, sal.shape[1] - 1)
            sal_val = float(sal[r_idx, c_idx])
            seq = foveal_sequence(image, px, py, self.foveal_radius_px,
                                  self.foveal_patch, self.n_pos_bins,
                                  self.quant_bits)
            result.append({'center_px': (px, py),
                           'saliency': sal_val,
                           'sequence': seq})
        return result

    def saliency(self, image) -> 'np.ndarray | None':
        """Return the 2D peripheral saliency map for a single image."""
        return saliency_map(image, self.peripheral, self.peripheral_patch)

    def evaluate_images(self, test_images: list,
                        train_images: list | None = None,
                        verbose: bool = True) -> dict:
        """Evaluate foveal prediction accuracy on test images.

        Runs fixate() on each image to get foveal sequences, then calls
        SequenceLearner.evaluate() with a train-pair set for the unseen split.

        Returns:
            Same dict structure as SequenceLearner.evaluate().
        """
        test_seqs = []
        for img in test_images:
            for fix in self.fixate(img):
                if fix['sequence']:
                    test_seqs.append(fix['sequence'])

        train_pairs: set = set()
        if train_images:
            for img in train_images:
                for fix in self.fixate(img):
                    seq = fix['sequence']
                    for i in range(len(seq) - 1):
                        train_pairs.add((seq[i], seq[i + 1]))

        if not test_seqs:
            if verbose:
                print('  evaluate_images: no foveal sequences from test images.')
            return {}

        return self.foveal.evaluate(test_seqs, train_pairs=train_pairs,
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
        description='VisionLearner / FovealVisionLearner: E0-E6 on image patches.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--demo',       action='store_true',
                   help='Run on synthetic random-stripe images')
    p.add_argument('--foveal',     action='store_true',
                   help='Use FovealVisionLearner (saliency + pixel-accurate fovea)')
    p.add_argument('--corpus',     default=None,
                   help='Path to directory of .png images')
    p.add_argument('--n_images',   type=int, default=100)
    p.add_argument('--n_clusters', type=int, default=32)
    p.add_argument('--patch_size', type=int, default=16,
                   help='Coarse patch size (VisionLearner) or peripheral patch size (FovealVisionLearner)')
    p.add_argument('--foveal_patch', type=int, default=1,
                   help='Foveal sub-patch size in pixels (1=pixel-accurate, 2=2×2)')
    p.add_argument('--foveal_radius', type=int, default=48,
                   help='Foveal crop half-width in pixels')
    p.add_argument('--n_fixations', type=int, default=3,
                   help='Max fixation points per image')
    args = p.parse_args()

    if args.demo:
        images = _synthetic_images(n=args.n_images)
        if not images:
            return
        n_train = int(len(images) * 0.8)
        train_imgs = images[:n_train]
        test_imgs  = images[n_train:]

        if args.foveal:
            print('=== FovealVisionLearner Demo (synthetic stripe images) ===')
            learner = FovealVisionLearner(
                peripheral_patch=args.patch_size,
                foveal_patch=args.foveal_patch,
                foveal_radius_px=args.foveal_radius,
                n_fixations=args.n_fixations,
                n_peripheral_clusters=args.n_clusters,
                n_foveal_clusters=args.n_clusters * 2,
            )
            learner.fit_images(train_imgs, verbose=True)
            print('\n--- Fixations on first test image ---')
            for i, fix in enumerate(learner.fixate(test_imgs[0])):
                print(f'  Fixation {i+1}: center={fix["center_px"]}  '
                      f'saliency={fix["saliency"]:.2f}  '
                      f'seq_len={len(fix["sequence"])}')
            print()
            learner.evaluate_images(test_imgs, train_imgs, verbose=True)
        else:
            print('=== VisionLearner Demo (synthetic stripe images) ===')
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

        if args.foveal:
            learner = FovealVisionLearner(
                peripheral_patch=args.patch_size,
                foveal_patch=args.foveal_patch,
                foveal_radius_px=args.foveal_radius,
                n_fixations=args.n_fixations,
                n_peripheral_clusters=args.n_clusters,
                n_foveal_clusters=args.n_clusters * 2,
            )
            learner.fit_images(train_imgs, verbose=True)
            learner.evaluate_images(test_imgs, train_imgs, verbose=True)
        else:
            learner = VisionLearner(patch_size=args.patch_size,
                                    n_clusters=args.n_clusters)
            learner.fit_images(train_imgs, verbose=True)
            learner.evaluate_images(test_imgs, train_imgs, verbose=True)

    else:
        p.print_help()
        print('\nRun with --demo or --demo --foveal for a quick sanity check.')


if __name__ == '__main__':
    main()
