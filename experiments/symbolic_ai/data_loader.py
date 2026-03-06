"""Data loading utilities for symbolic AI visual experiments.

Supports two sources:
  1. CIFAR-10  — auto-download + extract; cat class vs others.
  2. Folders   — load images from data/cats/ and data/negatives/ subfolders.

CIFAR-10 class index:
  0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
  5 dog, 6 frog, 7 horse, 8 ship, 9 truck

Usage:
    from data_loader import load_cifar10_cats, load_image_folder

    cats_tr, neg_tr, cats_te, neg_te = load_cifar10_cats()
    # Each element is a float32 numpy array shape (32, 32, 3), values in [0,1]

    pos, neg = load_image_folder()          # loads from data/cats/ + data/negatives/
    # Each element is a float32 numpy array (any size)
"""

from __future__ import annotations

import os
import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
DATA_DIR    = _HERE / 'data'
CIFAR10_DIR = DATA_DIR / 'cifar10'
CATS_DIR    = DATA_DIR / 'cats'
NEG_DIR     = DATA_DIR / 'negatives'

_CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
_CAT_CLASS   = 3   # CIFAR-10 class index for 'cat'

_IMAGE_EXTS  = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------

def download_cifar10(dest: Optional[Path] = None) -> Path:
    """Download and extract CIFAR-10 if not already present.

    Returns path to the extracted 'cifar-10-batches-py' directory.
    """
    dest = Path(dest or CIFAR10_DIR)
    dest.mkdir(parents=True, exist_ok=True)

    batches_dir = dest / 'cifar-10-batches-py'
    if batches_dir.exists():
        return batches_dir

    tar_path = dest / 'cifar10.tar.gz'
    print(f'Downloading CIFAR-10 (~170 MB) from {_CIFAR10_URL} ...')

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            print(f'  {pct}%', end='\r')

    urllib.request.urlretrieve(_CIFAR10_URL, tar_path, reporthook=_progress)
    print()

    print('Extracting ...')
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest)

    tar_path.unlink()
    print(f'CIFAR-10 ready at {batches_dir}')
    return batches_dir


def _load_cifar10_batch(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one CIFAR-10 batch file → (images NHWC float32, labels int)."""
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    # data is N×3072 uint8 in NCHW order; reshape to NHWC
    images = d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(d[b'labels'], dtype=np.int32)
    return images, labels


def load_cifar10_cats(
    data_dir: Optional[Path] = None,
    max_per_class_train: Optional[int] = None,
    max_per_class_test:  Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray],
           List[np.ndarray], List[np.ndarray]]:
    """Load CIFAR-10 cat vs non-cat splits.

    Auto-downloads if not present.

    Returns (cats_train, noncats_train, cats_test, noncats_test).
    Each element is a list of float32 numpy arrays, shape (32, 32, 3),
    values in [0.0, 1.0].

    Args:
        max_per_class_train: cap positive and negative training examples each.
        max_per_class_test:  cap positive and negative test examples each.
    """
    batches_dir = download_cifar10(data_dir)

    # Training batches
    train_imgs_list, train_lbls_list = [], []
    for i in range(1, 6):
        imgs, lbls = _load_cifar10_batch(batches_dir / f'data_batch_{i}')
        train_imgs_list.append(imgs)
        train_lbls_list.append(lbls)
    train_imgs = np.concatenate(train_imgs_list).astype(np.float32) / 255.0
    train_lbls = np.concatenate(train_lbls_list)

    test_imgs, test_lbls = _load_cifar10_batch(batches_dir / 'test_batch')
    test_imgs = test_imgs.astype(np.float32) / 255.0

    def _split(imgs, lbls, cap):
        cats    = [imgs[i] for i in range(len(lbls)) if lbls[i] == _CAT_CLASS]
        noncats = [imgs[i] for i in range(len(lbls)) if lbls[i] != _CAT_CLASS]
        if cap:
            cats    = cats[:cap]
            noncats = noncats[:cap]
        return cats, noncats

    cats_tr, neg_tr = _split(train_imgs, train_lbls, max_per_class_train)
    cats_te, neg_te = _split(test_imgs,  test_lbls,  max_per_class_test)

    print(f'CIFAR-10 loaded: '
          f'{len(cats_tr)} cat / {len(neg_tr)} non-cat train, '
          f'{len(cats_te)} cat / {len(neg_te)} non-cat test')
    return cats_tr, neg_tr, cats_te, neg_te


# ---------------------------------------------------------------------------
# Folder-based loader (Phase 17 user images)
# ---------------------------------------------------------------------------

def _load_image_file(path: Path) -> Optional[np.ndarray]:
    """Load a single image file → float32 H×W×C array, values in [0,1].

    Returns None if the file cannot be loaded.
    """
    try:
        try:
            from PIL import Image as _PIL
            img = _PIL.open(str(path)).convert('RGB')
            return np.array(img, dtype=np.float32) / 255.0
        except ImportError:
            pass
        # Fallback: matplotlib
        import matplotlib.pyplot as plt
        arr = plt.imread(str(path))
        if arr.dtype == np.uint8:
            return arr.astype(np.float32) / 255.0
        return arr.astype(np.float32)
    except Exception:
        return None


def load_image_folder(
    cats_dir: Optional[Path] = None,
    neg_dir:  Optional[Path] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load cat and non-cat images from the standard folder layout.

    Scans all sub-folders under cats_dir and neg_dir recursively.

    Args:
        cats_dir:    Root of cat images (default: data/cats/).
        neg_dir:     Root of negative images (default: data/negatives/).
        target_size: Optional (H, W) to resize all images to.

    Returns (positives, negatives) — lists of float32 numpy arrays.
    """
    cats_dir = Path(cats_dir or CATS_DIR)
    neg_dir  = Path(neg_dir  or NEG_DIR)

    def _scan(root: Path) -> List[np.ndarray]:
        images = []
        if not root.exists():
            return images
        for path in sorted(root.rglob('*')):
            if path.suffix.lower() not in _IMAGE_EXTS:
                continue
            arr = _load_image_file(path)
            if arr is None:
                print(f'  Warning: could not load {path}')
                continue
            if target_size is not None:
                try:
                    from PIL import Image as _PIL
                    h, w = target_size
                    pil = _PIL.fromarray((arr * 255).astype(np.uint8))
                    pil = pil.resize((w, h), _PIL.BILINEAR)
                    arr = np.array(pil, dtype=np.float32) / 255.0
                except ImportError:
                    pass  # keep original size if PIL unavailable
            images.append(arr)
        return images

    positives = _scan(cats_dir)
    negatives = _scan(neg_dir)

    print(f'Folder loader: {len(positives)} cat images, '
          f'{len(negatives)} negative images')
    return positives, negatives


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def balance(
    positives: List[np.ndarray],
    negatives: List[np.ndarray],
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Subsample the larger class to match the smaller, for balanced training."""
    import random
    rng = random.Random(seed)
    n = min(len(positives), len(negatives))
    pos = rng.sample(positives, n)
    neg = rng.sample(negatives, n)
    return pos, neg


def train_test_split(
    positives: List[np.ndarray],
    negatives: List[np.ndarray],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray],
           List[np.ndarray], List[np.ndarray]]:
    """Simple train/test split preserving class balance."""
    import random
    rng = random.Random(seed)

    def _split(items):
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_fraction))
        return shuffled[n_test:], shuffled[:n_test]

    pos_tr, pos_te = _split(positives)
    neg_tr, neg_te = _split(negatives)
    return pos_tr, neg_tr, pos_te, neg_te
