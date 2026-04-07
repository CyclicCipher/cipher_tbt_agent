"""Minimal MNIST loader using numpy only (no torchvision).

Downloads MNIST .gz files if not present, loads into numpy arrays.
"""
from __future__ import annotations

import gzip
import os
import struct
import urllib.request

import numpy as np

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "mnist")


def _download(filename: str):
    """Download a file from MNIST site if not cached."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, filename)
    if os.path.exists(path):
        return path
    url = MNIST_URL + filename
    print(f"  Downloading {url}...")
    urllib.request.urlretrieve(url, path)
    return path


def _load_images(path: str) -> np.ndarray:
    """Load IDX image file → (N, 28, 28) float32 array in [0, 1]."""
    with gzip.open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols).astype(np.float32) / 255.0


def _load_labels(path: str) -> np.ndarray:
    """Load IDX label file → (N,) int array."""
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist(verbose: bool = True) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Load MNIST train and test sets.

    Returns:
        ((train_images, train_labels), (test_images, test_labels))
        train_images: (60000, 28, 28) float32 in [0, 1]
        train_labels: (60000,) int
        test_images: (10000, 28, 28) float32 in [0, 1]
        test_labels: (10000,) int
    """
    if verbose:
        print("Loading MNIST...")
    train_img = _load_images(_download(MNIST_FILES["train_images"]))
    train_lbl = _load_labels(_download(MNIST_FILES["train_labels"]))
    test_img = _load_images(_download(MNIST_FILES["test_images"]))
    test_lbl = _load_labels(_download(MNIST_FILES["test_labels"]))
    if verbose:
        print(f"  Train: {train_img.shape}, Test: {test_img.shape}")
    return (train_img, train_lbl), (test_img, test_lbl)
