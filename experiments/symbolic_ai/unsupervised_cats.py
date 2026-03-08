"""unsupervised_cats.py — Unsupervised structure learning on cat photographs.

No labels are given during training.  After fitting, we ask:

  1. Does the saliency map consistently attend to the cat rather than background?
  2. Do foveal sequences carry pose-discriminative information — even though
     the model never saw a pose label?
  3. Can we recover the 4 pose groups (frontal / seated / side / three_quarter)
     from unsupervised clustering of foveal feature bags?

Methodology
-----------
  - Load all 75 cat photos (resize to RESIZE px square).
  - Train FovealVisionLearner (peripheral_color=True, no labels).
  - For each image, run fixate() to collect foveal sequences.
  - Build a feature bag per image: Counter of (pos:cluster_id) strings.
  - Cluster images (k-means, k=4) on TF-IDF-weighted feature bags.
  - Evaluate: cluster purity vs ground-truth pose groups + NMI.
  - Report: which features are most common in each cluster.
  - Report: within-group vs between-group cosine similarity.

Usage:
    python unsupervised_cats.py
    python unsupervised_cats.py --n_images 20      # quick smoke test
    python unsupervised_cats.py --resize 256        # smaller for speed
    python unsupervised_cats.py --peripheral_patch 16 --foveal_radius 64
    python unsupervised_cats.py --foveal_patch 4     # safe default (pixel-accurate=1 causes OOM)
"""
from __future__ import annotations

import argparse
import collections
import glob
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from vision_pipeline import FovealVisionLearner, foveal_radius_from_viewing_distance


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_images(cats_dir: str, resize: int, n_images: int | None = None,
                seed: int = 42) -> tuple[list, list, list]:
    """Load all cat photos, resize to (resize × resize), return (images, labels, paths).

    labels: pose subfolder name (frontal / seated / side / three_quarter).
    Images are returned as float32 arrays in [0, 1], shape (resize, resize, 3).
    """
    from PIL import Image

    pose_dirs = sorted(d for d in os.listdir(cats_dir)
                       if os.path.isdir(os.path.join(cats_dir, d)))
    all_paths, all_labels = [], []
    for pose in pose_dirs:
        files = sorted(glob.glob(os.path.join(cats_dir, pose, '*')))
        files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files:
            all_paths.append(f)
            all_labels.append(pose)

    if n_images is not None and n_images < len(all_paths):
        rng = random.Random(seed)
        idx = list(range(len(all_paths)))
        rng.shuffle(idx)
        idx = idx[:n_images]
        all_paths  = [all_paths[i]  for i in idx]
        all_labels = [all_labels[i] for i in idx]

    images = []
    good_paths, good_labels = [], []
    for path, label in zip(all_paths, all_labels):
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((resize, resize), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
            good_paths.append(path)
            good_labels.append(label)
        except Exception as e:
            print(f'  [warn] skipped {path}: {e}')

    return images, good_labels, good_paths


# ---------------------------------------------------------------------------
# Feature bags
# ---------------------------------------------------------------------------

def image_to_feature_bag(fvl: FovealVisionLearner, image) -> collections.Counter:
    """Run fixate() and return a Counter of (pos:cluster_id) feature strings."""
    bag: collections.Counter = collections.Counter()
    for fix in fvl.fixate(image):
        features = fvl._foveal_seq_to_features(fix['sequence'])
        bag.update(features)
    return bag


def build_feature_matrix(bags: list[collections.Counter]) -> tuple[np.ndarray, list]:
    """Convert list of Counters to a TF-IDF-weighted float32 matrix (n_images × n_feats).

    TF-IDF down-weights features that appear in every image (background patches)
    and up-weights features that are rare and therefore more discriminative.
    """
    # Vocabulary: union of all feature strings
    vocab: list[str] = sorted({f for bag in bags for f in bag})
    feat_idx = {f: i for i, f in enumerate(vocab)}
    n = len(bags)
    V = len(vocab)

    # Raw TF matrix
    tf = np.zeros((n, V), dtype=np.float32)
    for i, bag in enumerate(bags):
        total = sum(bag.values()) or 1
        for feat, cnt in bag.items():
            tf[i, feat_idx[feat]] = cnt / total

    # IDF
    doc_freq = np.sum(tf > 0, axis=0).astype(np.float32)
    idf = np.log((n + 1) / (doc_freq + 1)) + 1.0   # smoothed IDF

    tfidf = tf * idf
    # L2-normalise rows
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    tfidf /= norms
    return tfidf, vocab


# ---------------------------------------------------------------------------
# K-means clustering (pure numpy)
# ---------------------------------------------------------------------------

def kmeans(X: np.ndarray, k: int, n_iter: int = 50, seed: int = 0) -> np.ndarray:
    """K-means with cosine distance (input must be L2-normalised rows)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    centres = X[idx].copy()

    labels = np.zeros(len(X), dtype=int)
    for it in range(n_iter):
        # Assignment: cosine similarity = dot product (rows normalised)
        sims = X @ centres.T           # (n, k)
        new_labels = np.argmax(sims, axis=1)

        if np.all(new_labels == labels) and it > 0:
            break
        labels = new_labels

        # Update
        for c in range(k):
            mask = labels == c
            if mask.any():
                centre = X[mask].mean(axis=0)
                norm = np.linalg.norm(centre)
                centres[c] = centre / norm if norm > 0 else centre
    return labels


# ---------------------------------------------------------------------------
# Evaluation: cluster purity + NMI
# ---------------------------------------------------------------------------

def cluster_purity(cluster_labels: np.ndarray, gt_labels: list[str]) -> float:
    """Fraction of images assigned to the majority ground-truth class in their cluster."""
    gt = np.array(gt_labels)
    k = len(set(cluster_labels))
    correct = 0
    for c in range(k):
        mask = cluster_labels == c
        if not mask.any():
            continue
        gt_in_cluster = gt[mask]
        majority_count = collections.Counter(gt_in_cluster).most_common(1)[0][1]
        correct += majority_count
    return correct / len(gt_labels)


def nmi(cluster_labels: np.ndarray, gt_labels: list[str]) -> float:
    """Normalised mutual information between cluster assignment and pose group."""
    n = len(gt_labels)
    gt = np.array(gt_labels)
    clusters = sorted(set(cluster_labels))
    poses    = sorted(set(gt_labels))

    # Joint distribution P(c, g)
    joint = np.zeros((len(clusters), len(poses)), dtype=np.float64)
    c_idx = {c: i for i, c in enumerate(clusters)}
    p_idx = {p: i for i, p in enumerate(poses)}
    for cl, gt_l in zip(cluster_labels, gt):
        joint[c_idx[cl], p_idx[gt_l]] += 1
    joint /= n

    pc = joint.sum(axis=1)   # P(cluster)
    pg = joint.sum(axis=0)   # P(pose)

    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    Hc = entropy(pc)
    Hg = entropy(pg)
    # Mutual information
    MI = 0.0
    for i in range(len(clusters)):
        for j in range(len(poses)):
            if joint[i, j] > 0:
                MI += joint[i, j] * math.log2(joint[i, j] / (pc[i] * pg[j]))
    denom = (Hc + Hg) / 2
    return MI / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Structural analysis
# ---------------------------------------------------------------------------

def within_between_similarity(X: np.ndarray, gt_labels: list[str]) -> dict:
    """Compute mean cosine similarity within vs across pose groups."""
    gt = np.array(gt_labels)
    poses = sorted(set(gt_labels))
    sims = X @ X.T   # cosine similarity matrix (X is L2-normalised)

    within_sims, between_sims = [], []
    n = len(gt)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sims[i, j])
            if gt[i] == gt[j]:
                within_sims.append(s)
            else:
                between_sims.append(s)

    return {
        'within_mean':  float(np.mean(within_sims))  if within_sims  else 0.0,
        'within_std':   float(np.std(within_sims))   if within_sims  else 0.0,
        'between_mean': float(np.mean(between_sims)) if between_sims else 0.0,
        'between_std':  float(np.std(between_sims))  if between_sims else 0.0,
        'ratio':        (float(np.mean(within_sims)) /
                         max(float(np.mean(between_sims)), 1e-9))
                        if within_sims and between_sims else 1.0,
    }


def top_cluster_features(cluster_labels: np.ndarray, bags: list,
                          k: int, topn: int = 8) -> dict:
    """For each cluster, collect the most frequent features (raw counts)."""
    result = {}
    for c in range(k):
        combined: collections.Counter = collections.Counter()
        for i, bag in enumerate(bags):
            if cluster_labels[i] == c:
                combined.update(bag)
        result[c] = combined.most_common(topn)
    return result


def saliency_stats(fvl: FovealVisionLearner, images: list,
                   gt_labels: list[str]) -> dict:
    """Compute mean saliency concentration across images.

    concentration = fraction of total saliency in top-25% of patches.
    High concentration → attention is focused (cat found).
    Low concentration → attention is diffuse (no clear object).
    """
    import numpy as np
    from vision_pipeline import saliency_map

    concentrations: dict[str, list] = collections.defaultdict(list)
    for img, label in zip(images, gt_labels):
        sal = saliency_map(img, fvl.peripheral, fvl.peripheral_patch)
        if sal is None:
            continue
        flat = sal.ravel()
        flat_sorted = np.sort(flat)[::-1]
        total = flat_sorted.sum()
        if total <= 0:
            continue
        top_k = max(1, len(flat) // 4)
        conc = float(flat_sorted[:top_k].sum() / total)
        concentrations[label].append(conc)

    return {label: float(np.mean(vals)) for label, vals in concentrations.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cats_dir', default='data/cats')
    ap.add_argument('--n_images', type=int, default=None,
                    help='Limit total images (None = use all 75)')
    ap.add_argument('--resize',   type=int, default=400,
                    help='Resize all images to this square size (px)')
    ap.add_argument('--peripheral_patch', type=int, default=32)
    ap.add_argument('--foveal_radius',    type=int, default=None,
                    help='Foveal radius px (default: calibrated to 24" screen, 24" dist)')
    ap.add_argument('--foveal_patch',     type=int, default=4,
                    help='Foveal sub-patch size in px (1=pixel-accurate but huge sequences; '
                         '4 reduces tokens 16× and prevents memory explosion)')
    ap.add_argument('--n_fixations',      type=int, default=5)
    ap.add_argument('--k',                type=int, default=4,
                    help='Number of clusters (should match pose group count)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    # Calibrate foveal radius to screen if not provided
    if args.foveal_radius is None:
        # 400px image shown at 24" screen (20.8" wide) at 24" viewing distance
        # → 82px radius for full screen; scale to image fraction
        args.foveal_radius = max(32, args.resize // 5)

    print('=' * 60)
    print('Unsupervised Cat Photo Experiment')
    print('=' * 60)
    print(f'  resize={args.resize}px  peripheral_patch={args.peripheral_patch}px  '
          f'foveal_radius={args.foveal_radius}px  foveal_patch={args.foveal_patch}px  '
          f'n_fixations={args.n_fixations}')

    # --- Load images ---
    print(f'\n[1] Loading images from {args.cats_dir}...')
    images, gt_labels, paths = load_images(
        args.cats_dir, args.resize, args.n_images, args.seed)
    pose_counts = collections.Counter(gt_labels)
    print(f'  Loaded {len(images)} images')
    for pose, cnt in sorted(pose_counts.items()):
        print(f'    {pose}: {cnt}')

    if len(images) < 4:
        print('ERROR: need at least 4 images.')
        return

    # --- Train FovealVisionLearner (unsupervised) ---
    print('\n[2] Training FovealVisionLearner (no labels)...')
    fvl = FovealVisionLearner(
        peripheral_patch=args.peripheral_patch,
        foveal_patch=args.foveal_patch,
        foveal_radius_px=args.foveal_radius,
        n_fixations=args.n_fixations,
        n_peripheral_clusters=32,
        n_foveal_clusters=64,
        peripheral_color=True,    # use color in peripheral scan
    )
    fvl.fit_images(images, verbose=True)

    print(f'\n  Peripheral vocab size: {len(fvl.peripheral.learner.assignment)}')
    print(f'  Foveal assignment entries: {len(fvl.foveal.assignment)}')

    # --- Saliency analysis: is attention focused? ---
    print('\n[3] Saliency concentration by pose group...')
    print('  (fraction of saliency in top-25% of patches;'
          ' higher = more focused attention)')
    conc = saliency_stats(fvl, images, gt_labels)
    for pose in sorted(conc):
        print(f'    {pose}: {conc[pose]:.3f}')
    overall_conc = float(np.mean(list(conc.values())))
    print(f'    overall: {overall_conc:.3f}  '
          f'(baseline uniform = {0.25:.3f})')
    if overall_conc > 0.45:
        print('  → FOCUSED: model attends to specific regions (cat found?)')
    else:
        print('  → DIFFUSE: attention spread uniformly (no clear structure yet)')

    # --- Build feature bags ---
    print('\n[4] Building foveal feature bags...')
    bags = [image_to_feature_bag(fvl, img) for img in images]
    total_feats = sum(len(b) for b in bags)
    vocab_size = len({f for b in bags for f in b})
    print(f'  Total feature instances: {total_feats:,}')
    print(f'  Unique (pos:cluster) features: {vocab_size}')
    mean_bag_size = total_feats / len(bags) if bags else 0
    print(f'  Mean features per image: {mean_bag_size:.0f}')

    # --- Within vs between similarity ---
    print('\n[5] Within-pose vs between-pose feature similarity...')
    X, vocab = build_feature_matrix(bags)
    sim = within_between_similarity(X, gt_labels)
    print(f'  Within-pose  cosine sim: {sim["within_mean"]:.4f} ± {sim["within_std"]:.4f}')
    print(f'  Between-pose cosine sim: {sim["between_mean"]:.4f} ± {sim["between_std"]:.4f}')
    print(f'  Ratio (within/between):  {sim["ratio"]:.3f}')
    if sim['ratio'] > 1.15:
        print('  → STRUCTURE FOUND: same-pose images are more similar than different-pose')
    elif sim['ratio'] > 1.05:
        print('  → WEAK STRUCTURE: slight tendency for same-pose images to cluster')
    else:
        print('  → NO STRUCTURE: model treats all images the same (no pose signal)')

    # --- K-means clustering ---
    print(f'\n[6] K-means clustering (k={args.k})...')
    cluster_labels = kmeans(X, k=args.k, seed=args.seed)
    for c in range(args.k):
        count = (cluster_labels == c).sum()
        pose_breakdown = collections.Counter(
            gt_labels[i] for i in range(len(gt_labels)) if cluster_labels[i] == c)
        top_pose = pose_breakdown.most_common(1)[0] if pose_breakdown else ('?', 0)
        print(f'  Cluster {c}: {count} images  '
              f'(majority={top_pose[0]}, {top_pose[1]}/{count})')

    # --- Cluster quality ---
    print('\n[7] Cluster quality...')
    purity = cluster_purity(cluster_labels, gt_labels)
    nmi_val = nmi(cluster_labels, gt_labels)
    print(f'  Purity: {purity:.3f}  (random baseline ≈ {1/args.k:.3f})')
    print(f'  NMI:    {nmi_val:.3f}  (0 = no information, 1 = perfect)')
    if purity > 0.6 and nmi_val > 0.2:
        print('  → STRONG: pose structure recovered from unsupervised features')
    elif purity > 0.4 or nmi_val > 0.1:
        print('  → PARTIAL: some pose structure visible — model learned something')
    else:
        print('  → WEAK: clusters do not correspond to pose groups')

    # --- Contingency table ---
    print('\n[8] Contingency table (rows=cluster, cols=pose group)...')
    poses = sorted(set(gt_labels))
    header = '         ' + '  '.join(f'{p[:8]:>8}' for p in poses)
    print('  ' + header)
    for c in range(args.k):
        row = f'  Cluster {c}'
        for pose in poses:
            cnt = sum(1 for i in range(len(gt_labels))
                      if cluster_labels[i] == c and gt_labels[i] == pose)
            row += f'  {cnt:>8}'
        print(row)

    # --- Top features per cluster ---
    print('\n[9] Most common foveal features per cluster')
    print('    (format: pos_bin:cluster_id  e.g. D-2,1:15 = position Δrow=-2,Δcol=1, cluster 15)')
    top = top_cluster_features(cluster_labels, bags, args.k, topn=6)
    for c in range(args.k):
        pose_breakdown = collections.Counter(
            gt_labels[i] for i in range(len(gt_labels)) if cluster_labels[i] == c)
        majority = pose_breakdown.most_common(1)[0][0] if pose_breakdown else '?'
        print(f'  Cluster {c} (mainly {majority}):')
        for feat, cnt in top[c]:
            print(f'    {feat:20s}  count={cnt}')

    # --- Overall summary ---
    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    print(f'  Images: {len(images)}  Clusters: {args.k}')
    print(f'  Saliency concentration: {overall_conc:.3f}  '
          f'(>0.45 = focused, baseline=0.25)')
    print(f'  Within/between sim ratio: {sim["ratio"]:.3f}  (>1.15 = pose structure)')
    print(f'  Cluster purity: {purity:.3f}  NMI: {nmi_val:.3f}')

    learned = []
    if overall_conc > 0.45:
        learned.append('salient regions (attends to objects)')
    if sim['ratio'] > 1.05:
        learned.append('pose-discriminative texture patterns')
    if purity > 0.4:
        learned.append('partially recoverable pose clusters')
    if not learned:
        learned.append('nothing clearly discriminative yet')
    print(f'  Evidence of learning: {"; ".join(learned)}')


if __name__ == '__main__':
    main()
