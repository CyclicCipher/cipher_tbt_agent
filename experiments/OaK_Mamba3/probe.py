"""
probe.py - Structural probe suite for OaK-Mamba3.

Test 3: dt-spike probe
    Checks whether dt (the state-space step size) is systematically higher at
    SEP / QUERY boundary tokens than inside grid tokens.  If yes, the model
    treats example boundaries as hypothesis-revision events - the core OaK
    mechanism.

    Protocol:
      - Run N episodes through the model (no grad, single batch dim).
      - After each forward pass, read _last_dt from every OaKMixer (both fwd
        and bwd arms of each BiOaKMixer).  Shape: (1, T, nheads).
      - Classify every token position by type: sep_query / ex_in / ex_out /
        test_in / test_out / pad.
      - Accumulate per-type mean dt (averaged over heads) across all episodes
        and layers.
      - Report a table: rows = position type, cols = layer (fwd / bwd).
      - Pass criterion: mean dt at sep_query > mean dt at ex_in.

Test 4: k-means representational probe
    Checks whether the hidden representation at the QUERY token position is
    organised by rule type after training.  If purity > 1/k (chance), the
    model has compressed rule identity into the QUERY-position vector.

    Protocol:
      - Register a forward hook on model.norm to capture final hidden states
        h: (1, T, d_model).
      - For each episode collect h at the QUERY token position index.
      - Also record the first primitive of the episode's rule_name as label.
      - Run k-means (k=8 or --k_means) on the collected vectors.
      - Compute cluster purity = sum(max_class_count_per_cluster) / total.
      - Report: purity, chance baseline, and top-2 rules per cluster.

Usage:
    # Both tests on a saved checkpoint
    python experiments/OaK_Mamba3/probe.py --checkpoint path/to/model.pt

    # Single test
    python experiments/OaK_Mamba3/probe.py --checkpoint path/to/model.pt --test 3
    python experiments/OaK_Mamba3/probe.py --checkpoint path/to/model.pt --test 4

    # Untrained baseline (no checkpoint)
    python experiments/OaK_Mamba3/probe.py --no_checkpoint
"""

import argparse
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'Mamba3'))

from oak_model import (
    OaKConfig, OaKModel,
    SEP_TOKEN, QUERY_TOKEN, PAD_TOKEN, MASK_TOKEN, NUM_COLORS,
)
from env1 import sample_episode
from train_env1 import encode_episode


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> OaKModel:
    """Load OaKModel from a checkpoint saved by train_env1.py."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt['config_dict']
    config = OaKConfig(
        d_model     = cfg_dict['d_model'],
        d_state     = cfg_dict.get('d_state', 64),
        expand      = cfg_dict.get('expand', 2),
        headdim     = cfg_dict.get('headdim', 64),
        chunk_size  = cfg_dict.get('chunk_size', 64),
        n_layer     = cfg_dict['n_layer'],
        mlp_expand  = cfg_dict.get('mlp_expand', 4),
        stable_ssm  = cfg_dict.get('stable_ssm', True),
        num_options = cfg_dict.get('num_options', 1),
        d_option    = cfg_dict.get('d_option', 32),
        n_gvfs      = cfg_dict.get('n_gvfs', 5),
        n_segments  = cfg_dict.get('n_segments', 3),
    )
    model = OaKModel(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded checkpoint: {checkpoint_path}')
    print(f'  d_model={config.d_model}, n_layer={config.n_layer}, '
          f'num_options={config.num_options}')
    return model


def make_default_model(device: torch.device) -> OaKModel:
    """Create an untrained model with default config (baseline)."""
    config = OaKConfig(d_model=128, n_layer=4)
    model = OaKModel(config).to(device)
    model.eval()
    print('Using untrained model with default config (d_model=128, n_layer=4)')
    return model


# ---------------------------------------------------------------------------
# Position type classifier
# ---------------------------------------------------------------------------

# Position-type labels
PT_SEP_QUERY = 'sep_query'  # SEP or QUERY boundary token
PT_EX_IN     = 'ex_in'      # example input grid cell
PT_EX_OUT    = 'ex_out'     # example output grid cell
PT_TEST_IN   = 'test_in'    # test input grid cell
PT_TEST_OUT  = 'test_out'   # test output grid cell (target)
PT_PAD       = 'pad'        # PAD/MASK filler

_ALL_PTYPES = [PT_SEP_QUERY, PT_EX_IN, PT_EX_OUT, PT_TEST_IN, PT_TEST_OUT, PT_PAD]


def classify_positions(
    tokens: List[int],
    grid_segments: List[Tuple[int, int, int, int]],
) -> List[str]:
    """Assign a position-type string to every token in the sequence.

    grid_segments: list of (start, H, W, seg_id)
        seg_id 0 = input grid (example inputs + test input)
        seg_id 1 = output grid (example outputs)
        seg_id 2 = test output grid

    The test input is the LAST seg_id=0 segment (same convention as
    encode_episode and compute_oracle_omega).
    """
    T = len(tokens)
    ptype = [PT_SEP_QUERY] * T

    # Mark grid token positions from segments
    input_segs  = [(s, h, w) for s, h, w, sid in grid_segments if sid == 0]
    output_segs = [(s, h, w) for s, h, w, sid in grid_segments if sid == 1]
    test_out_segs = [(s, h, w) for s, h, w, sid in grid_segments if sid == 2]

    # Example input grids (all seg_id=0 except last)
    for start, h, w in input_segs[:-1]:
        for i in range(h * w):
            if start + i < T:
                ptype[start + i] = PT_EX_IN

    # Test input (last seg_id=0 segment)
    if input_segs:
        start, h, w = input_segs[-1]
        for i in range(h * w):
            if start + i < T:
                ptype[start + i] = PT_TEST_IN

    # Example output grids
    for start, h, w in output_segs:
        for i in range(h * w):
            if start + i < T:
                ptype[start + i] = PT_EX_OUT

    # Test output grid
    for start, h, w in test_out_segs:
        for i in range(h * w):
            if start + i < T:
                ptype[start + i] = PT_TEST_OUT

    # Override PAD/MASK tokens
    for i, tok in enumerate(tokens):
        if tok in (PAD_TOKEN, MASK_TOKEN):
            ptype[i] = PT_PAD

    return ptype


# ---------------------------------------------------------------------------
# Test 3: dt-spike probe
# ---------------------------------------------------------------------------

def run_dt_probe(
    model: OaKModel,
    n_episodes: int,
    difficulty: int,
    device: torch.device,
    rng: np.random.Generator,
) -> None:
    """Test 3: print dt statistics by position type for every layer.

    For each BiOaKMixer layer, accesses _last_dt from both the fwd and bwd
    OaKMixer arms.  Reports mean dt (averaged over heads) per position type
    and per layer arm.

    Pass criterion: mean dt at sep_query > mean dt at ex_in.
    """
    print('\n' + '=' * 70)
    print('TEST 3: dt-spike probe')
    print('=' * 70)
    print(f'Episodes: {n_episodes} | Difficulty: {difficulty}')
    print()

    n_layers = model.config.n_layer
    # Accumulators: dt_acc[layer_idx][arm]['ptype'] -> [list of values]
    # arm: 'fwd' or 'bwd'
    dt_acc: Dict[int, Dict[str, Dict[str, List[float]]]] = {
        i: {'fwd': defaultdict(list), 'bwd': defaultdict(list)}
        for i in range(n_layers)
    }

    # Sample representative (H, W, K) for the difficulty
    if difficulty == 1:
        H, W, K = 4, 4, 2
    elif difficulty == 2:
        H, W, K = 6, 6, 3
    else:
        H, W, K = 8, 8, 4

    with torch.no_grad():
        for ep_idx in range(n_episodes):
            ep = sample_episode(H, W, K, difficulty, rng)
            tokens, grid_segments, _label_start = encode_episode(ep, H, W)
            ptypes = classify_positions(tokens, grid_segments)

            # Pad to multiple of chunk_size
            cs = model.config.chunk_size
            T_orig = len(tokens)
            pad_len = (cs - T_orig % cs) % cs
            tokens_padded = tokens + [PAD_TOKEN] * pad_len
            ptypes_padded = ptypes + [PT_PAD] * pad_len

            tok_t = torch.tensor(tokens_padded, dtype=torch.long, device=device).unsqueeze(0)
            # Run forward pass (this sets _last_dt on all OaKMixer instances)
            _ = model(tok_t, grid_segments)

            # Collect dt from each layer
            for i, block in enumerate(model.layers):
                mixer = block.mixer  # BiOaKMixer
                for arm_name, arm_mixer in [('fwd', mixer.fwd), ('bwd', mixer.bwd)]:
                    dt_t = arm_mixer._last_dt  # (1, T_padded, nheads)
                    if dt_t is None:
                        continue
                    # Mean over heads: (T_padded,)
                    dt_mean = dt_t[0].mean(dim=-1).cpu().float()
                    for pos, pt in enumerate(ptypes_padded):
                        if pos < dt_mean.shape[0]:
                            dt_acc[i][arm_name][pt].append(dt_mean[pos].item())

    # Print results table
    # Column header
    header = f"{'pos_type':>12}"
    for i in range(n_layers):
        header += f"  L{i}_fwd  L{i}_bwd"
    print(header)
    print('-' * len(header))

    for pt in _ALL_PTYPES:
        row = f'{pt:>12}'
        for i in range(n_layers):
            for arm in ('fwd', 'bwd'):
                vals = dt_acc[i][arm][pt]
                if vals:
                    row += f'  {np.mean(vals):.4f}'
                else:
                    row += f'      --  '
        print(row)

    # Pass/fail verdict
    print()
    print('--- Pass criterion: mean dt(sep_query) > mean dt(ex_in) ---')
    all_sep_vals, all_exin_vals = [], []
    for i in range(n_layers):
        for arm in ('fwd', 'bwd'):
            all_sep_vals.extend(dt_acc[i][arm][PT_SEP_QUERY])
            all_exin_vals.extend(dt_acc[i][arm][PT_EX_IN])

    if all_sep_vals and all_exin_vals:
        mean_sep  = np.mean(all_sep_vals)
        mean_exin = np.mean(all_exin_vals)
        ratio     = mean_sep / (mean_exin + 1e-9)
        verdict   = 'PASS' if mean_sep > mean_exin else 'FAIL'
        print(f'  mean dt(sep_query) = {mean_sep:.5f}')
        print(f'  mean dt(ex_in)     = {mean_exin:.5f}')
        print(f'  ratio              = {ratio:.3f}x')
        print(f'  Verdict: {verdict}')
    else:
        print('  Insufficient data.')

    # Additional useful ratios
    print()
    print('--- Full cross-type ratios (relative to ex_in baseline) ---')
    baseline_vals = []
    for i in range(n_layers):
        for arm in ('fwd', 'bwd'):
            baseline_vals.extend(dt_acc[i][arm][PT_EX_IN])
    baseline = np.mean(baseline_vals) + 1e-9

    for pt in _ALL_PTYPES:
        agg = []
        for i in range(n_layers):
            for arm in ('fwd', 'bwd'):
                agg.extend(dt_acc[i][arm][pt])
        if agg:
            print(f'  {pt:>12}: {np.mean(agg):.5f}  ({np.mean(agg)/baseline:.2f}x)')


# ---------------------------------------------------------------------------
# k-means (pure NumPy, no sklearn required)
# ---------------------------------------------------------------------------

def _kmeans_np(
    X: np.ndarray,
    k: int,
    n_init: int = 10,
    max_iter: int = 300,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means++ initialisation + Lloyd's algorithm.

    Args:
        X: (N, d) float32 array
        k: number of clusters

    Returns:
        labels:   (N,) int cluster assignments
        centers:  (k, d) cluster centroids
    """
    rng = np.random.default_rng(seed)
    best_labels  = None
    best_inertia = float('inf')
    best_centers = None

    for _ in range(n_init):
        # k-means++ initialisation
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centers)
                for x in X
            ])
            probs = dists / (dists.sum() + 1e-12)
            centers.append(X[rng.choice(len(X), p=probs)])
        centers = np.stack(centers, axis=0)   # (k, d)

        labels = np.zeros(len(X), dtype=np.int32)
        for _it in range(max_iter):
            # Assignment
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=-1)  # (N, k)
            new_labels = dists.argmin(axis=1)
            if np.all(new_labels == labels):
                break
            labels = new_labels
            # Update
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    centers[j] = X[mask].mean(axis=0)

        inertia = sum(
            np.sum((X[labels == j] - centers[j]) ** 2)
            for j in range(k) if (labels == j).sum() > 0
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels  = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers


def _cluster_purity(
    labels: np.ndarray,
    class_labels: np.ndarray,
    k: int,
) -> float:
    """Compute cluster purity = sum(max_count_per_cluster) / N."""
    total_correct = 0
    for cluster_id in range(k):
        mask = labels == cluster_id
        if not mask.any():
            continue
        counts = np.bincount(class_labels[mask])
        total_correct += counts.max()
    return total_correct / len(labels)


# ---------------------------------------------------------------------------
# Test 4: k-means representational probe
# ---------------------------------------------------------------------------

def run_kmeans_probe(
    model: OaKModel,
    n_episodes: int,
    difficulty: int,
    k: int,
    device: torch.device,
    rng: np.random.Generator,
) -> None:
    """Test 4: k-means probe at QUERY token position.

    Hypothesis: after training, h[QUERY_pos] clusters by rule type.
    Pass criterion: purity > 1/k (above chance).

    For each episode, the hidden state at the QUERY token is extracted
    (the last pre-test-output position - the point where the model should
    have formed a rule hypothesis from K examples).

    Also probes the final SEP position before the QUERY for comparison.
    """
    print('\n' + '=' * 70)
    print('TEST 4: k-means representational probe')
    print('=' * 70)
    print(f'Episodes: {n_episodes} | Difficulty: {difficulty} | k={k}')
    print()

    # Collect vectors and labels
    vecs_query: List[np.ndarray] = []   # h at QUERY position
    vecs_sep:   List[np.ndarray] = []   # h at last SEP before QUERY
    rule_labels: List[str] = []

    # Hook on model.norm to capture h
    captured_h: Dict[str, Optional[Tensor]] = {'h': None}

    def _hook(_module, _input, output):
        captured_h['h'] = output.detach().cpu()

    hook_handle = model.norm.register_forward_hook(_hook)

    if difficulty == 1:
        H, W, K = 4, 4, 2
    elif difficulty == 2:
        H, W, K = 6, 6, 3
    else:
        H, W, K = 8, 8, 4

    cs = model.config.chunk_size

    with torch.no_grad():
        for ep_idx in range(n_episodes):
            ep = sample_episode(H, W, K, difficulty, rng)
            tokens, grid_segments, label_start = encode_episode(ep, H, W)

            # Pad
            T_orig = len(tokens)
            pad_len = (cs - T_orig % cs) % cs
            tokens_padded = tokens + [PAD_TOKEN] * pad_len

            tok_t = torch.tensor(
                tokens_padded, dtype=torch.long, device=device
            ).unsqueeze(0)
            _ = model(tok_t, grid_segments)

            h = captured_h['h']   # (1, T_padded, d_model)
            if h is None:
                continue

            # QUERY position: the token immediately before label_start
            # (encode_episode places QUERY_TOKEN at label_start - 1)
            query_pos = label_start - 1
            if 0 <= query_pos < h.shape[1]:
                vecs_query.append(h[0, query_pos].numpy())

            # Last SEP before QUERY (label_start - H*W - 2 in the format:
            # [SEP][test_input_grid][QUERY], so SEP is at label_start - H*W - 2)
            sep_before_test = label_start - H * W - 2
            if 0 <= sep_before_test < h.shape[1]:
                vecs_sep.append(h[0, sep_before_test].numpy())

            # First primitive of rule name as class label
            rule_name = ep.get('rule_name', '?')
            first_prim = rule_name.split('(')[0].split('+')[0]
            rule_labels.append(first_prim)

    hook_handle.remove()

    if not vecs_query:
        print('No data collected - check episode encoding.')
        return

    # Encode class labels as integers
    unique_classes = sorted(set(rule_labels))
    class_to_int   = {c: i for i, c in enumerate(unique_classes)}
    class_ints     = np.array([class_to_int[r] for r in rule_labels])
    n_classes      = len(unique_classes)
    chance         = 1.0 / k

    print(f'Unique rule first-primitives ({n_classes}): {", ".join(unique_classes)}')
    print()

    for probe_name, vecs in [('QUERY position', vecs_query),
                              ('Last SEP position', vecs_sep)]:
        if not vecs:
            print(f'{probe_name}: no data')
            continue

        X = np.stack(vecs, axis=0).astype(np.float32)

        # L2 normalise for cosine-equivalent clustering
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        X_norm = X / norms

        print(f'--- {probe_name} (N={len(X)}, d={X.shape[1]}) ---')

        # Use sklearn if available (faster / better), else fall back to np
        try:
            from sklearn.cluster import KMeans as SkKMeans
            km = SkKMeans(n_clusters=k, n_init=10, random_state=0, max_iter=300)
            km_labels = km.fit_predict(X_norm)
            print('  (using sklearn KMeans)')
        except ImportError:
            print('  (sklearn not available - using numpy k-means++)')
            km_labels, _ = _kmeans_np(X_norm, k, n_init=5, seed=0)

        purity = _cluster_purity(km_labels, class_ints, k)
        verdict = 'PASS' if purity > chance else 'FAIL'
        print(f'  Purity   : {purity:.3f}')
        print(f'  Chance   : {chance:.3f} (1/{k})')
        print(f'  Verdict  : {verdict}')

        # Top-2 rules per cluster
        print(f'  Cluster breakdown (top-2 rules per cluster):')
        for cluster_id in range(k):
            mask = km_labels == cluster_id
            if not mask.any():
                print(f'    cluster {cluster_id}: empty')
                continue
            cluster_classes = [rule_labels[i] for i in range(len(rule_labels)) if mask[i]]
            counts = defaultdict(int)
            for c in cluster_classes:
                counts[c] += 1
            top2 = sorted(counts.items(), key=lambda x: -x[1])[:2]
            top2_str = ', '.join(f'{r}({n})' for r, n in top2)
            print(f'    cluster {cluster_id} (n={mask.sum()}): {top2_str}')
        print()

    # Additional: variance explained by rule type (ANOVA-style R^2)
    X_all = np.stack(vecs_query, axis=0).astype(np.float32)
    grand_mean = X_all.mean(axis=0)
    ss_total = np.sum((X_all - grand_mean) ** 2)
    ss_between = 0.0
    for cls_int in range(n_classes):
        mask = class_ints == cls_int
        if mask.sum() > 1:
            cls_mean = X_all[mask].mean(axis=0)
            ss_between += mask.sum() * np.sum((cls_mean - grand_mean) ** 2)
    r_sq = ss_between / (ss_total + 1e-12)
    print(f'--- Rule-type R^2 (variance explained by rule class) ---')
    print(f'  R^2 = {r_sq:.4f}  (0 = no rule info, 1 = perfect separation)')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='OaK-Mamba3 structural probe suite (Tests 3 & 4)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint saved by train_env1.py')
    parser.add_argument('--no_checkpoint', action='store_true',
                        help='Use an untrained default model (baseline)')
    parser.add_argument('--test', type=int, default=0,
                        choices=[0, 3, 4],
                        help='Which test to run: 0=both, 3=dt_spike, 4=kmeans')
    parser.add_argument('--n_episodes', type=int, default=500,
                        help='Episodes per probe')
    parser.add_argument('--difficulty', type=int, default=2, choices=[1, 2, 3],
                        help='Episode difficulty (should match training difficulty)')
    parser.add_argument('--k_means', type=int, default=8,
                        help='Number of k-means clusters for Test 4')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if CUDA available')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'Device: {device}')

    # Load or create model
    if args.no_checkpoint:
        model = make_default_model(device)
    elif args.checkpoint:
        model = load_model(args.checkpoint, device)
    else:
        print('ERROR: Provide --checkpoint PATH or --no_checkpoint')
        parser.print_help()
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    run3 = args.test in (0, 3)
    run4 = args.test in (0, 4)

    if run3:
        run_dt_probe(
            model,
            n_episodes  = args.n_episodes,
            difficulty  = args.difficulty,
            device      = device,
            rng         = np.random.default_rng(args.seed),
        )

    if run4:
        run_kmeans_probe(
            model,
            n_episodes  = args.n_episodes,
            difficulty  = args.difficulty,
            k           = args.k_means,
            device      = device,
            rng         = np.random.default_rng(args.seed + 1),
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
