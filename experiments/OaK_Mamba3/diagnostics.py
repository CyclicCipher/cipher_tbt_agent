"""
OaK-Mamba3 Diagnostic Suite.

Runs 7 structural/mechanical checks on an untrained model.
No training required — these checks verify that the system is correctly
wired before a training run begins.

Usage:
    python experiments/OaK_Mamba3/diagnostics.py [--device cpu]
    python experiments/OaK_Mamba3/diagnostics.py --device cuda
"""

import os
import sys
import math

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'Mamba3'))

# ---------------------------------------------------------------------------
# Imports from oak_model.
# BiOaKMixer and _reverse_grid_positions are new exports — print a helpful
# error if oak_model.py has not been updated yet.
# ---------------------------------------------------------------------------
try:
    from oak_model import (
        OaKConfig, OaKModel, OaKOutput,
        SEP_TOKEN, QUERY_TOKEN, PAD_TOKEN, MASK_TOKEN, VOCAB_SIZE, NUM_COLORS,
        BiOaKMixer, _reverse_grid_positions,
    )
except ImportError as _e:
    print(
        f"[ERROR] Could not import from oak_model.py: {_e}\n"
        "        BiOaKMixer and _reverse_grid_positions must be exported from\n"
        "        oak_model.py before running this diagnostic suite.\n"
        "        Check that oak_model.py defines and exports both names."
    )
    sys.exit(1)

from env1 import sample_episode, generate_grid

# apply_masking and iterative_unmask may not yet be in train_env1.py.
# We try a combined import and fall back gracefully per function.
try:
    from train_env1 import (
        encode_episode, apply_masking, make_batch, compute_losses, iterative_unmask,
    )
    _HAS_MASKING = True
    _HAS_UNMASK  = True
except ImportError:
    # Partial import — encode_episode / make_batch / compute_losses are required.
    try:
        from train_env1 import encode_episode, make_batch, compute_losses
    except ImportError as _e2:
        print(f"[ERROR] Could not import core functions from train_env1.py: {_e2}")
        sys.exit(1)
    # Check individually which optional functions are available.
    try:
        from train_env1 import apply_masking
        _HAS_MASKING = True
    except ImportError:
        apply_masking = None
        _HAS_MASKING  = False

    try:
        from train_env1 import iterative_unmask
        _HAS_UNMASK = True
    except ImportError:
        iterative_unmask = None
        _HAS_UNMASK      = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config() -> OaKConfig:
    """Tiny model config used by all checks that need a model."""
    return OaKConfig(
        d_model=64, d_state=32, expand=2, headdim=32,
        chunk_size=64, n_layer=2, num_options=1, d_option=16, n_gvfs=2,
    )


def _make_small_batch(
    device: torch.device,
    H: int = 4,
    W: int = 4,
    K: int = 2,
    difficulty: int = 1,
    batch_size: int = 2,
    rng: np.random.Generator = None,
) -> dict:
    """Convenience wrapper around make_batch with small defaults."""
    if rng is None:
        rng = np.random.default_rng(42)
    config = _small_config()
    return make_batch(H, W, K, difficulty, batch_size, config.chunk_size, rng, device)


# ---------------------------------------------------------------------------
# Check 1: Random baseline comparison
# ---------------------------------------------------------------------------

def check_random_baseline(model: OaKModel, device: torch.device, n_samples: int = 20) -> bool:
    """Verify that an untrained model's task_loss is near the random baseline.

    Expected: task_loss ≈ log(10) ≈ 2.303 (uniform distribution over 10 colors).
    PASS: task_loss in [1.8, 3.0].
    FAIL: task_loss < 0.1 (impossible without training, indicates a bug) or NaN/inf.
    """
    model.eval()
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    losses = []
    with torch.no_grad():
        for _ in range(n_samples):
            batch   = _make_small_batch(device, rng=rng)
            outputs = model(batch['tokens'], batch['grid_segments'])
            _, task_loss, _ = compute_losses(outputs, batch)
            losses.append(task_loss.item())

    mean_loss       = float(np.mean(losses))
    random_baseline = math.log(NUM_COLORS)   # log(10) ≈ 2.303

    print(f"  mean task_loss   : {mean_loss:.4f}")
    print(f"  random baseline  : {random_baseline:.4f}  (log({NUM_COLORS}))")

    if not math.isfinite(mean_loss):
        print("  PROBLEM: task_loss is NaN or inf")
        return False
    if mean_loss < 0.1:
        print("  PROBLEM: task_loss < 0.1 — impossible on untrained model, likely a bug")
        return False
    if not (1.8 <= mean_loss <= 3.0):
        print(f"  PROBLEM: task_loss {mean_loss:.4f} outside expected range [1.8, 3.0]")
        return False
    return True


# ---------------------------------------------------------------------------
# Check 2: BiMamba bidirectionality
# ---------------------------------------------------------------------------

def check_bimamba_bidirectionality(model: OaKModel, device: torch.device) -> bool:
    """Verify that the backward sub-mixer contributes to BiOaKMixer output.

    PASS: norm(backward contribution) > 1% of norm(full output).
    Also checks that fwd and bwd have different parameter values.
    """
    torch.manual_seed(42)
    model.eval()

    T       = 128
    d_model = model.config.d_model
    u       = torch.randn(1, T, d_model, device=device)

    # One 4x4 grid starting at position 5
    grid_positions = [(5, 4, 4)]

    mixer: BiOaKMixer = model.layers[0].mixer

    with torch.no_grad():
        y_full     = mixer(u, grid_positions)              # (1, T, d_model)
        y_fwd_only = 0.5 * mixer.fwd(u, grid_positions)   # forward-only contribution

    y_bwd_contribution = y_full - y_fwd_only

    norm_full = y_full.norm().item()
    norm_fwd  = y_fwd_only.norm().item()
    norm_bwd  = y_bwd_contribution.norm().item()
    ratio     = norm_bwd / (norm_full + 1e-12)

    print(f"  norm(forward contribution) : {norm_fwd:.4f}")
    print(f"  norm(backward contribution): {norm_bwd:.4f}")
    print(f"  ratio (bwd/full)           : {ratio:.4f}  (need > 0.01)")

    # Check that fwd and bwd have different parameters (independent init)
    fwd_w        = mixer.fwd.in_proj.weight
    bwd_w        = mixer.bwd.in_proj.weight
    params_differ = not torch.allclose(fwd_w, bwd_w)
    print(f"  fwd/bwd in_proj.weight differ: {params_differ}")

    if ratio <= 0.01:
        print("  PROBLEM: backward contribution is < 1% of total — BiOaKMixer may not be bidirectional")
        return False
    if not params_differ:
        print("  PROBLEM: fwd and bwd in_proj.weight are identical — check initialization")
        return False
    return True


# ---------------------------------------------------------------------------
# Check 3: Masking correctness
# ---------------------------------------------------------------------------

def check_masking_correctness(device: torch.device) -> bool:
    """Verify that apply_masking correctly masks the test output region.

    Conditions checked:
      a) All positions [label_start, label_start+HW) are MASK_TOKEN in masked_tokens.
      b) All positions [label_start, label_start+HW) are True in mask_array.
      c) Positions before label_start are unchanged (ex_mask_prob=0.0).
      d) Original tokens at label_start..label_start+HW are all valid colors (0-9).
    """
    if not _HAS_MASKING:
        print("  SKIP: apply_masking not found in train_env1.py")
        print("        Expected signature:")
        print("          apply_masking(tokens, grid_segments, label_start, HW, rng,")
        print("                        test_mask_rate, ex_mask_prob) -> (masked_tokens, mask_array)")
        return False

    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    H, W, K = 4, 4, 2
    HW = H * W

    ep = sample_episode(H=H, W=W, K=K, difficulty=1, rng=rng)
    tokens, grid_segments, label_start = encode_episode(ep, H, W)

    # Fully mask the test output; do not touch example outputs.
    masked_tokens, mask_array = apply_masking(
        tokens, grid_segments, label_start, HW, rng,
        test_mask_rate=1.0, ex_mask_prob=0.0,
    )

    n_masked = int(mask_array.sum())
    print(f"  label_start  : {label_start}")
    print(f"  HW           : {HW}")
    print(f"  masked count : {n_masked}  (expected {HW})")

    violations = []

    # Condition a: test output positions are MASK_TOKEN
    for i in range(label_start, label_start + HW):
        tok = masked_tokens[i]
        if tok != MASK_TOKEN:
            violations.append(f"position {i}: expected MASK_TOKEN({MASK_TOKEN}), got {tok}")

    # Condition b: mask_array is True at test output positions
    for i in range(label_start, label_start + HW):
        if not bool(mask_array[i]):
            violations.append(f"mask_array[{i}] is False, expected True")

    # Condition c: positions before label_start are unchanged
    # (ex_mask_prob=0.0 so no example masking can occur)
    for i in range(label_start):
        if tokens[i] != masked_tokens[i]:
            violations.append(
                f"position {i} outside test region changed: {tokens[i]} -> {masked_tokens[i]}"
            )

    # Condition d: original tokens at test output are valid colors 0-9
    for i in range(label_start, label_start + HW):
        orig = tokens[i]
        if not (0 <= orig <= 9):
            violations.append(f"original token at {i} is {orig}, expected color 0-9")

    if violations:
        for v in violations[:5]:
            print(f"  VIOLATION: {v}")
        if len(violations) > 5:
            print(f"  ... and {len(violations) - 5} more violations")
        return False

    print("  All 4 masking conditions satisfied.")
    return True


# ---------------------------------------------------------------------------
# Check 4: Gradient flow through BiMamba
# ---------------------------------------------------------------------------

def check_gradient_flow(model: OaKModel, device: torch.device) -> bool:
    """Verify that gradients reach both the forward and backward sub-mixers.

    PASS: grad_norm > 1e-8 for both fwd.in_proj.weight and bwd.in_proj.weight
    in model.layers[0].mixer.
    """
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    model.train()
    batch   = _make_small_batch(device, batch_size=2, rng=rng)
    outputs = model(batch['tokens'], batch['grid_segments'])
    total_loss, _, _ = compute_losses(outputs, batch)

    model.zero_grad()
    total_loss.backward()

    mixer: BiOaKMixer = model.layers[0].mixer
    fwd_grad = mixer.fwd.in_proj.weight.grad
    bwd_grad = mixer.bwd.in_proj.weight.grad

    fwd_norm = fwd_grad.norm().item() if fwd_grad is not None else 0.0
    bwd_norm = bwd_grad.norm().item() if bwd_grad is not None else 0.0

    print(f"  layers[0].mixer.fwd.in_proj.weight grad_norm: {fwd_norm:.2e}")
    print(f"  layers[0].mixer.bwd.in_proj.weight grad_norm: {bwd_norm:.2e}")

    passed = True
    if fwd_grad is None or fwd_norm <= 1e-8:
        print("  PROBLEM: forward mixer has zero/None gradient on in_proj.weight")
        passed = False
    if bwd_grad is None or bwd_norm <= 1e-8:
        print("  PROBLEM: backward mixer has zero/None gradient on in_proj.weight")
        passed = False
    return passed


# ---------------------------------------------------------------------------
# Check 5: Iterative unmasking structural check
# ---------------------------------------------------------------------------

def check_iterative_unmasking(
    model: OaKModel,
    device: torch.device,
    n_episodes: int = 50,
) -> bool:
    """Verify that iterative_unmask runs without error and produces correct output.

    On an untrained model both step-1 and step-10 accuracy will be ~10%;
    the check only verifies structural correctness (no NaN/error, correct shape,
    step-10 accuracy >= step-1 accuracy by at least -5pp tolerance).

    iterative_unmask signature:
        (model, tokens_masked, tokens_orig, grid_segments, label_start, HW,
         n_steps, device) -> (predictions: Tensor, exact_match: float)
    """
    if not _HAS_UNMASK:
        print("  SKIP: iterative_unmask not found in train_env1.py")
        print("        Expected signature:")
        print("          iterative_unmask(model, tokens_masked, tokens_orig,")
        print("                           grid_segments, label_start, HW, n_steps, device)")
        print("          -> (predictions: Tensor(HW,), exact_match: float)")
        return False

    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    H, W, K = 5, 5, 3
    HW = H * W

    config = _small_config()

    accs_step1  = []
    accs_step10 = []
    output_shape = None

    for _ in range(n_episodes):
        batch = make_batch(H, W, K, 1, 1, config.chunk_size, rng, device)
        tokens_masked = batch['tokens']          # (1, T)
        tokens_orig   = batch['tokens_orig']     # (1, T)
        grid_segments = batch['grid_segments']
        label_start   = batch['label_starts'][0]

        try:
            preds1, em1 = iterative_unmask(
                model, tokens_masked, tokens_orig, grid_segments,
                label_start, HW, n_steps=1, device=device,
            )
            preds10, em10 = iterative_unmask(
                model, tokens_masked, tokens_orig, grid_segments,
                label_start, HW, n_steps=10, device=device,
            )
        except Exception as e:
            print(f"  ERROR during iterative_unmask: {e}")
            return False

        # Convert exact_match to per-cell accuracy for comparison
        # (exact_match is 1.0 if all HW cells correct, 0.0 otherwise)
        # Use it as a proxy — per-cell accuracy from argmax vs labels would be richer,
        # but iterative_unmask only returns exact_match.
        accs_step1.append(float(em1))
        accs_step10.append(float(em10))

        if output_shape is None:
            output_shape = tuple(preds1.shape)

    mean_acc1  = float(np.mean(accs_step1))
    mean_acc10 = float(np.mean(accs_step10))

    print(f"  step-1  exact-match accuracy : {mean_acc1*100:.1f}%  (expect ~10% untrained)")
    print(f"  step-10 exact-match accuracy : {mean_acc10*100:.1f}%  (expect ~10% untrained)")
    if output_shape is not None:
        print(f"  predictions shape            : {output_shape}  (expected ({HW},))")

    if not (math.isfinite(mean_acc1) and math.isfinite(mean_acc10)):
        print("  PROBLEM: NaN/inf accuracy value")
        return False

    # Structural shape check
    if output_shape is not None and output_shape != (HW,):
        print(f"  PROBLEM: predictions shape {output_shape} != expected ({HW},)")
        return False

    # Step-10 should not be dramatically worse than step-1 on an untrained model
    if mean_acc10 < mean_acc1 - 0.05:
        print(
            f"  PROBLEM: step-10 accuracy ({mean_acc10:.3f}) is much worse than "
            f"step-1 ({mean_acc1:.3f}) — iterative unmasking is regressing"
        )
        return False

    print("  Structural check passed: iterative_unmask runs cleanly.")
    return True


# ---------------------------------------------------------------------------
# Check 6: Environment diversity
# ---------------------------------------------------------------------------

def check_env_diversity(n_episodes: int = 200) -> bool:
    """Verify that the env1 rule sampler generates diverse, non-trivial episodes.

    PASS: < 5% trivial (output == input) AND >= 5 distinct rule types AND
          0.1 < mean_cell_change < 0.9.
    WARNING: if 0 < n_trivial < 10.
    """
    rng = np.random.default_rng(42)
    H, W, K = 6, 6, 3

    n_trivial     = 0
    rule_prefixes = set()
    cell_changes  = []

    for i in range(n_episodes):
        difficulty = 1 + (i % 3)   # rotate through 1, 2, 3
        ep = sample_episode(H=H, W=W, K=K, difficulty=difficulty, rng=rng)

        rule_prefix = ep['rule_name'][:20]
        rule_prefixes.add(rule_prefix)

        inp = ep['test_input']
        out = ep['test_output']
        changed_frac = float((inp != out).mean())
        cell_changes.append(changed_frac)

        if np.array_equal(inp, out):
            n_trivial += 1

    mean_cell_change = float(np.mean(cell_changes))
    n_rule_types     = len(rule_prefixes)

    print(f"  n_episodes       : {n_episodes}")
    print(f"  n_trivial        : {n_trivial}  ({n_trivial/n_episodes*100:.1f}%)")
    print(f"  n_rule_types     : {n_rule_types}  (first-20-char prefixes)")
    print(f"  mean_cell_change : {mean_cell_change:.3f}")

    if 0 < n_trivial < 10:
        print("  WARNING: small number of trivial episodes — probably fine")

    passed = (
        (n_trivial / n_episodes) < 0.05
        and n_rule_types >= 5
        and 0.1 < mean_cell_change < 0.9
    )

    if not passed:
        if (n_trivial / n_episodes) >= 0.05:
            print(f"  PROBLEM: {n_trivial/n_episodes*100:.1f}% trivial episodes (need < 5%)")
        if n_rule_types < 5:
            print(f"  PROBLEM: only {n_rule_types} distinct rule types (need >= 5)")
        if not (0.1 < mean_cell_change < 0.9):
            print(f"  PROBLEM: mean_cell_change {mean_cell_change:.3f} outside (0.1, 0.9)")

    return passed


# ---------------------------------------------------------------------------
# Check 7: Example sensitivity
# ---------------------------------------------------------------------------

def check_example_sensitivity(
    model: OaKModel,
    device: torch.device,
    n_episodes: int = 50,
) -> bool:
    """Verify that masking example outputs does not reduce task_loss (no cheating).

    On an untrained model both losses should be finite and similar.
    Key test: loss_without_examples >= loss_with_examples * 0.9
    (removing information should not help the model).

    To build the "no example outputs" batch, all positions with seg_id == 1
    (example output grids) in batch['tokens'] are replaced with MASK_TOKEN,
    and batch['mask'] is updated to mark those positions as needing prediction.
    The model should not gain from seeing masked-out example outputs.
    """
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    H, W, K = 5, 5, 3
    config = _small_config()

    losses_with    = []
    losses_without = []

    model.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            batch = make_batch(H, W, K, 1, 1, config.chunk_size, rng, device)

            # Normal forward pass (examples visible in masked tokens as usual)
            outputs_normal = model(batch['tokens'], batch['grid_segments'])
            _, task_loss_normal, _ = compute_losses(outputs_normal, batch)
            losses_with.append(task_loss_normal.item())

            # Build modified batch: replace all example OUTPUT token positions
            # (seg_id == 1) in tokens with MASK_TOKEN and mark them as masked.
            tokens_mod = batch['tokens'].clone()
            mask_mod   = batch['mask'].clone()

            for start, seg_H, seg_W, seg_id in batch['grid_segments']:
                if seg_id == 1:   # example output — not test output (seg_id=2)
                    end = start + seg_H * seg_W
                    tokens_mod[:, start:end] = MASK_TOKEN
                    mask_mod[:, start:end]   = True

            batch_mod = dict(batch)
            batch_mod['tokens'] = tokens_mod
            batch_mod['mask']   = mask_mod

            outputs_blind = model(tokens_mod, batch['grid_segments'])
            _, task_loss_blind, _ = compute_losses(outputs_blind, batch_mod)
            losses_without.append(task_loss_blind.item())

    mean_with    = float(np.mean(losses_with))
    mean_without = float(np.mean(losses_without))

    print(f"  mean task_loss (examples visible)      : {mean_with:.4f}")
    print(f"  mean task_loss (example outputs masked): {mean_without:.4f}")

    if not (math.isfinite(mean_with) and math.isfinite(mean_without)):
        print("  PROBLEM: NaN/inf in one of the loss values")
        return False

    # Masking examples should not help — loss should not drop significantly
    threshold = mean_with * 0.9
    if mean_without < threshold:
        print(
            f"  PROBLEM: loss dropped from {mean_with:.4f} to {mean_without:.4f} "
            f"after masking example outputs — model may be cheating"
        )
        return False

    print("  Both losses finite and similar — no cheating detected.")
    return True


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_diagnostics(device_str: str = 'cpu') -> None:
    device = torch.device(device_str)

    config = _small_config()
    torch.manual_seed(42)
    model = OaKModel(config).to(device)

    print("=" * 60)
    print("OaK-Mamba3 Diagnostic Suite")
    print("=" * 60)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Device: {device}")
    print()

    checks = [
        ("1. Random baseline",         lambda: check_random_baseline(model, device)),
        ("2. BiMamba bidirectionality", lambda: check_bimamba_bidirectionality(model, device)),
        ("3. Masking correctness",      lambda: check_masking_correctness(device)),
        ("4. Gradient flow",            lambda: check_gradient_flow(model, device)),
        ("5. Iterative unmasking",      lambda: check_iterative_unmasking(model, device)),
        ("6. Environment diversity",    lambda: check_env_diversity()),
        ("7. Example sensitivity",      lambda: check_example_sensitivity(model, device)),
    ]

    results = {}

    for name, fn in checks:
        print(f"--- {name} ---")
        try:
            passed = fn()
            status = "PASS" if passed else "FAIL"
        except Exception as e:
            status = f"ERROR: {e}"
            passed = False
        results[name] = status
        print(f"Result: {status}")
        print()

    print("=" * 60)
    print("Summary:")
    for name, status in results.items():
        print(f"  {name}: {status}")

    n_pass = sum(1 for s in results.values() if s == "PASS")
    print(f"\n{n_pass}/{len(results)} checks passed")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='OaK-Mamba3 Diagnostic Suite')
    parser.add_argument('--device', default='cpu', help='torch device (cpu or cuda)')
    args = parser.parse_args()
    run_all_diagnostics(args.device)
