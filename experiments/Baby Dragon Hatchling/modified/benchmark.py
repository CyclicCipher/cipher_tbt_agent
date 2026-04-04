"""BDH Benchmark Suite — systematic capability testing.

Tests a small BDH model on tasks of increasing difficulty.
Reports exact-match and character accuracy for each (config, level) pair.

Usage:
    # Single level, single config:
    python benchmark.py --level copy --config tiny --max_iters 500

    # All levels, single config:
    python benchmark.py --config small --all

    # All levels, all configs (full suite):
    python benchmark.py --all --all_configs
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from bdh import BDH, BDHConfig
from data.benchmarks import (
    generate_copy, generate_reverse, generate_sort,
    generate_arithmetic, generate_successor, generate_sudoku_4x4,
    find_output_start, OUT_MARKER_BYTES,
)


# ---------------------------------------------------------------------------
# Model configs (all <= 1.4M params)
# ---------------------------------------------------------------------------

CONFIGS = {
    "tiny":   BDHConfig(n_layer=4, n_embd=64,  n_head=4, mlp_internal_dim_multiplier=16,  vocab_size=256),
    "small":  BDHConfig(n_layer=6, n_embd=64,  n_head=4, mlp_internal_dim_multiplier=32,  vocab_size=256),
    "medium": BDHConfig(n_layer=6, n_embd=128, n_head=4, mlp_internal_dim_multiplier=16,  vocab_size=256),
    "large":  BDHConfig(n_layer=6, n_embd=128, n_head=4, mlp_internal_dim_multiplier=24,  vocab_size=256),
}


# ---------------------------------------------------------------------------
# Data loading per level
# ---------------------------------------------------------------------------

def load_level_data(level: str) -> tuple[list[str], list[str]]:
    """Return (train_examples, test_examples) for a benchmark level."""
    if level == "copy":
        train = generate_copy(800, length=8)
        test = generate_copy(200, length=8, seed=99)
        return train, test
    elif level == "reverse":
        train = generate_reverse(800, length=8)
        test = generate_reverse(200, length=8, seed=99)
        return train, test
    elif level == "sort":
        train = generate_sort(800, length=8)
        test = generate_sort(200, length=8, seed=99)
        return train, test
    elif level == "arithmetic":
        return generate_arithmetic()
    elif level == "successor":
        return generate_successor()
    elif level == "sudoku4":
        return generate_sudoku_4x4(n_train=5000, n_test=500, n_clues=8)
    elif level == "calendar":
        from data.calendar_data import calendar_training_strings, calendar_test_cases
        train_strs = calendar_training_strings()
        # Calendar training strings are already full sentences. We need to
        # reformat them with <out> markers for loss masking.
        train = []
        for s in train_strs[:200]:  # seconds only — first 200
            # Find "the date will be " and insert <out> there.
            marker = "the date will be "
            idx = s.find(marker)
            if idx != -1:
                input_part = s[:idx + len(marker)]
                output_part = s[idx + len(marker):]
                # Remove the <calendar> and <end> wrappers, add <cal> and <out>.
                input_clean = input_part.replace("<calendar>", "<cal>").replace("<end>", "")
                output_clean = output_part.replace("<end>", "").rstrip(".")
                train.append(f"{input_clean}<out>{output_clean}")
        test_cases = calendar_test_cases()
        test = []
        for inp, expected in test_cases:
            inp_clean = inp.replace("<calendar>", "<cal>")
            test.append(f"{inp_clean}<out>{expected}")
        return train, test
    elif level == "sudoku9":
        from data.sudoku import generate_dataset, format_for_training
        train_pairs, test_pairs = generate_dataset(
            n_train=1000, n_test=100, difficulty="easy",
        )
        train = [f"<sud9>{p}<out>{s}" for p, s in train_pairs]
        test = [f"<sud9>{p}<out>{s}" for p, s in test_pairs]
        return train, test
    else:
        raise ValueError(f"Unknown level: {level}")


# ---------------------------------------------------------------------------
# Encoding and batching with loss masking
# ---------------------------------------------------------------------------

def encode_example(example: str) -> tuple[list[int], int]:
    """Encode a string to byte tokens. Returns (tokens, output_start_idx)."""
    tokens = list(example.encode("utf-8"))
    out_idx = find_output_start(tokens)
    return tokens, out_idx


def make_batch(examples: list[str], batch_size: int, block_size: int,
               device: torch.device, rng: np.random.RandomState
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a batch with loss masking.

    Returns (x, y) where y has -100 for positions that should be ignored
    (everything before and including <out>).
    """
    indices = rng.randint(0, len(examples), size=(batch_size,))

    x_batch = []
    y_batch = []

    for idx in indices:
        tokens, out_start = encode_example(examples[idx])

        # Pad or truncate to block_size.
        if len(tokens) > block_size:
            tokens = tokens[:block_size]
            out_start = min(out_start, block_size)
        elif len(tokens) < block_size:
            tokens = tokens + [0] * (block_size - len(tokens))

        x = tokens[:-1]  # input: all but last
        y = tokens[1:]    # target: all but first

        # Mask: set positions before output start to -100.
        # out_start is the index of the first target byte in the original tokens.
        # In the shifted y, the first target byte is at position out_start - 1.
        mask_end = max(0, out_start - 1)
        for i in range(mask_end):
            y[i] = -100

        # Also mask padding.
        for i in range(len(y)):
            if x[i] == 0 and y[i] == 0:
                y[i] = -100

        x_batch.append(x)
        y_batch.append(y)

    return (
        torch.tensor(x_batch, dtype=torch.long, device=device),
        torch.tensor(y_batch, dtype=torch.long, device=device),
    )


# ---------------------------------------------------------------------------
# Evaluation: generate output and compare
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, test_examples: list[str], device: torch.device,
             max_gen: int = 100) -> dict:
    """Evaluate model on test examples.

    For each example, feed the input portion (before <out>), then
    generate tokens autoregressively and compare to expected output.
    """
    # Use the unwrapped model for generation (torch.compile + dynamic
    # sequence lengths causes recompilations).
    raw_model = getattr(model, '_orig_mod', model)
    raw_model.eval()
    exact_matches = 0
    total_chars = 0
    correct_chars = 0

    for example in test_examples:
        tokens, out_start = encode_example(example)
        if out_start < 0:
            continue

        input_tokens = tokens[:out_start]
        expected_tokens = tokens[out_start:]

        # Feed input.
        idx = torch.tensor([input_tokens], dtype=torch.long, device=device)

        # Generate.
        gen_len = min(len(expected_tokens), max_gen)
        generated = raw_model.generate(idx, max_new_tokens=gen_len, temperature=0.01, top_k=1)
        gen_tokens = generated[0, len(input_tokens):].tolist()

        # Compare.
        expected = expected_tokens[:gen_len]
        is_exact = gen_tokens == expected
        if is_exact:
            exact_matches += 1

        for g, e in zip(gen_tokens, expected):
            total_chars += 1
            if g == e:
                correct_chars += 1

    n = len(test_examples)
    model.train()
    return {
        "exact_match": exact_matches / max(n, 1),
        "char_accuracy": correct_chars / max(total_chars, 1),
        "n_test": n,
        "exact_matches": exact_matches,
    }


# ---------------------------------------------------------------------------
# Training loop for one (config, level) pair
# ---------------------------------------------------------------------------

def run_benchmark(
    config_name: str,
    level: str,
    max_iters: int = 2000,
    batch_size: int = 32,
    lr: float = 1e-3,
    eval_interval: int = 200,
    device: torch.device | None = None,
    use_compile: bool = False,
) -> dict:
    """Train and evaluate one (config, level) pair. Returns results dict."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = CONFIGS[config_name]
    n_params = 3 * config.mlp_internal_dim_multiplier * config.n_embd**2 + 2 * 256 * config.n_embd
    N = config.mlp_internal_dim_multiplier * config.n_embd // config.n_head

    print(f"\n{'='*60}")
    print(f"  {config_name} ({n_params:,} params, N={N}) x {level}")
    print(f"{'='*60}")

    # Load data.
    train_data, test_data = load_level_data(level)
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    if train_data:
        sample = train_data[0]
        print(f"  Example: {sample[:80]}{'...' if len(sample) > 80 else ''}")

    # Determine block_size from data.
    max_len = max(len(s.encode("utf-8")) for s in train_data + test_data) + 2
    block_size = min(max_len, 512)
    print(f"  Block size: {block_size}")

    # Create model.
    model = BDH(config).to(device)

    if use_compile and device.type == "cuda":
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # Precision.
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=(dtype == torch.float16))
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=dtype)
    rng = np.random.RandomState(42)

    best_exact = 0.0
    best_char = 0.0
    t0 = time.time()

    for step in range(max_iters):
        x, y = make_batch(train_data, batch_size, block_size, device, rng)

        with autocast_ctx:
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0:
            dt = time.time() - t0
            print(f"  step {step:5d} | loss {loss.item():.4f} | {dt:.1f}s")
            t0 = time.time()

        if (step > 0 and step % eval_interval == 0) or step == max_iters - 1:
            # Evaluate on a subset of test data (cap at 100 for speed).
            eval_subset = test_data[:100]
            results = evaluate(model, eval_subset, device)
            best_exact = max(best_exact, results["exact_match"])
            best_char = max(best_char, results["char_accuracy"])
            print(f"  EVAL step {step}: exact={results['exact_match']:.1%} "
                  f"char={results['char_accuracy']:.1%} "
                  f"({results['exact_matches']}/{results['n_test']})")

    # Save checkpoint.
    raw_model = getattr(model, '_orig_mod', model)
    ckpt_dir = os.path.join(_SCRIPT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{config_name}_{level}.pt")
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "config": config,
        "level": level,
        "config_name": config_name,
        "best_exact": best_exact,
        "best_char": best_char,
        "step": max_iters,
    }, ckpt_path)

    # Also save as latest.pt (overwriting).
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest_path):
        os.remove(latest_path)
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "config": config,
        "level": level,
        "config_name": config_name,
        "best_exact": best_exact,
        "best_char": best_char,
        "step": max_iters,
    }, latest_path)

    return {
        "config": config_name,
        "level": level,
        "params": n_params,
        "best_exact_match": best_exact,
        "best_char_accuracy": best_char,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_LEVELS = ["copy", "reverse", "sort", "arithmetic", "successor", "sudoku4", "calendar", "sudoku9"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="copy", choices=ALL_LEVELS)
    parser.add_argument("--config", type=str, default="small", choices=list(CONFIGS.keys()))
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--all", action="store_true", help="Run all levels")
    parser.add_argument("--all_configs", action="store_true", help="Run all configs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    levels = ALL_LEVELS if args.all else [args.level]
    configs = list(CONFIGS.keys()) if args.all_configs else [args.config]

    results = []

    for config_name in configs:
        for level in levels:
            result = run_benchmark(
                config_name=config_name,
                level=level,
                max_iters=args.max_iters,
                batch_size=args.batch_size,
                lr=args.lr,
                eval_interval=args.eval_interval,
                device=device,
                use_compile=args.compile,
            )
            results.append(result)

    # Print summary table.
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<10} {'Level':<12} {'Params':<10} {'Exact':>8} {'CharAcc':>8}")
    print(f"{'-'*10} {'-'*12} {'-'*10} {'-'*8} {'-'*8}")
    for r in results:
        print(f"{r['config']:<10} {r['level']:<12} {r['params']:<10,} "
              f"{r['best_exact_match']:>7.1%} {r['best_char_accuracy']:>7.1%}")


if __name__ == "__main__":
    main()
