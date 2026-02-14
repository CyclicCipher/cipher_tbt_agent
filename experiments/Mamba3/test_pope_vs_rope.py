"""
Compare RoPE vs PoPE on diagnostic tasks that expose what-where entanglement.

Three tasks (inspired by PoPE paper's indirect indexing diagnostic):

1. "Where" — predict the token at a fixed query position.
   Sequence: [random tokens...] | query = position index
   Target: token at that position.
   Requires: pure positional matching, content-irrelevant.

2. "What" — find a marked token regardless of position.
   Sequence: [random tokens... with one MARKER token]
   Target: token immediately after the marker.
   Requires: content matching (find marker), position-irrelevant.

3. "What+Where" — find a source token, then return the token at an
   offset from it (indirect indexing).
   Sequence: [random tokens... with one SOURCE token]  | query = offset
   Target: token at (position_of_source + offset).
   Requires: content match (find source) AND position match (apply offset).

RoPE should handle "What" fine but struggle on "Where" and "What+Where"
due to what-where entanglement.  PoPE should handle all three.

Usage:
    python experiments/Mamba3/test_pope_vs_rope.py [--epochs 100] [--device cuda]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.Mamba3.mamba3_block import Mamba3Config, Mamba3LM


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32          # 0..31: regular tokens
MARKER_TOKEN = VOCAB_SIZE      # 32: marker for "What" task
SOURCE_TOKEN = VOCAB_SIZE + 1  # 33: source for "What+Where" task
PAD_TOKEN = VOCAB_SIZE + 2     # 34: padding / query separator
TOTAL_VOCAB = VOCAB_SIZE + 3   # 35 tokens total


def generate_where_data(n: int, seq_len: int = 16) -> tuple[Tensor, Tensor]:
    """Where task: predict token at a queried position.

    Format: [t0, t1, ..., t_{L-2}, pos]
    Target: t_pos (the token at position `pos`)

    The last token is the position query (encoded as a token 0..seq_len-2).
    The model must output the token at that position.
    """
    data = torch.randint(0, VOCAB_SIZE, (n, seq_len))
    targets = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        pos = torch.randint(0, seq_len - 1, (1,)).item()
        targets[i] = data[i, pos].clone()
        data[i, -1] = pos  # query position encoded as token value
    return data, targets


def diagnose_where_by_position(model: nn.Module, seq_len: int, device: str,
                                n_per_pos: int = 500) -> dict[int, float]:
    """Per-position accuracy breakdown for the Where task.

    Generates n_per_pos test examples for EACH query position (0..seq_len-2)
    and measures accuracy separately.  This reveals whether the model can
    only retrieve from recent positions (decay hypothesis).

    Returns:
        Dict mapping query_position -> accuracy (0-100%).
    """
    model.eval()
    n_positions = seq_len - 1  # positions 0 .. seq_len-2
    pos_accs = {}

    with torch.no_grad():
        for pos in range(n_positions):
            data = torch.randint(0, VOCAB_SIZE, (n_per_pos, seq_len))
            data[:, -1] = pos  # all queries ask for same position
            targets = data[:, pos].clone()

            data_d, targets_d = data.to(device), targets.to(device)
            preds = model(data_d)[:, -1, :].argmax(-1)
            acc = (preds == targets_d).float().mean().item() * 100
            pos_accs[pos] = acc

    return pos_accs


def print_position_diagnosis(pos_accs: dict[int, float], mode_name: str,
                             seq_len: int):
    """Print per-position accuracy as a table and ASCII bar chart."""
    n_positions = seq_len - 1
    print(f"\n  {mode_name} — Per-position accuracy (query pos → accuracy):")
    print(f"  {'Pos':>4} {'Dist':>5} {'Acc':>7}  Bar")
    print(f"  {'----':>4} {'-----':>5} {'-------':>7}  ---")
    for pos in range(n_positions):
        acc = pos_accs.get(pos, 0.0)
        dist = n_positions - pos  # distance from query (last) position
        bar_len = int(acc / 2)  # 50 chars = 100%
        bar = '#' * bar_len
        print(f"  {pos:>4} {dist:>5} {acc:>6.1f}%  {bar}")

    # Summary stats
    accs = list(pos_accs.values())
    near_accs = [pos_accs[p] for p in range(n_positions - 3, n_positions)]
    far_accs = [pos_accs[p] for p in range(min(3, n_positions))]
    print(f"  Overall: {sum(accs)/len(accs):.1f}%  "
          f"Near(last 3): {sum(near_accs)/len(near_accs):.1f}%  "
          f"Far(first 3): {sum(far_accs)/len(far_accs):.1f}%")


def generate_what_data(n: int, seq_len: int = 16) -> tuple[Tensor, Tensor]:
    """What task: find the token after a MARKER.

    Format: [t0, ..., MARKER, t_answer, ..., t_{L-1}]
    Target: t_answer (token immediately after MARKER)

    Marker can appear at any position except the last two.
    """
    data = torch.randint(0, VOCAB_SIZE, (n, seq_len))
    targets = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        # Place marker at random position (not last 2, so answer fits)
        pos = torch.randint(0, seq_len - 2, (1,)).item()
        data[i, pos] = MARKER_TOKEN
        targets[i] = data[i, pos + 1].clone()
    return data, targets


def generate_whatwhere_data(n: int, seq_len: int = 16,
                           max_offset: int = 4) -> tuple[Tensor, Tensor]:
    """What+Where task: find SOURCE token, return token at (source_pos + offset).

    Format: [t0, ..., SOURCE, ..., t_{L-2}, offset]
    Target: t_{source_pos + offset}

    Last token encodes the offset. The model must find the SOURCE,
    then look at the position (source_pos + offset) to get the answer.
    """
    data = torch.randint(0, VOCAB_SIZE, (n, seq_len))
    targets = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        # Place source such that source_pos + max_offset < seq_len - 1
        max_src = seq_len - 2 - max_offset
        if max_src < 1:
            max_src = 1
        src_pos = torch.randint(0, max_src, (1,)).item()
        offset = torch.randint(1, max_offset + 1, (1,)).item()
        target_pos = src_pos + offset

        data[i, src_pos] = SOURCE_TOKEN
        data[i, -1] = offset  # offset query in last position
        targets[i] = data[i, target_pos].clone()
    return data, targets


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_and_eval(model: nn.Module, train_data: Tensor, train_targets: Tensor,
                   test_data: Tensor, test_targets: Tensor,
                   epochs: int, device: str, lr: float = 1e-3,
                   batch_size: int = 128) -> dict:
    """Train model on a task and return accuracy curves."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(train_data, train_targets)
    test_ds = TensorDataset(test_data, test_targets)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    train_accs, test_accs = [], []

    for epoch in range(epochs):
        # Train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            # Use last position's logits as prediction
            loss = F.cross_entropy(logits[:, -1, :], yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            correct_train, total_train = 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)[:, -1, :].argmax(-1)
                correct_train += (preds == yb).sum().item()
                total_train += yb.size(0)

            correct_test, total_test = 0, 0
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)[:, -1, :].argmax(-1)
                correct_test += (preds == yb).sum().item()
                total_test += yb.size(0)

        train_acc = correct_train / total_train * 100
        test_acc = correct_test / total_test * 100
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}: train {train_acc:5.1f}%  test {test_acc:5.1f}%")

    return {"train_accs": train_accs, "test_accs": test_accs}


def run_comparison(task_name: str, gen_fn, epochs: int, device: str,
                   seq_len: int = 16, n_train: int = 5000,
                   n_test: int = 1000, diagnose_where: bool = False,
                   **gen_kwargs):
    """Run RoPE vs PoPE on a task and report results."""
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    train_data, train_targets = gen_fn(n_train, seq_len=seq_len, **gen_kwargs)
    test_data, test_targets = gen_fn(n_test, seq_len=seq_len, **gen_kwargs)

    results = {}
    models = {}
    for mode_name, use_pope in [("RoPE", False), ("PoPE", True)]:
        print(f"\n--- {mode_name} ---")
        cfg = Mamba3Config(
            d_model=64, d_state=32, expand=2, headdim=32,
            n_layer=2, chunk_size=seq_len, use_pope=use_pope,
        )
        model = Mamba3LM(cfg, vocab_size=TOTAL_VOCAB)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,}")

        t0 = time.perf_counter()
        res = train_and_eval(
            model, train_data, train_targets,
            test_data, test_targets,
            epochs=epochs, device=device,
        )
        elapsed = time.perf_counter() - t0
        print(f"  time: {elapsed:.1f}s")
        results[mode_name] = res
        models[mode_name] = model

    # Summary
    for mode_name in ["RoPE", "PoPE"]:
        r = results[mode_name]
        print(f"  {mode_name}: final train={r['train_accs'][-1]:.1f}%  "
              f"test={r['test_accs'][-1]:.1f}%  "
              f"best_test={max(r['test_accs']):.1f}%")

    # Per-position diagnosis for Where task
    if diagnose_where:
        print(f"\n{'- '*30}")
        print("Per-position accuracy diagnosis (Where task)")
        print(f"{'- '*30}")
        for mode_name in ["RoPE", "PoPE"]:
            pos_accs = diagnose_where_by_position(
                models[mode_name], seq_len, device,
            )
            print_position_diagnosis(pos_accs, mode_name, seq_len)
            results[mode_name]["pos_accs"] = pos_accs

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RoPE vs PoPE diagnostic comparison")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--task", type=str, default="all",
                        choices=["where", "what", "whatwhere", "all"])
    parser.add_argument("--diagnose_where", action="store_true",
                        help="Run per-position accuracy breakdown on Where task")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Seq len: {args.seq_len}, Train: {args.n_train}, Test: {args.n_test}")

    all_results = {}

    if args.task in ("where", "all"):
        all_results["Where"] = run_comparison(
            "Where (position-only matching)",
            generate_where_data, args.epochs, args.device,
            seq_len=args.seq_len, n_train=args.n_train, n_test=args.n_test,
            diagnose_where=args.diagnose_where,
        )

    if args.task in ("what", "all"):
        all_results["What"] = run_comparison(
            "What (content-only matching)",
            generate_what_data, args.epochs, args.device,
            seq_len=args.seq_len, n_train=args.n_train, n_test=args.n_test,
        )

    if args.task in ("whatwhere", "all"):
        all_results["What+Where"] = run_comparison(
            "What+Where (indirect indexing)",
            generate_whatwhere_data, args.epochs, args.device,
            seq_len=args.seq_len, n_train=args.n_train, n_test=args.n_test,
            max_offset=4,
        )

    # Final summary table
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<20} {'RoPE test':>12} {'PoPE test':>12} {'Delta':>8}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*8}")
    for task_name, res in all_results.items():
        rope_best = max(res["RoPE"]["test_accs"])
        pope_best = max(res["PoPE"]["test_accs"])
        delta = pope_best - rope_best
        sign = "+" if delta > 0 else ""
        print(f"{task_name:<20} {rope_best:>11.1f}% {pope_best:>11.1f}% {sign}{delta:>6.1f}%")


if __name__ == "__main__":
    main()
