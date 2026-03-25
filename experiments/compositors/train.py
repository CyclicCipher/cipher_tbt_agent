"""
Training script for Compositor toy experiments.

Trains a Compositor model and evaluates on out-of-distribution test sequences.
Produces RESULTS.md -- a diagnostic log of successes and failures.

Optimised for fast iteration:
- torch.compile for kernel fusion
- BF16 autocast to halve memory bandwidth
- Large batches to maximise GPU utilisation
"""

import argparse
import os
import random
import time
from datetime import datetime

import torch
import torch.nn.functional as F

from data import (
    VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID,
    encode, decode, collate, make_train_test_split,
)
from model import Compositor
from structural_ops import seed_graph_from_data, seed_oracle_graph, run_structural_ops, graph_stats


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def classify_task(s: str) -> str:
    """Classify a sequence string into its task type."""
    if "+" in s and "=" in s:
        return "addition"
    elif "<" in s:
        return "less_than"
    elif ">" in s:
        return "greater_than"
    elif "," in s:
        return "succession"
    return "unknown"


def train_epoch(model, sequences, optimizer, device, batch_size=256, max_len=None):
    """Train one epoch. Returns average loss."""
    model.train()
    random.shuffle(sequences)

    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        input_ids, target_ids, mask = collate(batch_seqs, max_len=max_len)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        mask = mask.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss_per_token = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                target_ids.view(-1),
                reduction="none",
            ).view_as(target_ids)
            loss = (loss_per_token * mask).sum() / mask.sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += (loss_per_token * mask).sum().item()
        total_tokens += mask.sum().item()

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, sequences, device, max_len=None):
    """Evaluate on sequences. Returns per-token accuracy and loss."""
    model.eval()
    if not sequences:
        return 0.0, 0.0
    input_ids, target_ids, mask = collate(sequences, max_len=max_len)
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    mask = mask.to(device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids)

    preds = logits.argmax(dim=-1)
    correct = ((preds == target_ids).float() * mask).sum().item()
    total = mask.sum().item()

    loss_per_token = F.cross_entropy(
        logits.float().view(-1, VOCAB_SIZE),
        target_ids.view(-1),
        reduction="none",
    ).view_as(target_ids)
    loss = (loss_per_token * mask).sum().item() / max(total, 1)

    return correct / max(total, 1), loss


@torch.no_grad()
def evaluate_by_task(model, sequences, device, max_len=None):
    """Evaluate accuracy broken down by task type."""
    tasks = {}
    for s in sequences:
        t = classify_task(s)
        tasks.setdefault(t, []).append(s)

    results = {}
    for task_name, task_seqs in tasks.items():
        if task_seqs:
            acc, loss = evaluate(model, task_seqs, device, max_len=max_len)
            results[task_name] = {"acc": acc, "loss": loss, "n": len(task_seqs)}
    return results


@torch.no_grad()
def detailed_eval(model, sequences, device, batch_size=512, max_len=None):
    """Evaluate sequences in batches, returning per-sequence results."""
    model.eval()
    results = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        input_ids, target_ids, mask = collate(batch_seqs, max_len=max_len)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        mask = mask.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
        preds = logits.argmax(dim=-1)

        correct_mask = (preds == target_ids).float() * mask
        per_seq_correct = correct_mask.sum(dim=1)
        per_seq_total = mask.sum(dim=1).clamp(min=1)
        per_seq_acc = per_seq_correct / per_seq_total

        preds_cpu = preds.cpu().tolist()
        tgt_cpu = target_ids.cpu().tolist()
        mask_cpu = mask.cpu().tolist()
        acc_cpu = per_seq_acc.cpu().tolist()

        for j, seq in enumerate(batch_seqs):
            pred_str = "".join(
                decode([pid]) for pid, m in zip(preds_cpu[j], mask_cpu[j]) if m > 0
            )
            tgt_str = "".join(
                decode([tid]) for tid, m in zip(tgt_cpu[j], mask_cpu[j]) if m > 0
            )
            results.append({
                "seq": seq,
                "task": classify_task(seq),
                "predicted": pred_str,
                "correct": tgt_str,
                "acc": acc_cpu[j],
                "perfect": acc_cpu[j] == 1.0,
            })

    return results


@torch.no_grad()
def generate_completion(model, prompt: str, max_new_tokens: int, device) -> str:
    """Autoregressively generate tokens after the prompt."""
    model.eval()
    ids = encode(prompt)
    ids = ids[:-1]  # remove EOS

    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        next_id = logits[0, -1].argmax().item()
        if next_id == EOS_ID or next_id == PAD_ID:
            break
        ids.append(next_id)

    return decode(ids)


def write_results_log(model_results: dict, train_seqs: list, test_seqs: list,
                      args, sample_rate: float = 0.10):
    """Write RESULTS.md with sampled successes and failures."""
    path = os.path.join(SCRIPT_DIR, "RESULTS.md")
    rng = random.Random(42)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Compositor Experiment Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Config:** d_model={args.d_model}, n_heads={args.n_heads}, "
                f"n_layers={args.n_layers}, d_hidden={args.d_hidden}, "
                f"max_steps={args.max_steps}, "
                f"epochs={args.epochs}, lr={args.lr}, seed={args.seed}\n\n")
        f.write(f"**Data:** {len(train_seqs)} train, {len(test_seqs)} test\n\n")
        f.write(f"**Parameters:** {model_results['n_params']:,}\n\n")
        f.write(f"**Sample rate:** {sample_rate:.0%} of successes/failures shown\n\n")

        for split_name, split_results in [("Train", model_results["train_results"]),
                                           ("Test", model_results["test_results"])]:
            f.write(f"---\n\n## {split_name}\n\n")

            by_task = {}
            for r in split_results:
                by_task.setdefault(r["task"], []).append(r)

            for task in sorted(by_task.keys()):
                task_results = by_task[task]
                successes = [r for r in task_results if r["perfect"]]
                failures = [r for r in task_results if not r["perfect"]]
                task_acc = len(successes) / len(task_results) if task_results else 0

                f.write(f"### {task} ({len(successes)}/{len(task_results)} "
                        f"perfect = {task_acc:.1%})\n\n")

                if failures:
                    n_sample = max(1, int(len(failures) * sample_rate))
                    sampled = rng.sample(failures, min(n_sample, len(failures)))
                    f.write(f"**Failures** (showing {len(sampled)}/{len(failures)}):\n\n")
                    f.write("| Input | Expected | Predicted | Token Acc |\n")
                    f.write("|-------|----------|-----------|----------|\n")
                    for r in sampled:
                        f.write(f"| `{r['seq']}` | `{r['correct']}` | "
                                f"`{r['predicted']}` | {r['acc']:.1%} |\n")
                    f.write("\n")

                if successes:
                    n_sample = max(1, int(len(successes) * sample_rate))
                    sampled = rng.sample(successes, min(n_sample, len(successes)))
                    f.write(f"**Successes** (showing {len(sampled)}/{len(successes)}):\n\n")
                    f.write("| Input | Predicted |\n")
                    f.write("|-------|-----------|\n")
                    for r in sampled:
                        f.write(f"| `{r['seq']}` | `{r['predicted']}` |\n")
                    f.write("\n")

        # Generation examples
        f.write(f"---\n\n## Generation Examples\n\n")
        f.write("| Prompt | Output |\n")
        f.write("|--------|--------|\n")
        for prompt, output in model_results.get("generations", []):
            f.write(f"| `{prompt}` | `{output}` |\n")
        f.write("\n")

    print(f"Results written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train Compositor")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_hidden", type=int, default=256)
    parser.add_argument("--n_graph_nodes", type=int, default=64)
    parser.add_argument("--n_relations", type=int, default=8)
    parser.add_argument("--n_compose_hops", type=int, default=2,
                        help="(legacy) Only used by inspect_graph/structural_ops")
    parser.add_argument("--max_steps", type=int, default=5,
                        help="Number of loop iterations per block (Phase 6)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--inspect", action="store_true",
                        help="Run graph inspection after training")
    parser.add_argument("--oracle", action="store_true",
                        help="Use oracle graph (perfect knowledge) instead of data seeding")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # --- Data ---
    train_seqs, test_seqs = make_train_test_split()

    by_task_train = {}
    for s in train_seqs:
        by_task_train.setdefault(classify_task(s), []).append(s)
    by_task_test = {}
    for s in test_seqs:
        by_task_test.setdefault(classify_task(s), []).append(s)

    max_len = max(len(encode(s)) for s in train_seqs + test_seqs)

    print(f"Device: {args.device} | Vocab: {VOCAB_SIZE} | MaxLen: {max_len}")
    print(f"Train: {len(train_seqs)} ({', '.join(f'{len(v)} {k}' for k,v in sorted(by_task_train.items()))})")
    print(f"Test:  {len(test_seqs)} ({', '.join(f'{len(v)} {k}' for k,v in sorted(by_task_test.items()))})")

    # --- Model ---
    model = Compositor(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_hidden=args.d_hidden,
        n_graph_nodes=args.n_graph_nodes,
        n_relations=args.n_relations,
        n_compose_hops=args.n_compose_hops,
        max_steps=args.max_steps,
    ).to(args.device)
    print(f"Params: {model.count_parameters():,}")

    # --- Seed graph ---
    if args.oracle:
        print("\nSeeding graph with ORACLE knowledge...")
        seed_oracle_graph(model.graph, verbose=True)
    else:
        print("\nSeeding graph from training data...")
        seed_graph_from_data(model.graph, train_seqs, verbose=True)

    stats = graph_stats(model.graph)
    print(f"  After seeding: {stats['n_above_0.5']} strong edges, "
          f"{stats['n_above_0.1']} edges > 0.1, max={stats['max_edge']:.3f}")

    # Run initial composition closure
    from structural_ops import composition_closure
    n_closed = composition_closure(model.graph, threshold=0.3, verbose=True)
    stats = graph_stats(model.graph)
    print(f"  After closure: {stats['n_above_0.5']} strong edges, "
          f"{stats['n_above_0.1']} edges > 0.1, max={stats['max_edge']:.3f}")

    if not args.no_compile:
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: failed ({e}), continuing without")

    # --- Training ---
    # Separate param groups: zero weight decay on graph adjacency A
    # AdamW weight decay pushes A toward 0, which means sigmoid(A) toward 0.5
    # -- this DENSIFIES the graph, fighting the sparse init. Disable it for A.
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if hasattr(raw_model, "get_graph_params"):
        graph_params = raw_model.get_graph_params()
        graph_param_ids = {id(p) for p in graph_params}
        other_params = [p for p in model.parameters()
                        if p.requires_grad and id(p) not in graph_param_ids]
        optimizer = torch.optim.AdamW([
            {"params": other_params, "weight_decay": 0.01},
            {"params": graph_params, "weight_decay": 0.0},
        ], lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    best_test_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model, train_seqs, optimizer, args.device,
            batch_size=args.batch_size, max_len=max_len,
        )
        scheduler.step()

        # Periodic structural operations (Phase 4)
        raw_m = model._orig_mod if hasattr(model, "_orig_mod") else model
        run_structural_ops(raw_m.graph, epoch, interval=20,
                           threshold=0.3, verbose=(epoch % args.eval_every == 0))

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            train_acc, train_loss = evaluate(model, train_seqs, args.device, max_len=max_len)
            test_acc, test_loss = evaluate(model, test_seqs, args.device, max_len=max_len)
            elapsed = time.time() - t0

            marker = ""
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                marker = " *"

            print(
                f"  E{epoch:4d} | "
                f"trn {train_loss:.3f}/{train_acc:.3f} | "
                f"tst {test_loss:.3f}/{test_acc:.3f} | "
                f"{elapsed:.1f}s{marker}"
            )

            if epoch % (args.eval_every * 5) == 0 or epoch == args.epochs:
                task_results = evaluate_by_task(model, test_seqs, args.device, max_len=max_len)
                for task, res in sorted(task_results.items()):
                    print(f"    {task:15s}: {res['acc']:.3f} (n={res['n']})")

    print(f"\nBest test acc: {best_test_acc:.3f} | Total: {time.time()-t0:.1f}s")

    # --- Detailed eval + log ---
    # Unwrap compiled model for generation (compile doesn't support dynamic shapes well)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    train_detailed = detailed_eval(raw_model, train_seqs, args.device, max_len=max_len)
    test_detailed = detailed_eval(raw_model, test_seqs, args.device, max_len=max_len)

    gen_examples = [
        ("16,17,18,", 8), ("7+5=", 4), ("20>", 4), ("3+8=", 4),
        ("25,26,27,", 8), ("45,46,", 10), ("12+19=", 4), ("50>", 4),
    ]
    generations = []
    for prompt, max_new in gen_examples:
        result = generate_completion(raw_model, prompt, max_new, args.device)
        generations.append((prompt, result))
        print(f"  {prompt!r:20s} -> {result!r}")

    write_results_log(
        {
            "n_params": raw_model.count_parameters(),
            "best_test_acc": best_test_acc,
            "train_results": train_detailed,
            "test_results": test_detailed,
            "generations": generations,
        },
        train_seqs, test_seqs, args,
    )

    # --- Graph inspection ---
    if args.inspect:
        from inspect_graph import inspect_graph
        save_dir = os.path.join(SCRIPT_DIR, "inspect_output")
        report = inspect_graph(raw_model, save_dir=save_dir)
        print(report)

        # Also save the text report
        report_path = os.path.join(save_dir, "report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Inspection report saved to {report_path}")


if __name__ == "__main__":
    main()
