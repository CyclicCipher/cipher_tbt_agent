"""
Inspection tools for the Compositor's categorical graph (v0.3).

Dumps diagnostics about what the graph learned:
- Concept slot activity (which free nodes are used)
- Edge probability heatmaps (sigmoid-based, per relation type)
- Top-k morphisms per relation type
- Succession signal detection
- Composition effect (1-hop vs composed)
- Cosine similarity structure of node embeddings

Can be run standalone after training, or called from train.py with --inspect.
"""

import math
import os
import torch
import torch.nn.functional as F

from data import ALL_TOKENS, VOCAB_SIZE, ID_TO_TOKEN


def node_label(idx: int) -> str:
    """Human-readable label for a graph node."""
    if idx < VOCAB_SIZE:
        tok = ALL_TOKENS[idx]
        if tok == "\x00":
            return "PAD"
        elif tok == "\x01":
            return "BOS"
        elif tok == "\x02":
            return "EOS"
        elif tok == " ":
            return "SPC"
        else:
            return repr(tok)
    return f"C{idx - VOCAB_SIZE}"  # concept slot


def inspect_graph(model, save_dir: str = None):
    """Run all graph diagnostics. Returns a report string.

    Args:
        model: a Compositor model (unwrapped from torch.compile if needed)
        save_dir: if provided, save heatmap images here
    """
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    graph = model.graph
    lines = []
    lines.append("=" * 60)
    lines.append("GRAPH INSPECTION (v0.3)")
    lines.append("=" * 60)

    # --- 1. Node embedding norms ---
    lines.append("\n## Node Embedding Norms\n")
    norms = graph.nodes.data.norm(dim=1)  # (N,)

    lines.append("### Vocab tokens:")
    for i in range(VOCAB_SIZE):
        lines.append(f"  {node_label(i):6s}  norm={norms[i]:.4f}")

    lines.append("\n### Concept slots (sorted by norm):")
    concept_norms = [(i, norms[i].item()) for i in range(VOCAB_SIZE, graph.n_nodes)]
    concept_norms.sort(key=lambda x: -x[1])

    active_count = 0
    vocab_median_norm = norms[:VOCAB_SIZE].median().item()
    lines.append(f"  (vocab median norm: {vocab_median_norm:.4f})")

    for idx, norm in concept_norms:
        active = norm > vocab_median_norm
        if active:
            active_count += 1
        marker = " *ACTIVE*" if active else ""
        lines.append(f"  {node_label(idx):6s}  norm={norm:.4f}{marker}")

    lines.append(f"\n  Active concept slots: {active_count}/{graph.n_nodes - VOCAB_SIZE}")

    # --- 2. Edge probability analysis (v0.3: sigmoid-based) ---
    lines.append("\n## Edge Probability Analysis\n")

    with torch.no_grad():
        nodes = graph.nodes.data  # (N, D)
        # Edge probabilities (sigmoid of raw A, diagonal masked to 0)
        P = graph.get_edge_probs()  # (K, N, N)
        # Raw adjacency logits
        raw_A = graph.A.data  # (K, N, N)
        # Composed adjacency (P + P@P + ...)
        composed = graph.get_composed_adjacency()  # (K, N, N)

    lines.append("### Overall edge statistics:")
    lines.append(f"  Raw A (logits): range=[{raw_A.min():.3f}, {raw_A.max():.3f}], "
                 f"mean={raw_A.mean():.3f}, std={raw_A.std():.3f}")
    lines.append(f"  Edge probs P=sigmoid(A): range=[{P.min():.4f}, {P.max():.4f}], "
                 f"mean={P.mean():.4f}")
    lines.append(f"  Composed (P+P@P+...): range=[{composed.min():.4f}, {composed.max():.4f}], "
                 f"mean={composed.mean():.4f}")

    # Count edges above various thresholds
    for thresh in [0.1, 0.3, 0.5, 0.8]:
        n_above = (P > thresh).sum().item()
        total = P.numel()
        lines.append(f"  Edges with P > {thresh}: {n_above}/{total} "
                     f"({100*n_above/total:.1f}%)")
    lines.append("")

    # Per-relation analysis
    for k in range(graph.n_relations):
        P_k = P[k]  # (N, N)
        raw_k = raw_A[k]
        comp_k = composed[k]

        lines.append(f"### Relation {k}")
        lines.append(f"  Raw A range: [{raw_k.min():.3f}, {raw_k.max():.3f}], "
                     f"mean={raw_k.mean():.3f}")
        lines.append(f"  Edge prob range: [{P_k.min():.4f}, {P_k.max():.4f}], "
                     f"mean={P_k.mean():.4f}")

        n_strong = (P_k > 0.5).sum().item()
        lines.append(f"  Strong edges (P > 0.5): {n_strong}")

        # Top-10 strongest morphisms
        flat = P_k.flatten()
        top_vals, top_idxs = flat.topk(min(10, flat.numel()))

        lines.append(f"  Top-10 morphisms (by edge prob):")
        for val, idx in zip(top_vals, top_idxs):
            src_idx = idx.item() // graph.n_nodes
            tgt_idx = idx.item() % graph.n_nodes
            raw_val = raw_k[src_idx, tgt_idx].item()
            lines.append(
                f"    {node_label(src_idx):6s} -> {node_label(tgt_idx):6s}  "
                f"P={val.item():.4f} (logit={raw_val:.3f})"
            )

        # Succession signal
        digit_start = ALL_TOKENS.index("0")
        succ_probs = []
        for d in range(9):
            src = digit_start + d
            tgt = digit_start + d + 1
            succ_probs.append(P_k[src, tgt].item())
        avg_succ = sum(succ_probs) / len(succ_probs)

        non_succ = []
        for i in range(10):
            for j in range(10):
                if j != i + 1 and i != j:
                    non_succ.append(P_k[digit_start + i, digit_start + j].item())
        avg_non_succ = sum(non_succ) / max(len(non_succ), 1)

        ratio = avg_succ / max(avg_non_succ, 1e-6)
        lines.append(f"  Succession signal: avg_succ_P={avg_succ:.4f}, "
                     f"avg_other_P={avg_non_succ:.4f}, ratio={ratio:.2f}")
        lines.append("")

    # --- 3. Composition effect ---
    lines.append("\n## Composition Effect\n")
    lines.append("Comparing 1-hop P to composed (P+P@P) for digit succession:")

    digit_start = ALL_TOKENS.index("0")
    for k in range(min(3, graph.n_relations)):
        lines.append(f"\n### Relation {k}:")
        lines.append("  1-hop succession:")
        for d in range(9):
            src = digit_start + d
            tgt = digit_start + d + 1
            lines.append(f"    {d}->{d+1}: P={P[k, src, tgt]:.4f}, "
                         f"composed={composed[k, src, tgt]:.4f}")

        lines.append("  2-hop (skip-one, should strengthen from composition):")
        for d in range(8):
            src = digit_start + d
            tgt = digit_start + d + 2
            lines.append(f"    {d}->{d+2}: P={P[k, src, tgt]:.4f}, "
                         f"composed={composed[k, src, tgt]:.4f}")

    # --- 4. Morphism value norms ---
    lines.append("\n\n## Morphism Value Norms\n")
    if hasattr(graph, 'V'):
        V_norms = graph.V.data.norm(dim=-1)  # (K, N)
        for k in range(graph.n_relations):
            v_k = V_norms[k]
            lines.append(f"  Relation {k}: V norm range [{v_k.min():.3f}, {v_k.max():.3f}], "
                         f"mean={v_k.mean():.3f}")
    else:
        lines.append("  (V removed in v0.4 -- using node embeddings as values)")

    # --- 5. Cosine similarity structure ---
    lines.append("\n## Node Embedding Similarity\n")
    with torch.no_grad():
        normed = F.normalize(nodes, dim=1)
        cos_sim = normed @ normed.T

    lines.append("### Digit-digit cosine similarities:")
    digit_ids = [ALL_TOKENS.index(str(d)) for d in range(10)]
    header = "     " + " ".join(f"{d:6d}" for d in range(10))
    lines.append(header)
    for i, di in enumerate(digit_ids):
        row = f"  {i}  "
        for j, dj in enumerate(digit_ids):
            row += f"{cos_sim[di, dj].item():6.2f} "
        lines.append(row)

    lines.append("\n### Operator-token cosine similarities (avg with digits):")
    ops = ["+", "=", "<", ">", ","]
    for op in ops:
        op_id = ALL_TOKENS.index(op)
        avg_sim = sum(cos_sim[op_id, d].item() for d in digit_ids) / len(digit_ids)
        lines.append(f"  {repr(op):6s}  avg_cos_with_digits={avg_sim:.3f}")

    lines.append("\n### Concept slots closest to vocab tokens:")
    for c_idx in range(VOCAB_SIZE, graph.n_nodes):
        if norms[c_idx].item() < vocab_median_norm * 0.5:
            continue
        sims = cos_sim[c_idx, :VOCAB_SIZE]
        best_val, best_idx = sims.max(dim=0)
        lines.append(
            f"  {node_label(c_idx):6s} (norm={norms[c_idx]:.3f}) closest to "
            f"{node_label(best_idx.item()):6s} (cos={best_val.item():.3f})"
        )

    # --- 6. Save heatmaps ---
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _save_heatmaps(P, composed, cos_sim, norms, save_dir)
        lines.append(f"\nHeatmaps saved to {save_dir}/")

    report = "\n".join(lines)
    return report


def _save_heatmaps(P, composed, cos_sim, norms, save_dir):
    """Save edge probability and similarity heatmaps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K, N, _ = P.shape
        labels = [node_label(i) for i in range(N)]

        for k in range(K):
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))

            # 1-hop edge probabilities
            data_p = P[k].cpu().numpy()
            im0 = axes[0].imshow(data_p, cmap="YlOrRd", aspect="auto",
                                 vmin=0, vmax=1)
            axes[0].set_title(f"Relation {k} - Edge probs P=sigmoid(A)")
            axes[0].set_xticks(range(N))
            axes[0].set_yticks(range(N))
            axes[0].set_xticklabels(labels, rotation=90, fontsize=4)
            axes[0].set_yticklabels(labels, fontsize=4)
            plt.colorbar(im0, ax=axes[0])

            # Composed (P + P@P + ...)
            data_c = composed[k].cpu().numpy()
            im1 = axes[1].imshow(data_c, cmap="YlOrRd", aspect="auto")
            axes[1].set_title(f"Relation {k} - Composed (P + P@P)")
            axes[1].set_xticks(range(N))
            axes[1].set_yticks(range(N))
            axes[1].set_xticklabels(labels, rotation=90, fontsize=4)
            axes[1].set_yticklabels(labels, fontsize=4)
            plt.colorbar(im1, ax=axes[1])

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"relation_{k}.png"), dpi=150)
            plt.close()

        # Cosine similarity heatmap (vocab tokens only)
        V = len(ALL_TOKENS)
        fig, ax = plt.subplots(figsize=(8, 7))
        data = cos_sim[:V, :V].cpu().numpy()
        im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        vocab_labels = [node_label(i) for i in range(V)]
        ax.set_xticks(range(V))
        ax.set_yticks(range(V))
        ax.set_xticklabels(vocab_labels, rotation=90)
        ax.set_yticklabels(vocab_labels)
        ax.set_title("Vocab token cosine similarity")
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "vocab_cosine_sim.png"), dpi=150)
        plt.close()

        # Node norms bar chart
        fig, ax = plt.subplots(figsize=(14, 4))
        n_data = norms.cpu().numpy()
        colors = ["steelblue"] * V + ["coral"] * (len(n_data) - V)
        ax.bar(range(len(n_data)), n_data, color=colors)
        ax.set_xlabel("Node index")
        ax.set_ylabel("Embedding L2 norm")
        ax.set_title("Node embedding norms (blue=vocab, red=concept slots)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "node_norms.png"), dpi=150)
        plt.close()

        print(f"  Heatmap images saved to {save_dir}/")

    except ImportError:
        for k in range(P.shape[0]):
            path = os.path.join(save_dir, f"relation_{k}.csv")
            data = P[k].cpu().numpy()
            import numpy as np
            np.savetxt(path, data, delimiter=",", fmt="%.4f")
        print(f"  CSV data saved to {save_dir}/ (install matplotlib for images)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a trained Compositor graph")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to saved model checkpoint")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save heatmap images")
    args = parser.parse_args()

    if args.checkpoint:
        from model import Compositor
        model = Compositor(vocab_size=VOCAB_SIZE)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    else:
        from model import Compositor
        model = Compositor(vocab_size=VOCAB_SIZE)
        print("(Inspecting randomly initialized model -- pass --checkpoint for trained)")

    report = inspect_graph(model, save_dir=args.save_dir)
    print(report)
