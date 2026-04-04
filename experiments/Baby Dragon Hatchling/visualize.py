"""3D Brain Visualization for BDH.

Loads the most recent checkpoint, runs a forward pass on a sample input,
captures per-layer neuron activations, and renders an interactive 3D
visualization in the browser.

What you see:
- Each dot is a neuron (N neurons per head, across all heads)
- Position: PCA of encoder weight columns (neurons with similar encoding are nearby)
- Color: activation level (blue=inactive, red=highly active)
- Size: proportional to activation magnitude
- Use the layer slider to see how activations evolve through layers
- The ~3-5% active y_sparse neurons are highlighted with larger markers

Usage:
    python experiments/Baby\ Dragon\ Hatchling/visualize.py
    python experiments/Baby\ Dragon\ Hatchling/visualize.py --input "some text"
    python experiments/Baby\ Dragon\ Hatchling/visualize.py --layer 3
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

sys.path.insert(0, _SCRIPT_DIR)
from bdh import BDH, BDHConfig

CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "visualizations")


def load_model(checkpoint_path: str | None = None) -> tuple[BDH, dict]:
    """Load model from checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            "Run train.py first to create a checkpoint."
        )

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    config = ckpt["config"]
    model = BDH(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def get_neuron_positions(model: BDH) -> np.ndarray:
    """Compute 3D positions for each neuron via PCA of encoder weights.

    The encoder has shape (n_head, D, N). We treat each neuron as a
    D-dimensional vector (its column in the encoder matrix). PCA reduces
    to 3D. All heads are combined into one visualization.

    Returns: (total_neurons, 3) array of positions.
    """
    # encoder: (n_head, D, N) → reshape to (n_head*N, D)
    enc = model.encoder.detach().cpu().numpy()  # (nh, D, N)
    nh, D, N = enc.shape
    # Each neuron is a column: transpose to (nh, N, D) then reshape
    neurons = enc.transpose(0, 2, 1).reshape(nh * N, D)  # (nh*N, D)

    # PCA to 3D.
    neurons_centered = neurons - neurons.mean(axis=0)
    # SVD for PCA (more numerically stable than covariance).
    U, S, Vt = np.linalg.svd(neurons_centered, full_matrices=False)
    positions = U[:, :3] * S[:3]  # project onto top 3 components

    return positions


def get_activations(model: BDH, text: str, device: str = "cpu") -> list[dict]:
    """Run forward pass and return per-layer activations."""
    tokens = list(text.encode("utf-8"))
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        _, layer_activations = model.forward_with_activations(idx)

    return layer_activations


def _compute_edges(xy_act, positions, N, nh, top_k=50):
    """Compute top-K co-firing edges for visualization.

    xy_act: (nh, T, N) tensor — the gated activation per neuron per token.
    positions: (nh*N, 3) array — neuron 3D positions.
    Returns lists of edge coordinates for Scatter3d line traces.
    """
    # Reshape to (nh*N, T) — each neuron's activation profile across tokens.
    T = xy_act.shape[1]
    profiles = xy_act.reshape(nh * N, T)  # (total_neurons, T)

    # Only consider neurons that actually fire.
    active_mask = profiles.sum(axis=1) > 0
    active_idx = np.where(active_mask)[0]

    if len(active_idx) < 2:
        return [], [], [], []

    # Subsample if too many active neurons (keep computation tractable).
    if len(active_idx) > 500:
        rng = np.random.RandomState(0)
        active_idx = rng.choice(active_idx, 500, replace=False)

    active_profiles = profiles[active_idx]  # (n_active, T)

    # Normalize for cosine similarity.
    norms = np.linalg.norm(active_profiles, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = active_profiles / norms

    # Cosine similarity matrix (only upper triangle).
    sim = normed @ normed.T  # (n_active, n_active)
    np.fill_diagonal(sim, 0)

    # Find top-K edges by similarity.
    # Flatten upper triangle.
    n_active = len(active_idx)
    triu_i, triu_j = np.triu_indices(n_active, k=1)
    triu_vals = sim[triu_i, triu_j]

    if len(triu_vals) == 0:
        return [], [], [], []

    k = min(top_k, len(triu_vals))
    top_indices = np.argpartition(triu_vals, -k)[-k:]
    top_indices = top_indices[triu_vals[top_indices] > 0.1]  # threshold

    # Build edge coordinate lists for plotly (None-separated segments).
    ex, ey, ez, weights = [], [], [], []
    for idx in top_indices:
        i_local, j_local = triu_i[idx], triu_j[idx]
        i_global = active_idx[i_local]
        j_global = active_idx[j_local]
        w = float(triu_vals[idx])

        ex.extend([positions[i_global, 0], positions[j_global, 0], None])
        ey.extend([positions[i_global, 1], positions[j_global, 1], None])
        ez.extend([positions[i_global, 2], positions[j_global, 2], None])
        weights.append(w)

    return ex, ey, ez, weights


def _batch_top_connections(xy_act_np, N, nh, top_k=5):
    """Batch-compute top-K co-firing partners for ALL active neurons at once.

    Returns dict: neuron_idx → list of (partner_idx, similarity).
    Only computes for active neurons. O(n_active^2) not O(total^2).
    """
    T = xy_act_np.shape[1]
    total = nh * N
    profiles = xy_act_np.reshape(total, T)

    # Only consider neurons that fire.
    norms = np.linalg.norm(profiles, axis=1)
    active_mask = norms > 1e-8
    active_idx = np.where(active_mask)[0]

    result: dict[int, list[tuple[int, float]]] = {}
    if len(active_idx) < 2:
        return result

    # Subsample if too many (keep it fast).
    if len(active_idx) > 1000:
        rng = np.random.RandomState(0)
        active_idx = rng.choice(active_idx, 1000, replace=False)

    active_profiles = profiles[active_idx]
    active_norms = norms[active_idx]
    normed = active_profiles / active_norms[:, None]

    # Full cosine similarity matrix for active neurons.
    sim = normed @ normed.T  # (n_active, n_active)
    np.fill_diagonal(sim, 0)

    # For each active neuron, find top-K partners.
    for local_i in range(len(active_idx)):
        global_i = int(active_idx[local_i])
        row = sim[local_i]
        top_local = np.argsort(row)[-top_k:][::-1]
        partners = []
        for local_j in top_local:
            if row[local_j] > 0.01:
                partners.append((int(active_idx[local_j]), float(row[local_j])))
        if partners:
            result[global_i] = partners

    return result


def build_figure(
    positions: np.ndarray,
    layer_activations: list[dict],
    model_config: BDHConfig,
    input_text: str,
    step: int,
):
    """Build a plotly 3D figure with layer animation, edges, and weight inspection."""
    import plotly.graph_objects as go

    nh = model_config.n_head
    N = model_config.mlp_internal_dim_multiplier * model_config.n_embd // nh
    n_layers = len(layer_activations)

    # Precompute per-layer data.
    frames = []
    for layer_idx, acts in enumerate(layer_activations):
        x_act = acts['x_sparse'][0].numpy()  # (nh, T, N)
        y_act = acts['y_sparse'][0].numpy()
        xy_act = acts['xy_sparse'][0].numpy()

        x_mean = x_act.mean(axis=1).reshape(-1)   # (nh*N,)
        y_mean = y_act.mean(axis=1).reshape(-1)
        xy_mean = xy_act.mean(axis=1).reshape(-1)

        frames.append({
            'x_mean': x_mean,
            'y_mean': y_mean,
            'xy_mean': xy_mean,
            'xy_act': xy_act,  # keep full (nh, T, N) for edge computation
        })

    print("  Computing edges and connections per layer...")

    # Build animation frames (each frame has neurons trace + edges trace).
    fig_frames = []
    for layer_idx, frame_data in enumerate(frames):
        act = frame_data['xy_mean']
        y_act = frame_data['y_mean']

        act_max = max(act.max(), 1e-6)
        act_norm = act / act_max

        sizes = 2 + 6 * act_norm

        y_threshold = np.percentile(y_act[y_act > 0], 90) if (y_act > 0).any() else 0.01
        y_active = y_act > y_threshold
        sizes[y_active] = np.maximum(sizes[y_active], 10)

        colors = act_norm

        x_sparsity = (frame_data['x_mean'] > 0).sum() / len(frame_data['x_mean'])
        y_sparsity = (frame_data['y_mean'] > 0).sum() / len(frame_data['y_mean'])

        # Compute edges (top-K co-firing connections).
        ex, ey, ez, edge_weights = _compute_edges(
            frame_data['xy_act'], positions, N, nh, top_k=50,
        )
        n_edges = len(edge_weights)

        # Batch-compute top-5 connections for all active neurons.
        connections = _batch_top_connections(frame_data['xy_act'], N, nh, top_k=5)

        # Build hover text with connection details.
        hover_text = []
        for i in range(len(act)):
            head = i // N
            idx_in_head = i % N
            conn_text = ""
            if i in connections:
                for j, sim in connections[i]:
                    conn_text += f"  -> neuron {j} (h{j//N}.{j%N}) sim={sim:.3f}<br>"
            else:
                conn_text = "  (no strong connections)<br>"
            hover_text.append(
                f"<b>Neuron {i}</b> (head {head}, idx {idx_in_head})<br>"
                f"x_sparse: {frame_data['x_mean'][i]:.4f}<br>"
                f"y_sparse: {frame_data['y_mean'][i]:.4f}<br>"
                f"xy_gate: {act[i]:.4f}<br>"
                f"<br><b>Top connections:</b><br>"
                f"{conn_text}"
            )

        # Neuron markers trace.
        neuron_trace = go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='RdBu_r',
                cmin=0, cmax=1,
                colorbar=dict(title="Activation", x=0.92),
                opacity=0.7,
            ),
            text=hover_text,
            hoverinfo='text',
            name='neurons',
        )

        # Edge lines trace.
        if ex:
            mean_w = sum(edge_weights) / len(edge_weights) if edge_weights else 0
            edge_trace = go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode='lines',
                line=dict(
                    color='rgba(255, 100, 0, 0.4)',
                    width=2,
                ),
                hoverinfo='skip',
                name=f'edges ({n_edges})',
            )
            frame_traces = [neuron_trace, edge_trace]
        else:
            frame_traces = [neuron_trace]

        fig_frames.append(go.Frame(
            data=frame_traces,
            name=f"Layer {layer_idx}",
            layout=go.Layout(
                title=f"BDH Brain — Layer {layer_idx}/{n_layers-1} | "
                      f"x sparsity: {x_sparsity:.1%} | y sparsity: {y_sparsity:.1%} | "
                      f"{n_edges} edges"
            ),
        ))

    initial = fig_frames[0] if fig_frames else None

    fig = go.Figure(
        data=initial.data if initial else [],
        layout=go.Layout(
            title=f"BDH Brain Visualization (step {step}) — Input: \"{input_text[:50]}\"",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
                aspectmode='cube',
                domain=dict(x=[0, 0.9], y=[0.25, 1.0]),
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0.18,
                x=0.02,
                xanchor="left",
                yanchor="bottom",
                buttons=[
                    dict(label="  Play  ",
                         method="animate",
                         args=[None, {"frame": {"duration": 1000, "redraw": True},
                                      "fromcurrent": True}]),
                    dict(label=" Pause ",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                active=0,
                steps=[dict(
                    args=[[f"Layer {i}"], {"frame": {"duration": 0, "redraw": True},
                                           "mode": "immediate"}],
                    label=f"L{i}",
                    method="animate",
                ) for i in range(n_layers)],
                x=0.15, len=0.75,
                xanchor="left",
                y=0.2,
                currentvalue=dict(prefix="Layer: "),
            )],
            margin=dict(l=20, r=20, t=60, b=20),
            annotations=[dict(
                text=(
                    "<b>x sparsity</b> = % of neurons with excitatory activation > 0 (input encoding).  "
                    "<b>y sparsity</b> = % with output activation > 0 (the truly firing neurons).<br>"
                    "<b>Edges</b> = top-50 co-firing connections (cosine similarity of activation profiles across tokens).  "
                    "<b>Hover</b> a neuron to see its top-5 connections and weights."
                ),
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.02,
                xanchor="center", yanchor="top",
                font=dict(size=11, color="#555"),
                align="center",
            )],
        ),
        frames=fig_frames,
    )

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input", type=str,
                        default="<sudoku>5.3..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79<solve>")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("Loading model...")
    model, ckpt = load_model(args.checkpoint)
    step = ckpt.get("step", 0)
    config = ckpt["config"]
    N = config.mlp_internal_dim_multiplier * config.n_embd // config.n_head
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Step {step}, {n_params:,} params, N={N} neurons/head")

    print("Computing neuron positions (PCA of encoder)...")
    positions = get_neuron_positions(model)
    print(f"  {positions.shape[0]} neurons in 3D space")

    print(f"Running forward pass on: \"{args.input[:60]}\"...")
    activations = get_activations(model, args.input)
    print(f"  {len(activations)} layers captured")

    print("Building visualization...")
    fig = build_figure(positions, activations, config, args.input, step)

    # Save to HTML.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(OUTPUT_DIR, "brain.html")

    # Write HTML with full-viewport sizing (no scrollbar).
    html_header = (
        '<html><head><style>'
        'html, body { margin: 0; padding: 0; overflow: hidden; width: 100%; height: 100%; }'
        '</style></head><body>'
    )
    html_footer = '</body></html>'
    fig.write_html(
        out_path,
        auto_open=True,
        full_html=False,
        default_width="100%",
        default_height="100vh",
    )
    # Wrap the plotly div in a full-viewport HTML shell.
    with open(out_path, "r", encoding="utf-8") as f:
        plotly_div = f.read()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_header + plotly_div + html_footer)
    print(f"Visualization saved to {out_path}")


if __name__ == "__main__":
    main()
