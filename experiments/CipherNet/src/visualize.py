"""CipherNet Graph Visualizer — interactive 3D brain view in browser.

Renders the full graph with subgraphs color-coded. Nodes sized by
activation. Edges colored by type. Hover for details.

Usage:
    python visualize.py              # visualize a fresh brain
    python visualize.py --after-expr "3+4*2="  # visualize after processing
"""
from __future__ import annotations

import argparse
import os

try:
    from .graph import Graph, SPATIAL, TEMPORAL, BINDING, GATE
    from .prior_loader import load_priors
    from .brain import Brain
except ImportError:
    from graph import Graph, SPATIAL, TEMPORAL, BINDING, GATE
    from prior_loader import load_priors
    from brain import Brain

import plotly.graph_objects as go


# Subgraph color palette.
SUBGRAPH_COLORS = {
    "ans": "#2196F3",              # blue
    "pfc": "#FF9800",              # orange
    "basal_ganglia": "#F44336",    # red
    "thalamus": "#9C27B0",         # purple
    "output_cortex": "#4CAF50",    # green
    "broca": "#00BCD4",            # cyan
    "temporal_cortex": "#E91E63",  # pink
}
DEFAULT_COLOR = "#607D8B"       # grey for dynamic columns

EDGE_COLORS = {
    SPATIAL: "rgba(150, 150, 150, 0.3)",
    TEMPORAL: "rgba(50, 50, 200, 0.2)",
    BINDING: "rgba(50, 200, 50, 0.2)",
    GATE: "rgba(200, 50, 50, 0.3)",
}


def layout_graph(graph: Graph) -> dict[int, tuple[float, float, float]]:
    """Compute 3D positions for each node.

    Groups nodes by subgraph. Each subgraph gets a region in 3D space.
    Within a subgraph, nodes are arranged in a circle or grid.
    """
    import math

    positions: dict[int, tuple[float, float, float]] = {}
    subgraphs = graph.all_subgraphs()

    # Assign each subgraph a position on a circle in the XY plane.
    n_sg = max(len(subgraphs), 1)
    sg_positions: dict[str, tuple[float, float]] = {}
    for i, sg_name in enumerate(sorted(subgraphs)):
        angle = 2 * math.pi * i / n_sg
        radius = 5.0
        sg_positions[sg_name] = (radius * math.cos(angle), radius * math.sin(angle))

    # Place nodes within their subgraph.
    for sg_name in sorted(subgraphs):
        nodes = sorted(graph.nodes_in_subgraph(sg_name))
        cx, cy = sg_positions[sg_name]
        n = max(len(nodes), 1)

        for j, nid in enumerate(nodes):
            node = graph.get_node(nid)
            # Arrange in a small circle within the subgraph region.
            angle = 2 * math.pi * j / n
            r = min(1.5, 0.3 * math.sqrt(n))
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)

            # Z: layer determines height.
            layer = node.meta.get("layer", 0) if node else 0
            if layer == 4:
                z = -1.0
            elif layer == 23:
                z = 0.0
            elif layer == 5:
                z = 1.0
            elif layer == 6:
                z = -0.5
            else:
                z = 0.0

            positions[nid] = (x, y, z)

    # Handle nodes not in any subgraph.
    orphans = [nid for nid in range(graph.node_count())
               if nid not in positions and graph.get_node(nid) is not None]
    for i, nid in enumerate(orphans):
        positions[nid] = (8 + i * 0.3, 0, 0)

    return positions


def build_figure(graph: Graph, title: str = "CipherNet Brain") -> go.Figure:
    """Build a plotly 3D figure of the graph."""
    positions = layout_graph(graph)

    # Determine subgraph for each node.
    node_subgraph: dict[int, str] = {}
    for sg_name in graph.all_subgraphs():
        for nid in graph.nodes_in_subgraph(sg_name):
            node_subgraph[nid] = sg_name

    # Build node traces (one per subgraph for coloring).
    sg_traces: dict[str, dict] = {}
    for nid, (x, y, z) in positions.items():
        node = graph.get_node(nid)
        if node is None:
            continue

        sg = node_subgraph.get(nid, "other")
        if sg not in sg_traces:
            sg_traces[sg] = {"x": [], "y": [], "z": [], "text": [], "size": []}

        activation = node.activation
        label = node.label or f"node_{nid}"
        role = node.meta.get("role", "")
        layer = node.meta.get("layer", "")
        token = node.meta.get("token", "")

        error = node.error if hasattr(node, 'error') else 0.0

        hover = (f"<b>{label}</b><br>"
                 f"ID: {nid}<br>"
                 f"Subgraph: {sg}<br>"
                 f"Layer: {layer}<br>"
                 f"Role: {role}<br>"
                 f"Activation: {activation:.3f}<br>"
                 f"Error: {error:.3f}")
        if token:
            hover += f"<br>Token: '{token}'"

        sg_traces[sg]["x"].append(x)
        sg_traces[sg]["y"].append(y)
        sg_traces[sg]["z"].append(z)
        sg_traces[sg]["text"].append(hover)
        sg_traces[sg]["size"].append(4 + 12 * activation)

    traces = []
    for sg_name, data in sorted(sg_traces.items()):
        color = SUBGRAPH_COLORS.get(sg_name, DEFAULT_COLOR)
        # Dynamic columns get lighter shades.
        if sg_name.startswith("column:"):
            color = "#90A4AE"

        traces.append(go.Scatter3d(
            x=data["x"], y=data["y"], z=data["z"],
            mode="markers",
            marker=dict(
                size=data["size"],
                color=color,
                opacity=0.8,
            ),
            text=data["text"],
            hoverinfo="text",
            name=sg_name,
        ))

    # Build edge traces.
    edge_x, edge_y, edge_z = [], [], []
    for (src, tgt, etype), edge in graph._edges.items():
        if src in positions and tgt in positions:
            x0, y0, z0 = positions[src]
            x1, y1, z1 = positions[tgt]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

    traces.insert(0, go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="rgba(150, 150, 150, 0.15)", width=1),
        hoverinfo="skip",
        name="edges",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="",
            yaxis_title="",
            zaxis_title="Layer",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.01, y=0.99),
        width=1200,
        height=800,
    )

    return fig


def visualize_brain(brain: Brain, title: str = "CipherNet Brain",
                    output_path: str | None = None):
    """Visualize a Brain instance."""
    fig = build_figure(brain.graph, title)

    if output_path is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "brain.html")

    fig.write_html(output_path, auto_open=True)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--after-expr", type=str, default=None,
                        help="Visualize after processing this expression")
    args = parser.parse_args()

    brain = Brain(number_line_max=19)
    brain.teach_all_digits()

    title = "CipherNet Brain"

    if args.after_expr:
        result = brain.process(args.after_expr)
        title = f"CipherNet Brain after: {args.after_expr} = {result}"
        print(f"Processed: {args.after_expr} = {result}")

    print(f"Graph: {brain.graph.summary()['nodes']} nodes, "
          f"{brain.graph.summary()['edges']} edges")
    visualize_brain(brain, title)
