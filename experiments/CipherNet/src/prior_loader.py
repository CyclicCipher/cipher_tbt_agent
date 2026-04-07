"""Prior loader — builds a graph from JSON subgraph specifications.

Priors are DATA, not scripts. Each prior is a JSON file describing
nodes, edges, and properties. The loader reads them and creates the
corresponding structure in the graph.

No Python logic in the priors. All behavior emerges from the graph
structure — edge weights, inhibitory connections, layer assignments.
"""
from __future__ import annotations

import json
import os

try:
    from .graph import Graph, SPATIAL, TEMPORAL, BINDING, GATE
except ImportError:
    from graph import Graph, SPATIAL, TEMPORAL, BINDING, GATE

_CIPHER_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

EDGE_TYPES = {"spatial": SPATIAL, "temporal": TEMPORAL, "binding": BINDING, "gate": GATE}


def load_subgraph_from_json(graph: Graph, json_path: str) -> dict[str, int]:
    """Load a subgraph from a JSON file into the graph.

    Returns {node_id_in_json: node_id_in_graph}.
    """
    with open(json_path, "r") as f:
        spec = json.load(f)

    sg_name = spec["name"]
    graph.create_subgraph(sg_name)

    # Create nodes.
    id_map: dict[str, int] = {}
    for node_spec in spec.get("nodes", []):
        if "id" not in node_spec:
            continue  # skip _comment entries
        # Pass ALL fields from the JSON spec as metadata.
        meta = {k: v for k, v in node_spec.items() if k != "id"}
        nid = graph.add_node(
            label=f"{sg_name}:{node_spec['id']}",
            subgraph=sg_name,
            **meta,
        )
        id_map[node_spec["id"]] = nid

    # Create edges.
    for edge_spec in spec.get("edges", []):
        if "source" not in edge_spec:
            continue  # skip _comment entries
        src = id_map[edge_spec["source"]]
        tgt = id_map[edge_spec["target"]]
        etype = EDGE_TYPES.get(edge_spec.get("type", "spatial"), SPATIAL)
        weight = edge_spec.get("weight", 1.0)
        graph.add_edge(src, tgt, edge_type=etype, weight=weight,
                       description=edge_spec.get("description"))

    return id_map


def load_priors(config_path: str | None = None) -> tuple[Graph, dict[str, dict[str, int]]]:
    """Create a new graph with all configured priors loaded.

    Returns (graph, prior_nodes) where prior_nodes maps
    prior_name -> {json_node_id: graph_node_id}.
    """
    if config_path is None:
        config_path = os.path.join(_CIPHER_ROOT, "priors", "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    graph = Graph()
    prior_nodes: dict[str, dict[str, int]] = {}

    for prior_def in config.get("priors", []):
        name = prior_def["name"]
        json_file = prior_def["file"]
        json_path = os.path.join(_CIPHER_ROOT, json_file)

        id_map = load_subgraph_from_json(graph, json_path)
        prior_nodes[name] = id_map

    # Create connections between priors.
    for conn in config.get("connections", []):
        # Skip comment-only entries.
        if "_comment" in conn and "source_prior" not in conn:
            continue

        src_prior = conn["source_prior"]
        src_node = conn["source_node"]
        tgt_prior = conn["target_prior"]
        tgt_node = conn["target_node"]
        etype = EDGE_TYPES.get(conn.get("type", "spatial"), SPATIAL)
        weight = conn.get("weight", 1.0)

        src_id = prior_nodes[src_prior][src_node]
        tgt_id = prior_nodes[tgt_prior][tgt_node]
        graph.add_edge(src_id, tgt_id, edge_type=etype, weight=weight)

    # Set tonic activations for intrinsically active neurons.
    # GPi neurons fire tonically without input — they inhibit thalamus
    # by default. Gates are CLOSED until Go pathway actively opens them.
    if 'basal_ganglia' in prior_nodes:
        for node_key, node_id in prior_nodes['basal_ganglia'].items():
            node = graph.get_node(node_id)
            if node and node.meta.get('role') == 'gpi':
                node.activation = 0.8  # tonic firing

    return graph, prior_nodes


if __name__ == "__main__":
    graph, prior_nodes = load_priors()
    print(f"Graph: {graph.summary()}")
    for name, nodes in prior_nodes.items():
        print(f"\nPrior '{name}':")
        for json_id, graph_id in nodes.items():
            node = graph.get_node(graph_id)
            print(f"  {json_id} -> node {graph_id} "
                  f"(label={node.label}, layer={node.meta.get('layer')}, role={node.meta.get('role')})")

        # Show edges.
        for json_id, graph_id in nodes.items():
            for edge in graph.edges_from(graph_id):
                # Find the target's json id.
                tgt_json = next((jid for jid, gid in nodes.items() if gid == edge.target), f"node_{edge.target}")
                etype = {SPATIAL: "spatial", TEMPORAL: "temporal", BINDING: "binding"}.get(edge.edge_type, "?")
                print(f"  {json_id} --{etype}({edge.weight:.1f})--> {tgt_json}")
