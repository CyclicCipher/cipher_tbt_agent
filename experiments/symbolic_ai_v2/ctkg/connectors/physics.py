"""
Physics Connector — bridges physics_streams.py data to the v2 AgenticLoop.

Converts PhysicsStream observation sets (list of (dict[str, float], float)
tuples) into Graph instances that AgenticLoop.observe_named() can
consume directly.

Usage
-----
    from experiments.symbolic_ai_v2.ctkg.connectors.physics import (
        stream_to_graphs, feed_stream_to_loop,
    )
    from experiments.symbolic_ai_v2.ctkg.einstein.physics_streams import (
        newtonian_mechanics_stream,
    )
    from experiments.symbolic_ai_v2.ctkg.logic.AgenticLoop import AgenticLoop
    from experiments.symbolic_ai_v2.ctkg.logic.KnowledgeGraph import KnowledgeGraph

    loop = AgenticLoop(KnowledgeGraph())
    stream = newtonian_mechanics_stream()
    feed_stream_to_loop(loop, stream, stream_name="newtonian")
    loop.induct_named("newtonian")

Iron Law note
-------------
The physics quantity names ("force", "mass", "velocity", …) are role labels in
the Graph's fields dict.  They are NOT operator NodeIds and are NOT
subject to the Iron Law.  The Iron Law applies only to operator identities in
KG morphisms.
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.ctkg.logic.InputOutputTopology import Graph, sequence_graph
from experiments.symbolic_ai_v2.ctkg.einstein.physics_streams import PhysicsStream


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def obs_to_graph(obs: tuple[dict, float]) -> Graph:
    """
    Convert a single (fields_dict, output_float) physics observation to a Graph.

    Parameters
    ----------
    obs : (dict[str, float], float)
        A single observation from a PhysicsStream.observation_sets entry.

    Returns
    -------
    Graph with field nodes as inputs and an output node for the result.
    """
    fields, output = obs
    # Flatten fields (sorted alphabetically) + output into a sequence.
    sorted_vals = [v for _, v in sorted(fields.items())]
    return sequence_graph(sorted_vals + [output])


def stream_to_graphs(
    stream: PhysicsStream,
) -> list[list[Graph]]:
    """
    Convert a full PhysicsStream to nested lists of Graph instances.

    Returns a list of observation-set lists, mirroring the structure of
    stream.observation_sets.  Each outer list corresponds to one observation
    set; each inner element is one observation wrapped in a Graph.
    """
    return [
        [obs_to_graph(obs) for obs in obs_set]
        for obs_set in stream.observation_sets
    ]


def feed_stream_to_loop(
    loop: "AgenticLoop",
    stream: PhysicsStream,
    stream_name: str | None = None,
    obs_set_index: int | None = None,
) -> int:
    """
    Feed a PhysicsStream's observations into a named AgenticLoop stream.

    Parameters
    ----------
    loop         : The AgenticLoop (parent) to route observations through.
    stream       : PhysicsStream from physics_streams.py.
    stream_name  : Name for the child stream.  Defaults to stream.name.
    obs_set_index: If given, only feed the obs set at that index.
                   If None, feed all observation sets in order.

    Returns
    -------
    Total number of observations fed.
    """
    name = stream_name if stream_name is not None else stream.name
    obs_sets = (
        [stream.observation_sets[obs_set_index]]
        if obs_set_index is not None
        else stream.observation_sets
    )
    total = 0
    for obs_set in obs_sets:
        for obs in obs_set:
            loop.observe_named(name, obs_to_graph(obs))
            total += 1
    return total


def newtonian_predictions_for(
    stream: PhysicsStream,
    obs_set_index: int = 0,
) -> list[tuple[dict, float]]:
    """
    Return the Newtonian (ether) predictions from a PhysicsStream, if any.

    These are stored in stream.newtonian_predictions for streams where the
    classical prediction differs from the observation (e.g. Michelson-Morley).

    Returns an empty list if the stream has no Newtonian predictions.
    """
    if not stream.newtonian_predictions:
        return []
    if obs_set_index >= len(stream.newtonian_predictions):
        return []
    return stream.newtonian_predictions[obs_set_index]
