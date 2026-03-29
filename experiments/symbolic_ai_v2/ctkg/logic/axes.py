"""
Axis discovery — find the coordinate system from observation sequences.

Scans sequential observations for periodic patterns in token positions.
Each discovered period defines an axis of the space. The topology of
each axis (cyclic, with a specific period) emerges from the data.

Navigation: step along an axis. When the coordinate wraps past the
period, propagate (carry) to the next axis.

Decoding: read off the coordinate on each axis. If a coordinate
exceeds the period, recursively decompose it (same rule at every scale).

No domain knowledge. Discovers axes, periods, and topology from
observation order alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, NodeId,
)
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus


# ---------------------------------------------------------------------------
# Axis: a discovered periodic dimension
# ---------------------------------------------------------------------------

@dataclass
class Axis:
    """A discovered axis of the coordinate space."""
    period: int                      # how many steps before the token cycles
    token_sequence: list[NodeId]     # the tokens in cycle order (length = period)
    position_in_observation: int     # which position in the observation this axis reads from


@dataclass
class CoordinateSpace:
    """A multi-axis coordinate system discovered from data."""
    axes: list[Axis]                 # ordered: axis 0 = fastest-cycling (units)
    step_count: int = 0             # total observations used to build this


# ---------------------------------------------------------------------------
# Period detection from token sequences
# ---------------------------------------------------------------------------

def _detect_period(sequence: list[NodeId], min_period: int = 2, max_period: int = 50) -> int:
    """Find the smallest period p where sequence[i] == sequence[i+p] for most i.

    Returns the period, or 0 if no periodic structure found.
    """
    n = len(sequence)
    if n < min_period * 3:
        return 0  # not enough data to detect period

    for p in range(min_period, min(max_period, n // 2) + 1):
        max_check = min(n - p, n // 2)
        if max_check < min_period * 2:
            continue
        matches = sum(1 for i in range(max_check) if sequence[i] == sequence[i + p])
        if matches / max_check >= 0.9:
            return p

    return 0


def _extract_cycle_tokens(sequence: list[NodeId], period: int) -> list[NodeId]:
    """Extract the token order for one cycle of the given period.

    Takes the majority token at each phase position across all cycles.
    """
    from collections import Counter
    tokens_at_phase: list[Counter] = [Counter() for _ in range(period)]

    for i, nid in enumerate(sequence):
        phase = i % period
        tokens_at_phase[phase][nid] += 1

    return [c.most_common(1)[0][0] for c in tokens_at_phase]


# ---------------------------------------------------------------------------
# Axis discovery from observations
# ---------------------------------------------------------------------------

def discover_axes(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
    max_axes: int = 5,
) -> CoordinateSpace:
    """Discover the coordinate axes from observation sequences.

    For each token position in the observations (relative to a fixed
    reference point), extract the sequence of tokens across consecutive
    observations and check for periodicity.

    The fastest-cycling position becomes axis 0 (units). Slower-cycling
    positions become higher axes (tens, hundreds, etc.).

    Returns CoordinateSpace with discovered axes.
    """
    obs_list = hippo.all_observations()
    recent = obs_list[max(0, since_index):]
    if len(recent) < 10:
        return CoordinateSpace(axes=[], step_count=0)

    # Find the maximum observation length.
    max_len = max(len(obs.token_nids) for obs in recent)

    # Find a separator token: a token that appears in most observations at
    # a consistent position, splitting each observation into two halves.
    # (e.g., 'next_is' in counting observations [3, next_is, 4]).
    # If no separator found, use the full observation.
    from collections import Counter

    sep_counter: Counter = Counter()
    for obs in recent:
        for nid in set(obs.token_nids):
            sep_counter[nid] += 1

    # Separator candidate: appears in >80% of observations.
    n_obs = len(recent)
    separator_nid = None
    for nid, count in sep_counter.most_common():
        if count < n_obs * 0.8:
            continue
        # Check: does it appear at a consistent position?
        positions = [obs.token_nids.index(nid) for obs in recent
                     if nid in obs.token_nids]
        if not positions:
            continue
        # Must not be the first or last token in most observations.
        mid_count = 0
        for obs_i, obs in enumerate(recent):
            if nid not in obs.token_nids:
                continue
            p = obs.token_nids.index(nid)
            if 0 < p < len(obs.token_nids) - 1:
                mid_count += 1
        if mid_count > count * 0.5:
            separator_nid = nid
            break

    # Extract the INPUT side (tokens before separator) for period detection.
    # The input side encodes what number we're operating on. Its periodic
    # structure defines the coordinate system.
    input_sequences_by_pos: dict[int, list[NodeId]] = {}

    for obs in recent:
        nids = obs.token_nids
        if separator_nid is not None and separator_nid in nids:
            sep_pos = nids.index(separator_nid)
            input_tokens = nids[:sep_pos]
        else:
            input_tokens = nids

        if not input_tokens:
            continue

        # Collect by position from end of the input part.
        for pos_from_end in range(1, len(input_tokens) + 1):
            idx = len(input_tokens) - pos_from_end
            input_sequences_by_pos.setdefault(pos_from_end, []).append(
                input_tokens[idx]
            )

    # Detect periodicity at each position.
    candidates: list[tuple[int, int, list[NodeId]]] = []

    for pos_from_end, sequence in input_sequences_by_pos.items():
        if len(sequence) < 10:
            continue
        period = _detect_period(sequence)
        if period > 0:
            cycle_tokens = _extract_cycle_tokens(sequence, period)

            candidates.append((period, pos_from_end, cycle_tokens))

    if not candidates:
        return CoordinateSpace(axes=[], step_count=len(recent))

    # Sort by period (smallest first = fastest cycling = innermost axis).
    candidates.sort(key=lambda x: x[0])

    # Axis 0: the fastest-cycling position.
    axis0_period, axis0_pos, axis0_cycle = candidates[0]
    axes: list[Axis] = [Axis(
        period=axis0_period,
        token_sequence=axis0_cycle,
        position_in_observation=axis0_pos,
    )]

    # Higher axes: discovered from WRAP EVENTS of axis 0.
    # When axis 0 wraps (last cycle token → first cycle token), the next
    # position from the end increments. Extract that token sequence at
    # each wrap event and check if it uses the same cycle.
    #
    # This recursion handles arbitrary depth: axis 1 wraps trigger axis 2, etc.

    # First, rebuild the input token sequences to detect wraps.
    input_seqs: list[list[NodeId]] = []
    for obs in recent:
        nids = obs.token_nids
        if separator_nid is not None and separator_nid in nids:
            sep_pos = nids.index(separator_nid)
            input_seqs.append(nids[:sep_pos])
        else:
            input_seqs.append(list(nids))

    # Find wrap events for the current innermost axis.
    current_axis = 0
    while len(axes) < max_axes:
        cycle = axes[current_axis].token_sequence
        pos_fe = axes[current_axis].position_in_observation
        first_token = cycle[0]
        last_token = cycle[-1]

        # Detect wrap events: where this axis goes from last→first.
        wrap_steps: list[int] = []
        for i in range(1, len(input_seqs)):
            prev = input_seqs[i - 1]
            curr = input_seqs[i]
            prev_idx = len(prev) - pos_fe
            curr_idx = len(curr) - pos_fe
            if prev_idx < 0 or curr_idx < 0:
                continue
            if prev_idx < len(prev) and curr_idx < len(curr):
                if prev[prev_idx] == last_token and curr[curr_idx] == first_token:
                    wrap_steps.append(i)

        if len(wrap_steps) < 2:
            break  # not enough wraps to detect the next axis

        # Extract the token at next-position-from-end at each wrap event.
        next_pos_fe = pos_fe + 1
        wrap_tokens: list[NodeId] = []
        for step in wrap_steps:
            inp = input_seqs[step]
            idx = len(inp) - next_pos_fe
            if idx >= 0 and idx < len(inp):
                wrap_tokens.append(inp[idx])

        if len(wrap_tokens) < 2:
            break

        # Check: does this use the same cycle as axis 0?
        # (All axes use the same digit vocabulary in base-N systems.)
        # Verify by checking if the wrap tokens follow the same cycle.
        matches_cycle = all(
            wrap_tokens[i] in cycle for i in range(len(wrap_tokens))
        )

        if matches_cycle:
            axes.append(Axis(
                period=axis0_period,
                token_sequence=list(cycle),  # same cycle
                position_in_observation=next_pos_fe,
            ))
            current_axis += 1
        else:
            break

    return CoordinateSpace(axes=axes, step_count=len(recent))


# ---------------------------------------------------------------------------
# Encode: token sequence → coordinates
# ---------------------------------------------------------------------------

def encode(space: CoordinateSpace, token_nids: list[NodeId]) -> list[int] | None:
    """Map a token sequence to coordinates in the discovered space.

    For each digit position in the input (from the end), look up which
    token is there and find its phase in the cycle. If the input has
    more digit positions than discovered axes, use axis 0's cycle for
    the extra positions (self-similar: every digit position uses the
    same vocabulary).

    Returns list of coordinates [axis0_coord, axis1_coord, ...] or None
    if no axes discovered.
    """
    if not space.axes:
        return None

    # Use axis 0's cycle as the universal digit cycle.
    cycle = space.axes[0].token_sequence

    coords: list[int] = []
    n_digits = len(token_nids)

    for pos_from_end in range(1, n_digits + 1):
        idx = n_digits - pos_from_end
        token = token_nids[idx]
        try:
            phase = cycle.index(token)
        except ValueError:
            phase = 0
        coords.append(phase)

    return coords


# ---------------------------------------------------------------------------
# Navigate: step forward with boundary propagation
# ---------------------------------------------------------------------------

def successor_coords(space: CoordinateSpace, coords: list[int]) -> list[int]:
    """Compute the successor coordinates.

    Step forward on axis 0. If it wraps, propagate to axis 1. If that
    wraps, propagate to axis 2. Etc.

    This is the carry rule, expressed as boundary propagation on the
    discovered topology. Uses axis 0's period for ALL axes (self-similar
    at every scale — the same wrapping rule applies everywhere).
    """
    if not space.axes:
        return list(coords)

    period = space.axes[0].period
    result = list(coords)

    carry = True
    for i in range(len(result)):
        if not carry:
            break
        result[i] += 1
        if result[i] >= period:
            result[i] = 0
            carry = True
        else:
            carry = False

    # If carry propagates past all current coordinates, extend.
    if carry:
        result.append(1)

    return result


# ---------------------------------------------------------------------------
# Decode: coordinates → token sequence
# ---------------------------------------------------------------------------

def decode(space: CoordinateSpace, coords: list[int]) -> list[NodeId]:
    """Map coordinates back to a token sequence.

    For each axis with a nonzero coordinate (or axis 0 always), emit
    the token at that phase in the axis's cycle.

    Handles coordinates beyond the known axes by recursive decomposition:
    if a coordinate exceeds the period, it decomposes the same way
    (same wrapping rule at every scale).

    Returns list of NodeIds (the output token sequence).
    """
    if not space.axes or not coords:
        return []

    # Determine how many digit positions we need.
    # The coordinates may extend beyond the known axes.
    # Each coordinate maps to one token position.

    # First: normalize coordinates. If any coordinate >= period,
    # decompose recursively using the SAME period (the space is
    # self-similar at every scale).
    period = space.axes[0].period if space.axes else 10
    normalized: list[int] = []
    carry = 0
    for c in coords:
        val = c + carry
        normalized.append(val % period)
        carry = val // period
    while carry > 0:
        normalized.append(carry % period)
        carry //= period

    # Remove trailing zeros (leading zeros in the output).
    while len(normalized) > 1 and normalized[-1] == 0:
        normalized.pop()

    # Map each coordinate to a token using axis 0's cycle
    # (all axes use the same digit vocabulary).
    token_seq = normalized[::-1]  # reverse: highest axis first (like digits)
    result: list[NodeId] = []
    cycle = space.axes[0].token_sequence if space.axes else []
    for coord in token_seq:
        if coord < len(cycle):
            result.append(cycle[coord])
        else:
            result.append(cycle[0])  # fallback

    return result


# ---------------------------------------------------------------------------
# Full prediction: encode → navigate → decode
# ---------------------------------------------------------------------------

def predict_successor(
    space: CoordinateSpace,
    input_tokens: list[NodeId],
) -> list[NodeId]:
    """Predict the successor of a multi-token input.

    Encode the input to coordinates, step forward (with carry),
    decode back to tokens.
    """
    coords = encode(space, input_tokens)
    if coords is None:
        return []
    next_coords = successor_coords(space, coords)
    return decode(space, next_coords)
