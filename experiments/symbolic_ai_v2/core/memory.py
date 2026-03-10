"""Memory: checkpoint save/load and inspection utilities.

Short-term memory = MorphismGraph._buf  (the current chunk buffer).
Long-term memory  = the MorphismGraph itself (all learned edges and rules).

Checkpoint format: numpy .npz (bulk arrays) + JSON sidecar (metadata).
See DATA_FORMATS.md for the exact array layout and estimated file sizes.

save(mg, path, topology)  — save to <path>.npz + <path>.json
load(path)                — restore MorphismGraph from checkpoint
stats(mg)                 — dict of memory usage metrics
buffer_contents(mg)       — human-readable description of the current chunk
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .morphism import Atom, Composition, MorphismGraph, Symbol
from .topology import Topology


# ── Save ──────────────────────────────────────────────────────────────────────

def save(
    mg: MorphismGraph,
    path: str | Path,
    topology: Optional[Topology] = None,
) -> None:
    """Save a MorphismGraph to <path>.npz + <path>.json.

    Pairs are intentionally excluded from the checkpoint: they are only needed
    during learning, not for inference.  Relearning them from scratch on a new
    corpus is cheap; storing them for 89 books would add ~340 MB (raw).

    The checkpoint contains: symbol table, edge table, rule table.
    """
    path = Path(path)

    # ── Symbol table ──────────────────────────────────────────────────────────
    n = len(mg.symbols)
    sym_ids    = np.zeros(n, dtype=np.int32)
    sym_levels = np.zeros(n, dtype=np.int16)
    sym_types  = np.zeros(n, dtype=np.int8)   # 0 = atom, 1 = composition
    # Atom values: encode as a flat UTF-8 byte array with a separate lengths array
    atom_values_flat: list[bytes] = []
    atom_lengths:     list[int]   = []

    for sym in mg.symbols:
        sym_ids[sym.sid]    = sym.sid
        sym_levels[sym.sid] = sym.level
        if isinstance(sym, Atom):
            sym_types[sym.sid] = 0
            encoded = sym.value.encode("utf-8")
            atom_values_flat.append(encoded)
            atom_lengths.append(len(encoded))
        else:
            sym_types[sym.sid] = 1
            atom_values_flat.append(b"")
            atom_lengths.append(0)

    atom_bytes  = b"".join(atom_values_flat)
    atom_buf    = np.frombuffer(atom_bytes, dtype=np.uint8)
    atom_lens   = np.array(atom_lengths, dtype=np.int32)

    # ── Edge table (COO format) ───────────────────────────────────────────────
    E = len(mg.edges)
    edge_src   = np.zeros(E, dtype=np.int32)
    edge_etype = np.zeros(E, dtype=np.int8)
    edge_tgt   = np.zeros(E, dtype=np.int32)
    edge_count = np.zeros(E, dtype=np.int32)
    for i, ((src, et, tgt), cnt) in enumerate(mg.edges.items()):
        edge_src[i]   = src
        edge_etype[i] = et
        edge_tgt[i]   = tgt
        edge_count[i] = cnt

    # ── Rule table ────────────────────────────────────────────────────────────
    R = len(mg.rules)
    rule_comp  = np.zeros(R, dtype=np.int32)
    rule_left  = np.zeros(R, dtype=np.int32)
    rule_etype = np.zeros(R, dtype=np.int8)
    rule_right = np.zeros(R, dtype=np.int32)
    for i, (comp_id, (left, et, right)) in enumerate(mg.rules.items()):
        rule_comp[i]  = comp_id
        rule_left[i]  = left
        rule_etype[i] = et
        rule_right[i] = right

    # ── Save .npz ─────────────────────────────────────────────────────────────
    npz_path = path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        sym_ids=sym_ids, sym_levels=sym_levels, sym_types=sym_types,
        atom_buf=atom_buf, atom_lens=atom_lens,
        edge_src=edge_src, edge_etype=edge_etype,
        edge_tgt=edge_tgt, edge_count=edge_count,
        rule_comp=rule_comp, rule_left=rule_left,
        rule_etype=rule_etype, rule_right=rule_right,
    )

    # ── Save JSON metadata ────────────────────────────────────────────────────
    meta: dict[str, Any] = {
        "format_version": 1,
        "n_symbols":      n,
        "n_atoms":        mg.n_atoms(),
        "n_compositions": mg.n_compositions(),
        "n_edges":        E,
        "n_rules":        R,
        "n_observations": mg._n_obs,
        "n_boundaries":   mg._n_boundaries,
        "topology":       topology.name if topology else None,
        "edge_types":     topology.registry.names() if topology else [],
    }
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(meta, indent=2))


# ── Load ──────────────────────────────────────────────────────────────────────

def load(path: str | Path) -> tuple[MorphismGraph, dict]:
    """Restore a MorphismGraph from <path>.npz + <path>.json.

    Returns (mg, metadata_dict).
    """
    path = Path(path)
    npz_path  = path.with_suffix(".npz")
    json_path = path.with_suffix(".json")

    data = np.load(npz_path)
    meta = json.loads(json_path.read_text())

    mg = MorphismGraph()

    # ── Reconstruct atom values ───────────────────────────────────────────────
    atom_buf  = bytes(data["atom_buf"].tolist())
    atom_lens = data["atom_lens"]
    atom_strs: list[str] = []
    offset = 0
    for length in atom_lens:
        chunk = atom_buf[offset: offset + length]
        atom_strs.append(chunk.decode("utf-8") if length > 0 else "")
        offset += length

    # ── Rebuild symbol table ──────────────────────────────────────────────────
    sym_ids    = data["sym_ids"]
    sym_levels = data["sym_levels"]
    sym_types  = data["sym_types"]
    n = len(sym_ids)
    mg.symbols = [None] * n  # type: ignore[list-item]

    for i in range(n):
        sid   = int(sym_ids[i])
        level = int(sym_levels[i])
        stype = int(sym_types[i])
        if stype == 0:
            val  = atom_strs[sid]
            atom = Atom(sid=sid, level=level, value=val)
            mg.symbols[sid] = atom
            mg.atoms[val]   = sid
        else:
            mg.symbols[sid] = Composition(sid=sid, level=level, left=0, etype=0, right=0)

    # ── Rebuild edge table + output index ─────────────────────────────────────
    edge_src   = data["edge_src"]
    edge_etype = data["edge_etype"]
    edge_tgt   = data["edge_tgt"]
    edge_count = data["edge_count"]
    for i in range(len(edge_src)):
        src, et, tgt, cnt = int(edge_src[i]), int(edge_etype[i]), int(edge_tgt[i]), int(edge_count[i])
        mg.edges[(src, et, tgt)] = cnt
        mg._inc_out(src, et, tgt, cnt)

    # ── Rebuild rule table ────────────────────────────────────────────────────
    rule_comp  = data["rule_comp"]
    rule_left  = data["rule_left"]
    rule_etype = data["rule_etype"]
    rule_right = data["rule_right"]
    for i in range(len(rule_comp)):
        comp_id = int(rule_comp[i])
        left    = int(rule_left[i])
        et      = int(rule_etype[i])
        right   = int(rule_right[i])
        rule_key = (left, et, right)
        mg.rules[comp_id]       = rule_key
        mg.rules_inv[rule_key]  = comp_id
        # Fill in the Composition fields now that we know left/etype/right
        sym = mg.symbols[comp_id]
        if isinstance(sym, Composition):
            sym.left  = left
            sym.etype = et
            sym.right = right

    mg._n_obs        = meta.get("n_observations", 0)
    mg._n_boundaries = meta.get("n_boundaries", 0)
    return mg, meta


# ── Inspection ────────────────────────────────────────────────────────────────

def buffer_contents(mg: MorphismGraph) -> list[tuple[str, Optional[str]]]:
    """Return the current chunk buffer as a human-readable list.

    Each entry is (symbol_description, incoming_edge_name_or_None).
    Useful for debugging the segment boundary detection.
    """
    result = []
    for sid, etype in mg._buf:
        sym_desc = mg.value_of(sid)
        result.append((sym_desc, str(etype) if etype is not None else None))
    return result


def stats(mg: MorphismGraph) -> dict:
    """Return a dict of memory usage and learning statistics."""
    return {
        "n_symbols":       mg.n_symbols(),
        "n_atoms":         mg.n_atoms(),
        "n_compositions":  mg.n_compositions(),
        "n_edges":         mg.n_edges(),
        "n_pairs":         mg.n_pairs(),
        "n_observations":  mg._n_obs,
        "n_boundaries":    mg._n_boundaries,
        "buf_length":      len(mg._buf),
        "compression_ratio": (
            round(mg._n_obs / max(mg.n_compositions(), 1), 2)
            if mg.n_compositions() > 0 else None
        ),
        "bits_per_rule": (
            round(math.log2(mg._n_obs) / max(mg.n_compositions(), 1), 3)
            if mg._n_obs > 0 and mg.n_compositions() > 0 else None
        ),
    }
