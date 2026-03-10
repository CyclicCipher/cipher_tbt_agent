# DATA_FORMATS.md — Symbolic AI v2 Data Representation Specification

This document specifies the in-memory layout, key encoding, serialization format,
and Rust mapping for all data structures in symbolic_ai_v2. It is a companion to
BLUEPRINT.md and must be kept in sync with any architectural changes.

---

## Two graphs, two edge-type vocabularies

There are two distinct graph structures in v2, with **completely separate** edge-type
namespaces:

| Graph | Edge types | Example types | Governed by |
|-------|-----------|--------------|-------------|
| **MorphismGraph** | Input topology directions | `next`, `right`, `later` | `topology.py` |
| **CTKG** | Ontology relations | `requires`, `composes_with` | `ctkg/graph.py` |

The MorphismGraph sees only topology edges and has no knowledge of CTKG relations.
The CTKG organises the discovered morphisms and has no knowledge of raw topology.
**Do not mix the two namespaces.**

---

## MorphismGraph edge types — topology table

The number of edge types for the MorphismGraph is fixed at graph creation time by
the input topology. It never grows with data.

| Topology | Edge types | Count | Bits required |
|----------|-----------|-------|--------------|
| 1D sequence (text, arithmetic) | `next`, `prev` | 2 | 1 |
| 1D periodic / cycle | `next`, `prev` | 2 | 1 |
| 2D image, 4-connected | `right`, `left`, `down`, `up` | 4 | 2 |
| 2D image, 8-connected (diagonal) | + `dr`, `dl`, `ur`, `ul` | 8 | 3 |
| 3D volume, 6-connected | `+x`,`-x`,`+y`,`-y`,`+z`,`-z` | 6 | 3 |
| 3D volume, 6-connected + time | above + `later`, `earlier` | 8 | 3 |
| 3D volume, 18-connected (face+edge) | 18 directions | 18 | 5 |
| 3D volume, 18-connected + time | | 20 | 5 |
| 3D volume, 26-connected (full) | all 3×3×3−1 neighbors | 26 | 5 |
| **3D volume, 26-connected + time** | | **28** | **5** |
| Audio spectrogram | `next`, `prev`, `higher`, `lower` | 4 | 2 |
| Video (2D spatial + time) | 4 spatial + `later`, `earlier` | 6 | 3 |
| Mathematical expression tree | `left-child`, `right-child`, `parent` | 3 | 2 |
| Arbitrary relational / KG | user-defined | variable | variable |

**The worst-case perceptual topology in any foreseeable use case is 3D 26-connected
+ time = 28 edge types = 5 bits.** If a new topology requires more than 28 edge
types, update this table, recalculate the key layout below, and update the Rust
mapping.

Note: 3D data is the correct topology for CAD (solid geometry), medical imaging
(CT/MRI voxels), and scientific simulation grids. 26-connected is the full
neighbourhood; 6-connected is usually sufficient for smooth fields.

---

## Symbol IDs

Symbol IDs are non-negative integers assigned sequentially (0, 1, 2, …). Both
atoms (primitive observations) and compositions (discovered abstractions) share
the same ID space in the MorphismGraph.

Symbol count grows with corpus size. Indicative numbers from the v1 corpus:
- 89 Latin books, character level: ~1.2M compositions across 10 levels
- Word-level English (hypothetical): potentially tens of millions

**There is no hard cap on symbol IDs.** The representation must handle growth.

---

## In-memory key encoding

### Edge key: (src, edge_type, tgt) → count

```
Python:  tuple[int, int, int]   — (src_id, etype_int, tgt_id)
Rust:    struct EdgeKey { src: u32, etype: u8, tgt: u32 }   // 9 bytes → pad to 12
```

Edge type is always stored as a **small integer** (0-indexed), not a string. A
global `EdgeTypeRegistry` maps `str → int` at graph creation. This avoids string
hashing in the inner observation loop.

### Pair key: (Q, e1, P, e2, S) → count

The pair key represents the 5-tuple for SEQUITUR digram uniqueness checks. This
is the hottest data structure — called on every observation.

```
Python:  tuple[int, int, int, int, int]   — (Q, e1_int, P, e2_int, S)
Rust:    struct PairKey { q: u32, e1: u8, p: u32, e2: u8, s: u32 }  // 14 bytes → pad to 16
```

**Why not packed integers?** Packed int64 pair keys work only when symbol IDs are
small. The constraints are:

```
pair_key bits = 3 × symbol_bits + 2 × etype_bits ≤ 63  (signed int64)
```

| Max edge types | Etype bits | Max symbol IDs |
|---------------|-----------|----------------|
| 2  (1D) | 1 | 1,048,576 |
| 8  (2D diagonal / 3D 6-conn+time) | 3 | 524,288 |
| 28 (3D 26-conn+time) | 5 | 131,072 |

The 89-book corpus produced ~1.2M compositions, already above the 3D-diagonal
limit. **Packed int64 pair keys are only valid for small vocabularies (< 131K
symbols when using 3D diagonal topology).** They are not suitable as a general
solution.

For Python (validation phase): use plain tuple keys — correct for all corpus
sizes, no manual packing, no overflow risk.

For Rust (production): use `HashMap<PairKey, u32>` with AHash or FxHash.
The 16-byte struct key hashes in a single 128-bit SIMD operation on modern CPUs.
This is faster than manually packing bits and requires no size limits.

### Output index: src → edge_type → {tgt: count}

For O(degree) iteration over outgoing edges (used in predict()):

```
Python:  dict[int, dict[int, dict[int, int]]]   — _out[src][etype][tgt]
         Built lazily on first query; not populated during observe().

Rust:    Vec<HashMap<u8, HashMap<u32, u32>>>     indexed by src (Vec index = src ID)
         Or: CSR matrix after training is complete (frozen model)
```

---

## CTKG edge types

The CTKG (`experiments/ctkg/graph.py`) uses Python object references for edge
types (strings like `'requires'`), not small integers. This is separate from the
MorphismGraph edge type registry. The CTKG schema currently has approximately
20 relation types; this is expected to grow to perhaps 50-100.

The CTKG is not a performance bottleneck (it is updated only at segment boundaries,
not on every token). Its current representation in `graph.py` is appropriate.

---

## Serialization format

Target: compact, fast-loading, language-agnostic.

### Format: numpy `.npz` + JSON header

One `.npz` file contains named arrays. A JSON sidecar (or embedded in the `.npz`
as a string array) contains the schema version and metadata.

#### Symbol table
| Array name | dtype | Shape | Contents |
|-----------|-------|-------|----------|
| `sym_ids` | int32 | (N,) | Sequential symbol IDs 0..N-1 |
| `sym_levels` | int16 | (N,) | Hierarchy level (0=atom, >0=composition) |
| `sym_types` | int8 | (N,) | 0=atom, 1=composition |
| `sym_values` | uint8 | variable | Atom string values, UTF-8, length-prefixed |

#### Edge table (sorted by src, then etype)
| Array name | dtype | Shape | Contents |
|-----------|-------|-------|----------|
| `edge_src` | int32 | (E,) | Source symbol ID |
| `edge_etype` | int8 | (E,) | Edge type integer |
| `edge_tgt` | int32 | (E,) | Target symbol ID |
| `edge_count` | int32 | (E,) | Observation count |

CSR index: `edge_row_ptr[src]` = first index in edge arrays where src appears.
Enables O(degree) iteration without scanning all edges.

#### Pair table (COO format — no sort order required)
| Array name | dtype | Shape | Contents |
|-----------|-------|-------|----------|
| `pair_Q` | int32 | (P,) | Q symbol ID |
| `pair_e1` | int8 | (P,) | Edge type Q→P |
| `pair_P` | int32 | (P,) | P symbol ID |
| `pair_e2` | int8 | (P,) | Edge type P→S |
| `pair_S` | int32 | (P,) | S symbol ID |
| `pair_count` | int32 | (P,) | Co-occurrence count |

#### Composition rules
| Array name | dtype | Shape | Contents |
|-----------|-------|-------|----------|
| `rule_comp` | int32 | (R,) | Composition symbol ID |
| `rule_left` | int32 | (R,) | Left constituent ID |
| `rule_etype` | int8 | (R,) | Edge type left→right |
| `rule_right` | int32 | (R,) | Right constituent ID |

#### Edge type registry (JSON metadata)
```json
{
  "format_version": 1,
  "n_symbols": 12345,
  "n_edges": 67890,
  "n_pairs": 11111,
  "n_rules": 2222,
  "edge_types": ["next", "prev"],
  "topology": "sequence_1d"
}
```

#### Expected sizes (89-book Latin corpus estimate)
| Table | Rows | Bytes per row | Estimated size |
|-------|------|--------------|----------------|
| Symbols | 1.2M | 9 bytes | ~11 MB |
| Edges | ~5M | 13 bytes | ~65 MB |
| Pairs | ~20M | 17 bytes | ~340 MB |
| Rules | 1.2M | 13 bytes | ~16 MB |

Raw total: ~430 MB. With zstd compression (typical ratio 4-8×): **54–108 MB**.
v1 pickle was 87 MB — comparable, but v2 format is language-agnostic and
mmap-loadable.

Note: if the pair table is too large, it can be discarded after training (it is
only needed for SEQUITUR composition detection during learning, not for inference).
Without pairs: ~90 MB → **12–23 MB compressed**.

---

## Rust mapping

When the Python validation is complete and the architecture is confirmed correct,
the Rust rewrite will use:

```rust
use ahash::AHashMap;

type SymbolId = u32;   // up to 4 billion symbols
type EdgeType  = u8;   // up to 256 edge types (covers 3D 26-conn+time = 28)
type Count     = u32;  // up to 4 billion observations

#[derive(Hash, Eq, PartialEq)]
struct EdgeKey { src: SymbolId, etype: EdgeType, tgt: SymbolId }

#[derive(Hash, Eq, PartialEq)]
struct PairKey { q: SymbolId, e1: EdgeType, p: SymbolId, e2: EdgeType, s: SymbolId }

struct MorphismGraph {
    symbols:   Vec<Symbol>,                          // indexed by SymbolId
    atoms:     AHashMap<Box<[u8]>, SymbolId>,        // value bytes → ID
    edges:     AHashMap<EdgeKey, Count>,
    pairs:     AHashMap<PairKey, Count>,
    rules:     Vec<(SymbolId, EdgeType, SymbolId)>,  // indexed by comp ID
    rules_inv: AHashMap<EdgeKey, SymbolId>,          // (left,etype,right) → comp
    // CSR output index (built lazily or after freeze):
    out_ptr:   Vec<u32>,     // out_ptr[src] = first edge index for src
    out_etype: Vec<EdgeType>,
    out_tgt:   Vec<SymbolId>,
    out_cnt:   Vec<Count>,
    buf:       Vec<(SymbolId, EdgeType)>,            // current chunk buffer
}
```

Key performance properties of this layout:
- `symbols` is a Vec → O(1) indexed access, cache-friendly sequential scan
- `EdgeKey` is 9 bytes → 16-byte aligned struct, 1-2 cache line hits per lookup
- `PairKey` is 14 bytes → 16-byte aligned, single cache line lookup
- AHashMap with 16-byte keys → hashed in a single 128-bit SIMD operation
- CSR output index → sequential memory access for outgoing edge iteration

---

## Known limits and upgrade paths

| Limit | Current value | Upgrade path |
|-------|--------------|-------------|
| Max edge types (Python) | 256 (u8) | Increase to u16; update pair tuple |
| Max edge types (Rust) | 256 (u8) | Change `EdgeType = u16`; recompile |
| Max symbol IDs (Python) | unlimited (Python int) | No limit |
| Max symbol IDs (Rust) | 4,294,967,295 (u32) | Change to u64; 2× memory for keys |
| Max edge count | 4,294,967,295 (u32) | Change to u64 if needed |
| Max topology dimensions | no hard limit | Add new edge type names |

**3D+1T with full 26-connectivity (28 edge types)** fits comfortably within u8
edge types and requires no changes to data structures. This covers CAD solids,
medical volumetric imaging, and physics simulation grids.

If a future topology requires more than 256 edge types (e.g., a raw Wikidata
knowledge graph with ~10,000 property types): the MorphismGraph should not be
used directly on that data. Instead, apply a functor to collapse the 10,000
Wikidata property types into a smaller set of CTKG-level abstract relation types
before feeding into the MorphismGraph. This is architecturally correct: Wikidata
properties are not perceptual topology — they are knowledge structure that belongs
in the CTKG.
