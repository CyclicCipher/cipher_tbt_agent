"""
Unlabeled knowledge graph node identifiers (Phase XXIII).

Every node in the CTKG is an opaque integer NodeId — no string stored in the
node itself.  Identity is defined entirely by edges, not by internal labels.
This implements the grounding principle: meaning emerges from relationships.

Reserved NodeId ranges
----------------------
  0 – 127   : ASCII character atoms.  NodeId == ord(char).
               These are the only nodes whose "content" can be recovered
               without graph traversal — chr(node_id) gives the character.
  128+       : Dynamically allocated by TokenGraph on first observation.

Character-level encoding
------------------------
Token 'add' is NOT stored as a string.  It is stored as:
  _id_to_chars[add_node] = (ord('a'), ord('d'), ord('d')) = (97, 100, 100)

String recovery traverses _id_to_chars and maps each element through chr():
  ''.join(chr(c) for c in _id_to_chars[node_id])

The character tuple is the ONLY data stored per token node.

Well-known structural nodes
---------------------------
Imported at module level:
  EQ_NODE, STEP_NODE, ANS_NODE, EOS_NODE, PAD_NODE
  P0_NODE … P9_NODE
  OUTPUT_DELIMS, POSITIONAL_ROLE_NODES
  is_positional_role(nid) -> bool
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# NodeId type
# ---------------------------------------------------------------------------

# Opaque integer identifier for a knowledge graph node.
# Do not compare NodeIds to strings — only to other NodeIds.
NodeId = int

# Reserved range boundaries
CHAR_MIN: int = 0
CHAR_MAX: int = 127
_USER_START: int = 128


# ---------------------------------------------------------------------------
# TokenGraph
# ---------------------------------------------------------------------------

class TokenGraph:
    """Bidirectional string ↔ NodeId registry with character-level encoding.

    Single source of truth for all NodeId assignments in the system.
    Use the module-level ``TOKEN_GRAPH`` singleton — do not instantiate
    additional TokenGraph objects unless for isolated testing.

    Thread safety: NOT thread-safe.  All usage is single-threaded.
    """

    def __init__(self) -> None:
        # Character atoms: NodeId 0-127 pre-populated
        self._str_to_id: dict[str, NodeId] = {}
        self._id_to_chars: dict[NodeId, tuple[NodeId, ...]] = {}
        self._next_id: int = _USER_START

        # Pre-register all ASCII printable characters
        for code in range(CHAR_MAX + 1):
            ch = chr(code)
            self._str_to_id[ch] = code
            self._id_to_chars[code] = (code,)

    # ------------------------------------------------------------------
    # Core encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, token: str) -> NodeId:
        """Return the NodeId for *token*, creating a new one if necessary.

        Single characters (len==1, code≤127) always map to their ASCII code.
        All other tokens are lazily allocated starting at NodeId 128.
        """
        existing = self._str_to_id.get(token)
        if existing is not None:
            return existing

        # Allocate a new NodeId
        nid = self._next_id
        self._next_id += 1
        self._str_to_id[token] = nid
        # Store character sequence
        self._id_to_chars[nid] = tuple(ord(c) for c in token)
        return nid

    def decode(self, node_id: NodeId) -> str:
        """Recover the string for *node_id* by graph traversal.

        Traverses _id_to_chars to reconstruct the original token.
        Returns '<unknown>' for unregistered ids (should not happen in normal use).
        """
        char_seq = self._id_to_chars.get(node_id)
        if char_seq is None:
            return f'<node:{node_id}>'
        return ''.join(chr(c) for c in char_seq)

    def encode_seq(self, tokens: list[str]) -> list[NodeId]:
        """Encode a list of string tokens to NodeIds."""
        return [self.encode(t) for t in tokens]

    def decode_seq(self, node_ids: list[NodeId]) -> list[str]:
        """Decode a list of NodeIds to string tokens."""
        return [self.decode(n) for n in node_ids]

    def encode_multi(self, joined: str) -> tuple[NodeId, ...]:
        """Encode a '\x00'-joined multi-token string to a tuple of NodeIds.

        Used for RelationRule multi-token output values during migration.
        """
        return tuple(self.encode(t) for t in joined.split('\x00'))

    def decode_multi(self, node_ids: tuple[NodeId, ...]) -> str:
        """Decode a tuple of NodeIds back to a '\x00'-joined string."""
        return '\x00'.join(self.decode(n) for n in node_ids)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_char_atom(self, node_id: NodeId) -> bool:
        """Return True if *node_id* is a single ASCII character atom."""
        return CHAR_MIN <= node_id <= CHAR_MAX

    def __len__(self) -> int:
        return self._next_id

    def __repr__(self) -> str:
        return f'TokenGraph(size={self._next_id})'


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

TOKEN_GRAPH: TokenGraph = TokenGraph()
"""Global singleton TokenGraph — the single source of truth for all NodeIds.

Import this directly in all pipeline modules:
    from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
"""


def enc(token: str) -> NodeId:
    """Shorthand for TOKEN_GRAPH.encode(token).  Intended for test files."""
    return TOKEN_GRAPH.encode(token)


def dec(node_id: NodeId) -> str:
    """Shorthand for TOKEN_GRAPH.decode(node_id).  Intended for test files."""
    return TOKEN_GRAPH.decode(node_id)


# ---------------------------------------------------------------------------
# Well-known structural node constants
# ---------------------------------------------------------------------------
# These are pre-registered in TOKEN_GRAPH at module import.
# All pipeline code that previously compared `tok == 'eq'` should now
# compare `tok_id == EQ_NODE`.

EQ_NODE:   NodeId = TOKEN_GRAPH.encode('eq')
STEP_NODE: NodeId = TOKEN_GRAPH.encode('step')
ANS_NODE:  NodeId = TOKEN_GRAPH.encode('ans')
EOS_NODE:  NodeId = TOKEN_GRAPH.encode('<eos>')
PAD_NODE:  NodeId = TOKEN_GRAPH.encode('<pad>')

# Positional role names (p0 … p9) used in RelationStore schemas
P0_NODE:  NodeId = TOKEN_GRAPH.encode('p0')
P1_NODE:  NodeId = TOKEN_GRAPH.encode('p1')
P2_NODE:  NodeId = TOKEN_GRAPH.encode('p2')
P3_NODE:  NodeId = TOKEN_GRAPH.encode('p3')
P4_NODE:  NodeId = TOKEN_GRAPH.encode('p4')
P5_NODE:  NodeId = TOKEN_GRAPH.encode('p5')
P6_NODE:  NodeId = TOKEN_GRAPH.encode('p6')
P7_NODE:  NodeId = TOKEN_GRAPH.encode('p7')
P8_NODE:  NodeId = TOKEN_GRAPH.encode('p8')
P9_NODE:  NodeId = TOKEN_GRAPH.encode('p9')

# Frozen sets for fast membership tests
OUTPUT_DELIMS: frozenset[NodeId] = frozenset({
    EQ_NODE, STEP_NODE, ANS_NODE, EOS_NODE,
})

POSITIONAL_ROLE_NODES: frozenset[NodeId] = frozenset({
    P0_NODE, P1_NODE, P2_NODE, P3_NODE, P4_NODE,
    P5_NODE, P6_NODE, P7_NODE, P8_NODE, P9_NODE,
})

# Identity morphism sentinel — replaces the string "__identity__"
IDENTITY_NODE: NodeId = TOKEN_GRAPH.encode('__identity__')


# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------

def is_positional_role(node_id: NodeId) -> bool:
    """Return True iff *node_id* is one of the positional role nodes p0…p9.

    Replaces: rname.startswith('p') and rname[1:].isdigit()
    """
    return node_id in POSITIONAL_ROLE_NODES


def positional_role_index(node_id: NodeId) -> Optional[int]:
    """Return the integer index of positional role node p<i>, or None."""
    _ORDER = (P0_NODE, P1_NODE, P2_NODE, P3_NODE, P4_NODE,
              P5_NODE, P6_NODE, P7_NODE, P8_NODE, P9_NODE)
    try:
        return _ORDER.index(node_id)
    except ValueError:
        return None
