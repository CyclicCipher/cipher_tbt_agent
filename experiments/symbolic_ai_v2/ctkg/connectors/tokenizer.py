"""
Tokenizer — converts atomic sensory inputs into opaque integer tokens.

Lives at the connector boundary.  All external input passes through here
before entering the graph.  The system never sees raw characters or floats —
only opaque token IDs.

Design:
  - Every unique symbol gets a unique integer token ID.
  - The vocabulary is built incrementally: new symbols get new IDs.
  - A fixed base vocabulary covers all printable ASCII (keyboard symbols).
  - Additional symbols (Unicode, domain-specific) are added on first encounter.
  - Token IDs are opaque — the system cannot inspect their content.
  - The tokenizer is invertible: token ID → original symbol for display.

Usage:
    tok = Tokenizer()
    token_ids = tok.encode("F=ma")     # [tok['F'], tok['='], tok['m'], tok['a']]
    original = tok.decode(token_ids)   # "F=ma"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# All printable ASCII characters (codes 32-126) — the "keyboard" vocabulary.
_PRINTABLE_ASCII = [chr(c) for c in range(32, 127)]


@dataclass
class Tokenizer:
    """Character-level tokenizer for the CTKG connector boundary.

    Every unique symbol maps to a unique opaque integer token.
    The base vocabulary covers all printable ASCII.  New symbols
    encountered at runtime are added automatically.
    """

    _sym_to_id: dict[str, int] = field(default_factory=dict)
    _id_to_sym: dict[int, str] = field(default_factory=dict)
    _next_id: int = 0

    def __post_init__(self):
        """Seed the base vocabulary with all printable ASCII."""
        if not self._sym_to_id:
            for ch in _PRINTABLE_ASCII:
                self._register(ch)

    def _register(self, symbol: str) -> int:
        """Register a new symbol and return its token ID."""
        if symbol in self._sym_to_id:
            return self._sym_to_id[symbol]
        tid = self._next_id
        self._next_id += 1
        self._sym_to_id[symbol] = tid
        self._id_to_sym[tid] = symbol
        return tid

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def tokenize(self, symbol: str) -> int:
        """Convert a single symbol (character) to its token ID.

        If the symbol hasn't been seen before, it's added to the vocabulary.
        """
        tid = self._sym_to_id.get(symbol)
        if tid is not None:
            return tid
        return self._register(symbol)

    # Alias for use by AgenticLoop
    encode_token = tokenize

    def detokenize(self, token_id: int) -> str:
        """Convert a token ID back to its original symbol."""
        sym = self._id_to_sym.get(token_id)
        if sym is None:
            raise KeyError(f"Unknown token ID: {token_id}")
        return sym

    def encode(self, text: str) -> list[int]:
        """Tokenize a string into a list of token IDs (one per character)."""
        return [self.tokenize(ch) for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        return "".join(self.detokenize(tid) for tid in token_ids)

    def tokenize_any(self, value: Any) -> list[int]:
        """Tokenize any atomic value by converting to its string representation.

        Numbers, booleans, etc. are converted to their string form first,
        then each character is tokenized individually.
        """
        return self.encode(str(value))

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return self._next_id

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._sym_to_id

    def __getitem__(self, symbol: str) -> int:
        """Shorthand: tok['a'] returns the token ID for 'a'."""
        return self.tokenize(symbol)

    def __len__(self) -> int:
        return self._next_id
