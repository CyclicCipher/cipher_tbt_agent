"""
Tokenizer tests — character-level tokenization at the connector boundary.

Run with:
    ./venv/Scripts/python.exe -m pytest experiments/symbolic_ai_v2/tests/test_tokenizer.py -v
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.ctkg.connectors.tokenizer import Tokenizer


class TestTokenizerBasics:
    """Basic tokenization: symbols → opaque IDs → symbols."""

    def test_printable_ascii_preloaded(self):
        tok = Tokenizer()
        assert tok.vocab_size == 95  # 127 - 32

    def test_single_char_roundtrip(self):
        tok = Tokenizer()
        for ch in "abcXYZ019!@#":
            tid = tok.tokenize(ch)
            assert isinstance(tid, int)
            assert tok.detokenize(tid) == ch

    def test_same_char_same_id(self):
        tok = Tokenizer()
        assert tok.tokenize("a") == tok.tokenize("a")

    def test_different_chars_different_ids(self):
        tok = Tokenizer()
        ids = [tok.tokenize(ch) for ch in "abc"]
        assert len(set(ids)) == 3

    def test_encode_decode_roundtrip(self):
        tok = Tokenizer()
        text = "F=ma"
        assert tok.decode(tok.encode(text)) == text

    def test_novel_symbol_auto_registered(self):
        tok = Tokenizer()
        before = tok.vocab_size
        tid = tok.tokenize("\u03b1")  # Greek alpha
        assert tok.vocab_size == before + 1
        assert tok.detokenize(tid) == "\u03b1"

    def test_contains(self):
        tok = Tokenizer()
        assert "a" in tok
        assert "\u03b1" not in tok
        tok.tokenize("\u03b1")
        assert "\u03b1" in tok

    def test_getitem(self):
        tok = Tokenizer()
        assert tok["a"] == tok.tokenize("a")

    def test_encode_token_alias(self):
        """encode_token is an alias for tokenize (used by AgenticLoop)."""
        tok = Tokenizer()
        assert tok.encode_token("a") == tok.tokenize("a")


class TestTokenizerAny:
    """tokenize_any: convert non-string values to token sequences."""

    def test_number_to_tokens(self):
        tok = Tokenizer()
        tokens = tok.tokenize_any(42)
        assert tok.decode(tokens) == "42"

    def test_float_to_tokens(self):
        tok = Tokenizer()
        tokens = tok.tokenize_any(3.14)
        assert tok.decode(tokens) == "3.14"

    def test_bool_to_tokens(self):
        tok = Tokenizer()
        tokens = tok.tokenize_any(True)
        assert tok.decode(tokens) == "True"


class TestTokenizerOpacity:
    """Token IDs are opaque integers — no semantic content."""

    def test_ids_are_plain_ints(self):
        tok = Tokenizer()
        assert type(tok.tokenize("a")) is int

    def test_registration_order(self):
        tok = Tokenizer()
        # Space is first registered (ASCII 32), so tid_space = 0
        assert tok.tokenize(" ") == 0
        # 'a' is 66th registered (97 - 32 = 65)
        assert tok.tokenize("a") == ord("a") - 32
