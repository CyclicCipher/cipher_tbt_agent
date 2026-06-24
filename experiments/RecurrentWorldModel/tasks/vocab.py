"""Shared token vocabulary for the few-shot arithmetic line of experiments (FEW_SHOT_ARITHMETIC.md).

Discovery (number_line.py) and arithmetic (arithmetic.py) MUST share one vocab so the value rows
`VAL(v)` are the *same embedding table rows* across phases -- the number-line structure discovered
in phase 0 (succ/pred/compare on Z_m) is exactly what the value embeddings carry into phase 1
(few-shot addition). Non-VAL op/relation tokens occupy distinct low rows per phase; only the VAL rows
need to align, and they do (VAL0 onward).
"""

from __future__ import annotations

PAD = 0
EQ = 1     # sequence marker; the scored slot sits here, its next token is the answer
ADD = 2    # arithmetic: a + b
MUL = 3    # arithmetic: a * b
SUCC = 4   # discovery: successor movement (+1 mod m)
PRED = 5   # discovery: predecessor movement (-1 mod m)
CMP = 6    # discovery: circular-distance comparison
VAL0 = 7   # VAL(v) = VAL0 + v  (v in 0..m-1)

OP = {"add": ADD, "mul": MUL}


def val(v: int) -> int:
    return VAL0 + v


def vocab_size(m: int) -> int:
    return VAL0 + m
