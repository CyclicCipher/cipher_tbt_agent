"""
Working memory: stateless prefix parser and Store comonad for the CTKG pipeline.

The module provides two complementary interfaces:

1. **parse_prefix** (stateless, counit):
   A pure function that reads a flat token list and returns a `MemoryState`
   describing the current generation phase.  This is the counit of the Store
   comonad — it projects the full memory state down to the current focus.

   Phase transitions (for a unary operator like succ/pred):
       []                          → START   (nothing seen)
       [op]                        → INPUT   (reading input digits)
       [op, d₁, …]                → INPUT   (still reading input)
       [op, d₁, …, eq]            → OUTPUT  (eq seen, ready to generate output)
       [op, d₁, …, eq, r₁, …]    → OUTPUT  (generating output digits)
       […, <eos>]                  → EOS     (sequence complete)

2. **WorkingMemory** (stateful Store comonad):
   Implements the categorical Store comonad `Store s a = (s → a, s)` where:
     s = focus position (integer index into the prefix)
     a = type distribution at that position (TypeDist = dict[ConceptId, float])

   The memory holds the full list of per-position type distributions `T[i]`
   (the "function" component of the comonad) and the current focus position.
   The `Spine` is stored alongside to track compression-tree state.

   Comonad operations:
     extract() → T[focus]              (counit: type dist at current focus)
     extend(f) → WorkingMemory         (cobind: map positions via f)

   The type distributions T[] are initialized from the `_fixpoint_iteration`
   output in `predict.py` and updated online as new tokens arrive.

See CTKG_ARCHITECTURE.md §Working Memory for the full specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from experiments.symbolic_ai_v2.ctkg.core.spine import Spine
from experiments.symbolic_ai_v2.ctkg.core.node import NodeId, TOKEN_GRAPH

# Type alias for a probability distribution over concept IDs.
# Mirrors the TypeDist alias in operad.py for standalone use.
TypeDist = dict[int, float]


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class MemoryState:
    """Snapshot of the generation state at a given prefix.

    Attributes
    ----------
    phase:
        'START'  — nothing observed yet.
        'INPUT'  — reading/saw input operand digits (before eq or step/ans).
        'OUTPUT' — past the eq / first step/ans delimiter; generating output.
        'EOS'    — sequence terminated with eos_token.
    op:
        The operator atom, if seen (e.g. 'succ', 'pred').  None in START.
    input_digits:
        Digit tokens observed between op and eq (MSB-first).  Fold ops only.
    output_digits:
        Digit tokens observed after eq (MSB-first, excluding eos).  Fold ops only.
    input_tokens:
        ALL tokens between op and first step/ans delimiter.  Chain ops only.
        For fold ops this is always empty.
    output_tokens:
        ALL tokens from first step/ans delimiter onward (excluding eos).
        Chain ops only.  For fold ops this is always empty.
    input_tree:
        Parsed Expr tree for the input operand(s), if already parsed.
        None when no rewrite rules are available or parsing has not been
        attempted.  Set by the Predictor after term_algebra.parse() succeeds.
    output_tree:
        Parsed Expr tree for the output so far (partial or complete).
        None when no output has been generated or parsing has not been
        attempted.
    """

    phase: str = "START"
    op: Optional[str] = None
    input_digits: list[str] = field(default_factory=list)
    output_digits: list[str] = field(default_factory=list)
    input_tokens: list[str] = field(default_factory=list)
    output_tokens: list[str] = field(default_factory=list)
    input_tree: Optional[Any] = None   # Optional[Expr] — avoid circular import
    output_tree: Optional[Any] = None  # Optional[Expr] — avoid circular import

    def __repr__(self) -> str:
        return (
            f"MemoryState(phase={self.phase!r}, op={self.op!r}, "
            f"inp={self.input_digits}, out={self.output_digits})"
        )


# ---------------------------------------------------------------------------
# Digit vocabulary (the only thing known a priori)
# Operators are discovered by process_discover and passed in at call time.
# ---------------------------------------------------------------------------

_DIGIT_SET: frozenset[str] = frozenset("0123456789")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_prefix(
    prefix: list[str],
    eq_token: str = "eq",
    eos_token: str = "<eos>",
    op_atoms: Optional[frozenset[NodeId]] = None,
) -> MemoryState:
    """Parse a token prefix and return the current MemoryState.

    Parameters
    ----------
    prefix:
        The token sequence observed so far (including the operator and any
        already-generated output tokens).
    eq_token:
        The separator token that marks the boundary between input and output.
        Default 'eq'.
    eos_token:
        The end-of-sequence token.  Default '<eos>'.
    op_atoms:
        Set of known operator NodeIds discovered by process_discover.
        The caller (Predictor) passes frozenset(self._rules.keys()) where
        keys are NodeIds.
        If None or empty, no process-rule phase detection is performed and
        the prefix parser returns START/EOS/OUTPUT only via the eq delimiter.

        Sprint B: previously frozenset[str]; now frozenset[NodeId].
        Callers must encode string op atoms to NodeId before passing.

    Returns
    -------
    MemoryState with phase, op, input_digits, output_digits populated.
    """
    if op_atoms is None:
        op_atoms = frozenset()

    if not prefix:
        return MemoryState(phase="START")

    # EOS check: sequence is done if last token is eos
    if prefix[-1] == eos_token:
        op = _find_op(prefix, op_atoms)
        eq_idx = _find_eq(prefix, eq_token)
        inp, out = _split_at_eq(prefix, eq_idx, eos_token)
        return MemoryState(phase="EOS", op=op, input_digits=inp, output_digits=out)

    # Past-eq check
    eq_idx = _find_eq(prefix, eq_token)
    if eq_idx is not None:
        op = _find_op(prefix[:eq_idx], op_atoms)
        inp = _digits_in(prefix[1:eq_idx] if op is not None else prefix[:eq_idx])
        out = _digits_in(prefix[eq_idx + 1:])
        return MemoryState(phase="OUTPUT", op=op, input_digits=inp, output_digits=out)

    # No eq seen yet — we're in INPUT phase if an operator was seen
    op = _find_op(prefix, op_atoms)
    if op is not None:
        op_idx = prefix.index(op)
        inp = _digits_in(prefix[op_idx + 1:])
        return MemoryState(phase="INPUT", op=op, input_digits=inp, output_digits=[])

    # Nothing recognisable yet
    return MemoryState(phase="START")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_op(tokens: list[str], op_atoms: frozenset[NodeId]) -> Optional[str]:
    """Return the first token that is a known operator, or None.

    Sprint B: op_atoms is now frozenset[NodeId].  Tokens are encoded to NodeId
    for the membership test; the original string is returned for backward
    compatibility with MemoryState.op (which stays Optional[str]).
    """
    for t in tokens:
        if TOKEN_GRAPH.encode(t) in op_atoms:
            return t
    return None


def _find_eq(tokens: list[str], eq_token: str) -> Optional[int]:
    """Return the index of eq_token, or None if not present."""
    try:
        return tokens.index(eq_token)
    except ValueError:
        return None


def _digits_in(tokens: list[str]) -> list[str]:
    """Return only the digit tokens from a list (preserving order)."""
    return [t for t in tokens if t in _DIGIT_SET]


def _split_at_eq(
    prefix: list[str],
    eq_idx: Optional[int],
    eos_token: str,
) -> tuple[list[str], list[str]]:
    """Split prefix into (input_digits, output_digits) around eq."""
    if eq_idx is None:
        return [], []
    inp = _digits_in(prefix[1:eq_idx])   # skip op at index 0
    raw_out = prefix[eq_idx + 1:]
    out = _digits_in([t for t in raw_out if t != eos_token])
    return inp, out


# ---------------------------------------------------------------------------
# Chain-format parser (step/ans delimiters instead of eq)
# ---------------------------------------------------------------------------

_CHAIN_DELIMITERS: frozenset[str] = frozenset({"step", "ans"})


def parse_chain_prefix(
    prefix: list[str],
    chain_op_atoms: frozenset[NodeId],
    step_token: str = "step",
    ans_token: str = "ans",
    eq_token: str = "eq",
    eos_token: str = "<eos>",
) -> Optional[MemoryState]:
    """Parse a chain-format prefix.

    Handles two formats:
      • step/ans format (no eq):  [op, in₁, …, step, out₁, …, ans, final₁, …]
      • eq format (mixed input):  [op, in₁, x, in₂, at, …, eq, out₁, out₂, …]

    For both formats the returned state uses input_tokens / output_tokens
    (not input_digits / output_digits).

    Parameters
    ----------
    prefix:
        Token sequence observed so far.
    chain_op_atoms:
        Set of operator NodeIds that produce chain (step/ans or eq-mixed) format
        sequences.  Sprint B: previously frozenset[str]; now frozenset[NodeId].
    step_token, ans_token, eq_token, eos_token:
        Delimiter tokens.

    Returns
    -------
    MemoryState if op ∈ chain_op_atoms, None otherwise.
    """
    if not prefix:
        return None
    op = prefix[0]
    # Sprint B: chain_op_atoms is frozenset[NodeId]; encode op for membership test.
    # op stays as str for backward compatibility in MemoryState.op.
    if TOKEN_GRAPH.encode(op) not in chain_op_atoms:
        return None

    step_ans_delims = frozenset({step_token, ans_token})

    # EOS check (applies to both formats)
    if prefix[-1] == eos_token:
        first_sa = _first_delimiter_idx(prefix, step_ans_delims)
        eq_idx = _find_eq_idx(prefix, eq_token)

        if first_sa is not None and (eq_idx is None or first_sa < eq_idx):
            # step/ans format
            inp_toks = prefix[1:first_sa]
            out_toks = [t for t in prefix[first_sa:-1] if t != eos_token]
        elif eq_idx is not None:
            # eq format
            inp_toks = prefix[1:eq_idx]
            out_toks = [t for t in prefix[eq_idx + 1:-1] if t != eos_token]
        else:
            inp_toks = prefix[1:]
            out_toks = []
        return MemoryState(phase="EOS", op=op, input_tokens=inp_toks, output_tokens=out_toks)

    # Determine output delimiter
    first_sa = _first_delimiter_idx(prefix, step_ans_delims)
    eq_idx = _find_eq_idx(prefix, eq_token)

    if first_sa is not None and (eq_idx is None or first_sa <= eq_idx):
        # step/ans format (step/ans takes priority over eq if both present)
        inp_toks = prefix[1:first_sa]
        out_toks = [t for t in prefix[first_sa:] if t != eos_token]
        return MemoryState(phase="OUTPUT", op=op, input_tokens=inp_toks, output_tokens=out_toks)

    if eq_idx is not None:
        # eq format: eq is the delimiter
        inp_toks = prefix[1:eq_idx]
        out_toks = [t for t in prefix[eq_idx + 1:] if t != eos_token]
        phase = "OUTPUT" if out_toks or prefix[-1] == eq_token else "INPUT"
        return MemoryState(phase=phase, op=op, input_tokens=inp_toks, output_tokens=out_toks)

    # No delimiter found — still in INPUT phase
    inp_toks = prefix[1:]
    return MemoryState(phase="INPUT", op=op, input_tokens=inp_toks, output_tokens=[])


def _first_delimiter_idx(
    tokens: list[str],
    delimiters: frozenset[str],
) -> Optional[int]:
    """Return the index of the first delimiter token, or None."""
    for i, tok in enumerate(tokens):
        if tok in delimiters:
            return i
    return None


def _find_eq_idx(tokens: list[str], eq_token: str) -> Optional[int]:
    """Return the index of eq_token, or None."""
    try:
        return tokens.index(eq_token)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Store comonad
# ---------------------------------------------------------------------------

def _copy_spine(spine: Spine) -> Spine:
    """Return a shallow copy of a Spine (independent stack, same rule bodies)."""
    new_spine = Spine()
    for frame in spine._stack:
        new_spine.push(frame.rule_id, list(frame.body))
        # Restore position in the copied frame
        new_spine.peek().pos = frame.pos  # type: ignore[union-attr]
    return new_spine


class WorkingMemory:
    """Store comonad over the compression tree spine.

    Implements  Store s a = (s → a, s)  where:
      s  = focus position (integer index into the prefix)
      a  = TypeDist — type distribution at that position

    The comonad's "function" component is the list T of per-position type
    distributions; the "store" component is the focus position.  The Spine
    is carried alongside to track compression-tree state.

    Parameters
    ----------
    prefix:
        The token sequence observed so far.
    type_dists:
        Per-position type distributions; T[i] is the TypeDist at position i.
        Initialised from ``_fixpoint_iteration`` in ``predict.py`` and
        updated online as new tokens arrive.  Must have the same length as
        ``prefix`` (or be empty for a fresh start).
    focus:
        Current focus position (index into ``prefix`` / ``type_dists``).
    op_atoms:
        Known operator atoms for ``parse_prefix``.  Defaults to empty.
    spine:
        Compression-tree position tracker.  A fresh ``Spine()`` is used if
        not provided.
    """

    def __init__(
        self,
        prefix: list[str],
        type_dists: list[TypeDist],
        focus: int = 0,
        op_atoms: Optional[frozenset] = None,
        spine: Optional[Spine] = None,
    ) -> None:
        self._prefix: list[str] = list(prefix)
        self._T: list[TypeDist] = list(type_dists)
        self._focus: int = focus
        self._op_atoms: frozenset = op_atoms if op_atoms is not None else frozenset()
        self._spine: Spine = spine if spine is not None else Spine()

    # ------------------------------------------------------------------
    # Comonad operations
    # ------------------------------------------------------------------

    def extract(self) -> TypeDist:
        """Counit: return the type distribution at the current focus.

        Returns
        -------
        T[focus], or {} if focus is out of range.
        """
        if not self._T or self._focus < 0 or self._focus >= len(self._T):
            return {}
        return dict(self._T[self._focus])

    def extend(
        self,
        f: Callable[["WorkingMemory"], TypeDist],
    ) -> "WorkingMemory":
        """Cobind: apply f at each position to produce a new WorkingMemory.

        Creates a new WorkingMemory whose type distributions are
        ``T'[i] = f(WorkingMemory focused at position i)``.

        Parameters
        ----------
        f:
            Function applied at each focus position.  Receives a
            WorkingMemory with the same prefix, type_dists, and spine but
            with focus set to i.

        Returns
        -------
        New WorkingMemory with updated type distributions and the same focus.
        """
        new_T: list[TypeDist] = []
        for i in range(len(self._T)):
            wm_i = WorkingMemory(
                prefix=self._prefix,
                type_dists=self._T,
                focus=i,
                op_atoms=self._op_atoms,
                spine=self._spine,
            )
            new_T.append(f(wm_i))
        return WorkingMemory(
            prefix=self._prefix,
            type_dists=new_T,
            focus=self._focus,
            op_atoms=self._op_atoms,
            spine=self._spine,
        )

    # ------------------------------------------------------------------
    # Token advance
    # ------------------------------------------------------------------

    def advance(
        self,
        token: str,
        token_type_dist: Optional[TypeDist] = None,
    ) -> "WorkingMemory":
        """Append a new token, advancing the focus to the new position.

        Parameters
        ----------
        token:
            The newly observed or generated token.
        token_type_dist:
            Pre-computed type distribution for the new token.
            Defaults to {} (unknown).

        Returns
        -------
        New WorkingMemory with the token appended and focus at the new end.
        """
        new_prefix = self._prefix + [token]
        new_T = self._T + [token_type_dist if token_type_dist is not None else {}]
        new_focus = len(new_T) - 1

        # Advance spine: move to the next symbol in the top frame
        new_spine = _copy_spine(self._spine)
        new_spine.advance()
        # Pop exhausted frames (compression trigger)
        while not new_spine.is_empty() and new_spine.compression_trigger():
            new_spine.pop()

        return WorkingMemory(
            prefix=new_prefix,
            type_dists=new_T,
            focus=new_focus,
            op_atoms=self._op_atoms,
            spine=new_spine,
        )

    # ------------------------------------------------------------------
    # Stateless counit (delegates to parse_prefix)
    # ------------------------------------------------------------------

    def parse_state(
        self,
        eq_token: str = "eq",
        eos_token: str = "<eos>",
    ) -> MemoryState:
        """Return the MemoryState for the prefix up to and including focus.

        This is the counit projection: maps the Store to the current
        generation phase (START / INPUT / OUTPUT / EOS).
        """
        return parse_prefix(
            self._prefix[: self._focus + 1],
            eq_token=eq_token,
            eos_token=eos_token,
            op_atoms=self._op_atoms,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def focus(self) -> int:
        """Current focus position."""
        return self._focus

    @property
    def prefix(self) -> list[str]:
        """Copy of the current token prefix."""
        return list(self._prefix)

    @property
    def type_dists(self) -> list[TypeDist]:
        """Copy of the per-position type distributions."""
        return list(self._T)

    @property
    def spine(self) -> Spine:
        """The compression-tree spine."""
        return self._spine

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"WorkingMemory(focus={self._focus}/{len(self._T)}, "
            f"spine={self._spine!r})"
        )
