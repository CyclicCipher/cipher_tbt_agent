"""Process language interpreter.

Executes the process blocks from .ctkg files.  Process blocks are stored
as List[str] in Concept.process — this module makes them executable.

Supported process line forms:
    emit(e1, e2, ...)           — output a tuple (terminates the process)
    var = expr                  — single assignment
    v1, v2 = expr               — tuple-unpacking assignment

Supported expression constructs:
    succ(x)                     — integer x + 1  (non-wrapping; fold needs >9)
    pred(x)                     — integer x - 1  (non-wrapping)
    compare(a, b)               — 'GT' | 'LT' | 'EQ'
    fold(n, init, fn)           — apply fn n times from init
    if(cond, then, else)        — conditional
    equal(a, b)                 — True | False
    lookup(concept, *args)      — call a learned concept via the engine
    expr == TOKEN               — equality comparison (returns bool)
    expr + N  /  expr - N       — integer arithmetic
    name                        — variable or known string literal
    N                           — integer literal

Input naming convention (from input_type List[str]):
    'op' type  → env['op']
    all others → env['a'], env['b'], env['c'], ... (positional)

    Example: input_type = ['digit', 'op', 'digit']
             inputs     = (3, 'ADD', 4)
             env        = {'a': 3, 'op': 'ADD', 'b': 4}

Design note on succ:
    The predecessor/successor concepts in the CTKG wrap at digit boundaries
    (SUCC(9) = 0 in the neural training context).  But the process for
    single_digit_addition uses fold(b, a, succ) and then checks whether
    the result exceeded 9.  This requires a non-wrapping succ so the
    intermediate sum can reach values like 13 or 17.  Therefore succ(x)
    here is plain integer x + 1, and the digit type constraint applies
    only to I/O, not intermediate values.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Known string literals — names that are not variables but symbolic tokens
# ---------------------------------------------------------------------------

_LITERALS: FrozenSet[str] = frozenset({
    'GT', 'LT', 'EQ',
    'ADD', 'SUB',
    'DOT', 'TEN', 'STOP',
    'True', 'False',
})


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""
    (?P<WS>\s+)                           |
    (?P<EQ2>==)                           |
    (?P<PLUS>\+)                          |
    (?P<MINUS>-)                          |
    (?P<LPAREN>\()                        |
    (?P<RPAREN>\))                        |
    (?P<COMMA>,)                          |
    (?P<INT>[0-9]+)                       |
    (?P<NAME>[A-Za-z_][A-Za-z0-9_]*)
""", re.VERBOSE)


def _tokenize(s: str) -> List[Tuple[str, Any]]:
    """Tokenize an expression string into (kind, value) pairs."""
    tokens: List[Tuple[str, Any]] = []
    for m in _TOKEN_RE.finditer(s):
        kind = m.lastgroup
        if kind == 'WS':
            continue
        elif kind == 'INT':
            tokens.append((kind, int(m.group())))
        else:
            tokens.append((kind, m.group()))
    return tokens


# ---------------------------------------------------------------------------
# Recursive descent expression evaluator
# ---------------------------------------------------------------------------

class _Parser:
    """Evaluate a tokenized process expression in a given environment."""

    def __init__(
        self,
        tokens: List[Tuple[str, Any]],
        env: Dict[str, Any],
        engine_ask: Optional[Callable],
        concept_names: FrozenSet[str],
    ):
        self.tokens = tokens
        self.pos = 0
        self.env = env
        self.engine_ask = engine_ask
        self.concept_names = concept_names

    def peek(self) -> Optional[Tuple[str, Any]]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> Tuple[str, Any]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    # Grammar:
    #   expr  ::= term (('==' | '+' | '-') term)?
    #   term  ::= NAME '(' arglist ')' | NAME | INT
    #   arglist ::= empty | expr (',' expr)*

    def parse_expr(self) -> Any:
        left = self.parse_term()
        tok = self.peek()
        if tok is not None and tok[0] in ('EQ2', 'PLUS', 'MINUS'):
            op = self.consume()[0]
            right = self.parse_term()
            if op == 'EQ2':
                return left == right
            elif op == 'PLUS':
                return int(left) + int(right)
            elif op == 'MINUS':
                return int(left) - int(right)
        return left

    def parse_term(self) -> Any:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")

        if tok[0] == 'INT':
            self.consume()
            return tok[1]

        if tok[0] == 'NAME':
            name = self.consume()[1]
            if self.peek() is not None and self.peek()[0] == 'LPAREN':
                self.consume()          # '('
                args = self.parse_arglist()
                self.consume()          # ')'
                return self._apply(name, args)
            else:
                return self._resolve(name)

        raise ValueError(f"Unexpected token: {tok!r}")

    def parse_arglist(self) -> List[Any]:
        if self.peek() is not None and self.peek()[0] == 'RPAREN':
            return []
        args = [self.parse_expr()]
        while self.peek() is not None and self.peek()[0] == 'COMMA':
            self.consume()
            args.append(self.parse_expr())
        return args

    def _resolve(self, name: str) -> Any:
        """Look up a name: variable, built-in literal, or concept name."""
        if name in self.env:
            return self.env[name]
        if name in _LITERALS:
            return name
        if name in self.concept_names:
            return name  # concept name used as a string in lookup(...)
        raise ValueError(
            f"Undefined name {name!r}. "
            f"env keys: {sorted(self.env.keys())}"
        )

    def _apply(self, name: str, args: List[Any]) -> Any:
        """Apply a built-in function to already-evaluated arguments."""

        if name == 'succ':
            return int(args[0]) + 1

        elif name == 'pred':
            return int(args[0]) - 1

        elif name == 'compare':
            a, b = int(args[0]), int(args[1])
            if a > b:   return 'GT'
            elif a < b: return 'LT'
            else:       return 'EQ'

        elif name == 'fold':
            # fold(n, init, fn): apply fn n times from init.
            # fn may be a callable (resolved from env) or a string name.
            n, init, fn = args[0], args[1], args[2]
            n = int(n)
            if callable(fn):
                fn_callable = fn
            elif isinstance(fn, str):
                fn_callable = self.env.get(fn)
                if fn_callable is None or not callable(fn_callable):
                    raise ValueError(f"fold: {fn!r} is not a callable in env")
            else:
                raise TypeError(f"fold: expected callable or name, got {fn!r}")
            result = init
            for _ in range(n):
                result = fn_callable(result)
            return result

        elif name == 'if':
            cond, then, else_ = args[0], args[1], args[2]
            return then if cond else else_

        elif name == 'equal':
            return args[0] == args[1]

        elif name == 'lookup':
            if self.engine_ask is None:
                raise RuntimeError(
                    "lookup() called but engine_ask is not configured"
                )
            concept_name = str(args[0])
            inputs = tuple(args[1:])
            result = self.engine_ask(concept_name, inputs)
            if result is None:
                raise RuntimeError(
                    f"lookup({concept_name!r}, {inputs!r}) returned None — "
                    f"concept has no process and no matching example"
                )
            return result

        else:
            raise ValueError(f"Unknown function: {name!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_assignment_eq(line: str) -> int:
    """Return the index of the assignment '=' in a process line, or -1.

    Skips '==' operators and '=' characters inside parentheses.
    Example:
        'var = expr'                   → 4
        'c = if(compare(r, 9) == GT)'  → 2   (skips the inner ==)
    """
    depth = 0
    for i, c in enumerate(line):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == '=' and depth == 0:
            prev = line[i - 1] if i > 0 else ' '
            nxt  = line[i + 1] if i + 1 < len(line) else ' '
            if prev != '=' and nxt != '=':
                return i
    return -1


def _split_top_level_commas(s: str) -> List[str]:
    """Split s by commas not inside parentheses."""
    parts: List[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ',' and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# ProcessInterpreter
# ---------------------------------------------------------------------------

class ProcessInterpreter:
    """Execute process language expressions from .ctkg files.

    Usage:
        interp = ProcessInterpreter(engine_ask=ai.ask,
                                    concept_names=frozenset(graph.concepts))
        outputs = interp.run(concept.process, inputs=(3, 'ADD', 4),
                             input_type=['digit', 'op', 'digit'])
        # returns (0, 7)
    """

    def __init__(
        self,
        engine_ask: Optional[Callable] = None,
        concept_names: FrozenSet[str] = frozenset(),
    ):
        self.engine_ask = engine_ask
        self.concept_names = concept_names

    def run(
        self,
        process_lines: List[str],
        inputs: tuple,
        input_type: List[str],
    ) -> tuple:
        """Execute process_lines with given inputs; return the emitted tuple."""
        env = self._make_env(inputs, input_type)

        # Inject built-in function references so fold(n, init, succ) works.
        env['succ'] = lambda x: int(x) + 1
        env['pred'] = lambda x: int(x) - 1

        for raw_line in process_lines:
            line = raw_line.strip()
            if not line:
                continue
            result = self._eval_line(line, env)
            if result is not None:
                return result

        raise RuntimeError(
            f"Process did not emit for concept with input_type={input_type}: "
            f"{process_lines!r}"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_env(self, inputs: tuple, input_type: List[str]) -> Dict[str, Any]:
        """Assign variable names to inputs per the naming convention."""
        env: Dict[str, Any] = {}
        letter_names = 'abcdefgh'
        digit_idx = 0
        for i, type_name in enumerate(input_type):
            if type_name == 'op':
                env['op'] = inputs[i]
            else:
                if digit_idx >= len(letter_names):
                    raise ValueError(
                        f"Too many non-op inputs: {input_type}"
                    )
                env[letter_names[digit_idx]] = inputs[i]
                digit_idx += 1
        return env

    def _eval_line(
        self, line: str, env: Dict[str, Any]
    ) -> Optional[tuple]:
        """Evaluate one process line.

        Returns None for assignment lines.
        Returns a tuple for emit lines (terminates the process).
        """
        # --- emit ---
        if line.startswith('emit(') and line.endswith(')'):
            inner = line[5:-1]
            parts = _split_top_level_commas(inner)
            return tuple(self._eval_expr(p.strip(), env) for p in parts)

        # --- assignment (single or tuple unpacking) ---
        idx = _find_assignment_eq(line)
        if idx < 0:
            raise ValueError(f"Unrecognized process line: {line!r}")

        lhs = line[:idx].strip()
        rhs = line[idx + 1:].strip()
        value = self._eval_expr(rhs, env)

        if ',' in lhs:
            # Tuple unpacking: v1, v2 = expr
            names = [n.strip() for n in lhs.split(',')]
            if not isinstance(value, tuple):
                value = (value,)
            for name, val in zip(names, value):
                env[name] = val
        else:
            env[lhs] = value

        return None

    def _eval_expr(self, s: str, env: Dict[str, Any]) -> Any:
        """Evaluate an expression string in the given environment."""
        tokens = _tokenize(s)
        parser = _Parser(
            tokens=tokens,
            env=env,
            engine_ask=self.engine_ask,
            concept_names=self.concept_names,
        )
        return parser.parse_expr()
