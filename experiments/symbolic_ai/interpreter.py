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
    fold_until(max_steps, init, step_fn, stop_pred)
                                — apply step_fn repeatedly until stop_pred(state) is True,
                                  or until max_steps iterations (HARD SAFETY CAP — never infinite).
                                  stop_pred is checked BEFORE each step; if True on init, returns init.
                                  If max_steps is reached without stopping, returns current state.
                                  Use for: division, GCD, integer sqrt — any algorithm where
                                  the number of steps depends on the data.
    fn(param, body)             — anonymous function (closure); body is lazily captured
                                  e.g. fn(k, fold(a, k, succ)) computes k -> a+k
                                  Enables: multiplication = fold(b, 0, fn(k, fold(a, k, succ)))
                                           exponentiation = fold(b, 1, fn(acc, fold(a, 0, fn(k, fold(acc, k, succ)))))
    pair(a, b)                  — Python tuple (a, b); multi-value state for fold_until
    triple(a, b, c)             — Python tuple (a, b, c)
    first(s)                    — s[0]; first element of a pair/triple state
    second(s)                   — s[1]; second element of a pair/triple state
    third(s)                    — s[2]; third element of a triple state
    if(cond, then, else)        — conditional
    equal(a, b)                 — True | False
    lookup(concept, *args)      — call a learned concept via the engine
    expr == TOKEN               — equality comparison (returns bool)
    expr + N  /  expr - N       — integer arithmetic
    name                        — variable or known string literal
    N                           — integer literal

Level C — Symbolic expression primitives (Gaps 2–5 toward ODEs):
    Expressions are tagged Python tuples:
        ('NUM', int)            — numeric constant
        ('VAR', str)            — symbolic variable (use literals X, Y, Z, T)
        ('ADD', e, e)           — addition node
        ('SUB', e, e)           — subtraction node
        ('MUL', e, e)           — multiplication node
        ('DIV', e, e)           — division node (used for rational coefficients)
        ('POW', e, int)         — integer power node
        ('NEG', e)              — negation node

    Constructors (with constant folding):
    sym_num(n)                  — ('NUM', n)
    sym_var(v)                  — ('VAR', v); v is a literal like X, Y, Z, T
    sym_add(e1, e2)             — ('ADD', e1, e2)
    sym_sub(e1, e2)             — ('SUB', e1, e2)
    sym_mul(e1, e2)             — ('MUL', e1, e2)
    sym_div(e1, e2)             — ('DIV', e1, e2); exact division folds to NUM
    sym_pow(e, n)               — ('POW', e, n); n must be a non-negative integer
    sym_neg(e)                  — ('NEG', e)

    Calculus:
    sym_diff(expr, var)         — differentiate expr w.r.t. var (power + product + quotient rule)
    sym_integrate(expr, var)    — antiderivative of polynomial expr w.r.t. var (no constant term)
                                  Supports: constants, monomials, linear combos, c*xⁿ.
                                  Does NOT support: products where both factors contain var
                                  (would require integration by parts).

    Evaluation and rewriting:
    sym_eval(expr, var, val)    — evaluate expr numerically with var=val; returns int/float
    sym_subst(expr, var, repl)  — substitute ('VAR', var) → repl throughout expr
    sym_str(expr)               — human-readable string representation of expr

    Inspection (expression tree introspection — Gap 5):
    sym_tag(expr)               — tag string of the node: 'NUM'|'VAR'|'ADD'|'SUB'|'MUL'|'DIV'|
                                  'POW'|'NEG'; comparable with == in process lines
    sym_lhs(expr)               — first child of a binary node (ADD, SUB, MUL, DIV, POW)
    sym_rhs(expr)               — second child of a binary node
    sym_arg(expr)               — single child of a unary node (NEG)
    sym_val(expr)               — numeric value of a NUM node (returns int)
    sym_name(expr)              — variable name of a VAR node (returns str)
    sym_exp(expr)               — exponent of a POW node (returns int)
    sym_expand(expr)            — distribute MUL over ADD/SUB (FOIL / distributivity)
                                  e.g. sym_expand((x+1)*(x-1)) → x²-1
                                  Does NOT collect like terms; use sym_coeff after expanding.
    sym_coeff(expr, var, deg)   — extract coefficient of var^deg from a polynomial expression
                                  Returns a float. Assumes expr is in expanded form.
                                  e.g. sym_coeff(3x²+5x-2, X, 2) → 3.0

Float arithmetic (Gap 7 — exact floating-point):
    float_num(n)                — float(n); converts any number to Python float
    float_sqrt(x)               — sqrt of float(x); returns float
    float_abs(x)                — abs of float(x)
    float_exp(x)                — e^float(x); math.exp
    float_log(x)                — natural log of float(x); math.log
    float_pow(x, n)             — float(x) ** int(n); integer exponent
    float_pi()                  — math.pi (3.14159...)
    float_add(a, b)             — float(a) + float(b)
    float_sub(a, b)             — float(a) - float(b)
    float_mul(a, b)             — float(a) * float(b)
    float_div(a, b)             — float(a) / float(b)

Sequence primitives (Gap 4):
    scan(n, init, step)         — apply step n times from init, collecting all states:
                                  [step(init), step²(init), ..., stepⁿ(init)]
                                  Returns a Python list. step may be fn(...) or a named fn.
    seq_get(seq, i)             — element at index i (0-based)
    seq_len(seq)                — number of elements
    seq_cons(head, tail)        — prepend head to list, return new list
    seq_nil()                   — empty list []
    seq_head(seq)               — first element (= seq_get(seq, 0))
    seq_tail(seq)               — all elements except the first

Design note on fn:
    fn(param, body) is a special form — param is read as a literal name (not
    evaluated), and body tokens are captured and re-evaluated on each call.
    This gives lexical scoping: free variables in body are bound at fn
    creation time.  Nested fns work correctly because each call re-parses
    the body with the updated environment.

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
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Phase D: dry-run protection
# ---------------------------------------------------------------------------

class DryRunError(RuntimeError):
    """Raised when an effectful primitive is called in dry-run mode.

    Synthesis always runs in dry-run mode so that template testing never
    sends live commands to the game environment.  If a template requires
    an effectful primitive (e.g. mc_attack), this error is raised and the
    template is skipped.

    Catch this in _test_template() to safely reject action-dependent templates.
    """
    def __init__(self, primitive_name: str):
        self.primitive_name = primitive_name
        super().__init__(
            f"Effectful primitive {primitive_name!r} called in dry-run mode. "
            f"This template requires a live game environment."
        )


# ---------------------------------------------------------------------------
# Known string literals — names that are not variables but symbolic tokens
# ---------------------------------------------------------------------------

_LITERALS: FrozenSet[str] = frozenset({
    'GT', 'LT', 'EQ',
    'ADD', 'SUB',
    'DOT', 'TEN', 'STOP',
    'True', 'False',
    'X', 'Y', 'Z', 'T', 'R',  # symbolic variable names for Level C primitives
    # Symbolic node tags — returned by sym_tag(), comparable with ==
    'NUM', 'VAR', 'MUL', 'POW', 'NEG', 'DIV',
})


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""
    (?P<WS>\s+)                           |
    (?P<EQ2>==)                           |
    (?P<GTE>>=)                           |
    (?P<LTE><=)                           |
    (?P<GT>>)                             |
    (?P<LT><)                             |
    (?P<PLUS>\+)                          |
    (?P<MINUS>-)                          |
    (?P<LPAREN>\()                        |
    (?P<RPAREN>\))                        |
    (?P<COMMA>,)                          |
    (?P<FLOAT>[0-9]+\.[0-9]+)            |
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
        elif kind == 'FLOAT':
            tokens.append((kind, float(m.group())))
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
        modality_fns: Optional[Dict[str, Callable]] = None,
        dry_run: bool = False,
        effectful_fns: Optional[Set[str]] = None,
    ):
        self.tokens = tokens
        self.pos = 0
        self.env = env
        self.engine_ask = engine_ask
        self.concept_names = concept_names
        self.modality_fns: Dict[str, Callable] = modality_fns if modality_fns is not None else {}
        self.dry_run: bool = dry_run
        self.effectful_fns: Set[str] = effectful_fns if effectful_fns is not None else set()

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
        if tok is not None and tok[0] in ('EQ2', 'GT', 'LT', 'GTE', 'LTE', 'PLUS', 'MINUS'):
            op = self.consume()[0]
            right = self.parse_term()
            if op == 'EQ2':
                return left == right
            elif op == 'GT':
                return float(left) > float(right)
            elif op == 'LT':
                return float(left) < float(right)
            elif op == 'GTE':
                return float(left) >= float(right)
            elif op == 'LTE':
                return float(left) <= float(right)
            elif op == 'PLUS':
                # Preserve int type when both operands are integral
                l, r = float(left) + 0.0, float(right) + 0.0
                result_f = l + r
                return int(result_f) if result_f == int(result_f) else result_f
            elif op == 'MINUS':
                l, r = float(left) + 0.0, float(right) + 0.0
                result_f = l - r
                return int(result_f) if result_f == int(result_f) else result_f
        return left

    def parse_term(self) -> Any:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")

        if tok[0] == 'INT':
            self.consume()
            return tok[1]

        if tok[0] == 'FLOAT':
            self.consume()
            return tok[1]  # already a Python float from _tokenize

        if tok[0] == 'NAME':
            name = self.consume()[1]
            if self.peek() is not None and self.peek()[0] == 'LPAREN':
                self.consume()          # '('

                # ----------------------------------------------------------
                # Special form: fn(param, body)
                # param is captured as a literal name (not evaluated).
                # body tokens are captured unparsed and re-evaluated on each
                # call, providing lexical scoping.
                # ----------------------------------------------------------
                if name == 'fn':
                    param_tok = self.peek()
                    if param_tok is None or param_tok[0] != 'NAME':
                        raise ValueError(
                            f"fn: expected parameter name, got {param_tok!r}"
                        )
                    param_name = self.consume()[1]
                    comma = self.peek()
                    if comma is None or comma[0] != 'COMMA':
                        raise ValueError(
                            f"fn: expected ',' after parameter '{param_name}'"
                        )
                    self.consume()  # ','

                    # Capture body tokens up to the matching closing ')'.
                    # depth 0 = we are directly inside fn(...); the first
                    # RPAREN at depth 0 is the fn's own closing paren.
                    body_tokens: List[Tuple[str, Any]] = []
                    depth = 0
                    while self.pos < len(self.tokens):
                        t = self.tokens[self.pos]
                        if t[0] == 'RPAREN' and depth == 0:
                            break
                        if t[0] == 'LPAREN':
                            depth += 1
                        elif t[0] == 'RPAREN':
                            depth -= 1
                        body_tokens.append(t)
                        self.pos += 1
                    self.consume()  # closing ')' of fn(...)

                    # Build lexically-scoped closure.
                    captured_env      = dict(self.env)
                    captured_ask      = self.engine_ask
                    captured_cnames   = self.concept_names
                    captured_mfns     = self.modality_fns
                    captured_dry_run  = self.dry_run
                    captured_effectfl = self.effectful_fns

                    def _make_closure(param, tokens, base_env, ask, cnames, mfns,
                                      dry_run, effectful):
                        def _closure(arg_val: Any) -> Any:
                            call_env = dict(base_env)
                            call_env[param] = arg_val
                            return _Parser(
                                tokens, call_env, ask, cnames, mfns,
                                dry_run=dry_run, effectful_fns=effectful,
                            ).parse_expr()
                        return _closure

                    return _make_closure(
                        param_name, body_tokens,
                        captured_env, captured_ask, captured_cnames, captured_mfns,
                        captured_dry_run, captured_effectfl,
                    )
                # ----------------------------------------------------------

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

        elif name == 'fold_until':
            # fold_until(max_steps, init, step_fn, stop_pred):
            #   Bounded iteration — CANNOT loop forever.
            #   stop_pred is checked BEFORE each step.
            #   If max_steps is reached, returns current state (safe fallback).
            if len(args) != 4:
                raise ValueError(
                    f"fold_until requires exactly 4 args: "
                    f"(max_steps, init, step_fn, stop_pred), got {len(args)}"
                )
            max_steps, state, step_fn, stop_pred = args
            max_steps = int(max_steps)
            if max_steps < 0:
                raise ValueError(
                    f"fold_until: max_steps must be non-negative, got {max_steps}"
                )
            for step_fn_arg in (step_fn, stop_pred):
                if not callable(step_fn_arg):
                    raise TypeError(
                        f"fold_until: step_fn and stop_pred must be callables, "
                        f"got {step_fn_arg!r}"
                    )
            for _ in range(max_steps):
                if stop_pred(state):
                    break
                state = step_fn(state)
            return state

        # ----------------------------------------------------------------
        # Tuple state constructors and accessors for fold_until
        # ----------------------------------------------------------------

        elif name == 'pair':
            if len(args) != 2:
                raise ValueError(f"pair requires exactly 2 args, got {len(args)}")
            return (args[0], args[1])

        elif name == 'triple':
            if len(args) != 3:
                raise ValueError(f"triple requires exactly 3 args, got {len(args)}")
            return (args[0], args[1], args[2])

        elif name == 'first':
            if len(args) != 1:
                raise ValueError(f"first requires exactly 1 arg, got {len(args)}")
            s = args[0]
            if not isinstance(s, tuple) or len(s) < 1:
                raise TypeError(f"first: expected a tuple with ≥1 element, got {s!r}")
            return s[0]

        elif name == 'second':
            if len(args) != 1:
                raise ValueError(f"second requires exactly 1 arg, got {len(args)}")
            s = args[0]
            if not isinstance(s, tuple) or len(s) < 2:
                raise TypeError(f"second: expected a tuple with ≥2 elements, got {s!r}")
            return s[1]

        elif name == 'third':
            if len(args) != 1:
                raise ValueError(f"third requires exactly 1 arg, got {len(args)}")
            s = args[0]
            if not isinstance(s, tuple) or len(s) < 3:
                raise TypeError(f"third: expected a tuple with ≥3 elements, got {s!r}")
            return s[2]

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

        # ----------------------------------------------------------------
        # Level C: symbolic expression constructors and operations
        # ----------------------------------------------------------------

        elif name == 'sym_num':
            if len(args) != 1:
                raise ValueError(f"sym_num requires 1 arg, got {len(args)}")
            return ('NUM', int(args[0]))

        elif name == 'sym_var':
            if len(args) != 1:
                raise ValueError(f"sym_var requires 1 arg, got {len(args)}")
            return ('VAR', str(args[0]))

        elif name == 'sym_add':
            if len(args) != 2:
                raise ValueError(f"sym_add requires 2 args, got {len(args)}")
            return _sym_simplify_add(args[0], args[1])

        elif name == 'sym_sub':
            if len(args) != 2:
                raise ValueError(f"sym_sub requires 2 args, got {len(args)}")
            return _sym_simplify_sub(args[0], args[1])

        elif name == 'sym_mul':
            if len(args) != 2:
                raise ValueError(f"sym_mul requires 2 args, got {len(args)}")
            return _sym_simplify_mul(args[0], args[1])

        elif name == 'sym_pow':
            if len(args) != 2:
                raise ValueError(f"sym_pow requires 2 args, got {len(args)}")
            return _sym_simplify_pow(args[0], int(args[1]))

        elif name == 'sym_neg':
            if len(args) != 1:
                raise ValueError(f"sym_neg requires 1 arg, got {len(args)}")
            e = args[0]
            if isinstance(e, tuple) and e[0] == 'NUM':
                return ('NUM', -e[1])
            return ('NEG', e)

        elif name == 'sym_eval':
            if len(args) != 3:
                raise ValueError(
                    f"sym_eval requires 3 args (expr, var, val), got {len(args)}"
                )
            expr, var, val = args
            return _sym_eval_helper(expr, {str(var): val})

        elif name == 'sym_diff':
            if len(args) != 2:
                raise ValueError(
                    f"sym_diff requires 2 args (expr, var), got {len(args)}"
                )
            expr, var = args
            return _sym_diff_helper(expr, str(var))

        elif name == 'sym_subst':
            if len(args) != 3:
                raise ValueError(
                    f"sym_subst requires 3 args (expr, var, replacement), got {len(args)}"
                )
            expr, var, replacement = args
            return _sym_subst_helper(expr, str(var), replacement)

        elif name == 'sym_str':
            if len(args) != 1:
                raise ValueError(f"sym_str requires 1 arg, got {len(args)}")
            return _sym_str_helper(args[0])

        elif name == 'sym_div':
            if len(args) != 2:
                raise ValueError(f"sym_div requires 2 args, got {len(args)}")
            return _sym_simplify_div(args[0], args[1])

        elif name == 'sym_integrate':
            if len(args) != 2:
                raise ValueError(
                    f"sym_integrate requires 2 args (expr, var), got {len(args)}"
                )
            expr, var = args
            return _sym_integrate_helper(expr, str(var))

        elif name == 'sym_expand':
            if len(args) != 1:
                raise ValueError(f"sym_expand requires 1 arg, got {len(args)}")
            return _sym_expand(args[0])

        elif name == 'sym_coeff':
            if len(args) != 3:
                raise ValueError(
                    f"sym_coeff requires 3 args (expr, var, degree), got {len(args)}"
                )
            return _sym_coeff(args[0], str(args[1]), int(args[2]))

        # ----------------------------------------------------------------
        # Float arithmetic primitives (Gap 7)
        # ----------------------------------------------------------------

        elif name == 'float_num':
            if len(args) != 1:
                raise ValueError(f"float_num requires 1 arg, got {len(args)}")
            return float(args[0])

        elif name == 'float_sqrt':
            if len(args) != 1:
                raise ValueError(f"float_sqrt requires 1 arg, got {len(args)}")
            x = float(args[0])
            if x < 0:
                raise ValueError(f"float_sqrt: cannot take sqrt of negative {x}")
            return x ** 0.5

        elif name == 'float_abs':
            if len(args) != 1:
                raise ValueError(f"float_abs requires 1 arg, got {len(args)}")
            return abs(float(args[0]))

        elif name == 'float_exp':
            if len(args) != 1:
                raise ValueError(f"float_exp requires 1 arg, got {len(args)}")
            import math
            return math.exp(float(args[0]))

        elif name == 'float_add':
            if len(args) != 2:
                raise ValueError(f"float_add requires 2 args, got {len(args)}")
            return float(args[0]) + float(args[1])

        elif name == 'float_sub':
            if len(args) != 2:
                raise ValueError(f"float_sub requires 2 args, got {len(args)}")
            return float(args[0]) - float(args[1])

        elif name == 'float_mul':
            if len(args) != 2:
                raise ValueError(f"float_mul requires 2 args, got {len(args)}")
            return float(args[0]) * float(args[1])

        elif name == 'float_div':
            if len(args) != 2:
                raise ValueError(f"float_div requires 2 args, got {len(args)}")
            denom = float(args[1])
            if denom == 0.0:
                raise ValueError("float_div: division by zero")
            return float(args[0]) / denom

        elif name == 'float_log':
            if len(args) != 1:
                raise ValueError(f"float_log requires 1 arg, got {len(args)}")
            import math
            x = float(args[0])
            if x <= 0.0:
                raise ValueError(f"float_log: argument must be positive, got {x}")
            return math.log(x)

        elif name == 'float_pow':
            if len(args) != 2:
                raise ValueError(f"float_pow requires 2 args (x, n), got {len(args)}")
            return float(args[0]) ** int(args[1])

        elif name == 'float_pi':
            if len(args) != 0:
                raise ValueError(f"float_pi takes no args, got {len(args)}")
            import math
            return math.pi

        # ----------------------------------------------------------------
        # Inspection primitives — expose the expression tree structure
        # ----------------------------------------------------------------

        elif name == 'sym_tag':
            if len(args) != 1:
                raise ValueError(f"sym_tag requires 1 arg, got {len(args)}")
            e = args[0]
            if not isinstance(e, tuple) or not e:
                raise TypeError(f"sym_tag: expected symbolic expression, got {e!r}")
            return e[0]

        elif name == 'sym_lhs':
            if len(args) != 1:
                raise ValueError(f"sym_lhs requires 1 arg, got {len(args)}")
            e = args[0]
            if not isinstance(e, tuple) or len(e) < 2:
                raise TypeError(f"sym_lhs: expected binary node, got {e!r}")
            return e[1]

        elif name == 'sym_rhs':
            if len(args) != 1:
                raise ValueError(f"sym_rhs requires 1 arg, got {len(args)}")
            e = args[0]
            if not isinstance(e, tuple) or len(e) < 3:
                raise TypeError(f"sym_rhs: expected binary node with ≥2 children, got {e!r}")
            return e[2]

        elif name == 'sym_arg':
            if len(args) != 1:
                raise ValueError(f"sym_arg requires 1 arg, got {len(args)}")
            e = args[0]
            if not isinstance(e, tuple) or len(e) < 2:
                raise TypeError(f"sym_arg: expected unary node, got {e!r}")
            return e[1]

        elif name == 'sym_val':
            if len(args) != 1:
                raise ValueError(f"sym_val requires 1 arg, got {len(args)}")
            e = args[0]
            if not isinstance(e, tuple) or e[0] != 'NUM':
                raise TypeError(f"sym_val: expected NUM node, got {e!r}")
            return e[1]

        elif name == 'sym_name':
            if len(args) != 1:
                raise ValueError(f"sym_name requires 1 arg, got {len(args)}")
            e = args[0]
            if not isinstance(e, tuple) or e[0] != 'VAR':
                raise TypeError(f"sym_name: expected VAR node, got {e!r}")
            return e[1]

        elif name == 'sym_exp':
            if len(args) != 1:
                raise ValueError(f"sym_exp requires 1 arg, got {len(args)}")
            e = args[0]
            if not isinstance(e, tuple) or e[0] != 'POW':
                raise TypeError(f"sym_exp: expected POW node, got {e!r}")
            return e[2]

        # ----------------------------------------------------------------
        # Sequence primitives (Gap 4)
        # ----------------------------------------------------------------

        elif name == 'scan':
            if len(args) != 3:
                raise ValueError(
                    f"scan requires 3 args (n, init, step), got {len(args)}"
                )
            n, init, step_fn = args[0], args[1], args[2]
            n = int(n)
            if callable(step_fn):
                fn_callable = step_fn
            elif isinstance(step_fn, str):
                fn_callable = self.env.get(step_fn)
                if fn_callable is None or not callable(fn_callable):
                    raise ValueError(f"scan: {step_fn!r} is not a callable in env")
            else:
                raise TypeError(f"scan: expected callable or name, got {step_fn!r}")
            result = []
            state = init
            for _ in range(n):
                state = fn_callable(state)
                result.append(state)
            return result

        elif name == 'seq_get':
            if len(args) != 2:
                raise ValueError(f"seq_get requires 2 args (seq, i), got {len(args)}")
            seq, i = args[0], int(args[1])
            if not isinstance(seq, list):
                raise TypeError(f"seq_get: expected list, got {type(seq).__name__}")
            return seq[i]

        elif name == 'seq_len':
            if len(args) != 1:
                raise ValueError(f"seq_len requires 1 arg, got {len(args)}")
            seq = args[0]
            if not isinstance(seq, list):
                raise TypeError(f"seq_len: expected list, got {type(seq).__name__}")
            return len(seq)

        elif name == 'seq_cons':
            if len(args) != 2:
                raise ValueError(f"seq_cons requires 2 args (head, tail), got {len(args)}")
            head, tail = args[0], args[1]
            if not isinstance(tail, list):
                raise TypeError(f"seq_cons: tail must be list, got {type(tail).__name__}")
            return [head] + tail

        elif name == 'seq_nil':
            return []

        elif name == 'seq_head':
            if len(args) != 1:
                raise ValueError(f"seq_head requires 1 arg, got {len(args)}")
            seq = args[0]
            if not isinstance(seq, list) or len(seq) == 0:
                raise TypeError(f"seq_head: expected non-empty list")
            return seq[0]

        elif name == 'seq_tail':
            if len(args) != 1:
                raise ValueError(f"seq_tail requires 1 arg, got {len(args)}")
            seq = args[0]
            if not isinstance(seq, list) or len(seq) == 0:
                raise TypeError(f"seq_tail: expected non-empty list")
            return seq[1:]

        elif name in self.modality_fns:
            if self.dry_run and name in self.effectful_fns:
                raise DryRunError(name)
            return self.modality_fns[name](*args)

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
# Level C: symbolic expression helpers
# ---------------------------------------------------------------------------
# All expressions are tagged tuples: ('TAG', arg1, arg2, ...)
# Tags: NUM, VAR, ADD, SUB, MUL, POW, NEG


def _sym_simplify_add(e1: Any, e2: Any) -> Any:
    if isinstance(e1, tuple) and e1[0] == 'NUM' and isinstance(e2, tuple) and e2[0] == 'NUM':
        return ('NUM', e1[1] + e2[1])
    if isinstance(e1, tuple) and e1[0] == 'NUM' and e1[1] == 0:
        return e2
    if isinstance(e2, tuple) and e2[0] == 'NUM' and e2[1] == 0:
        return e1
    return ('ADD', e1, e2)


def _sym_simplify_sub(e1: Any, e2: Any) -> Any:
    if isinstance(e1, tuple) and e1[0] == 'NUM' and isinstance(e2, tuple) and e2[0] == 'NUM':
        return ('NUM', e1[1] - e2[1])
    if isinstance(e2, tuple) and e2[0] == 'NUM' and e2[1] == 0:
        return e1
    return ('SUB', e1, e2)


def _sym_simplify_mul(e1: Any, e2: Any) -> Any:
    if isinstance(e1, tuple) and e1[0] == 'NUM' and isinstance(e2, tuple) and e2[0] == 'NUM':
        return ('NUM', e1[1] * e2[1])
    if isinstance(e1, tuple) and e1[0] == 'NUM' and e1[1] == 0:
        return ('NUM', 0)
    if isinstance(e2, tuple) and e2[0] == 'NUM' and e2[1] == 0:
        return ('NUM', 0)
    if isinstance(e1, tuple) and e1[0] == 'NUM' and e1[1] == 1:
        return e2
    if isinstance(e2, tuple) and e2[0] == 'NUM' and e2[1] == 1:
        return e1
    # c * (expr / c) → expr   (cancellation for rational integration coefficients)
    if (isinstance(e1, tuple) and e1[0] == 'NUM'
            and isinstance(e2, tuple) and e2[0] == 'DIV'
            and isinstance(e2[2], tuple) and e2[2][0] == 'NUM'
            and e1[1] != 0 and e2[2][1] != 0 and e1[1] == e2[2][1]):
        return e2[1]
    # (expr / c) * c → expr
    if (isinstance(e2, tuple) and e2[0] == 'NUM'
            and isinstance(e1, tuple) and e1[0] == 'DIV'
            and isinstance(e1[2], tuple) and e1[2][0] == 'NUM'
            and e2[1] != 0 and e1[2][1] != 0 and e2[1] == e1[2][1]):
        return e1[1]
    return ('MUL', e1, e2)


def _sym_simplify_div(e1: Any, e2: Any) -> Any:
    """Division node with constant folding and exact-integer simplification."""
    if isinstance(e2, tuple) and e2[0] == 'NUM' and e2[1] == 1:
        return e1  # e / 1 = e
    if isinstance(e1, tuple) and e1[0] == 'NUM' and e1[1] == 0:
        return ('NUM', 0)  # 0 / e = 0
    if isinstance(e1, tuple) and e1[0] == 'NUM' and isinstance(e2, tuple) and e2[0] == 'NUM':
        if e2[1] != 0 and e1[1] % e2[1] == 0:
            return ('NUM', e1[1] // e2[1])  # exact integer division
    return ('DIV', e1, e2)


def _sym_simplify_pow(e: Any, n: int) -> Any:
    if n == 0:
        return ('NUM', 1)
    if n == 1:
        return e
    if isinstance(e, tuple) and e[0] == 'NUM':
        return ('NUM', e[1] ** n)
    return ('POW', e, n)


def _sym_eval_helper(expr: Any, bindings: dict) -> Any:
    """Evaluate a symbolic expression to a number given variable bindings."""
    if not isinstance(expr, tuple):
        return expr
    tag = expr[0]
    if tag == 'NUM':
        return expr[1]
    elif tag == 'VAR':
        name = expr[1]
        if name not in bindings:
            raise ValueError(f"sym_eval: unbound variable {name!r}")
        return bindings[name]
    elif tag == 'ADD':
        return _sym_eval_helper(expr[1], bindings) + _sym_eval_helper(expr[2], bindings)
    elif tag == 'SUB':
        return _sym_eval_helper(expr[1], bindings) - _sym_eval_helper(expr[2], bindings)
    elif tag == 'MUL':
        return _sym_eval_helper(expr[1], bindings) * _sym_eval_helper(expr[2], bindings)
    elif tag == 'POW':
        return _sym_eval_helper(expr[1], bindings) ** expr[2]
    elif tag == 'DIV':
        denom = _sym_eval_helper(expr[2], bindings)
        if denom == 0:
            raise ValueError("sym_eval: division by zero")
        return _sym_eval_helper(expr[1], bindings) / denom
    elif tag == 'NEG':
        return -_sym_eval_helper(expr[1], bindings)
    else:
        raise ValueError(f"sym_eval: unknown tag {tag!r}")


def _sym_diff_helper(expr: Any, var: str) -> Any:
    """Symbolic differentiation of expr with respect to var."""
    if not isinstance(expr, tuple):
        return ('NUM', 0)
    tag = expr[0]
    if tag == 'NUM':
        return ('NUM', 0)
    elif tag == 'VAR':
        return ('NUM', 1) if expr[1] == var else ('NUM', 0)
    elif tag == 'ADD':
        return _sym_simplify_add(
            _sym_diff_helper(expr[1], var),
            _sym_diff_helper(expr[2], var),
        )
    elif tag == 'SUB':
        return _sym_simplify_sub(
            _sym_diff_helper(expr[1], var),
            _sym_diff_helper(expr[2], var),
        )
    elif tag == 'MUL':
        # Product rule: (fg)' = f'g + fg'
        f, g = expr[1], expr[2]
        fpg = _sym_simplify_mul(_sym_diff_helper(f, var), g)
        fgp = _sym_simplify_mul(f, _sym_diff_helper(g, var))
        return _sym_simplify_add(fpg, fgp)
    elif tag == 'POW':
        # Power rule: (x^n)' = n * x^(n-1) * x'
        f, n = expr[1], expr[2]
        if not isinstance(n, int) or n == 0:
            return ('NUM', 0)
        coeff = _sym_simplify_mul(('NUM', n), _sym_diff_helper(f, var))
        base  = _sym_simplify_pow(f, n - 1)
        return _sym_simplify_mul(coeff, base)
    elif tag == 'DIV':
        # Quotient rule: (f/g)' = (f'g - fg') / g²
        f, g = expr[1], expr[2]
        fp = _sym_diff_helper(f, var)
        gp = _sym_diff_helper(g, var)
        numerator = _sym_simplify_sub(
            _sym_simplify_mul(fp, g),
            _sym_simplify_mul(f, gp),
        )
        denominator = _sym_simplify_pow(g, 2)
        return _sym_simplify_div(numerator, denominator)
    elif tag == 'NEG':
        d = _sym_diff_helper(expr[1], var)
        if isinstance(d, tuple) and d[0] == 'NUM':
            return ('NUM', -d[1])
        return ('NEG', d)
    else:
        return ('NUM', 0)


def _sym_subst_helper(expr: Any, var: str, replacement: Any) -> Any:
    """Replace ('VAR', var) nodes with replacement throughout expr."""
    if not isinstance(expr, tuple):
        return expr
    tag = expr[0]
    if tag == 'NUM':
        return expr
    elif tag == 'VAR':
        return replacement if expr[1] == var else expr
    elif tag in ('ADD', 'SUB', 'MUL', 'DIV'):
        e1 = _sym_subst_helper(expr[1], var, replacement)
        e2 = _sym_subst_helper(expr[2], var, replacement)
        if tag == 'ADD':
            return _sym_simplify_add(e1, e2)
        elif tag == 'SUB':
            return _sym_simplify_sub(e1, e2)
        elif tag == 'DIV':
            return _sym_simplify_div(e1, e2)
        else:
            return _sym_simplify_mul(e1, e2)
    elif tag == 'POW':
        e = _sym_subst_helper(expr[1], var, replacement)
        return _sym_simplify_pow(e, expr[2])
    elif tag == 'NEG':
        e = _sym_subst_helper(expr[1], var, replacement)
        if isinstance(e, tuple) and e[0] == 'NUM':
            return ('NUM', -e[1])
        return ('NEG', e)
    else:
        return expr


def _sym_str_helper(expr: Any) -> str:
    """Human-readable string for a symbolic expression."""
    if not isinstance(expr, tuple):
        return str(expr)
    tag = expr[0]
    if tag == 'NUM':
        return str(expr[1])
    elif tag == 'VAR':
        return expr[1]
    elif tag == 'ADD':
        return f'({_sym_str_helper(expr[1])} + {_sym_str_helper(expr[2])})'
    elif tag == 'SUB':
        return f'({_sym_str_helper(expr[1])} - {_sym_str_helper(expr[2])})'
    elif tag == 'MUL':
        return f'({_sym_str_helper(expr[1])} * {_sym_str_helper(expr[2])})'
    elif tag == 'DIV':
        return f'({_sym_str_helper(expr[1])}/{_sym_str_helper(expr[2])})'
    elif tag == 'POW':
        return f'({_sym_str_helper(expr[1])}^{expr[2]})'
    elif tag == 'NEG':
        return f'(-{_sym_str_helper(expr[1])})'
    else:
        return str(expr)


def _sym_expand(expr: Any) -> Any:
    """Distribute MUL over ADD/SUB (FOIL / distributivity).

    Expands expressions like (a+b)*(c+d) → ac + ad + bc + bd.
    Recurses until no more distributivity applies.
    Does NOT collect like terms — use _sym_coeff to read coefficients.
    """
    if not isinstance(expr, tuple):
        return expr
    tag = expr[0]
    if tag in ('NUM', 'VAR'):
        return expr
    if tag == 'ADD':
        return _sym_simplify_add(_sym_expand(expr[1]), _sym_expand(expr[2]))
    if tag == 'SUB':
        return _sym_simplify_sub(_sym_expand(expr[1]), _sym_expand(expr[2]))
    if tag == 'NEG':
        e = _sym_expand(expr[1])
        return ('NUM', -e[1]) if isinstance(e, tuple) and e[0] == 'NUM' else ('NEG', e)
    if tag == 'DIV':
        return _sym_simplify_div(_sym_expand(expr[1]), _sym_expand(expr[2]))
    if tag == 'POW':
        return _sym_simplify_pow(_sym_expand(expr[1]), expr[2])
    if tag == 'MUL':
        e1 = _sym_expand(expr[1])
        e2 = _sym_expand(expr[2])
        # Distribute e1 over e2
        if isinstance(e2, tuple) and e2[0] == 'ADD':
            return _sym_simplify_add(
                _sym_expand(_sym_simplify_mul(e1, e2[1])),
                _sym_expand(_sym_simplify_mul(e1, e2[2])),
            )
        if isinstance(e2, tuple) and e2[0] == 'SUB':
            return _sym_simplify_sub(
                _sym_expand(_sym_simplify_mul(e1, e2[1])),
                _sym_expand(_sym_simplify_mul(e1, e2[2])),
            )
        # Distribute e2 over e1
        if isinstance(e1, tuple) and e1[0] == 'ADD':
            return _sym_simplify_add(
                _sym_expand(_sym_simplify_mul(e1[1], e2)),
                _sym_expand(_sym_simplify_mul(e1[2], e2)),
            )
        if isinstance(e1, tuple) and e1[0] == 'SUB':
            return _sym_simplify_sub(
                _sym_expand(_sym_simplify_mul(e1[1], e2)),
                _sym_expand(_sym_simplify_mul(e1[2], e2)),
            )
        return _sym_simplify_mul(e1, e2)
    return expr


def _sym_coeff(expr: Any, var: str, degree: int) -> float:
    """Extract the coefficient of var^degree from a polynomial expression.

    Returns a float.  Assumes expr is in expanded (distributed) form.
    Example: _sym_coeff(3x²+5x-2, 'X', 2) → 3.0

    Works by recursing over ADD/SUB nodes and reading off MUL(scalar, POW(var,n))
    monomials.
    """
    if not isinstance(expr, tuple):
        return 0.0
    tag = expr[0]

    if tag == 'NUM':
        return float(expr[1]) if degree == 0 else 0.0

    elif tag == 'VAR':
        if expr[1] == var:
            return 1.0 if degree == 1 else 0.0
        return 0.0  # other variable treated as symbolic constant → 0

    elif tag == 'ADD':
        return _sym_coeff(expr[1], var, degree) + _sym_coeff(expr[2], var, degree)

    elif tag == 'SUB':
        return _sym_coeff(expr[1], var, degree) - _sym_coeff(expr[2], var, degree)

    elif tag == 'NEG':
        return -_sym_coeff(expr[1], var, degree)

    elif tag == 'MUL':
        f, g = expr[1], expr[2]
        f_const = not _sym_contains_var(f, var)
        g_const = not _sym_contains_var(g, var)
        if f_const:
            try:
                c = float(_sym_eval_helper(f, {}))
                return c * _sym_coeff(g, var, degree)
            except Exception:
                return 0.0
        if g_const:
            try:
                c = float(_sym_eval_helper(g, {}))
                return c * _sym_coeff(f, var, degree)
            except Exception:
                return 0.0
        return 0.0  # product where both factors contain var — not in expanded form

    elif tag == 'POW':
        base, n = expr[1], expr[2]
        if isinstance(base, tuple) and base[0] == 'VAR' and base[1] == var:
            return 1.0 if n == degree else 0.0
        if not _sym_contains_var(base, var) and degree == 0:
            try:
                return float(_sym_eval_helper(expr, {}))
            except Exception:
                return 0.0
        return 0.0

    elif tag == 'DIV':
        if not _sym_contains_var(expr[2], var):
            try:
                denom = float(_sym_eval_helper(expr[2], {}))
                return _sym_coeff(expr[1], var, degree) / denom
            except Exception:
                return 0.0
        return 0.0

    return 0.0


def _sym_contains_var(expr: Any, var: str) -> bool:
    """Return True if expr contains ('VAR', var) anywhere in the tree."""
    if not isinstance(expr, tuple):
        return False
    if expr[0] == 'VAR':
        return expr[1] == var
    return any(_sym_contains_var(child, var) for child in expr[1:] if isinstance(child, tuple))


def _sym_integrate_helper(expr: Any, var: str) -> Any:
    """Polynomial antiderivative of expr with respect to var (no constant of integration).

    Handles: constants, var itself, c*xⁿ, linear combinations (ADD/SUB),
    scalar multiples (MUL where one factor is constant w.r.t. var), NEG.
    Does NOT handle: products where both factors contain var (needs parts).
    """
    if not isinstance(expr, tuple):
        return ('NUM', 0)
    tag = expr[0]
    if tag == 'NUM':
        # ∫c dx = c*x
        if expr[1] == 0:
            return ('NUM', 0)
        if expr[1] == 1:
            return ('VAR', var)
        return _sym_simplify_mul(expr, ('VAR', var))
    elif tag == 'VAR':
        if expr[1] != var:
            # treat as constant w.r.t. var: ∫c dx = c*x
            return _sym_simplify_mul(expr, ('VAR', var))
        # ∫x dx = x²/2
        return _sym_simplify_div(_sym_simplify_pow(('VAR', var), 2), ('NUM', 2))
    elif tag == 'POW':
        base, n = expr[1], expr[2]
        if isinstance(base, tuple) and base[0] == 'VAR' and base[1] == var:
            # ∫xⁿ dx = xⁿ⁺¹/(n+1)
            return _sym_simplify_div(
                _sym_simplify_pow(('VAR', var), n + 1),
                ('NUM', n + 1),
            )
        # base doesn't contain var — treat whole expr as constant
        return _sym_simplify_mul(expr, ('VAR', var))
    elif tag == 'MUL':
        f, g = expr[1], expr[2]
        f_has_var = _sym_contains_var(f, var)
        g_has_var = _sym_contains_var(g, var)
        if not f_has_var:
            # ∫(c · f(x)) dx = c · ∫f(x) dx
            return _sym_simplify_mul(f, _sym_integrate_helper(g, var))
        if not g_has_var:
            return _sym_simplify_mul(g, _sym_integrate_helper(f, var))
        raise ValueError(
            f"sym_integrate: cannot integrate product where both factors contain "
            f"{var!r}: {_sym_str_helper(expr)}"
        )
    elif tag == 'DIV':
        # ∫(f/c) dx = (∫f dx)/c  when denominator is constant w.r.t. var
        if not _sym_contains_var(expr[2], var):
            return _sym_simplify_div(_sym_integrate_helper(expr[1], var), expr[2])
        raise ValueError(
            f"sym_integrate: cannot integrate rational expression with var in "
            f"denominator: {_sym_str_helper(expr)}"
        )
    elif tag == 'ADD':
        return _sym_simplify_add(
            _sym_integrate_helper(expr[1], var),
            _sym_integrate_helper(expr[2], var),
        )
    elif tag == 'SUB':
        return _sym_simplify_sub(
            _sym_integrate_helper(expr[1], var),
            _sym_integrate_helper(expr[2], var),
        )
    elif tag == 'NEG':
        inner = _sym_integrate_helper(expr[1], var)
        if isinstance(inner, tuple) and inner[0] == 'NUM':
            return ('NUM', -inner[1])
        return ('NEG', inner)
    else:
        raise ValueError(f"sym_integrate: unknown tag {tag!r} in {expr!r}")


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
        # Modality primitives registered via register_modality().
        # Keys are function names (e.g. 'img_dog'); values are callables.
        self._modality_fns: Dict[str, Callable] = {}
        # Phase D: names of effectful primitives (may not be called in dry-run mode).
        self._effectful_fns: Set[str] = set()
        # Internal flag set during run() when dry_run=True.
        self._dry_run: bool = False

    def register_modality(self, modality: 'Modality') -> None:  # type: ignore[name-defined]
        """Register all primitives from a Modality into the interpreter.

        If the modality exposes an EFFECTFUL class attribute (a set of
        primitive names), those names are marked as effectful and will
        raise DryRunError if called during synthesis (dry-run mode).

        After registration, process lines may call e.g. 'img = img_load(path)'
        and the interpreter will dispatch to the modality's implementation.

        A modality's primitives may shadow built-in names if there is a name
        collision, but that should not happen in practice because vision
        primitive names all start with 'img_'.
        """
        self._modality_fns.update(modality.primitives)
        # Register effectful names for dry-run protection (Phase D).
        if hasattr(modality, 'EFFECTFUL'):
            self._effectful_fns.update(modality.EFFECTFUL)

    def run(
        self,
        process_lines: List[str],
        inputs: tuple,
        input_type: List[str],
        dry_run: bool = False,
    ) -> tuple:
        """Execute process_lines with given inputs; return the emitted tuple.

        Args:
            dry_run: If True, effectful primitives (e.g. mc_attack) raise
                     DryRunError instead of executing.  Always True during
                     synthesis so that template testing never sends live
                     game commands.
        """
        self._dry_run = dry_run
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
            modality_fns=self._modality_fns,
            dry_run=self._dry_run,
            effectful_fns=self._effectful_fns,
        )
        return parser.parse_expr()
