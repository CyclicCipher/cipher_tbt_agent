"""Math corpus generator — 10 levels from counting to fluid dynamics.

The model is given NO explicit rules.  It sees only examples and must
discover the compositional structure of mathematics itself.

Token format: prefix (Polish) notation, each token a string.
Each fact is one sequence (list[str]).  Sequences are split into
train / test with no overlap so generalisation is testable.

ALL numeric values are tokenized as individual digit characters so that
the digit-level k-gram knowledge from the arithmetic corpus transfers
to every level.  Symbolic/operator tokens (e.g. 'pow', 'eq', 'x',
'bernoulli') remain as single tokens.

Level  0  counting        '0 1 2 3 4 5' runs — discovers successor chain
Level  1  successor       'succ 0 eq 1' facts — formalises the chain
Level  2  addition        'add 2 3 eq 5' — discovers add = iterated succ
Level  3  subtraction     'sub 5 2 eq 3' — discovers sub as inverse of add
Level  4  multiplication  'mul 3 4 eq 12' — discovers mul = iterated add
Level  5  powers          'pow 2 3 eq 8', 'sq 4 eq 16' — iterated mul
Level  6  linearity       linear poly eval: 'eval add mul A x B val eq R'
Level  7  derivative      'd_dx monomial' rules — discovers power rule
Level  8  integral        'int monomial dx eq antideriv' — discovers FTC
Level  9  conservation    'conserve LHS eq RHS' — discovers equality invariant
Level 10  bernoulli       Bernoulli's equation (ρ=2, h=0): sq v + P = const

Each level's generator returns (train_seqs, test_seqs) — lists of list[str].
"""

from __future__ import annotations

import random
from typing import Iterator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _digits(n: int) -> list[str]:
    """Split non-negative integer into a list of digit character strings.

    Examples:
        _digits(0)   -> ['0']
        _digits(5)   -> ['5']
        _digits(64)  -> ['6', '4']
        _digits(100) -> ['1', '0', '0']
    """
    return list(str(n))


def _zfill2(n: int) -> list[str]:
    """Zero-padded 2-digit representation (plain digits)."""
    return list(str(n).zfill(2))


def _split(seqs: list[list[str]], test_frac: float = 0.2, seed: int = 0) -> tuple[list, list]:
    rng = random.Random(seed)
    shuffled = seqs[:]
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * (1 - test_frac)))
    return shuffled[:cut], shuffled[cut:]


# ── Level 0: Counting ──────────────────────────────────────────────────────────

def counting_seqs(n_max: int = 30, run_len: int = 8) -> tuple[list, list]:
    """Short counting runs of length run_len starting at each position.

    e.g.  ['0', '1', '2', '3', '4', '5', '6', '7']
          ['5', '6', '7', '8', '9', '1', '0', '1', '1', '1', '2']

    Numbers are tokenized as digit characters.  The model must discover
    the digit-level successor pattern: each number is followed by n+1.
    """
    seqs = []
    for start in range(n_max - run_len + 1):
        seq = []
        for i in range(run_len):
            seq.extend(_digits(start + i))
        seqs.append(seq)
    return _split(seqs, test_frac=0.2)


# ── Level 1: Successor / predecessor ──────────────────────────────────────────

def successor_seqs(n_max: int = 50) -> tuple[list, list]:
    """succ 0 eq 1 , succ 1 eq 2 , ... and pred 1 eq 0 , pred 2 eq 1 , ...

    The model must discover: succ(n) = n+1, pred(n) = n-1.
    Numbers are tokenized as digit characters.
    """
    seqs = []
    for n in range(n_max):
        seqs.append(['succ'] + _digits(n) + ['eq'] + _digits(n + 1))
    for n in range(1, n_max + 1):
        seqs.append(['pred'] + _digits(n) + ['eq'] + _digits(n - 1))
    return _split(seqs, test_frac=0.2)


# ── Level 2: Addition ─────────────────────────────────────────────────────────

def addition_seqs(a_max: int = 9, b_max: int = 9) -> tuple[list, list]:
    """add a b eq c  for all a in 0..a_max, b in 0..b_max.

    Also includes: add b a eq c (commutativity examples).
    Numbers are tokenized as digit characters.
    """
    seqs = []
    for a in range(a_max + 1):
        for b in range(b_max + 1):
            seqs.append(['add'] + _digits(a) + _digits(b) + ['eq'] + _digits(a + b))
            if a != b:
                seqs.append(['add'] + _digits(b) + _digits(a) + ['eq'] + _digits(a + b))
    return _split(seqs, test_frac=0.2)


# ── Level 3: Subtraction ──────────────────────────────────────────────────────

def subtraction_seqs(max_val: int = 18) -> tuple[list, list]:
    """sub c b eq a  where c = a + b, a >= 0.

    Numbers are tokenized as digit characters.
    """
    seqs = []
    for a in range(10):
        for b in range(10):
            c = a + b
            if c <= max_val:
                seqs.append(['sub'] + _digits(c) + _digits(b) + ['eq'] + _digits(a))
    return _split(seqs, test_frac=0.2)


# ── Level 4: Multiplication ───────────────────────────────────────────────────

def multiplication_seqs(a_max: int = 9, b_max: int = 9) -> tuple[list, list]:
    """mul a b eq c.  Includes commutative pairs.
    Numbers are tokenized as digit characters.
    """
    seqs = []
    for a in range(a_max + 1):
        for b in range(b_max + 1):
            seqs.append(['mul'] + _digits(a) + _digits(b) + ['eq'] + _digits(a * b))
            if a != b:
                seqs.append(['mul'] + _digits(b) + _digits(a) + ['eq'] + _digits(a * b))
    return _split(seqs, test_frac=0.2)


# ── Level 5: Powers ───────────────────────────────────────────────────────────

def power_seqs() -> tuple[list, list]:
    """pow base exp eq result  for small integers.

    Also: sq n eq n*n  (square as a named operation).
    Numbers are tokenized as digit characters.
    """
    seqs = []
    for base in range(2, 8):
        for exp in range(0, 5):
            val = base ** exp
            if val <= 1000:
                seqs.append(['pow'] + _digits(base) + _digits(exp) + ['eq'] + _digits(val))
    # Square as a named operation
    for n in range(1, 13):
        seqs.append(['sq'] + _digits(n) + ['eq'] + _digits(n * n))
    # Square root (perfect squares only)
    for n in range(1, 11):
        seqs.append(['sqrt'] + _digits(n * n) + ['eq'] + _digits(n))
    return _split(seqs, test_frac=0.25)


# ── Level 6: Linear polynomial evaluation ─────────────────────────────────────

def linear_eval_seqs() -> tuple[list, list]:
    """eval a x b at v eq result

    Evaluating the linear polynomial a*x + b at x=val:
    result = a * val + b

    Numbers are tokenized as digit characters.
    Token format: eval A x B at V eq R
    """
    seqs = []
    for a in range(1, 7):
        for b in range(0, 6):
            for v in range(0, 8):
                result = a * v + b
                if result <= 50:
                    seqs.append(
                        ['eval'] + _digits(a) + ['x'] + _digits(b) +
                        ['at'] + _digits(v) + ['eq'] + _digits(result)
                    )
    return _split(seqs, test_frac=0.2)


# ── Level 7: Derivatives (power rule) ─────────────────────────────────────────

def derivative_seqs() -> tuple[list, list]:
    """Derivative of monomials via the power rule: d/dx x^n = n * x^(n-1).

    Numeric coefficients are tokenized as digit characters.
    """
    seqs: list[list[str]] = []

    # Monomials x^n for n = 0..6
    seqs.append(['d', 'x', 'eq', '1'])
    seqs.append(['d', 'sq', 'x', 'eq', 'mul', '2', 'x'])
    seqs.append(['d', 'pow', 'x', '3', 'eq', 'mul', '3', 'sq', 'x'])
    seqs.append(['d', 'pow', 'x', '4', 'eq', 'mul', '4', 'pow', 'x', '3'])
    seqs.append(['d', 'pow', 'x', '5', 'eq', 'mul', '5', 'pow', 'x', '4'])
    seqs.append(['d', 'pow', 'x', '6', 'eq', 'mul', '6', 'pow', 'x', '5'])

    # Scalar multiples: d/dx(c*x^n) = c*n*x^(n-1)
    for c in range(2, 6):
        seqs.append(['d', 'mul'] + _digits(c) + ['x', 'eq'] + _digits(c))
        seqs.append(['d', 'mul'] + _digits(c) + ['sq', 'x', 'eq', 'mul'] + _digits(2 * c) + ['x'])
        seqs.append(['d', 'mul'] + _digits(c) + ['pow', 'x', '3',
                     'eq', 'mul'] + _digits(3 * c) + ['sq', 'x'])

    # Linearity: d/dx(ax² + bx) = 2ax + b
    for a in range(1, 5):
        for b in range(1, 5):
            seqs.append(
                ['d', 'add', 'mul'] + _digits(a) + ['sq', 'x', 'mul'] + _digits(b) + ['x',
                 'eq', 'add', 'mul'] + _digits(2 * a) + ['x'] + _digits(b)
            )

    return _split(seqs, test_frac=0.25)


# ── Level 8: Indefinite integration (antiderivatives) ─────────────────────────

def integral_seqs() -> tuple[list, list]:
    """Indefinite integration of monomials.

    Numbers are tokenized as digit characters.
    Token format: int EXPR dx eq ANTIDERIV C
    """
    seqs: list[list[str]] = []

    seqs.append(['int', '1', 'dx', 'eq', 'x', 'C'])
    seqs.append(['int', 'x', 'dx', 'eq', 'mul', 'half', 'sq', 'x', 'C'])
    seqs.append(['int', 'sq', 'x', 'dx', 'eq', 'mul', 'third', 'pow', 'x', '3', 'C'])

    # ∫c dx = c*x
    for c in range(2, 7):
        seqs.append(['int'] + _digits(c) + ['dx', 'eq', 'mul'] + _digits(c) + ['x', 'C'])

    # ∫2ax dx = a*x² + c
    for a in range(1, 6):
        seqs.append(
            ['int', 'mul'] + _digits(2 * a) + ['x', 'dx', 'eq', 'add', 'mul'] +
            _digits(a) + ['sq', 'x', 'C']
        )

    # ∫3ax² dx = a*x³ + C
    for a in range(1, 5):
        seqs.append(
            ['int', 'mul'] + _digits(3 * a) + ['sq', 'x', 'dx',
             'eq', 'add', 'mul'] + _digits(a) + ['pow', 'x', '3', 'C']
        )

    # Fundamental theorem pairs
    seqs.append(['ftc', 'd', 'x', 'eq', '1', 'and', 'int', '1', 'dx', 'eq', 'x', 'C'])
    seqs.append(['ftc', 'd', 'sq', 'x', 'eq', 'mul', '2', 'x',
                 'and', 'int', 'mul', '2', 'x', 'dx', 'eq', 'add', 'sq', 'x', 'C'])

    return _split(seqs, test_frac=0.25)


# ── Level 9: Conservation laws ────────────────────────────────────────────────

def conservation_seqs() -> tuple[list, list]:
    """Equality-based conservation: LHS = RHS.

    Numbers are tokenized as digit characters.
    """
    seqs: list[list[str]] = []

    # Simple two-term conservation: a+b = c+d when a+b == c+d
    seen = set()
    for a in range(1, 8):
        for b in range(1, 8):
            s = a + b
            for c in range(1, s):
                d = s - c
                if (a == c and b == d) or (a == d and b == c):
                    continue
                key = tuple(sorted([(a, b), (c, d)]))
                if key not in seen:
                    seen.add(key)
                    seqs.append(
                        ['conserve', 'add'] + _digits(a) + _digits(b) +
                        ['eq', 'add'] + _digits(c) + _digits(d)
                    )

    # Kinetic energy terms: ke A V eq (A*V²)//2
    for rho in [1, 2, 4]:
        for v in range(1, 7):
            ke = (rho * v * v) // 2
            seqs.append(['ke'] + _digits(rho) + _digits(v) + ['eq'] + _digits(ke))

    # Potential energy: pe rho g h
    for h in range(1, 6):
        for rho in [1, 2, 4]:
            pe = rho * 10 * h   # g = 10
            seqs.append(['pe'] + _digits(rho) + _digits(h) + ['eq'] + _digits(pe))

    return _split(seqs, test_frac=0.2)


# ── Level 10: Bernoulli's equation ────────────────────────────────────────────

def bernoulli_seqs() -> tuple[list, list]:
    """Bernoulli's principle: P + ½ρv² + ρgh = const along a streamline.

    Simplified for h₁ = h₂ = 0 (horizontal flow), ρ = 2:
        P + v² = const
    so  P1 + v1² = P2 + v2²

    Numbers are tokenized as digit characters.
    Token format:
        bernoulli P1_digits V1_digits P2_digits V2_digits
    """
    seqs: list[list[str]] = []

    valid = []
    for v1 in range(1, 9):
        for v2 in range(1, 9):
            if v1 == v2:
                continue
            min_p1 = max(1, v2 ** 2 - v1 ** 2 + 1)
            for p1 in range(min_p1, min_p1 + 10):
                p2 = p1 + v1 ** 2 - v2 ** 2
                if p2 > 0:
                    valid.append((p1, v1, p2, v2))

    rng = random.Random(7)
    rng.shuffle(valid)

    for p1, v1, p2, v2 in valid[:120]:
        # Compact symbolic form
        seqs.append(
            ['bernoulli'] + _digits(p1) + _digits(v1) + _digits(p2) + _digits(v2)
        )
        # Expanded conservation form
        seqs.append(
            ['add'] + _digits(p1) + ['sq'] + _digits(v1) +
            ['eq', 'add'] + _digits(p2) + ['sq'] + _digits(v2)
        )

    return _split(seqs, test_frac=0.25)


# ── Phase D: Proof trace levels ───────────────────────────────────────────────

def power_trace_seqs() -> tuple[list, list]:
    """Proof traces for pow and sq: step shows b^k for k=1..e-1, ans shows b^e.

    Format (e >= 2):
        pow 4 3 step 4 step 1 6 ans 6 4 <eos>
    where step tokens mark intermediate accumulated powers (digit-tokenized).
    step/ans tokens act as unambiguous segment delimiters regardless of digit
    width — the model discovers their role from co-occurrence statistics.

    Trivial base cases (e=0,1) use only ans (no intermediate steps needed).
    """
    seqs: list[list[str]] = []

    for base in range(2, 8):
        for exp in range(0, 5):
            val = base ** exp
            if val > 1000:
                continue
            seq = ['pow'] + _digits(base) + _digits(exp)
            if exp == 0:
                seq += ['ans', '1']
            elif exp == 1:
                seq += ['ans'] + _digits(base)
            else:
                for k in range(1, exp):
                    seq += ['step'] + _digits(base ** k)
                seq += ['ans'] + _digits(val)
            seq += ['<eos>']
            seqs.append(seq)

    # sq n: zero-padded to 2 digits for unambiguous tokenisation
    for n in range(1, 13):
        n2 = _zfill2(n)
        seq = ['sq'] + n2 + ['step'] + n2 + ['ans'] + _digits(n * n) + ['<eos>']
        seqs.append(seq)

    return _split(seqs, test_frac=0.25, seed=5)


def linear_eval_trace_seqs() -> tuple[list, list]:
    """Proof traces for linear polynomial evaluation a*x+b at x=v.

    Format:
        eval 3 x 2 at 5 step 1 5 ans 1 7 <eos>
    One intermediate step showing a*v, then ans showing a*v+b.
    """
    seqs: list[list[str]] = []
    for a in range(1, 7):
        for b in range(0, 6):
            for v in range(0, 8):
                result = a * v + b
                if result > 50:
                    continue
                mul_step = a * v
                seq = (
                    ['eval'] + _digits(a) + ['x'] + _digits(b) +
                    ['at'] + _digits(v) +
                    ['step'] + _digits(mul_step) +
                    ['ans'] + _digits(result) +
                    ['<eos>']
                )
                seqs.append(seq)
    return _split(seqs, test_frac=0.2, seed=5)


def algebra_trace_seqs() -> tuple[list, list]:
    """Phase E: solve the linear equation A*x + B = C for x.

    Format:
        linsolve A B C step R1 ans X <eos>

    where R1 = C - B  (subtract constant from RHS)
    and   X  = R1 / A (divide by coefficient — always an integer here).

    The model must discover from examples that:
      - step gives R1 = C - B
      - ans  gives X  = R1 / A

    Construction: fix A ∈ 1..5, B ∈ 0..9, X ∈ 1..9, compute C = A*X + B.
    This guarantees R1 / A = X exactly (integer division).
    Filter: C ≤ 99 (at most 2 digits).
    """
    seqs: list[list[str]] = []
    for a in range(1, 6):          # coefficient
        for b in range(0, 10):     # constant term
            for x in range(1, 10): # answer
                c = a * x + b
                if c > 99:
                    continue
                r1 = c - b         # = a * x
                seq = (
                    ['linsolve'] + _digits(a) + _digits(b) + _digits(c) +
                    ['step'] + _digits(r1) +
                    ['ans'] + _digits(x) +
                    ['<eos>']
                )
                seqs.append(seq)
    return _split(seqs, test_frac=0.2, seed=7)


# ── Phase F: Multi-scenario conservation and Bernoulli ────────────────────────

def conservation_scenario_seqs() -> tuple[list, list]:
    """Phase F: four-scenario traces for the conservation law A + B = C + D.

    Each scenario expresses one unknown as a function of the other three.
    Operators cs4/cs3/cs2/cs1 indicate which position is being solved for:

        cs4 A B C  step <A+B>  ans D  <eos>   (find D = A+B-C)
        cs3 A B D  step <A+B>  ans C  <eos>   (find C = A+B-D)
        cs2 A C D  step <C+D>  ans B  <eos>   (find B = C+D-A)
        cs1 B C D  step <C+D>  ans A  <eos>   (find A = C+D-B)

    One intermediate step shows the sum of the known side; the ans completes.
    """
    seqs: list[list[str]] = []

    seen: set = set()
    for a in range(1, 9):
        for b in range(1, 9):
            s = a + b
            for c in range(1, s):
                d = s - c
                if (a == c and b == d) or (a == d and b == c):
                    continue
                key = (a, b, c, d)
                rev = (c, d, a, b)
                if key in seen or rev in seen:
                    continue
                seen.add(key)
                lhs = s   # = a + b = c + d

                seqs.append(
                    ['cs4'] + _digits(a) + _digits(b) + _zfill2(c) +
                    ['step'] + _zfill2(lhs) + ['ans'] + _zfill2(d) + ['<eos>']
                )
                seqs.append(
                    ['cs3'] + _digits(a) + _digits(b) + _zfill2(d) +
                    ['step'] + _zfill2(lhs) + ['ans'] + _zfill2(c) + ['<eos>']
                )
                seqs.append(
                    ['cs2'] + _digits(a) + _zfill2(c) + _zfill2(d) +
                    ['step'] + _zfill2(lhs) + ['ans'] + _digits(b) + ['<eos>']
                )
                seqs.append(
                    ['cs1'] + _digits(b) + _zfill2(c) + _zfill2(d) +
                    ['step'] + _zfill2(lhs) + ['ans'] + _digits(a) + ['<eos>']
                )

    return _split(seqs, test_frac=0.2, seed=9)


def bernoulli_trace_seqs() -> tuple[list, list]:
    """Phase F: two-scenario Bernoulli traces for P1 + v1² = P2 + v2².

    Simplified Bernoulli (ρ=2, h=0): P + v² = const along a streamline.

    Two scenarios:
        bern_p2 P1 V1 V2  step <V1²>  step <V2²>  ans P2  <eos>  (find P2)
        bern_p1 P2 V1 V2  step <V1²>  step <V2²>  ans P1  <eos>  (find P1)

    Two intermediate steps expose the squared velocities; the ans follows.
    P values are zero-padded to 2 digits; V values (1-7) are single-digit.
    Zero-padding P eliminates the k-gram boundary ambiguity that arises when
    P is single-digit: 'bern_p2 4 5 3' would otherwise be ambiguous between
    P=4,V1=5,V2=3 (→ step next) and P=45,V1=3,V2=? (→ digit next).
    With zero-padding: 'bern_p2 0 4 5 3' unambiguously means P=04=4, V1=5, V2=3.
    """
    seqs: list[list[str]] = []

    for v1 in range(1, 8):
        for v2 in range(1, 8):
            if v1 == v2:
                continue
            v1sq = v1 * v1
            v2sq = v2 * v2
            min_p1 = max(1, v2sq - v1sq + 1)
            for p1 in range(min_p1, min_p1 + 8):
                p2 = p1 + v1sq - v2sq
                if p2 <= 0 or p2 > 99:
                    continue
                seqs.append(
                    ['bern_p2'] + _zfill2(p1) + _digits(v1) + _digits(v2) +
                    ['step'] + _digits(v1sq) +
                    ['step'] + _digits(v2sq) +
                    ['ans'] + _zfill2(p2) + ['<eos>']
                )
                seqs.append(
                    ['bern_p1'] + _zfill2(p2) + _digits(v1) + _digits(v2) +
                    ['step'] + _digits(v1sq) +
                    ['step'] + _digits(v2sq) +
                    ['ans'] + _zfill2(p1) + ['<eos>']
                )

    return _split(seqs, test_frac=0.25, seed=9)


# ── Phase G: Expanded calculus traces ─────────────────────────────────────────

def derivative_trace_seqs() -> tuple[list, list]:
    """Phase G: power-rule proof traces for d/dx(c * x^n) = (c*n) * x^(n-1).

    Format (same input tokens as derivative_seqs, eq replaced with step/ans):
        d [mul C] [x | sq x | pow x N]  step <c*n>  ans <result>  <eos>

    The step shows the numeric new coefficient c*n; the ans shows the symbolic
    result in the notation used by derivative_seqs (e.g. 'mul 6 x').
    """
    seqs: list[list[str]] = []

    def input_toks(c: int, n: int) -> list[str]:
        if c == 1:
            if n == 1:   return ['d', 'x']
            if n == 2:   return ['d', 'sq', 'x']
            return ['d', 'pow', 'x'] + _digits(n)
        else:
            if n == 1:   return ['d', 'mul'] + _digits(c) + ['x']
            if n == 2:   return ['d', 'mul'] + _digits(c) + ['sq', 'x']
            return ['d', 'mul'] + _digits(c) + ['pow', 'x'] + _digits(n)

    def result_toks(cn: int, n: int) -> list[str]:
        """Tokens for (cn) * x^(n-1)."""
        if n == 1:   return _digits(cn)            # just a constant
        if n == 2:   return ['mul'] + _digits(cn) + ['x']
        if n == 3:   return ['mul'] + _digits(cn) + ['sq', 'x']
        return ['mul'] + _digits(cn) + ['pow', 'x'] + _digits(n - 1)

    for c in range(1, 6):
        for n in range(1, 6):
            cn = c * n
            if cn > 25:
                continue
            seq = input_toks(c, n) + ['step'] + _digits(cn) + ['ans'] + result_toks(cn, n) + ['<eos>']
            seqs.append(seq)

    return _split(seqs, test_frac=0.2, seed=11)


def integral_trace_seqs() -> tuple[list, list]:
    """Phase G: reverse-power-rule traces for ∫ R*M * x^(M-1) dx = R * x^M.

    Parametrised by result coefficient R ∈ 1..5 and result exponent M ∈ 1..4
    so that the integrand coefficient R*M is always an integer.

    Format:
        int [mul <R*M>] [x | sq x | pow x (M-1)] dx  step R  ans mul R [x | ...] <eos>

    The step shows R (result coefficient); ans shows the antiderivative.
    """
    seqs: list[list[str]] = []

    def integrand_toks(rm: int, m: int) -> list[str]:
        if m == 1:   return ['int'] + _digits(rm) + ['dx']
        if m == 2:   return ['int', 'mul'] + _digits(rm) + ['x', 'dx']
        if m == 3:   return ['int', 'mul'] + _digits(rm) + ['sq', 'x', 'dx']
        return ['int', 'mul'] + _digits(rm) + ['pow', 'x'] + _digits(m - 1) + ['dx']

    def result_toks(r: int, m: int) -> list[str]:
        if m == 1:   return ['mul'] + _digits(r) + ['x']
        if m == 2:   return ['mul'] + _digits(r) + ['sq', 'x']
        return ['mul'] + _digits(r) + ['pow', 'x'] + _digits(m)

    for r in range(1, 6):
        for m in range(1, 5):
            rm = r * m
            if rm > 20:
                continue
            seq = integrand_toks(rm, m) + ['step'] + _digits(r) + ['ans'] + result_toks(r, m) + ['<eos>']
            seqs.append(seq)

    return _split(seqs, test_frac=0.2, seed=13)


# ── All levels ─────────────────────────────────────────────────────────────────

LEVELS: list[tuple[str, callable]] = [
    ('counting',      counting_seqs),
    ('successor',     successor_seqs),
    ('addition',      addition_seqs),
    ('subtraction',   subtraction_seqs),
    ('multiplication', multiplication_seqs),
    ('powers',        power_seqs),
    ('linear_eval',   linear_eval_seqs),
    ('derivatives',   derivative_seqs),
    ('integrals',     integral_seqs),
    ('conservation',  conservation_seqs),
    ('bernoulli',     bernoulli_seqs),
]


def all_levels() -> Iterator[tuple[str, list[list[str]], list[list[str]]]]:
    """Yield (name, train_seqs, test_seqs) for each level."""
    for name, gen_fn in LEVELS:
        train, test = gen_fn()
        yield name, train, test


if __name__ == '__main__':
    # Quick sanity check: print sample seqs from each level
    for name, train, test in all_levels():
        print(f"\n{'='*60}")
        print(f"Level: {name}  ({len(train)} train, {len(test)} test)")
        print("  Samples:")
        for seq in train[:3]:
            print(f"    {' '.join(seq)}")
        print("  Test samples:")
        for seq in test[:2]:
            print(f"    {' '.join(seq)}")
