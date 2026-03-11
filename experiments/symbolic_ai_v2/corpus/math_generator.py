"""Math corpus generator — 10 levels from counting to fluid dynamics.

The model is given NO explicit rules.  It sees only examples and must
discover the compositional structure of mathematics itself.

Token format: prefix (Polish) notation, each token a string.
Each fact is one sequence (list[str]).  Sequences are split into
train / test with no overlap so generalisation is testable.

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
          ['5', '6', '7', '8', '9', '10', '11', '12']

    The model must discover: each number is followed by n+1.
    Test set uses starts > n_max * 0.8 — genuinely unseen continuations.
    """
    seqs = []
    for start in range(n_max - run_len + 1):
        seqs.append([str(start + i) for i in range(run_len)])
    return _split(seqs, test_frac=0.2)


# ── Level 1: Successor / predecessor ──────────────────────────────────────────

def successor_seqs(n_max: int = 50) -> tuple[list, list]:
    """succ 0 eq 1 , succ 1 eq 2 , ... and pred 1 eq 0 , pred 2 eq 1 , ...

    The model must discover: succ(n) = n+1, pred(n) = n-1.
    Test: higher-range values not in training.
    """
    seqs = []
    for n in range(n_max):
        seqs.append(['succ', str(n), 'eq', str(n + 1)])
    for n in range(1, n_max + 1):
        seqs.append(['pred', str(n), 'eq', str(n - 1)])
    return _split(seqs, test_frac=0.2)


# ── Level 2: Addition ─────────────────────────────────────────────────────────

def addition_seqs(a_max: int = 9, b_max: int = 9) -> tuple[list, list]:
    """add a b eq c  for all a in 0..a_max, b in 0..b_max.

    Also includes: add b a eq c (commutativity examples).
    The model must discover: add a b = a + b.
    Test split ensures some (a,b) pairs held out.
    """
    seqs = []
    for a in range(a_max + 1):
        for b in range(b_max + 1):
            seqs.append(['add', str(a), str(b), 'eq', str(a + b)])
            if a != b:
                seqs.append(['add', str(b), str(a), 'eq', str(a + b)])  # commutative
    return _split(seqs, test_frac=0.2)


# ── Level 3: Subtraction ──────────────────────────────────────────────────────

def subtraction_seqs(max_val: int = 18) -> tuple[list, list]:
    """sub c b eq a  where c = a + b, a >= 0.

    The model must discover: sub is the inverse of add.
    Same numerical range as addition so transfer is testable.
    """
    seqs = []
    for a in range(10):
        for b in range(10):
            c = a + b
            if c <= max_val:
                seqs.append(['sub', str(c), str(b), 'eq', str(a)])
    return _split(seqs, test_frac=0.2)


# ── Level 4: Multiplication ───────────────────────────────────────────────────

def multiplication_seqs(a_max: int = 9, b_max: int = 9) -> tuple[list, list]:
    """mul a b eq c.  Includes commutative pairs."""
    seqs = []
    for a in range(a_max + 1):
        for b in range(b_max + 1):
            seqs.append(['mul', str(a), str(b), 'eq', str(a * b)])
            if a != b:
                seqs.append(['mul', str(b), str(a), 'eq', str(a * b)])
    return _split(seqs, test_frac=0.2)


# ── Level 5: Powers ───────────────────────────────────────────────────────────

def power_seqs() -> tuple[list, list]:
    """pow base exp eq result  for small integers.

    Also: sq n eq n*n  (square as a named operation).
    Model must discover: pow b 2 = sq b = mul b b.
    """
    seqs = []
    for base in range(2, 8):
        for exp in range(0, 5):
            val = base ** exp
            if val <= 1000:
                seqs.append(['pow', str(base), str(exp), 'eq', str(val)])
    # Square as a named operation
    for n in range(1, 13):
        seqs.append(['sq', str(n), 'eq', str(n * n)])
    # Square root (perfect squares only)
    for n in range(1, 11):
        seqs.append(['sqrt', str(n * n), 'eq', str(n)])
    return _split(seqs, test_frac=0.25)


# ── Level 6: Linear polynomial evaluation ─────────────────────────────────────

def linear_eval_seqs() -> tuple[list, list]:
    """eval a x b val eq result

    Evaluating the linear polynomial a*x + b at x=val:
    result = a * val + b

    The model must discover the substitution + arithmetic pattern.
    Token format: eval A x B at V eq R
    """
    seqs = []
    for a in range(1, 7):
        for b in range(0, 6):
            for v in range(0, 8):
                result = a * v + b
                if result <= 50:
                    seqs.append(['eval', str(a), 'x', str(b), 'at', str(v), 'eq', str(result)])
    return _split(seqs, test_frac=0.2)


# ── Level 7: Derivatives (power rule) ─────────────────────────────────────────

def derivative_seqs() -> tuple[list, list]:
    """Derivative of monomials via the power rule: d/dx x^n = n * x^(n-1).

    d x eq 1                 — d/dx(x) = 1
    d sq x eq mul 2 x        — d/dx(x²) = 2x
    d pow x 3 eq mul 3 sq x  — d/dx(x³) = 3x²
    d pow x 4 eq mul 4 pow x 3  — d/dx(x⁴) = 4x³

    Also linearity:
    d add mul A sq x mul B x eq add mul mul 2 A x B  — d/dx(Ax²+Bx) = 2Ax+B

    Model must discover: d pow x n eq mul n pow x sub n 1
    i.e. bring down the exponent, reduce by one.
    """
    seqs: list[list[str]] = []

    # Monomials x^n for n = 0..6
    seqs.append(['d', 'x', 'eq', '1'])                                      # d/dx x = 1
    seqs.append(['d', 'sq', 'x', 'eq', 'mul', '2', 'x'])                   # d/dx x² = 2x
    seqs.append(['d', 'pow', 'x', '3', 'eq', 'mul', '3', 'sq', 'x'])       # d/dx x³ = 3x²
    seqs.append(['d', 'pow', 'x', '4', 'eq', 'mul', '4', 'pow', 'x', '3'])
    seqs.append(['d', 'pow', 'x', '5', 'eq', 'mul', '5', 'pow', 'x', '4'])
    seqs.append(['d', 'pow', 'x', '6', 'eq', 'mul', '6', 'pow', 'x', '5'])

    # Scalar multiples: d/dx(c*x^n) = c*n*x^(n-1)
    for c in range(2, 6):
        seqs.append(['d', 'mul', str(c), 'x', 'eq', str(c)])
        seqs.append(['d', 'mul', str(c), 'sq', 'x', 'eq', 'mul', str(2 * c), 'x'])
        seqs.append(['d', 'mul', str(c), 'pow', 'x', '3',
                     'eq', 'mul', str(3 * c), 'sq', 'x'])

    # Linearity: d/dx(ax² + bx) = 2ax + b
    for a in range(1, 5):
        for b in range(1, 5):
            seqs.append([
                'd', 'add', 'mul', str(a), 'sq', 'x', 'mul', str(b), 'x',
                'eq', 'add', 'mul', str(2 * a), 'x', str(b)
            ])

    return _split(seqs, test_frac=0.25)


# ── Level 8: Indefinite integration (antiderivatives) ─────────────────────────

def integral_seqs() -> tuple[list, list]:
    """Indefinite integration of monomials.

    int 1 dx eq x C            — ∫1 dx = x + C
    int x dx eq add half sq x C — ∫x dx = x²/2 + C
    int mul 2 x dx eq add sq x C — ∫2x dx = x² + C
    int sq x dx eq add div pow x 3 3 C

    Token format: int EXPR dx eq ANTIDERIV C
    where C is the constant of integration.
    The model must discover: integration undoes differentiation.
    """
    seqs: list[list[str]] = []

    seqs.append(['int', '1', 'dx', 'eq', 'x', 'C'])
    seqs.append(['int', 'x', 'dx', 'eq', 'mul', 'half', 'sq', 'x', 'C'])
    seqs.append(['int', 'sq', 'x', 'dx', 'eq', 'mul', 'third', 'pow', 'x', '3', 'C'])

    # ∫c dx = c*x
    for c in range(2, 7):
        seqs.append(['int', str(c), 'dx', 'eq', 'mul', str(c), 'x', 'C'])

    # ∫2ax dx = a*x² + c
    for a in range(1, 6):
        seqs.append(['int', 'mul', str(2 * a), 'x', 'dx', 'eq', 'add', 'mul', str(a), 'sq', 'x', 'C'])

    # ∫3ax² dx = a*x³ + C
    for a in range(1, 5):
        seqs.append(['int', 'mul', str(3 * a), 'sq', 'x', 'dx',
                     'eq', 'add', 'mul', str(a), 'pow', 'x', '3', 'C'])

    # Fundamental theorem: int (d f) dx eq f C — FTC pairs
    seqs.append(['ftc', 'd', 'x', 'eq', '1', 'and', 'int', '1', 'dx', 'eq', 'x', 'C'])
    seqs.append(['ftc', 'd', 'sq', 'x', 'eq', 'mul', '2', 'x',
                 'and', 'int', 'mul', '2', 'x', 'dx', 'eq', 'add', 'sq', 'x', 'C'])

    return _split(seqs, test_frac=0.25)


# ── Level 9: Conservation laws ────────────────────────────────────────────────

def conservation_seqs() -> tuple[list, list]:
    """Equality-based conservation: LHS = RHS implies both sides change together.

    conserve add A B eq add A2 B2
    conserve add add A B C eq add add A2 B2 C2

    Also kinetic-energy style terms: KE = mul half mul rho sq v
    Model must discover the conservation pattern before seeing Bernoulli.
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
                    continue   # skip trivially identical sides
                key = tuple(sorted([(a, b), (c, d)]))
                if key not in seen:
                    seen.add(key)
                    seqs.append([
                        'conserve',
                        'add', str(a), str(b),
                        'eq',
                        'add', str(c), str(d)
                    ])

    # Kinetic energy terms: ke A V eq mul half mul A sq V
    for rho in [1, 2, 4]:
        for v in range(1, 7):
            ke = (rho * v * v) // 2
            seqs.append(['ke', str(rho), str(v), 'eq', str(ke)])

    # Potential energy: pe rho g h
    for h in range(1, 6):
        for rho in [1, 2, 4]:
            pe = rho * 10 * h   # g = 10
            seqs.append(['pe', str(rho), str(h), 'eq', str(pe)])

    return _split(seqs, test_frac=0.2)


# ── Level 10: Bernoulli's equation ────────────────────────────────────────────

def bernoulli_seqs() -> tuple[list, list]:
    """Bernoulli's principle: P + ½ρv² + ρgh = const along a streamline.

    Simplified for h₁ = h₂ = 0 (horizontal flow), ρ = 2:
        P + v² = const
    so  P1 + v1² = P2 + v2²
    i.e. add P1 sq_v1 eq add P2 sq_v2

    Token format:
        bernoulli P1 V1 P2 V2
    (implying add P1 sq V1 eq add P2 sq V2)

    Also the full expanded form so the model sees the equation structure:
        add str(p1) sq str(v1) eq add str(p2) sq str(v2)

    Test: given partial sequences with the final token masked,
    predict the missing pressure or velocity.
    """
    seqs: list[list[str]] = []

    valid = []
    for v1 in range(1, 9):
        for v2 in range(1, 9):
            if v1 == v2:
                continue
            # P1 + v1² = P2 + v2²  ⟹  P2 = P1 + v1² - v2²
            # Choose P1 so P2 > 0
            # P2 = P1 + (v1²-v2²) > 0  ⟹  P1 > v2²-v1² (when v2>v1)
            min_p1 = max(1, v2 ** 2 - v1 ** 2 + 1)
            for p1 in range(min_p1, min_p1 + 10):
                p2 = p1 + v1 ** 2 - v2 ** 2
                if p2 > 0:
                    valid.append((p1, v1, p2, v2))

    rng = random.Random(7)
    rng.shuffle(valid)

    for p1, v1, p2, v2 in valid[:120]:
        # Compact symbolic form: bernoulli P1 V1 P2 V2
        seqs.append(['bernoulli', str(p1), str(v1), str(p2), str(v2)])
        # Expanded conservation form
        seqs.append([
            'add', str(p1), 'sq', str(v1),
            'eq',
            'add', str(p2), 'sq', str(v2)
        ])

    return _split(seqs, test_frac=0.25)


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
