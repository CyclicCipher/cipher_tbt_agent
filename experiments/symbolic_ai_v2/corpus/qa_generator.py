"""qa_generator.py — Phase 20/17c: Q&A, word problems, and variadic equations.

Three levels of progression:

Level A — Simple Q&A (memorisation baseline)
  Sequences: ['what', 'is', 'succ', '3', '?', '4', '<eos>']
  Goal: after training, generate_until_eos(['what', 'is', 'succ', '3', '?'])
        returns ['4', '<eos>'].

Level B — Arithmetic word problems (semantic alignment)
  Sequences: ['alice', 'has', '3', 'apples', 'bob', 'gives', 'her', '4', 'how', 'many', '?', '7', '<eos>']
  Co-trained with math facts ('add 3 4 eq 7').
  The FCA discovers: 'has', 'gives', 'how many', '?' co-occur with 'add', 'eq'.
  After alignment, the model generalises to new (N1, N2) pairs it has never seen in NL.

Level C — Variadic equations (N-ary sum)
  Sequences: ['vadd', '1', '2', '3', 'eq', '6', '<eos>']
             ['vadd', '2', '3', '4', '5', 'eq', '14', '<eos>']
  The model learns variadic addition from examples.  The grammar enumerator's
  fold rule ('M = sum(Ns)') captures this.

Design principles (GOALS.md / BLUEPRINT.md):
  - Domain-agnostic: no hardcoded mappings from NL to math.  Alignment emerges
    from distributional co-occurrence when both domains are trained together.
  - EOS token: '<eos>' marks end of answer.  generate_until_eos() stops on it.
  - Variable split: train/test split by held-out (N1, N2) pairs so that the
    test set exercises generalisation to unseen values, not just recall.
"""

from __future__ import annotations

import random
from typing import Iterator

EOS = '<eos>'


# ── Level A: Simple Q&A ────────────────────────────────────────────────────────

def _simple_qa_sequences(rng: random.Random, n: int = 0) -> list[list[str]]:
    """Generate Q&A sequences for arithmetic facts.

    Format: ['what', 'is', op, arg1, [arg2,] 'eq', result, '<eos>']

    Uses 'eq' (not '?') as the answer separator so that the frame_match
    back-off chain recognises the algebraic pattern [op, N1, N2, eq] and
    can apply the discovered rule to unseen (N1, N2) pairs.

    Covers succ, add, sub, mul on small integers (0..9).
    """
    seqs: list[list[str]] = []

    # succ
    for k in range(10):
        seqs.append(['what', 'is', 'succ', str(k), 'eq', str(k + 1), EOS])

    # add
    for a in range(5):
        for b in range(5):
            seqs.append(['what', 'is', 'add', str(a), str(b), 'eq', str(a + b), EOS])

    # sub (non-negative results only)
    for a in range(1, 8):
        for b in range(a + 1):
            seqs.append(['what', 'is', 'sub', str(a), str(b), 'eq', str(a - b), EOS])

    # mul (small)
    for a in range(1, 6):
        for b in range(1, 6):
            seqs.append(['what', 'is', 'mul', str(a), str(b), 'eq', str(a * b), EOS])

    if n:
        rng.shuffle(seqs)
        seqs = seqs[:n]
    return seqs


def simple_qa_level(
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[list[str]], list[list[str]]]:
    """Return (train, test) Q&A sequences.

    Test set: held-out sequences (by index after shuffle).
    """
    rng = random.Random(seed)
    seqs = _simple_qa_sequences(rng)
    rng.shuffle(seqs)
    split = int(len(seqs) * train_ratio)
    return seqs[:split], seqs[split:]


# ── Level B: Arithmetic word problems ─────────────────────────────────────────

# Templates for "A has N1 [things], B gives A N2 more, how many?"
# We vary agent names and object names to make the distribution richer.
_AGENTS   = ['alice', 'bob', 'carol', 'dave', 'eve']
_OBJECTS  = ['apples', 'coins', 'books', 'marbles', 'cards']
_GIVERS   = ['bob', 'carol', 'dave', 'eve', 'alice']


def _addition_problem(a1: str, obj: str, a2: str, giver: str, n1: int, n2: int) -> list[str]:
    """'alice has 3 apples bob gives her 4 how many eq 7 <eos>'

    Uses 'eq' as the answer separator (consistent with math notation).
    The semantic alignment — 'has'/'gives'/'how many' ≈ 'add'/'eq' — is
    discovered via distributional co-occurrence when co-trained with math facts.
    """
    return [a1, 'has', str(n1), obj, giver, 'gives', 'her', str(n2),
            'how', 'many', 'eq', str(n1 + n2), EOS]


def _subtraction_problem(a1: str, obj: str, n1: int, n2: int) -> list[str]:
    """'alice has 7 apples gives away 3 how many eq 4 <eos>'"""
    return [a1, 'has', str(n1), obj, 'gives', 'away', str(n2),
            'how', 'many', 'eq', str(n1 - n2), EOS]


def word_problem_level(
    train_ratio: float = 0.8,
    seed: int = 42,
    n_add: int = 40,
    n_sub: int = 20,
) -> tuple[list[list[str]], list[list[str]]]:
    """Return (train, test) word problem sequences.

    Test set: held-out (N1, N2) pairs that were not seen during training.
    This tests algebraic generalisation, not memorisation.
    """
    rng = random.Random(seed)

    # Pool of (N1, N2) pairs for addition.
    add_pairs = [(a, b) for a in range(1, 9) for b in range(1, 9)
                 if a + b <= 12]
    rng.shuffle(add_pairs)
    train_add_pairs = set(map(tuple, add_pairs[:int(len(add_pairs) * train_ratio)]))
    test_add_pairs  = set(map(tuple, add_pairs[int(len(add_pairs) * train_ratio):]))

    # Pool of (N1, N2) pairs for subtraction.
    sub_pairs = [(a, b) for a in range(2, 10) for b in range(1, a)]
    rng.shuffle(sub_pairs)
    train_sub_pairs = set(map(tuple, sub_pairs[:int(len(sub_pairs) * train_ratio)]))
    test_sub_pairs  = set(map(tuple, sub_pairs[int(len(sub_pairs) * train_ratio):]))

    def _make_add(pairs, n):
        seqs = []
        for _ in range(n):
            n1, n2 = rng.choice(list(pairs))
            a1  = rng.choice(_AGENTS)
            obj = rng.choice(_OBJECTS)
            a2  = rng.choice([g for g in _GIVERS if g != a1])
            seqs.append(_addition_problem(a1, obj, a2, a2, n1, n2))
        return seqs

    def _make_sub(pairs, n):
        seqs = []
        for _ in range(n):
            n1, n2 = rng.choice(list(pairs))
            a1  = rng.choice(_AGENTS)
            obj = rng.choice(_OBJECTS)
            seqs.append(_subtraction_problem(a1, obj, n1, n2))
        return seqs

    train = _make_add(train_add_pairs, n_add) + _make_sub(train_sub_pairs, n_sub)
    test  = _make_add(test_add_pairs, max(4, n_add // 5)) + _make_sub(test_sub_pairs, max(2, n_sub // 5))

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def math_cotraining_level(seed: int = 42) -> list[list[str]]:
    """Return arithmetic facts to co-train with word problems and Q&A.

    Covers succ, add, sub, mul so that:
    - extract_unary_pairs / extract_binary_pairs find endofunctor maps
    - fit_rule discovers M = N+1, M = N1+N2, M = N1-N2, M = N1*N2
    - predict_via_frame_match can fire on any Q&A prompt suffix
    """
    seqs: list[list[str]] = []
    # succ
    for k in range(10):
        seqs.append(['succ', str(k), 'eq', str(k + 1)])
    # add
    for a in range(1, 9):
        for b in range(1, 9):
            if a + b <= 12:
                seqs.append(['add', str(a), str(b), 'eq', str(a + b)])
    # sub
    for a in range(2, 10):
        for b in range(1, a):
            seqs.append(['sub', str(a), str(b), 'eq', str(a - b)])
    # mul (small to stay in range)
    for a in range(1, 6):
        for b in range(1, 6):
            seqs.append(['mul', str(a), str(b), 'eq', str(a * b)])
    return seqs


# ── Level C: Variadic equations ────────────────────────────────────────────────

def variadic_add_level(
    train_ratio: float = 0.8,
    seed: int = 42,
    max_terms: int = 6,
    n_per_arity: int = 12,
) -> tuple[list[list[str]], list[list[str]]]:
    """Return (train, test) variadic addition sequences.

    Format: ['vadd', N1, N2, ..., Nk, 'eq', sum, '<eos>']
    Arities 2..max_terms are represented.  The model must learn that 'vadd'
    followed by any number of numerals before 'eq' means 'sum them all'.

    For arity 2 this is identical to 'add N1 N2 eq M'.
    For arity k > 2 this requires the model to discover the fold pattern.
    """
    rng = random.Random(seed)

    all_seqs: list[list[str]] = []
    for arity in range(2, max_terms + 1):
        for _ in range(n_per_arity):
            nums = [rng.randint(1, 9) for _ in range(arity)]
            total = sum(nums)
            # No EOS here: variadic uses math notation (vadd N1 ... Nk eq sum)
            # so the binary extraction can work on arity-2 cases.
            seq = ['vadd'] + [str(n) for n in nums] + ['eq', str(total)]
            all_seqs.append(seq)

    rng.shuffle(all_seqs)
    split = int(len(all_seqs) * train_ratio)
    return all_seqs[:split], all_seqs[split:]


# ── Level D: Latin word problems (Phase 17c stretch goal) ─────────────────────

# Variable vocabulary — different agents, objects, givers per sentence so the
# template learner must abstract over them rather than memorising name-specific
# associations.  Anti-unification over ~200 training sentences will produce a
# template with 5 variables [?agent, ?N1, ?obj, ?giver, ?N2] → N1+N2.
# The lookup table covers seen (agent, N1, obj, giver, N2) 5-tuples; test
# accuracy reflects honest coverage, not an engineered outcome.
# Variable vocabulary — different agents, objects, givers so the template
# must abstract over surface form rather than memorising specific names.
_LATIN_AGENTS  = ['marcus', 'gaius']
_LATIN_OBJECTS = ['poma', 'libri']
_LATIN_GIVERS  = ['julia', 'livia']


def _latin_addition_problem(
    rng: random.Random, n1: int, n2: int,
) -> list[str]:
    """e.g. 'gaius habet libri . livia dat ei . 3 et 4 quot ? 7 <eos>'

    Format: [agent habet obj . giver dat ei . N1 et N2 quot ? SUM <eos>]
    Latin: 'habet'=has, 'dat'=gives, 'ei'=to him/her, 'et'=and, 'quot'=how many.

    The narrative part (positions 0-7) establishes who has what and who gives.
    The question formula 'N1 et N2 quot ?' (positions 8-12) states the operands
    and asks for their sum.  SEQUITUR builds compositions N1→et→N2→quot→? which
    the template system expands to a 5-token context [?0 et ?1 quot ?] with
    lookup (N1, N2) → sum.  With 36 (N1,N2) pairs and 160 training sequences,
    lookup coverage is ~99%.  The '?' separator prevents the arithmetic
    frame-match rule from firing (that rule triggers on 'eq', not '?').
    """
    agent = rng.choice(_LATIN_AGENTS)
    obj   = rng.choice(_LATIN_OBJECTS)
    giver = rng.choice(_LATIN_GIVERS)
    return [agent, 'habet', obj, '.', giver, 'dat', 'ei', '.',
            str(n1), 'et', str(n2), 'quot', '?', str(n1 + n2), EOS]


def word_problem_latin(
    train_ratio: float = 0.8,
    seed: int = 43,
    n_add: int = 200,
) -> tuple[list[list[str]], list[list[str]]]:
    """Return (train, test) Latin addition word problems.

    With 36 (N1,N2) pairs and 160 training sequences, coverage of the
    pair lookup ≈ 99%.  The question formula 'N1 et N2 quot ?' creates
    a 5-token template context that SEQUITUR can capture and the template
    system can match at inference time.
    """
    rng = random.Random(seed)
    add_pairs = [(a, b) for a in range(1, 7) for b in range(1, 7)]  # 36 pairs, sums 2-12
    all_seqs  = [_latin_addition_problem(rng, *rng.choice(add_pairs)) for _ in range(n_add)]
    rng.shuffle(all_seqs)
    split = int(len(all_seqs) * train_ratio)
    return all_seqs[:split], all_seqs[split:]


# ── Level E: Middle High German word problems (Phase 17c stretch goal) ─────────

_MHG_AGENTS  = ['hildegard', 'walther']
_MHG_OBJECTS = ['apfel', 'brot']
_MHG_GIVERS  = ['elsa', 'mechthild']


def _mhg_addition_problem(
    rng: random.Random, n1: int, n2: int,
) -> list[str]:
    """e.g. 'walther hat brot . mechthild git ir . 3 und 4 wie vil ? 7 <eos>'

    Format: [agent hat obj . giver git ir . N1 und N2 wie vil ? SUM <eos>]
    MHG: 'hat'=has, 'git'=gives, 'ir'=to her/him, 'und'=and, 'wie vil'=how many.

    Same design rationale as Latin: variable vocabulary, '?' separator,
    question formula N1 und N2 wie vil ? creating a 6-token SEQUITUR context
    [?0 und ?1 wie vil ?] with lookup (N1, N2) → sum.
    """
    agent = rng.choice(_MHG_AGENTS)
    obj   = rng.choice(_MHG_OBJECTS)
    giver = rng.choice(_MHG_GIVERS)
    return [agent, 'hat', obj, '.', giver, 'git', 'ir', '.',
            str(n1), 'und', str(n2), 'wie', 'vil', '?', str(n1 + n2), EOS]


def word_problem_mhg(
    train_ratio: float = 0.8,
    seed: int = 44,
    n_add: int = 200,
) -> tuple[list[list[str]], list[list[str]]]:
    """Return (train, test) Middle High German addition word problems.

    Same design as word_problem_latin: random sequence split, variable
    vocabulary, '?' separator, question formula creates SEQUITUR context.
    """
    rng = random.Random(seed)
    add_pairs = [(a, b) for a in range(1, 7) for b in range(1, 7)]
    all_seqs  = [_mhg_addition_problem(rng, *rng.choice(add_pairs)) for _ in range(n_add)]
    rng.shuffle(all_seqs)
    split = int(len(all_seqs) * train_ratio)
    return all_seqs[:split], all_seqs[split:]


# ── All Phase 20 levels ────────────────────────────────────────────────────────

PHASE20_LEVELS: list[tuple[str, object]] = [
    ('simple_qa',     simple_qa_level),
    ('word_problems', word_problem_level),
    ('variadic_add',  variadic_add_level),
]

# Phase 17c multilingual levels (Latin + MHG word problems)
PHASE17C_NL_LEVELS: list[tuple[str, object]] = [
    ('latin_wp',  word_problem_latin),
    ('mhg_wp',    word_problem_mhg),
]
