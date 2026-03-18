"""digit_math_generator.py — Digit-level tokenised arithmetic sequences.

Numbers are tokenised as sequences of individual digit characters rather than
atomic number strings.  This lets the model generalise to numbers outside the
training range by learning the positional structure of decimal notation.

Two tokenisation formats are provided:

──────────────────────────────────────────────────────────────────────────────
Format 1 — Flat (MSB-first input, MSB-first output):

    succ_seq(211)  = ['succ', '2', '1', '1', 'eq', '2', '1', '2', '<eos>']
    pred_seq(212)  = ['pred', '2', '1', '2', 'eq', '2', '1', '1', '<eos>']
    succ_seq(99)   = ['succ', '9', '9', 'eq', '1', '0', '0', '<eos>']

The carry pattern '...X 9 eq ... X+1 0' is visible in k-grams, but the
context window must span all n digits of the input to be position-correct,
making generalisation beyond the training magnitude range unreliable.

──────────────────────────────────────────────────────────────────────────────
Format 2 — Interleaved LSB-first with position markers (the working approach):

    succ_seq_interleaved(211, n_pairs=4)
        = ['succ','eq','i','1','o','2','i','1','o','1','i','2','o','2','i','0','o','0','<eos>']
          (pairs: units→(1,2), tens→(1,1), hundreds→(2,2), thousands→(0,0))

Input/output digits are interleaved as ('i', in_i, 'o', out_i) quads, LSB-first,
padded to n_pairs positions with leading-zero quads.  During guided generation:

    buf = [op, 'eq']
    for each input digit in_i (LSB-first):
        buf += ['i', in_i, 'o']
        out_i = argmax sp.predict(buf)
        buf.append(out_i)

The 'i'/'o' markers eliminate the context collision that arises in the unmarked
format when the same digit triple (A, B, C) appears both predicting an INPUT
token and predicting an OUTPUT token with different targets.

Key k=6 contexts (independent of magnitude):
  ('succ','eq','i', D,'o')  → (D+1)%10           # units, carry-in = 1
  ('pred','eq','i', D,'o')  → (D-1+10)%10        # units, borrow-in = 1
  (9,'o',0,'i', D,'o')      → (D+1)%10           # carry propagates
  (0,'o',9,'i', D,'o')      → (D-1+10)%10        # borrow propagates
  (D,'o',D+1,'i',E,'o')     → E                  # carry absorbed; copy E
  (D,'o', D,'i',E,'o')      → E                  # no carry/borrow; copy E
  (0,'o',0,'i',E,'o')       → E                  # padding copy

These rules fully determine succ/pred for any magnitude, and are ALL present
in training data for 0–99 padded to n_pairs=6.
"""

from __future__ import annotations

import random


# ── Primitive helpers ─────────────────────────────────────────────────────────

def digits(n: int) -> list[str]:
    """Split a non-negative integer into a list of decimal digit characters.

    Examples
    --------
    digits(0)   → ['0']
    digits(9)   → ['9']
    digits(42)  → ['4', '2']
    digits(211) → ['2', '1', '1']
    """
    assert n >= 0, f"digits() requires n >= 0, got {n}"
    return list(str(n))


def succ_seq(n: int) -> list[str]:
    """Digit-level successor sequence ending with '<eos>'.

    Format: ['succ'] + digits(n) + ['eq'] + digits(n+1) + ['<eos>']
    """
    return ['succ'] + digits(n) + ['eq'] + digits(n + 1) + ['<eos>']


def pred_seq(n: int) -> list[str]:
    """Digit-level predecessor sequence ending with '<eos>'.

    Format: ['pred'] + digits(n) + ['eq'] + digits(n-1) + ['<eos>']
    Requires n >= 1 (predecessor of 0 is undefined in ℕ).
    """
    assert n >= 1, f"pred_seq() requires n >= 1, got {n}"
    return ['pred'] + digits(n) + ['eq'] + digits(n - 1) + ['<eos>']


# ── Interleaved LSB-first sequences ──────────────────────────────────────────

def succ_seq_interleaved(n: int, n_pairs: int = 6) -> list[str]:
    """Interleaved LSB-first successor sequence with position markers.

    Format: ['succ', 'eq', 'i', in_0, 'o', out_0, 'i', in_1, 'o', out_1, ...,
             'i', in_{n_pairs-1}, 'o', out_{n_pairs-1}, '<eos>']

    The 'i' and 'o' marker tokens precede each input and output digit
    respectively.  They eliminate context collisions that occur in the
    unmarked format when the same digit triple appears both predicting an
    input token and predicting an output token with conflicting targets.

    Padding to n_pairs ≥ max_input_digits + 1 ensures:
      • EOS follows only all-zero padding, never a mid-sequence quad.
      • All carry and copy patterns appear in training.
      • Guided generation generalises to any input length ≤ n_pairs.

    Parameters
    ----------
    n       : input integer (n >= 0)
    n_pairs : number of ('i', in, 'o', out) quads; must be ≥
              max(len(str(n)), len(str(n+1))).

    Example
    -------
    succ_seq_interleaved(211, n_pairs=4)
        → ['succ','eq','i','1','o','2','i','1','o','1','i','2','o','2',
           'i','0','o','0','<eos>']
    """
    n_out = n + 1
    min_pairs = max(len(str(n)), len(str(n_out)))
    assert n_pairs >= min_pairs, (
        f"n_pairs={n_pairs} is too small for succ({n})={n_out} "
        f"(need ≥ {min_pairs})"
    )
    inp_lsb = list(str(n).zfill(n_pairs))[::-1]      # LSB-first, zero-padded
    out_lsb = list(str(n_out).zfill(n_pairs))[::-1]
    tokens: list[str] = ['succ', 'eq']
    for i, o in zip(inp_lsb, out_lsb):
        tokens.append('i')
        tokens.append(i)
        tokens.append('o')
        tokens.append(o)
    tokens.append('<eos>')
    return tokens


def pred_seq_interleaved(n: int, n_pairs: int = 6) -> list[str]:
    """Interleaved LSB-first predecessor sequence with position markers.

    Same format as succ_seq_interleaved but for pred(n) = n - 1.
    Requires n >= 1 (predecessor of 0 is undefined in N).
    """
    assert n >= 1, f"pred_seq_interleaved() requires n >= 1, got {n}"
    n_out = n - 1
    min_pairs = max(len(str(n)), len(str(n_out)) if n_out > 0 else 1)
    assert n_pairs >= min_pairs, (
        f"n_pairs={n_pairs} is too small for pred({n})={n_out} "
        f"(need ≥ {min_pairs})"
    )
    inp_lsb = list(str(n).zfill(n_pairs))[::-1]
    out_lsb = list(str(n_out).zfill(n_pairs))[::-1]
    tokens: list[str] = ['pred', 'eq']
    for i, o in zip(inp_lsb, out_lsb):
        tokens.append('i')
        tokens.append(i)
        tokens.append('o')
        tokens.append(o)
    tokens.append('<eos>')
    return tokens


def digit_succ_pred_split_interleaved(
    train_max: int = 99,
    test_min:  int = 100,
    test_max:  int = 999,
    *,
    n_pairs:   int = 6,
    seed:      int = 42,
) -> tuple[str, list[list[str]], list[list[str]]]:
    """Interleaved-format train/test split for digit-level succ/pred.

    All sequences are padded to n_pairs (input, output) digit pairs, which
    must satisfy n_pairs >= len(str(test_max + 1)) + 1 for full coverage.
    Setting n_pairs = 6 covers training 0–99 and testing up to 5-digit numbers
    (test_max ≤ 99999).

    Parameters
    ----------
    train_max : largest number in the training set (inclusive)
    test_min  : smallest number in the test set (inclusive)
    test_max  : largest number in the test set (inclusive)
    n_pairs   : fixed number of (in, out) digit pairs per sequence
    seed      : random seed for training-set shuffle

    Returns
    -------
    (name, train_seqs, test_seqs)
    """
    assert train_max < test_min, (
        f"train_max ({train_max}) must be < test_min ({test_min})"
    )

    train: list[list[str]] = []
    for n in range(0, train_max + 1):
        train.append(succ_seq_interleaved(n, n_pairs=n_pairs))
    for n in range(1, train_max + 2):
        train.append(pred_seq_interleaved(n, n_pairs=n_pairs))

    rng = random.Random(seed)
    rng.shuffle(train)

    test: list[list[str]] = []
    for n in range(test_min, test_max + 1):
        test.append(succ_seq_interleaved(n, n_pairs=n_pairs))
    for n in range(test_min + 1, test_max + 2):
        test.append(pred_seq_interleaved(n, n_pairs=n_pairs))

    name = (
        f"digit_succ_pred_interleaved "
        f"(train 0-{train_max}, test {test_min}-{test_max}, n_pairs={n_pairs})"
    )
    return name, train, test


# ── Train / test split generator ──────────────────────────────────────────────

def digit_succ_pred_split(
    train_max: int = 99,
    test_min:  int = 100,
    test_max:  int = 999,
    *,
    seed:      int = 42,
) -> tuple[str, list[list[str]], list[list[str]]]:
    """Generate train and test digit-level succ/pred sequences.

    Training range : succ(0 .. train_max)   and  pred(1 .. train_max + 1)
    Test range     : succ(test_min .. test_max)  and  pred(test_min + 1 .. test_max + 1)

    The train and test ranges are disjoint by construction.  The training set
    is shuffled so that the Hankel estimator sees the patterns in mixed order.

    Parameters
    ----------
    train_max : largest number in the training set (inclusive)
    test_min  : smallest number in the test set (inclusive)
    test_max  : largest number in the test set (inclusive)
    seed      : random seed for training-set shuffle

    Returns
    -------
    (name, train_seqs, test_seqs)
    """
    assert train_max < test_min, (
        f"train_max ({train_max}) must be < test_min ({test_min}) "
        f"to keep train and test disjoint."
    )

    train: list[list[str]] = []
    for n in range(0, train_max + 1):
        train.append(succ_seq(n))
    for n in range(1, train_max + 2):
        train.append(pred_seq(n))

    rng = random.Random(seed)
    rng.shuffle(train)

    test: list[list[str]] = []
    for n in range(test_min, test_max + 1):
        test.append(succ_seq(n))
    for n in range(test_min + 1, test_max + 2):
        test.append(pred_seq(n))

    name = (
        f"digit_succ_pred "
        f"(train 0–{train_max}, test {test_min}–{test_max})"
    )
    return name, train, test


# ── Binary-operation interleaved sequences ────────────────────────────────────
#
# Format for two-operand operations (add, sub, mul):
#
#   [op, 'eq', 'i', a_0, 'j', b_0, 'o', c_0, 'i', a_1, 'j', b_1, 'o', c_1,
#    ..., 'i', a_{n-1}, 'j', b_{n-1}, 'o', c_{n-1}, '<eos>']
#
# a_i, b_i are the i-th digits (LSB-first) of the two operands; c_i is the
# corresponding output digit.  All three are zero-padded to n_pairs.
#
# Key k=10 context at position j >= 1 (encodes full carry/borrow state):
#   (a_{j-1}, 'j', b_{j-1}, 'o', c_{j-1}, 'i', a_j, 'j', b_j, 'o') → c_j
#
# At j=0 the k=7 context ('op', 'eq', 'i', a_0, 'j', b_0, 'o') is unique
# for each (a_0, b_0) pair and maps directly to (a_0 op b_0) mod 10.
#
# Guided generation feeds (a_j, b_j) at each step and reads the predicted c_j.


def add_seq_interleaved(a: int, b: int, n_pairs: int = 6) -> list[str]:
    """Interleaved LSB-first addition sequence with position markers.

    Format: ['add','eq','i',a_0,'j',b_0,'oa',c_0,...,'<eos>']

    Uses output marker 'oa' (add-output) rather than the generic 'o' used by
    succ/pred.  This prevents context collisions: add(5,5)=10 and sub(5,5)=0
    share the same (a,b,c_0) digits at position j=1 but differ in operator,
    so they must use different output markers to disambiguate the k=10 carry
    context.

    n_pairs must be >= max(len(str(a)), len(str(b)), len(str(a+b))).
    """
    c = a + b
    min_pairs = max(len(str(a)), len(str(b)), len(str(c)))
    assert n_pairs >= min_pairs, (
        f"n_pairs={n_pairs} too small for add({a},{b})={c} (need >= {min_pairs})"
    )
    a_lsb = list(str(a).zfill(n_pairs))[::-1]
    b_lsb = list(str(b).zfill(n_pairs))[::-1]
    c_lsb = list(str(c).zfill(n_pairs))[::-1]
    tokens: list[str] = ['add', 'eq']
    for ai, bi, ci in zip(a_lsb, b_lsb, c_lsb):
        tokens += ['i', ai, 'j', bi, 'oa', ci]
    tokens.append('<eos>')
    return tokens


def sub_seq_interleaved(a: int, b: int, n_pairs: int = 6) -> list[str]:
    """Interleaved LSB-first subtraction sequence with position markers.

    Format: ['sub','eq','i',a_0,'j',b_0,'os',c_0,...,'<eos>']

    Uses output marker 'os' (sub-output) to avoid context collisions with
    add (which uses 'oa').  Requires a >= b (natural-number subtraction).
    """
    assert a >= b, f"sub_seq_interleaved requires a >= b, got sub({a},{b})"
    c = a - b
    min_pairs = max(len(str(a)), len(str(b)), len(str(c)) if c > 0 else 1)
    assert n_pairs >= min_pairs, (
        f"n_pairs={n_pairs} too small for sub({a},{b})={c} (need >= {min_pairs})"
    )
    a_lsb = list(str(a).zfill(n_pairs))[::-1]
    b_lsb = list(str(b).zfill(n_pairs))[::-1]
    c_lsb = list(str(c).zfill(n_pairs))[::-1]
    tokens: list[str] = ['sub', 'eq']
    for ai, bi, ci in zip(a_lsb, b_lsb, c_lsb):
        tokens += ['i', ai, 'j', bi, 'os', ci]
    tokens.append('<eos>')
    return tokens


def mul_seq_interleaved(a: int, b: int, n_pairs: int = 4) -> list[str]:
    """Interleaved LSB-first multiplication sequence with position markers.

    Format: ['mul','eq','i',a_0,'j',b_0,'om',c_0,...,'<eos>']

    Uses output marker 'om' (mul-output) to avoid collisions with add/sub.
    For single-digit operands (a, b in 0–9) the result fits in 2 digits;
    n_pairs=4 provides two overflow slots.  Multi-digit multiplication is not
    decomposable position-by-position like add/sub, so this encoding treats
    each (a,b) pair as a direct lookup: the k=7 context
    ('mul','eq','i',a,'j',b,'om') uniquely determines c_0 for each (a,b).
    """
    c = a * b
    min_pairs = max(len(str(a)), len(str(b)), len(str(c)) if c > 0 else 1)
    assert n_pairs >= min_pairs, (
        f"n_pairs={n_pairs} too small for mul({a},{b})={c} (need >= {min_pairs})"
    )
    a_lsb = list(str(a).zfill(n_pairs))[::-1]
    b_lsb = list(str(b).zfill(n_pairs))[::-1]
    c_lsb = list(str(c).zfill(n_pairs))[::-1]
    tokens: list[str] = ['mul', 'eq']
    for ai, bi, ci in zip(a_lsb, b_lsb, c_lsb):
        tokens += ['i', ai, 'j', bi, 'om', ci]
    tokens.append('<eos>')
    return tokens


def digit_arithmetic_corpus(
    succ_pred_max: int = 999,
    add_sub_max:   int = 99,
    *,
    n_pairs: int = 7,
    seed:    int = 42,
) -> list[list[str]]:
    """Combined digit-level arithmetic training corpus.

    Includes:
      • succ(0 .. succ_pred_max)  and  pred(1 .. succ_pred_max + 1)
      • add(a, b) for a, b in 0 .. add_sub_max  (all pairs)
      • sub(a, b) for a, b in 0 .. add_sub_max  with a >= b
      • mul(a, b) for a, b in 0 .. 9            (all 100 single-digit facts)

    All sequences use the same interleaved LSB-first format with 'i'/'o'
    (and 'j' for the second operand) markers.

    Parameters
    ----------
    succ_pred_max : inclusive upper bound for succ/pred training range
    add_sub_max   : inclusive upper bound for each operand in add/sub training
    n_pairs       : sequence length (digit pairs); must accommodate all results
    seed          : random seed for shuffle
    """
    seqs: list[list[str]] = []

    # succ / pred
    for n in range(succ_pred_max + 1):
        seqs.append(succ_seq_interleaved(n, n_pairs=n_pairs))
    for n in range(1, succ_pred_max + 2):
        seqs.append(pred_seq_interleaved(n, n_pairs=n_pairs))

    # add / sub  (all (a,b) pairs in [0, add_sub_max])
    for a in range(add_sub_max + 1):
        for b in range(add_sub_max + 1):
            seqs.append(add_seq_interleaved(a, b, n_pairs=n_pairs))
    for a in range(add_sub_max + 1):
        for b in range(a + 1):          # b <= a
            seqs.append(sub_seq_interleaved(a, b, n_pairs=n_pairs))

    # mul  (single-digit × single-digit: all 100 facts)
    for a in range(10):
        for b in range(10):
            seqs.append(mul_seq_interleaved(a, b, n_pairs=n_pairs))

    random.Random(seed).shuffle(seqs)
    return seqs


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Flat (MSB-first) sequences ===")
    for n in [0, 9, 10, 99, 100]:
        print(f"  succ_seq({n:4d}) = {succ_seq(n)}")
    for n in [1, 10, 100]:
        print(f"  pred_seq({n:4d}) = {pred_seq(n)}")

    print()
    print("=== Interleaved LSB-first sequences with markers (n_pairs=4) ===")
    for n in [0, 9, 10, 99, 211]:
        print(f"  succ_interleaved({n:4d}) = {succ_seq_interleaved(n, n_pairs=4)}")
    for n in [1, 10, 100]:
        print(f"  pred_interleaved({n:4d}) = {pred_seq_interleaved(n, n_pairs=4)}")

    print()
    name, train, test = digit_succ_pred_split()
    print(f"Flat split: {name}")
    print(f"  #train={len(train)}  #test={len(test)}")

    print()
    name2, train2, test2 = digit_succ_pred_split_interleaved()
    print(f"Interleaved split: {name2}")
    print(f"  #train={len(train2)}  #test={len(test2)}")
    print(f"  train[0] = {train2[0]}")
    print(f"  test[0]  = {test2[0]}")
