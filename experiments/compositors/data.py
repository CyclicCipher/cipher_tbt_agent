"""
Data generation for Compositor toy experiments.

Character-level tokenisation. All sequences are strings of individual characters.
The model sees raw characters and must discover structure (multi-digit numbers,
operators, relationships) from co-occurrence alone.
"""

import random
from typing import Optional


# --- Vocabulary ---

CHARS = list("0123456789+= <>,")
PAD = "\x00"
BOS = "\x01"
EOS = "\x02"

ALL_TOKENS = [PAD, BOS, EOS] + CHARS
TOKEN_TO_ID = {ch: i for i, ch in enumerate(ALL_TOKENS)}
ID_TO_TOKEN = {i: ch for ch, i in TOKEN_TO_ID.items()}
VOCAB_SIZE = len(ALL_TOKENS)

PAD_ID = TOKEN_TO_ID[PAD]
BOS_ID = TOKEN_TO_ID[BOS]
EOS_ID = TOKEN_TO_ID[EOS]


def encode(s: str) -> list[int]:
    """String -> list of token IDs."""
    return [BOS_ID] + [TOKEN_TO_ID[ch] for ch in s] + [EOS_ID]


def decode(ids: list[int]) -> str:
    """List of token IDs -> string (strips BOS/EOS/PAD)."""
    return "".join(
        ID_TO_TOKEN[i] for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)
    )


# --- Sequence generators ---


def gen_succession(n_seqs: int, min_start: int, max_start: int,
                   min_len: int = 3, max_len: int = 8,
                   rng: random.Random = None) -> list[str]:
    """Generate number line sequences: '4,5,6,7,8'

    Varying start points and lengths to teach succession from many angles.
    """
    rng = rng or random.Random()
    seqs = []
    for _ in range(n_seqs):
        start = rng.randint(min_start, max_start)
        length = rng.randint(min_len, max_len)
        nums = list(range(start, start + length))
        seqs.append(",".join(str(n) for n in nums))
    return seqs


def gen_comparisons(n_seqs: int, min_n: int, max_n: int,
                    rng: random.Random = None) -> list[str]:
    """Generate comparison sequences: '3<7', '9>2', etc."""
    rng = rng or random.Random()
    seqs = []
    for _ in range(n_seqs):
        a = rng.randint(min_n, max_n)
        b = rng.randint(min_n, max_n)
        while b == a:
            b = rng.randint(min_n, max_n)
        if rng.random() < 0.5:
            # less-than: smaller < larger
            if a < b:
                seqs.append(f"{a}<{b}")
            else:
                seqs.append(f"{b}<{a}")
        else:
            # greater-than: larger > smaller
            if a > b:
                seqs.append(f"{a}>{b}")
            else:
                seqs.append(f"{b}>{a}")
    return seqs


def gen_addition(n_seqs: int, min_op: int, max_op: int,
                 rng: random.Random = None) -> list[str]:
    """Generate addition sequences: '2+3=5', etc.

    Samples random operand pairs within range. With replacement to hit
    target count even when range is small.
    """
    rng = rng or random.Random()
    seqs = []
    for _ in range(n_seqs):
        a = rng.randint(min_op, max_op)
        b = rng.randint(min_op, max_op)
        seqs.append(f"{a}+{b}={a + b}")
    return seqs


# --- Dataset construction ---


def make_train_test_split(
    n_train_per_task: int = 1000,
    n_test_per_task: int = 200,
    # Succession ranges
    train_succ_max_start: int = 40,
    test_succ_min_start: int = 41,
    test_succ_max_start: int = 90,
    # Comparison ranges
    train_comp_max: int = 40,
    test_comp_min: int = 20,
    test_comp_max: int = 90,
    # Addition ranges
    train_add_max_op: int = 15,
    test_add_min_op: int = 10,
    test_add_max_op: int = 30,
    seed: int = 42,
):
    """Create train/test split.

    Train: numbers 1-40 for succession, 1-40 for comparisons, operands 1-15
    Test: numbers 41-90+ for succession, 20-90 for comparisons (at least one
          operand >15), operands 10-30 for addition (at least one >15)

    The test set is out-of-distribution: larger numbers and unseen combinations.
    """
    rng = random.Random(seed)

    train_seqs = []
    test_seqs = []

    # --- Succession ---
    train_seqs.extend(gen_succession(
        n_train_per_task, min_start=1, max_start=train_succ_max_start, rng=rng,
    ))
    test_seqs.extend(gen_succession(
        n_test_per_task, min_start=test_succ_min_start,
        max_start=test_succ_max_start, rng=rng,
    ))

    # --- Comparisons ---
    train_seqs.extend(gen_comparisons(
        n_train_per_task, min_n=1, max_n=train_comp_max, rng=rng,
    ))
    # Test comparisons: at least one number > train range
    test_comp = []
    while len(test_comp) < n_test_per_task:
        a = rng.randint(test_comp_min, test_comp_max)
        b = rng.randint(test_comp_min, test_comp_max)
        if a == b:
            continue
        if max(a, b) <= train_comp_max:
            continue  # ensure at least one OOD number
        if rng.random() < 0.5:
            test_comp.append(f"{min(a,b)}<{max(a,b)}")
        else:
            test_comp.append(f"{max(a,b)}>{min(a,b)}")
    test_seqs.extend(test_comp)

    # --- Addition ---
    train_seqs.extend(gen_addition(
        n_train_per_task, min_op=1, max_op=train_add_max_op, rng=rng,
    ))
    # Test addition: at least one operand > train range
    test_add = []
    while len(test_add) < n_test_per_task:
        a = rng.randint(test_add_min_op, test_add_max_op)
        b = rng.randint(test_add_min_op, test_add_max_op)
        if max(a, b) <= train_add_max_op:
            continue  # ensure OOD
        test_add.append(f"{a}+{b}={a + b}")
    test_seqs.extend(test_add)

    return train_seqs, test_seqs


def collate(sequences: list[str], max_len: Optional[int] = None) -> tuple:
    """Encode and pad a batch of string sequences.

    Returns:
        input_ids: (batch, max_len) -- BOS + content + EOS, right-padded
        target_ids: (batch, max_len) -- shifted left by 1 (next-token prediction)
        mask: (batch, max_len) -- 1.0 where target is valid, 0.0 on padding
    """
    import torch

    encoded = [encode(s) for s in sequences]
    if max_len is None:
        max_len = max(len(e) for e in encoded)

    input_ids = []
    target_ids = []
    mask = []

    for e in encoded:
        # Pad to max_len + 1 so we can shift
        padded = e + [PAD_ID] * (max_len + 1 - len(e))
        input_ids.append(padded[:max_len])
        target_ids.append(padded[1 : max_len + 1])
        m = [1.0] * (len(e) - 1) + [0.0] * (max_len - len(e) + 1)
        mask.append(m[:max_len])

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(target_ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.float32),
    )


if __name__ == "__main__":
    train, test = make_train_test_split()
    print(f"Train sequences: {len(train)}")
    print(f"Test sequences: {len(test)}")
    print(f"Vocab size: {VOCAB_SIZE}")

    # Task breakdown
    for label, seqs in [("Train", train), ("Test", test)]:
        succ = [s for s in seqs if "," in s]
        comp = [s for s in seqs if "<" in s or (">" in s and "=" not in s)]
        add = [s for s in seqs if "+" in s]
        print(f"\n{label}: {len(succ)} succession, {len(comp)} comparison, {len(add)} addition")
        print(f"  Sample succession: {succ[:3]}")
        print(f"  Sample comparison: {comp[:3]}")
        print(f"  Sample addition:   {add[:3]}")

    # Max encoded length
    all_seqs = train + test
    max_len = max(len(encode(s)) for s in all_seqs)
    print(f"\nMax encoded length: {max_len}")
