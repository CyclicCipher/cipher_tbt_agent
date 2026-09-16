import itertools

from env.dsl import (ADD, ARITY, CSUM, DIFF, DROP, FEVEN, FGT, MUL, N_OPS, NEG, REV, ROT, SORT, SWP, TAKE, UNIQ,
                     run, prog_to_str, str_to_prog)

X = (3, 1, 4, 1, 5, 9, 2)


def test_every_op_by_hand():
    assert run(((ADD, 7),), X) == (0, 8, 1, 8, 2, 6, 9)
    assert run(((MUL, 3),), X) == (9, 3, 2, 3, 5, 7, 6)
    assert run(((NEG, None),), X) == (7, 9, 6, 9, 5, 1, 8)
    assert run(((REV, None),), X) == (2, 9, 5, 1, 4, 1, 3)
    assert run(((SORT, None),), X) == (1, 1, 2, 3, 4, 5, 9)
    assert run(((ROT, 2),), X) == (4, 1, 5, 9, 2, 3, 1)
    assert run(((ROT, 9),), X) == (4, 1, 5, 9, 2, 3, 1)           # k mod len
    assert run(((TAKE, 3),), X) == (3, 1, 4)
    assert run(((TAKE, 0),), X) == ()
    assert run(((DROP, 2),), X) == (4, 1, 5, 9, 2)
    assert run(((FGT, 3),), X) == (4, 5, 9)
    assert run(((FEVEN, None),), X) == (4, 2)
    assert run(((UNIQ, None),), X) == (3, 1, 4, 5, 9, 2)
    assert run(((CSUM, None),), X) == (3, 4, 8, 9, 4, 3, 5)
    assert run(((DIFF, None),), X) == (8, 3, 7, 4, 4, 3)
    assert run(((SWP, None),), X) == (1, 3, 1, 4, 9, 5, 2)
    assert run(((ADD, 0),), X) == X                                # identity


def test_totality_for_every_digit_argument():
    inputs = [(), (5,), (0, 9), X, tuple(range(8))]
    for op in range(N_OPS):
        for k in (range(10) if ARITY[op] else [None]):
            for x in inputs:
                y = run(((op, k),), x)
                assert isinstance(y, tuple) and len(y) <= 8 and all(0 <= v <= 9 for v in y)


def test_string_round_trip():
    prog = ((ADD, 3), (REV, None), (ROT, 0), (SWP, None))
    assert str_to_prog(prog_to_str(prog)) == prog
