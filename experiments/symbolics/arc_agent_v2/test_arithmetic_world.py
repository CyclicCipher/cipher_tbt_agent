"""Tests of the arithmetic world: it shows only structureless SDRs, splits cleanly, and its scorer is sound (an oracle
scores 1, a guess scores chance). Run from experiments/symbolics:
    ../../venv312/Scripts/python.exe -m pytest arc_agent_v2/test_arithmetic_world.py -q
"""
import numpy as np

from arc_agent_v2.arithmetic_world import ArithmeticWorld
from arc_agent_v2.HDClibrary import SDR, overlap, thin


def test_symbols_are_structureless_sdrs():
    w = ArithmeticWorld(p=13)
    sdrs = [w.sdr[s] for s in w.symbols]
    assert len(sdrs) == 15 and all(isinstance(s, SDR) and s.d == 2048 and s.w == 40 for s in sdrs)
    assert max(overlap(a, b) for i, a in enumerate(sdrs) for b in sdrs[i + 1:]) <= 6   # "3" is no nearer "4" than "+"


def test_split_is_disjoint_and_complete():
    w = ArithmeticWorld(p=13, ops=("+", "*"), train_frac=0.5)
    tr, ho = set(w.train), set(w.held_out)
    assert not tr & ho and len(tr | ho) == 2 * 169 and abs(len(tr) - 169) <= 1


def test_episodes_are_the_written_equation():
    w = ArithmeticWorld(p=13)
    ep = w.episode("+", 9, 7)
    assert [w.decode(s) for s in ep] == ["9", "+", "7", "=", "3"]
    assert w.written("+", 12, 1)[-1] == "0"
    rng = np.random.default_rng(0)
    assert len(w.epoch(rng)) == len(w.train)


def test_decode_tolerates_noise():
    w = ArithmeticWorld(p=13)
    noisy = thin(w.sdr["5"], 24, np.random.default_rng(1)) | SDR.random(2048, 16, np.random.default_rng(2))
    assert w.decode(noisy) == "5" and w.decode(None) is None


def test_scorer_oracle_and_chance():
    w = ArithmeticWorld(p=13)
    answers = {tuple(s.idx.tobytes() for s in prefix): ans for prefix, ans, _ in w.queries()}
    oracle = lambda prefix: answers[tuple(s.idx.tobytes() for s in prefix)]
    assert w.score(oracle) == 1.0
    guess = lambda prefix: w.sdr["0"]
    assert w.score(guess) < 0.2                                  # chance is 1/13
    assert w.score(lambda prefix: None) == 0.0
