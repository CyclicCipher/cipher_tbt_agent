"""Calendar task — can the CTKG predict date arithmetic answers?

Tests the full loop: train on calendar sentences via read(), then predict
the answer portion of unseen test sentences character by character.

Start with the simplest sub-task: second succession (+1 second).
The answer is always the same as the input except seconds increment.

The key mechanism being tested: surprise-gated working memory holds
the input date digits active across the template span, so they're
still available when predicting the answer.
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from datetime import datetime, timedelta

from experiments.symbolic_ai_v2.ctkg.logic.graph import KnowledgeGraph
from experiments.symbolic_ai_v2.ctkg.logic.loop import AgenticLoop
from experiments.symbolic_ai_v2.environments.calendar import (
    _make_observation,
    _format_time,
    _add_time,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train_on_sentences(loop: AgenticLoop, sentences: list[list[str]]) -> None:
    """Feed each sentence through loop.read()."""
    for sentence in sentences:
        loop.read(sentence)


def _predict_answer(
    loop: AgenticLoop,
    input_chars: list[str],
    answer_length: int,
) -> list[str]:
    """Read input, then predict answer_length characters autoregressively.

    The input is everything up to (but not including) the answer portion.
    After reading, we predict one char at a time, feeding each prediction
    back to update context.
    """
    # Read input portion (builds activation + WM).
    loop.read(input_chars)

    # Now predict answer characters one at a time.
    predicted: list[str] = []
    for _ in range(answer_length):
        next_char = loop.predict_next()
        if next_char is None:
            predicted.append("?")
            # Feed a placeholder to keep context moving.
            loop.observe(["?"])
        else:
            predicted.append(next_char)
            # Feed prediction back to update context for next character.
            loop.observe([next_char])
    return predicted


def _split_sentence(sentence: list[str]) -> tuple[list[str], list[str]]:
    """Split a calendar sentence into (input_portion, answer_portion).

    The answer is the last 21 chars: 20-char datetime + final '.'.
    But we only predict the 20-char datetime (excluding the trailing '.').

    Sentence structure:
    "The current date is SS:MM:HH MM/DD/YYYY. In N unit, the date will be SS:MM:HH MM/DD/YYYY."

    The answer datetime starts 21 chars from the end (20 chars + '.').
    """
    # Find "the date will be " and split there.
    s = "".join(sentence)
    marker = "the date will be "
    idx = s.find(marker)
    if idx == -1:
        raise ValueError(f"Cannot find answer marker in: {s[:50]}...")
    answer_start = idx + len(marker)
    input_portion = list(s[:answer_start])
    answer_portion = list(s[answer_start:answer_start + 19])  # SS:MM:HH MM/DD/YYYY (19 chars)
    return input_portion, answer_portion


def _char_accuracy(predicted: list[str], expected: list[str]) -> float:
    """Per-character accuracy."""
    if not expected:
        return 0.0
    correct = sum(1 for p, e in zip(predicted, expected) if p == e)
    return correct / len(expected)


# ---------------------------------------------------------------------------
# Training data: second succession only (fast)
# ---------------------------------------------------------------------------

def _seconds_training_data(n_sentences: int = 200) -> list[list[str]]:
    """Generate n_sentences of "+1 second" examples."""
    base = datetime(2000, 6, 15, 14, 58, 0)
    sentences = []
    for i in range(n_sentences):
        dt = base + timedelta(seconds=i)
        sentences.append(_make_observation(dt, 1, "seconds"))
    return sentences


def _seconds_test_cases() -> list[tuple[list[str], list[str]]]:
    """Test cases for second succession. Mix of seen and unseen times."""
    cases = []
    # Case 1: simple increment (no wrap).
    dt = datetime(2000, 6, 15, 14, 58, 30)
    obs = _make_observation(dt, 1, "seconds")
    inp, ans = _split_sentence(obs)
    cases.append((inp, ans))

    # Case 2: second wrap 59→00 (carry to minutes).
    dt = datetime(2000, 6, 15, 14, 58, 59)
    obs = _make_observation(dt, 1, "seconds")
    inp, ans = _split_sentence(obs)
    cases.append((inp, ans))

    # Case 3: unseen time (different hour).
    dt = datetime(2000, 6, 15, 10, 30, 15)
    obs = _make_observation(dt, 1, "seconds")
    inp, ans = _split_sentence(obs)
    cases.append((inp, ans))

    # Case 4: another simple case.
    dt = datetime(2000, 6, 15, 14, 58, 45)
    obs = _make_observation(dt, 1, "seconds")
    inp, ans = _split_sentence(obs)
    cases.append((inp, ans))

    # Case 5: second 00→01 (low digits).
    dt = datetime(2000, 6, 15, 14, 59, 0)
    obs = _make_observation(dt, 1, "seconds")
    inp, ans = _split_sentence(obs)
    cases.append((inp, ans))

    return cases


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCalendarSmoke:
    """Fast smoke tests: can the system predict anything at all?"""

    def test_split_sentence(self):
        """Verify sentence splitting works."""
        dt = datetime(2000, 6, 15, 14, 58, 30)
        obs = _make_observation(dt, 1, "seconds")
        inp, ans = _split_sentence(obs)
        # Answer should be the result datetime.
        result_dt = _add_time(dt, 1, "seconds")
        expected = _format_time(result_dt)
        assert ans == expected, f"Expected {expected}, got {ans}"

    def test_wm_holds_surprising_tokens(self):
        """After reading a sentence, WM should contain some tokens."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        loop.CONSOLIDATION_INTERVAL = 0  # disable auto-consolidation

        # Train on a few sentences so template becomes predictable.
        sentences = _seconds_training_data(20)
        _train_on_sentences(loop, sentences)

        # Now read one more sentence and check WM.
        test_sentence = _seconds_training_data(1)[0]
        loop.read(test_sentence)
        wm = kg.wm_contents()
        # WM should have SOME entries (the surprising tokens).
        # After training, template text is predictable; date digits are surprising.
        assert len(wm) > 0, "Working memory should hold surprising tokens"

    def test_predict_next_returns_something(self):
        """After training, predict_next() should return a character."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        loop.CONSOLIDATION_INTERVAL = 0

        sentences = _seconds_training_data(50)
        _train_on_sentences(loop, sentences)

        # Read a test input up to the answer portion.
        test_cases = _seconds_test_cases()
        inp, ans = test_cases[0]
        loop.read(inp)

        pred = loop.predict_next()
        # Should predict SOMETHING (not None).
        assert pred is not None, "predict_next() should return a character after training"


class TestCalendarAccuracy:
    """Accuracy tests on second succession."""

    @pytest.fixture(scope="class")
    def trained_system(self):
        """Train on 200 seconds sentences, return (loop, kg)."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        loop.CONSOLIDATION_INTERVAL = 0  # consolidate manually

        sentences = _seconds_training_data(200)
        _train_on_sentences(loop, sentences)

        # Consolidate once after all training.
        loop.consolidate()

        return loop, kg

    def test_seconds_char_accuracy(self, trained_system):
        """Character accuracy on second succession should beat random."""
        loop, kg = trained_system
        test_cases = _seconds_test_cases()

        total_chars = 0
        correct_chars = 0
        exact_matches = 0

        for inp, expected_ans in test_cases:
            predicted = _predict_answer(loop, inp, len(expected_ans))
            acc = _char_accuracy(predicted, expected_ans)

            total_chars += len(expected_ans)
            correct_chars += sum(1 for p, e in zip(predicted, expected_ans) if p == e)
            if predicted == expected_ans:
                exact_matches += 1

            # Diagnostic output.
            pred_str = "".join(predicted)
            exp_str = "".join(expected_ans)
            print(f"  Expected: {exp_str}")
            print(f"  Predicted: {pred_str}")
            print(f"  Char accuracy: {acc:.1%}")
            print(f"  WM size: {len(kg.wm_contents())}")
            print()

        overall_acc = correct_chars / total_chars if total_chars > 0 else 0.0
        print(f"Overall char accuracy: {overall_acc:.1%}")
        print(f"Exact matches: {exact_matches}/{len(test_cases)}")

        # Random baseline: 1/34 unique chars ≈ 3%.
        # We should beat random substantially.
        assert overall_acc > 0.03, (
            f"Char accuracy {overall_acc:.1%} should beat random (3%)"
        )

    def test_template_chars_preserved(self, trained_system):
        """The non-varying parts of the answer (colons, slashes, spaces)
        should be predicted correctly — they're always the same."""
        loop, kg = trained_system
        test_cases = _seconds_test_cases()

        # Template positions in SS:MM:HH MM/DD/YYYY:
        # positions 2,5,8,11,14 are ':', ':', ':', '/', '/'
        # position 8 is ' '
        template_positions = {
            2: ':',   # after SS
            5: ':',   # after MM
            8: ' ',   # after HH
            11: '/',  # after MM (month)
            14: '/',  # after DD
        }

        template_correct = 0
        template_total = 0

        for inp, expected_ans in test_cases:
            predicted = _predict_answer(loop, inp, len(expected_ans))
            for pos, expected_char in template_positions.items():
                if pos < len(predicted):
                    template_total += 1
                    if predicted[pos] == expected_char:
                        template_correct += 1

        if template_total > 0:
            template_acc = template_correct / template_total
            print(f"Template char accuracy: {template_acc:.1%}")
            # Template characters should be very predictable.
            assert template_acc > 0.5, (
                f"Template accuracy {template_acc:.1%} should be high"
            )


# ---------------------------------------------------------------------------
# Standalone runner with diagnostics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Calendar Task: Second Succession ===\n")

    kg = KnowledgeGraph()
    loop = AgenticLoop(kg)
    loop.CONSOLIDATION_INTERVAL = 0

    print("Training on 200 second-succession sentences...")
    sentences = _seconds_training_data(200)
    _train_on_sentences(loop, sentences)

    print(f"Graph: {kg.node_count()} nodes, {kg.edge_count()} edges")
    print(f"Consolidating...")
    stats = loop.consolidate()
    print(f"Consolidation: {stats}")
    print(f"Graph after: {kg.node_count()} nodes, {kg.edge_count()} edges")
    print()

    test_cases = _seconds_test_cases()
    total_chars = 0
    correct_chars = 0

    for i, (inp, expected_ans) in enumerate(test_cases):
        predicted = _predict_answer(loop, inp, len(expected_ans))
        acc = _char_accuracy(predicted, expected_ans)

        pred_str = "".join(predicted)
        exp_str = "".join(expected_ans)
        inp_str = "".join(inp[-30:])  # last 30 chars of input

        print(f"Test {i+1}:")
        print(f"  Input (last 30): ...{inp_str}")
        print(f"  Expected answer:  {exp_str}")
        print(f"  Predicted answer: {pred_str}")
        print(f"  Char accuracy:    {acc:.1%}")
        print(f"  WM contents:      {len(kg.wm_contents())} nodes")
        # Show what's in WM.
        wm_labels = []
        for nid in kg.wm_contents():
            label = kg.label_for_node(nid)
            wm_labels.append(label)
        print(f"  WM labels:        {wm_labels[:15]}...")
        print()

        total_chars += len(expected_ans)
        correct_chars += sum(1 for p, e in zip(predicted, expected_ans) if p == e)

    overall = correct_chars / total_chars if total_chars > 0 else 0.0
    print(f"Overall char accuracy: {overall:.1%} ({correct_chars}/{total_chars})")
