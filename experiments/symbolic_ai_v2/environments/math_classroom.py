"""MathClassroom — a textworld where the agent learns arithmetic.

The agent is a "student" in a classroom. The teacher presents problems on
a blackboard. The agent answers by selecting from available actions. The
teacher reveals the correct answer after each response. The agent learns
from the feedback — not from special math code, but from the same
observe/act loop used for the science lab.

There is NO special handling of numbers. The tokens "3", "+", "4", "=", "7"
are opaque strings, identical in kind to "AT_corridor" or "SEE_wrench".
The agent must discover that "3 + 4" is followed by "7" the same way it
discovers that "go_west" from corridor leads to supply_closet.

**CRITICAL: No BOARD_ or ANSWER_ prefixes.** Digits are bare tokens: "3",
not "BOARD_3". The number "3" is the same node whether it appears as a
question operand, an answer, or a counting step. This forces the system to
use context (where the digit appears in the observation sequence) to
determine role, not type tags.

Two modes:

**Mode A — Interleaved training and testing:**
  For each problem type (succession, addition, subtraction, ...):
    10 training examples (agent sees problem + answer)
    10 test questions (agent must answer, then sees correct answer)
    Repeat for up to 100 cycles.
  Score = fraction of test questions answered correctly.

**Mode B — Test-first adaptive:**
  Present 1 test question. If wrong, train on that problem + one more.
  Repeat up to 1000 times.
  Score = fraction of first-attempt correct answers.

Observation encoding (same intero/extero as science lab):

intero (edge type 1):
    PHASE_train or PHASE_test     — current mode
    SCORE_X                       — running score (correct/total)
    PROBLEM_TYPE_succession       — what kind of problem

extero (edge type 0):
    3, +, 4, =, ?                 — the problem on the board (bare tokens)
    FEEDBACK_correct or FEEDBACK_wrong  — after answering

Action format: bare digit tokens 0..9 (autoregressive, one digit at a time)
    0, 1, ..., 9, done — same tokens as observation digits
"""

from __future__ import annotations

import random
from typing import Optional

from ..environment import Environment


# ── Problem universes (exhaustive, then split train/test) ────────────────────
#
# All numbers are tokenized to individual digits. The number 42 becomes
# the token sequence ["4", "2"]. This is character-level tokenization —
# the same principle used by the tokenizer at the connector boundary.
# The model sees individual digits, not whole numbers.

def _digits(n: int) -> list[str]:
    """Convert a number to a list of individual digit tokens."""
    return list(str(n))

SP = " "   # space token separating numbers/operators within an equation
SEP = ","  # separator between clauses (e.g., problem clause, answer clause)

def _all_succ() -> list[tuple[list[str], str]]:
    """All succession problems: succ(X) = X+1 for X in 0..8."""
    return [(_digits(x) + [SP, "succ"], str(x + 1)) for x in range(9)]

def _all_pred() -> list[tuple[list[str], str]]:
    """All predecessor problems: pred(X) = X-1 for X in 1..9."""
    return [(_digits(x) + [SP, "pred"], str(x - 1)) for x in range(1, 10)]

def _all_add() -> list[tuple[list[str], str]]:
    """All single-digit addition: X + Y = Z for X,Y in 0..9."""
    return [(_digits(x) + [SP, "+", SP] + _digits(y) + [SP, "="], str(x + y))
            for x in range(10) for y in range(10)]

def _all_sub() -> list[tuple[list[str], str]]:
    """All single-digit subtraction: X - Y = Z for X >= Y."""
    return [(_digits(x) + [SP, "-", SP] + _digits(y) + [SP, "="], str(x - y))
            for x in range(10) for y in range(x + 1)]

def _all_mul() -> list[tuple[list[str], str]]:
    """All single-digit multiplication: X * Y = Z."""
    return [(_digits(x) + [SP, "*", SP] + _digits(y) + [SP, "="], str(x * y))
            for x in range(10) for y in range(10)]


def _split_train_test(
    universe: list[tuple[list[str], str]],
    test_fraction: float,
    rng: random.Random,
) -> tuple[list[tuple[list[str], str]], list[tuple[list[str], str]]]:
    """Split a problem universe into disjoint train and test pools.

    Test problems are NEVER seen during training. This is a hard guarantee.
    """
    shuffled = list(universe)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_fraction))
    test_pool = shuffled[:n_test]
    train_pool = shuffled[n_test:]
    return train_pool, test_pool


PROBLEM_UNIVERSES = {
    "succession": _all_succ,
    "predecessor": _all_pred,
    "addition": _all_add,
    "subtraction": _all_sub,
    "multiplication": _all_mul,
}


# ── MathClassroomEnv ─────────────────────────────────────────────────────────

class MathClassroomEnv(Environment):
    """A textworld where the agent learns arithmetic from a teacher.

    Parameters
    ----------
    problem_type : which arithmetic operation to test.
    mode : "A" (interleaved train/test cycles) or "B" (test-first adaptive).
    max_cycles : maximum number of cycles (Mode A) or attempts (Mode B).
    train_per_cycle : training examples per cycle (Mode A only).
    test_per_cycle : test questions per cycle (Mode A only).
    seed : random seed for reproducibility.
    answer_range : range of valid answer tokens. Default 0..20.
    """

    def __init__(
        self,
        problem_type: str = "succession",
        mode: str = "A",
        max_cycles: int = 100,
        train_per_cycle: int = 10,
        test_per_cycle: int = 10,
        seed: int = 42,
        answer_range: tuple[int, int] = (0, 199),
        test_fraction: float = 0.3,
        counting_warmup: int = 5,
    ) -> None:
        self._problem_type = problem_type
        self._mode = mode
        self._max_cycles = max_cycles
        self._train_per_cycle = train_per_cycle
        self._test_per_cycle = test_per_cycle
        self._rng = random.Random(seed)
        self._answer_range = answer_range
        self._counting_warmup = counting_warmup

        # Split problem universe into disjoint train/test pools.
        # Test problems NEVER appear during training. Hard guarantee.
        universe = PROBLEM_UNIVERSES[problem_type]()
        split_rng = random.Random(seed)  # deterministic split
        self._train_pool, self._test_pool = _split_train_test(
            universe, test_fraction, split_rng,
        )
        # Separate RNG for sampling within pools (so split is independent).
        self._sample_rng = random.Random(seed + 1000)

        self.reset()

    def reset(self) -> None:
        self._cycle = 0
        self._step_in_cycle = 0
        self._total_correct = 0
        self._total_tested = 0
        self._done = False

        # Counting warmup: the teacher counts on the board before problems.
        # The agent sees 0, 1, 2, ..., max in order, repeated counting_warmup
        # times. This builds the number line (NNO) as transition edges.
        self._in_warmup: bool = self._counting_warmup > 0
        self._warmup_pass: int = 0
        self._warmup_position: int = 0

        # Current problem state.
        self._board: list[str] = []       # tokens on the board
        self._correct_answer: str = ""    # correct answer as a string (e.g., "42")
        self._phase: str = "train"        # "train", "test", "counting", or "reproduce"
        self._last_feedback: str = ""     # "correct", "wrong", or ""
        self._last_answer_shown: str = "" # revealed answer after feedback
        self._agent_answered: bool = False
        self._waiting_for_answer: bool = False

        # Autoregressive answer state: the agent emits one digit at a time.
        # _answer_buffer accumulates digits until the agent says "done".
        self._answer_buffer: list[str] = []
        # Training reproduction: after seeing the answer, agent must reproduce it.
        self._awaiting_reproduction: bool = False

        # Generate first problem (or start warmup).
        if self._in_warmup:
            self._advance_warmup()
        else:
            self._advance()

    def _advance_warmup(self) -> None:
        """Present the next number in the counting sequence.

        The board shows the current number. The answer shows the next.
        The agent sees: [3, next_is, 4] (training, answer shown).
        Both 3 and 4 are the same bare tokens — no BOARD/ANSWER prefixes.
        Counts 0..max_number, repeat for counting_warmup passes.
        """
        lo, hi = self._answer_range
        max_number = hi  # count 0..hi (e.g., 0..199)

        self._phase = "counting"
        self._board = _digits(self._warmup_position) + [SP, "next_is"]
        if self._warmup_position < max_number:
            self._correct_answer = str(self._warmup_position + 1)
        else:
            # End of number line — wrap to 0 for next pass.
            self._correct_answer = str(0)
        self._last_feedback = ""
        self._last_answer_shown = ""
        self._waiting_for_answer = False
        self._agent_answered = False

        # Advance position.
        self._warmup_position += 1
        if self._warmup_position > max_number:
            self._warmup_position = 0
            self._warmup_pass += 1
            if self._warmup_pass >= self._counting_warmup:
                self._in_warmup = False

    def _advance(self) -> None:
        """Move to the next problem."""
        if self._in_warmup:
            self._advance_warmup()
            return
        if self._mode == "A":
            self._advance_mode_a()
        else:
            self._advance_mode_b()

    def _sample_from_pool(self, pool: list[tuple[list[str], str]]) -> tuple[list[str], str]:
        """Sample one problem from a pool (with replacement)."""
        return self._sample_rng.choice(pool)

    def _advance_mode_a(self) -> None:
        """Mode A: interleaved train/test cycles."""
        total_per_cycle = self._train_per_cycle + self._test_per_cycle

        if self._cycle >= self._max_cycles:
            self._done = True
            return

        step = self._step_in_cycle
        if step < self._train_per_cycle:
            self._phase = "train"
            self._board, self._correct_answer = self._sample_from_pool(self._train_pool)
        else:
            self._phase = "test"
            self._board, self._correct_answer = self._sample_from_pool(self._test_pool)

        self._last_feedback = ""
        self._last_answer_shown = ""
        self._waiting_for_answer = (self._phase == "test")
        self._agent_answered = False

        # Advance cycle counter.
        self._step_in_cycle += 1
        if self._step_in_cycle >= total_per_cycle:
            self._step_in_cycle = 0
            self._cycle += 1

    def _advance_mode_b(self) -> None:
        """Mode B: test-first adaptive."""
        if self._cycle >= self._max_cycles:
            self._done = True
            return

        # Always present as test first — drawn from test pool.
        self._phase = "test"
        self._board, self._correct_answer = self._sample_from_pool(self._test_pool)
        self._last_feedback = ""
        self._last_answer_shown = ""
        self._waiting_for_answer = True
        self._agent_answered = False

    # ── observe ───────────────────────────────────────────────────────────

    def observe(self) -> list[tuple[str, Optional[int]]]:
        toks: list[tuple[str, Optional[int]]] = []

        def intero(val: str) -> None:
            toks.append((val, None if not toks else 1))

        def extero(val: str) -> None:
            toks.append((val, None if not toks else 0))

        # Interoception: phase, score, problem type.
        intero(f"PHASE_{self._phase}")
        intero(f"PROBLEM_TYPE_{self._problem_type}")
        if self._total_tested > 0:
            pct = int(100 * self._total_correct / self._total_tested)
            intero(f"SCORE_{pct}")

        # Exteroception: the board.
        # Each element of self._board is a single character (digit or operator).
        # The generators produce them in reading order: ["1", "2", "+", "5", "="]
        # No BOARD_ prefix — tokens are bare. "3" is the same node whether it
        # appears in a question, an answer, or a counting step.
        for tok in self._board:
            extero(tok)
        if self._waiting_for_answer:
            extero("?")
            # Show digits emitted so far during autoregressive answering.
            for d in self._answer_buffer:
                extero(f"EMIT_{d}")  # EMIT prefix distinguishes "typed" from "shown"

        # Feedback from last answer (if any).
        if self._last_feedback:
            extero(f"FEEDBACK_{self._last_feedback}")
        if self._last_answer_shown:
            # Show correct answer digit by digit — bare tokens, no ANSWER_ prefix.
            for d in self._last_answer_shown:
                extero(d)

        # In training or counting phase, show the answer digit by digit.
        if self._phase in ("train", "counting") and not self._awaiting_reproduction:
            for d in self._correct_answer:
                extero(d)

        # In reproduce phase, show the answer (what to copy) + reproduction prompt.
        if self._awaiting_reproduction:
            for d in self._correct_answer:
                extero(d)
            extero("REPRODUCE_?")

        return toks

    # ── act ────────────────────────────────────────────────────────────────

    def act(self, action: str) -> None:
        if self._done:
            return

        if self._waiting_for_answer or self._awaiting_reproduction:
            if action == "done":
                # Agent signals answer/reproduction is complete.
                answer = "".join(self._answer_buffer)
                self._answer_buffer = []

                if self._awaiting_reproduction:
                    # Training reproduction complete.
                    self._awaiting_reproduction = False
                    self._agent_answered = True
                    if answer == self._correct_answer:
                        self._last_feedback = "correct"
                    else:
                        self._last_feedback = "wrong"
                else:
                    # Test answer complete.
                    self._agent_answered = True
                    self._waiting_for_answer = False

                    if answer == self._correct_answer:
                        self._last_feedback = "correct"
                        self._total_correct += 1
                    else:
                        self._last_feedback = "wrong"

                    self._total_tested += 1
                    self._last_answer_shown = self._correct_answer

                    # Mode B: if wrong, convert to training + one extra.
                    if self._mode == "B" and self._last_feedback == "wrong":
                        self._cycle += 1
                        return

                    if self._mode == "B":
                        self._cycle += 1

            elif len(action) == 1 and action.isdigit():
                # Agent emits one digit (bare token, same node as observation).
                self._answer_buffer.append(action)
                # Auto-submit if answer buffer reaches correct answer length.
                # This prevents the agent from emitting infinite digits.
                if len(self._answer_buffer) >= len(self._correct_answer):
                    self.act("done")
                    return

        elif action == "next":
            if self._phase == "train" and not self._awaiting_reproduction and not self._agent_answered:
                # In training: after seeing the answer, enter reproduction mode.
                # The agent must reproduce the answer it just saw.
                self._awaiting_reproduction = True
                self._answer_buffer = []
            else:
                # Move to next problem.
                self._agent_answered = False
                self._awaiting_reproduction = False
                self._advance()

    # ── available actions ─────────────────────────────────────────────────

    def available_actions(self) -> list[str]:
        actions: list[str] = []

        if self._waiting_for_answer or self._awaiting_reproduction:
            # 10 digit choices + done signal.
            # Bare digit tokens — same nodes as observation digits.
            for d in range(10):
                actions.append(str(d))
            # Can signal "done" if at least one digit has been emitted.
            if self._answer_buffer:
                actions.append("done")
        else:
            actions.append("next")

        return actions

    # ── terminal conditions ───────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return self._done

    @property
    def won(self) -> bool:
        # "Won" if score >= 90% and at least 10 tested.
        if self._total_tested < 10:
            return False
        return self._total_correct / self._total_tested >= 0.9

    # ── diagnostics ───────────────────────────────────────────────────────

    @property
    def score(self) -> float:
        if self._total_tested == 0:
            return 0.0
        return self._total_correct / self._total_tested

    @property
    def total_tested(self) -> int:
        return self._total_tested

    @property
    def total_correct(self) -> int:
        return self._total_correct

    def summary(self) -> str:
        return (f"MathClassroom(type={self._problem_type}, mode={self._mode}, "
                f"cycle={self._cycle}/{self._max_cycles}, "
                f"score={self._total_correct}/{self._total_tested})")
