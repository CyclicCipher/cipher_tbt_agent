"""Counting problem generators.

Stage 1 — Query counting: "how many DOTs?" or "how many TENs?"
Stage 2 — Combined counting: count both with scratchpad cues
"""

import random
from typing import Any, List

from ..framework import Problem, ProblemGenerator, Step, Vocab


def _setup_counting_vocab(vocab: Vocab) -> None:
    """Ensure counting-specific tokens exist in vocab."""
    for d in range(10):
        vocab.add(str(d))
    vocab.add('DOT')
    vocab.add('TEN')
    vocab.add('STOP')


class QueryCountingGenerator(ProblemGenerator):
    """Stage 1: query-based counting.

    Input: shuffled DOTs and TENs, then NOTE <QUERY> to specify what to count.
    Work: WORK <count>
    The query token (DOT or TEN) appears in the input so the model knows
    which type to count BEFORE the work area begins.

    Example: DOT TEN DOT TEN DOT NOTE DOT WORK 3
    Example: TEN TEN TEN NOTE TEN WORK 3

    Rubric: 1 step ("count") with 1 graded token.
    """

    @property
    def name(self) -> str:
        return 'query_counting'

    def enumerate_all(self) -> List[Any]:
        """All (dot_count, ten_count) pairs, 0-9 each. 100 total."""
        return [(d, t) for d in range(10) for t in range(10)]

    def generate(self, specs: List[Any], n_samples: int,
                 vocab: Vocab) -> List[Problem]:
        _setup_counting_vocab(vocab)
        problems = []
        for _ in range(n_samples):
            d, t = random.choice(specs)
            # Build shuffled input
            input_toks = [vocab['DOT']] * d + [vocab['TEN']] * t
            random.shuffle(input_toks)
            # Random query — placed in input (after NOTE) so model can see it
            if random.random() < 0.5:
                query_tok, answer = vocab['DOT'], d
            else:
                query_tok, answer = vocab['TEN'], t
            input_toks.append(vocab.NOTE)
            input_toks.append(query_tok)

            problems.append(Problem(
                question=input_toks,
                steps=[
                    Step('count', [vocab[str(answer)]], weight=1.0),
                ],
            ))
        return problems


class CombinedCountingGenerator(ProblemGenerator):
    """Stage 2: combined counting with count-up process tokens.

    Input: shuffled DOTs and TENs.
    Work: WORK DOT <1..d> <STOP padding> TEN <1..t> <STOP padding>

    The model counts up using the successor function, then emits STOP
    tokens for remaining slots. Each count section is fixed at 9 tokens.
    This forces the model to demonstrate the counting PROCESS (successor
    chain), not just output the final count digit.

    Example (d=2, t=3):
      DOT TEN DOT TEN TEN WORK DOT 1 2 STOP STOP STOP STOP STOP STOP STOP
                              TEN 1 2 3 STOP STOP STOP STOP STOP STOP

    Example (d=0, t=1):
      TEN WORK DOT STOP STOP STOP STOP STOP STOP STOP STOP STOP
               TEN 1 STOP STOP STOP STOP STOP STOP STOP STOP

    Rubric: 4 steps — DOT cue (ungraded), dot_count (graded, 9 tokens),
            TEN cue (ungraded), ten_count (graded, 9 tokens).
    n_result = 20 for all problems.
    """

    @property
    def name(self) -> str:
        return 'combined_counting'

    def enumerate_all(self) -> List[Any]:
        return [(d, t) for d in range(10) for t in range(10)]

    def generate(self, specs: List[Any], n_samples: int,
                 vocab: Vocab) -> List[Problem]:
        _setup_counting_vocab(vocab)
        MAX_COUNT = 9  # max single-digit count
        problems = []
        for _ in range(n_samples):
            d, t = random.choice(specs)
            input_toks = [vocab['DOT']] * d + [vocab['TEN']] * t
            random.shuffle(input_toks)

            # Count-up process: 1, 2, ..., count then STOP to fill 9 slots
            dot_toks = ([vocab[str(i)] for i in range(1, d + 1)]
                        + [vocab['STOP']] * (MAX_COUNT - d))
            ten_toks = ([vocab[str(i)] for i in range(1, t + 1)]
                        + [vocab['STOP']] * (MAX_COUNT - t))

            problems.append(Problem(
                question=input_toks,
                steps=[
                    Step('dot_cue', [vocab['DOT']], grading='ungraded'),
                    Step('dot_count', dot_toks, weight=1.0),
                    Step('ten_cue', [vocab['TEN']], grading='ungraded'),
                    Step('ten_count', ten_toks, weight=1.0),
                ],
            ))
        return problems
