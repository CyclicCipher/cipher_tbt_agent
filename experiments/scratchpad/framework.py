"""Core scratchpad framework: Vocab, Problem, Grader, Curriculum.

Design goals:
  - Model-agnostic: works with any model that consumes/produces token sequences
  - Unified rubric system: same grading API for math, language, any domain
  - Process supervision: grade intermediate steps, not just final answers
  - Extensible: new problem types just implement ProblemGenerator
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocab:
    """Dynamic vocabulary that grows as tokens are added.

    Special tokens (PAD, WORK, NOTE, SEP) are reserved at construction.
    Domain-specific tokens are added by generators.
    """

    def __init__(self):
        self._tok2id: Dict[str, int] = {}
        self._id2tok: Dict[int, str] = {}
        self._next_id = 0

        # Reserve structural tokens
        self.PAD = self.add('PAD')    # padding
        self.WORK = self.add('WORK')  # start of work area (= in old format)
        self.NOTE = self.add('NOTE')  # start of notepad area
        self.SEP = self.add('SEP')    # step separator within work area

    def add(self, token: str) -> int:
        """Add a token. Returns its ID (existing ID if already present)."""
        if token not in self._tok2id:
            tid = self._next_id
            self._tok2id[token] = tid
            self._id2tok[tid] = token
            self._next_id += 1
        return self._tok2id[token]

    def add_many(self, tokens: List[str]) -> List[int]:
        """Add multiple tokens. Returns list of IDs."""
        return [self.add(t) for t in tokens]

    def __getitem__(self, token: str) -> int:
        return self._tok2id[token]

    def __contains__(self, token: str) -> bool:
        return token in self._tok2id

    def decode(self, token_id: int) -> str:
        return self._id2tok.get(token_id, f'?{token_id}')

    def decode_sequence(self, ids) -> str:
        """Decode a sequence of IDs to a human-readable string."""
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        return ' '.join(self.decode(t) for t in ids)

    def __len__(self) -> int:
        return self._next_id


# ---------------------------------------------------------------------------
# Problem structure
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """One named step in a solution process.

    Each step has a name (for diagnostics), expected tokens (ground truth),
    a weight (for weighted scoring), and a grading mode.

    Grading modes:
      'exact'    — token-by-token exact match (default, for math)
      'custom'   — use custom_fn(expected, actual) -> float in [0,1]
      'ungraded' — not scored (e.g. notepad content, formatting tokens)
    """
    name: str
    tokens: List[int]
    weight: float = 1.0
    grading: str = 'exact'
    custom_fn: Optional[Callable] = None

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)


@dataclass
class Problem:
    """A single problem instance with input, solution steps, and notepad.

    The full token sequence is:
        [question] [WORK] [step1 tokens] [step2 tokens] ... [NOTE] [notepad]

    For training, the model must predict everything from WORK onward.
    For grading, only the work area steps are evaluated (notepad is free).
    """
    question: List[int]           # input tokens (read-only)
    steps: List[Step]             # ordered solution steps (graded)
    notepad: List[int] = field(default_factory=list)  # free scratch (future)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def work_tokens(self) -> List[int]:
        """All tokens in the work area (concatenation of all steps)."""
        tokens = []
        for s in self.steps:
            tokens.extend(s.tokens)
        return tokens

    @property
    def n_result(self) -> int:
        """Number of result tokens after the WORK marker.

        Matches the old convention where n_result excluded the '=' separator.
        The training script computes loss on the last n_result positions.
        """
        return len(self.work_tokens)

    def to_tokens(self, vocab: Vocab) -> List[int]:
        """Full token sequence: question + WORK + work + [NOTE + notepad]."""
        tokens = list(self.question)
        tokens.append(vocab.WORK)
        tokens.extend(self.work_tokens)
        if self.notepad:
            tokens.append(vocab.NOTE)
            tokens.extend(self.notepad)
        return tokens


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class Grader:
    """Grades model output against a Problem's rubric (steps).

    Returns per-step accuracy, per-token accuracy, and weighted overall score.
    """

    def grade(self, problem: Problem, output_tokens: List[int]) -> Dict[str, Any]:
        """Grade output tokens against problem steps.

        Args:
            problem: the Problem with expected steps
            output_tokens: model's predicted tokens for the work area
                           (excluding WORK marker, just the content)

        Returns dict with:
            'per_step': {step_name: accuracy} for each graded step
            'per_token': list of 0/1 for each token position
            'overall': weighted average of graded steps
            'exact_match': 1 if all graded tokens correct, else 0
        """
        per_step = {}
        per_token = []
        pos = 0

        for step in problem.steps:
            expected = step.tokens
            actual = output_tokens[pos:pos + len(expected)]

            if step.grading == 'exact':
                tok_correct = []
                for j, exp_t in enumerate(expected):
                    got = actual[j] if j < len(actual) else -1
                    tok_correct.append(1 if got == exp_t else 0)
                step_acc = sum(tok_correct) / len(expected) if expected else 1.0
                per_token.extend(tok_correct)

            elif step.grading == 'custom' and step.custom_fn is not None:
                step_acc = step.custom_fn(expected, actual)
                # Custom grading doesn't produce per-token detail
                per_token.extend([step_acc] * len(expected))

            elif step.grading == 'ungraded':
                per_token.extend([float('nan')] * len(expected))
                pos += len(expected)
                continue  # skip weight accumulation

            else:
                raise ValueError(f"Unknown grading mode: {step.grading}")

            per_step[step.name] = step_acc
            pos += len(expected)

        # Weighted overall
        graded_steps = [s for s in problem.steps if s.grading != 'ungraded']
        total_weight = sum(s.weight for s in graded_steps)
        if total_weight > 0:
            overall = sum(per_step[s.name] * s.weight for s in graded_steps) / total_weight
        else:
            overall = 1.0

        # Exact match (all graded tokens correct)
        graded_tokens = [t for t in per_token if t == t]  # exclude NaN
        exact = 1 if graded_tokens and all(t == 1 for t in graded_tokens) else 0

        return {
            'per_step': per_step,
            'per_token': per_token,
            'overall': overall,
            'exact_match': exact,
        }

    def grade_batch(self, problems: List[Problem], output_batch: Tensor,
                    vocab: Vocab) -> Dict[str, Any]:
        """Grade a batch of outputs. Returns aggregated metrics."""
        results = []
        for prob, out_row in zip(problems, output_batch):
            # Extract work area tokens (skip WORK marker)
            work_start = len(prob.question) + 1  # +1 for WORK token
            out_tokens = out_row[work_start:].tolist()
            results.append(self.grade(prob, out_tokens))

        # Aggregate
        n = len(results)
        avg_overall = sum(r['overall'] for r in results) / n
        avg_exact = sum(r['exact_match'] for r in results) / n

        # Per-step averages
        step_names = list(results[0]['per_step'].keys()) if results else []
        avg_per_step = {}
        for name in step_names:
            avg_per_step[name] = sum(r['per_step'][name] for r in results) / n

        return {
            'overall': avg_overall,
            'exact_match': avg_exact,
            'per_step': avg_per_step,
            'individual': results,
        }


# ---------------------------------------------------------------------------
# Problem Generator (abstract base)
# ---------------------------------------------------------------------------

class ProblemGenerator(ABC):
    """Abstract base for problem generators.

    Subclasses define a problem domain (counting, arithmetic, etc.)
    and produce Problem instances with appropriate rubrics.
    """

    @abstractmethod
    def enumerate_all(self) -> List[Any]:
        """Enumerate all problem specifications (for train/test splitting).

        Returns a list of problem specs (tuples, dicts, etc.) that uniquely
        identify each problem. These specs are split into train/test sets,
        then passed to generate() to produce Problem instances.
        """
        ...

    @abstractmethod
    def generate(self, specs: List[Any], n_samples: int,
                 vocab: Vocab) -> List[Problem]:
        """Generate n_samples Problem instances from the given specs.

        Args:
            specs: problem specifications (subset of enumerate_all output)
            n_samples: how many samples to generate (with replacement)
            vocab: shared vocabulary to use for token encoding
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this problem type."""
        ...


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    """One stage in a curriculum."""
    number: int
    generator: ProblemGenerator
    pass_threshold: float = 0.95   # test accuracy to advance
    max_epochs: int = 100

    @property
    def name(self) -> str:
        return self.generator.name


class Curriculum:
    """Ordered sequence of stages with progression rules."""

    def __init__(self, stages: List[Stage]):
        self.stages = {s.number: s for s in stages}

    def get_stage(self, number: int) -> Stage:
        return self.stages[number]

    @property
    def stage_numbers(self) -> List[int]:
        return sorted(self.stages.keys())

    def __len__(self) -> int:
        return len(self.stages)


# ---------------------------------------------------------------------------
# Tensor utilities
# ---------------------------------------------------------------------------

def problems_to_tensors(problems: List[Problem], vocab: Vocab,
                        seq_len: int) -> Tuple[Tensor, int]:
    """Convert Problem list to a left-padded tensor batch.

    Returns:
        seqs: (n_samples, seq_len) long tensor
        n_result: number of result tokens (consistent across batch)
    """
    n_result = problems[0].n_result
    seqs = torch.zeros(len(problems), seq_len, dtype=torch.long)

    for i, prob in enumerate(problems):
        tokens = prob.to_tokens(vocab)
        pad_n = seq_len - len(tokens)
        assert pad_n >= 0, (
            f"Problem {i} has {len(tokens)} tokens, exceeds seq_len={seq_len}"
        )
        padded = [vocab.PAD] * pad_n + tokens
        seqs[i] = torch.tensor(padded, dtype=torch.long)

    return seqs, n_result


def split_problems(generator: ProblemGenerator, vocab: Vocab,
                   n_train: int = 5000, n_test: int = 1000,
                   test_fraction: float = 0.2, seq_len: int = 48,
                   seed: int = 42, min_for_split: int = 30
                   ) -> Dict[str, Any]:
    """Generate train/test data with held-out problem splits.

    Enumerates all problems, splits by spec (not by sample), then generates
    training and test samples from disjoint spec sets.
    """
    import random as _random

    all_specs = generator.enumerate_all()
    rng = _random.Random(seed)
    shuffled = list(all_specs)
    rng.shuffle(shuffled)

    if len(shuffled) < min_for_split:
        train_specs = shuffled
        test_specs = shuffled
    else:
        n_held = max(1, int(len(shuffled) * test_fraction))
        test_specs = shuffled[:n_held]
        train_specs = shuffled[n_held:]

    train_problems = generator.generate(train_specs, n_train, vocab)
    test_problems = generator.generate(test_specs, n_test, vocab)

    train_seqs, n_result = problems_to_tensors(train_problems, vocab, seq_len)
    test_seqs, _ = problems_to_tensors(test_problems, vocab, seq_len)

    return dict(
        train_seqs=train_seqs,
        test_seqs=test_seqs,
        train_problems=train_problems,
        test_problems=test_problems,
        n_result_tokens=n_result,
        vocab_size=len(vocab),
        n_train_specs=len(train_specs),
        n_test_specs=len(test_specs),
    )
