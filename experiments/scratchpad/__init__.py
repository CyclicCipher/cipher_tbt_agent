"""General-purpose scratchpad framework for curriculum learning.

Model-agnostic: defines problems, rubrics, grading, and sequence formatting.
Any model that consumes/produces integer token sequences can use this.

Structure of a scratchpad sequence:
    [PAD...] [question tokens] [WORK] [solution step tokens] [NOTE] [notepad tokens]

- Question: the problem input (read-only for the model)
- Work area: step-by-step solution graded by rubric criteria
- Notepad: free scratch space for the model (not graded, future use)
"""

from .framework import (
    Vocab,
    Step,
    Problem,
    Grader,
    ProblemGenerator,
    Stage,
    Curriculum,
    problems_to_tensors,
    split_problems,
)

__all__ = [
    'Vocab', 'Step', 'Problem', 'Grader', 'ProblemGenerator',
    'Stage', 'Curriculum', 'problems_to_tensors', 'split_problems',
]
