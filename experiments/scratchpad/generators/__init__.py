"""Problem generators for the scratchpad framework.

Each generator implements ProblemGenerator and produces Problem instances
with appropriate rubric steps for a specific domain.
"""

from .counting import QueryCountingGenerator, CombinedCountingGenerator
from .arithmetic import (
    SingleDigitArithmeticGenerator,
    TwoDigitSingleArithmeticGenerator,
    TwoDigitArithmeticGenerator,
)
from .syntax import (
    PosTagGenerator,
    NpChunkGenerator,
    PpChunkGenerator,
    VpChunkGenerator,
    ClauseStructureGenerator,
)

__all__ = [
    'QueryCountingGenerator',
    'CombinedCountingGenerator',
    'SingleDigitArithmeticGenerator',
    'TwoDigitSingleArithmeticGenerator',
    'TwoDigitArithmeticGenerator',
    'PosTagGenerator',
    'NpChunkGenerator',
    'PpChunkGenerator',
    'VpChunkGenerator',
    'ClauseStructureGenerator',
]
