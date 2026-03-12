"""conftest.py — pytest configuration for symbolic_ai_v2 tests.

Markers:
  no_ait — legacy marker preserved for compatibility.  Originally disabled an
            ActiveInferenceTracker autouse fixture; the MorphismGraph and
            ActiveInference layer have been removed (Phase G / ROADMAP_REDESIGN
            §IV.5).  The marker is kept so that existing @pytest.mark.no_ait
            decorations do not produce PytestUnknownMarkWarning.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "no_ait: legacy marker (no-op since MorphismGraph removal). "
        "Kept to suppress PytestUnknownMarkWarning on test modules that "
        "predate the Phase G cleanup.",
    )
