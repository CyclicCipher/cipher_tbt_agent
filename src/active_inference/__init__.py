"""
Active Inference and Curiosity-Driven Learning

This module implements active inference mechanisms for data-efficient learning,
based on:
- Friston's Free Energy Principle and Active Inference
- Oudeyer et al.'s Learning Progress framework
- Schmidhuber's Compression Progress theory

Key Components:
- LearningProgressTracker: Tracks per-sample learning dynamics
- ActiveCurriculumManager: Selects samples based on epistemic value
- ExpectedFreeEnergyCalculator: Computes EFE for sample selection
- ActiveInferenceDiagnostics: Visualizes and analyzes training dynamics
"""

from .learning_progress import LearningProgressTracker, SampleStats
from .curriculum_manager import ActiveCurriculumManager
from .expected_free_energy import (
    ExpectedFreeEnergyCalculator,
    compute_information_gain,
    compute_expected_information_gain,
)
from .diagnostics import ActiveInferenceDiagnostics

__all__ = [
    'LearningProgressTracker',
    'SampleStats',
    'ActiveCurriculumManager',
    'ExpectedFreeEnergyCalculator',
    'compute_information_gain',
    'compute_expected_information_gain',
    'ActiveInferenceDiagnostics',
]
