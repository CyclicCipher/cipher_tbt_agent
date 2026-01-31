"""Vision processing utilities for predictive coding agent."""

from .retinal_preprocessing import (
    RetinalPreprocessor,
    flatten_visual_input,
    get_visual_input_size
)

__all__ = [
    'RetinalPreprocessor',
    'flatten_visual_input',
    'get_visual_input_size'
]
