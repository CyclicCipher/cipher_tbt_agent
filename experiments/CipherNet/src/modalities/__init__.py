"""Sensor modality adapters.

Each modality handles the input-specific pipeline:
  1. preprocess(raw_input)           — once per stimulus (e.g. DoG filter)
  2. encode(prepared, fixation)      — once per fixation

The cortex only sees (sdr, location_key) pairs; it never imports
modality-specific code (Eye, librosa, tokenizers, etc.).
"""
from modalities.base import SensorModality
from modalities.vision import VisualModality

__all__ = ['SensorModality', 'VisualModality']
