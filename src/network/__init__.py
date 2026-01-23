"""
Predictive coding network implementation.

Implements a layered predictive coding architecture with prospective learning.
"""

from .neuron import TwoCompartmentNeuron
from .layer import PredictiveCodingLayer
from .backbone import BackboneNetwork

__all__ = [
    'TwoCompartmentNeuron',
    'PredictiveCodingLayer',
    'BackboneNetwork',
]

__version__ = '0.1.0'
