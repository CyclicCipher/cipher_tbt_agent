"""
Predictive coding network implementation.

Minimal standard implementation based on:
- Bogacz Group reference implementation
- Whittington & Bogacz (2017)
- Standard PC algorithm (not custom neuron designs)
"""

from .pc_layer import PCLayer, PCNetwork
from .pc_trainer import PCTrainer

# Old implementations (deprecated - see MISTAKES.md)
# from .neuron import TwoCompartmentNeuron
# from .layer import PredictiveCodingLayer
# from .backbone import BackboneNetwork

__all__ = [
    'PCLayer',
    'PCNetwork',
    'PCTrainer',
]

__version__ = '0.2.0'
