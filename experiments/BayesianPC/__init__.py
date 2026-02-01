"""
Bayesian Predictive Coding Experiment

Tests Bayesian inference with uncertainty quantification on MNIST.

Key questions:
1. Does KL divergence improve generalization?
2. Does uncertainty tracking help with data efficiency?
3. How does uncertainty evolve during training?
4. Can we detect when the model is uncertain (for active learning)?
"""

from .bayesian_pc_layer import BayesianPCLayer, BayesianPCNetwork

__all__ = ['BayesianPCLayer', 'BayesianPCNetwork']
