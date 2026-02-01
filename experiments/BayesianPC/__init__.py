"""
Bayesian Predictive Coding

Correct implementation following Algorithm 1 from:
"Bayesian Predictive Coding" (Tschantz et al., 2025, arXiv:2503.24016)

Key features:
- Value nodes: MAP estimates (point values), optimized via gradient descent
- Weights: Matrix Normal Wishart posterior distributions
- Learning: Closed-form Hebbian updates (Equation 7)
- Architecture: Weights OUTSIDE activation function for conjugacy
"""

from .bayesian_pc_layer import BayesianPCLayer, BayesianPCNetwork
from .bayesian_pc_trainer import BayesianPCTrainer

__all__ = ['BayesianPCLayer', 'BayesianPCNetwork', 'BayesianPCTrainer']
