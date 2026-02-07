"""
eBPC: Error-based Bayesian Predictive Coding

Combines ePC (Goemaere et al. 2025) with BPC (Tschantz et al. 2025):
- Inference: Error reparameterization with global backprop (no signal decay)
- Learning: Closed-form Hebbian updates via MNW posterior (Equation 7)
"""

from .ebpc_layer import eBPCLayer, eBPCNetwork
from .ebpc_trainer import eBPCTrainer

__all__ = ['eBPCLayer', 'eBPCNetwork', 'eBPCTrainer']
