"""Optimizers for predictive coding networks."""

from .muon import Muon, MuonWithActivityReg
from .stable_prospective import StableProspectiveLearning

__all__ = ['Muon', 'MuonWithActivityReg', 'StableProspectiveLearning']
