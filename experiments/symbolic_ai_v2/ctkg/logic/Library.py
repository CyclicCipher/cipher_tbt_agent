"""
Library — foundational data types, classes, and functions for the CTKG.

This module is the only place where core mathematical and categorical structures
are defined. All other modules in ctkg/logic/ import from here.

Contents (to be implemented):
- Node / Edge / Morphism data types
- Category-theoretic operations: composition, identity, functor application
- Type system: objects, arrows, natural transformations
- MDL scoring primitives
- Expression trees (Expr) with opaque integer operator IDs
- Observation types: discrete sequences, continuous (input, output) pairs
- Helper math: OLS fitting, MSE, log-probability
"""
from __future__ import annotations
