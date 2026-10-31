"""The cell types (LAB.md E0b): type A, human layer 2/3 (`l23`); type B, layer 5-like (`l5`)."""
from .l23 import l23
from .l5 import l5

TYPES = dict(A=l23, B=l5)
