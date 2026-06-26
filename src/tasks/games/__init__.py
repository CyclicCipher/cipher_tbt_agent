"""Concrete games for the ARC-AGI-3 replica."""

from .lockpath import LockPath
from .multikey import MultiKey
from .sokoban import Sokoban
from .collectall import CollectAll
from .toggle import Toggle

__all__ = ["LockPath", "MultiKey", "Sokoban", "CollectAll", "Toggle"]
