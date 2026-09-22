"""ziplib — the library both BrainBuilder and ZipLearn import (DESIGN §21.2; the module spec).

Re-exports, and nothing else: `price` (the module), `Store`, `Layout`, `Codec`, `INSTRUCTIONS`, `Blueprint`, `Arch`,
`Brain`. The imports are LAZY (PEP 562 module `__getattr__`): the package imports even while a sibling module is still
being written, and `from ziplib import Layout` only imports `ziplib.layout`.
"""
from __future__ import annotations

import importlib

_EXPORTS = {
    "price": ("ziplib.price", None),
    "Store": ("ziplib.store", "Store"),
    "Layout": ("ziplib.layout", "Layout"),
    "Codec": ("ziplib.codec", "Codec"),
    "INSTRUCTIONS": ("ziplib.instructions", "INSTRUCTIONS"),
    "Blueprint": ("ziplib.blueprint", "Blueprint"),
    "Arch": ("ziplib.blueprint", "Arch"),
    "Brain": ("ziplib.brain", "Brain"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'ziplib' has no attribute {name!r}")
    module_name, attr = _EXPORTS[name]
    module = importlib.import_module(module_name.replace("ziplib", __name__, 1))
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value                                           # cache: the next access is a plain lookup
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
