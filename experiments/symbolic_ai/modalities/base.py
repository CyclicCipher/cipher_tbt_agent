"""Abstract base class for input modalities.

A Modality is a plugin that extends the process language interpreter with
new primitive functions specific to a sensory channel (e.g. vision, audio).

Usage pattern:
    from modalities.vision import VisionModality
    ai = SymbolicAI(graph, modalities=[VisionModality()])

    # Now process lines may call img_load, img_dog, img_gabor, etc.
    concept.process = [
        'img = img_load(path)',
        'gray = img_to_gray(img)',
        'edges = img_dog(gray, 1.0, 2.0)',
        'emit(edges)',
    ]

Design contract:
    - Modality.primitives maps str → Callable.
    - Each callable receives already-evaluated arguments (Python objects),
      same as built-in process functions.
    - Primitives must be pure or have bounded side effects — they run inside
      the interpreter's expression evaluator.
    - Primitive names should be prefixed by modality (e.g. 'img_', 'aud_')
      to avoid collisions with built-in names.
    - preprocess(raw) converts external data into the modality's native
      representation before it enters the process language.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List


class Modality(ABC):
    """Abstract base for sensory modality plugins.

    Subclass this and implement `name` and `primitives`.  Optionally
    override `preprocess` if the modality needs to normalise raw inputs
    before they are handed to process language expressions.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'vision', 'audio'."""
        ...

    @property
    @abstractmethod
    def primitives(self) -> Dict[str, Callable]:
        """Map from function name to callable.

        Each callable receives positional args as Python objects.
        Example:
            {'img_load': self._img_load,
             'img_to_gray': self._img_to_gray, ...}
        """
        ...

    def preprocess(self, raw: Any) -> Any:
        """Convert raw external input to the modality's native form.

        Default: identity (no preprocessing).  Override in subclasses.
        For vision: raw might be a file path or a numpy array; preprocess
        returns a normalised float32 array in [0, 1].
        """
        return raw

    def type_names(self) -> List[str]:
        """CTKG input type names this modality handles.

        Used for documentation and future type-checking integration.
        Default: [self.name + '_input'].
        """
        return [self.name + '_input']
