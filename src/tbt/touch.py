"""touch.py -- the agent's BODY surface and the SKIN peripheral: the touch modality's front-end (notes/touch_and_body_design.md).

Contact is somatosensory, not geometric: it needs a body with a sensed surface. This gives the self a BODY (its occupied
cells) with a SURFACE (the outward faces), and a SKIN sense that reports, per surface face, what is in contact across it. It is
a sensor front-end -- exactly parallel to `perceive.segment` (the retina): it TRANSDUCES the world into a touch signal and
decides nothing. The touch COLUMN models that signal; the contact-DYNAMICS reads the leading face. No object semantics, no
identity here (`feedback_bitter_lesson`): just "a surface of colour c is pressed against this face."
"""

from __future__ import annotations

# The four face normals of a grid cell -- the outward directions of a body's surface, in screen coords (+y is down).
_NORMALS = ((0, -1), (0, 1), (-1, 0), (1, 0))   # N, S, W, E


def body_surface(body_cells) -> list:
    """The outward FACES of a body (a set of cells): each face is `(cell, normal)` where the neighbour across `normal` is NOT a
    body cell -- the body's boundary in its own layout. A single-cell body has all four faces; a multi-cell body's shared
    internal faces are excluded (they are inside the body, not skin). Deterministic order."""
    body = set(body_cells)
    faces = []
    for cell in sorted(body):
        for normal in _NORMALS:
            if (cell[0] + normal[0], cell[1] + normal[1]) not in body:
                faces.append((cell, normal))
    return faces


def sense_contact(grid, body_cells, background: int = 0) -> dict:
    """The SKIN sense: for each body-surface face, the contact FEATURE across it -- the colour of whatever occupies the abutting
    cell, or None if that cell is empty (background) or off the grid (no reading; a real frame edge would be a boundary contact,
    deferred). Returns `{(cell, normal): feature_or_None}`. Raw touch -- no grouping, no identity; the touch column does that."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    contact = {}
    for cell, normal in body_surface(body_cells):
        nx, ny = cell[0] + normal[0], cell[1] + normal[1]
        contact[(cell, normal)] = grid[ny][nx] if (0 <= nx < w and 0 <= ny < h and grid[ny][nx] != background) else None
    return contact


def contact_toward(contact: dict, direction) -> int:
    """The feature felt across the LEADING face -- a face whose normal points along `direction` (the way a move would press). It
    is what the body would push INTO under that motion; None if that face is free. `direction` is a unit step (the sign of the
    efference); for a single-cell body there is one face per direction. Multi-cell: the first pressed leading face."""
    d = (0 if direction[0] == 0 else (1 if direction[0] > 0 else -1),
         0 if direction[1] == 0 else (1 if direction[1] > 0 else -1))
    for (cell, normal), feature in contact.items():
        if normal == d and feature is not None:
            return feature
    return None
