"""test_touch.py -- the BODY surface + the SKIN peripheral (touch.py), the touch modality's front-end.

The self's body has a surface (its outward faces); the skin reports, per face, the colour pressed against it -- a wall, a
block, or nothing. Grounded on the real Push frame, since contact is what the push mechanic turns on.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.touch import body_surface, sense_contact, contact_toward   # noqa: E402
from tasks.games.push import Push                                   # noqa: E402

N, S, W, E = (0, -1), (0, 1), (-1, 0), (1, 0)


def test_single_cell_body_has_four_faces():
    faces = body_surface({(1, 1)})
    assert len(faces) == 4 and {n for _, n in faces} == {N, S, W, E}, faces


def test_multicell_body_excludes_internal_faces():
    """A 2-cell horizontal body has 6 outward faces, not 8 -- the shared internal faces are inside the body, not skin."""
    faces = body_surface({(1, 1), (2, 1)})
    assert ((1, 1), E) not in faces and ((2, 1), W) not in faces, "internal faces must be excluded"
    assert len(faces) == 6, faces


def test_skin_reads_the_contact_around_the_agent():
    """On Push L0 the agent starts at (1,1): walls to its North and West, empty floor South and East -- felt exactly."""
    game = Push(); game.load_level(0)
    grid = game.render()[0]
    contact = sense_contact(grid, {game.agent})
    assert game.agent == (1, 1)
    assert contact[((1, 1), N)] == 1 and contact[((1, 1), W)] == 1, "walls felt N and W"
    assert contact[((1, 1), S)] is None and contact[((1, 1), E)] is None, "floor felt as no contact"


def test_leading_face_feels_the_block_it_would_push():
    """Standing just left of the block, a move EAST presses into the block (colour 6); moving WEST presses into nothing that
    way -- the leading-face contact is exactly what the push dynamics needs to condition on."""
    game = Push(); game.load_level(0)
    game.agent = (1, 2)                                   # left of the block at (2,2)
    contact = sense_contact(game.render()[0], {game.agent})
    assert contact_toward(contact, E) == 6, "moving east presses into the block"
    assert contact_toward(contact, N) is None, "moving north presses into nothing"


if __name__ == "__main__":
    game = Push(); game.load_level(0); game.agent = (1, 2)
    for face, feat in sense_contact(game.render()[0], {game.agent}).items():
        print(f"face {face}: contact={feat}")
