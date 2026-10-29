"""L5: the operation (ARCHITECTURE.md section 1): what each input does to the place.

Every input is a move (P1). A move is remembered as an association from the conjunction of (place, input), a code
that depends on both jointly (HDClibrary.Conjunction), to the place it leads to. The moves come from the frame
(frameLearning.py), observed and filled in alike. L5 doesn't know what any input means: it only knows where an input
has led, or must lead, from each place.
"""
from arc_agent_v2.HDClibrary import Associations, Conjunction


class L5:
    def __init__(self, d, w, seed=0):
        self.key = Conjunction(d, w, seed)
        self.moves = Associations(d, theta=w // 2)

    def clear(self):
        self.moves = Associations(self.key.d, theta=self.key.w // 2)

    def learn(self, place, x, nxt):
        self.moves.store(self.key(place, x), nxt)

    def move(self, place, x):
        """Where input x leads from place, or None if the frame doesn't know."""
        return self.moves.best(self.key(place, x))
