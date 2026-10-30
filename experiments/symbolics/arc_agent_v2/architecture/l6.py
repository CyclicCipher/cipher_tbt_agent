"""L6: the coordinate system (ARCHITECTURE.md section 1): where the column is in its frame.

The place is an SDR. It means nothing in itself: it is a place because moves lead to it and from it (L5), and content
is remembered at it (L2/3). Every episode starts at the same place (P1). The place is None when the column is lost: it
made a move its frame doesn't know, and it stays lost until the next episode.

**Part 5:** where the frame is factored, L6 also holds where the column is as (phase, coordinate). That pair covers
places never visited, because a coordinate moves by shifts and a phase by its transitions.

**Part 9:** that state is a union of hypotheses, a posterior, which the column updates by Bayes' rule and widens by
bursting (column.py). **Part 12:** a hypothesis is (object, phase, coordinate), or (object, None, place) where the frame
isn't factored. `fac` is the most probable hypothesis's (phase, coordinate).
"""


class L6:
    def __init__(self):
        self.start = None                               # the frame's start place, once the frame exists
        self.place = None
        self.fac_start = None                           # part 5: (phase, coordinate code) where episodes start
        self.hyps = {}                                  # hypothesis -> log-weight (the column sets it at a reset)

    @property
    def fac(self):
        """The most probable (phase, coordinate), or None."""
        return max(self.hyps, key=self.hyps.get)[1:] if self.hyps else None

    def lost(self):
        return self.place is None
