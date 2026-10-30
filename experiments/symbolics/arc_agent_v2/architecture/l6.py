"""L6: the coordinate system (ARCHITECTURE.md section 1): where the column is in its frame.

The place is an SDR. It means nothing in itself: it is a place because moves lead to it and from it (L5), and content
is remembered at it (L2/3). Every episode starts at the same place (P1). The place is None when the column is lost: it
made a move its frame doesn't know, and it stays lost until the next episode.

**Part 5:** where the frame is factored, L6 also holds where the column is as (phase, coordinate). That pair covers
places never visited, because a coordinate moves by shifts and a phase by its transitions.

**Part 9:** that factored state is a union of hypotheses, (phase, coordinate) -> log-weight: a posterior, which the
column updates by Bayes' rule and widens by bursting (column.py). `fac` is its most probable hypothesis.
"""


class L6:
    def __init__(self):
        self.start = None                               # the frame's start place, once the frame exists
        self.place = None
        self.fac_start = None                           # part 5: (phase, coordinate code) where episodes start
        self.hyps = {}                                  # part 9: (phase, coordinate code) -> log-weight

    def reset(self):
        self.place = self.start
        self.hyps = {self.fac_start: 0.0} if self.fac_start is not None else {}

    @property
    def fac(self):
        """The most probable (phase, coordinate), or None."""
        return max(self.hyps, key=self.hyps.get) if self.hyps else None

    def lost(self):
        return self.place is None
