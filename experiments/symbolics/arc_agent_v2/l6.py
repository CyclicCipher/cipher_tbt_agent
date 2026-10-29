"""L6: the coordinate system (ARCHITECTURE.md section 1): where the column is in its frame.

The place is an SDR. It means nothing in itself: it is a place because moves lead to it and from it (L5), and content
is remembered at it (L2/3). Every episode starts at the same place (P1). The place is None when the column is lost: it
made a move its frame doesn't know, and it stays lost until the next episode.

**Part 5:** where the frame is factored, L6 also holds where the column is as (phase, coordinate). That pair covers
places never visited, because a coordinate moves by shifts and a phase by its transitions.
"""


class L6:
    def __init__(self):
        self.start = None                               # the frame's start place, once the frame exists
        self.place = None
        self.fac_start = self.fac = None                # part 5: (phase, coordinate code)

    def reset(self):
        self.place = self.start
        self.fac = self.fac_start

    def lost(self):
        return self.place is None
