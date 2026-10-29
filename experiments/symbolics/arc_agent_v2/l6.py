"""L6: the coordinate system (ARCHITECTURE.md section 1): where the column is in its frame.

The place is an SDR. It means nothing in itself: it is a place because moves lead to it and from it (L5), and content
is remembered at it (L2/3). Every episode starts at the same place (P1). The place is None when the column is lost: it
made a move its frame doesn't know, and it stays lost until the next episode.
"""


class L6:
    def __init__(self):
        self.start = None                               # the frame's start place, once the frame exists
        self.place = None

    def reset(self):
        self.place = self.start

    def lost(self):
        return self.place is None
