"""L4: the input at the place (ARCHITECTURE.md section 1; the user's "place / arguments?").

The input arrives here, at the column's current place, and meets the prediction L2/3 made for it. A surprise is an
input the prediction didn't match (less than half its bits). The count of surprises is the column's prediction error.
"""
from arc_agent_v2.architecture.HDClibrary import overlap


class L4:
    def __init__(self, w):
        self.w, self.input, self.surprises, self.seen = w, None, 0, 0

    def sense(self, x, predicted):
        """Take input x; True if the prediction didn't match it."""
        self.input = x
        self.seen += 1
        surprise = predicted is None or overlap(x, predicted) < self.w // 2
        self.surprises += surprise
        return surprise
