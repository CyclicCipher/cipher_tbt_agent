"""L4: the input at the place (ARCHITECTURE.md section 1; the user's "place / arguments?").

The input arrives here, at the column's current place, and meets what the column expected of it. A surprise is an input
the column expected less than it would have knowing nothing (its likelihood under the column's state below the prior's,
information.py; part 12). The count of surprises is the column's prediction error.
"""


class L4:
    def __init__(self, w):
        self.w, self.input, self.surprises, self.seen = w, None, 0, 0

    def sense(self, x, logp, prior_logp):
        """Take input x, with its log-likelihood under the column's state and under the prior; True if a surprise."""
        self.input = x
        self.seen += 1
        surprise = logp < prior_logp
        self.surprises += surprise
        return surprise
