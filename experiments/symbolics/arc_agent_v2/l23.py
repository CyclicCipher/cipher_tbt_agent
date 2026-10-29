"""L2/3: content (ARCHITECTURE.md section 1): what is at each place.

The content at a place is what followed there: counts of the inputs that came next, keyed by the place's SDR. It is the
column's prediction of the next input (P2): the input that most often came next at this place.
"""
from arc_agent_v2.HDClibrary import Associations


class L23:
    def __init__(self, d, w):
        self.d, self.w = d, w
        self.clear()

    def clear(self):
        self.content = Associations(self.d, theta=self.w // 2)

    def learn(self, place, x, count=1):
        self.content.store(place, x, count)

    def predict(self, place):
        return None if place is None else self.content.best(place)
