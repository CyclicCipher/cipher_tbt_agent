"""L2/3: content (ARCHITECTURE.md section 1): what is at each place.

The content at a place is what came next there, by sensor: counts of the inputs, keyed by the place's SDR. It is the
column's prediction of the next input from a sensor (P2): the input of that sensor that most often came next here.
"""
from arc_agent_v2.HDClibrary import Associations


class L23:
    def __init__(self, d, w):
        self.d, self.w = d, w
        self.clear()

    def clear(self):
        self.content = Associations(self.d, theta=self.w // 2)

    def learn(self, place, sensor, x, count=1):
        self.content.store(place, (sensor, x), count)

    def predict(self, place, sensor="symbols"):
        if place is None:
            return None
        came = {x: n for (s, x), n in self.content.recall(place).items() if s == sensor}
        return max(came, key=came.get) if came else None
