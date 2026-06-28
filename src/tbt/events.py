"""Event boundaries -- segment the experience stream at discontinuities the agent's actions cannot explain.

A game-over/reset/level-change/scene-cut is a full-frame discontinuity, NOT a normal action effect. Feeding it to
the operator learner corrupts the operators -- a single unhandled death made every cell look dynamic in a live
probe. The biological analogue is the reafference principle (von Holst & Mittelstaedt 1950): the brain uses the
efference copy to predict the change its OWN action causes and flags change it cannot account for as exafferent /
world-caused; a large unexplained change is an EVENT BOUNDARY (event-segmentation theory, Zacks et al. 2007), across
which the brain does NOT path-integrate but RE-LOCALISES (relocalisation after sleep, the kidnapped-robot problem;
hippocampal map reactivation, Jezek et al. 2011).

So a transition is a boundary when its change is anomalously large versus the agent's running NORMAL per-action
effect (the reafference test), or a lifecycle cue says so (the score/level changed, or the state is GAME_OVER/WIN).
Boundaries are EXCLUDED from operator learning, trigger re-localisation, and a death carries an aversive value.
This is the lifecycle handling the real games need, with no hand-coded "if GAME_OVER" rule -- the prediction error
detects it, and the lifecycle flag (raw data the benchmark provides) is used when available. Pure stdlib.
"""

from __future__ import annotations


class EventSegmenter:
    """Online event-boundary detection from the reafference signal (change magnitude vs a running normal) plus
    lifecycle cues, so operator learning sees only clean within-event transitions. `anomaly_factor` = how many
    times the running-normal change a transition must exceed to count as a boundary (the exafference threshold);
    `warmup` = steps to learn the normal before the magnitude test fires."""

    def __init__(self, anomaly_factor: float = 4.0, warmup: int = 4):
        self.anomaly_factor = anomaly_factor
        self.warmup = warmup
        self.n = 0                                            # within-event steps seen (the running-normal sample)
        self.mean_change = 0.0                               # running mean of the NORMAL per-action change magnitude

    def is_boundary(self, change_size: float, lifecycle: bool = False) -> bool:
        """Is this transition an event boundary? True if a lifecycle cue says so, or (after warmup) the change is
        more than `anomaly_factor` times the running normal -- a change the agent's action cannot explain. A
        boundary does NOT update the running normal (so it cannot inflate it and mask the next boundary)."""
        anomalous = self.n >= self.warmup and change_size > self.anomaly_factor * max(self.mean_change, 1.0)
        if lifecycle or anomalous:
            return True
        self.n += 1                                          # a normal within-event step: fold it into the normal
        self.mean_change += (change_size - self.mean_change) / self.n
        return False
