"""Event boundaries -- segment the experience stream at discontinuities the agent's actions cannot explain.

A game-over/reset/level-change/scene-cut is a full-frame discontinuity, NOT a normal action effect. Feeding it to
the operator learner corrupts the operators -- a single unhandled death made every cell look dynamic in a live
probe. The biological analogue is the reafference principle (von Holst & Mittelstaedt 1950): the brain uses the
efference copy to predict the change its OWN action causes and flags change it cannot account for as exafferent /
world-caused; a large unexplained change is an EVENT BOUNDARY (event-segmentation theory, Zacks et al. 2007), across
which the brain does NOT path-integrate but RE-LOCALISES (relocalisation after sleep, the kidnapped-robot problem;
hippocampal map reactivation, Jezek et al. 2011).

So a transition is a boundary when its UNEXPLAINED change is anomalously large versus the agent's running NORMAL, or a
lifecycle cue says so (the score/level changed, or the state is GAME_OVER/WIN). The caller supplies the unexplained
magnitude -- the REAFFERENCE RESIDUAL: the observed changed cells minus what the learned operators predict the tracked
objects would change (vacate + enter). Before any operator exists that residual equals the raw change (the magnitude
bootstrap); as the operators sharpen it shrinks for explained motion, so a big EXPLAINED move stops looking like a
boundary while a scene-cut's huge unexplained change still fires -- the prediction error and the operators co-bootstrap
(`play.py` computes the residual; `forward.predict_cells` supplies the prediction). Boundaries are EXCLUDED from
operator learning, trigger re-localisation, and a death carries an aversive value. No hand-coded "if GAME_OVER" rule --
the residual detects it, and the lifecycle flag (raw data the benchmark provides) is used when available. Pure stdlib.
"""

from __future__ import annotations


class EventSegmenter:
    """Online event-boundary detection from the reafference signal (the UNEXPLAINED change magnitude vs a running
    normal -- the caller passes the residual: observed change minus the operators' prediction) plus lifecycle cues, so
    operator learning sees only clean within-event transitions. `anomaly_factor` = how many times the running-normal
    residual a transition must exceed to count as a boundary (the exafference threshold); `warmup` = steps to learn the
    normal before the test fires."""

    def __init__(self, anomaly_factor: float = 4.0, warmup: int = 4):
        self.anomaly_factor = anomaly_factor
        self.warmup = warmup
        self.n = 0                                            # within-event steps seen (the running-normal sample)
        self.mean_change = 0.0                               # running mean of the NORMAL per-action change magnitude

    def is_boundary(self, change_size: float, lifecycle: bool = False) -> bool:
        """Is this transition an event boundary? `change_size` is the UNEXPLAINED change magnitude (the reafference
        residual). True if a lifecycle cue says so, or (after warmup) the residual is more than `anomaly_factor` times
        the running normal -- change the agent's operators cannot explain. A boundary does NOT update the running normal
        (so it cannot inflate it and mask the next boundary)."""
        anomalous = self.n >= self.warmup and change_size > self.anomaly_factor * max(self.mean_change, 1.0)
        if lifecycle or anomalous:
            return True
        self.n += 1                                          # a normal within-event step: fold it into the normal
        self.mean_change += (change_size - self.mean_change) / self.n
        return False
