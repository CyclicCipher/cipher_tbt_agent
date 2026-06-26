"""The basal ganglia — the gate SELECTOR (architecture doc §4, §12.3). Pure stdlib.

Columns are a POOL of identical units; the BG decides which column handles which structure, by learned value
(dopamine-RPE) with load-balancing so one column cannot win everything (MoE gating). Roles are NOT assigned —
which column becomes (say) the digit line vs the position line EMERGES from competition + reinforcement
(Mountcastle: specialization follows inputs, not design; do not hand-assign column roles).

Mechanism, minimal but faithful to §4:
  default-closed   : a column is gated only when selected (GPi/SNr tonic inhibition).
  Go   (direct/D1) : disinhibit the highest-AFFINITY column for this structure.
  NoGo (indirect/D2): suppress columns already recruited to other structures (a per-column LOAD penalty) —
                      this is what stops one column swallowing every structure.
  dopamine (RPE)   : after a column models a structure, raise that structure's affinity for it — so the gate
                      routes the structure back to its specialist next time.
Symmetry of identical columns is broken by a tiny random affinity init (random niches)."""

from __future__ import annotations

import random


class BasalGanglia:
    def __init__(self, n_columns: int, balance: float = 0.5, lr: float = 1.0, seed: int = 0):
        self.n = n_columns
        self.balance = balance                               # NoGo: penalty per distinct structure already on a column
        self.lr = lr                                         # dopamine learning rate
        self._rng = random.Random(seed)
        self.aff: dict = {}                                  # structure-key -> [affinity per column]
        self.assigned = [set() for _ in range(n_columns)]    # distinct structures recruited per column
        self.load = [0.0] * n_columns
        self._opt_aff: dict = {}                             # subgoal-option -> learned affinity (the gate, below)

    def _aff(self, key):
        if key not in self.aff:                              # tiny random init breaks the symmetry of identical columns
            self.aff[key] = [self._rng.uniform(0, 1e-3) for _ in range(self.n)]
        return self.aff[key]

    def select(self, key):
        """Disinhibit the highest-value, least-loaded column for this structure (Go − NoGo)."""
        aff = self._aff(key)
        return max(range(self.n), key=lambda c: aff[c] - self.balance * self.load[c])

    def reinforce(self, key, column, value):
        """Dopamine-RPE: raise this structure's affinity for the column that modelled it well; recruit it."""
        self._aff(key)[column] += self.lr * value
        self.assigned[column].add(key)
        self.load[column] = len(self.assigned[column])       # load = number of DISTINCT structures on the column

    def gate(self, options, values, lr=0.3):
        """The SUBGOAL gate (doc §4): Go disinhibits the highest combined CRITIC value (reward.py) + learned
        affinity; dopamine-RPE nudges the chosen option's affinity toward its critic value (so the gate learns
        which subgoal is worth selecting, trained by the critic). `options` are hashable subgoal identities;
        `values` are reward.py's values of their outcomes. Returns the chosen index."""
        a = self._opt_aff
        i = max(range(len(options)), key=lambda j: values[j] + a.get(options[j], 0.0))
        o = options[i]
        a[o] = a.get(o, 0.0) + lr * (values[i] - a.get(o, 0.0))
        return i
