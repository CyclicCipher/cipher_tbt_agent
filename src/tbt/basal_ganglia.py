"""The basal ganglia — the value-driven ACTION SELECTOR (ARCHITECTURE.md §3, the loop's "select" arm).

The cortex is value-free; the BG is the ONE place arbitration between competing options is allowed (rule 4 — it is the
brain's arbitration organ). It selects by learned value (dopamine-RPE), Go/NoGo opponent weights (OpAL), with an STN
"hold-your-horses" commitment so near-tied options don't trigger a dithering switch. Grounds: reference_basal_ganglia
(Collins & Frank OpAL; Frank 2006 STN; Redgrave/Gurney/Prescott selection-by-disinhibition). The OpponentActor is reused
verbatim from the legacy `basal_ganglia.py` (it is sound + cited); `BasalGanglia` is a thin selector over it.

SCOPE (2026-07-10, first slice): the value comes from the OpAL actor the reward-RPE trains directly — for an
IMMEDIATE-reward task the RPE is a simple centered reward (a proper TD critic / `reward.py` comes with a task that needs
multi-step value). The OTHER BG role — MoE column-allocation (which column models which structure, by emergent
competition) — is deferred to a multi-column-pool task. Pure stdlib.
"""

from __future__ import annotations

import random


class OpponentActor:
    """The OpAL opponent Go/NoGo actor (Collins & Frank; reference_basal_ganglia) — the basal-ganglia ACTOR the critic's
    dopamine RPE trains. TWO opponent weights: **Go** learns BENEFITS (potentiated by DA bursts, δ>0), **NoGo** learns COSTS
    (potentiated by DA dips, δ<0), each by the THREE-FACTOR Hebbian rule (the weight scales its OWN update —
    `w ← w + α·w·δ`), so the two SPECIALIZE by reward range rather than duplicating one value. Choice value
    `Act = βg·Go − βn·NoGo`; **tonic dopamine `ρ`** sets `βg = β·max(0,1+ρ)`, `βn = β·max(0,1−ρ)` — the explore/exploit +
    vigor gain (ρ>0 rich → Go/exploit; ρ<0 lean → NoGo/avoid). `N` gives principled AVERSION a single reward cannot represent.

    PHASE 4 — DISTRIBUTED (per-bit) READ-OFF. The weights are held per `(SDR bit, action)`, not per whole-context key, and the
    Go/NoGo value is the SUM over the context's active bits: `Go(s,a) = Σ_{bit∈s} WG[bit,a]`. So SDR OVERLAP = value SIMILARITY
    — value learned in one context TRANSFERS to another that shares bits (`reference_sdr_encoder_library`), which the old
    exact-frozenset keying could not do (a novel context was a fresh key with no value). The three-factor OpAL specialization
    is preserved PER BIT; the update is normalised by |active bits| so the effective rate is stable regardless of SDR sparsity
    (the same fix the value critic needed). Weights seed at `init>0` (three-factor needs a non-zero seed); Go=NoGo, ρ=0 ⇒
    contribution 0, so it is behaviour-neutral until the RPE trains them apart."""

    def __init__(self, alpha_g: float = 0.1, alpha_n: float = 0.1, beta: float = 1.0, init: float = 1.0):
        self.alpha_g, self.alpha_n, self.beta, self.init = alpha_g, alpha_n, beta, init
        self.WG: dict = {}                                   # (bit, action) -> Go weight  (benefits, distributed)
        self.WN: dict = {}                                   # (bit, action) -> NoGo weight (costs, distributed)

    def _go_nogo(self, context, action):
        """The distributed read-off: Go/NoGo summed over the context's active bits (an unseen bit contributes `init`)."""
        g = sum(self.WG.get((b, action), self.init) for b in context)
        n = sum(self.WN.get((b, action), self.init) for b in context)
        return g, n

    def learn(self, context, action, delta: float) -> None:
        """Three-factor OpAL update, PER active bit, by the RPE `delta`: Go up on δ>0 (a benefit), NoGo up on δ<0 (a cost).
        Each bit's weight scales its own update (`w ← w + α·w·δ/|context|`), clamped ≥0. Because it is per-bit, the update
        credits the FEATURES present — so a shared feature accrues value across every context it appears in."""
        context = list(context)
        m = max(1, len(context))
        for b in context:
            g, n = self.WG.get((b, action), self.init), self.WN.get((b, action), self.init)
            self.WG[(b, action)] = max(g + self.alpha_g * g * delta / m, 0.0)
            self.WN[(b, action)] = max(n - self.alpha_n * n * delta / m, 0.0)

    def act_value(self, context, action, rho: float = 0.0) -> float:
        """The actor's salience contribution: `Act = βg·Go − βn·NoGo`, with tonic-DA `rho` setting the gains (rho>0 rich →
        Go/benefits dominate = exploit/vigor; rho<0 lean → NoGo/costs dominate = avoid)."""
        g, n = self._go_nogo(context, action)
        return self.beta * max(0.0, 1.0 + rho) * g - self.beta * max(0.0, 1.0 - rho) * n


class BasalGanglia:
    """The selector: pick the highest-value action (Go/NoGo via the OpAL actor), with STN commitment. Default-closed —
    only the selected action is disinhibited (the thalamus enacts it). Trained by `learn(context, action, rpe)`."""

    def __init__(self, alpha_g: float = 0.2, alpha_n: float = 0.2, commit_frac: float = 0.0, seed: int = 0):
        self.actor = OpponentActor(alpha_g=alpha_g, alpha_n=alpha_n)
        self.commit_frac = float(commit_frac)               # STN margin (0 = pure argmax; >0 = hysteresis for sequences)
        self._rng = random.Random(seed)

    def select(self, context, n_actions: int, rho: float = 0.0, current=None, explore: float = 0.0) -> int:
        """Disinhibit the highest OpAL-value action for this context (Go − NoGo). `explore` = ε random (for training an
        immediate-reward task, so every action is sampled). `current` + `commit_frac` = the STN 'hold your horses' brake
        for sequential decisions (hold the current action unless a competitor wins by a margin; off by default)."""
        if explore > 0.0 and self._rng.random() < explore:
            return self._rng.randrange(n_actions)
        vals = [self.actor.act_value(context, a, rho) for a in range(n_actions)]
        m = max(vals)
        ties = [a for a in range(n_actions) if vals[a] == m]
        best = self._rng.choice(ties) if len(ties) > 1 else ties[0]
        if current is None or not (0 <= current < n_actions):
            return best
        margin = self.commit_frac * max((abs(v) for v in vals), default=0.0)   # STN: scale-free, delays a near-tied switch
        return best if vals[best] - vals[current] > margin else current

    def learn(self, context, action, rpe: float) -> None:
        """Dopamine-RPE update of the chosen (context, action). `rpe` is the reward prediction error (for an immediate
        reward, a centered reward — δ>0 rewarded → Go, δ<0 punished → NoGo)."""
        self.actor.learn(context, action, rpe)
