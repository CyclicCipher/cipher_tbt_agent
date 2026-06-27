"""The one value-search planner — domain-general core of the agent (the EZ-V2 `V` + planning, consolidated).

It plans over a learned forward model `G` and an SR-frame column, by a depth-D model-rollout search whose leaf
value is a TD-learned `V` over a bound latent, shaped online by an opaque PRAGMATIC signal (progress toward `F`'s
learned goal). One model, every mechanic: navigation, a door (a state bit gates traversability), a push (a mover
coordinate), a multi-object goal — they differ only in the coords/gates/pragmatic that PERCEPTION hands in.

Everything domain-specific is opaque here:
  - `coords` — a factored state tuple (agent + each mover, egocentric, + state bits); the planner only indexes it;
  - the `column` + `cid` — the SR map; the planner only reads place codes;
  - `gates` — `{cell: bit-index}`; a move onto a gated cell is blocked while that coord-bit reads 1;
  - `pragmatic(coords) -> [0,1]` — `F`-progress, a callable.
There is no grid / colour / delta / door / push in this file — swap perception (even to a non-spatial game) and
this planner is unchanged. The latent binds a place code per position-slot ⊗ a tag for the bit-slots and is unit-
normalised; the value is updated by EZ-V2 multi-step (n-step) TD with potential-based pragmatic shaping.
"""

from __future__ import annotations

import random

import torch


class Value:
    """TD readout over a latent: V(z) = w · z; w is updated by the planner's n-step `flush`."""

    def __init__(self, d_mem: int, gamma: float = 0.9, alpha: float = 0.3):
        self.w = torch.zeros(d_mem)
        self.gamma = gamma
        self.alpha = alpha

    def value(self, z: torch.Tensor) -> float:
        return float(self.w @ z)


class ValuePlanner:
    def __init__(self, G, n_movers, n_bit, n_actions, gamma=0.9, alpha=0.3, depth=5, kappa=1.0, l=5, seed=0):
        self.G = G
        self.n_movers, self.n_bit, self.n_actions = n_movers, n_bit, n_actions
        self.gamma, self.alpha, self.depth, self.kappa, self.l = gamma, alpha, depth, kappa, l
        self.val = None
        self.rhat, self.term = {}, {}                          # (coords, action) -> reward, terminal (the G head)
        self._sv, self._zc, self._slotc = {}, {}, {}           # bit-combo->tag; (coords,focus)->latent; coords->cells
        self._focus = tuple(range(n_movers))                   # which movers the VALUE binds (default: all)
        self._stage_fn = lambda c: ()                          # coords -> the discrete STAGE tag (factors done)
        self._gen = torch.Generator().manual_seed(1234 + seed)
        self.rng = random.Random(seed)
        self._sel, self._traj = None, []
        self._col = self._cid = None

    def set_map(self, col, cid):
        """Bind the SR-frame column for the current state space (built by perception from its walkable set)."""
        if cid is not self._cid:
            self._col, self._cid, self._zc = col, cid, {}
        if self.val is None:
            self.val = Value(col.d_mem, self.gamma, self.alpha)

    def reset(self):                                            # new episode
        self._sel, self._traj = None, []

    # ---- the bound, normalised latent (place codes ⊗ a bit tag) -------------------------------------------
    def _slots(self, coords):
        s = self._slotc.get(coords)
        if s is None:
            ax, ay = coords[0], coords[1]
            s = [(ax, ay)] + [(ax + coords[2 + 2 * i], ay + coords[2 + 2 * i + 1]) for i in range(self.n_movers)]
            self._slotc[coords] = s
        return s

    def _svec(self, bits):
        v = self._sv.get(bits)
        if v is None:
            v = torch.randn(self._col.d_mem, generator=self._gen); v = v / v.norm(); self._sv[bits] = v
        return v

    def set_focus(self, focus, stage_fn):
        """The task column narrows the VALUE: bind agent × the ONE object being manipulated (`focus`) — so V
        FACTORS instead of binding the JOINT of all movers (the 2^K problem `control_loop` solved by factoring) —
        while keeping the small discrete STAGE (which factors are done, `stage_fn(coords)`) in the tag, so the
        value still knows where it is in the sequence. G still sees all movers; only the value narrows."""
        self._focus, self._stage_fn = tuple(focus), stage_fn

    def _latent(self, coords):
        zk = (coords, self._focus)
        z = self._zc.get(zk)
        if z is not None:
            return z
        col, cid = self._col, self._cid
        ax, ay = coords[0], coords[1]
        zz = col.place_code(cid[(ax, ay)]) if (ax, ay) in cid else torch.ones(col.d_mem)
        for i in self._focus:                                  # only the FOCUSED mover enters the value latent
            cell = (ax + coords[2 + 2 * i], ay + coords[2 + 2 * i + 1])
            if cell in cid:
                zz = zz * col.place_code(cid[cell])
        zz = zz * self._svec((coords[2 + 2 * self.n_movers:], self._stage_fn(coords)))   # tag: door bits + stage
        zz = zz / (zz.norm() + 1e-8)
        self._zc[zk] = zz
        return zz

    # ---- the learned model G, gated by traversability (the map + a closed gate) ---------------------------
    def _traversable(self, cell, coords, gates):
        if cell not in self._cid:
            return False
        b = gates.get(cell)
        return not (b is not None and coords[b] == 1)

    def _forward(self, coords, action, gates):
        nxt = self.G.predict(coords, action)
        for cell in self._slots(nxt):
            if not self._traversable(cell, coords, gates):     # agent or a pushed mover lands non-walkable
                return coords
        return nxt

    def _reward(self, coords, s2, action, pragmatic):
        """Learned env reward + potential-based pragmatic shaping (γ·Φ(s') − Φ(s)) toward F's goal."""
        return (self.rhat.get((coords, action), 0.0)
                + self.kappa * (self.gamma * pragmatic(s2) - pragmatic(coords)))

    # ---- the depth-D model-rollout search (memoised over the reachable set) -------------------------------
    def _rollout(self, coords, depth, gates, pragmatic, memo):
        if depth <= 0:
            return self.val.value(self._latent(coords))
        mk = (coords, depth)
        if mk in memo:
            return memo[mk]
        best = -1e30
        for a in range(self.n_actions):
            s2 = self._forward(coords, a, gates)
            d = self.term.get((coords, a), False)
            q = (self._reward(coords, s2, a, pragmatic)
                 + (0.0 if d else self.gamma * self._rollout(s2, depth - 1, gates, pragmatic, memo)))
            if q > best:
                best = q
        memo[mk] = best
        return best

    def act(self, coords, gates, pragmatic, explore=0.0, explore_fn=None):
        if coords is None or self._col is None:
            self._sel = None
            return self.rng.randrange(self.n_actions)
        if explore and self.rng.random() < explore:                  # biased exploration (skips the rollout)
            move = explore_fn(coords) if explore_fn else self.rng.randrange(self.n_actions)
        else:
            memo, qs = {}, []
            for a in range(self.n_actions):
                s2 = self._forward(coords, a, gates)
                d = self.term.get((coords, a), False)
                qs.append(self._reward(coords, s2, a, pragmatic)
                          + (0.0 if d else self.gamma * self._rollout(s2, self.depth - 1, gates, pragmatic, memo)))
            mx = max(qs)
            move = self.rng.choice([m for m, q in enumerate(qs) if q == mx])
        self._sel = (coords, move)
        return move

    # ---- learning: EZ-V2 multi-step TD with potential-based pragmatic shaping -----------------------------
    def learn(self, reward, done, pragmatic):
        if self._sel is None:
            return
        coords, move = self._sel
        self._traj.append((self._latent(coords), reward, pragmatic(coords), done))   # z, env_r, Φ, done
        self.rhat[(coords, move)] = reward
        self.term[(coords, move)] = done
        self._sel = None

    def flush(self):
        traj, T = self._traj, len(self._traj)
        shaped = []
        for t in range(T):
            phi = traj[t][2]
            phi_next = traj[t + 1][2] if (t + 1 < T and not traj[t][3]) else 0.0     # Φ(terminal) = 0
            shaped.append(traj[t][1] + self.kappa * (self.gamma * phi_next - phi))
        for t in range(T):
            R, disc, n, terminal = 0.0, 1.0, 0, False
            for k in range(self.l):
                if t + k >= T:
                    break
                R += disc * shaped[t + k]; disc *= self.gamma; n = k + 1
                if traj[t + k][3]:
                    terminal = True; break
            if not terminal and t + n < T:
                R += disc * self.val.value(traj[t + n][0])
            self.val.w += self.alpha * (R - self.val.value(traj[t][0])) * traj[t][0]
        self._traj = []


class FactoredPlanner:
    """The task column over the value-search — `control_loop`'s additive loop, made to drive the ValuePlanner.

    A conjunctive goal (`cover₁ ∧ cover₂ ∧ … ∧ reach`) is too deep for one pragmatic gradient: the search covers
    one factor and stalls (no gradient to sequence the rest). So sequence them — REVEAL the factors one at a time
    (perception orders them, terminal `reach` last); the value-search satisfies the newly-revealed one while every
    already-revealed factor stays in the pragmatic set, so the done ones are PRESERVED (un-satisfying one drops
    the progress). When the revealed set is all satisfied, reveal the next. K independent satisfactions sequenced,
    never the joint 2^K.

    Domain-general: `factors` and `satisfied(coords, factor)->bool` are OPAQUE (from perception). For any game the
    loop sequences whatever sub-conditions F learned — pads to cover, or contradictions to resolve."""

    def __init__(self, vp: "ValuePlanner", satisfied, route_proximity, focus_mover):
        self.vp, self.satisfied = vp, satisfied
        self.route_proximity, self.focus_mover = route_proximity, focus_mover
        self._k = 1
        self._prag = lambda c: 0.0
        self._explore_fn = None

    def reset(self):
        self.vp.reset()
        self._k = 1

    def set_map(self, col, cid):
        self.vp.set_map(col, cid)

    def act(self, coords, gates, factors, explore=0.0):
        if coords is not None and factors:
            while self._k < len(factors) and all(self.satisfied(coords, f) for f in factors[:self._k]):
                self._k += 1                                   # the revealed set is met → reveal the next factor
            active = tuple(factors[:self._k])
            n = len(factors)                                   # the VALUE is the sparse discrete satisfaction over
            self._prag = lambda c, a=active, n=n: sum(self.satisfied(c, f) for f in a) / n  # all factors (fixed
            #                                                  # denominator so revealing the next never drops Φ)
            target = factors[self._k - 1]                      # the current (revealed, unsatisfied) factor
            occ = frozenset(f[0] for f in factors[:self._k - 1] if f[1] == "cover")   # movers parked on done cells
            fm = self.focus_mover(coords, target, occ)         # FACTOR the value: agent × the ONE current object,
            stage = lambda c, fac=tuple(factors): frozenset(   # + the discrete STAGE (which factors are done) so
                j for j, f in enumerate(fac) if self.satisfied(c, f))   # the value still knows the sequence point
            self.vp.set_focus((fm,) if fm is not None else (), stage)
            def explore_fn(c, t=target, occ=occ, g=gates):     # EXPLORATION (not the value) heads toward the
                best, bp = self.vp.rng.randrange(self.vp.n_actions), -1.0             # current factor's object, so
                for a in range(self.vp.n_actions):                                   # the agent reaches the next
                    p = self.route_proximity(self.vp._forward(c, a, g), t, occ)      # block/goal; the value-search
                    if p > bp:                                                       # covers it from there (L0)
                        bp, best = p, a
                return best
            self._explore_fn = explore_fn
        return self.vp.act(coords, gates, self._prag, explore, self._explore_fn)

    def learn(self, reward, done):
        self.vp.learn(reward, done, self._prag)

    def flush(self):
        self.vp.flush()
