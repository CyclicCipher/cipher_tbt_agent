"""A TD-learned value over a column's place codes + a model-rollout search — the EZ-V2 `V` + planning, general.

The successor-representation IS the value basis (V = M·R) and the column's place codes ARE the SR eigenframe, so
a LINEAR readout `V(z) = w · z` can represent the value, with `w` learned by TD from scalar rewards. The latent
`z` the value reads is a BIND of the column's place code (the "where") with an observable STATE code (the "what":
which learned effects have fired — a door opened, a pad covered). Binding is what de-aliases a cell whose value
depends on world state that the place code alone can't see (a door is in the frame, but the door cell stays
walkable, so the place code is identical open or shut). `z` is unit-normalised: the Hadamard bind has norm ~1/√d,
which would otherwise shrink `V(z)` to ~0 and stall TD.

Acting is a depth-D model rollout (EZ-V2's search, exhaustive over the replica's 4 moves): from the current
latent, roll the learned model forward — the column's walkable map advances the "where", the learned effects
advance the "what" — accumulating the learned reward/termination head `r̂` and bootstrapping `V` at the leaves;
take the best root move. This is what a 1-step greedy cannot do: 'fire the key first' EMERGES because the search
sees the door open and the goal beyond it — no subgoal types, no 2^K enumeration, no hand-coded composite. (For
the real ARC click(x,y), `4^depth` is infeasible; that is where the Gumbel-Top-k sampling + Sequential Halving in
the notes replace the exhaustive expansion — parked until the click action.)

The colour-dependent pieces (`_state`, `_predict`) are flagged for a later acceptance-test cleanup: they belong
in perception (which legitimately knows colours), passing the planner an opaque state token + transition, so
`tbt/` ends up vector-only. The mechanism here — bind, normalise, TD `V`, model rollout — is already domain-free.
"""

from __future__ import annotations

import torch

from .planner import Planner


class Value:
    """The TD(0) readout over a latent: V(z) = w·z, w learned from scalar rewards. Domain-free — it sees only
    latent VECTORS and scalar rewards, never a grid / colour / action / effect."""

    def __init__(self, d_mem: int, gamma: float = 0.9, alpha: float = 0.3):
        self.w = torch.zeros(d_mem)
        self.gamma = gamma
        self.alpha = alpha

    def value(self, z: torch.Tensor) -> float:
        return float(self.w @ z)

    def update(self, z: torch.Tensor, reward: float, z_next: torch.Tensor, done: bool) -> float:
        """One TD(0) step: w += α·(r + γ·V(s') − V(s))·z. Returns the TD error (a surprise signal). On a terminal
        the bootstrap is zeroed — the never-visited leaf's garbage value must not repel the agent from the goal."""
        target = reward + (0.0 if done else self.gamma * float(self.w @ z_next))
        td = target - float(self.w @ z)
        self.w += self.alpha * td * z
        return td


class ValuePlanner(Planner):
    """S3 — the same column-map + recurrence as `Planner`, but it PLANS by a model-rollout search over a
    TD-learned `V` (bound latent) instead of the fire/cover/goal subgoal enumeration. Subgoal TYPES do not exist
    here; composite order emerges from the value + search."""

    def __init__(self, world, deltas, seed=0, gamma=0.9, alpha=0.3, depth=5):
        self.gamma, self.alpha, self.depth = gamma, alpha, depth
        self.val = None                                   # lazily sized to the column's d_mem on first act
        self.rhat, self.term = {}, {}                     # (state, cell, move) -> reward, terminal — the G head
        self._sv, self._zc = {}, {}                       # state -> state code;  (cell, state) -> cached latent
        self._gen = torch.Generator().manual_seed(1234 + seed)
        self._sel = None                                  # the move just selected, awaiting its `learn`
        super().__init__(world, deltas, seed)             # calls reset -> new_level (uses the attrs above)

    def new_level(self):
        super().new_level()
        self._zc = {}                                     # the map (and so the place codes) changed
        self._sel = None

    # ---- the bound, normalised latent z = place ⊗ state -----------------------------------------------------
    def _svec(self, d, state):
        v = self._sv.get(state)
        if v is None:
            v = torch.randn(d, generator=self._gen)
            v = v / v.norm()
            self._sv[state] = v                           # a fixed random tag per observed abstract state
        return v

    def _z(self, col, cid, cell, state):
        z = self._zc.get((cell, state))
        if z is None:
            z = col.place_code(cid[cell]) * self._svec(col.d_mem, state)     # VSA bind (Hadamard)
            z = z / (z.norm() + 1e-8)                                        # unit norm -> V(z) at O(1) scale
            self._zc[(cell, state)] = z
        return z

    # ---- the learned forward model G (colour-dependent -> perception's job in the cleanup) -----------------
    def _state(self, by_color):
        """The observable abstract state: which learned-effect ('door') colours are currently removed."""
        return frozenset(c for c in self.world.doors if c not in by_color)

    def _predict(self, cid, cell2color, cell, state, move):
        """G: (cell, state, move) -> (next_cell, next_state), from the learned map + effects. Static cell colours
        + the dynamic `state` (which effects fired) — so an imagined rollout opens doors / fires keys correctly."""
        dx, dy = self.deltas[move]
        tgt = (cell[0] + dx, cell[1] + dy)
        if tgt not in cid:
            return cell, state                            # a wall / off-map -> stay
        tcolor = cell2color.get(tgt)
        if tcolor in self.world.doors and tcolor not in state:
            return cell, state                            # a still-shut door blocks
        ns = frozenset(state | self.world.effects[tcolor]) if tcolor in self.world.effects else state
        return tgt, ns                                    # step (maybe firing an effect)

    def _rollout(self, col, cid, c2c, cell, state, depth):
        """Best discounted return over `depth` model steps, bootstrapping V at the leaf (EZ-V2 SVE, exhaustive
        over the 4 moves). r̂ / term come from the learned head; the leaf value from the SR readout."""
        if depth <= 0:
            return self.val.value(self._z(col, cid, cell, state))
        best = -1e30
        for m in range(len(self.deltas)):
            ncell, nstate = self._predict(cid, c2c, cell, state, m)
            r = self.rhat.get((state, cell, m), 0.0)
            d = self.term.get((state, cell, m), False)
            q = r + (0.0 if d else self.gamma * self._rollout(col, cid, c2c, ncell, nstate, depth - 1))
            if q > best:
                best = q
        return best

    # ---- act / learn --------------------------------------------------------------------------------------
    def act(self, scene, explore=0.0):
        by_color = scene.by_color
        self._spatial(by_color)
        body = self._track(scene.body_pos, by_color)
        col, cid = self._cache["col"], self._cache["cid"]
        if self.val is None:
            self.val = Value(col.d_mem, self.gamma, self.alpha)
        if body is None or body not in cid:
            self._sel = None
            return self.rng.randrange(len(self.deltas))
        c2c = {cell: c for c, cells in by_color.items() for cell in cells}   # cell -> colour, O(1) in the rollout
        state = self._state(by_color)
        qs = []
        for m in range(len(self.deltas)):
            ncell, nstate = self._predict(cid, c2c, body, state, m)
            r = self.rhat.get((state, body, m), 0.0)
            d = self.term.get((state, body, m), False)
            qs.append(r + (0.0 if d else self.gamma * self._rollout(col, cid, c2c, ncell, nstate, self.depth - 1)))
        if explore and self.rng.random() < explore:
            move = self.rng.randrange(len(self.deltas))
        else:
            mx = max(qs)
            move = self.rng.choice([m for m, q in enumerate(qs) if q == mx])  # randomised ties
        self._sel = (col, cid, body, state, move)
        return move

    def learn(self, next_scene, reward, done):
        """TD-update V for the move just taken + fill the reward/termination head. The next latent is read from
        the next state's own map (which may have rebuilt) and its own abstract state; `done` zeroes it."""
        if self._sel is None:
            return
        col, cid, body, state, move = self._sel
        z = self._z(col, cid, body, state)
        z_next = z
        if not done:
            nb = next_scene.body_pos
            nstate = self._state(next_scene.by_color)
            nc = self._spatial(next_scene.by_color)
            if nb is not None and nb in nc["cid"]:
                z_next = self._z(nc["col"], nc["cid"], nb, nstate)
        self.val.update(z, reward, z_next, done)
        self.rhat[(state, body, move)] = reward
        self.term[(state, body, move)] = done
        self._sel = None
