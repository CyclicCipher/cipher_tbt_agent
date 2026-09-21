"""ZipLearner on weights — the core loop of DESIGN.md §4 for ONE matrix over one-hot digits.

A layer over one-hot digits is a V x V matrix W: row x is what the layer says about input x. Learning a task is
choosing the cheapest DESCRIPTION of that matrix from a small library of structures (§5) and WRITING the matrix from it:

    fit    -- a structure solves its few parameters from the (x -> y) pairs seen so far;
    write  -- it then writes its WHOLE matrix, rows for inputs never seen included;
    price  -- every pair is paid for in bits as it arrives: log2(V) when the structure had no prediction yet (the pair
              is a parameter it must store), ~0 when its written matrix already predicted y, log2(V) + 1 when it was
              wrong (an escape, then the digit); plus log2(library size) to name the structure;
    keep   -- the structure with the lowest total is the layer's description; a query reads a row of its matrix.

Parameters are paid for THROUGH the observations that determine them (the shift's one pair IS its one parameter), so
there is no separate parameter term to double count. No gradients anywhere; everything is counting, solving, writing.

Library, version 1 (§5): table, identity, shift, affine permutation, general permutation. The cost when a structure has
no prediction is log2(number of outputs it could still be) -- log2(V) for most, but log2(V - seen) for the general
permutation, whose entries must be distinct, so its full table costs log2(V!) as §5 says.
"""
from __future__ import annotations

import math

import numpy as np


def inv(a, p):
    return pow(int(a) % p, p - 2, p)                                  # p prime


PRICE = "rate"                                                        # E4: "flat" was not a valid code; "rate" is (DESIGN §13)


def flag_bits(is_exception, n_right, n_wrong):
    """The RATE price of saying whether this observation is an exception (DESIGN §8, change 1): an adaptive code for a
    two-symbol source, P(exception) = (n_wrong + 1/2) / (n + 1) -- the Krichevsky-Trofimov estimator. Summed over n
    observations it converges to n * H(epsilon) + O(log n): exceptions are paid for by their RATE, not remembered."""
    n = n_right + n_wrong
    p_exc = (n_wrong + 0.5) / (n + 1.0)
    return -math.log2(p_exc if is_exception else 1.0 - p_exc)


def pay(s, right):
    """Charge structure `s` (any object with V, cost, n_right, n_wrong, price) for one predicted digit."""
    if right:
        s.cost += flag_bits(False, s.n_right, s.n_wrong) if s.price == "rate" else math.log2(1 + 1.0 / (1 + 4 * s.n_right))
        s.n_right += 1
    else:
        s.cost += (flag_bits(True, s.n_right, s.n_wrong) + math.log2(s.V - 1)) if s.price == "rate" else math.log2(s.V) + 1.0
        s.n_wrong += 1


class Structure:
    """One way of describing a V x V matrix. Subclasses define how to predict, how to absorb a pair, and how to write."""
    name = "structure"
    n_params = 0                                                      # for tie-breaks: simpler wins at equal price

    def __init__(self, V, price=None):
        self.V = V
        self.price = price or PRICE
        self.cost = 0.0                                               # bits paid so far (parameters + exceptions)
        self.n_right = self.n_wrong = self.n_new = 0

    @property
    def rate(self):                                                   # the exception rate this description reports
        n = self.n_right + self.n_wrong
        return self.n_wrong / n if n else 0.0

    def pay_right(self):
        if self.price == "rate":
            self.cost += flag_bits(False, self.n_right, self.n_wrong)
        else:
            self.cost += math.log2(1 + 1.0 / (1 + 4 * self.n_right))   # v1: nearly free once the model has stood up
        self.n_right += 1

    def pay_wrong(self):
        if self.price == "rate":
            self.cost += flag_bits(True, self.n_right, self.n_wrong) + math.log2(self.V - 1)
        else:
            self.cost += math.log2(self.V) + 1.0                       # v1: a flat escape, then the digit
        self.n_wrong += 1

    # -- what subclasses fill in -----------------------------------------------------------------------------------
    def predict(self, x):                                             # digit, or None if this structure cannot say yet
        raise NotImplementedError

    def absorb(self, x, y):                                           # update parameters / entries with this pair
        raise NotImplementedError

    def write(self):                                                  # the written V x V matrix (rows unknown = zeros)
        W = np.zeros((self.V, self.V))
        for x in range(self.V):
            y = self.predict(x)
            if y is not None:
                W[x, y] = 1.0
        return W

    def bits_new(self, x):                                            # price of a pair the structure had no guess for
        return math.log2(self.V)

    # -- the core loop for one pair --------------------------------------------------------------------------------
    def observe(self, x, y):
        g = self.predict(x)
        if g is None:
            self.cost += self.bits_new(x)
            self.n_new += 1
        elif g == y:
            self.pay_right()
        else:
            self.pay_wrong()
        self.absorb(x, y)


class Table(Structure):
    """The dictionary: one entry per input seen, majority output. Captures anything; cannot extrapolate."""
    name, n_params = "table", 10 ** 6

    def __init__(self, V, price=None):
        super().__init__(V, price)
        self.counts = {}

    def predict(self, x):
        c = self.counts.get(x)
        return max(c, key=c.get) if c else None

    def absorb(self, x, y):
        self.counts.setdefault(x, {}).setdefault(y, 0)
        self.counts[x][y] += 1


class Identity(Structure):
    """This layer copies its input. Zero parameters."""
    name, n_params = "identity", 0

    def predict(self, x):
        return x

    def absorb(self, x, y):
        pass


class Shift(Structure):
    """y = x + b. One parameter, solved from the first pair."""
    name, n_params = "shift", 1

    def __init__(self, V, price=None):
        super().__init__(V, price)
        self.b = None

    def predict(self, x):
        return None if self.b is None else (x + self.b) % self.V

    def absorb(self, x, y):
        if self.b is None:
            self.b = (y - x) % self.V


class Affine(Structure):
    """y = a*x + b. Two parameters, solved from the first two pairs with different x; a seen x is answered meanwhile."""
    name, n_params = "affine", 2

    def __init__(self, V, price=None):
        super().__init__(V, price)
        self.points = {}
        self.a = self.b = None

    def predict(self, x):
        if self.a is not None:
            return (self.a * x + self.b) % self.V
        return self.points.get(x)

    def bits_new(self, x):
        # the first pair fixes b (any of V digits); the second fixes a, which cannot be 0 -- the map is one-to-one, so
        # y1 != y0 and there are V - 1 options. The family has V*(V-1) members, and the price says so.
        return math.log2(self.V if not self.points else self.V - 1)

    def observe(self, x, y):
        if self.a is None and self.points and x not in self.points and y in self.points.values():
            self.pay_wrong()                                          # y1 == y0 would need a = 0: not one-to-one; escape
            return
        super().observe(x, y)

    def absorb(self, x, y):
        if self.a is None:
            self.points.setdefault(x, y)
            if len(self.points) == 2:
                (x0, y0), (x1, y1) = self.points.items()
                self.a = (y1 - y0) * inv(x1 - x0, self.V) % self.V
                self.b = (y0 - self.a * x0) % self.V


class Permutation(Structure):
    """Any one-to-one relabelling. Each new input costs log2(outputs still unused), so a full table costs log2(V!);
    the last unseen input is inferred for free."""
    name, n_params = "permutation", 10 ** 3

    def __init__(self, V, price=None):
        super().__init__(V, price)
        self.entries, self.used = {}, set()

    def predict(self, x):
        if x in self.entries:
            return self.entries[x]
        if len(self.entries) == self.V - 1:
            return next(y for y in range(self.V) if y not in self.used)
        return None

    def bits_new(self, x):
        return math.log2(max(1, self.V - len(self.entries)))

    def observe(self, x, y):
        if x not in self.entries and y in self.used:                  # a collision refutes one-to-one: an escape,
            self.pay_wrong()                                          # and the pair is not stored
            return
        super().observe(x, y)

    def absorb(self, x, y):
        if x not in self.entries:
            self.entries[x] = y
            self.used.add(y)


LIBRARY = {s.name: s for s in (Table, Identity, Shift, Affine, Permutation)}


class Layer:
    """One matrix, described by whichever structure prices the pairs seen so far cheapest."""

    def __init__(self, V=11, library=("table", "identity", "shift", "affine", "permutation"), price=None):
        self.V = V
        self.structs = [LIBRARY[n](V, price) for n in library]
        self.bits_name = math.log2(len(self.structs))                 # naming the structure, paid once
        self.seen = set()

    def observe(self, x, y):
        x, y = int(x), int(y)
        for s in self.structs:
            s.observe(x, y)
        self.seen.add(x)

    def observe_demo(self, xs, ys):
        for x, y in zip(xs, ys):
            self.observe(x, y)

    def price(self, s):
        return s.cost + self.bits_name

    def keep(self):
        return min(self.structs, key=lambda s: (round(self.price(s), 9), s.n_params))

    def matrix(self):
        return self.keep().write()

    def predict(self, x):
        row = self.matrix()[int(x)]
        return None if row.sum() == 0 else int(row.argmax())

    def predict_seq(self, xs):
        return [self.predict(x) for x in xs]

    def report(self):
        return {s.name: round(self.price(s), 2) for s in self.structs}


# ── sequences: a position layer, and two layers solved together (DESIGN §6) ─────────────────────────────────────────
class PositionPerm:
    """Output position j reads input position pi(j): a permutation of the L positions, solved by intersecting
    candidates -- after each demonstration, pi(j) can only be a position whose digit equalled y_j -- with the
    one-to-one constraint propagated. Digits not yet predictable cost log2(V) (they are the parameters); a prediction
    that fails is an escape and refutes the structure for good."""
    name, n_params = "position permutation", 10

    def __init__(self, L, V, price=None):
        self.L, self.V, self.price = L, V, price or PRICE
        self.cand = [set(range(L)) for _ in range(L)]
        self.cost, self.dead = 0.0, False
        self.n_right = self.n_wrong = 0

    def pi(self, j):
        c = self.cand[j]
        return next(iter(c)) if len(c) == 1 else None

    def predict(self, x_seq):
        return [None if self.pi(j) is None else int(x_seq[self.pi(j)]) for j in range(self.L)]

    def observe_demo(self, x_seq, y_seq):
        x_seq, y_seq = [int(v) for v in x_seq], [int(v) for v in y_seq]
        for j in range(self.L):
            p = self.pi(j)
            if self.dead or (p is not None and x_seq[p] != y_seq[j]):
                pay(self, False)
                self.dead = True
                continue
            if p is None:
                self.cost += math.log2(self.V)
                self.cand[j] &= {i for i in range(self.L) if x_seq[i] == y_seq[j]}
                if not self.cand[j]:
                    self.dead = True
            else:
                pay(self, True)
        # one-to-one: a settled source is unavailable to the others (repeat until nothing changes)
        changed = True
        while changed and not self.dead:
            changed = False
            settled = {self.pi(j) for j in range(self.L) if self.pi(j) is not None}
            for j in range(self.L):
                if self.pi(j) is None and self.cand[j] & settled:
                    self.cand[j] -= settled
                    changed = True
                    if not self.cand[j]:
                        self.dead = True


class PositionIdentity:
    name, n_params = "position identity", 0

    def __init__(self, L, V, price=None):
        self.L, self.V, self.price, self.cost, self.n_right, self.n_wrong = L, V, price or PRICE, 0.0, 0, 0

    def predict(self, x_seq):
        return [int(v) for v in x_seq]

    def observe_demo(self, x_seq, y_seq):
        for x, y in zip(x_seq, y_seq):
            pay(self, int(x) == int(y))


def value_hypotheses(V):
    """Every description the value layer can have at this V, with the bits its parameters cost: identity (0),
    shift b (log2 V), affine a,b with a != 0,1 (log2 V + log2(V-1) in total, as in §5)."""
    hyps = [("identity", (1, 0), 0.0)]
    hyps += [("shift", (1, b), math.log2(V)) for b in range(1, V)]
    hyps += [("affine", (a, b), math.log2(V) + math.log2(V - 1)) for a in range(2, V) for b in range(V)]
    return hyps


class TwoLayer:
    """Layer 1 = positions, layer 2 = values, described together at the lowest total price. The plain sweep of §6
    (freeze one layer, solve the other, alternate) is started from EVERY description of the value layer -- there are
    only 1 + (V-1) + (V-2)*V of them -- because from the identity start alone it deadlocks whenever both layers act
    (`sweep_from_identity` measures that). Inverting the value layer is exact: every hypothesis is a bijection."""

    def __init__(self, L, V):
        self.L, self.V = L, V
        self.hyps = value_hypotheses(V)
        self.bits_value_name = math.log2(3)                       # identity / shift / affine
        self.bits_pos_name = math.log2(2)                         # identity / permutation
        self.runs = [(h, PositionIdentity(L, V), PositionPerm(L, V)) for h in self.hyps]

    @staticmethod
    def invert(a, b, y, V):
        return (inv(a, V) * (y - b)) % V

    def observe_demo(self, x_seq, y_seq):
        for (name, (a, b), _bits), pid, pperm in self.runs:
            t = [self.invert(a, b, int(y), self.V) for y in y_seq]   # what layer 1 must produce for layer 2 to give y
            pid.observe_demo(x_seq, t)
            pperm.observe_demo(x_seq, t)

    def best(self):
        best, best_price = None, (float("inf"), 0)
        for (name, ab, bits), pid, pperm in self.runs:
            for pos in (pid, pperm):
                price = bits + self.bits_value_name + pos.cost + self.bits_pos_name
                key = (round(price, 9), pos.n_params)
                if best is None or key < best_price:
                    best, best_price = (name, ab, pos), key
        return best

    def predict_seq(self, x_seq):
        name, (a, b), pos = self.best()
        mid = pos.predict(x_seq)
        return [None if m is None else (a * m + b) % self.V for m in mid]

    def describe(self):
        name, (a, b), pos = self.best()
        return f"{pos.name} then {name}{'' if name == 'identity' else f' a={a} b={b}'}"


# ── library discovery across tasks (DESIGN §7 step 3) ───────────────────────────────────────────────────────────────
class NamedPerm:
    """A position permutation the library already knows by name. It predicts from the first digit and pays only the
    bits to NAME it (log2 of the number of named items) instead of solving its entries."""
    name, n_params = "named permutation", 1

    def __init__(self, L, V, pi, bits_name, price=None):
        self.L, self.V, self._pi, self.price = L, V, tuple(pi), price or PRICE
        self.cost, self.n_right, self.n_wrong, self.dead = bits_name, 0, 0, False

    def pi(self, j):
        return self._pi[j]

    def predict(self, x_seq):
        return [int(x_seq[self._pi[j]]) for j in range(self.L)]

    def observe_demo(self, x_seq, y_seq):
        for j in range(self.L):
            right = int(x_seq[self._pi[j]]) == int(y_seq[j]) and not self.dead
            pay(self, right)
            if not right:
                self.dead = True                                    # one miss refutes a named item for this task


class PermLibrary:
    """Position permutations seen to recur across tasks. A solved permutation is a candidate; the second time the same
    one is solved it is promoted to a named item (ZIP's dictionary rule: a recurring substring becomes an entry)."""

    def __init__(self):
        self.candidates, self.named = {}, []

    def bits_name(self):
        return math.log2(len(self.named)) if self.named else 0.0

    def record(self, pi):
        pi = tuple(pi)
        if pi in self.named:
            return False
        self.candidates[pi] = self.candidates.get(pi, 0) + 1
        if self.candidates[pi] >= 2:
            self.named.append(pi)
            return True
        return False


class TwoLayerWithLibrary(TwoLayer):
    """TwoLayer whose position layer also tries every named permutation in the library."""

    def __init__(self, L, V, library):
        super().__init__(L, V)
        self.library = library
        self.bits_pos_name = math.log2(2 + (1 if library.named else 0))    # identity / permutation / a named item
        self.runs = [(h, PositionIdentity(L, V), PositionPerm(L, V),
                      [NamedPerm(L, V, pi, library.bits_name()) for pi in library.named]) for h in self.hyps]

    def observe_demo(self, x_seq, y_seq):
        for (name, (a, b), _bits), pid, pperm, named in self.runs:
            t = [self.invert(a, b, int(y), self.V) for y in y_seq]
            pid.observe_demo(x_seq, t)
            pperm.observe_demo(x_seq, t)
            for s in named:
                s.observe_demo(x_seq, t)

    def best(self):
        best, best_price = None, (float("inf"), 0)
        for (name, ab, bits), pid, pperm, named in self.runs:
            for pos in (pid, pperm, *named):
                price = bits + self.bits_value_name + pos.cost + self.bits_pos_name
                key = (round(price, 9), pos.n_params)
                if best is None or key < best_price:
                    best, best_price = (name, ab, pos), key
        return best

    def solved_pi(self):
        """The position permutation of the kept description, if fully solved (general) or named."""
        _n, _ab, pos = self.best()
        if isinstance(pos, PositionPerm) and not pos.dead and all(pos.pi(j) is not None for j in range(self.L)):
            return tuple(pos.pi(j) for j in range(self.L))
        if isinstance(pos, NamedPerm):
            return pos._pi
        return None


# ── continual learning without a replay buffer (DESIGN §9) ──────────────────────────────────────────────────────────
class ContinualLayer:
    """One matrix with a BLOCK of rows per context (§9). Pairs arrive with no task labels. Each block is a `Layer` of
    its own -- its description is never overwritten by another context's data (rule 1); the active block is the one
    that prices the recent window cheapest (rule 2, context selects); a new block is minted only when a fresh
    description of the recent window would be cheaper than any existing block's account of it (mint on refutation).
    What is kept between contexts is the descriptions and their counts -- no examples, no replay."""

    def __init__(self, V=11, window=4, library=("table", "identity", "shift", "affine", "permutation")):
        self.V, self.window, self.library = V, window, library
        self.blocks = []                                                 # each: dict(layer=Layer, n=evidence count)
        self.recent = []                                                 # the last `window` pairs: the context
        self.n_minted = 0

    # -- pricing a window under a block's CURRENT description, without learning from it --------------------------
    MIN_EVIDENCE = 3                                                     # pairs before a block's description is fixed

    @classmethod
    def _window_cost(cls, layer, pairs, n=None):
        """What a block would pay for these pairs. An ESTABLISHED block (evidence >= MIN_EVIDENCE) is priced under its
        kept structure, frozen as it stands: it must not quietly re-describe itself (become a table, say) to swallow
        another rule's pairs -- routing asks what each description SAYS, and foreign pairs are exceptions to it. A
        YOUNG block is still forming and is priced with all its structures live, so its own rule's next pairs can
        still correct an early guess. Rule 1 of §9, literally: protection in proportion to evidence."""
        import copy
        if n is None or n >= cls.MIN_EVIDENCE:
            s = copy.deepcopy(layer.keep())
            before = s.cost
            for x, y in pairs:
                s.observe(x, y)
            return s.cost - before
        probe = copy.deepcopy(layer)
        before = probe.price(probe.keep())
        for x, y in pairs:
            probe.observe(x, y)
        return probe.price(probe.keep()) - before

    def bits_block_name(self):
        return math.log2(len(self.blocks)) if len(self.blocks) > 1 else 0.0

    def select(self, pairs):
        """The block that explains these pairs cheapest, and its cost; (None, inf) if there are no blocks."""
        best, best_cost = None, float("inf")
        for b in self.blocks:
            c = self._window_cost(b["layer"], pairs, b["n"])
            if c < best_cost:
                best, best_cost = b, c
        return best, best_cost

    def observe(self, x, y):
        """Route the pair to a block, minting one if the recent window is cheaper described WITH A CHANGE in it. For
        every split point k the window is priced as "an existing block explains window[:k], then either another
        existing block or a fresh one explains window[k:]" (a change costs log2 V bits; a fresh block also costs its
        name), against "one existing block explains it all". The cheapest wins; a fresh block starts from the part
        after the change -- never from a window that straddles two rules -- and a rule that returns is routed to its
        old block, not given a new one."""
        x, y = int(x), int(y)
        self.recent = (self.recent + [(x, y)])[-self.window:]
        W = len(self.recent)
        change = math.log2(self.V)
        whole_block, whole = self.select(self.recent)
        options = [(whole, whole_block, None)]                            # (cost, block to route to, mint from k)
        for k in range(W):
            _pb, prefix = self.select(self.recent[:k]) if k else (None, 0.0)
            sb, suffix = self.select(self.recent[k:])
            fresh = self._window_cost(Layer(self.V, self.library), self.recent[k:], 0) + self.bits_block_name()
            if sb is not None and suffix <= fresh:
                options.append((prefix + change + suffix, sb, None))
            else:
                options.append((prefix + change + fresh, None, k))
        cost, block, mint_k = min(options, key=lambda o: o[0])
        if block is None:                                                # refuted: mint on the post-change part
            fresh = Layer(self.V, self.library)
            for px, py in self.recent[mint_k:-1]:
                fresh.observe(px, py)
            block = dict(layer=fresh, n=len(self.recent[mint_k:-1]))
            self.blocks.append(block)
            self.n_minted += 1
        block["layer"].observe(x, y)
        block["n"] += 1
        return block

    def predict_in_context(self, context_pairs, x):
        """Answer x for the context these pairs establish, learning nothing: rule 2's selector, then a row read."""
        block, _c = self.select(context_pairs)
        return None if block is None else block["layer"].predict(x)

    def total_bits(self):
        return sum(b["layer"].price(b["layer"].keep()) for b in self.blocks) + len(self.blocks) * self.bits_block_name()

    def describe(self):
        out = []
        for b in self.blocks:
            s = b["layer"].keep()
            params = {"shift": lambda s: f"b={s.b}", "affine": lambda s: f"a={s.a} b={s.b}"}.get(s.name, lambda s: "")(s)
            out.append(f"{s.name} {params} (n={b['n']}, exceptions {s.rate:.2f})")
        return out
