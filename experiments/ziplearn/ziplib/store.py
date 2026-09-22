"""ziplib.store — the ONE count table and its three tensor forms (DESIGN §21.2, §21.7; the module spec's `store.py`).

A `Store` is a table from MASKED WINDOWS (the cells of a receptive `Field` the mask keeps) to the counts of an outcome,
plus the EVIDENCE it was built from (`full`: counts over whole windows), so that a sleep pass can try smaller masks and
adopt one when the merged table is cheaper (E24). It is `arcgames.LocalRule` moved whole, with `GoalModel`'s outcome
keys ([confirmed, refuted] per window) and `ActionModel`'s radius competition (`best_store`) beside it, and
`textlm.ContextLM`'s window extraction, backoff chain and prequential sleep price folded in: one table for the games'
rules, the goal keys and the text contexts. Its three TENSOR FORMS are regenerated from the evidence on demand:

  as_tokens(codec)   E28's memory tokens, one per entry (`codec.entry`; a cell the mask dropped is an all-zero WILDCARD)
  as_bank(codec)     the same rows split into (keys, values) for the k/v cache — the GD-compatible form (DESIGN §21.8)
  as_rows(layout)    the MLP form: (key_row, threshold = k_in - 1/2, value_row) per entry (DESIGN §21.7), the key one-hot
                     under `code="onehot"` or a k-of-d code under `code="sparse"` (`SparseProjection`)

What moved from where (the check script `_check_store.py` compares each against its origin):
  Field.windows_grid    <- LocalRule._rows        BORDER padding + sliding_window_view, (H*W, cells) rows in raster order
  Field.windows_seq     <- ContextLM._contexts    the k-back context of every position, column j = offsets[j] back
  Store.observe / predict / predict_frame / sleep(price="table") / merged / price   <- LocalRule (whole; the online
                        cost with `price.flag_bits`, the greedy strict/loose sleep with the centre always kept)
  Store.sleep(price="prequential") / chain   <- ContextLM.fit's greedy drop under `price.prequential_bits` / _chain_of
  Store.record / refute / live / predicts / known   <- GoalModel.record / refute / live / predicts / known (per `tag`)
  Store.nearest / predict_nearest   <- e35.NearestRule._keys_array / predict_nearest (the Hamming-nearest stored key,
                        ties by stored index; here as the matmul of one-hot codes, which is what the `Match` head computes)
  best_store            <- ActionModel.best        the radius competition on price
  Store.exception_rate  <- ActionModel.exception_rate

Offline readers (`nearest`, `candidates`, `loo_kernel_width`) and `consolidate` are ZipLearn's apparatus (DESIGN §21.1
item 2): never called from `Brain.run`.

Conventions. Symbols are 0..V-1; the padding symbol is BORDER = -1 in the DATA (LocalRule's convention), which the
codec writes at the END of a one-hot subspace (index C - 1 = V of `colour(C = V + 1)`): the text store pads with the
same -1 where `ContextLM` used V — the prices depend only on key equality, so E33's numbers do not move. Keys are the
int16 bytes of the window's cells (LocalRule's `tobytes()`); `full` keys are whole windows, `table` keys masked ones.
Field cell -> subspace: the centre is `field.centre_name` ("colour"), an offset (di, dj) is `field.template`
("nbr[{di},{dj}]", the blueprint README's convention); both are arguments, not constants of the table.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from . import price as P

BORDER = -1                                                  # the data's padding symbol (LocalRule.BORDER)
HS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)                   # expI's kernel-width grid (research/expI_frames_chowliu_loo.py)


# ── the receptive field ──────────────────────────────────────────────────────────────────────────────────────────────
class Field:
    """The cells of a window as offsets (di, dj) in order, the index of the centre cell (None for a context field,
    whose target is the next symbol, not a cell of the window), and the subspace each cell is written to.

    `Field.grid(r)` is LocalRule's window: the (2r+1)^2 cells in raster order, centre at n // 2 (always kept by sleep).
    `Field.seq(k)` is ContextLM's context: `back(k)` = [(0, -1), ..., (0, -k)], column j = j + 1 characters back.
    `Field.lattice(r)` (the born receptive field of a blueprint, centre excluded) and `Field.back(k)` are the offset lists."""

    def __init__(self, offsets, centre=None, kind="lattice", names=None, centre_name="colour", template="nbr[{di},{dj}]"):
        self.offsets = [tuple(int(v) for v in o) for o in offsets]
        if len(set(self.offsets)) != len(self.offsets):
            raise ValueError(f"Field: an offset is repeated in {self.offsets}")
        self.centre = None if centre is None else int(centre)
        if self.centre is not None and not (0 <= self.centre < len(self.offsets)):
            raise ValueError(f"Field: centre {centre} is outside {len(self.offsets)} cells")
        self.kind = kind
        self.n = len(self.offsets)
        self.r = max((max(abs(di), abs(dj)) for di, dj in self.offsets), default=0)
        self.centre_name, self.template = centre_name, template
        self.names = list(names) if names is not None else [
            centre_name if i == self.centre else template.format(di=di, dj=dj) for i, (di, dj) in enumerate(self.offsets)]
        if len(self.names) != self.n:
            raise ValueError(f"Field: {len(self.names)} names for {self.n} cells")

    @staticmethod
    def lattice(r):
        """Every (di, dj) with |di|, |dj| <= r, the centre excluded, in raster order (8 at r = 1, 24 at r = 2, 80 at r = 4)."""
        return [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1) if (di, dj) != (0, 0)]

    @staticmethod
    def back(k):
        """[(0, -1), ..., (0, -k)]: the k previous positions of a sequence, most recent first."""
        return [(0, -j) for j in range(1, k + 1)]

    @classmethod
    def grid(cls, r, **kw):
        window = [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)]
        return cls(window, centre=len(window) // 2, kind="lattice", **kw)

    @classmethod
    def seq(cls, k, **kw):
        return cls(cls.back(k), centre=None, kind="back", **kw)

    def name(self, offset):
        return self.names[self.offsets.index(tuple(offset))]

    def windows_grid(self, frame, mask=None):
        return windows_grid(frame, self, mask)

    def windows_seq(self, seq, mask=None):
        return windows_seq(seq, self, mask)

    def to_dict(self):
        return dict(offsets=self.offsets, centre=self.centre, kind=self.kind, names=self.names,
                    centre_name=self.centre_name, template=self.template)

    @classmethod
    def from_dict(cls, d):
        return cls(d["offsets"], d.get("centre"), d.get("kind", "lattice"), d.get("names"),
                   d.get("centre_name", "colour"), d.get("template", "nbr[{di},{dj}]"))

    def __repr__(self):
        return f"Field({self.kind}, {self.n} cells, r={self.r}, centre={self.centre})"


def windows_grid(frame, field, mask=None):
    """MOVES LocalRule._rows: every cell's window as a row, (H*W, cells) in raster order, the frame BORDER-padded by the
    field's extent; `(full, masked)` with `masked = full[:, mask]` (`masked is full` when no mask is given). The
    columns are the field's cells in its order (identity for `Field.grid(r)`, whose cells are the whole square)."""
    r = field.r
    frame = np.asarray(frame)
    if frame.ndim != 2:
        raise ValueError(f"windows_grid wants an H x W frame, got shape {frame.shape}")
    H, W = frame.shape
    pad = np.full((H + 2 * r, W + 2 * r), BORDER, dtype=np.int16)
    pad[r:r + H, r:r + W] = frame
    full = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(pad, (2 * r + 1, 2 * r + 1)).reshape(H * W, -1))
    side = 2 * r + 1
    cols = [(di + r) * side + (dj + r) for di, dj in field.offsets]
    if cols != list(range(side * side)):
        full = np.ascontiguousarray(full[:, cols])
    masked = full if mask is None else np.ascontiguousarray(full[:, mask])
    return full, masked


def windows_seq(seq, field, mask=None):
    """MOVES ContextLM._contexts: for every position of a sequence the context row (n, cells), column j = the symbol
    `field.offsets[j]` positions back ((0, -k): k back), BORDER-padded before the start; `(full, masked)`. The target
    of row i is `seq[i]` itself (ContextLM's `nxt`), which the caller passes to `observe` as `y`."""
    s = np.asarray(seq).reshape(-1).astype(np.int16)
    n = len(s)
    R = max((-dj for _di, dj in field.offsets), default=0)
    if any(di != 0 or dj > 0 for di, dj in field.offsets):
        raise ValueError(f"windows_seq wants a `back` field (offsets (0, -k)), got {field.offsets}")
    pad = np.concatenate([np.full(R, BORDER, dtype=np.int16), s])
    cols = [pad[R + dj: R + dj + n] for _di, dj in field.offsets]
    full = np.ascontiguousarray(np.stack(cols, 1)) if cols else np.zeros((n, 0), dtype=np.int16)
    masked = full if mask is None else np.ascontiguousarray(full[:, mask])
    return full, masked


# ── a fixed random sparse code ───────────────────────────────────────────────────────────────────────────────────────
class SparseProjection:
    """A FIXED random binary matrix (d_in x d_out) with k ones per row, seeded: a one-hot key maps to a k-of-d_out code
    (its row); a multi-hot key (a window of several cells) maps to the UNION of its dims' rows (Willshaw 1969's sparse
    address; DESIGN §21.6 axis 2, §21.8a item 1a). Genome-written, never learned: `SparseProjection(d, d, 11, seed)`
    is the same matrix on every machine."""

    def __init__(self, d_in, d_out, k, seed=0):
        if not 0 < k <= d_out:
            raise ValueError(f"SparseProjection: k = {k} must be in 1..d_out = {d_out}")
        self.d_in, self.d_out, self.k, self.seed = int(d_in), int(d_out), int(k), int(seed)
        rng = np.random.default_rng(seed)
        m = np.zeros((self.d_in, self.d_out), dtype=np.float32)
        for i in range(self.d_in):
            m[i, rng.choice(self.d_out, size=self.k, replace=False)] = 1.0
        self.matrix = torch.from_numpy(m)

    def __call__(self, x):
        """x: Tensor[..., d_in] of 0/1 -> Tensor[..., d_out] of 0/1 (the union of the active dims' codes)."""
        x = torch.as_tensor(x, dtype=torch.float32)
        return (x @ self.matrix > 0.5).float()

    def __repr__(self):
        return f"SparseProjection({self.d_in} -> {self.d_out}, k={self.k}, seed={self.seed})"


# ── the store ────────────────────────────────────────────────────────────────────────────────────────────────────────
class Store:
    """One count table over masked windows (module docstring). `Store(field, V)`; `observe(keys, y)` with `keys` the
    WHOLE windows (rows of `windows_grid` / `windows_seq`) — the mask projection is the store's own, as it was
    LocalRule's; `predict(keys)` by hash; `sleep(strict)`; the tensor forms; the offline readers.

    `keep_order=True` keeps the observations in order (`log`), which the prequential sleep price (E33) needs; the table
    price (E24) needs only the counts. `value` names the subspace the majority is written to in the tensor forms."""

    def __init__(self, field, V, keep_order=False, value="out", capacity=None):
        self.field, self.V = field, int(V)
        self.mask = np.ones(field.n, dtype=bool)
        self.centre = field.centre
        self.full = {}                                       # whole window -> {y: count}: the evidence
        self.table = {}                                      # masked window -> {y: count}: the description
        self.majority = {}                                   # masked window -> the y predicted (the entry's majority)
        self.stats = {}                                      # masked window -> [n_right, n_wrong]
        self.cost = 0.0                                      # the online two-part code paid so far (LocalRule.cost)
        self.n_obs = 0                                       # observe() calls (ActionModel.n_obs)
        self.outcome_keys = {}                               # tag -> {whole window: [confirmed, refuted]} (GoalModel)
        self.log = [] if keep_order else None                # [(rows, ys)] in observation order (the prequential price)
        self.forms = {}                                      # masked window -> "rows" once consolidated (else tokens)
        self.value, self.capacity = value, capacity
        self.bits_before = self.bits_after = None            # the last sleep's (before, after)
        self.proj = None                                     # the SparseProjection `as_rows(code="sparse")` used last
        self._karr = self._kval = None                       # the nearest reader's cache (NearestRule._keys_array)
        self._karr_n = -1

    # -- keys ------------------------------------------------------------------------------------------------------
    @property
    def r(self):
        return self.field.r

    @property
    def kept(self):
        """Indices of the cells the mask keeps."""
        return np.flatnonzero(self.mask)

    def _rows(self, keys):
        keys = np.asarray(keys)
        if keys.ndim == 1:
            keys = keys[None, :]
        if keys.ndim != 2 or keys.shape[1] != self.field.n:
            raise ValueError(f"Store: keys must be whole windows of {self.field.n} cells, got shape {keys.shape}")
        return np.ascontiguousarray(keys.astype(np.int16, copy=False))

    def _project(self, full_key):
        return np.frombuffer(full_key, dtype=np.int16)[self.mask].tobytes()

    def key_fields(self, key):
        """A masked key (int16 array over the kept cells, or its bytes) as {subspace: symbol} for the codec — a dropped
        cell is absent (a wildcard); BORDER = -1 stays -1 (the codec writes it at the end of the subspace)."""
        if isinstance(key, (bytes, bytearray)):
            key = np.frombuffer(key, dtype=np.int16)
        return {self.field.names[c]: int(v) for c, v in zip(self.kept, key)}

    # -- observing (MOVES LocalRule.observe) -----------------------------------------------------------------------
    def observe(self, keys, y):
        """Count one or many (whole window, outcome) pairs in order, paying the online two-part code: a new masked
        window is a parameter (log2 V), a repeat pays its exception flag at the rate price (+ log2(V - 1) when wrong)."""
        rows = self._rows(keys)
        ys = np.asarray(y).reshape(-1)
        if len(ys) != len(rows):
            raise ValueError(f"Store.observe: {len(rows)} windows but {len(ys)} outcomes")
        if self.log is not None:
            self.log.append((rows.copy(), ys.astype(np.int16)))
        V, flag_bits = self.V, P.flag_bits
        full, table, majority, stats, mask = self.full, self.table, self.majority, self.stats, self.mask
        cost = 0.0
        masked = rows[:, mask]
        for t in range(len(rows)):
            fk, k, yy = rows[t].tobytes(), masked[t].tobytes(), int(ys[t])
            f = full.setdefault(fk, {})
            f[yy] = f.get(yy, 0) + 1
            e = table.get(k)
            if e is None:                                    # a new window: its outcome is a parameter
                cost += math.log2(V)
                table[k] = {yy: 1}
                majority[k] = yy
                stats[k] = [0, 0]
                continue
            st = stats[k]
            if majority[k] == yy:
                cost += flag_bits(False, st[0], st[1])
                st[0] += 1
            else:
                cost += flag_bits(True, st[0], st[1]) + math.log2(V - 1)
                st[1] += 1
            e[yy] = e.get(yy, 0) + 1
            majority[k] = max(e, key=e.get)
        self.cost += cost
        self.n_obs += 1
        self._karr_n = -1
        return cost

    def observe_frame(self, before, after):
        """LocalRule.observe(before, after): every cell's whole window -> the cell's new symbol."""
        full, _m = windows_grid(before, self.field)
        return self.observe(full, np.asarray(after).reshape(-1))

    def observe_seq(self, seq):
        """ContextLM.fit's counting for one text: every position's context -> the symbol there."""
        full, _m = windows_seq(seq, self.field)
        return self.observe(full, np.asarray(seq).reshape(-1))

    # -- the hash reader (MOVES LocalRule.predict) -----------------------------------------------------------------
    def predict(self, keys):
        """The majority for each whole window whose masked key is stored, None where it is not: y | None for one
        window, a list for many."""
        rows = self._rows(keys)
        masked = rows[:, self.mask]
        out = [self.majority.get(masked[t].tobytes()) for t in range(len(rows))]
        return out[0] if np.asarray(keys).ndim == 1 else out

    def predict_frame(self, frame):
        """The predicted frame and the number of cells whose (masked) window was never seen (left unchanged)."""
        frame = np.asarray(frame)
        _full, masked = windows_grid(frame, self.field, self.mask)
        out = frame.copy().reshape(-1)
        unknown = 0
        majority = self.majority
        for t in range(out.size):
            y = majority.get(masked[t].tobytes())
            if y is None:
                unknown += 1
            else:
                out[t] = y
        return out.reshape(frame.shape), unknown

    def exception_rate(self):
        """MOVES ActionModel.exception_rate: wrong / (right + wrong) over the table's stats."""
        st = self.stats.values()
        right, wrong = sum(s[0] for s in st), sum(s[1] for s in st)
        return wrong / (right + wrong) if right + wrong else 0.0

    # -- the sleep pass (MOVES LocalRule.sleep; ContextLM.fit's drop loop) ------------------------------------------
    def merged(self, mask):
        """The evidence merged under a mask: {masked key: {y: count}} (LocalRule._merged)."""
        merged = {}
        for fk, counts in self.full.items():
            k = np.frombuffer(fk, dtype=np.int16)[mask].tobytes()
            m = merged.setdefault(k, {})
            for y, c in counts.items():
                m[y] = m.get(y, 0) + c
        return merged

    @staticmethod
    def _wrong(merged):
        return sum(sum(c.values()) - max(c.values()) for c in merged.values())

    def price(self, mask=None):
        """The two-part code of the evidence under the table a mask gives (`price.table_price`; the current mask by default)."""
        return P.table_price(self.merged(self.mask if mask is None else mask), self.V)

    def _positions(self, mask):
        """The mask as `prequential_bits` takes it: kept columns j as positions j + 1 (ctx[:, k - 1] = column k - 1)."""
        return [int(j) + 1 for j in np.flatnonzero(mask)]

    def _prequential(self, mask, limit):
        if not self.log:
            raise ValueError("Store.sleep(price='prequential') needs the observations in order: build with keep_order=True")
        ctx = np.concatenate([r for r, _y in self.log]).astype(np.int64)
        nxt = np.concatenate([y for _r, y in self.log]).astype(np.int64)
        return P.prequential_bits(ctx, nxt, self._positions(mask), self.V, limit)

    def sleep(self, strict=True, price="table", limit=200000):
        """Drop cells while the table gets cheaper AND (strict) explains the evidence no worse -- no new exceptions
        (greedy; the centre, when the field has one, is always kept). `strict=False` lets the price alone decide (E35's
        noisy targets). `price="table"` is E24's two-part code; `price="prequential"` is E33's blended-backoff code
        (the store must keep its observations in order). Returns (bits before, bits after, cells kept); afterwards the
        table, majority and stats are the merged evidence under the new mask and `cost` is its price."""
        if not self.full:
            return 0.0, 0.0, int(self.mask.sum())
        if price == "table":
            merged0 = self.merged(self.mask)
            before, wrong0 = P.table_price(merged0, self.V), self._wrong(merged0)
            score = lambda m: P.table_price(self.merged(m), self.V)               # noqa: E731
            ok = (lambda m: True) if not strict else (lambda m: self._wrong(self.merged(m)) <= wrong0)
        elif price == "prequential":
            before = self._prequential(self.mask, limit)
            score = lambda m: self._prequential(m, limit)                         # noqa: E731
            ok = lambda m: True                                                   # noqa: E731
        else:
            raise ValueError(f"Store.sleep: price must be 'table' or 'prequential', got {price!r}")
        mask, cur = self.mask.copy(), before
        while True:
            best_mask, best_price = None, cur
            for c in np.flatnonzero(mask):
                if self.centre is not None and c == self.centre:
                    continue
                cand = mask.copy()
                cand[c] = False
                pr = score(cand)
                if pr < best_price and ok(cand):
                    best_mask, best_price = cand, pr
            if best_mask is None:
                break
            mask, cur = best_mask, best_price
        self.mask = mask
        merged = self.merged(mask)
        self.table = merged
        self.majority = {k: max(c, key=c.get) for k, c in merged.items()}
        self.stats = {k: [max(c.values()), sum(c.values()) - max(c.values())] for k, c in merged.items()}
        self.cost = cur
        self.bits_before, self.bits_after = before, cur
        self.forms = {}
        self._karr_n = -1
        return before, cur, int(mask.sum())

    def chain(self):
        """E33's backoff chain as masks: the mask, then without its farthest kept cell (for a `back` field the oldest
        position: `price.backoff_chain`'s rule), ..., the empty mask (the unigram). The reader falls back along it."""
        chain, mask = [], self.mask.copy()
        while True:
            chain.append(mask.copy())
            kept = np.flatnonzero(mask)
            if len(kept) == 0:
                break
            far = max(kept, key=lambda c: (max(abs(self.field.offsets[c][0]), abs(self.field.offsets[c][1])),
                                           self.field.offsets[c][0] ** 2 + self.field.offsets[c][1] ** 2, c))
            mask[far] = False
        return chain

    # -- the entries -------------------------------------------------------------------------------------------------
    def entries(self):
        """[(key, counts, stats)] in table order: key an int16 array over the kept cells, counts {y: n}, stats [right, wrong]."""
        return [(np.frombuffer(k, dtype=np.int16), c, self.stats[k]) for k, c in self.table.items()]

    def __len__(self):
        return len(self.table)

    # -- outcome keys (MOVES GoalModel.record / refute / live / predicts / known) ---------------------------------------
    def record(self, keys, tag=None):
        """Confirm each DISTINCT whole window once: the windows around the cells that changed at a scoring (or fatal)
        transition (GoalModel.record; the caller selects the rows of those cells)."""
        rows = self._rows(keys)
        target = self.outcome_keys.setdefault(tag, {})
        for fk in {rows[t].tobytes() for t in range(len(rows))}:
            target.setdefault(fk, [0, 0])[0] += 1

    def refute(self, keys, tag=None):
        """Hindsight: every stored key among these windows is refuted once per window it appears in (GoalModel.refute)."""
        rows = self._rows(keys)
        target = self.outcome_keys.get(tag, {})
        for t in range(len(rows)):
            fk = rows[t].tobytes()
            if fk in target:
                target[fk][1] += 1

    def live(self, tag=None):
        """The keys trusted now: confirmed more often than refuted (GoalModel.live)."""
        return {k for k, (c, r) in self.outcome_keys.get(tag, {}).items() if c > r}

    def predicts(self, keys, tag=None):
        """True when a live key is among these whole windows (GoalModel.predicts)."""
        keys_live = self.live(tag)
        if not keys_live:
            return False
        rows = self._rows(keys)
        return any(rows[t].tobytes() in keys_live for t in range(len(rows)))

    def known(self):
        return any(self.live(tag) for tag in self.outcome_keys)

    # -- the tensor forms -------------------------------------------------------------------------------------------
    def _token_ids(self, which):
        ids = range(len(self.table))
        if which == "tokens":
            return [i for i, k in zip(ids, self.table) if self.forms.get(k) != "rows"]
        if which == "rows":
            return [i for i, k in zip(ids, self.table) if self.forms.get(k) == "rows"]
        return list(ids)

    def as_tokens(self, codec, value=None, conf=False, which="tokens", **consts):
        """E28's memory tokens: one per entry in table order (`codec.entry`), the kept cells as one-hots in their
        subspaces (a dropped cell all-zero: a wildcard), `**consts` written into every entry (E28: `action=ai`), the
        majority in `value` (default the store's). `conf=True` also writes right / (right + wrong) into the class's
        `conf` field. `which="tokens"` leaves out the entries `consolidate` moved to rows."""
        value = self.value if value is None else value
        keys = list(self.table)
        ids = self._token_ids(which)
        if not ids:
            return torch.zeros(0, codec.d)
        out = []
        for i in ids:
            k = keys[i]
            fields = self.key_fields(k)
            fields.update(consts)
            c = None
            if conf:
                right, wrong = self.stats[k]
                c = right / (right + wrong) if right + wrong else 0.0
            out.append(codec.entry(fields, {value: int(self.majority[k])}, conf=c))
        return torch.stack(out)

    def as_bank(self, codec, value=None, which="tokens", **consts):
        """The same rows split for the k/v cache: keys = the token with its value subspace zeroed, values = the value
        subspace alone (identical arithmetic to attending the tokens; positions free — the GD-compatible form)."""
        value = self.value if value is None else value
        tok = self.as_tokens(codec, value=value, which=which, **consts)
        s = codec.layout[value]
        keys, values = tok.clone(), torch.zeros_like(tok)
        keys[:, s] = 0.0
        values[:, s] = tok[:, s]
        return keys, values

    def _onehot_key_row(self, layout, key, consts):
        row = torch.zeros(layout.d)
        for name, sym in {**self.key_fields(key), **consts}.items():
            s = layout[name]
            w = s.stop - s.start
            if not -w <= int(sym) < w:
                raise ValueError(f"symbol {sym} is outside subspace {name!r} of width {w}")
            row[s.start + (int(sym) % w)] = 1.0
        return row

    def as_rows(self, layout, code="onehot", k=None, proj=None, value=None, which="all", **consts):
        """The MLP form (DESIGN §21.7; the `Row` instruction): per entry (key_row, threshold, value_row) with
        key_row the entry's one-hot key over the layout (`code="onehot"`) or its k-of-d code (`code="sparse"`, through
        `proj` or a `SparseProjection(layout.d, layout.d, k)` built once and kept as `self.proj`), threshold =
        k_in - 1/2 with k_in the ones in key_row (a full match fires, a partial one does not), value_row the majority's
        one-hot in `value`. `which="rows"` gives only the entries `consolidate` moved."""
        value = self.value if value is None else value
        keys = list(self.table)
        ids = self._token_ids(which)
        if code == "sparse":
            if proj is None:
                if self.proj is None or self.proj.d_in != layout.d:
                    if k is None:
                        raise ValueError("as_rows(code='sparse') needs k (the ones per code) or a SparseProjection")
                    self.proj = SparseProjection(layout.d, layout.d, k)
                proj = self.proj
        elif code != "onehot":
            raise ValueError(f"as_rows: code must be 'onehot' or 'sparse', got {code!r}")
        s = layout[value]
        out = []
        for i in ids:
            key = keys[i]
            key_row = self._onehot_key_row(layout, key, consts)
            if code == "sparse":
                key_row = proj(key_row)
            k_in = float(key_row.sum())
            value_row = torch.zeros(layout.d)
            value_row[s.start + int(self.majority[key])] = 1.0
            out.append((key_row, k_in - 0.5, value_row))
        return out

    # -- the offline readers (ZipLearn's apparatus; never inside a pass) ---------------------------------------------
    def _keys_array(self):
        """MOVES NearestRule._keys_array: the stored masked keys (N, kept) int16 in table order and their majorities."""
        if self._karr is None or self._karr_n != len(self.table):
            keys = list(self.table.keys())
            self._karr = (np.stack([np.frombuffer(k, dtype=np.int16) for k in keys]) if keys
                          else np.zeros((0, int(self.mask.sum())), np.int16))
            self._kval = np.array([self.majority[k] for k in keys], dtype=np.int16)
            self._karr_n = len(self.table)
        return self._karr, self._kval

    def _onehot(self, rows):
        """Masked rows (Q, kept) of symbols in -1..V-1 -> one-hot codes (Q, kept * (V + 1)) with -1 at index V."""
        rows = np.asarray(rows, dtype=np.int64)
        C = self.V + 1
        q = torch.from_numpy(rows % C)
        oh = torch.zeros(len(rows), rows.shape[1] * C, dtype=torch.float32)
        if rows.shape[1]:
            oh.scatter_(1, q + torch.arange(rows.shape[1])[None, :] * C, 1.0)
        return oh

    def _matches(self, masked):
        """(Q, N) matching-cell counts between query masked rows and every stored key = kept - Hamming distance: the
        matmul of one-hot codes, which is exactly what a `Match` head's q.k computes over `as_tokens`."""
        karr, _kval = self._keys_array()
        return self._onehot(masked) @ self._onehot(karr).T

    def nearest(self, keys, M=None):
        """MOVES NearestRule.predict_nearest's default: for each whole window, the majority of the Hamming-NEAREST
        stored key (ties by stored index). `M=None` is the exact hard read (argmax of the matches); a finite `M` is the
        soft read the attention head computes (softmax(M * matches) over the stored values, then argmax)."""
        rows = self._rows(keys)
        karr, kval = self._keys_array()
        if not len(karr):
            raise ValueError("Store.nearest: the table is empty")
        matches = self._matches(rows[:, self.mask])
        if M is None:
            idx = matches.argmax(1)
            out = kval[idx.numpy()]
        else:
            p = torch.softmax(float(M) * matches, 1)
            votes = p @ torch.nn.functional.one_hot(torch.from_numpy(kval.astype(np.int64)), self.V).float()
            out = votes.argmax(1).numpy().astype(np.int16)
        return int(out[0]) if np.asarray(keys).ndim == 1 else out

    def predict_nearest(self, frame):
        """MOVES NearestRule.predict_nearest: known windows by hash, unknown ones by the nearest stored key; returns
        (frame, number of unknown cells)."""
        frame = np.asarray(frame)
        _full, masked = windows_grid(frame, self.field, self.mask)
        out = frame.copy().reshape(-1)
        unknown = []
        for t in range(out.size):
            y = self.majority.get(masked[t].tobytes())
            if y is None:
                unknown.append(t)
            else:
                out[t] = y
        if unknown and len(self.table):
            q = masked[unknown]
            for s in range(0, len(q), 256):
                matches = self._matches(q[s:s + 256])
                out[np.array(unknown[s:s + 256])] = self._kval[matches.argmax(1).numpy()]
        return out.reshape(frame.shape), len(unknown)

    def candidates(self, keys, k):
        """Product-key candidates (Lample et al. 2019, as DESIGN §21.1 item 8 corrects it): the kept cells split into
        two halves; per query the top-k stored sub-keys of each half (by matching cells, ties by index); the CANDIDATES
        are the stored entries whose two halves are both selected; they are rescored EXACTLY on the whole key and
        returned best first. One list of entry ids (table order) per query."""
        rows = self._rows(keys)
        karr, _kval = self._keys_array()
        if not len(karr):
            return [[] for _ in rows]
        masked = rows[:, self.mask]
        n_kept = karr.shape[1]
        m = max(1, n_kept // 2)
        halves = [(0, m), (m, n_kept)] if n_kept > 1 else [(0, n_kept)]
        sel = []
        for a, b in halves:
            sub_u, inv = np.unique(karr[:, a:b], axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            scores = (self._onehot(masked[:, a:b]) @ self._onehot(sub_u).T).numpy()
            order = np.argsort(-scores, axis=1, kind="stable")[:, :k]     # top-k sub-keys, ties by index
            sel.append((inv, [set(o.tolist()) for o in order]))
        exact = self._matches(masked).numpy()
        out = []
        for qi in range(len(rows)):
            ids = [e for e in range(len(karr)) if all(inv[e] in chosen[qi] for inv, chosen in sel)]
            ids.sort(key=lambda e: (-exact[qi, e], e))
            out.append(ids)
        return out

    def _counts_matrix(self):
        """(N, V) counts per stored entry in table order (the kernel readers' votes)."""
        C = np.zeros((len(self.table), self.V), dtype=np.float32)
        for i, counts in enumerate(self.table.values()):
            for y, c in counts.items():
                C[i, y] = c
        return torch.from_numpy(C)

    def loo_kernel_width(self, sample=3000, hs=HS, seed=0, eps=1e-3, chunk=1024):
        """F2's leave-one-out choice of the Hamming kernel's width (research/expI_frames_chowliu_loo.py `loo_select`,
        the `loo_window` score at k = all): on a subsample of `sample` stored windows, each window's own counts are
        predicted from ALL OTHER stored windows with weights exp(-d_H / h) times their counts (+ eps), and h is the
        grid value with the largest leave-one-out log-likelihood. Returns h; `self.loo_table` holds bits per h."""
        karr, _kval = self._keys_array()
        N = len(karr)
        if N < 2:
            raise ValueError("loo_kernel_width needs at least two stored entries")
        rng = np.random.default_rng(seed)
        if isinstance(sample, (int, np.integer)):
            sub = np.sort(rng.choice(N, size=min(int(sample), N), replace=False))
        else:
            sub = np.asarray(sample, dtype=np.int64)
        counts = self._counts_matrix()
        koh = self._onehot(karr)
        kept = karr.shape[1]
        ll = {h: 0.0 for h in hs}
        for s in range(0, len(sub), chunk):
            ids = sub[s:s + chunk]
            D = kept - koh[ids] @ koh.T                                        # exact Hamming distances
            D[torch.arange(len(ids)), torch.from_numpy(ids)] = 1e6            # leave the window itself out
            Cs = counts[ids]
            for h in hs:
                votes = torch.exp(-D / h) @ counts + eps
                p = votes / votes.sum(1, keepdim=True)
                ll[h] += float((Cs * torch.log2(p)).sum())
        tot = float(counts[sub].sum())
        self.loo_table = {h: -ll[h] / tot for h in hs}                        # bits per observation
        return max(hs, key=lambda h: ll[h])

    # -- consolidation (DESIGN §21.7) ---------------------------------------------------------------------------------
    def consolidate(self, n0=8, eps0=0.1, price=None, layout=None, code="onehot", k=None, proj=None, **consts):
        """Move settled entries from tokens to rows: an entry qualifies when its evidence is settled (n >= n0
        observations, exception rate <= eps0) AND the interference exceptions the row form produces on the evidence
        (`full` replayed through the written rows: a row fires on a window iff its key is a subset of the window's
        code, and the fired rows' values decide) cost fewer bits than the tokens save. Rows are built as `as_rows`
        does (over `layout`, or over the store's own one-hot code of the kept cells when `layout` is None).

        Where §21.7 is silent, the saving is the KEY's bits: a token stores its address explicitly
        (kept cells x log2(V + 1) bits per entry), a row on shared sparse units stores it superposed (Willshaw), so
        `price(store, ids) -> bits saved` defaults to len(ids) * kept * log2(V + 1); the interference is priced by
        the two-part code's exception terms (n H(wrong/n) + wrong log2(V - 1)) over the replayed windows, minus what
        those windows already paid as tokens. The whole settled set is tried first, then each entry alone.
        Returns the ids (table order) moved; `forms[key] = "rows"` for them."""
        keys = list(self.table)
        settled = []
        for i, kk in enumerate(keys):
            right, wrong = self.stats[kk]
            n = right + wrong
            if n >= n0 and (wrong / n if n else 0.0) <= eps0 and self.forms.get(kk) != "rows":
                settled.append(i)
        if not settled:
            return []
        kept = len(self.kept)
        if price is None:
            price = lambda store, ids: len(ids) * kept * math.log2(store.V + 1)          # noqa: E731
        # the key codes: over the layout, or the store's own one-hot code of the kept cells
        if layout is not None:
            key_rows = torch.stack([self._onehot_key_row(layout, kk, consts) for kk in keys])
        else:
            karr, _kval = self._keys_array()
            key_rows = self._onehot(karr)
        if code == "sparse":
            if proj is None:
                if k is None:
                    raise ValueError("consolidate(code='sparse') needs k or a SparseProjection")
                proj = SparseProjection(key_rows.shape[1], key_rows.shape[1], k)
            key_rows = proj(key_rows)
        thresholds = key_rows.sum(1) - 0.5
        # the evidence's queries (whole windows projected by the mask) in the same code, with their counts
        full_keys = list(self.full)
        full_masked = np.stack([np.frombuffer(fk, dtype=np.int16)[self.mask] for fk in full_keys])
        if layout is not None:
            q_rows = torch.stack([self._onehot_key_row(layout, fm, consts) for fm in full_masked])
        else:
            q_rows = self._onehot(full_masked)
        if code == "sparse":
            q_rows = proj(q_rows)
        values = torch.tensor([int(self.majority[kk]) for kk in keys], dtype=torch.long)
        counts = [self.full[fk] for fk in full_keys]
        n_tot = np.array([sum(c.values()) for c in counts], dtype=np.float64)
        own_wrong = np.array([sum(c.values()) - max(c.values()) for c in counts], dtype=np.float64)

        def interference_bits(ids):
            idx = torch.tensor(ids, dtype=torch.long)
            fire = (q_rows @ key_rows[idx].T) >= thresholds[idx][None, :]           # (n_full, len(ids))
            bits = 0.0
            for w in np.flatnonzero(fire.any(1).numpy()):
                votes = torch.zeros(self.V)
                for j in np.flatnonzero(fire[w].numpy()):
                    votes[values[idx[j]]] += 1.0
                answer = int(votes.argmax())
                wrong_rows = n_tot[w] - counts[w].get(answer, 0)
                n = n_tot[w]
                bits += (n * P.entropy(wrong_rows / n) + wrong_rows * math.log2(self.V - 1)
                         - (n * P.entropy(own_wrong[w] / n) + own_wrong[w] * math.log2(self.V - 1)))
            return bits

        def adopt(ids):
            return interference_bits(ids) < price(self, ids)

        moved = settled if adopt(settled) else [i for i in settled if adopt([i])]
        for i in moved:
            self.forms[keys[i]] = "rows"
        return moved

    # -- the evidence on disk -----------------------------------------------------------------------------------------
    def save_npz(self, path):
        """Keys (int16), values, counts (int32) and stats: the evidence the tensor forms are regenerated from, plus
        the mask, the online cost, the outcome keys, the ordered log (if kept) and the consolidation forms."""
        n = self.field.n
        fk, fy, fc = [], [], []
        for k, counts in self.full.items():
            for y, c in counts.items():
                fk.append(np.frombuffer(k, dtype=np.int16)); fy.append(y); fc.append(c)
        tk, ty, tc, tstat, tmaj, tform = [], [], [], [], [], []
        for k, counts in self.table.items():
            arr = np.frombuffer(k, dtype=np.int16)
            for y, c in counts.items():
                tk.append(arr); ty.append(y); tc.append(c)
            tstat.append(self.stats[k]); tmaj.append(self.majority[k]); tform.append(1 if self.forms.get(k) == "rows" else 0)
        ok_tag, ok_key, ok_cr = [], [], []
        for tag, keys in self.outcome_keys.items():
            for k, cr in keys.items():
                ok_tag.append("" if tag is None else str(tag)); ok_key.append(np.frombuffer(k, dtype=np.int16)); ok_cr.append(cr)
        kept = int(self.mask.sum())
        arrays = dict(
            field=np.array(json.dumps(self.field.to_dict())), V=np.array(self.V), mask=self.mask,
            cost=np.array(self.cost), n_obs=np.array(self.n_obs), value=np.array(self.value),
            capacity=np.array(-1 if self.capacity is None else self.capacity),
            full_keys=_stack(fk, n), full_values=np.array(fy, dtype=np.int16),
            full_counts=np.array(fc, dtype=np.int32),
            table_keys=_stack(tk, kept), table_values=np.array(ty, dtype=np.int16),
            table_counts=np.array(tc, dtype=np.int32),
            entry_keys=_stack([np.frombuffer(k, dtype=np.int16) for k in self.table], kept),
            stats=_stack(tstat, 2, np.int32), majority=np.array(tmaj, dtype=np.int16),
            forms=np.array(tform, dtype=np.int8),
            outcome_tags=np.array(ok_tag), outcome_keys=_stack(ok_key, n),
            outcome_cr=_stack(ok_cr, 2, np.int32),
            keep_order=np.array(self.log is not None),
        )
        if self.log:
            arrays["log_rows"] = np.concatenate([r for r, _y in self.log]).astype(np.int16)
            arrays["log_ys"] = np.concatenate([y for _r, y in self.log]).astype(np.int16)
            arrays["log_lens"] = np.array([len(y) for _r, y in self.log], dtype=np.int64)
        np.savez(Path(path), **arrays)

    @classmethod
    def load_npz(cls, path):
        z = np.load(Path(path), allow_pickle=False)
        field = Field.from_dict(json.loads(str(z["field"])))
        cap = int(z["capacity"])
        store = cls(field, int(z["V"]), keep_order=bool(z["keep_order"]), value=str(z["value"]), capacity=None if cap < 0 else cap)
        store.mask = z["mask"].astype(bool)
        store.cost, store.n_obs = float(z["cost"]), int(z["n_obs"])
        for row, y, c in zip(z["full_keys"], z["full_values"], z["full_counts"]):
            store.full.setdefault(np.ascontiguousarray(row).tobytes(), {})[int(y)] = int(c)
        for row, st, maj, form in zip(z["entry_keys"], z["stats"], z["majority"], z["forms"]):
            k = np.ascontiguousarray(row).tobytes()
            store.table[k] = {}
            store.stats[k] = [int(st[0]), int(st[1])]
            store.majority[k] = int(maj)
            if form:
                store.forms[k] = "rows"
        for row, y, c in zip(z["table_keys"], z["table_values"], z["table_counts"]):
            store.table[np.ascontiguousarray(row).tobytes()][int(y)] = int(c)
        for tag, row, cr in zip(z["outcome_tags"], z["outcome_keys"], z["outcome_cr"]):
            t = None if str(tag) == "" else str(tag)
            store.outcome_keys.setdefault(t, {})[np.ascontiguousarray(row).tobytes()] = [int(cr[0]), int(cr[1])]
        if "log_rows" in z:
            rows, ys, lens = z["log_rows"], z["log_ys"], z["log_lens"]
            store.log, at = [], 0
            for n in lens:
                store.log.append((np.ascontiguousarray(rows[at:at + n]), np.ascontiguousarray(ys[at:at + n])))
                at += int(n)
        return store

    def __repr__(self):
        return (f"Store({self.field}, V={self.V}, {len(self.table)} entries over {int(self.mask.sum())} cells, "
                f"{len(self.full)} whole windows, cost {self.cost:.1f} bits)")


def _stack(rows, width, dtype=np.int16):
    """Rows of `width` as a (len(rows), width) array, also when there are none or width is 0."""
    out = np.zeros((len(rows), width), dtype=dtype)
    for i, r in enumerate(rows):
        out[i] = r
    return out


def best_store(stores):
    """MOVES ActionModel.best: among stores of one action at different radii, the cheapest description predicts
    (price + log2(number of candidates), rounded to 1e-9; ties to the smaller radius)."""
    stores = list(stores)
    return min(stores, key=lambda s: (round(s.cost + math.log2(len(stores)), 9), s.field.r))


__all__ = ["BORDER", "HS", "Field", "windows_grid", "windows_seq", "SparseProjection", "Store", "best_store"]
