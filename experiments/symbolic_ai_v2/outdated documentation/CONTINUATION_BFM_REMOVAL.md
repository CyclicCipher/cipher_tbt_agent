# CONTINUATION: Remove BFM — Route All Arithmetic Through `_compose`

**Status:** Not started. Documentation only (Mistake #48 written in MISTAKES.md).
**Date written:** 2026-03-16
**Priority:** Fix NL benchmark (all trace-format ops currently 0% in NL mode).

---

## The Problem in One Sentence

`_binary_fmaps` (BFM) is a pre-computed lookup table. It is empty in NL mode
because `build_binary_functional_maps` rejects all tokens with `len(t) != 1`,
filtering out every NL word token (`'three'`, `'five'`, etc.). This causes
`discover_relation_rules` to find zero rules, the lambda library to be empty,
and all trace-format ops to return `{}`.

---

## Why BFM Exists (and Why It Shouldn't)

BFM was introduced to give `RelationRule.evaluate()` a fast way to compute
`add(a, b)`, `mul(a, b)`, etc. at inference time. But `_compose` (Level 0.7,
NNO fold engine) already does this correctly for any token vocabulary. BFM is
just a pre-computed cache of `_compose` results — and it bakes in the
assumption that tokens are single characters, breaking everything for NL tokens.

Every "fix" to BFM has produced a new intractable problem. See MISTAKES.md #48.

---

## Four Cascading Bugs in BFM (all from `len(t) != 1`)

### #48a — Input filter rejects NL tokens
`process_discover.py` line 1663:
```python
if any(len(t) != 1 for t in inputs):
    continue  # BUG: 'three' has len=3, so entire BFM is empty in NL mode
```

### #48b — Result join corrupts multi-token NL outputs
`process_discover.py` line 1666:
```python
result_tok = ''.join(val)  # BUG: ('one','four') → 'onefour' (not two tokens)
```

### #48c — Two-digit extension code uses character indexing
`predict.py` lines 240–313: hardcoded `len(_res) == 2`, `_R[0]`, `_R[1]`,
`tuple(_R)` — all character-level assumptions that break for NL tokens.

### #48d — Output split uses `list(result_str)` (character split)
`relation_store.py` lines 752, 863:
```python
output_toks.extend(list(result_str))  # BUG: 'eleven' → ['e','l','e','v','e','n']
```
`predict.py` line 620:
```python
_kl_out.extend(list(_kl_vals[_kl_rule.output_role]))  # same bug
```
`lambda_term.py` line 205:
```python
for ch in result_str:  # BUG: iterates characters, not tokens
```

---

## The Fix: Replace BFM With a `ComposeEngine`

### Design

Create a `ComposeEngine` dataclass in `process_discover.py` that bundles all
NNO state and provides a `compute(op, a_tok, b_tok) -> tuple[str, ...] | None`
method. This replaces the BFM dict everywhere. It calls `_compose` on demand
so it works for any token vocabulary.

```python
@dataclass
class ComposeEngine:
    fc_lookup: dict
    fold_rules: dict
    succ_map: dict[str, str]
    carry_el: str
    carry_out: tuple
    zero: str
    cache: dict = field(default_factory=dict)

    def compute(self, op: str, a: str, b: str) -> tuple[str, ...] | None:
        """Compute op(a, b) via NNO fold. a and b are single token strings."""
        return _compose(
            op, ((a,), (b,)),
            self.fc_lookup, self.fold_rules,
            self.succ_map, self.carry_el, self.carry_out, self.zero,
            self.cache,
        )

    def get_op_map(self, op: str) -> '_OpProxy':
        """Returns a proxy so bfm.get(op, {}).get((a, b)) still works."""
        return _OpProxy(self, op)


@dataclass
class _OpProxy:
    engine: ComposeEngine
    op: str

    def get(self, key: tuple, default=None):
        if len(key) != 2:
            return default
        result = self.engine.compute(self.op, key[0], key[1])
        if result is None:
            return default
        return result  # returns tuple[str,...], not str
```

**IMPORTANT:** The result is now a `tuple[str, ...]`, not a concatenated string.
All sites that previously did `list(result_str)` or `for ch in result_str:`
must be updated to treat the result as a token tuple directly.

---

## All Files That Must Change

### 1. `process_discover.py`

**Add:** `ComposeEngine` and `_OpProxy` classes (see above).

**Remove:** `build_binary_functional_maps` — this function builds the BFM
table. Delete it (or keep as a stub returning `{}` for backward compat).

**Also export** `ComposeEngine` so `predict.py` can import it.

---

### 2. `predict.py` — Predictor `__init__`

**Lines 190–313: Delete entirely.** This is all the BFM building code:
- Lines 190–238: `_binary_fmaps = build_binary_functional_maps(...)` + NNO completion
- Lines 240–283: Two-digit add/sub extension (char-indexed, wrong)
- Lines 285–313: Two-digit mul extension (char-indexed, wrong)

**Replace lines 190–238 with:**
```python
# Build ComposeEngine — replaces BFM (Mistake #48).
# No pre-computed table; _compose is called on demand at inference time.
if self._compose_succ_map:
    from experiments.symbolic_ai_v2.ctkg.learning.process_discover import ComposeEngine
    self._engine = ComposeEngine(
        fc_lookup=self._fc_lookup,
        fold_rules=self._fold_rules,
        succ_map=self._compose_succ_map,
        carry_el=self._compose_carry_el,
        carry_out=self._compose_carry_out,
        zero=self._compose_zero,
        cache=self._compose_cache,
    )
else:
    self._engine = None
```

**Lines 326–395 (concat/div/fst + RelationStore + discover_relation_rules):**
Change `_binary_fmaps or {}` → `self._engine` throughout:
```python
_op_rr = discover_relation_rules(
    _op_rels, self._engine, mismatch_tolerance=_mm_tol   # engine, not bfm
)
...
_disc_role, _chains = discover_kleisli_chains(
    _op_rels, self._engine                                 # engine, not bfm
)
```

Remove `self._binary_fmaps` attribute entirely. It is referenced in:
- Line 375: `self._binary_fmaps: dict = _binary_fmaps or {}`
- Line 425: `self._binary_fmaps: dict = {}`
- Line 516: `lambda_predict(..., self._binary_fmaps, ...)`
- Line 541: `predict_alternatives_from_rules(..., self._binary_fmaps)`
- Line 609: `_kl_rule.evaluate(_kl_vals, self._binary_fmaps)`
- Line 637: `lambda_predict(..., self._binary_fmaps, ...)`
- Line 649: `_equalizer_predict(..., self._binary_fmaps, ...)`
- Line 665: `_pullback_predict(..., self._binary_fmaps, ...)`

Replace all `self._binary_fmaps` → `self._engine`.

**Line 620 (Kleisli output split):**
```python
# OLD:
_kl_out.extend(list(_kl_vals[_kl_rule.output_role]))
# NEW: result is already a token tuple from engine.compute()
_result_toks = _kl_vals[_kl_rule.output_role]
if isinstance(_result_toks, tuple):
    _kl_out.extend(_result_toks)
else:
    _kl_out.append(_result_toks)
```

**`_equalizer_predict` and `_pullback_predict`:** These take `bfm` as a
parameter. Update their signatures to take `engine` instead and update internal
`bfm.get(op, {}).get((a, b))` calls to `engine.compute(op, a, b)` with result
unpacked from tuple.

---

### 3. `relation_store.py`

**`RelationRule.evaluate()` (lines 515–536):**
```python
# OLD signature:
def evaluate(self, role_values: dict[str, str], bfm: dict) -> dict[str, float]:
    ...
    result = bfm.get(self.op_name, {}).get((v1, v2))
    ...

# NEW signature:
def evaluate(self, role_values: dict[str, str], engine) -> dict[str, float]:
    v1 = role_values.get(self.arg1)
    v2 = role_values.get(self.arg2)
    if v1 is None or v2 is None:
        return {}
    result_toks = engine.compute(self.op_name, v1, v2)
    if result_toks is None:
        return {}
    # Join tokens with '\x00' as internal separator for multi-token results.
    result_key = '\x00'.join(result_toks)
    return {result_key: self.confidence}
```

**`discover_relation_rules()` (line 539):**
Change signature: `bfm: dict` → `engine`.
Inner loop (line 646): `for op_name, op_map in bfm.items():` →
```python
# Enumerate all ops the engine knows about
for op_name in engine.known_ops():
    for role_a in source_roles:
        for role_b in source_roles:
            n_match = n_unknown = n_mismatch = 0
            for role_vals, target_val in examples:
                va = role_vals.get(role_a)
                vb = role_vals.get(role_b)
                if va is None or vb is None:
                    n_unknown += 1
                    continue
                result_toks = engine.compute(op_name, va, vb)
                if result_toks is None:
                    n_unknown += 1
                    continue
                result = '\x00'.join(result_toks)
                if result != target_val:
                    n_mismatch += 1
                else:
                    n_match += 1
            ...
```

**Add `known_ops()` to `ComposeEngine`:**
```python
def known_ops(self) -> list[str]:
    """Return list of ops the fold_rules knows about, plus adjunction inverses."""
    ops = list(self.fold_rules.keys())
    # Add standard adjunction pairs if discovered
    return ops
```

**`predict_from_relation_rules()` lines 736, 752:**
```python
# OLD line 736:
role_values[rname] = ''.join(toks)
# NEW — join with separator:
role_values[rname] = '\x00'.join(toks)

# OLD line 752:
output_toks.extend(list(result_str))
# NEW — split by separator:
output_toks.extend(result_str.split('\x00'))
```

**`predict_alternatives_from_rules()` lines 809, 863:**
Same fix: `''.join(toks)` → `'\x00'.join(toks)`, `list(result_str)` → `result_str.split('\x00')`.

**`discover_kleisli_chains()` — same signature fix:** `bfm: dict` → `engine`.

**`discover_relation_rules` target_val collection (lines 621, 625):**
```python
# OLD:
role_vals[rname] = ''.join(toks)
# NEW:
role_vals[rname] = '\x00'.join(toks)
```

---

### 4. `lambda_term.py`

**`eval_expr()` (lines 112–138):**
```python
# OLD signature:
def eval_expr(expr, env, bfm: dict) -> Optional[str]:
    ...
    return bfm.get(expr.head, {}).get((a_val, b_val))

# NEW signature:
def eval_expr(expr, env, engine) -> Optional[str]:
    ...
    result_toks = engine.compute(expr.head, a_val, b_val)
    if result_toks is None:
        return None
    return '\x00'.join(result_toks)
```

**`eval_term()` (lines 141–215):**
```python
# OLD signature:
def eval_term(term, arg_tokens, bfm: dict, output_so_far) -> ...:

# NEW signature:
def eval_term(term, arg_tokens, engine, output_so_far) -> ...:
```

**Line 205 (character iteration of result):**
```python
# OLD:
for ch in result_str:
    if pos == k:
        return {ch: acc_conf}
    ...

# NEW: result_str is '\x00'-joined; split back to tokens:
result_toks = result_str.split('\x00')
for tok in result_toks:
    if pos == k:
        return {tok: acc_conf}
    if output_so_far[pos] != tok:
        return None
    pos += 1
```

**`lambda_predict()` (line 342, 382, 412):** `bfm: dict` → `engine` in signature,
pass through to `eval_term`.

---

### 5. `_equalizer_predict` and `_pullback_predict` in `predict.py`

These functions take `bfm` as a parameter and call `bfm.get(op, {}).get((a,b))`.
Update to take `engine` and call `engine.compute(op, a, b)` → unpack result tuple.

---

## `'\x00'` Separator Rationale

Multi-token results (e.g. `add('nine', 'five') = ('one', 'four')` in NL mode)
must be stored as a single string key in `role_values` dicts (the current dict
type is `dict[str, str]`). Using `'\x00'` as separator:
- Is invisible in any real token vocabulary
- Allows reversible round-trips: `'\x00'.join(toks)` / `toks.split('\x00')`
- Means single-token results are unchanged (`'eight'.split('\x00') = ['eight']`)

The alternative is to change `role_values` from `dict[str, str]` to
`dict[str, tuple[str,...]]` everywhere — this is cleaner but requires more churn.
Use `'\x00'` for the first pass; refactor to tuples in a follow-up if desired.

---

## Concat / Div / Fst (currently BFM-only ops)

These are currently injected into `_binary_fmaps` at Predictor init:
- `concat(a, b) = a + b` (string concatenation, used for `sq` trace identity)
- `div(r, a) = b` where `mul(a, b) = r` (adjunction inverse)
- `fst(a, b) = a` (first projection for Kleisli base step)

These are NOT arithmetic ops discoverable by `fold_rules`. Add them to
`ComposeEngine.compute()` as special cases:
```python
def compute(self, op: str, a: str, b: str) -> tuple[str, ...] | None:
    # Special non-NNO ops
    if op == 'concat':
        return (a + b,)                    # only correct for single-char tokens!
    if op == 'fst':
        return (a,)
    if op == 'div':
        # enumerate: find b such that mul(a_candidate, b) = a
        # use adj_solve if available, otherwise brute force over NNO chain
        ...
    # NNO fold
    return _compose(op, ((a,), (b,)), ...)
```

**NOTE:** `concat(a, b) = a + b` is character concatenation — it was only ever
meaningful in standard digit mode. For NL mode it produces `'threefive'` which is
meaningless. If `sq` trace needs a token-join identity, use a proper `pair` op
that emits two tokens. Revisit when sq/pow trace benchmarks are addressed.

---

## Test Plan

After implementing:

```bash
cd C:/Users/julia/GitHub/predictive-coding-agent
./venv/Scripts/python.exe -m pytest experiments/symbolic_ai_v2/ -x -q
```

Then run the NL benchmark specifically:
```bash
./venv/Scripts/python.exe experiments/symbolic_ai_v2/ctkg/benchmarks/anon_math_benchmark.py
```

Expected: `linear_trace`, `algebra_trace`, `bernoulli_trace` all reach 100% OOD
in both standard and NL mode. `power_trace` may improve from 30% but is not
the primary target (it has a separate depth-mismatch issue).

---

## What NOT to Do

- Do not add new BFM-like caches. Mistake #48.
- Do not add `int()` calls. Iron Rule.
- Do not hardcode operator names (`'add'`, `'mul'`, etc.) anywhere outside
  `ComposeEngine.compute()` special cases.
- Do not create new dict-of-dicts structures that pre-enumerate token pairs.
- `_compose` is the ground truth; everything else must call it.
