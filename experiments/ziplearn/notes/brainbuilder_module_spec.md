# BrainBuilder — module spec (the file-by-file skeleton), 2026-09-22

*The implementer's map for DESIGN.md §21. One line per class or function: what it is, and what existing code it MOVES
(the old location is deleted in the same commit — `feedback_decisive_full_cutover`). Nothing here calls `backward`.
Regression numbers that must not move: E18 1.000; E28 300/300 + 230/230 (seed 0), 277/277 + 232/232 (seed 1), ties
1,434 / 1,475; E24 LockPath masks [3, 3, 3, 2] and 9,192 → 660 bits; E6 retention 1.000, 3 blocks; E33 1.873 frozen at
2.7M characters; E35 strict 0.946 / 0.880 / 0.800.*

Build order: `h1_lid.py` edits → `ziplib/price.py` → `ziplib/layout.py` → `ziplib/codec.py` → `ziplib/store.py` →
`ziplib/instructions.py` → `ziplib/blueprint.py` → `ziplib/brain.py` → `brainbuilder.py` (B0 must pass before
anything else) → `ziplearn.py` → `blueprints/*.json`. Every file runs on CPU; the written weights are ordinary tensors
(`Brain.to("cuda")` is the whole GPU story).

---

## `experiments/transformers/h1_lid.py` — the substrate (five flag-edits; defaults unchanged; no copies)

- `Attn.forward(x, cache=None, start=0, coords=None)` — NEW argument `coords` (LongTensor T×2): when given and
  `pos == "rope"`, pairs `0 .. P/2−1` rotate by `coords[:, 0]·θ` (row axis) and pairs `P/2 .. P−1` by `coords[:, 1]·θ`
  (column axis) — the `rope2d` codec; `coords=None` is today's behaviour exactly. With `cache`, the bank's k/v are
  computed once and reused across passes (already supported).
- `Block(d_model, n_head, pos, res="std", n_zero=0, norm="layer")` — NEW `norm="none"` sets `n1, n2 = nn.Identity()`
  (E28 bypassed `Block` because LayerNorm rescales one-hot codes).
- `Model(..., n_head=int | list[int])`, `LoopedModel(..., n_head=int | list[int])` — a per-layer head count, so a
  layer holding one whole-window `Match` is a one-head layer with `hd = d` (what `WrittenSim.lookup = Attn(d, 1)` is).
- `LoopedModel(..., core_layers=1, boundary=None)` — NEW: the core is `core_layers` Blocks applied per pass (E28's
  core is gather + lookup); `boundary`, when given, replaces `rms_norm(x) + alpha * anchor` with `BoundaryOp`.
- `class BoundaryOp(nn.Module)` — NEW, written tensors only: `keep` (a d-vector mask: 1 keep, 0 clear), `quantise`
  (a list of (read matrix d×g, write slice) — argmax over the g pre-activations, one-hot into the slice), `commit`
  (a list of (from slice, to slice) copies after quantise), `anchor` (the slice the anchor vector is copied into, on
  cell tokens only), `halt_flag` (a dim index), `register_dims` (the dims compared between passes for convergence).
  `forward(x, anchor_vec=None, prev_regs=None) -> (x, halted: bool tensor)`. Moves `WrittenSim.rollout` lines
  178–189 (the anchor fill, the re-quantise, the clear) into a module.
- Everything else (the tasks, `make_batch`, the training loop, `routes`, `forward_embedded`, `new_caches`) stays.

## `experiments/ziplearn/ziplib/__init__.py`

- Re-exports: `price`, `Store`, `Layout`, `Codec`, `INSTRUCTIONS`, `Blueprint`, `Arch`, `Brain`. Nothing else.

## `experiments/ziplearn/ziplib/price.py` — the codes both scripts price with (no model code)

- `flag_bits(is_exception, n_right, n_wrong) -> float` — the KT rate price of one exception flag. MOVES
  `ziplearner.flag_bits`.
- `pay(s, right) -> None` — charge a structure for one prediction. MOVES `ziplearner.pay` (drops the retired "flat" branch).
- `precision_bits(n, span, sigma) -> float` — E15's 1/√n law. MOVES `ziplearner.precision_bits`.
- `entropy(p) -> float` — MOVES `arcgames.entropy`.
- `table_price(merged: dict[key, dict[y, count]], V) -> float` — the two-part code of a merged table (one parameter
  per entry, exceptions by rate). MOVES `LocalRule._price` (V becomes an argument).
- `kt_bits(ctx_keys: ndarray, nxt: ndarray, V) -> float` — the per-context KT code. MOVES `ContextLM._bits`.
- `prequential_bits(ctx, nxt, mask, V, limit) -> float` — the blended-backoff code the predictor pays (E33's
  corrected sleep price). MOVES `ContextLM._prequential` and `_chain_of` (as the helper `backoff_chain(mask)`).

## `experiments/ziplearn/ziplib/layout.py` — the residual layout

- `@dataclass Subspace(name, width: int | str, kind: "onehot" | "flag" | "scalar" | "dist")`.
- `class Layout` — `allocate(subspaces, d=None, dims=None) -> Layout`: sequential slices in declaration order (the
  `base += C` of `WrittenSim.__init__` lines 75–84), symbolic widths bound from `dims`; `d_layout`; `__getitem__(name)
  -> slice`; `flag(name) -> int`; `free(n) -> slice` (the spare-dim free-list; raises when exhausted);
  `to_json()` / `from_json()`; `error` when `d_layout > d` naming the dims needed (superposition refused).
- `channels(head_dim, needs) -> ChannelPlan(content_pairs, position_pairs)` — content dims in the LOWEST rotary pairs,
  position dims in the HIGHEST (E18 `write` lines 56 and 67 as a rule); raises naming the head on a conflict.

## `experiments/ziplearn/ziplib/codec.py` — data ↔ tokens

- `@dataclass TokenClass(name, flag, fields: dict[subspace, "input" | "coord.row" | "coord.col" | "const:<v>" | "store" | "runner"])`.
- `class Codec(layout, token_classes, pos: "onehot" | "rope2d")` —
  `encode(cls, **fields) -> Tensor[d]` (one token; `onehot` writes ROW/COL one-hots, `rope2d` writes the `bias` flag
  and leaves coordinates to `coords`); `encode_frame(frame) -> (Tensor[H·W, d], coords[H·W, 2])` (MOVES
  `WrittenSim.encode`); `encode_seq(seq) -> (Tensor[L, d], coords)` (row 0, col = position); `decode(v, tol) ->
  dict[subspace, value | None]` (argmax per one-hot subspace, `None` when not one-hot to `tol`; the decompiler's
  primitive); `nulls() -> Tensor[2, d]` (the border and zero tokens, MOVES lines 151–155 of e28.py);
  `entry(key: dict[subspace, value], value: dict, conf) -> Tensor[d]` (a store entry as a token: an all-zero subspace
  where the mask dropped a cell = a wildcard; MOVES lines 134–150); `register(name, slot) -> Tensor[d]`.
- `coords_for(tokens) -> LongTensor[T, 2]` — the coordinate list `Attn.forward(coords=)` takes; nulls, entries and
  registers get (0, 0) and are excluded from gathers by their class flags, as in E28.

## `experiments/ziplearn/ziplib/store.py` — the ONE count table and its three tensor forms

- `class Field` — `lattice(r) -> [offsets]` (every (di, dj) with |di|, |dj| ≤ r, centre excluded) and `back(k) ->
  [(0, −1) … (0, −k)]`; `windows_grid(frame, field) -> (full, masked)` (MOVES `LocalRule._rows`: BORDER padding,
  `sliding_window_view`) and `windows_seq(seq, back) -> (full, masked)` (MOVES `ContextLM._contexts`).
- `class Store(field, V)` — MOVES `LocalRule` whole: `full` (evidence), `mask`, `table`, `majority`, `stats`,
  `cost`; `observe(keys, y)`; `predict(keys) -> (y | None)` (the hash reader); `sleep(strict=True) -> (before, after,
  cells)` (MOVES `LocalRule.sleep`; on an overproduced field the dropped cells name the `Gather` heads to zero);
  `chain() -> [mask]` (E33's backoff chain, MOVES `ContextLM._chain_of`'s use); `price() -> bits` (via
  `price.table_price`); `entries() -> [(key, counts, stats)]`; `outcome_keys` for goal keys with [confirmed, refuted]
  (MOVES `GoalModel.record / live / refute`).
- Tensor forms: `as_tokens(codec) -> Tensor[N, d]` (E28's memory tokens); `as_bank(codec) -> (keys[N, d], values[N,
  d])` (the same rows split for the k/v cache — the GD-compatible form, DESIGN §21.8); `as_rows(layout, code="onehot"
  | "sparse", k=None, proj=None) -> [(key_row, threshold, value_row)]` (the MLP form; `sparse` uses
  `SparseProjection`).
- `class SparseProjection(d_in, d_out, k, seed)` — a FIXED random binary matrix: a one-hot key → a k-of-d_out code
  (genome-written, never learned). NEW.
- Offline readers (ZipLearn's apparatus only; never called from `Brain.run`): `nearest(keys, M) -> y` (matmul over
  `as_tokens`; MOVES `e35.NearestRule.predict_nearest`), `candidates(keys, k) -> [ids]` (product-key: top-k per key
  half, then the stored candidates rescored exactly — Lample et al. 2019), `loo_kernel_width(sample) -> h`
  (F2's leave-one-out choice of the kernel width).
- `consolidate(n0, eps0, price) -> [moved ids]` — an entry moves tokens → rows when settled (n ≥ n0, rate ≤ eps0) AND
  the replayed interference exceptions of the row form cost fewer bits than the token saves (DESIGN §21.7). NEW.
- `save_npz(path)` / `load_npz(path)` — keys (int16), values, counts (int32), stats — the evidence the tensors are
  regenerated from.

## `experiments/ziplearn/ziplib/instructions.py` — the instruction set (weight templates)

- `@dataclass Write(tensor: str, index: tuple, value: float)`; `@dataclass Needs(heads, hd_min, content_dims,
  position_pairs, rows)`.
- `class Instruction` — `needs(params, layout, arch) -> Needs`; `emit(params, layout, arch, slot, M) -> [Write]`
  (addresses: `blocks.{L}.attn.qkv.weight[Q|K|V + h·hd + c, slice]`, `blocks.{L}.attn.proj.weight[slice, h·hd + c]`,
  `blocks.{L}.mlp.0.weight/bias[r]`, `blocks.{L}.mlp.2.weight[:, r]`, `emb.weight`, `head.weight`); `test(params,
  layout, brain, n=256) -> Report` (argmax on target for 100% of queries, logit gap ≥ M, written subspace exact).
- `Gather(offset, src, dst, null, cls)` — one head. `onehot`: MOVES e28.py lines 97–115 (permuted coordinate query,
  the null key at 1.5·M, value/proj copy). `rope2d`: MOVES e18.py lines 56–59 per axis (bias-channel q/k with key
  phases `cos(di·θ_c), sin(di·θ_c)` on the row pairs and `dj` on the column pairs); `profile(params, arch) ->
  min gap` computes the score over every (Δi, Δj) in `max_extent` and the compiler raises the pairs until gap ≥ 1.
- `Match(q, k, v, dst, key_class, null, weights, mode)` — one head. MOVES e28.py lines 117–132 (window q/k at M,
  value = v − dst for `mode="replace"`, the class gate, the null catch); `weights[sub] = "auto"` → `(other key dims)
  + 2` (E28's `g = nO + 2`, derived). E18's induction head = `Match(q=tok, k=prev, v=tok, dst=out)` in the lowest
  pairs (MOVES e18.py lines 66–75), with the check `max_len·θ_c < 0.1`.
- `Pool(reg, key_class, src, W, dst, gate=None, sign=+1)` — one head: the register's class-flag query (+ `M·gate`),
  value `W·src` over the class, the mean into `dst`; ungated lands on `zero`. NEW.
- `Broadcast(reg, sub, dst)` — one head: every token reads the register's `sub`. NEW (E28's `x[0, n:, self.A] = 1`
  as a head).
- `Row(key: dict, threshold, value: dict)` — one MLP unit: `mlp.0.weight[r] = M·key`, `bias = −M·threshold`,
  `mlp.2.weight[:, r] = value / (M·margin)`. NEW (Geva et al.'s key-value row; the consolidated entry; the halt row).
- `Compare(a, b, flag)` — C rows `GELU(M(a_c + b_c) − 1.5M)` each weighted −2/M into `flag`, plus a bias row +1:
  `flag = 1` iff the codes differ. MOVES the gate templates of `e5.py`.
- `Branch(mixer, table: dict[flag, source])` — an `AttnRes` query with M on the flag dims (DESIGN §15). NEW.
- `Readout(sub, vocab, M_out)` — `head.weight[v, sub.start + v] = M_out`. MOVES e18.py line 77.
- Boundary parameters (no weights of their own; they fill `BoundaryOp`): `Keep(subs)`, `Clear(subs)`,
  `Quantise(read: [subs], write: sub)`, `Commit(from, to)`, `Anchor(sub, from)`, `Halt(flag)`.
- `INSTRUCTIONS: dict[str, type[Instruction]]` — the whitelist `validate` checks. Nothing outside it compiles.

## `experiments/ziplearn/ziplib/blueprint.py` — the schema (DESIGN §21.2.1)

- `@dataclass Blueprint(name, field, subspaces, tokens, circuits, stores, registers, loop, routes, tests)` with
  nested dataclasses `Circuit(name, instr, params, per, place, reads, writes, prunable)`, `StoreSpec(name, cls, key,
  value, form, capacity, nulls)`, `Register(name, slots, holds, filled_by)`, `Loop(core, tied, boundary, sequence)`,
  `Boundary(keep, clear, quantise, commit, anchor, halt)`, `Route(mixer, table)`, `Test(name, generator, oracle,
  criterion)`.
- `@dataclass Arch(d_model, n_head, head_dim, n_layer, pos, max_len, p_star=0.99, dims)` — `Arch.auto(dims)`,
  `Arch.load(path)`.
- `Blueprint.load(path)` / `save(path)` / `expand()` (the `per: "offset"` circuits over `Field.lattice(r)`) /
  `validate() -> [str]`: every `instr` in `INSTRUCTIONS`; every `reads`/`writes` a declared subspace; ≥ 1 test; no
  `params` value that is a content symbol (a colour, a character, an offset list outside `field`); no register `init`.

## `experiments/ziplearn/ziplib/brain.py` — the compiled artefact and its runtime plumbing

- `class Brain(nn.Module)` — fields: `model` (`LoopedModel`, `norm="none"`), `layout`, `codec`, `blueprint`, `arch`,
  `boundary` (`BoundaryOp`), `banks: nn.ParameterDict` (per store: keys, values), `W: nn.ParameterDict` (the `Pool`
  value matrices), `registers` (initial register tokens), `stores: dict[str, Store]` (the evidence), `M`,
  `verified=False`.
- `run(tokens, coords, anchors=None, max_passes=8) -> (tokens, passes)` — plain torch: prepend nulls + registers +
  memory tokens (or attach the banks through the caches), then per pass: core → `BoundaryOp` (anchor from `anchors[k]`
  or the ACTION register; noise field refilled from the seeded generator) → stop on the halt flag, on register
  convergence, or when the anchors run out. MOVES `WrittenSim.rollout` (the loop; the operator is now the module).
- `think(tokens, coords, n=None) -> tokens` — the sequence axis: append a thought position whose embedding is
  `BoundaryOp(final residual of the last position)` through `loop.sequence.feedback`, via `model.forward_embedded(h,
  caches, start)`; stops by the same halt rule or after `n`.
- `rows.write(key_row, threshold, value_row) -> r` — claims the next free MLP row (the headroom).
- `param_groups() -> {"structure": [...], "state": [...]}` — STRUCTURE = attention/route/boundary/readout/embedding
  tensors; STATE = `banks`, `W`, `registers`, the read-out calibration (DESIGN §21.8).
- `readout(name) -> logits` — `"action"` (the ACTION register through `head`) or `"pred"`; a sigmoid over the halt
  row's pre-activation as `halt_prob` (the PonderNet form of the same tensor).
- `anatomy() -> table` — every nonzero traced to its circuit name (uses the `Write` lists kept from compile).
- `save(dir)` / `load(dir)` — `state.safetensors` (h1_lid's tensor names), `blueprint.json`, `layout.json`,
  `<store>.npz` per store. `to(device)` inherited.

## `experiments/ziplearn/brainbuilder.py` — runs BEFORE ZipLearn

- `compile(blueprint, arch) -> Brain` — DESIGN §21.4 steps 1–7: bind dims → `Layout.allocate` → order layers by
  reads/writes (topological; pins; cycle error) → heads and channels (first-fit; one-head layers for whole-window
  `Match`; `Layout.channels`) → `M = ln((max_len − 1)·p*/(1 − p*))` → emit + conflict check + one `index_put_` per
  tensor → materialise `LoopedModel(core_layers, boundary, norm="none", n_head=[...])` → `Brain(verified=False)`.
  Generalises `WrittenSim.__init__` and `e18.write`, which are deleted.
- `verify(brain, seed=0) -> Report` — per-instruction `test` (argmax 100%, gap ≥ M, exact subspace, the mean
  `|∂loss/∂logits|` per head reported as the gradient-flow number); purity of onehot subspaces on those batches;
  the blueprint's own tests against their oracle with the tie count; writes `runs/brainbuilder/<name>.json`; sets
  `brain.verified`.
- `anatomy(brain) -> table`, `ablate(blueprint, circuit_name) -> Blueprint`.
- `main()` — `python brainbuilder.py blueprints/gridworld.json --arch auto|<file> --r 1 --pos onehot|rope2d --verify
  --out runs/brainbuilder/`.
- The blueprint experiments, one function each with its pre-registered pass/refute printed: `b0_regressions`
  (E18 tensor equality; E28 rollouts + ties; width/depth/codec invariance), `b1_generality` (with `ziplearn`),
  `b6_sequence_axis`, `b7_ablation_ladder`, `b8_compile_cost`.

## `experiments/ziplearn/ziplearn.py` — knowledge and skills into a VERIFIED brain, by counting

- `class ZipLearn(brain)` — refuses `brain.verified == False`.
- `observe(before, action, after, outcome)` — `Store.observe` per action (MOVES `Player.observe`); appends a store
  entry only when the brain's `surprise` flag fired on the pass that predicted `after` (the write gate); records goal
  keys on win/death (MOVES `GoalModel.record`).
- `observe_text(seq)` — the E33 contexts into the text store (MOVES `ContextLM.fit`'s counting).
- `sleep(strict=True, consolidate=True, n0=8, eps0=0.1) -> report` — `Store.sleep` per store (E24's masks; on text the
  prequential price, E33); zero the `Gather` heads no mask keeps (writes zeros into STRUCTURE — the one thing ZipLearn
  does to attention weights, and only ever zeros); `Store.consolidate` → `brain.rows.write`; E6's block minting →
  TASK ids (MOVES `ContinualLayer.observe`'s minting; `ContinualLayer.select` is deleted — the TASK `Broadcast` does it).
- `write() -> bits` — regenerate `brain.banks` / memory tokens / rows from the stores' evidence (`as_tokens`,
  `as_bank`, `as_rows`); returns the bits paid.
- `count_inverse(transitions)` — GCML eq. 14: `W += a (s' − s)ᵀ` into `brain.W["inverse"]`; refuted actions into
  `brain.W["nogo"]`. NEW.
- `mix(stream) -> table` — the per-context grid posterior applied to the coda read-out (MOVES `GridPosterior` from
  `research/expA_exponent_grid.py`); no learned exponents.
- `lift(tol=1e-2) -> {store: Store}` — the DECOMPILER: every bank row / MLP row decoded through `Codec.decode`; rows
  that are not one-hot to `tol` are counted as unliftable; rebuilds `Store`s with `stats` from the value's sharpness.
  NEW (DESIGN §21.8 obligation 4).
- `ledger() -> {circuit: bits}` — bits paid per store and per consolidated row.
- `play(env, budget_per_level, max_levels, seed) -> results` — the harness over `src/tasks/games` (MOVES
  `arcgames.play` minus the environment adapter, which stays in `arcgames.py`).
- `read(train_seqs, test_seq) -> (bpc_frozen, bpc_online)` — the text harness (MOVES `ContextLM.bits_per_char`,
  `bits_per_char_online`, `xz_bits_per_char` stays as the reference).
- The knowledge experiments, one function each: `g1_gd_pretrain_lands_in_store`, `g2_post_training`, `g3_round_trip`
  (the gradient-compatibility tests — the only functions in the three scripts that import an optimiser, and they
  train STATE only), `b2_executive_e6`, `b3_planner_is_the_loop`, `b4_capacity`, `b5_consolidation`.
- `main()` — `python ziplearn.py --brain runs/brainbuilder/gridworld --game LockPath --levels 0 1 --sleep --out
  runs/ziplearn/` and `--corpus "corpora/latin books" --train_chars 826605 --test_chars 20000`.

## `experiments/ziplearn/arcgames.py` — thinned to the environment adapter

- Keeps `crop_box`, the exploring `Player` (`explore`, `choose` — E27's price of ignorance; the only chooser), and the
  env loop; `LocalRule`, `ActionModel`, `GoalModel` are deleted (they ARE `Store`). `play` moves to `ziplearn.py`.

## `experiments/ziplearn/blueprints/`

- `gridworld.json` — DESIGN §21.2.2 verbatim (field r as a parameter; the 12 circuits of §21.2.2; 2 stores; 4 registers; the
  boundary; the E28 test).
- `induction.json` — E18: `Gather(offset=(0, −1), src=tok, dst=prev)` under `rope2d` + `Match(q=tok, k=prev, v=tok,
  dst=out)` + `Readout(out)`; test = E18's generator, criterion 1.0; B0 demands tensor equality with `e18.write`.

## Deleted in the same commit

`e28.py` (`WrittenSim` → `compile(gridworld)`; its `main` becomes `b0_regressions`), `e18.py` (`write` →
`compile(induction)`), `textlm.py` (`ContextLM` → `Store` + `ziplearn.read`), `e35.py`'s `NearestRule` (→
`Store.nearest`), `ziplearner.py`'s `flag_bits`/`pay`/`precision_bits` (→ `price.py`) and `ContinualLayer.select`;
`ziplearner.py`'s structure library (`Table`/`Shift`/`Affine`/`Permutation`, `TwoLayer`, `WordLibrary`) stays as the
arithmetic-task learner until a blueprint uses it, imported by `ziplearn.py`, not copied.

## Not designed (an implementer stops here and says so)

The sparse top-k reader inside `Attn` for banks above 10⁵ entries; the price of the nearest-key default; what fills
`util` when no goal is held; relational goals; the KT escape as a written read-out; pricing `p*`, the noise gain,
`max_passes`, `n0`, `eps0`; minting a new instruction; superposed subspaces.
