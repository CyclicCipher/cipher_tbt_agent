# `blueprints/` — the genomes BrainBuilder compiles

A blueprint is one JSON document following the schema of `DESIGN.md` §21.2.1 field by field: `name`, `field`,
`subspaces`, `tokens`, `circuits`, `stores`, `registers`, `loop`, `routes`, `tests`. It holds repeatable top-down
instructions for building a brain — which circuits exist, where they sit, how they connect — and never a weight that
encodes a fact. `brainbuilder.compile(Blueprint.load(path), arch)` turns it into a `Brain`; `ziplearn.ZipLearn` then
fills the brain's stores by counting. Widths and constants may be symbols bound from `Arch.dims` (`H`, `W`, `C`, `V`,
`nA`, `L`); the size numbers (`d_model`, heads, layers, `pos`, `max_len`, `p_star`, `dims`) are the `Arch`, not the
blueprint, so one file compiles at any width.

## `gridworld.json` — the E28 brain (DESIGN §21.2.2)

The worked example, as §21.2.2 gives it. The born receptive field is `field.r` (1 for the E28 regression, `--r 4`
for the B1 generality test); the compiler expands the one `per: "offset"` circuit `gather[o]` into one head per
lattice offset (8 at r = 1, 80 at r = 4) and the sleep pass later zeroes the heads no mask keeps, so `gather[o]` is
the only `prunable` circuit. Twelve circuits are declared: `gather[o]`, `lookup` (the E28 window match into `pred`,
with the action's weight derived as `"auto"`), `goal_read`, `bcast_act`, `bcast_goal`, `surprise` (the Compare rows),
`inverse` and `nogo` (the GCML Go/NoGo pools into `util`), `task_write` (the surprise-gated pool into TASK),
`halt_row`, `read_act`, `read_pred`. Two stores (`rules`: window + action → `out`; `goals`: window → `goal_met`),
four registers (`ACTION`, `GOAL`, `TASK`, `STEP`), the boundary (keep / clear / two quantisations / the `pred → colour`
commit / the action anchor / the halt flag) and the E28 test `rollout_vs_table` (criterion: written = planner on
≥ 0.98 of the all-known plans; E28 measured 1.000). With dims `H 8, W 11, C 17, V 16, nA 4` (the crop boxes of
LockPath levels 0–2, 16 colours + BORDER, 4 actions) the 13 subspace groups of §21.2.2 allocate `d_layout = 251` at
r = 1 and 1,475 at r = 4, and the memory budget `2 nulls + 192 + 64 entries + 88 cells + 4 registers = 350 = max_len`
gives `M = 10.45` at `p* = 0.99`, the numbers §21.2.2 states.

## `induction.json` — the E18 circuit

E18's two written heads as a blueprint: `prev_token = Gather(offset=(0, −1), src=tok, dst=prev)` — under `rope2d`
the sequence is the row-0 case, so the offset is one position back along the column axis — then
`induction = Match(q=tok, k=prev, v=tok, dst=out, mode=add)` in the lowest rotary pairs, then `Readout(out, V,
M_out=10.0)`. One token class (`tok`, its constant `bias` channel doubling as the class flag), no stores, no
registers, a no-op boundary (one pass: nothing is anchored and there is no register to converge). Dims `C 9, V 8`:
the nine token identities (8 digits + BOS, as `C` = 16 colours + BORDER in gridworld) and the 8-way read-out. Test
`induction`: E18's generator (256 patterns of 8 distinct tokens repeated twice, seed 0) scored on the second copy
against the next token, criterion 1.0 (E18 measured 1.000). B0 additionally compares the compiled state_dict with
`e18.write`'s tensors.

## Conventions the compiler reads (chosen where §21 is silent; the full list is in the builder's report)

- A subspace or circuit name containing `[o]` is a template over the field's offsets (`nbr[o]`, `gather[o]`):
  `Blueprint.expand` instantiates one per `Field.lattice(r)` offset, named `nbr[di,dj]` (as `ziplib/_check_codec.py`
  names them); the `per: "offset"` circuit's `params.offset = "o"` is the bound offset. `x[*]` names every instance
  of `x` and appears in token fields, store keys, `Match` q/k lists, circuit reads/writes and the boundary's `clear`;
  `expand` resolves it in all of them before the codec sees a class (the codec takes concrete names only).
- The nine flag dims of §21.2.2's `flags(9)` are declared as nine width-1 subspaces of kind `flag`, in the same
  order and at the same dims (242–250 at r = 1), because token classes, the boundary's `halt`, `Compare`, `Row` and
  routes name flags individually and `Layout.flag(name)` resolves them directly; `flags.halt` in the DESIGN text is
  the subspace `halt` here.
- `boundary.keep` lists every subspace that survives the boundary on the tokens it operates on (cells and
  registers), `clear` the scratch it zeroes (`nbr[*]`, `util`); together they partition the layout, so the
  `BoundaryOp.keep` mask is the same whether the compiler builds it from `keep` or from the complement of `clear`.
  E28's boundary cleared only the gathered neighbours and re-quantised the colour; `pred` is kept so the next pass's
  `surprise` compares the last prediction with the colour the new frame carries.
- A token class's coordinate fields (`row: coord.row`, `col: coord.col`) name subspaces under `onehot` (gridworld's
  `row(H)`, `col(W)`) and name no subspace under `rope2d`, where the position lives in `coords` (induction's `tok`);
  the codec finds the positioned class by its `coord.row` field.
- `const:<v>` is a symbol INDEX, negative from the end: the border token's colour is `const:-1`, the last dim of the
  `C`-wide colour code, which is BORDER (`WrittenSim.BORDER = V`) — never a literal colour index.
- `tests[].generator` / `tests[].oracle` are dotted names of callables, optionally followed by a parenthesised list
  of literal keyword arguments (`ast.literal_eval` per value) carrying the test's protocol numbers; the seed is
  `verify(brain, seed)`'s.
- `Pool.src` may be a difference of two subspaces written `"a - b"` (§21.2.2's `goal - colour`); `Pool.W = null` is
  the identity value matrix (a named `W` is a slot in `Brain.W` that ZipLearn counts).

## The rule: no `params` value is a content symbol

A circuit's `params` may name subspaces, token classes, registers, value-matrix slots (`W`), a mode, `"auto"`, the
bound offset `"o"`, and the operation constants of its instruction (`Row.threshold`, `Readout.M_out`, `Pool.sign`, a
flag value of 1, a single offset that lies inside `field`). It may never hold a colour index or colour name, a
character, a word, a hand-picked offset list, or any value chosen because a particular game or corpus uses it
— an offset list is a FIELD, declared once under `field`; goals are filled by the goal model, never by a register
`init`. `Blueprint.validate` refuses these mechanically (the E29 guard, DESIGN §21.2.1); a blueprint that needs one
is asking for a new instruction, and a new kind is a program (§19), not a parameter.
