# Blueprint — the grid-cell format as the model's design basis

This consolidates the learnings from the few-shot-arithmetic thread into one design and proposes the
**redesigned model**. Theory backing: `PURE_MATH_FOR_ML.md` (esp. §8–§10); experiment harness:
`FEW_SHOT_ARITHMETIC.md`; code: `tasks/{vocab,number_line,arithmetic}.py`, `train_numberline.py`,
`train_arith.py`, `binding.py`.

## Status (the gap roadmap §H, tracked)

- **Gap A · geometry — ✅ COMPLETE** (Findings 4–7): zero-shot add/sub/mul/div; 2-torus; n-D; multi-scale
  (exponential range + error-correction); hexagonal/conformal isometry. The full grid-cell geometry.
- **Gap C · feature⊗location binding — ✅ COMPLETE** (Findings 8–9): VSA outer-product binding; lossless
  movement (`S·Rᵀ`); grid→place orthogonalisation closes capacity to the ceiling. **Two coupled codes** —
  grid (movement) + place (memory), Fourier-dual (`GRID_PLACE_REFERENCE.md`).
- **Gap B · agency — NEXT**: active self-movement (efference copy), the recurrent navigator + learned
  symbol→movement binding, goals/value (vector-to-goal), the *perceive→bind→plan→act→predict* loop.
- **Gap D · frames & online discovery — after B (the hard part)**: object-centric frames + selection;
  online discovery of the world's generators from interaction; sensory anchoring / drift correction.
- **Bridge — displacement cells** (`DISPLACEMENT_CELLS.md`): the movement `R^d` as a first-class object;
  spans C→B (position-invariant object identity + the agent's action/plan representation).

> ## ★ Headline result (Finding 4, 2026-06-16): zero-shot arithmetic by path integration
> A reference frame + its movement generator `R`, **discovered from succ/pred/compare alone — addition
> never seen**, compute addition at **100% (289/289), zero-shot, parameter-free**, via `Rᵇ·z(a) = z(a+b)`.
> Robust across seeds *and* across frame quality (even the single-frequency frame: `R` composes
> regardless). The original "few-shot" goal was the wrong frame: with the right structure it's
> **zero-shot**; the few labels were only ever for the trivial *binding* "symbol `+b` ⟼ apply `R` `b`
> times." **Design lesson:** an operation must be computed by **structurally applying the generator**
> (a *navigator*: start at a location, move, read the landing) — **not** relearned by a generic readout.
> A generic transformer given the same frame as init reached only ~0.14 (it doesn't rediscover
> composition); the navigator gets 1.00. This is `project_vector_navigation_not_autoregressive` realised.
>
> ## ★ Finding 5: the mechanism is operation-agnostic (multiplication + division, zero-shot)
> Fed only the *multiplicative* step `×g` (a primitive root), the **same** discovery machinery discovers
> the **log-scale frame** (values ordered by exponent, `1,3,9,10,…`) and gets **multiplication 256/256 =
> 100%** and **division 256/256 = 100%, zero-shot** (`tasks/multiplicative.py`). A *different* frame for a
> *different* operation, with a *non-trivial* binding (the discrete log `log_g b`), still composes exactly.
> **Caveat:** mult-mod-prime is still a *cyclic* group (isomorphic to addition on exponents); the real
> frontier is **non-cyclic / 2-D** structure (e.g. mult mod a *composite* — `ℤ/8*` ≅ ℤ/2×ℤ/2 needs two
> generators / a 2-torus). That's the §G #4 direction and the honest next stress before the agent.
>
> ## ★ Finding 8: gap C foundation — feature⊗location binding, with lossless movement (`binding.py`)
> VSA outer-product binding `S = Σ fᵢ ⊗ z(locᵢ)`: query-what (`S·z`), query-where (`Sᵀ·f`), and **move the
> whole scene** (`S·Rᵀ`) all work. On **idealised orthonormal keys: 100% at every K≤8** (mechanism + move
> validated). **Movement is lossless** — for the grid code, `move-what` *exactly equals* `what` at every K,
> so `S·Rᵀ` shifts every bound object coherently with **zero added error** (a bound scene that survives
> self-motion = TBT sensorimotor prediction over objects). **Binding capacity scales with key orthogonality:**
> single ring (2 eff-dim) → 0.77/0.60 at K=8; multi-scale (8 eff-dim) → 0.87/0.83; orthonormal → 1.00. So
> the **multiple-modules** machinery that gave exponential *range* (F7) also gives binding *capacity* — and
> the route to full capacity is the brain's **grid→place orthogonalisation** (grid for path-integration,
> place for memory).
>
> ## ★ Finding 9: grid→place orthogonalisation closes the binding gap (analytic, validated)
> Per the Fourier duality (`GRID_PLACE_REFERENCE.md`): place = each location's **similarity profile across
> the grid code** (inverse-Fourier synthesis, a circulant peaked at the location) + **top-k sparsification**
> (the DG nonlinearity); the grid generator `R` becomes a **cyclic-shift permutation** in place space (the
> regular rep's action), and top-k commutes with it so **movement is preserved by construction**
> (`binding.py:place_from_grid`). Result: place top-k≤3 → **what/where/move = 1.00 at every K≤8** (vs the raw
> grid code's 0.87/0.83, vs the orthonormal ceiling 1.00). Sparser k → more orthogonal → higher capacity
> (top-k=10 drifts back toward the grid) = the Babadi-Sompolinsky tradeoff / DG sparse regime. So the
> closed-form grid→place transform **closes the gap to the ceiling** — no learning needed; the group-rep
> theory delivered. **Two coupled codes confirmed:** grid (movement/path-integration) + place (binding/memory),
> Fourier-dual. Next: bound scenes as *agent state*, and the displacement-cell upgrade (`DISPLACEMENT_CELLS.md`).
>
> ## ★ Finding 6: the 2-torus works — zero-shot 2-D navigation (multi-generator)
> Two generators `Rₓ,R_y` discovered from x-/y-step experience on a 5×5 torus (`tasks/torus2d.py`): they
> **learn to commute** (path independence = 2-D reference-frame coherence), and `Rₓᵃ·R_yᵇ·z(x,y)` lands on
> `z(x+a,y+b)` for **all 625 (start, vector) pairs = 100% zero-shot**. This extends the kernel from cyclic
> groups to **finite abelian groups** (products of cyclics = a multi-torus). **Lesson:** multi-generator
> frames are *more collapse-prone* — at default anti-collapse the rep collapsed (equivariance satisfied
> *vacuously* via `z→const, R≈I`, nav = chance); needed `w_sig≈5` (~5–50×) to hold the generators apart.
> **Anti-collapse must scale with the number of generators**; collapse remains the main failure mode.
>
> ## ★ Finding 7: multi-scale modules — exponential range (the grid code's superpower)
> Four small rings (periods 13,15,16,17), each composing 100%, **jointly code 53,040 positions** and add
> over the *entire* range **100% zero-shot**, combined by the **Chinese Remainder Theorem** — exactly how
> grid modules combine. K small modules → `lcm(periods)` unique positions: exponential capacity from small,
> robust components, the reason the brain uses multiple scales. **Caveat (separable engineering, not a flaw):**
> very small rings (m≤9) train unstably — a dim-64 transformer is wildly overparameterised for a ~5-point
> ring, so the learned `R` mixes off-ring dimensions and `Rʲ` drifts (decode → chance). Proven range
> (13–17) is flawless; the fix is per-ring capacity control (smaller dim / normalisation), not the idea.

## A · Learnings → principles (what we established, in order)

1. **A positional encoding is not a reference frame** (§9). PoPE says *where a token arrived in the
   input*; the number line must be the **learned geometry of the value representations**. `coord` does
   only sequence position (`value_coord=False`).
2. **Relational cross-entropy buys separability, not geometry** (§9, Finding 1). succ/pred/compare under
   CE are solved and even generalise (~0.63 held-out) but grow **no clean metric space** — a ring is
   *sufficient* to classify, not *necessary*, so it doesn't appear. Structure must be made an objective.
3. **Equivariance forces the space** (§9, Finding 2). Require the `+1` movement to act as a **single
   shared transform** `z(a+1)=R z(a)` with the **orbit closing** + SIGReg → a ring emerges (corr 0→0.68+).
4. **The transform must be a rotation** (§10). `R = exp(G−Gᵀ)`: (a) removes the over-weighting collapse
   (a rotation can't shrink `z`), (b) makes `R` a **Lie generator** so `R^t` gives **fractional/continuous**
   movements, (c) **is conformal isometry** — metric preservation.
5. **Single-frequency purity is *collapse*, not success** (§9 caveat + the neural-collapse literature).
   Over-weighting equivariance drives all energy into the **fundamental** frequency (`ring_var`→1); that
   2-D circle lacks the harmonics any nonlinear readout needs, so downstream generalisation drops
   (comparison 0.85→0.56). Neural collapse provably trades train-fit for transfer.
6. **The fix is multiplicity of scale/frequency.** Grid cells are *multi-module* precisely because one
   scale is only locally unique; grokked modular addition uses **3–8 frequencies**. The target is a
   **multi-frequency / high-effective-dimensional** code, not a pure circle.
7. **The movement `b` is given here (passive); the real test is the active agent** (§10). Long-term:
   make this a `WorldModelAgent`-style agent where `b` is a policy action known via efference copy
   (= path integration). `Rᵇ = exp(b·G)` is the same object as action-conditioning and a learned-phase PoPE.

## B · The blueprint (grid-cell format, three pillars)

The mammalian grid code is the convergent optimum of three constraints (`PURE_MATH §8`), and each is a
design choice here:

| brain | constraint | our model |
|---|---|---|
| hexagonal lattice / conformal isometry (3 plane waves; **6-cell torus**) | metric preservation | movement = **rotation** `R=exp(G)` |
| toroidal continuous attractor | periodicity / indefinite integration | **closing orbit** `Rᵐ≈I` |
| multiple modules at geometric scales (~3/2) | unique long range + error correction | **multi-frequency** code |

2-D hexagonal because physical space is 2-D; a 1-D quantity (number line) is the **1-torus (circle)**;
abstract spaces reuse the machinery (Constantinescu 6-fold fMRI).

## C · The redesigned model — a *discovered multi-scale toroidal reference frame*

The current model has pillars 1–2 (rotation `R`, closing orbit) but **fails pillar 3** (single-frequency
collapse). The redesign makes the reference frame **multi-scale** by construction/pressure.

**Representation.** A token's "where" is a point on a learned **multi-scale torus** Tᴷ (K circles). The
value code `z` carries K rotational components (modules), each a phase; a number is its **vector of
phases across scales** — the Fourier/grid code.

**Movement (generator).** The unit `+1` is a single metric-preserving rotation, **block-diagonal across
modules**: `R = ⊕ₖ Rot(θₖ)`, `R = exp(G)`, `G` skew-symmetric. **Closure** `Rᵐ=I ⟹ θₖ = 2π·nₖ/m`
(integer harmonics = the ℤ/m characters = the grid scales). Continuous movement: `R^t = exp(t·G)`.

**Objective (discovery).**
- *Sensorimotor*: predict succ/pred (the movement's effect).
- *Equivariance / conformal isometry*: `z(a+1) = R z(a)`, one shared rotation (the movement is the same
  metric-preserving transform everywhere).
- *Closure*: the orbit closes after `m` (→ torus).
- *Multi-scale* **(the new pillar — revised, see Finding 3)**: the code should use **a few distinct
  frequencies** (grid scales), not one. *How* to get there is open. A direct spectral-spread regulariser
  (Herfindahl on the power spectrum) was tried and **demoted to a diagnostic** — it overshoots to a
  *uniform* spectrum, which destroys the ordered ring (Finding 3). Current best understanding: multi-scale
  is **task-driven** — keep equivariance *moderate* (over-weighting causes the single-frequency collapse)
  and let the **task demand** recruit frequencies (Nanda's 3–8 emerged from *addition*, not a prior).
- *Metric*: comparison/distance — both a downstream check and a task that *needs* multiple frequencies.

**Readout — a *navigator*, not a generic head (revised by Finding 4).** The operation is computed by
**structurally applying the generator**: `a∘b` = start at `z(a)`, apply the movement `b` times
(`Rᵇ·z(a)`), decode the landing by nearest value code. This is *zero-shot* once the frame+generator are
discovered (Finding 4: 100% on addition). A generic transformer readout given the same frame does **not**
rediscover this composition (~0.14) — so the navigator must be **structural**, not a free MLP/attention
head. The only learnable part is the **binding** (the symbol/operand `b` ⟼ "apply `R` `b` times"), which
is trivial. Open: how the binding is learned, and how a *single* readout selects which generator to apply.

**Probe (replaces `ring_var`).** Effective number of frequencies / participation ratio of the orbit's
power spectrum **+** circular-distance generalisation. Target: *multi-frequency and metric*, not pure circle.

### The one open fork (needs a decision)

How explicit to make the multi-scale structure:
- **(1) Explicit modules.** Fix K 2-D rotation blocks, learn their frequencies `θₖ`. Interpretable,
  multi-frequency *by construction*, a literal grid code. More baked-in.
- **(2) Emergent spectrum.** One rotation `R=exp(G)` on the full width (modules = `G`'s eigenplanes),
  plus a **spectral-diversity** objective so multiple frequencies get used; K and the `θₖ` *emerge*.
  More bitter-lesson (don't hand-pick scales).

**Recommendation: (2)** — keep `R=exp(G)` general, add the multi-frequency anti-collapse term so it can't
collapse to one frequency, and let the scales emerge; use (1)'s explicit blocks only as an interpretable
diagnostic. Rationale: the bitter lesson (no hand-set module count/scales) + grid scales are themselves
emergent from pattern formation in the brain.

## D · Build plan (increments)

1. **Spectral probe.** `spectrum_probe` = participation ratio (`n_eff_freq`) of the value code's power
   spectrum. ✅ Confirmed the §9 diagnosis: w_equiv=1 → n_eff≈2 (multi), w_equiv=3 → n_eff≈1 (collapse).
2. **Multi-frequency anti-collapse term.** ✅ Built (`_spectral_concentration`) and tested — **demoted to
   a diagnostic** (overshoots to uniform; Finding 3). Multi-scale instead handled by moderate equivariance
   + task demand. *Options kept open* — a "few-but-not-uniform" objective (soft `n_eff` floor) is unexplored.
3. **Phase-1 transfer** (`train_arith.py`) — ✅ **DONE → Finding 4.** Arms: A0 random+CE (chance),
   (b) frame+CE (chance — CE can't use the ring), (b-coh) frame+coherence (~0.14, only arm above chance
   from a generic head; coherence *needs* the frame — random+coherence = chance), and the **structural
   `Rᵇ` navigator = 100%, zero-shot** (the answer). Frame is *preserved/enriched* by phase-1 finetuning
   (hid_dc 0.57→0.58, n_eff 1.97→2.85), so the weak generic-readout number is a *readout* limitation,
   not frame erosion.
4. **Next — the navigator + binding.** Build the structural navigator into the model and learn the
   symbol→movement binding (the only learnable part; expected trivially few-shot). Then: **multiplication**
   needs a *different* frame (log scale, where `×`=`+`; the exp/ln isomorphism, PURE_MATH §2–3) — tests
   generality. Then the **active agent** (`b` a self-chosen action via efference copy; §10/§7).

## G · Roadmap (all worth doing — saved 2026-06-16)

Post-Finding-4 directions, none discarded:
1. **Navigator + binding.** Make the structural `Rᵇ` computation the model's actual readout; learn the
   trivial symbol→movement binding. Consolidates the win into an end-to-end model.
2. **Multiplication via a log-scale frame.** ✅ **DONE → Finding 5: 100% zero-shot `×` and `÷`** (same
   machinery, `×g` step, discovered log frame, discrete-log binding). Next stress: **mult mod a composite**
   (non-cyclic → needs a 2-D/multi-generator frame; merges with #4) — the real generality frontier.
3. **Active agent (efference copy).** Movement `b` becomes a self-chosen action known via efference copy
   (= path integration); bridge to the `WorldModelAgent`. The setting these ideas are really for.
4. **2-D / hexagonal extension.** From the 1-D circle to a 2-D hexagonal toroidal frame with two
   generators — the literal grid-cell geometry; tests scaling past one dimension.

## H · Gap to the full grid system (the agent build order)

We have the **kernel**: a 1-D ring (one cyclic group), one generator, one frame, *passive* movement,
discovered *offline*, with a *hand-coded* readout — and it gives zero-shot, transportable operations
(add/sub/mul/div). The mammalian grid system has all of this *plus* the following. (Inductive biases
aren't the problem — the brain is full of them; the bitter lesson forbids domain-specific *rules*, not
general structural priors like the grid code. These gaps are the experiment program.)

**A · Geometry (shape of the frame)**
- 1-torus → **2-torus** ✅ (Finding 6: 625/625 zero-shot 2-D nav) → **n-D / abstract** (movements are vectors).
- single scale → **multiple scales/modules** (~5–10, ratio ~3/2): unique long range + error correction.
- (2-D) → **hexagonal / conformal isometry** (3 plane waves) — the isotropic metric-preserving lattice.

**B · Agency (how movement happens)**
- passive given `b` → **active self-motion + efference copy** (the move is a *commanded action*, known
  because issued) — the agent core.
- hand-coded `Rᵇ` → **trained recurrent navigator** (apply-operator-until-halt; a decision each step).
- no goal → **vector navigation to an inferred goal** (Banino 2018); ARC must *infer + pursue* a win condition.

**C · Content (what is represented)**
- location-only → **feature ⊗ location binding** (TBT: *what* bound to *where*) — to represent a board,
  not just a position.

**D · Frames & discovery (the hard part for ARC)**
- one frame → **many frames + selection + hierarchical composition** (the Merge/CTKG line; the "many
  frames" problem we parked).
- offline curated successor → **online discovery of the world's generators from raw interaction**.
- free-floating → **sensory anchoring / drift correction** (place cells re-anchor the grid to percepts).

**For the ARC-AGI-3 agent specifically**, the binding gaps are **D + B + C** (discover the frame online,
move by deciding actions, bind features to locations, run/select multiple frames, infer a goal) — most
of an agent. The geometry gaps (A) are about robustness at scale. Each piece has a concrete target, and
several map onto existing repo work (`WorldModelAgent` for active goals; CTKG/Merge for multi-frame
composition).

**Chosen ordering (owner, 2026-06-16): finish A → then C → then B/D.**
- **A · geometry — ✅ COMPLETE.** All pieces validated:
  - 2-torus ✅ (F6); **multi-scale modules** ✅ (F7: 53,040-range zero-shot via CRT — exponential capacity);
  - **multi-scale error-correction** ✅ (25% per-module noise: naive CRT → 0.30, error-corrected → 0.98 —
    Sreenivasan & Fiete's analog error-correcting code; redundant modules outvote noise);
  - **n-D** ✅ (3-torus 4×4×4: 3-D navigation 4096/4096 = 100%, 3 commuting generators; `tasks/torus_nd.py`)
    — multi-attribute state;
  - **hexagonal / conformal isometry** ✅ (nav 100%, closure `‖R0R1R2−I‖≈0.05`, step-mag isotropy ratio
    1.002 with the conformal term, ~1.03 even without — isotropy is largely free from rotation generators +
    symmetric directions; `tasks/hextorus.py`, `train_hex`). "Hexagons all the way down."
  Functions recap: n-D = multi-D state; multi-scale = exponential range; error-correction = robustness;
  hexagonal = isotropic metric. **The full grid-cell geometry is reproduced.**

### Collapse control (the `w_sig` question — *tested, resolved*)
Collapse (Findings 6/7) has **no clean closed-form `w_sig`** — it's a *basin* problem (the equivariance
term has both a correct ring minimum and a trivial low-rank/`R≈I` one). We proposed `eff_rank(cov z) ≥ 2G`
as the criterion and a hard VICReg-style anti-collapse as the fix, then **ran the head-to-head ladder
(SIGReg vs VICReg × G=2,3) to decide on evidence.** Result:
- **VICReg is *not* better.** Same compute (58 vs 57 ms/step — the anti-collapse term is negligible vs the
  transformer forwards), **same tuning need** (`vicreg@1` collapsed, `vicreg@5` works — exactly like SIGReg),
  identical quality (nav 1.0, rank 6.0 at w=5). The "parameter-free" promise didn't hold → **keep SIGReg**.
- **`eff_rank ≥ 2G` is necessary but NOT sufficient.** `sigreg@0.1` had *high* rank (7.6, 14.3) yet failed
  (spread but no ring). The real success signature is **rank = *exactly* 2G AND nav = 1.0** (G rings live
  in precisely 2G dims; below = missing rings, above-with-bad-nav = unstructured). nav is the true metric.
- **The standard, settled:** **SIGReg, `w_sig = 5`** for multi-generator discovery (20 also works, more
  margin). The Findings 6–8 tuning was a *one-time* fact — single self-closing ring stable at 0.1,
  multi-generator needs ~5 — not a per-G fragility (5 held across G=2,3). Not a loose end.
- *(Untested: w=5 at G≥4 — slow, and the G=2,3 evidence + mechanism suggest it holds. VICReg kept in
  `train_numberline.vicreg` as an option, not the default.)*
- **C · feature⊗location binding — ✅ (Findings 8 + 9):** VSA outer-product binding with lossless movement
  (`S·Rᵀ`); **grid→place orthogonalisation** (inverse-DFT similarity + top-k; `binding.py:place_from_grid`)
  **closes the capacity gap to the ceiling** (top-k≤3 → 1.00 binding, movement-preserving). Two coupled
  codes (grid=movement, place=memory), Fourier-dual. *Next: bound scenes as agent state; displacement-cell
  upgrade for position-invariance (`DISPLACEMENT_CELLS.md`).*
- **B · agency** and **D · frames/online-discovery** — built on top of A+C.
- **Bridge concept — displacement cells** (`DISPLACEMENT_CELLS.md`): the *movement* `R^d` as a first-class
  represented object (= the operand/velocity; a grid-basis phase difference; translation-invariant). It
  upgrades gap C from absolute-location to *relative-arrangement* (position-invariant) object codes, **is**
  gap B's action/plan representation (displacement-to-goal = the plan; apply it = forward model), and is the
  relational glue for CTKG/Merge composition. Triad: location (where) + feature (what) + displacement (how).

## F · Open questions (new research — hold conclusions loosely)

Nobody has the answers here yet; keep options open.
- **Finding 3.** A spectral-spread regulariser can't distinguish "a few scales" (grid) from "uniform"
  (noise); the latter maximises `n_eff` but destroys the ring (`hid_dc`→0.01). `cmp_ho` (task readout)
  and `hid_dc` (geometric ring) **diverge** — readout wants many modes, the clean ring wants few.
- Is "a few frequencies" best obtained by task demand, weight-decay/efficiency, or a *shaped* spectral
  prior (floor, not uniform)? Untested.
- Does few-shot **addition** itself recruit more frequencies (à la grokking)? Phase-1 will tell us.
- Which frame property actually predicts few-shot transfer — `hid_dc` (ring), `n_eff` (scales), or the
  rotation `R`'s cleanliness? Phase-1 measures this.

## E · What stays / what changes

- **Stays:** `FixedDepthTransformer` (PoPE) as readout substrate; `coord = position` only; shared vocab;
  SIGReg; rotation `R = exp(G)`; closing-orbit equivariance; the discovery→phase-1 two-phase structure.
- **Changes:** add the **multi-scale/multi-frequency** pillar (objective + probe); stop treating
  `ring_var`→1 as good; pick the operating point by *spectral diversity + transfer*, not ring purity.
- **Deferred:** explicit learned-phase→PoPE merge (the "many frames" frame-selection problem, §9); the
  2-D hexagonal version (we use the 1-D circle / 1-torus for the number line).
