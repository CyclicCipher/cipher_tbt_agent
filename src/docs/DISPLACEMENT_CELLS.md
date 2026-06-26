# Displacement Cells — the movement as a first-class object (design note)

Design note (2026-06-16). TBT hypothesises **displacement cells** alongside location (grid) and feature
cells. This note works out what they are *in our framework* and why they're a clean bridge across three
things we'd otherwise treat separately: **gap C's position-invariance, gap B's action representation, and
the CTKG/Merge composition line.** Status: displacement cells are *hypothesised* in TBT (Lewis, Purdy,
Ahmad & Hawkins 2019, *Locations in the Neocortex*), not as experimentally pinned as grid/place cells.

## What a displacement cell is (in our group/Fourier terms)

Location cells say *where you are*; a **displacement cell encodes the offset *between* two locations**, and
it is **invariant to absolute position** (same relative offset → same displacement, wherever you start).

In our group framing this is exact. With `z(a)=Rᵃz(0)`, the displacement from `a` to `b` is the **group
element that maps one to the other**: `z(b)=R^{(b−a)}·z(a)`, so displacement = `R^{b−a}` (Lie view: the
algebra element `(b−a)·G` — a *velocity*). Three facts make it the right object:
- It is the **movement itself, represented as a first-class thing** — not a location but the operator/vector
  relating locations. In our arithmetic it is literally the **operand `b`** ("apply `R` `b` times"); in
  navigation it is the **velocity / efference copy**.
- It is **translation-invariant**: `displacement(a+h, b+h)=R^{b−a}` for any `h`.
- In the Fourier (grid) basis it is a **phase difference** — which is why grid codes are *good at computing
  displacements* (subtract phases), and why the brain uses them for vector navigation.

So displacement cells are the **"how / relate" half** of the world model, dual to location cells ("where")
and feature cells ("what"). The triad: **location = where, feature = what, displacement = the movement
relating them.**

## Role 1 — feature/location binding (the position-invariance upgrade to gap C)

Absolute binding (`S = Σ fᵢ ⊗ z(locᵢ)`) is *not* object recognition — an object is the **relative
arrangement** of its features (a cup is a cup wherever it sits). Displacement cells supply that:

1. **Position-invariant object identity.** An object = the set of **feature-to-feature displacements**
   (`fᵢ` at `R^{dᵢⱼ}` from `fⱼ`), unchanged when the whole scene moves (`S·Rᵀ` shifts every location but
   every displacement is fixed). The set of pairwise displacements ≈ the scene's **autocorrelation**; its
   Fourier transform is the **power spectrum** — the canonical *translation-invariant* representation. So
   "recognise the object regardless of where it is" = read the displacement (power-spectrum) code, not the
   location code. (Ties to `GRID_PLACE_REFERENCE.md`: grid/place are Fourier-dual; displacement = the
   translation-invariant magnitude.)
2. **Relational queries.** "What sits at displacement `d` from the handle?" — probe by *relative* offset,
   the egocentric lookup an agent needs.
3. **Hierarchical composition = Merge.** An object of sub-objects = each sub-object's *reference frame* at a
   **displacement** from the parent's (cup = cylinder ⊕ handle-at-`d`). Displacement cells are the glue that
   **binds whole frames into composite objects** — exactly the compositional `Merge` of the CTKG line, now
   grounded in grid machinery.

## Role 2 — agentic behavior (gap B: the action/plan representation)

For an agent, **the displacement *is* the plan and the action**:

1. **Goal-directed navigation = compute a displacement.** Displacement from *here* to *the goal* is the
   movement to execute (`d = goal ⊖ current`, a phase subtraction in the grid basis) — Banino-style
   vector navigation; the displacement cell holds "the vector to the reward."
2. **Forward model / planning.** Apply a candidate displacement to the bound scene → predict the next scene
   (`S → S·(R^d)ᵀ`): simulate "if I move by `d`, what will I sense?" before acting. Planning = search over
   displacements.
3. **Subgoal composition.** Multi-step plans chain displacements (`R^{d₁}R^{d₂}=R^{d₁+d₂}`); hierarchical
   plans are nested displacements — the same Merge as in binding, over actions.
4. **Transfer / sample-efficiency.** Displacements are position-invariant, so a relational policy ("move by
   `d` toward feature X for reward") **generalises across all absolute positions** — the `PURE_MATH §5`
   few-shot/transport property, now for *behavior*: learn the relation once, not per-location.
5. **Efference copy.** The commanded movement *is* a displacement; the agent knows its own displacement
   because it issued the action — how path integration stays grounded (the active-`b` of gap B).

## Where it plugs into the build

A **displacement code** (the represented movement — `b`/velocity/Lie-element, computed as grid-basis phase
differences) sits between the gaps:
- **gap C+**: upgrades binding from *absolute-location* to *relative-arrangement* (position-invariant)
  object codes.
- **gap B**: *is* the agent's action/plan representation — the operator we already apply as `R^d`, now made
  a first-class, comparable, composable vector.
- **CTKG/Merge**: the relational glue for hierarchical composition of both objects and plans.

**The agent loop it implies:** perceive (bind features to locations) → compute displacement-to-goal →
execute the movement → predict (apply displacement to the scene) → repeat. Location + feature +
displacement is the full TBT triad; we already have location (gaps A/B) and feature-binding (gap C
foundation), and displacement is the small, principled addition that makes both *relational* and
*actionable*.

## Connections in this repo
- `GRID_PLACE_REFERENCE.md` — grid/place Fourier duality; displacement = the translation-invariant
  (power-spectrum / autocorrelation) view.
- `PURE_MATH_FOR_ML.md §5` — transport/few-shot ⟺ the position-invariance of displacement policies.
- `BLUEPRINT.md` §H — gaps B (agency) and C (binding); this note is the bridge between them.
- Prior project notes: cortical-column "displacement layer" (`experiments/ManuallyCodedTBT`),
  vector-navigation-not-autoregressive, and the CTKG/Merge composition line.
