# Implementation plan — dynamics + subgoals EMERGE from the columns (no env-specific code)

Extends `RESEARCH.md` (R1–R8). Goal stated by Cipher: the agent must contain **no code that depends on the
specific environment**. The hand-coded residue we are deleting is (1) the feature extractor that hands the
DynamicsModel `(stepped_on, present)`, and (2) the enumerated subgoal types `fire / cover / goal / collect`
(and the `if key[0]=="cover"` dispatch). Both move INTO the cortical column's learned world model.

Two standing facts make this tractable, not a moonshot:
- the agent is already ~90% emergent (every colour/role is learned; only the *dispatch structure* is hard-coded);
- the toggle proved the **core is sound** — facing a mechanic it could not model at all (`effects={}`), the agent
  still completed by online-blocking + flailing. Robust-but-blind. The missing piece is the *self-discovered,
  augmented, non-stationary state* — not the core idea.

## The research clues (what the literature + our own R1–R8 say to do)

1. **Non-stationarity = aliasing → CLONE on exafference.** The toggle's door cell is an *aliased* observation
   (same look, different transitions). CSCG (cloned HMM, George 2021) disambiguates aliasing by making
   **context-dependent clones** of an observation, split by sequential context. R1's reframe: the trigger is
   **exafference** (a transition that fails the prediction) and binding is **one-shot** (not EM) — observe the
   surprising transition, mint a clone. The recurrence (L6 path-integrated state, R8) supplies the context.
2. **Conditional effects are ALREADY learnable — `residual.py` (R7).** Recursive-residual learns context-
   dependence `F_τ(C)` (29/29), carry, exceptions, hierarchy as ONE mechanism — *if the context is a
   coordinate*. So there is no "effects detector" to write; there is a coordinate to provide.
3. **Factors come from ACTION ORBITS — `factorize.py` (R3, Higgins/Locatello).** Disentanglement is impossible
   from statistics alone; it falls out of how states transform under action. Built; open hardness = non-factored
   action spaces (one button → several factors) and granularity (model selection).
4. **Subgoals = EIGENOPTIONS (SR eigenframe, already computed) + AFFORDANCES (effect preconditions).**
   Eigenoptions (Machado 2017) read straight off `_sr_frame`'s eigenvectors — reward-free navigation/exploration
   options that cut diffusion time even from pixels. Affordance-options = "reach the precondition of a discovered
   effect." Both valued by `reward.py`. Caveat: pure bottleneck-targeting can *hurt* — keep eigenoptions
   task-independent and let value pick.
5. **Incremental, LOCAL learning replaces the batch eigh (R2 #2, Monty).** Numenta's Monty modifies only the
   current reference frame at the current location (Hebbian, continual, huge FLOP cut). Our column's batch
   `eigh()` becomes an incremental **TD-learned SR** — dissolves the O(n³) cost AND gives smooth online revision.
6. **Exafference is the universal trigger (R1, R5).** The reafference/exafference split (efference copy) already
   separates self-motion from world-change; that residual triggers BOTH a new causal edge AND a clone.
7. **Long-range credit = learned reverse replay (R1 correction, Mattar–Daw).** The real remaining frontier; parked.

## Target architecture (where everything ends up)

```
raw frame ─► factors (objects/colours, action-orbit discovered)  ┐
                                                                  ├─► AUGMENTED STATE  s = (position × factors × context-clones)
recurrence (L6) ─► sequential context ─► clone id on exafference  ┘
        │
        ▼
   the COLUMN learns transitions over s  ── recursive_residual ──►  conditional effects  (no DynamicsModel, no hand features)
        │                                                                   │
        ▼                                                                   ▼
   SR-frame eigenoptions (navigation/explore)        +     affordances "reach an effect's precondition"   = SUBGOALS (no types)
        └────────────────────────────  reward.py values · thalamus routes · BG gates · recurrence tracks  ◄┘
```

## Recurrence is the spine (not a separate behaviour)

`SelectiveRecurrence` (`h = gate⊙A(h) + (1−gate)⊙drive`) underlies every stage. Today `A` = the L5 operator and
`h` = "where am I" (path integration). The plan keeps the mechanism and enriches what it carries:
- `h`: position → the **augmented state** (position × factors × context);
- `A`: L5 → **L5 ⊕ the learned residual effects** (a step moves me AND can flip a factor) — so `h` becomes a
  **recurrent world-model belief** whose transition IS the learned dynamics.

Consequences the stages depend on:
- `h` is the **belief that tracks a non-stationary world** — a stationary map holds all (position × factor)
  states; `h` is which one we are in now. For an **OBSERVABLE** latent (the toggle's door, visible in the frame)
  this is just an extra coordinate the recurrence carries — handled in Stage B/C. **Cloning (Stage D) is reserved
  for an ALIASED latent the frame does not reveal** (two contexts that look identical).
- `h`'s prediction error IS the **exafference** that triggers learning an effect; rolling `h` forward is how a
  subgoal is **valued without acting** (vicarious evaluation).
- The per-channel **gate** (predict vs reset) is the context-boundary detector (the R8 "a preposition resets
  context" gate).
- Caveat: `h` transitions only POSITION today, with 1-step-local learning (R8). Teaching `A` to roll the
  factor/context channels is the Stage B/D work; the 1-step locality is the long-range-credit limit.

## Staged plan — each stage gated on the regression (LockPath 100 / MultiKey 100 / Sokoban 79 + arithmetic/graph) and the two targets

- **Stage A — the augmented factored state (substrate).** Give the column a state that is a *factored coordinate
  vector* (position + object/colour factors), not position only. Bridge: seed the coordinates from the objects
  the agent already perceives (proper discovery is Stage E). *Gate:* the agent still solves LockPath/MultiKey/
  Sokoban over the augmented state.
- **Stage B — effects from the column's residual (delete the hand features + the separate DynamicsModel).** Drive
  `recursive_residual` over the augmented coordinates; the conditional effects (a colour-delta under a learned
  predicate) emerge. Effects are **symmetric** (a colour can appear ⇄ disappear), and the coordinates include the
  per-colour **presence-context**, so an OBSERVABLE reversible effect (the toggle's switch↔door) is modelled here
  — its `effects` is no longer empty. *Gate:* the same effects appear (key→door, hazard→death) AND the toggle's
  switch↔door rule is learned (both directions), with no hand-extracted features; regression holds.
- **Stage C — subgoals emerge: affordances + eigenoptions (delete the enumeration).** Replace `_subgoals`/
  `_navigate`'s `fire/cover/goal` dispatch with: affordance-options (reach a discovered effect's precondition,
  the navigation mode *read from the effect*) + eigenoptions (off `_sr_frame`). *Gate:* regression holds AND
  **collect-all passes** (its visit-to-clear is just the consume affordance — no new type). [B+C together is
  "dynamics and subgoals at the same time."]
- **Stage D — cloning on exafference for an ALIASED latent (a context the frame does NOT reveal).** When a
  transition fails prediction and the disambiguating state is *not* an observable coordinate (unlike the toggle's
  visible door), mint a **context clone** of the aliased observation (CSCG), disambiguated by the recurrence's
  context. *Gate:* a new aliased-context mechanic (built when we reach this stage) solved; regression +
  collect-all + toggle still hold.
- **Stage E — discover the factors live (delete the last hand-extraction).** Wire `factorize.py` (action-orbit
  disentanglement) into perception so the coordinates are *discovered*, not hand-fed; body/pushable/blocking/roles
  emerge. Handle non-factored actions + granularity via the MDL/residual stack. *Gate:* the agent works with NO
  hand-coded perception features + all targets.
- **Stage F — incremental SR (scale + smooth revision).** Replace batch `eigh` with an incremental TD-SR
  (Monty-style local updates). *Gate:* scales to 64×64; online revision without a batch re-consolidate.
- **Frontier — long-range credit via learned reverse replay.** Parked until a deep dependency chain needs it.

## Honest open hard parts (do not pretend these are solved)

- **One-shot online cloning.** CSCG normally learns clones by EM over many passes; we need the R1 one-shot
  exafference-triggered version. Risk: over-cloning (a clone per noise) — needs the MDL stop, same as the residual.
- **Factor discovery from non-factored actions** (R3 open) — one ARC click may touch several factors; orbit
  disentanglement degrades, the statistical/MDL route stacks on top.
- **Incremental SR** (R2 #2 open) — TD-SR that stays near-orthonormal at scale is unbuilt.
- **Credit assignment** (R1 frontier) — long chains need reverse replay.

## Success criteria

collect-all **0/3 → pass** (Stage C); toggle **44.7% → ~100%** (Stage C — observable, via the augmented dynamics
+ the recurrence; Stage D's cloning is reserved for an aliased latent); regression green at every stage; and the
end state contains **no `if`/enumeration keyed on a game-specific role** in `unified_agent.py`.

Sources: [CSCG (George 2021, Nat. Commun.)](https://www.nature.com/articles/s41467-021-22559-5) ·
[Space is a latent sequence (2022)](https://arxiv.org/abs/2212.01508) ·
[Eigenoption discovery via the SR (Machado 2017)](https://arxiv.org/abs/1710.11089) ·
[Thousand Brains Project / Monty (2024–25)](https://arxiv.org/abs/2507.04494) · plus RESEARCH.md R1–R8.
