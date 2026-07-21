# notes/touch_and_body_design.md — the TOUCH modality and the agent's BODY (design of record)

Status: DESIGN (2026-07-21). Motivated by the Gap-1 soundness check (the "contact-dynamics" thread): a direction-general
push requires a *contact* condition, and contact is not a geometric predicate — in TBT it is an **active-touch sensorimotor
event**, which the agent cannot have without a **body** and a **touch sense**. This note is the mechanism-of-record for adding
that modality. It supersedes the hard-coded contact/solidity in `hippocampus/replay.py` and the push crutches (#3/#4/#5/#9 in
commit `bf04c9e`).

## 1. Why — the real gap

The agent is a **disembodied eye**: it has vision (`perceive.segment` = a retina peripheral) and a *point* self (the
discovered controllable root, one cell). But pushing, solidity, and occlusion-by-contact are all **somatosensory** phenomena.
Every push fix so far (the `snap`, the rigid-body coupling in `replay.py`, the inferred landing) is a stand-in for a **touch
signal the agent does not have**. You cannot cleanly model contact in a system with no organ that senses contact. So the fix
is not another predicate — it is a modality.

## 2. What contact IS (TBT)

- A **touch column** attends to a **receptive field on the body surface** — the somatotopic map / homunculus is exactly the
  arrangement of these fields over the skin. It runs the *same* sensorimotor reference-frame algorithm a vision column runs
  (feature-at-location), but its sensor is skin (Numenta's canonical example: a fingertip modelling a coffee cup).
- **Contact = the body-surface frame and an object-surface frame coincide at a sensed point.** The touch column reports "a
  surface is *here* on my body," and because the body's pose is known (nav L6a + efference), that body-location maps to a
  world/object location. Two reference frames meet.
- It is **active**: touch ⊕ efference ⊕ the sensed OUTCOME. The agent presses (efference) into the felt object and it either
  **yields** (moves — felt give), **resists** (no motion — felt force), or there is **no touch** (it passes through). That
  yield / resist / pass trichotomy *is* the push / block / passable distinction — **felt, then LEARNED per object**. Solidity
  is felt resistance, not a universal law (the correction that killed the "all matter is solid" prior).

## 3. The pieces

1. **Body** — the self's occupied cells (a rigid set) in a **BODY FRAME** anchored to the nav L6a pose; it moves with the self
   via the *same operator* (reuse, not new geometry). The **surface** is the outward faces (faces of body cells whose neighbour
   is not a body cell). Single-cell now (4 faces in 2-D); MULTI-CELL is the real-ARC future — and it is the same machinery that
   will segment multi-cell OBJECTS, so body and object embodiment share code.
2. **Skin (peripheral)** — per body-surface face, the contact feature (the object across that face) or None. A sensor
   front-end, exactly parallel to `segment` (the retina). It feeds the touch column; it does NOT decide anything.
3. **Touch column (S1)** — a `Column` whose location frame is the **body surface** (faces as locations) and whose sensed
   feature is contact; it does feature-at-body-location sensorimotor modelling. Over motion it integrates the felt surface →
   **recognise objects by touch** (shape-from-contact) — the growth that later makes occlusion / partial observability
   tractable. First slice: it senses the current contact configuration.
4. **Contact-dynamics** — the object forward model is conditioned on **FELT contact at the LEADING face** (the face in the
   efference direction), not a geometric predicate. On felt contact with object O under efference e, O's response is learned
   per O: **yields** → O moves by `T·e` (the validated efference-parameterised T; T=I default, LMS-fit, revisable), and the
   self advances into O's vacated cell; **resists** → O stays (its own target blocked), the self is blocked; **passable** →
   no touch event, the self moves through. Replaces `replay.py`'s hand-coded solidity + the snap.

## 4. Reference frames & reuse (the "don't build parallel machinery" check)

- Body frame = the self's frame **given extent**, anchored to the nav L6a pose; the `operator` moves it. REUSE `operator.py`.
- Touch column = `Column(location = GridEncoder(small body frame))`. REUSE `column.py` AND the grid code — a body face is a
  POSITION in the body's 2-D coordinates (the somatotopic map is a 2-D body-surface map), so no new encoder (verified: 4/4 clean
  contact-at-face binding through the reused `GridEncoder`; the modular-metric aliasing degrades only recognise-by-touch, not the
  dynamics' identity binding). Instantiated by the `modality.py` factory from a four-field spec.
- Skin = a peripheral parallel to `segment`. New file `touch.py`, one concept.
- Efference = the L5PT copy (`broadcast_efference`, built) selects the LEADING face and parameterises the push. REUSE.
- Recognise-by-touch = the touch column's `recognize`/`perceive` (same machinery as vision recognition). DEFERRED growth.

## 5. Learned vs given (the bitter-lesson line)

- **Given** (architecture, like *having a retina*): the self HAS a body (extent) and a touch sense. Existence of the organ,
  not its content.
- **Learned**: the body's SHAPE (from the discovered self object's cells); per-object contact behaviour (yield-`T` / resist /
  pass); object identity by touch.
- **NOT assumed**: solidity (a revisable default — retracted the first time the self is observed to pass through an object);
  co-motion (`T` learned); which face leads (read from the efference).

## 6. Build slices (vertical)

1. **Body + skin peripheral** — the body surface of the discovered self, and per-face contact sensing (`touch.py`). [this slice]
2. **Touch column** — a `Column` sensing contact at body-surface locations (feature-at-body-location), wired from the agent.
3. **Contact-dynamics** — condition the object forward model on FELT contact at the leading face; yield-`T` / resist / pass
   learned per object; delete the `snap` and the hand-coded coupling. Re-run Push: kills #3/#4/#5/#9.
Later: recognise-by-touch (the column's `recognize`); multi-cell body; object-object contact (a box pushed by a box).
