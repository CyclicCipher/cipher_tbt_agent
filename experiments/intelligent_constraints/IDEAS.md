# Candidate constraints

Each entry gives the constraint, why it might hold for every case (the generality test in `README.md`), where it came
from, its status, and the planned test.

## The HNN analogy, made exact

| | HNN | Dagger memory (C1) |
|---|---|---|
| learned | a scalar H | a memory matrix: keys K, values V |
| fixed structure | the symplectic matrix J (motion = J ∇H) | the converse read is the transpose of the forward read |
| guarantee | energy is conserved | forward and converse scores are the same number |
| applies to | conservative systems only | every binary relation, because every relation has a converse |

---

## C1. Dagger: the converse read is the transpose

**Constraint.** One stored fact serves both directions. A key–value memory y = V σ(Kx) also reads backwards through the
same parameters: Kᵀ σ(Vᵀx). A unit trained to map the cue for "Jason, father" to the value "Joseph" is then cued by
"Joseph" and writes back "Jason".

**Why general.** Relations form a dagger category:
- every relation R has a converse R°;
- (R°)° = R and (S∘R)° = R°∘S°;
- as a 0/1 matrix, the converse is the transpose.

The transpose always exists, while the inverse exists only for one-to-one relations. A further fact: in the category of
relations, a relation is a function exactly when it has a right adjoint, and that adjoint is its converse (f ⊣ f°). The
unit says f is total and the counit says it is single-valued. The CTKG's `succ ⊣ pred` naming was therefore literally
correct.

**What is still learned.** Which token names the converse ("son" is father°). That is a few parameters per relation,
learnable from the facts seen both ways, and it should transfer to every one-way fact because the facts are shared.

**Origin:**
- the CTKG FrameSolver (converse pairs discovered by unit/counit coverage, then applied to held-out queries);
- Kosko's bidirectional associative memory (1988), which is the linear case;
- the HNN analogy (the user).

**Status:** Experiment 1, pre-registered.

## C2. Functoriality: composing relations is multiplying their maps

**Constraint.** If relation R is the map F(R), then F(S∘R) = F(S) F(R). A two-step question ("the father of the actor
who played Aquaman") is two chained reads. For the chain to work, the output of one read must be a valid cue for the
next, in the same code space. Together with C1, the converse of a chain comes for free: (S∘R)° = R°∘S°.

**Why general.** Composition of relations is always defined; it is the category's composition.

**Known failure it targets.**
- Transformers resolve the first step too late in their layers for the second lookup ("Hopping Too Late", Biran et al.
  2024).
- They compose out of distribution only if the second-step facts are stored again in later layers (Wang et al. 2024,
  grokked transformers).

**Status:** planned as Experiment 2. It needs a looped or shared-weight read, so the same memory can be applied twice.
This connects to the Neural Turing Architecture's looped model.

## C3. One code per entity

**Constraint.** The code that cues an entity (reading it) is the code the memory writes for it (saying it).

**Why general.** Identity is reflexive: an entity is the same thing in every role.

**Evidence it matters:**
- Wang & Sun 2025 name "inconsistent concept representations" as a cause of the curse.
- Ma et al. 2026 ("identity bridge") get about 40% reversal by training "the name of A is A", which forces the two codes
  to agree.

**Simplest form:** tied input and output embeddings.

**Status:** tested partly in Experiment 1 (arm D: the dagger memory with untied embeddings).

## C4. Allocation: create a new entity only on refutation, never on incompleteness

**Constraint.** A store that replaces explicit node creation (Charles Simon's UKS) needs a rule for when two mentions
are the same entity. The rule: allocate a new code (a new attractor) only when every existing code is contradicted by
the evidence, never merely because the evidence is partial.

**Origin:** the TBT recognition-under-occlusion work (memory `reference_recognition_under_occlusion`), where one rule
works at three scales.

**Status:** an idea. It needs a continuous store with an allocation step; there is no design yet.

## C5. Set-valued reads

**Constraint.** A one-to-many question ("Joseph's children") returns a set. Under C1 this comes free: the transpose of a
many-to-one map returns the superposition of all subjects.

**Status:** measured inside Experiment 1, which includes one-to-many and many-to-one relations. Score = probability mass
on the correct set.

---

## Rejected candidates, and why

These come from Gemini's answer (`notes/gemini_continuous_uks_*.png`), evaluated 2026-10-18.

| candidate | why rejected |
|---|---|
| invertible operators (RotatE, Lie-group relations) | an inverse exists only for one-to-one relations, so this is HNN-narrow; the converse (transpose) is what every relation has |
| "relations must be symmetric" | father-of is not symmetric; what must be shared is the stored fact, while direction lives in the role labels |
| box or hyperbolic embeddings | they serve is-a hierarchies, not reversal; pure containment breaks on exceptions ("penguins don't fly") |
| VSA "lossless" binding | unbinding is exact for one binding, noisy for a superposition of many; the roles are chosen by hand (the rigid parse again) |

**Overstated diagnoses in the same answer:**
- "Conflating computation with memory" is not the cause. A separate memory that is a directed key→value table has the
  same curse.
- One fact written by backprop does not cause catastrophic forgetting (ROME edits single facts locally). The real
  failures are that edits don't spread, and that models degrade over thousands of sequential edits.

**The hard part it hid.** "The encoder projects the statement into a structured query vector": that step is the parse,
plus deciding which mentions are the same entity. These are C4 and an open problem: learning the roles.
