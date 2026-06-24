# Chollet's Program Synthesis ↔ Our Deep Models

> **What this is.** A side-project scoping note: François Chollet's program-synthesis
> thesis for AGI, and where it confirms, names, threatens, or extends the theorizing in
> `experiments/RecurrentWorldModel/`. Status: research + connection mapping, pre-implementation.
>
> **Status legend** (borrowed from RWM): `established` empirically confirmed · `directional`
> promising, unproven · `open` active question · `theoretical` unconfirmed · `tension` the
> two frameworks pull in opposite directions and we should be honest about it.
>
> **Update (RWM refocus).** This note was originally written against RWM's exploratory
> "clamped settling core," which has since been **shelved** — RWM's current work is improving
> **transformers** through controlled experiments (the temporal-PoPE results cited below are part
> of that). The Chollet content here is **architecture-general**: value-centric vs program-centric
> abstraction, transduction vs induction, and the A∘B composition test apply directly to the
> transformer agent in [LEARNING_AGENT.md](LEARNING_AGENT.md). Where the text below says
> "Risk 2 / Risk 4 / clamp / settling," read it as the historical packaging of those general
> points, not a current architecture commitment.

---

## 1 · Chollet's thesis, in the terms that matter to us

**Intelligence = skill-acquisition efficiency**, not skill. ("On the Measure of
Intelligence," 2019.) A system that interpolates within its training distribution has
*crystallized skill*; intelligence is the efficiency with which it handles **genuine novelty**
— situations not reducible to anything seen. ARC-AGI is the benchmark built to isolate that.

**ARC-AGI is a program-synthesis problem.** Each task gives a few input→output demos; you must
infer the *transformation rule* and apply it to a new input. Chollet's claim: the right solution
is to **search the space of discrete programs that exactly explain the demos**, not to fit a
continuous function to them.

**The bottleneck of pure program synthesis is combinatorial explosion.** Discrete search over a
DSL blows up. Chollet's fix (and the entire thesis of his startup **Ndea**): **use deep learning
to *guide* the discrete search** — intuition prunes the tree, search supplies the exactness and
the generalization. "Deep learning guided program synthesis." Neither half alone; the fusion.

**The Kaleidoscope Hypothesis** (the *why* underneath). The world's apparent infinite novelty is
recombination of a *small* set of recurring "atoms of meaning." Intelligence is the engine that
(a) **mines** experience for these reusable atoms (abstractions) and (b) **recombines** them on
the fly to cover novelty. A growing, reusable *library* of abstractions is the substrate.

**Two kinds of abstraction — both are analogy-making** (the load-bearing distinction for us):

| | **Value-centric** | **Program-centric** |
|---|---|---|
| Compares via | continuous **distance** | exact **structure match / isomorphism** |
| Domain | continuous, geometric | discrete, graph/program-like |
| Analogy type | value analogy (similarity) | program analogy (same structure) |
| Underlies | perception, intuition, pattern cognition, **deep learning** | reasoning, exact composition, **program search** |
| Cognitive type | **Type 1** (fast, intuitive) | **Type 2** (slow, deliberate) |

All cognition is a *combination* of the two. Deep learning is stuck in the left column; ARC needs
the right column; AGI needs the bridge.

**Transduction vs Induction** (the empirical crystallization, from the ARC Prize 2024 Technical
Report, arXiv:2412.04604). The two abstraction types showed up as two *winning machine strategies*:

- **Transduction** — directly predict the output (test-time-trained deep nets, e.g. MindsAI/the
  ARChitects → up to 55.5%). Value-centric. Wins on perceptual/fuzzy tasks.
- **Induction** — synthesize an explicit *program*, then run it (DSL search / LLM-generated Python
  → ~40%). Program-centric. Wins on compositional/precise tasks.
- **Top scores ensemble both** — "all top scores use a combination of transduction and induction,"
  because *each solves task categories the other can't.* Neither dominates.

---

## 2 · The one connection that matters most

> **RWM's Risk 2 ("consistency ≠ correctness") *is* Chollet's value-centric / program-centric
> split, rediscovered from the inside.** `directional`

RWM Risk 2 asks (architecture.md §9): is goal-clamped settling *genuine multi-step inference, or
a fancy Hopfield net that settles to the nearest stored pattern?* In Chollet's vocabulary that is
**exactly** the question: does the settling do **program-centric** recombination (exact structural
composition of novel rule combinations), or only **value-centric** interpolation (snap to the
nearest attractor = nearest stored pattern = continuous similarity)?

A Hopfield net is the *purest* value-centric machine: it relaxes to the nearest stored memory by a
distance metric. RWM's fear "is it just a Hopfield net" = Chollet's diagnosis "deep learning only
does value-centric abstraction." **They are the same worry.** This is worth a lot: it means RWM's
single most-cited internal risk has an external theory, a name, a 5-year literature, and — crucially
— a **test**.

> **RWM's decisive experiment (A∘B never seen together) is Chollet's program-centric test.**
> `established` (as a framing)

The A∘B test (README §"single most decisive experiment"): train on mechanics A and B separately,
test on their composition. Passing requires **exact structural recombination** of two known atoms
into an unseen whole — this is the *definition* of program-centric abstraction and the *definition*
of the Kaleidoscope recombination engine. A∘B is not a generic OOD probe; it is precisely the assay
that separates Chollet's two columns. If RWM passes A∘B, it is doing program-centric abstraction
inside a continuous substrate. If it fails, Chollet's framework predicted exactly why: continuous
interpolation cannot do discrete exact recombination.

This sharpens what A∘B is *for*. Not "does the architecture generalize" but "**which column of the
abstraction table is this architecture in?**"

---

## 3 · Where RWM is secretly already Chollet-aligned

**RWM's "Learn" mode = Test-Time Training, made native.** `directional` The ARC 2024 result that
shocked people: TTT (fine-tune on the handful of demos at inference) was *essential* — it carried
the top transduction scores. RWM's **Learn clamp mode** (architecture.md §2.3 item 4: clamp goal +
input, settle, consolidate weights locally toward that equilibrium) is *test-time adaptation as a
clamp mode rather than a separate fine-tuning pass.* Chollet's empirical finding that TTT is
load-bearing is independent support for RWM's decision to make learning a clamp of the one operator
instead of a frozen-then-finetuned pipeline.

**RWM's settling-as-search is a candidate dissolution of the discrete/continuous split.**
`theoretical` Ndea's literal recipe is *handcoded DSL + neural heuristic to guide the tree search* —
which **violates RWM's bitter-lesson constraint** (memory: `feedback_bitter_lesson` — no handcoded
rules, no DSL of domain primitives). But the *spirit* — "intuition guides search toward exact,
recombinable structure" — survives translation. In RWM the **settling dynamics already are a
search**: a continuous descent on an energy landscape (architecture.md §5, energy ↔ diffusion). So
RWM can be read as the bet that you can get "deep-learning-guided program search" **without a
discrete DSL**, by making the programs be *attractors/trajectories* in latent space and the search
be *settling*. The deep net's intuition isn't a heuristic bolted onto a separate symbolic searcher;
it **is** the search operator. If that works, it dissolves Chollet's duality instead of bridging it
— a stronger and more bitter-lesson-compatible claim than Ndea's.

**The hinge is composition fidelity.** `open` Whether the above bet pays off reduces to RWM's
existing open question Q1/A4 (architecture.md §10): do learned relation-transformations *compose*,
`M_{r2} M_{r1} = M_{r2∘r1}`? That equation **is** program-centric abstraction expressed in the
continuous substrate: if latent relations compose like discrete program primitives, RWM gets the
right column for free, inside the left column's machinery. Composition fidelity is therefore not a
side-question — it is *the* measurement of whether the duality-dissolution bet is real. **Recommend
elevating Q1/A4 from "tested alongside" to a primary gate, co-equal with A∘B** (they test the same
thing from two sides: A∘B at the behavioral level, M·M at the mechanistic level).

---

## 4 · Where Chollet threatens RWM — the honest tensions

**`tension` 1 — Chollet would bet RWM fails A∘B, for a principled reason.** His whole position is
that *no* purely continuous/transductive system does genuine recombination — that's why 5 years of
deep learning bounced off ARC until discrete induction was added. RWM is, architecturally, a
transduction machine: every mode (even Reason) settles to an output *state*, never emits an
inspectable, verifiable, **reusable program**. The null hypothesis Chollet hands us is strong and
specific, and we should hold A∘B to it.

**`tension` 2 — RWM has no library of abstractions; the Kaleidoscope says it needs one.** Chollet's
engine has two halves: mine reusable atoms **and store them in a growing library** to recombine
later. RWM has the recombination substrate (the relational block) but **no explicit, inspectable,
accumulating store** of named abstractions. Its working-memory clamp carries *one* latent conclusion
forward; it is not a library. Open question for the side-project: does RWM need an explicit
program-centric memory bolted alongside the continuous core (→ neuro-symbolic, against the
one-operator thesis), or can `continual_learning_design.md` + the relational monoid serve as an
*implicit* library? This is the sharpest new design question Chollet surfaces.

**`tension` 3 — "Neither dominates" is a caution to the one-operator bet.** The ARC report's
empirical headline is that transduction and induction solve *different* task categories and the
best systems **ensemble two mechanisms**. RWM bets *one* operator covers all modes. Chollet's data
is the strongest external evidence for **RWM's Risk 4** (representation/reasoning interference on
shared weights): if value-centric (Represent/Perceive) and program-centric (Reason) genuinely want
different machinery — as the ARC ensembles suggest — then forcing them onto shared weights is
exactly the fight Risk 4 names. The transduction/induction split is a second, independent lens on
Risk 4, and a reason to take that risk seriously rather than assume unification.

---

## 5 · What this buys us (and a proposed first probe)

**Conceptual payoff (no new code):**
1. Risk 2 gets an external name, theory, and the A∘B test reframed as "which abstraction column?"
2. Q1/A4 (composition fidelity) gets promoted: it is the mechanistic measurement of program-centric
   abstraction, co-equal with A∘B. Recommend making it a primary gate.
3. Risk 4 gets a second lens (transduction vs induction) and external evidence it's real.
4. A new, sharp design question (tension 2): does the system need an explicit abstraction library?

**Proposed first probe (reuses RWM infrastructure, GPU job for the user):** run the existing A∘B
test, but **instrument it as Chollet's discriminator**. Alongside pass/fail, log the two diagnostics
that distinguish the columns:
- **Value-centric signature:** does the settled state for A∘B sit on the *interpolation manifold*
  between the A-attractor and the B-attractor (nearest-stored-pattern behavior)? Measure latent
  distance of the A∘B equilibrium to the A and B equilibria.
- **Program-centric signature:** does composition fidelity hold on the held-out combination —
  `M_{B}M_{A}` applied to a fresh input matches the true `M_{B∘A}` transform? (Q1/A4 measured *on
  the unseen composition*, not on seen ones.)

A system that passes A∘B *by interpolation* (first signature high, second low) is a sophisticated
Hopfield net — Risk 2 realized. A system that passes *by composition* (second signature holds OOD)
is in Chollet's right column inside a continuous substrate — the duality-dissolution win. **The
point of the probe is not whether A∘B passes, but by which mechanism** — that is the question
Chollet's framework lets us ask precisely, and the one RWM couldn't name before.

---

## Sources

- Chollet, *On the Measure of Intelligence*, 2019 — [arXiv:1911.01547](https://arxiv.org/abs/1911.01547).
  Intelligence as skill-acquisition efficiency; Core Knowledge priors; ARC. The Kaleidoscope
  Hypothesis and the value-centric / program-centric abstraction dichotomy are in §III ("A new
  perspective").
- *ARC Prize 2024: Technical Report* — [arXiv:2412.04604](https://arxiv.org/abs/2412.04604).
  Transduction vs induction; test-time training; deep-learning-guided program synthesis; "all top
  scores combine transduction and induction."
- **Ndea** (Chollet's startup) thesis — [ndea.com](https://ndea.com). Deep learning guided program
  synthesis: search discrete programs that *exactly* explain data, with deep learning guiding the
  search to beat combinatorial explosion; "program synthesis and deep learning equally important."
- Secondary on the Kaleidoscope Hypothesis & abstraction types: MLST "It's Not About Scale, It's
  About Abstraction"; alphanome.ai write-up.
