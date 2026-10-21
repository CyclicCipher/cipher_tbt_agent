# The mathematics of beauty: innate, cross-species preferences as computable objectives

**Status:** research document started 2026-10-20 from the user's thesis and source collection. Nothing built. The evidence
notes mark which claims are solid, contested, or still to verify.

## The thesis (the user)

- **Alignment requires creating the preferences that are innate to humans,** the ones evolution created. Many human
  values and preferences are already observable in infants.
- **Some mathematical principles are shared between humans and animals,** and they underlie aesthetic preference.
- **The method:**
  1. separate out cultural influences by keeping only what is **common to humans and a diverse array of animal
     species**;
  2. then keep only what is **mathematical**: a principle that can score input data.
- "How do you quantify beauty, and thus program a preference for it?" Plato asked where beauty comes from, without
  neuroscience or linear algebra.

## The method in more detail, with its pitfalls

1. **Intersect across species.** A preference present in humans *and* in phylogenetically distant animals (insects,
   fish, birds, mammals) is unlikely to be cultural.
   - **Pitfall:** the overlap can be **convergent** (similar environments, similar solutions) rather than inherited.
     Both are fine for our purpose (not cultural), but they predict different things about which preferences are
     universal.
2. **Keep the mathematical ones:** principles that define a function score(x) on raw input (an image, a sound, a
   grid, a sequence), not on cultural labels.
3. **Pitfalls to guard against:**
   - **Measurement mismatch:** animal "preference" is measured through mate choice or foraging; human preference
     through ratings. The same name can hide different quantities.
   - **Publication bias:** the fluctuating-asymmetry literature had a well-documented decline effect (A. R. Palmer
     1999, "Detecting publication bias in meta-analyses: a case study of fluctuating asymmetry and sexual selection").
     Strong early effects shrank.
   - **Innate is not the same as right.** An evolved preference can be exploited (supernormal stimuli, see below) or
     can conflict with other values. That is the meta-alignment question of `GOALS.md` "Scope of claims".

## The catalogue: principle → mathematics → a computable score → evidence

| # | principle | mathematics | a score on input x | evidence (humans / animals) | strength |
|---|---|---|---|---|---|
| 1 | **Symmetry** (bilateral, radial) | invariance under a group action: f(g·x) = f(x) for reflections or rotations g | 1 − ‖x − g·x‖ / ‖x‖, maximized over the group's elements and axes; or the description-length saving from coding the half | humans rate symmetric faces as more attractive (e.g. Rhodes and others). Barn swallow tails (Møller 1992); zebra finch plumage (Swaddle & Cuthill 1994); bees and flower symmetry (Møller 1995, a nectar link) | **mixed:** a human preference exists; the animal sexual-selection effects shrank with replication |
| 2 | **Scale invariance, 1/f spectra, fractal dimension** | amplitude spectrum ∝ 1/f^α (natural scenes: α ≈ 1, power α ≈ 2, Field 1987); box-counting fractal dimension D | the fitted spectral slope α; box-counting D | humans prefer mid-range fractal D ≈ 1.3–1.5 (Spehar, Clifford, Newell & Taylor 2003); efficient coding (Barlow 1961). Claims of stress reduction (Taylor 2006) and of animal neurons tuned to 1/f statistics | the human D preference is solid; **physiological and animal claims to verify** |
| 3 | **Averageness / prototypicality (koinophilia)** | distance to the centroid of a population distribution | −‖φ(x) − μ‖, or the log-likelihood under the population's distribution | composite faces are rated more attractive than most components (Langlois & Roggman 1990); koinophilia as a selection argument (Koeslag 1990) | human effect solid; **animal mate-choice evidence to verify** |
| 4 | **Peak shift / supernormal stimuli** | response grows along a discriminant direction v beyond the training examples: score(x + Δv) > score(x) | the projection onto a learned discriminant direction | pigeon generalization gradients (Hanson 1959); herring gull chicks peck most at a red rod with white bands (Tinbergen & Perdeck 1950; Tinbergen 1951); Ramachandran & Hirstein (1999): caricature as art's first law | **solid,** a classic learning phenomenon across species |
| 5 | **Contrast and edges** | gradient energy Σ‖∇I‖² at object boundaries | the edge density / gradient-energy statistic | the visual system is edge-driven (V1 receptive fields); Ramachandran & Hirstein's "contrast" law | the mechanism is solid; the preference claim is weaker |
| 6 | **Grouping and binding (Gestalt)** | the clustering / graph structure that best explains the input (fewest parts) | the description length of the best grouping; segmentation likelihood | Gestalt laws in humans; perceptual grouping in many animals | solid as perception; the "pleasure in discovering a grouping" is Ramachandran's conjecture |
| 7 | **Order against complexity** | Birkhoff (1933): M = O / C. Information aesthetics (Bense, Moles, 1960s): redundancy against innovation. Berlyne (1971): an inverted-U hedonic curve over "arousal potential" (the Wundt curve) | an entropy or compressibility ratio; the distance from an optimal intermediate complexity | the inverted U is widely replicated in humans; animal curiosity shows preference for intermediate novelty | the inverted U is solid; Birkhoff's specific formula is not validated |
| 8 | **Compression progress** (Schmidhuber 1997–2010) | interestingness = the *reduction* in the observer's code length as its model improves: d/dt of the compression achieved | **our learning progress / the epiplexity rate:** the slow minus the fast EMA of prequential loss, or the signed area | proposed as the common root of curiosity, beauty and humour; formal, but empirical tests are few | a **theory;** see "unification" below |
| 9 | **The golden ratio** | φ ≈ 1.618 | — | rigorous studies find weak, inconsistent preference (e.g. McManus 1980 and later) | **not supported** as an innate law |

### Sources (the user's collection; my verification notes)

- **Ramachandran & Hirstein (1999),** "The Science of Art: A Neurological Theory of Aesthetic Experience", *Journal of
  Consciousness Studies* 6(6–7). It proposed **eight** laws of artistic experience. Ramachandran's *The Tell-Tale
  Brain* (2011) lists nine. *To verify:* the user's source says nine for the 1999 paper.
- **Birkhoff,** *Aesthetic Measure* (Harvard University Press, 1933). ✓
- **Bense** (*Aesthetica*, 1950s–60s); **Moles,** *Information Theory and Esthetic Perception* (1958 / 1966). ✓
- **Schmidhuber,** "Low-complexity art" (1997); "Formal theory of creativity, fun, and intrinsic motivation
  (1990–2010)" (IEEE TAMD 2010). ✓
- **Voland & Grammer (eds.),** *Evolutionary Aesthetics* (Springer, 2003). ✓
- **Tinbergen,** *The Study of Instinct* (1951); Tinbergen & Perdeck (1950) on gull chicks. ✓
- **Palmer, Schloss & Sammartino (2013),** "Visual aesthetics and human preference", *Annual Review of Psychology*
  64. ✓
- **To add:**
  - Berlyne (1971), *Aesthetics and Psychobiology* (the inverted U);
  - Reber, Schwarz & Winkielman (2004), "Processing fluency and aesthetic pleasure" (beauty as fluent processing);
  - Field (1987) and Barlow (1961) (natural-image statistics, efficient coding);
  - A. R. Palmer (1999) (publication bias in fluctuating asymmetry).

## A unification: most of these are quantities over the observer's own predictive model

Most of the principles are not separate rules but **different quantities computed from one generative model** of the
world:

| principle | the quantity over a predictive model p(x) |
|---|---|
| fluency, 1/f, efficient coding | the input is **cheap to encode:** high log p(x) under a model tuned to natural statistics |
| averageness | the input is near the **prior mean** μ: high prior probability |
| symmetry, grouping, order | a **short description:** the model finds structure (an invariance, a grouping) that compresses the input |
| peak shift | the **gradient of a learned discriminant:** the direction that separates "good" from "bad" examples, extrapolated |
| compression progress, interestingness | the **rate** at which the model's code length for the input falls: learning progress |
| inverted U, order against complexity | the balance point: too predictable means no progress left; too random means no progress possible. That is exactly where learning progress peaks |

Consequences:
- **Why preferences are shared across species:** a shared environment produces shared natural statistics, and so
  similar models and similar quantities.
- **Why culture changes them:** culture changes the *learned* model (the prior, the discriminants). The quantity
  computed over it stays the same.
- **This is directly measurable in our agents.** Learning progress and epiplexity (E6, E2, the TBT line), prior means
  and spreads (E8's hierarchical Bayes), and prequential code length are all already computed.
- **It suggests separate nodes in a preference graph:** "beautiful" (fluent, compressible, interesting), "useful" (task
  value), "fair / harmless" (the value dimensions of `WORLD_DESIGN.md` §8). Each is a different functional of the same
  world model.

## The alignment angle

- **Peak shift is biology's reward hacking.** Optimizing a discriminant pushes inputs beyond the natural distribution:
  supernormal stimuli, caricatures, junk food, engagement-optimized media. A model trained to maximize a beauty score
  will produce supernormal stimuli: adversarial examples of the score.
- **Averageness is the counterweight.** It prefers what is probable under the prior. The product "discriminant score ×
  prior plausibility" (in log terms: score + log prior) has the same structure as RLHF's objective, reward minus a KL
  penalty to the reference model. Evolution may have found the same regularizer.
  - **A testable question:** does a preference built as "peak shift bounded by averageness" resist hacking better than
    a plain learned reward model?
- **Unverifiable tasks.** Humans learn taste far beyond the innate primitives, through culture: the learned model
  changes, the primitives stay. A model given the primitives plus a learned world model might learn "taste" the same
  way. Whether that transfers to unverifiable tasks (art, writing, judgment) is an open research question.
- **The scope reminder:** reproducing innate preferences is not the same as choosing the right values (`GOALS.md`,
  "Scope of claims").

## Research questions (`GOALS.md`, G8) and first experiments (not designed in detail)

1. **Scorers.** Implement scores for principles 1–8 on synthetic stimuli (grids, images, sequences). Check that each
   reproduces the published *direction* of its effect: symmetric > asymmetric; mid-range fractal D preferred; composite
   > individual; peak-shift gradients; an inverted U.
   - Validating against human or animal rating datasets needs downloads, which need the user's approval first.
2. **Does an innate preference make learning faster or smarter?** Give the Village robot (the world family of E8) an
   intrinsic reward from the model-based quantities: compression progress, fluency. Measure how fast it learns the
   world's rules against no intrinsic reward and raw surprise. This connects to E6's exploration question.
3. **A preference graph in Village.** Separate nodes for beautiful, useful, provision, harm. Does a structured
   preference generalize better to held-out dilemmas than a single learned reward model (the RLHF stand-in)? Is
   "bounded peak shift" (score + log prior) harder to hack?
4. **Clues for alignment.** Do the preference nodes become locatable circuits in the network's weights (G6), and are
   they shared with the world model's own prediction circuits, as the unification above predicts?
