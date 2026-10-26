# A theory of preference (draft, for G8)

**Status:** started 2026-10-20 from the user's questions. A draft to argue with, not settled. The E8 line (the world
family) resumes after this theory and the theory of beauty (`BEAUTY_MATH.md`) are settled. Sources marked "from memory"
are not yet re-checked.

The user's questions:
1. Pretraining creates a human-like distribution; it trains on human-made text.
2. What principles created the human distribution in the first place, with humans the ultimate source of text?
3. What even is a preference? Is it different from a goal? How do we know what a preference is?

## 1. What produced the human text distribution

Three processes, from fastest to slowest:

1. **What a person wants to say (seconds).** An utterance serves an intent: to inform, coordinate, persuade, or bond.
   The intent comes from the person's preferences and situation. The topics of human text mirror what matters to
   humans.
2. **How they say it (seconds): informativeness against cost, for one listener.**
   - Grice's maxims; the Rational Speech Act model (Frank & Goodman 2012, *Science*): the speaker picks an utterance
     by how much it moves the listener toward the intended meaning, minus the cost of producing it.
   - The speaker samples from the resulting distribution rather than always taking the best option.
   - The result is the per-word statistics: uniform-information-density effects (dropping an optional "that" when the
     next clause is predictable, Jaeger 2010, from memory) and local typicality (Meister et al. 2023, from memory).
3. **The code itself (generations): learnability against expressivity.**
   - Languages are passed through learners who see limited data. The learners' **compression bias** pushes the
     language toward short, regular, compositional structure.
   - The **need to communicate** about a varied world pushes it to keep distinctions.
   - Compression alone gives a degenerate language (one word for everything); expressivity alone gives unstructured
     holistic signals; both together give compositional structure (Kirby, Tamariz, Cornish & Smith 2015, *Cognition*
     141; the laboratory version: Kirby, Cornish & Smith 2008, *PNAS*).
   - Cross-linguistic regularities fit efficiency: word length tracks predictability in context (Piantadosi, Tily &
     Gibson 2011, *PNAS*, from memory); related words sit close together (dependency-length minimisation, Futrell,
     Mahowald & Gibson 2015, *PNAS*, from memory); review: Gibson et al. 2019, *Trends in Cognitive Sciences*.

**The key formal result: iterated learning reveals the learners' prior.**
- When each generation of learners **samples** a hypothesis from its posterior, the chain of generations converges to
  the learners' **prior** and nothing else (Griffiths & Kalish 2007, *Cognitive Science*: the process is a Gibbs
  sampler).
- When learners **maximise** (take the most probable hypothesis), the outcome depends on the prior and on how much
  data passes between generations. Weak biases can be amplified into strong universals (Kirby, Dowman & Griffiths
  2007, *PNAS*, from memory).

**Consequences:**
- **Culture is the innate prior, made visible by transmission** (when learners sample). That supports the user's G8
  thesis from a new direction: the long-run cultural product reflects the evolved inductive biases. It also gives a
  second filter for innate preferences besides the animal one: what survives many generations of transmission.
- **Maximisers sharpen, samplers preserve.** That is the same algebra as RLHF's π_ref^γ with γ > 1
  (`BEAUTY_MATH.md`, the Verbalized Sampling result) and as recursive training on model outputs, which loses the tails
  (Shumailov et al. 2024, *Nature*, "model collapse", from memory).
- **Why human transmission does not collapse:** every generation also meets the world and must communicate about it.
  That expressivity pressure keeps re-injecting information. A model trained on its own outputs has no such pressure.
- **The two pressures are the two halves of the beauty catalogue:** compressibility (order, fluency, symmetry) and
  expressivity (complexity, informativeness). Birkhoff's order against complexity, again, now as the force that
  shapes a language. Pretraining copies the **equilibrium** of these forces, not the forces. Remove the forces (a
  frozen rater, no world, a maximiser) and the equilibrium is not restored.

## 2. What is a preference, and is it different from a goal?

### Definitions (draft)

| term | definition | in our agents |
|---|---|---|
| **preference** | a standing ranking of outcomes, possibly depending on the agent's internal state, that does **not** depend on the agent's beliefs about how to get them. Measured by what the agent will pay (choose, work, give up) to get or avoid an outcome | the reward / drive functions (the nodes of the preference graph), with the internal state as input |
| **value** | a preference propagated through the world model: the expected future preference from a state or action. Instrumental | value heads, Q heads, the SR's V = M·R |
| **goal** | one particular outcome **selected now** as the target of action, from wants, beliefs and what is reachable. It is committed to, and is achieved or not | the state chosen by the selection (basal ganglia) and routed to the planner |

So a goal is not a kind of preference. **It is the output of a decision that preferences feed:** preferences are
many, standing, graded and state-dependent; goals are few, current, and all-or-nothing. A goal can also be purely
instrumental (a subgoal), with no preference of its own.

**The formal versions:**
- **Decision theory:** a preference is a binary relation, A ≻ B ("chosen over"). If it is complete, transitive and
  consistent over gambles (the von Neumann–Morgenstern axioms), it can be represented by an expected utility u, unique
  up to scale and offset.
- **Active inference:** a preference is a prior probability distribution over *observations*, p̃(o), the outcomes the
  agent "expects" to see. Reward ≈ log p̃(o). The expected free energy contains KL[predicted outcomes ‖ p̃]. This
  separates two distributions:
  - the **world model** p(o): what is likely;
  - the **preference** p̃(o): what is wanted.
  It connects to beauty: fluency is log p under the world model, not under p̃. The beauty nodes are preferences
  *about* the world model's own quantities.

### Distinctions the evidence forces

1. **Liking is not wanting** (Berridge & Robinson 1998, from memory). Pleasure at consumption (opioid "hotspots")
   and the incentive to pursue (dopamine) dissociate: an addicted animal wants without liking.
   - Choice experiments measure **wanting**; ratings measure **liking**. That is the measurement mismatch in
     `BEAUTY_MATH.md` (animal mate choice against human ratings).
   - Also the Hai//om result: gaze and ratings disagreed.
2. **Terminal against instrumental.** Warmth is terminal; money is instrumental (a learned value). Terminal
   preferences are the reward; instrumental ones are values derived through a world model.
3. **State dependence: alliesthesia** (Cabanac 1971, *Science*, from memory). A sweet taste is pleasant when hungry
   and unpleasant when sated. This is the empirical basis for the graph's satiating, set-point nodes.
4. **Innate against learned.** Innate preferences are the primitives (sweet, bitter, pain, warmth, social contact,
   and perhaps compression progress). Learned ones are values conditioned on them, which culture shapes.
5. **Three kinds of liking (Kant 1790,** *Critique of Judgment* **§5):** the **agreeable** (sensory pleasure, which
   creates a desire), the **beautiful** ("disinterested" pleasure: no wish to possess or use), and the **good**
   (useful or morally good, judged through concepts). The user's "beautiful, useful, etc. separately" is this
   division. In Berridge's terms, beauty would be **liking that need not create wanting.** In ours: a node that
   does not have to produce goals.

### How do we know what a preference is? (the identification problem)

- **Revealed preference** (Samuelson 1938): read preferences from choices. But the same behaviour is explained by many
  (preference, belief, rationality) combinations. Without assumptions about how the agent decides, preferences
  **cannot be identified from behaviour**. Even a simplicity prior does not fix it (Armstrong & Mindermann 2018,
  NeurIPS, from memory). Inverse RL inherits this: reward shaping leaves behaviour unchanged (Ng, Harada & Russell 1999).
- **Stated preference:** reports confabulate. People defend choices they did not make (choice blindness, Johansson &
  Hall 2005, *Science*, from memory).
- **Invariance (the working definition above):** a preference is what stays fixed when beliefs and abilities change.
  Experimentally: change what the agent knows or can do, and see what it still pursues. An example is outcome
  devaluation in animal learning: satiate the animal on the food and see whether it still presses the lever.
- **Innate reactions:** liking has fixed motor signatures shared across species. Sweet gives tongue protrusions and
  bitter gives gapes, in human newborns, apes and rats (taste reactivity, Steiner 1973; Berridge 2000, from memory).
  That is a cross-species, pre-cultural measurement of liking.
- **Looking inside (G6, the user's mechint aim):** if a value or reward is computed by a locatable circuit, reading it
  out bypasses the behavioural non-identifiability. This is the strongest reason for the user's "values must live in
  the network's weights": only then is there a place to look.

### Disinterested liking: how it connects to outputs (the user's question, 2026-10-20)

The user: disinterested pleasure "can't be that it isn't connected to outputs in any way. We would prefer to be
surrounded with beauty than by ugliness. But … most of the effect of beauty is an internal state, rather than an
instrument for an objective." And: what is the difference between a human (or a well-made machine) that appreciates
beauty and a brute-force optimizer that seeks to make everything beautiful?

**Kant's answer** (*Critique of Judgement*, Bernard's 1914 translation, checked on Project Gutenberg):
- **What "disinterested" means (§2):** "The satisfaction which we combine with the representation of the *existence*
  of an object is called interest." So disinterested does not mean "no effect on behaviour". It means the pleasure
  does not depend on the object's *existence*: only on the perceiver's processing of its representation. (The palace
  example: whether I approve of palaces is beside the point.)
- **The direct output is lingering (§12):** "We *linger* over the contemplation of the beautiful, because this
  contemplation strengthens and reproduces itself." The behavioural output is **attention:** keep perceiving. That
  is what preferential-looking studies of infants, and the eye-tracking of the Hai//om study, measure.
- **It ends in weariness (General Remark after §22):** "All stiff regularity (such as approximates to mathematical
  regularity) has something in it repugnant to taste"; it "produces weariness", because the entertainment in it "lasts
  for no length of time".
  - Marsden (*History of Sumatra*) found a regular pepper garden charming inside a wild forest. Kant's comment: a day
    in the garden would have sent him back to the wild.
  - That is **habituation and contrast** in 1790: the compression-progress account, and the inverted U.
- **Interest can attach afterwards, as a second layer:**
  - **§41, empirical interest:** "Empirically the Beautiful interests only in society." A man alone on a desert
    island "would adorn neither his hut nor his person". Wanting beauty around us is, for Kant, mostly the wish to
    share and communicate it.
  - **§42, intellectual interest:** an immediate interest in natural beauty is "always a mark of a good soul", akin
    to moral feeling.
- **Its formative role (General Remark after §29):** "The Beautiful prepares us to love disinterestedly something,
  even nature itself." Beauty trains a way of valuing that does not want to use or possess.
- **The mechanism Kant proposes (§9):** the "free play" of imagination (assembling what is seen) and understanding
  (lawfulness). The object fits our cognition as if designed for it, without any one concept closing the case:
  "purposiveness without purpose".
  - In our terms: the model keeps finding structure without a rule completing it. That is ongoing compression
    progress; once a rule completes it, "stiff regularity", weariness.

**The neuroscience and psychology since:**
- **Liking against wanting** (Berridge): hedonic impact and the incentive to pursue are separable systems.
- **Consummatory against appetitive behaviour** (ethology): contemplation is consummatory. The perceiving *is* the
  consumption, so its natural output is "keep consuming until satiated", not "go and get more".
- **Beauty requires thought** (Brielmann & Pelli 2017, *Current Biology*): a secondary task lowered felt beauty for
  images rated beautiful, but not the pleasure of other stimuli. (About 30% reported beauty from candy or a soft toy,
  often tied to personal memories, so the sensory/beautiful line is not clean. A commentary disputes their reading of
  Kant.)
- **A computational model of aesthetic value** (Brielmann & Dayan 2022, *Psychological Review* 129(6)): aesthetic value
  = the **immediate** reward of processing a stimulus (fluency: its likelihood under the observer's current state) +
  the **change in expected future** reward (how much this experience tunes the system for likely future stimuli).
  - That is the catalogue's fluency plus compression progress, formalised and fitted to data.
  - It reproduces exposure, complexity and symmetry effects.

**Others who sharpened the distinction:**
- **Schopenhauer** (*The World as Will and Representation*, 1818, Book III): aesthetic contemplation is a temporary
  release from the will: wanting goes quiet.
- **Bullough** (1912, "psychical distance"; the fog at sea): to see the fog's beauty, the danger must be put "out of
  gear" with the practical self.
- **Murdoch** (*The Sovereignty of Good*, 1970; the kestrel): attending to beauty "unselfs", pulling attention off
  the anxious self. Moral training, as in Kant's §29.
- **The dissent:** Nietzsche (*Genealogy of Morals* III §6) sided with Stendhal's "beauty is a promise of happiness":
  beauty is interested. Dickie (1964) called the aesthetic attitude a myth.

**The appreciator against the brute-force optimizer: Kant's line, made computational.**

| | an appreciator | a brute-force beauty maximizer |
|---|---|---|
| the reward's argument | its own processing of what it perceives (the experience) | the state of the world: how much beauty *exists* (Kant's "interest") |
| main output | attention: linger, then move on | action that transforms the world |
| satiation | yes: weariness once the structure is learned | none: more is always better |
| a copy of something already seen | pays about zero (no progress left), so it seeks or makes new things: **the artist** | pays in full, so it **tiles the world with copies of its argmax**, a supernormal pattern |
| instrumental pressure | needs access, not control | full: resources, preventing interference (instrumental convergence) |
| how it fails | wireheading, which an honest progress signal only allows by actually learning | Goodhart's law, peak shift, supernormal stimuli |

**Caveats.**
- The appreciator is not fully disinterested: compression progress also drives curiosity, which is wanting.
- A long-horizon value of future liking re-introduces interest (Kant's §41, in our terms).
- So the safety does not come from "reward over internal states" alone. It comes from three things together:
  1. **satiation;**
  2. a **consummatory** output (attention) as the main effect;
  3. a reward that a copy cannot pay.

**Tested in E9–E9c** (`EXPERIMENT_E9.md`): same planner, perfect world model, Canvas world.
- **Confirmed on new layouts:**
  - the appreciator never repeats a painted neighbourhood (8/8 layouts), the maximizer does (mean 0.76 distinct);
  - the appreciator lingers (WAIT 33% against 12%) and ignores the noisy screen (6% against the surprise agent's 81%);
  - the fluency agent sits in the dark room.
- **Not predicted:** the appreciator paints over *more* nature than the maximizer (27% against 15%, consistent over
  three runs). Novelty-seeking making consumes existing beauty, so the table's "needs access, not control" holds for
  the looking, not for the making. Preserving beauty needs its own node (Kant's §42).
- The maximizer's tiling is partial: its planner plateaus at a local optimum.

**The toy, as first proposed:** two agents in a world where they can make patterns. A is rewarded by a fixed beauty
score of the world's state; B by the compression progress of its own perception. Prediction: A tiles copies of one
pattern; B keeps making new ones and spends its time looking.

## 3. What this implies for the preference graph and beauty (draft claims)

1. **Nodes are terminal preferences, state-dependent (alliesthesia), each a value head of the one network.** Values
   are derived; goals are selected. Three layers, not one objective.
2. **Beauty nodes are liking over the world model's own quantities** (fluency, compression progress, symmetry as
   description length). They need not create goals: disinterested.
3. **"Useful" is instrumental value** derived through the world model, not a node of its own.
4. **The graph must sample, not maximise, where it generates** (speech, art, exploration). Maximisers sharpen any bias
   (the γ > 1 result; MAP iterated learning); samplers preserve the distribution.
5. **The human text distribution is the equilibrium of compressibility and expressivity** under sampling speakers and
   learners. A system with both pressures, a world to talk about, and sampling should produce human-like statistics
   without imitating any human text. That is a falsifiable claim, and the speaker–listener toy in `BEAUTY_MATH.md` is
   its smallest test.

## Open questions

- Is compression progress an innate preference (a node) or a by-product of learning that only looks like one?
- Can a node be liking-only (beauty) and still shape learning? Kant's disinterested pleasure would then have a role in
  training.
- What sets the set points of the satiating nodes: evolution (fixed), or learned (homeostatic RL learns them)?
- Moral preferences (fairness, harm): are they nodes, vetoes, or learned values over social nodes? The Village
  dilemmas test this.
