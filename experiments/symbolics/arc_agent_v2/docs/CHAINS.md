# Chained expressions: the research before the design (2026-10-29)

The next step after A2 (ARCHITECTURE.md §7) is chained expressions, "a + b + c =". The user asked for research first:
1. working short-term memory;
2. the thalamus as an active gate, where the columns plug into it, and how the PFC does its job mechanically;
3. how brains solve chained expressions;
4. how these interact without exponential blowups.

The user also asked whether the brain implements anything like an LLM's attention (§5), and suggested a `thalamus.py`
as the complement to `bus.py` (§6).

Nothing here is designed yet. §6 lists proposals for the user to choose from.

## 1. What fails in v2 today

After A2, the brain answers "a op b =" for any a, b and op. Here is what it would do with "a + b + c =" (from the code,
not run):

1. **After "a + b"** the column is at the result place C_r (r = a + b) with the factored state (phase C, coordinate r).
2. **The next "+":**
   - **The place move:** L5 has no move ("+" was never read at C_r), so the column's place is lost.
   - **The phase move:** the phase automaton has no transition from C by "+", so the factored state is lost too.
   - **The fallback:** the hippocampus never lived through the prefix, so it predicts nothing.

So the answer is wrong by construction, at every length above one operation.

The coordinate part would already be right: the running sum is the coordinate. "+ c" would shift it by c, exactly as
after "a". What is missing is control: knowing that after a result, an operator continues from that result as if it
had just been read as a digit. Two routes to that come up below:
- **Generalization:** the place after a result is treated as the place after a digit (§6, proposal A).
- **A held value routed into an argument slot:** working memory (§6, proposal B).

## 2. How brains chain operations

**Chaining is serial, effortful and conscious, even when each step is not.**
- **The experiment** (Sackur & Dehaene 2009): participants added or subtracted 2 from a digit, then compared the result
  with 5.
- **Each operation alone can run on a subliminal (unseen) digit; their chaining does not.** It needs conscious
  perception, is slow and error-prone, and costs time beyond the classic dual-task costs (Fan et al. 2012).
- **The overlap:** the second operation starts before the first has fully finished, so the chain is not strictly
  serial.

**The "human Turing machine"** (Zylberberg, Dehaene, Roelfsema & Sigman 2011): a mental program is a loop of two
phases.
- **Parallel processing** within each step: perception, memory retrieval, an arithmetic fact.
- **A slow serial production system** that chooses the next step. Productions compete as accumulators (ramping neurons
  racing to a threshold); the winner inhibits the rest, so one production runs at a time.
- **The program:** a production's effect is to change sensory and memory neurons, which sets up the next competition.
  Iterating the cycle is the program.

**A circuit for multi-step computation** (Zylberberg, Paz, Roelfsema, Dehaene & Sigman 2013, arXiv 1312.6660) makes
this concrete:
- **Pointers:** neurons that bind to a value in a value domain (a network of number-selective neurons) by transient
  synaptic strengthening. The binding is "activity-silent" and decays in a few hundred milliseconds, so it must be
  refreshed.
- **Retrieval:** activate the pointer and put the value domain into winner-take-all mode. The strengthened synapses
  bias the winner to the bound value.
- **Assignment (Y ← X):** retrieve X, then activate Y, which binds to the value now active.
- **Comparison:** two pointers co-activated; a coincidence-detecting value neuron fires only if both point to it.
- **Its counter** ("count from 2 to 6") cycles six productions: prepare the successor, rebind Count to it, clear,
  refresh End, compare, halt or repeat.
- **Its limits:**
  - one binding per value domain at a time, which forces seriality;
  - the successor table and the productions' order are hand-wired ("all parameters were set by hand"), and their
    learning is not addressed.

**Nested expressions cost by depth, not by combinations.**
- **Nakai & Sakai 2014:** activity in left IFG and left IPS/SMG while computing nested arithmetic without writing or
  speaking partial results is best explained by the Degree of Merger, roughly the number of pending sub-results. The
  same network handles nested sentences; mathematics adds both IPS.
- **Maruyama et al. 2012:** the cortex represents an expression's constituent structure, not only its symbols.
- **Dehaene et al. 2022** frame this as "mental programs": compressed, recursive descriptions in a language of thought.

**What this means for v2:**
- One operation at a time, with intermediate results in a small held store, reused by the same machinery at every
  step.
- The cost grows with the number of pending sub-results (the depth), not with the number of possible expressions.

## 3. Working memory

**Where it is held:**
- **Persistent activity in thalamocortical loops.**
  - **Guo et al. 2017:** a premotor cortex (ALM) and thalamus loop sustains the delay activity; silencing either part
    collapses it.
  - **Bolkan et al. 2017:** projections from the mediodorsal thalamus (MD) sustain prefrontal activity during
    maintenance.
  - **Schmitt et al. 2017:** MD doesn't relay the rule's content; it amplifies the PFC's own local connectivity, which
    keeps the rule active.
- **Activity-silent synaptic traces.** Short-term synaptic facilitation holds items for about a second without firing
  (Mongillo, Barak & Tsodyks 2008; Stokes 2015). This is the binding Zylberberg et al. 2013 use for pointers.

**Capacity:** about 4 chunks (Cowan 2001). Nothing in the brain keeps an unbounded context in active memory.

**What gets in, what stays, what is read** (PBWM, prefrontal cortex and basal ganglia working memory; O'Reilly & Frank
2006):
- **Stripes:** PFC is divided into "stripes", patches that each hold one item.
- **The basal ganglia's three gates:**
  - **input gating:** a stripe updates to new content;
  - **maintenance:** it holds;
  - **output gating:** its content is released to drive processing elsewhere.
- **Learned, not wired:** the gating policy is learned by reinforcement (dopamine). The BG decides when and where; the
  PFC holds what.

**Indirection: pointers between stripes** (Kriete, Noelle, Cohen & O'Reilly 2013):
- **Roles hold addresses:** role stripes (agent, verb, patient) hold the address of the filler stripe that holds the
  word, not the word itself.
- **Recall:** output-gate the role stripe; the address it releases output-gates the filler stripe.
- **The test:** 10 words in 3 roles (1,000 sentences), trained on 200. In the hardest test, 2 words never appear as
  patients in training.
- **The result:** only the indirection model generalized in all tests. Maintenance-only and output-gating PBWM failed
  the combinatorial tests, and a simple recurrent network struggled on all.
- **Its limits:**
  - pointers only to locations it has experience with;
  - "neural pointers cannot be nested at arbitrary levels of complexity or depth";
  - extensive training.

A related machine-learning model separates control from content the same way: Webb, Sinha & Cohen 2021, "emergent
symbols through binding in external memory".

## 4. The thalamus as an active gate; the PFC's mechanics

The TBT line's July review (`src/tbt/notes/bg_thalamus_value_research.md`, thread 3) found the thalamus is a gated
router, not a relay, and that it mostly doesn't learn: the routing decisions are learned upstream and applied there. The
working-memory papers add what that gate does for control.

**Where columns plug in** (Sherman & Guillery; Halassa & Kastner 2017):
- **L6 corticothalamic cells → the thalamus and its reticular nucleus (TRN):**
  - **The synapse type:** a modulator (many small synapses that set gain), not a driver.
  - **What it gives a column:** feedback control over its own input relay.
- **L5 pyramidal-tract cells → higher-order thalamus → other cortical areas (transthalamic):**
  - **The synapse type:** a driver, carrying the message.
  - **What it is:** the route by which one area's output (and its efference copy) drives another.
- **Core and matrix thalamus:**
  - **core:** specific, to the middle layers (the input);
  - **matrix:** diffuse, to layer 1 and the apical dendrites (context, binding).
- **Basal ganglia output (GPi/SNr) → ventral, motor and mediodorsal thalamus:**
  - **The default:** these outputs inhibit the thalamus tonically, so every channel is blocked.
  - **Selection:** a selected channel is released by disinhibition.

**What the thalamus does for control:**
- **It selects which sensory stream passes.** In divided attention, PFC (via the basal ganglia) drives the TRN to
  suppress the irrelevant modality's relay (Wimmer et al. 2015).
- **It sustains the relevant context and suppresses the others.**
  - **MD's two subsets** (Rikhye, Gilra & Halassa 2018): one sustains the context-relevant PFC representations, the
    other suppresses the irrelevant ones.
  - **The function:** MD activity tracks which cueing context is in force, and is needed to switch between contexts.
  - **A side effect:** suppressed traces are protected from being overwritten.
- **It sustains working memory** (§3): thalamocortical loops keep the PFC's delay activity alive.

**How the PFC acts on the rest of the brain** (Miller & Cohen 2001, "an integrative theory of prefrontal cortex
function"):
- **Its content:** the PFC holds goals and rules as patterns of activity.
- **Its effect:** those patterns bias the competition in posterior areas, so the right pathway wins ("laying down
  tracks").
- **Its commands are biases and gates, never the content of the operation:**
  - **top-down:** feedback to layer 1 and the apical dendrites of posterior areas (a gain change);
  - **into the basal ganglia:** the striatum's input, which selects what enters or leaves working memory, and which
    action or step runs;
  - **onto the TRN, via the basal ganglia:** what passes.

**For v2:**
- **The operation stays in the columns** (L5's displacement in L6's frame).
- **What is new for chains is control:**
  - holding a value;
  - routing the held value into a column's argument slot (L4, which ARCHITECTURE §1 calls the place or argument
    slots);
  - choosing when to update and when to read.
- **The division of labour** in the literature:
  - the BG learns the choice;
  - the PFC holds the rule and the held values;
  - the thalamus applies the gates.

## 5. Does the brain implement attention as in an LLM?

**Not as one operation over a stored context.** An LLM keeps every past token in parallel and, at each step, weights
all of them by query·key similarity. The brain holds about 4 items in active memory (§3), and the rest of its past is
compressed into state or recalled on demand. But each part of attention has a counterpart.

**Content-addressed recall: the softmax(QK)V part.**
- **Hippocampus as key-value memory** (Gershman, Fiete & Irie 2025, *Neuron*):
  - **keys:** the medial temporal lobe holds them, optimized to tell memories apart;
  - **values:** the neocortex holds them, optimized for fidelity;
  - **forgetting:** often a failure to address a memory, not its loss.
- **The keys are positions, path-integrated** (Whittington, Warren & Behrens 2022, ICLR): TEM, a model of the
  hippocampal formation, is mathematically a transformer whose position encodings are updated recurrently by actions.
  With that, place and grid cells emerge. The brain's "positional encoding" is a grid code moved by each action, which
  is v2's L6.
- **Sparse distributed memory** (Bricken & Pehlevan 2021, NeurIPS):
  - **The read:** Kanerva's SDM reads by activating every stored address within a Hamming radius of the query and
    summing their contents.
  - **The match:** that approximates softmax attention, with the radius in the role of the temperature, and the
    conditions hold in trained GPT-2.
  - **In v2:** `Associations.recall` (exact, else overlap ≥ θ) is a hard-threshold form of the same read.
- **Attention is one step of a modern Hopfield network** (Ramsauer et al. 2021).
- **Induction heads behave like human free recall** (Ji-An, Zhou, Benna & Mattar 2024, NeurIPS):
  - **The match:** the LLM heads that copy from context behave like CMR (context maintenance and retrieval), the
    standard model of human free recall.
  - **CMR's key:** a slowly drifting temporal context.
  - **The evidence:** ablating these heads hurts in-context learning.
- **The softmax's competition:** divisive normalization and k-winners-take-all inhibition (the normalization model of
  attention, Reynolds & Heeger 2009). v2's `kwta` is the hard version.

**Fast weights: attention's memory of the context.**
- Linear attention is formally a fast-weight memory (Schlag, Irie & Schmidhuber 2021).
- Short-term synaptic plasticity can store bindings for about a second (Mongillo et al. 2008).
- Short-term Hebbian plasticity can implement transformer-like attention (Ellwood 2024).
- Zylberberg et al. 2013's pointers are exactly this.

**Routing: what neuroscience calls attention.**
- **What it decides:** which inputs reach which targets.
- **How:** TRN gating (Crick's "searchlight"), BG disinhibition, MD sustaining one context (§4), and the "dynamic
  routing" circuits proposed for invariant recognition (Olshausen, Anderson & Van Essen 1993).
- **How it differs from an LLM's attention:** the weights are near 0 or 1 and set by the task (PFC/BG), not by
  similarity of content.

**What has no counterpart:**
- a parallel weighting over thousands of past tokens at every step;
- a step that reads everything at once. The brain's operations on working memory are serial (§2), which is what keeps
  them from blowing up.

**In v2 today:**
- **What we have:** the hippocampus is a key-value store (the DG place is the key, what followed is the value), read by
  overlap, the SDM form of attention.
- **What chains need:** the routing kind, putting a held value into a column's argument slot at the right step.

## 6. Interactions without blowup, and what it suggests for v2

**Where the blowups would be:**
1. **Memorizing chains:** p^n expressions of n operands.
2. **A phase per chain position:** the automaton grows with the length, and never reaches lengths it hasn't seen.
3. **Searching over parses:** the bracketings of n operands grow as the Catalan numbers.

**How the brain avoids them:**
- **Reuse:** the same operation machinery is reused at every step (one production at a time).
- **A small, fixed store:** intermediate results are held in a small store and rebound, not given new places.
- **One left-to-right pass:** the cost of nesting is the number of pending results, bounded by working memory, not the
  number of parses.

**For v2:** the control automaton must loop, and the running result must sit in a slot that the next step reads.

**Proposals for the user to choose from** (not designed yet):

**A. Chains of one precedence level ("a + b − c ="), with no new memory.**
- **The register already exists:** the running result is the coordinate (L6's place).
- **What is missing:** the phase after a result must continue under an operator as the phase after a digit does.
- **The rule:** P3 lifted to phases. Two phases whose moves and contents never disagree, read as functions of the
  coordinate, are one phase.
- **The bold step:** the phase after a digit and the phase after a result have the same future under an operator,
  except that only the result has seen "=". So they merge, generalize-then-correct. The automaton becomes a loop:
  digit → operator → digit → … → "=".
- **The data question:** does it need chains of length 2 in training, or none?
  - **None:** a zero-shot claim, the stronger test. "+" after a result is never seen.
  - **With chains of length 2:** test on longer chains.

**B. Precedence and parentheses ("a + b × c", "(a + b) × c") need a second held value.**
- **What has to happen:** the pending "a +" must wait while "b × c" is computed.
- **Where it lives:** this is where working memory and the thalamus come in.
- **`thalamus.py` (the user's suggestion), as the complement to `bus.py`:**
  - `bus.py` is the transport: every message delivered once, stamped, by receptive field (the anatomy).
  - `thalamus.py` would be the active gate on that transport, with three jobs from §3–4:
    1. **holding:** values kept by a loop with a column, as MD sustains PFC activity;
    2. **gating:** update, hold, or read for each held value (PBWM's input and output gates);
    3. **routing:** a held value into a column's argument slot, by address (indirection).
  - **What it does not do:** decide. In the literature, the policy is learned by the BG from reward; the thalamus
    applies it.
- **The hard rule's demand:** under ARCHITECTURE §0, nothing may say "the result goes to the first argument". Which
  value to read when must be learned.
- **In a world with no reward:** the candidate signal is v2's own prediction error. A gate choice that makes the next
  symbol predictable is reinforced (`src/tbt/notes/bg_thalamus_value_research.md`: the BG's three-factor rule).

**C. Order:** A first. It is cheap, uses only what exists, and tests whether "a result is a value like a digit" can be
learned without being shown. Then B, with `thalamus.py` built when the first expression with precedence needs it
(vertical slices: nothing counts until it is wired and exercised).

**Questions for the user:**
1. **For A:** chains in training or not? And may the test lengths exceed the trained ones?
2. **For B:** standard precedence (× before +), or left-to-right with parentheses only?
3. **The gate's learning signal:** prediction error alone, or a reward the world gives?

## 7. The user's choices (2026-10-29), and what designing A found

**The choices:**
- **A first:** compare one chain in training with none (LAB part 7).
- **Then B:** standard precedence.
- **Fix the hippocampus's gaps** (§10).

**Zero-shot may work.**
- **The evidence it rests on:** part 6 found that the start and "after +" share every digit's shift (one shift class).
- **The rule:** P3 lifted to phases ("same future, same phase", with an input's shared shift as the evidence). It
  merges those two phases.
- **The consequence:** a digit read from either must then lead to one phase, so "after a digit" and "after a op b"
  merge too.
- **The result:** the automaton loops, and no chain need be seen.

**One chain is the harder case.**
- **The split:** the chain "a + b − c = d" gives the place after "a + b" a future ("−") that the other places of
  result a + b never had. Strict P3 then keeps it apart from them.
- **Full P3 can't merge it either:** its content ({=, −}, or "−") identifies nothing.
- **The collisions:** a translation t then maps both it and the merged place of the same result to the place of
  result a + b + t: two places to one. P4 refutes every such translation. With about 48 digits trained after "a +" at
  p = 97, that is about half of "+"'s displacements.
- **So:** a collision must first be tested as a merge (P3's fold), and only then refute.

The chain's own places also need naming to be placed in the right shift class.

## 8. Can local circuits chain, and what holds the state?

**Chaining moves within one frame is local.**
- **Path integration is chained addition:** a sum of displacements, kept by the entorhinal grid system without the PFC
  (McNaughton et al. 2006).
- **A practiced, fixed sequence is local too:**
  - songbird HVC runs a learned chain of neurons (Long, Jin & Fee 2010);
  - practiced motor sequences are chunked in striatum and premotor cortex (Graybiel 1998).
- **Arithmetic shifts with practice:** children move from counting procedures to retrieving the facts, with
  hippocampal-neocortical reorganization (Qin et al. 2014).
- **Proposal A is this kind:** the column's coordinate is the running result, as a grid code is the running position.

**The PFC is needed when the chain is novel, or crosses circuits.**
- A result must be held while something else is computed (precedence: proposal B).
- A result must be routed from one circuit into another's argument.
- Each step must be chosen under control (Sackur & Dehaene 2009; Zylberberg et al. 2011).

**Short-term memory is the state; long-term memory is the weights. There is a middle.**

| timescale | what holds it | in v2 |
|---|---|---|
| ms–s | activity: the current state (persistent firing, an attractor) | L6's place and (phase, coordinate) |
| s–min | short-term synaptic change, "activity-silent" (Mongillo et al. 2008; Stokes 2015): fast weights | not built (B's held values) |
| hours–years | the hippocampus's one-shot plasticity, until consolidated | `hippocampus.py`'s stores |
| a lifetime | cortical synapses | L5's and L2/3's tables, installed at sleep |

## 9. Working memory: what it is, its limits, what fills the slots

**What it is:** the few things held active and open to manipulation for seconds.
- **Where:** prefrontal, parietal and sensory cortex, sustained by thalamocortical loops (§3).
- **Who decides what enters and leaves:** the basal ganglia.
- **The focus:** one item at a time is directly operated on (Oberauer 2002).

**How many:**
- Miller's 7 ± 2 (1956) counted chunks, with rehearsal and grouping allowed.
- With those prevented, about 4 (Cowan 2001).
- Visual working memory holds about 3–4 objects, each with all its features (Luck & Vogel 1997).

**Why so few: competing accounts, none settled.**
1. **Interference in one population.** Items held at once compete through lateral inhibition; past a few, they merge
   or one dies (attractor models: Edin et al. 2009).
2. **Time-multiplexing.** Each item fires in its own gamma cycle within a theta cycle, and the ratio of the two
   rhythms is about 4–8 (Lisman & Idiart 1995; Lundqvist et al. 2016 for gamma and beta bursts).
3. **A shared resource, no slots.** Precision is divided among the items and falls as they increase (Ma, Husain &
   Bays 2014).
4. **Binding.** Each item must be bound to a distinct role or position tag, and there are few distinguishable tags.

**What this means for SDRs (my inference, not a source):**
- **Membership is cheap:** a union of k SDRs (w = 40, d = 2048) still tells its members from non-members for large
  k.
- **Roles are not:** which item fills which role is lost unless each item is bound to a distinct role code. The limit
  that matters for chains is the binding, not the storage.

**What goes in a slot: a chunk, a unit of long-term memory.**
- **Examples:** a digit, a word, a familiar phrase, an object with its features bound.
- **Experts hold bigger chunks** (chess: Chase & Simon 1973).
- **Often a slot holds a pointer:** a cue that retrieves long-term content.
  - Cowan's "activated long-term memory";
  - Ericsson & Kintsch's "long-term working memory" (1995);
  - Kriete et al.'s indirection (§3).
- **For B:** a slot would hold a pending result (a coordinate) bound to its role, for example "the left operand of the
  pending +".

## 10. What teaches the gate: the PFC research

**PBWM's gate is trained by dopamine.**
- **The signal:** dopamine prediction errors, from task reward (O'Reilly & Frank 2006).
- **The hierarchy:** rostral PFC's content conditions the gating of caudal PFC, so a rule held higher selects which
  gate opens below (Frank & Badre 2012).

**Dopamine signals more than reward, so a self-supervised signal can train a gate.**
- Dopamine neurons signal errors in predicting the sensory features of outcomes (Takahashi et al. 2017).
- Dopamine as a generalized prediction error (Gardner, Schoenbaum & Gershman 2018).

**The thalamus may be where prediction meets outcome** (O'Reilly et al. 2021, deep predictive learning).
- **The comparison:** every alpha cycle (about 100 ms), the pulvinar gets the cortex's prediction (from L6) and then
  the actual input (an L5 driver).
- **The learning:** the difference trains the cortex.
- **The candidate rule for B:** a gating choice that makes the next symbol predicted is reinforced.

**The medial PFC computes predictions and the value of control.**
- It predicts outcomes and signals their errors (Alexander & Brown 2011).
- It decides how much control to apply: expected value minus cost (Shenhav, Botvinick & Cohen 2013).

**The PFC builds reusable rules.**
- **Task sets:** it clusters contexts into task-sets, and reuses a set in new contexts (Collins & Frank 2013).
- **Meta-RL** (Wang et al. 2018): dopamine trains the PFC's recurrent dynamics, which then learn a new task within
  their activity. The slow weights learn how to gate; the fast state does the gating.

**For v2:** B's gate can be taught by v2's own prediction error, with no reward from the world. This is the user's
option 3 (CHAINS §6), now with a mechanism in the literature.

## 11. The hippocampus's gaps, given attention (to fix: LAB part 8, after A)

**What's built:** the hippocampus is a key-value store. The DG place is the key, what followed is the value, and the
read is by overlap (§5).

**Its gaps against the attention literature:**
1. **Its keys have no structure.**
   - **Today:** a key is the hash of the exact path (the DG conjunction of the previous place and the input), so a
     new path never overlaps an old one. Recall is a lookup table, not attention.
   - **In TEM-t** (Whittington et al. 2022), the keys are the position codes the cortex path-integrates.
   - **The fix:**
     - after sleep, the hippocampus keys what it stores by the cortex's location (the column's (phase, coordinate),
       arriving as the entorhinal input);
     - a new path to a known location then recalls what was stored there;
     - new content can be learned in one shot at a location the cortex already has.
   - **A test:** learn a new answer from one episode, then recall it through a different equation with the same
     result.
2. **No temporal context.**
   - **What's missing:** nothing retrieves an item by when it occurred in the episode (CMR's drifting context, which
     LLM induction heads resemble; Ji-An et al. 2024).
   - **Why it matters:** B could need it to recall a pending operand. Whether B uses the hippocampus or the
     thalamus's held values is B's design question.
3. **No CA1.** Recalled content is never mapped back into the cortex's places. It is not needed until 1 or 2 is.

## 12. What the thalamus learns, and what it does, compared with a column (the user, 2026-10-29)

**The short answer:**
- **A column learns content and operations:** what is where, what comes next, how a move changes a place.
- **The thalamus learns almost no content.** Its learnable parts set how much and when (gains and gates). The choice
  of which gate opens is learned elsewhere: by the basal ganglia from dopamine, and, for timing, by the cerebellum.

**What a column learns:**
- **Feedforward synapses into L4 and L2/3:** features, receptive fields.
- **Lateral and recurrent L2/3 connections:** associations, voting, and sequence memory on distal dendrites (HTM's
  temporal memory).
- **L5 and L6:** the frame and the displacements that move a place in it.
- **Apical feedback synapses in L1:** learned context from higher areas.

**What the thalamus learns:**

| part | what is plastic | what it sets |
|---|---|---|
| driver synapses (retina → LGN; L5 → higher-order thalamus) | mostly fixed in adults (plastic in development) | nothing learned: the message passes as it is |
| modulatory synapses from L6 (corticothalamic) | NMDA-receptor LTP/LTD | the gain of a column's own relay |
| the reticular nucleus (TRN) | short-term plasticity; long-term depression of its electrical coupling (Landisman & Connors 2005) | how strongly, and how in sync, it suppresses relays |
| thalamocortical input to cortex | plastic in the critical period; in adults, higher-order thalamic input paired with sensory input drives LTP in L2/3 (Gambino et al. 2014) | the thalamus helps teach the cortex, rather than learning itself |
| mediodorsal (MD) responses | track the statistics of which context holds (Rikhye, Gilra & Halassa 2018) | which context to sustain; the learning may sit in its PFC inputs |

**What the thalamus does** (its operations, none of which needs content learning):
1. **Relay and route.**
   - **First-order:** sense → cortex.
   - **Higher-order:** a column's L5 output → thalamus → another area. This is the transthalamic route, the efference
     copy's path.
2. **Gate.**
   - **The default:** the basal ganglia inhibit the thalamus tonically, so every channel is closed.
   - **Selection:** the selected channel is released by disinhibition.
   - **The TRN** suppresses the relays not attended to (Crick's "searchlight").
3. **Set gain and mode.**
   - **Gain:** L6 feedback scales its own relay.
   - **Mode:** tonic firing relays faithfully; bursts signal novelty, "wake up" (Sherman 2001).
4. **Sustain:** thalamocortical loops keep the PFC's working-memory activity alive, and amplify the PFC's own
   connections (§3).
5. **Switch context:** MD sustains the relevant PFC representations and suppresses the others (§4).
6. **Synchronize and bind:**
   - **The matrix's diffuse projections** to layer 1 coordinate the timing of areas.
   - **The pulvinar** synchronizes two visual areas according to attention, regulating what passes between them
     (Saalmann et al. 2012).
7. **Compare** (a hypothesis): the pulvinar sets the cortex's prediction against the next input every ~100 ms, and the
   difference trains the cortex (O'Reilly et al. 2021, §10).

**For `thalamus.py`:**
- **It is mostly deterministic operations:** route, gate, hold, switch, synchronize, compare.
- **Its learned state is small:** a gain per channel, and how strongly each gate closes.
- **The policy of which gate opens when** is learned outside it: by a basal-ganglia learner taught by prediction error,
  and later by the goal cortex's drives ("be correct"; ARCHITECTURE §7).
- **This matches** the TBT line's July review (`src/tbt/notes/bg_thalamus_value_research.md`, thread 3).

## Sources

- Alexander & Brown 2011, "Medial prefrontal cortex as an action-outcome predictor", *Nature Neuroscience* 14.
- Bolkan et al. 2017, "Thalamic projections sustain prefrontal activity during working memory maintenance", *Nature
  Neuroscience* 20.
- Chase & Simon 1973, "Perception in chess", *Cognitive Psychology* 4.
- Collins & Frank 2013, "Cognitive control over learning: creating, clustering, and generalizing task-set structure",
  *Psychological Review* 120.
- Edin et al. 2009, "Mechanism for top-down control of working memory capacity", *PNAS* 106.
- Ericsson & Kintsch 1995, "Long-term working memory", *Psychological Review* 102.
- Frank & Badre 2012, "Mechanisms of hierarchical reinforcement learning in corticostriatal circuits 1", *Cerebral
  Cortex* 22.
- Gardner, Schoenbaum & Gershman 2018, "Rethinking dopamine as generalized prediction error", *Proc. R. Soc. B* 285.
- Gambino et al. 2014, "Sensory-evoked LTP driven by dendritic plateau potentials in vivo", *Nature* 515.
- Graybiel 1998, "The basal ganglia and chunking of action repertoires", *Neurobiology of Learning and Memory* 70.
- Landisman & Connors 2005, "Long-term modulation of electrical synapses in the mammalian thalamus", *Science* 310.
- Saalmann, Pinsk, Wang, Li & Kastner 2012, "The pulvinar regulates information transmission between cortical areas
  based on attention demands", *Science* 337.
- Sherman 2001, "Tonic and burst firing: dual modes of thalamocortical relay", *Trends in Neurosciences* 24.
- Lisman & Idiart 1995, "Storage of 7 ± 2 short-term memories in oscillatory subcycles", *Science* 267.
- Long, Jin & Fee 2010, "Support for a synaptic chain model of neuronal sequence generation", *Nature* 468.
- Luck & Vogel 1997, "The capacity of visual working memory for features and conjunctions", *Nature* 390.
- Lundqvist et al. 2016, "Gamma and beta bursts underlie working memory", *Neuron* 90.
- Ma, Husain & Bays 2014, "Changing concepts of working memory", *Nature Neuroscience* 17.
- McNaughton et al. 2006, "Path integration and the neural basis of the 'cognitive map'", *Nature Reviews
  Neuroscience* 7.
- Miller 1956, "The magical number seven, plus or minus two", *Psychological Review* 63.
- Oberauer 2002, "Access to information in working memory: exploring the focus of attention", *JEP: Learning, Memory,
  and Cognition* 28.
- O'Reilly, Russin, Zolfaghar & Rohrlich 2021, "Deep predictive learning in neocortex and pulvinar", *Journal of
  Cognitive Neuroscience* 33.
- Qin et al. 2014, "Hippocampal-neocortical functional reorganization underlies children's cognitive development",
  *Nature Neuroscience* 17.
- Shenhav, Botvinick & Cohen 2013, "The expected value of control", *Neuron* 79.
- Takahashi et al. 2017, "Dopamine neurons respond to errors in the prediction of sensory features of expected
  rewards", *Neuron* 95.
- Wang et al. 2018, "Prefrontal cortex as a meta-reinforcement learning system", *Nature Neuroscience* 21.
- [Bricken & Pehlevan 2021, "Attention approximates sparse distributed memory", NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/8171ac2c5544a5cb54ac0f38bf477af4-Abstract.html).
- Cowan 2001, "The magical number 4 in short-term memory", *Behavioral and Brain Sciences* 24.
- Dehaene, Al Roumi, Lakretz, Planton & Sablé-Meyer 2022, "Symbols and mental programs: a hypothesis about human
  singularity", *Trends in Cognitive Sciences* 26.
- Ellwood 2024, "Short-term Hebbian learning can implement transformer-like attention", *PLOS Computational Biology*.
- [Fan et al. 2012, "The cost of serially chaining two cognitive operations", *Psychological Research* 76](https://link.springer.com/doi/10.1007/s00426-011-0375-y).
- [Gershman, Fiete & Irie 2025, "Key-value memory in the brain", *Neuron* (arXiv 2501.02950)](https://arxiv.org/abs/2501.02950v2).
- Guo et al. 2017, "Maintenance of persistent activity in a frontal thalamocortical loop", *Nature* 545.
- Halassa & Kastner 2017, "Thalamic functions in distributed cognitive control", *Nature Neuroscience* 20.
- [Ji-An, Zhou, Benna & Mattar 2024, "Linking in-context learning in transformers to human episodic memory", NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0ba385c3ea3bb417ac6d6a33e24411bc-Abstract.html).
- [Kriete, Noelle, Cohen & O'Reilly 2013, "Indirection and symbol-like processing in the prefrontal cortex and basal ganglia", *PNAS* 110](https://pmc.ncbi.nlm.nih.gov/articles/PMC3799308).
- Maruyama, Pallier, Jobert, Sigman & Dehaene 2012, "The cortical representation of simple mathematical expressions",
  *NeuroImage* 61.
- Miller & Cohen 2001, "An integrative theory of prefrontal cortex function", *Annual Review of Neuroscience* 24.
- Mongillo, Barak & Tsodyks 2008, "Synaptic theory of working memory", *Science* 319.
- [Nakai & Sakai 2014, "Neural mechanisms underlying the computation of hierarchical tree structures in mathematics", *PLOS ONE* 9](https://pmc.ncbi.nlm.nih.gov/articles/PMC4224410).
- O'Reilly & Frank 2006, "Making working memory work: a computational model of learning in the prefrontal cortex and
  basal ganglia", *Neural Computation* 18.
- Olshausen, Anderson & Van Essen 1993, "A neurobiological model of visual attention and invariant pattern recognition
  based on dynamic routing of information", *Journal of Neuroscience* 13.
- Ramsauer et al. 2021, "Hopfield networks is all you need", ICLR.
- Reynolds & Heeger 2009, "The normalization model of attention", *Neuron* 61.
- [Rikhye, Gilra & Halassa 2018, "Thalamic regulation of switching between cortical representations enables cognitive flexibility", *Nature Neuroscience* 21](https://pmc.ncbi.nlm.nih.gov/articles/PMC7225728).
- [Sackur & Dehaene 2009, "The cognitive architecture for chaining of two mental operations", *Cognition* 111](https://researchportal.ip-paris.fr/en/publications/the-cognitive-architecture-for-chaining-of-two-mental-operations/).
- Schlag, Irie & Schmidhuber 2021, "Linear transformers are secretly fast weight programmers", ICML.
- Schmitt et al. 2017, "Thalamic amplification of cortical connectivity sustains attentional control", *Nature* 545.
- Stokes 2015, "'Activity-silent' working memory in prefrontal cortex: a dynamic coding framework", *Trends in
  Cognitive Sciences* 19.
- Webb, Sinha & Cohen 2021, "Emergent symbols through binding in external memory", ICLR.
- [Whittington, Warren & Behrens 2022, "Relating transformers to models and neural representations of the hippocampal formation", ICLR](https://arxiv.org/abs/2112.04035).
- Wimmer et al. 2015, "Thalamic control of sensory selection in divided attention", *Nature* 526.
- [Zylberberg, Dehaene, Roelfsema & Sigman 2011, "The human Turing machine: a neural framework for mental programs", *Trends in Cognitive Sciences* 15](https://www.unicog.org/publications/ZylberbergDehaeneRoelfsemaSigman_TuringMachine_TICS2011.pdf).
- [Zylberberg, Paz, Roelfsema, Dehaene & Sigman 2013, "A neuronal device for the control of multi-step computations", arXiv 1312.6660](https://ar5iv.labs.arxiv.org/html/1312.6660).
- `src/tbt/notes/bg_thalamus_value_research.md` (this repo, 2026-07-10): the thalamus and basal ganglia review, thread 3.
