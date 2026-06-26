# Cipher's TBT Agent

A from-scratch AI agent built on the **Thousand Brains Theory** of the neocortex.

It is **not** a neural network in the usual sense, and it is **not** trained by gradient descent on a large
dataset. Instead, one small, reusable *cortical column* learns the **structure** of whatever it is given — a
space, a number line, a game's rules, a language's words — as a geometry it can navigate. The same column,
composed into a small "neocortex," plays games, does exact arithmetic, and models language, with no
task-specific code.

> **Honest framing.** This is a **novel if limited** research prototype — not artificial general intelligence.
> It is early-stage and small-scale. It is interesting because of *how* it works and *what it suggests*: it
> learns the rules of an unfamiliar game by playing it, generalizes arithmetic to numbers it has never seen,
> and finds its way through a world it can only partly observe — all without hand-coded rules. Those are
> genuine signs of *fluid* intelligence, on a small scale.

## The core idea: one mechanism, many domains

A **cortical column** does one thing: *learn a map of some structure, then predict by navigating that map.*
Everything else is the same mechanism applied to different inputs. A column has four parts (loosely mirroring
cortical layers):

- **Where (L6) — a location code.** It places what it sees into a coordinate frame computed from how things
  connect. Mathematically this is the *successor-representation eigenbasis* — the same code grid cells use in
  the brain. It comes out grid-like for open space, ring-like for a cycle, and correct even for a branching
  tree, all from one rule.
- **How (L5) — displacement operators.** Movement is a first-class object: applying an operator moves you
  through the frame. "Add 5" is just stepping the successor operator five times.
- **What (L4) — a content codebook.** What sits at each location (a digit, a word, a tile).
- **Memory (L23) — an object store.** Bound "what-is-where" facts, recalled later.

Because structure is learned as geometry, the *same* column learns a number line (and does arithmetic by
walking it), a 2-D grid (and navigates it), a word-transition graph (and gets a word-embedding-like geometry),
or a game's state space.

## Memory: a state that knows where it is

The location code can be driven as a **recurrent state** — a selective, gated integrator (the same idea as a
modern state-space model like Mamba). This lets the agent **path-integrate**: it knows where it is from its
own movements, without seeing its absolute position, and it remembers what it has seen. Under partial
observation — when it sees only a small window around itself — this is what lets it build a map and navigate;
a memoryless agent simply cannot.

## Composition: a small neocortex

Hard tasks need more than one column. A **task** column (the sub-goals) and a **space** column (the map) are
composed *additively* through a **thalamus** (which routes a goal from one column into the other) and a
**basal-ganglia gate** (which decides which column handles which structure — the roles *emerge*, they are not
assigned). Keeping the columns separate and additive avoids the combinatorial blow-up of cramming everything
into one model.

## What it can do (so far)

None of these use hand-coded, task-specific rules — the agent learns the rules from experience.

- **Plays a game it has never seen.** On a replica of an [ARC-AGI-3](https://arcprize.org/arc-agi/3)-style game
  ("LockPath"), it learns the dynamics from playing — that a key opens doors, that blocks must be pushed onto
  pads, that the goal only counts once the pads are covered — and then **solves all four levels on every seed,
  near-optimally** (96.5% on an action-efficiency proxy). It decomposes the task into sub-goals and navigates
  each, never searching the whole joint state.
- **Models language.** From only a small classical-Latin and Middle/Old-High-German corpus, the *same column
  mechanism* produces a coherent word geometry (related words land near each other) and a working next-token
  model that **beats n-gram baselines exactly where data is sparse** — the generalization regime. With a
  recurrent state it discovers, from raw prediction alone, that prepositions reset context while particles
  carry it.
- **Does exact arithmetic.** It learns a number line from scratch and adds by navigation — including place
  value and carry — and **generalizes to numbers it never saw** (trained on single digits, correct on 8-digit
  sums).
- **Navigates the partly-unseen.** Seeing only a small window around itself, it path-integrates its position
  and remembers the map, solving levels a memoryless agent fails — and far more efficiently.

## Honest limitations

- **Small scale.** Tiny corpora, small grids, a single 4 GB GPU. Results are demonstrations, not benchmarked at
  scale.
- **Perception is hand-fed.** The agent currently receives clean symbolic input (objects with known roles).
  Turning raw pixels into objects — real perception — is the main unbuilt piece and the next step toward
  generality.
- **Single results.** Most numbers come from one configuration, not extensive sweeps.
- **No general-intelligence claim.** A promising, limited research direction.

## Repository layout

```
corpora/                               the language data (Latin, Middle/Old High German)
experiments/RecurrentWorldModel/
  tbt/                                 the cortical column — the core mechanism
    column.py                          the column (L6 / L5 / L4 / L23)
    recurrence.py                      the selective recurrence (the memory)
    thalamus.py, basal_ganglia.py      multi-column routing + emergent allocation
    RESEARCH.md, THALAMO_CORTICAL_ARCHITECTURE.md   how and why it works
  precursor/                           runnable demos (number line, arithmetic, language, control loop, ...)
experiments/ProgramSynthesis/
  agent/column/                        the TBT agent playing the ARC replica
  agent/wm/                            the predecessor symbolic agent, kept as a reference (see below)
  arc_agi_3/                           the LockPath game replica
```

The directory names `RecurrentWorldModel` and `ProgramSynthesis` are **historical** — the experiment folders
this work grew up in. They are kept to avoid breaking the code's import paths.

## Getting started

```bash
python -m venv venv && . venv/Scripts/activate      # or: source venv/bin/activate   (Linux/macOS)
pip install -r requirements.txt

# Watch the agent learn an ARC-style game from scratch and solve all 4 levels:
cd experiments/ProgramSynthesis && python -m agent.column.multicolumn_agent

# The same column does exact arithmetic by navigating a learned number line:
cd ../RecurrentWorldModel && python -m precursor.arithmetic

# ...and models language (Latin + German):
python -m precursor.language_recurrent

# Memory under partial observation (path integration + remembered map):
cd ../ProgramSynthesis && python -m agent.column.recurrent_agent
```

## The predecessor: a symbolic world-model agent

`experiments/ProgramSynthesis/agent/wm/` holds the agent's direct ancestor: a hand-written **symbolic**
world-model agent that already solved the same ARC replica (perceive → induce the rules → infer the goal →
plan), winning all four levels from frame + score alone. It is kept as a reference for how these ideas
developed — the TBT column grew out of it — and it still provides the scoring used to evaluate the column
agent.

## Credits & license

By **Cipher** (CyclicCipher). Built on the Thousand Brains Theory (Hawkins / Numenta) and the
successor-representation view of grid cells (Stachenfeld et al.). License: TBD.
