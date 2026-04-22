# PFC Research — In Context of Thousand Brains Theory

## The Key Insight: PFC Shares the 6-Layer Blueprint But Has Real Differences

Mountcastle proposed a uniform cortical circuit. 2024 PNAS research
confirms [common modular architecture across diverse cortical areas
during early development](https://pmc.ncbi.nlm.nih.gov/articles/PMC10945769/).
But the MATURE PFC is NOT identical to sensory cortex. The 6-layer
blueprint is shared, but the details differ significantly.

### Structural differences from sensory cortex

**1. Layer 4 varies dramatically across PFC.**
Sensory cortex has a thick, well-developed layer 4 (granular cortex)
packed with stellate cells for receiving thalamic input.
PFC ranges from [granular (DLPFC) to dysgranular to agranular
(ACC, orbitofrontal)](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2024.1389110/full)
— a gradient from full L4 to absent L4. This means different PFC
regions receive thalamic input very differently from sensory cortex.

**2. PFC pyramidal neurons are far more complex.**
PFC pyramidal cells have [up to 23 times more dendritic spines
than V1 pyramidal cells](https://academic.oup.com/cercor/article/13/11/1124/274054).
Larger and more complex basal dendrites, more extensive branching.
These extra spines = more synaptic inputs per neuron = more
integration capacity. PFC neurons can combine more information
streams than sensory neurons.

**3. PFC pyramidal neurons grow spines; sensory neurons prune them.**
V1 pyramidal cells prune more spines than they grow during development.
PFC pyramidal cells [grow more than they prune]
(https://carta.anthropogeny.org/moca/topics/prefrontal-cortex-pyramidal-cell-morphology).
PFC connections INCREASE with experience; sensory connections get
refined (sharpened) with experience. Different developmental trajectory.

**4. PFC has slower, stronger NMDA receptors.**
NMDA receptor-mediated currents at PFC synapses have a [twofold longer
decay time than V1](https://www.pnas.org/doi/10.1073/pnas.0804318105).
Higher NR2B subunit expression in PFC. These slow kinetics are
ESSENTIAL for persistent firing — the sustained activity that
underlies working memory. [NR2B-containing NMDARs in PFC are
essential for working memory across mammalian species]
(https://pmc.ncbi.nlm.nih.gov/articles/PMC12501000/).

**5. PFC has greater electrophysiological diversity.**
PFC neurons show [greater electrophysiological diversity than neurons
from other cortical areas](https://www.jneurosci.org/content/39/37/7277),
including multiple subtypes within layer 5 with distinct projection
targets. This diversity may enable the more complex computations
performed in PFC.

**6. Recurrent connectivity is stronger in PFC.**
PFC has denser recurrent excitatory connections between pyramidal cells
in L2/3, which sustain persistent firing. These recurrent loops are
what make working memory possible — [persistent firing arises from
recurrent excitation within a network of pyramidal Delay cells]
(https://pmc.ncbi.nlm.nih.gov/articles/PMC3584418/).

### What this means for CipherNet

The PFC prior CANNOT be identical to arithmetic columns. It needs:
- **Stronger recurrence** (higher-weight self-reinforcing edges at L2/3)
  to model the slower NMDA dynamics and persistent activity
- **More input connections** (more edges into L4) to model the
  23x greater spine density
- **Variable layer 4** (some PFC subregions have reduced L4,
  meaning they get less direct thalamic/feedforward input and more
  top-down/lateral input)
- **Multiple L5 subtypes** (different output targets for different
  aspects of executive control)

The 6-layer blueprint is shared. But the WEIGHTS, CONNECTION DENSITY,
TIME CONSTANTS, and LAYER THICKNESS are different. The PFC column
is a modified version, not a carbon copy.

## What Makes PFC Different: Connectivity AND Internal Properties

### Different inputs

Sensory columns receive input from sensory organs (eyes, ears, skin).
PFC columns receive input from OTHER CORTICAL COLUMNS. The PFC's
L4 layer gets its input not from the thalamic relay of a sensor,
but from the L5 outputs of other cortical areas.

This is the key: **PFC columns model the OUTPUTS of other columns,
not the external world directly.** A PFC column learns models of
what other columns are doing, just as a visual column learns models
of what the retina is seeing.

Ref: [Prefrontal cortex connectivity to most cortical and subcortical
regions](https://www.nature.com/articles/s41386-021-01132-0)

### Different connectivity patterns

PFC has denser and more diverse long-range connections than sensory
cortex. Adjacent PFC columns can belong to entirely different networks.
This high connectivity is what enables:

- **Integration**: combining information from many different cortical
  areas into a coherent representation
- **Broadcast**: sending control signals back to many cortical areas
  simultaneously
- **Recurrence**: forming loops with basal ganglia and thalamus that
  create persistent activity (working memory)

Ref: [Executive functions from fine-grained cortical architecture,
Cerebral Cortex 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10839840/)

### Working memory from the same circuit

PFC working memory uses the same reverberatory mechanisms as any
cortical column's persistent activity:

- L2/3 horizontal connections between neurons with similar tuning
  create self-sustaining loops
- NMDA receptor slow time constants maintain depolarization
- Cellular bistability (from the column's intrinsic circuit) creates
  stable on/off states

The DIFFERENCE is that PFC columns maintain representations of
ABSTRACT states (goals, rules, intermediate results) rather than
sensory features. But the mechanism is identical.

Ref: [PFC minicolumns and executive control](https://pmc.ncbi.nlm.nih.gov/articles/PMC4065017/)

### Executive function = modeling other columns

In TBT terms: the PFC is a set of cortical columns that build
models of OTHER COLUMNS' behavior. Just as a visual column builds
a model of a cup by sensing features at locations, a PFC column
builds a model of a TASK by sensing column outputs at time steps.

- **Working memory**: a PFC column holding the output of another
  column in its persistent state
- **Sequencing**: a PFC column that has learned a temporal model
  of which column outputs should follow which
- **Task switching**: different PFC columns modeling different tasks,
  with the most active one controlling behavior (voting/competition)
- **Gating**: the basal ganglia loop decides which PFC columns
  update their state (via Go/NoGo — see PFC_PLAN.md)

## Implications for CipherNet PFC Design

### The PFC is a MODIFIED column whose inputs are other columns' outputs

```
Sensory column:              PFC column:
  L4 ← sensory organ           L4 ← other columns' L5 outputs (many more inputs)
  L2/3: model of object        L2/3: model of TASK (stronger recurrence, slower decay)
  L5 → motor/other columns     L5 → control signals (multiple subtypes, diverse targets)
  L6 → prediction error        L6 → prediction error
```

The PFC column's "object" is the task. Its "features" are the
outputs of arithmetic columns at each step. Its "locations" are
time steps in the task. Its "displacement" is "after this step,
the next column output should be X."

The MECHANISM is the same (displacement learning, predictive coding).
The PARAMETERS are different (stronger recurrence, more inputs,
slower dynamics, greater diversity).

### The PFC learns tasks SIMILARLY to how sensory columns learn objects

A sensory column learns a cup by:
1. Sense brown at location (3,5,2)
2. Move to (3,5,4), sense brown with higher curvature
3. Build a model: brown at (3,5,2) → move(0,0,2) → brown-curved at (3,5,4)

A PFC column learns "3 + 7 + 1 = 11" by:
1. Observe: addition column outputs 10 (from 3+7)
2. Time step: next input is 1
3. Observe: addition column outputs 11 (from 10+1)
4. Build a model: output(10) → next_input(1) → output(11)

The PFC model is a temporal graph of column outputs at time steps,
exactly as a sensory model is a spatial graph of features at locations.

### Working memory = persistent activity in PFC L2/3

When the addition column outputs 10, the PFC column's L2/3 maintains
this value through persistent activity (the same bistability that any
column uses). The value stays active while the next input is being
processed. This IS working memory — no special mechanism needed.

The gating question (when to update vs maintain) is handled by the
basal ganglia connection — same as in the PBWM model. But the
column itself doesn't need to know about gating. It just has persistent
states that either update (when gate is open) or hold (when gate is
closed).

### PFC prior structure

The PFC prior shares the 6-layer blueprint but with modifications
reflecting the real biological differences:

1. **A set of MODIFIED columns** (working memory stripes) with:
   - Higher-weight L2/3 recurrent edges (modeling stronger NMDA recurrence)
   - More L4 input nodes (modeling the 23x spine density increase)
   - Self-reinforcing edges at L2/3 with slow decay (modeling NR2B
     NMDA dynamics — the "bistability" that sustains working memory)
   - Multiple L5 output types (modeling PFC's diverse projection targets)

2. **Input connections** from OTHER columns' L5 outputs to PFC L4.
   More input edges per column than sensory columns have.

3. **Output connections** from PFC L5 to other columns' L4 (control
   signals) AND to basal ganglia analog nodes (gating).

4. **Gating nodes** (basal ganglia analog) connected to each stripe.
   Go/NoGo competition controls whether a stripe updates or holds.

5. **Lateral connections** between PFC stripes — mutual inhibition
   (modeling the competition between task representations).

6. **Variable L4** — some PFC stripes may have reduced direct
   feedforward input (dysgranular/agranular pattern), relying more
   on recurrent and top-down connections.

The PFC prior is a JSON file, same format as ANS, same loader.
The differences from an arithmetic column are in BOTH the connectivity
(where edges go) AND the internal parameters (edge weights, decay
rates, number of inputs).

## PFC JSON Structure (Draft)

```json
{
  "name": "pfc",
  "nodes": [
    -- Working memory stripe 0 (standard column) --
    {"id": "wm0:L4", "layer": 4, "role": "input"},
    {"id": "wm0:L23", "layer": 23, "role": "process"},
    {"id": "wm0:L5", "layer": 5, "role": "output"},
    {"id": "wm0:L6", "layer": 6, "role": "feedback"},
    {"id": "wm0:gate", "layer": 23, "role": "gate"},

    -- Working memory stripe 1 --
    {"id": "wm1:L4", "layer": 4, "role": "input"},
    {"id": "wm1:L23", "layer": 23, "role": "process"},
    {"id": "wm1:L5", "layer": 5, "role": "output"},
    {"id": "wm1:L6", "layer": 6, "role": "feedback"},
    {"id": "wm1:gate", "layer": 23, "role": "gate"},

    -- Sequencer (another column that models task order) --
    {"id": "seq:L4", "layer": 4, "role": "input"},
    {"id": "seq:L23", "layer": 23, "role": "process"},
    {"id": "seq:L5", "layer": 5, "role": "output"},
    {"id": "seq:L6", "layer": 6, "role": "feedback"}
  ],
  "edges": [
    -- Standard column wiring for each stripe --
    ...
    -- Gate controls update of L23 --
    {"source": "wm0:gate", "target": "wm0:L23", "type": "temporal", "weight": 1.0},
    -- Stripe-to-stripe lateral connections --
    {"source": "wm0:L5", "target": "wm1:L4", "type": "temporal"},
    -- Sequencer observes stripe outputs --
    {"source": "wm0:L5", "target": "seq:L4", "type": "temporal"},
    {"source": "wm1:L5", "target": "seq:L4", "type": "temporal"}
  ]
}
```

The connections from PFC to arithmetic columns (and from arithmetic
columns to PFC) are specified in the MAIN config.json connections
section, not inside the PFC prior itself. This keeps the PFC
modular — it doesn't know which specific columns it will control.

## Open Questions

1. **How many WM stripes?** Humans have ~4. Start with 2 (enough
   for binary operations with one accumulator and one operand).

2. **How does the sequencer learn task structure?** It's a column
   that models temporal sequences of other columns' outputs. This
   is exactly what HTM sequence memory does. The sequencer's L2/3
   learns "after output A, the next output is usually B."

3. **How does gating actually work in graph terms?** The gate node
   modulates whether L4 input reaches L2/3. When gate=open, the
   temporal edge from L4→L23 has high weight (input passes through).
   When gate=closed, the edge weight is low (input blocked, L2/3
   maintains its current state). The basal ganglia analog learns
   to control the gate weight.

4. **How do we test this?** The simplest test: "3 + 7 + 1 = 11".
   The PFC must: (a) route 3,7 to addition column, (b) hold the
   result 10 in WM, (c) route 10,1 to addition column, (d) output 11.

## References

- [Common modular architecture across cortical areas, PNAS 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10945769/)
- [PFC minicolumns for executive control, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4065017/)
- [PFC role in cognitive control, Nature 2021](https://www.nature.com/articles/s41386-021-01132-0)
- [Executive functions from cortical architecture, Cerebral Cortex 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10839840/)
- [Hawkins grid cell framework, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6336927/)
- [Thousand Brains Project 2024, arXiv](https://arxiv.org/html/2412.18354v1)
