# DSNN lab book

One entry per experiment: its pre-registration (written and committed before the code), then amendments (before the
run), then the results. The design is in `DESIGN.md`. Run from `experiments/` with `../venv312/Scripts/python.exe`.

## E0: the neuron (pre-registered 2026-10-30)

**The question:** can a reduced human layer 2/3 pyramidal neuron, with literature parameters and no tuning, reproduce
the six classic dendritic experiments of DESIGN.md §4.1 (N1–N6)? And what does it cost to run?

**The user's decisions (2026-10-30):**
- "2 wide" = 2 pyramidal neurons per layer.
- The target is the classic experiments. A NEURON-based reference model is not part of E0.

### The cell (v0): `cell.py`

**Compartments (15):**

| # | name | parent | length (µm) | diameter (µm) | mechanisms |
|---|---|---|---|---|---|
| 0 | soma | — | lumped: 25,000 µm² of membrane (soma plus the dendrites not modelled) | — | AdEx spiking; the spike's waveform |
| 1–3 | b1.p, b1.m, b1.d (basal branch 1: proximal, middle, distal) | soma, 1, 2 | 60 each | 1.0, 0.9, 0.8 | AMPA + NMDA synapses |
| 4–6 | b2.p, b2.m, b2.d | soma, 4, 5 | 60 each | 1.0, 0.9, 0.8 | AMPA + NMDA |
| 7 | trunk (apical, proximal) | soma | 150 | 2.5 | passive |
| 8 | zone (apical, distal: the dCaAP site) | trunk | 100 | 2.0 | dCaAP; AMPA + NMDA |
| 9–11 | t1.p, t1.m, t1.d (tuft branch 1) | zone, 9, 10 | 60 each | 0.9, 0.8, 0.7 | AMPA + NMDA |
| 12–14 | t2.p, t2.m, t2.d | zone, 12, 13 | 60 each | 0.9, 0.8, 0.7 | AMPA + NMDA |

**Why 25,000 µm² for the soma:** it gives an input resistance of 80 MΩ and a capacitance of 125 pF at the constants
below. That is a typical somatic input resistance, not a human measurement.

**Passive constants:**

| parameter | value | source |
|---|---|---|
| specific capacitance Cm | 0.5 µF/cm² | human layer 2/3 (Eyal et al. 2016) |
| specific membrane resistance Rm | 20,000 Ω·cm², so τ = Rm·Cm = 10 ms | a typical cortical value, not human-specific |
| axial resistivity Ra | 150 Ω·cm | typical, not human-specific |
| leak reversal E_L | −70 mV | typical |

**Coupling:** between a parent and a child compartment, g = 1 / (R_parent/2 + R_child/2), with R = 4·Ra·L / (π·d²).
The lumped soma counts as isopotential (its half-resistance is 0).

**Synapses** (per contact; one activation opens AMPA and NMDA together):

| part | conductance | rise / decay | other | source |
|---|---|---|---|---|
| AMPA | 0.88 nS | 0.3 / 1.8 ms | E = 0 mV | conductance: Eyal et al. 2018 (human layer 2/3); kinetics typical |
| NMDA | 1.31 nS | 2 / 70 ms | E = 0 mV | conductance: Eyal et al. 2018; kinetics typical |
| NMDA's Mg²⁺ block | B(V) = 1/(1 + [Mg]/3.57 · e^(−0.062·V)) | — | [Mg²⁺] = 1 mM | Jahr & Stevens 1990 |

Each conductance is a difference of two exponentials, scaled so its peak equals the conductance given.

**The soma: AdEx** (Brette & Gerstner 2005), with the "adapting" set of Naud et al. 2008:
- **Parameters:** V_T = −50 mV, Δ_T = 2 mV, a = 2 nS, τ_w = 300 ms, b = 60 pA, V_r = −58 mV.
- **A spike:** V crosses 0 mV. The soma is then held at +30 mV for 1 ms, then reset to V_r, and w increases by b.
- **Why the 1-ms hold:** it is the spike's waveform. Through the coupling, it is the backpropagating action potential
  (bAP), so the bAP's attenuation comes from the morphology, not from a separate rule.

**The dCaAP** (Gidon et al. 2020's model, as implemented by Hashemi et al. 2025, `Gidon_mc_lif_group_current.py`):
- **Onset:** the zone's voltage crosses θ = −36 mV upward, outside a 200 ms refractory period.
- **Its current:** I(t) = w·K·(σ(t/τ_A) − σ((t − Δ)/τ_B)), with σ the logistic function, τ_A = 3 ms, τ_B = 0.4 ms,
  Δ = 21 ms. That is a pulse of about 21 ms.
- **Its amplitude falls with the input's strength:** K = exp(−(I_in − I_th) / (0.3·I_th)), capped at 1.
  - **I_th** is the zone's outward current at θ: its leak plus the axial currents to its neighbours.
  - **I_in − I_th** is measured as C·dV/dt at the crossing step.
  - **Why this form:** it is the current-based version of the cited code's rule. The code uses the voltage overshoot
    within one time step, which depends on the step size.
- **w** is the one value not taken from a source. Its rule: the smallest w, in 10 pA steps, for which a dCaAP evoked by
  a threshold-level current step drives the zone to 0 mV. *This is an assumption: a typical peak for a dendritic
  spike, not a measured dCaAP amplitude.*

**Numerics:** backward Euler for the leak, coupling and synaptic conductances, at dt = 0.05 ms. The Mg²⁺ block, the AdEx
exponential, the adaptation and the dCaAP current are taken from the previous step.

**No tuning:** E0 runs with these values. The only values set by rule are w and each protocol's input strengths (below).
Any other change is an amendment, recorded before the run.

### The protocols (`protocols.py`) and pass criteria (`tests/test_e0.py`)

Each protocol is a set of interventions on one neuron, compared side by side. Input strengths are set by the stated
rule, never by hand. "Spiking off" means the soma's spike mechanism is disabled, as with TTX, to measure depolarization
below threshold, as the original experiments did.

| # | protocol | the rule for strengths | passes if |
|---|---|---|---|
| N1 | two groups of k contacts each, activated at once: A on b1.d; B on b1.m (the same branch) or on b2.d (another branch); k = 2, 4, …, 20; spiking off; the soma's peak depolarization | — | the same-branch ratio (A+B) / (A + B) reaches ≥ 1.5 for some k, and the other-branch ratio stays in [0.85, 1.15] for every k |
| N2 | a 2 ms somatic current pulse, and n tuft contacts spread evenly over the 6 tuft segments; the tuft input's time relative to the pulse, Δt = −10, −5, 0, 2, 5, 10, 20 ms | pulse: the smallest amplitude, in 50 pA steps, that gives exactly 1 spike. n = (the smallest even n whose tuft input alone gives a dCaAP) − 2 | pulse alone 1 spike; tuft alone 0 spikes; at some Δt in {0, 2, 5}, ≥ 3 spikes within 30 ms of the pulse; at Δt = 20, ≤ 1 spike in 50 ms |
| N3 | inputs A and B, each n contacts on the zone, at t = 10 ms; conditions none, A, B, A+B | n = the smallest count for which A alone gives a dCaAP | soma spikes: none 0, A ≥ 1, B ≥ 1, A+B 0 (XOR) |
| N4 | s contacts at each of b1's 3 segments, activated in order: inward (d, m, p) or outward (p, m, d); intervals 1, 2, 4 ms; s = 2, 4, 6, 8; spiking off; the soma's peak | — | inward/outward ≥ 1.15 at ≥ 2 of the 3 intervals, for some s |
| N5 | 2 s of Poisson input: 60 basal contacts (10 per basal segment) at rate r_b; 60 apical contacts (10 per tuft segment) at rate r_a; conditions r_b ∈ {r, 2r} × r_a ∈ {r′, 2r′}; 3 seeds. An *event* is a spike more than 16 ms after the previous spike, plus the spikes that follow within 16 ms; a *burst* is an event of ≥ 2 spikes | r = the smallest basal rate (grid 0.5–20 Hz, steps of 0.5) giving ≥ 3 events/s with no apical input. r′ = the smallest apical rate (same grid), with basal at r, giving a burst fraction ≥ 0.1 | doubling r_b changes the event rate ≥ 2× as much as doubling r_a does, and doubling r_a changes the burst fraction ≥ 2× as much as doubling r_b does (absolute changes, averaged over the other factor and the seeds) |
| N6 | conditioning: c contacts on b1.d at t = 10 ms; test: s contacts on b2.d at 10 + Δ ms; Δ = 50, 100, 300 | c = the smallest count giving b1.d > −40 mV for ≥ 30 ms (a plateau). s = the largest count, up to 40, whose test input alone gives no spike | no spike in the test window [10+Δ, 10+Δ+30] for conditioning alone or test alone; ≥ 1 spike there for both at Δ = 50 and 100; none at Δ = 300 |

**Checks on the apparatus** (they must pass, or no verdict counts):
- **dCaAP shape:** for current steps into the zone at 1, 1.5 and 2 times the threshold current, the measured K matches
  exp(−(I − I_th)/(0.3·I_th)) within 10%.
- **Step size:** the whole of E0 rerun at dt = 0.025 ms gives the same verdicts, and every scored number within 5%.

**The cost** (reported, not scored): wall-clock time per simulated second, on this CPU, for:
- one neuron;
- 12 neurons run as a batch;
- the 2 × 3 network (6 neurons).

Also compartment updates per second.

### Predictions (2026-10-30, before the code)

| # | prediction | confidence |
|---|---|---|
| N1 | passes: NMDA in a thin branch is supralinear; branches sum linearly at the lumped soma | high |
| N2 | **fails.** A bAP rises fast, so a dCaAP it triggers has a small K, and no burst follows. BAC firing is a rodent layer 5 finding; this cell is human layer 2/3 | medium |
| N3 | passes: XOR follows from K's shape, provided A+B's direct EPSP doesn't fire the soma | medium |
| N4 | uncertain: three segments per branch may be too coarse for the impedance gradient that Branco et al. describe | low |
| N5 | uncertain: the dCaAP's 200 ms refractory period caps bursting at 5 Hz, which may flatten the burst fraction | low |
| N6 | passes at Δ = 50 and 300; uncertain at Δ = 100, since the NMDA decay of 70 ms sets the plateau's length | medium |
| cost | under 5 s of wall-clock per simulated second for the 2 × 3 network (numpy, no compiler) | medium |

**What each outcome means:**
- **N2 fails while N3 passes:** a human layer 2/3 cell with a dCaAP is not a coincidence detector in Larkum's sense.
  Bursts would need a second, layer 5-like Ca²⁺ zone, which would be a design decision for the user.
- **N4 fails:** a branch needs more segments, or tapering, which is a measured cost.
- **N5 fails:** events and bursts aren't separate channels in this cell, so the burst-dependent rule (DESIGN.md §5,
  candidate 1) has no signal to use.

### Amendments (before the run)

1. **The dCaAP's I_th is a constant of the cell** (2026-10-30, during calibration, before any protocol was scored).
   - **The problem:** as pre-registered, I_th was the zone's instantaneous outward current at θ. That current goes
     negative whenever a neighbour is above θ, as the trunk is during a bAP (it reaches +19 mV), and K is then
     undefined. The first calibration hit this as an overflow.
   - **The new definition:** I_th is the zone's threshold current for a long step: the smallest 100 ms step, in 5 pA
     steps, that takes the zone to θ, with no dCaAP and spiking off.
   - **Why it's closer to Gidon's form:** the threshold current there is a fixed property of the cell, and the
     stimulus is measured against it. The drive at the crossing is still C·dV/dt.
2. **Details the pre-registration left open,** decided before the run:
   - every rule's search runs on the default cell;
   - w's threshold-level step is that same smallest step, with spiking off;
   - N5's rate searches use seed 0.

### The instrument: `viz/` (not an experiment)

A local web page served by `viz/server.py`. Every simulation runs in the same `sim.py`, so there is one model and no
second implementation.
- **The neuron, or the 2 × 2 or 2 × 3 network,** drawn as morphology. Each compartment is coloured by its voltage at
  the cursor's time, synapse activations show as dots, and spikes, bursts and dCaAPs are marked.
- **Interventions side by side:** each protocol opens with its runs (e.g. pulse alone, tuft alone, both). Each run is a
  row of voltage traces (soma, zone, a branch) on one shared time axis.
- **Adding a run:** change one input or cell parameter, then add the run as a new row, so the effect of one change
  shows against the others.
- **Time:** a scrubber and play button animate the morphology through the selected run.
