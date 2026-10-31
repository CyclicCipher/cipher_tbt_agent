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

### Run 1 (2026-10-30, at commit 265ec00; `runs/e0_dt0.05.json`): void, because the dCaAP check failed

**What the rules set:**
- **dCaAP:** the zone's threshold current is 270 pA, and w = 320 pA.
- **N2:** the pulse is 1950 pA, and n = 30 tuft contacts.
- **N3:** n = 17.
- **N6:** c = 9, and s = 40, which is the rule's cap.

The input resistance is 70.1 MΩ. The pre-registration's 80 MΩ was the lumped soma alone; the dendrites lower it.

**The check on the apparatus failed:**
- **The numbers:** for steps of 1, 1.5 and 2 times the threshold current, the measured K is 0.998, 0.932 and 0.780.
  Gidon's curve gives 1.0, 0.189 and 0.036.
- **Why:** the zone is a small compartment coupled to a large load, the trunk and the soma. Most of a step's excess
  current charges those neighbours, so C·dV/dt at the crossing is a small part of I − I_th. My measure of the input's
  strength does not implement Gidon's curve.
- **So:** by the pre-registration, no verdict of this run counts.

**What the run shows anyway** (a diagnosis, not verdicts):

| # | outcome | what it shows |
|---|---|---|
| N1 | the same-branch ratio peaks at 1.17 (k = 6), then falls to 0.65 at k = 20; the other-branch ratio stays between 0.89 and 1.10 | the NMDA nonlinearity is there in the branch, but at the soma, saturation swamps it. A thin segment near 0 mV stops passing current |
| N2 | pulse alone 1 spike, tuft alone 0. Together: 3 spikes at Δt = −10, **26 at −5**, then 2, 1, 1, 1, 1 at Δt = 0 to 20 | as predicted, a dCaAP that the bAP triggers has a small K (0.32 at Δt = 2, 0 at 5). And a pathology: the soma has no refractory period, so after the 1 ms hold the depolarized dendrite fires it again at once, a spike every 1.1 ms |
| N3 | spikes: neither 0, A 10, B 10, A+B 2. K: A 0.79, A+B 0.08 | the XOR mechanism works in the dendrite, where K falls tenfold. But A+B's direct input fires 2 spikes, and A alone fires 10 spikes about 1 ms apart (the same pathology) |
| N4 | inward/outward between 1.01 and 1.11 | order matters in the right direction, but too weakly with 3 segments per branch |
| N5 | passes as written. Doubling basal input adds 5.0 events/s, doubling apical adds 1.5. The burst fraction changes by +0.044 with basal and −0.113 with apical | not the phenomenon: the burst fraction **falls** with apical input. My criterion took absolute values, which was a mistake. Also, r′ hit the grid's floor (0.5 Hz), and with about 10 events per run, the fraction is noise |
| N6 | no spikes anywhere; the test rule hit its cap of 40 | a volley on one distal segment can't fire the soma at any size. It saturates: none up to 240 contacts, measured afterwards. The same volley spread over a branch's 3 segments fires it at 36 |
| cost | 1.46 s per simulated second for one cell; 2.27 for 12 as a batch (0.19 each); 1.61 for the 2 × 3 network | the prediction (under 5 s) holds |

**The step-size check:** not run, since the run is void.

**After the run, outside E0's scoring:** for the viz, the networks' connections were changed (`protocols.network`).
Each connection now spreads its contacts over one branch's three segments, and the two input fibres are asymmetric. The
cost figure for the network above used the old wiring.

**Proposed for run 2** (for the user to approve; nothing is changed yet):
2. **The dCaAP's input strength:** read Gidon's stimulus current directly. That is the injected and synaptic current
   into the apical subtree (the zone and the tuft) at the crossing, against I_th, instead of C·dV/dt.
   - **For current steps:** it reproduces Gidon's curve exactly, so the shape check becomes a check of the code.
   - **The consequence:** a bAP arriving from the trunk no longer counts as input. That changes N2's prediction:
     coincidence bursts become possible.
3. **A refractory period:** after the 1 ms hold, the soma is held at V_r for 2 ms. Without it, the cell fires at about
   900 Hz.
4. **N5's criterion gets a sign,** and more data:
   - doubling apical input must *raise* the burst fraction at least twice as much as doubling basal input does;
   - runs of 10 s, so each fraction rests on about 50 events.
5. **N6's test becomes a 2 ms somatic current pulse:** the largest, in 50 pA steps, that alone gives no spike. A
   volley on b2.d can't fire the soma at any size.
6. **N1 stays as it is:** its failure is the model's, not the protocol's.

### Amendments 2–5 (approved by the user 2026-10-30, before run 2)

2. **The dCaAP's input strength:** I_in is the injected and synaptic current into the zone and the six tuft segments,
   at the crossing step. K = exp(−max(0, I_in − I_th) / (0.3·I_th)), capped at 1, with I_th as in amendment 1. Current
   arriving through the trunk (a bAP) is not input.
3. **A refractory period:** after the 1 ms hold at +30 mV, the soma is held at V_r for t_ref = 2 ms, and only then
   released.
4. **N5:**
   - **Signed criterion:** doubling basal input must raise the event rate, and by at least twice the absolute change
     that doubling apical input gives. Doubling apical input must *raise* the burst fraction, and by at least twice the
     absolute change that doubling basal input gives.
   - **Longer runs:** each scored condition runs 10 s, not 2 s. The rate rules keep their 2 s searches.
5. **N6's test:** a 2 ms somatic current pulse at 10 + Δ ms. Its amplitude is the largest, in 50 pA steps, that alone
   gives no spike in the test window. The window, the conditioning and the criterion are unchanged.

**Predictions for run 2:**
- **The dCaAP check:** passes, by construction (it now checks the code).
- **N2:** may now pass, since a bAP meeting tuft input that is just below threshold gives a full-strength dCaAP.
- **N3:** still fails at A+B if the direct input alone fires the soma. Spike counts drop with the refractory period.
- **N5:** uncertain.
- **N6:** passes at Δ = 50, and is uncertain at Δ = 100.

### Run 2 (2026-10-30, at commit 81937d8): void, because of a bug in the simulator found after the run

**What the bug was:**
- **The mechanism:** the soma's AdEx exponential current is taken from the previous step. So on the step where the
  soma crosses V_detect, its new voltage overshoots by tens to hundreds of mV. The coupled solve passes that overshoot
  to the dendrites for that one step, before the soma is set to +30 mV.
- **How it was found:** after run 2, a diagnostic gave a full-strength dCaAP plus a zone step. The trunk read
  **+692 mV**, the zone +149 mV, and at w = 640 pA the zone read +11,720 mV.
- **What it touched:** every spike in runs 1 and 2 kicked the dendrites by an amount that depended on the step, not
  on the spike's defined waveform. That may account for run 1's 26 spikes in 30 ms, the dCaAP that a pulse alone
  triggered, and the network's dense bursts.
- **Why no check caught it:** none of the pre-registered checks looked at voltages. The step-size check runs after
  the main run.

**Run 2's numbers, kept for the record only:**
- dCaAP check: passes (K = 1.0, 0.189, 0.036).
- N1 1.17.
- N2: 1–2 spikes at every Δt.
- N3: A 0, B 0, A+B 1, with K = 0.019 for A alone.
- N4 at most 1.11.
- N5: events +6.87 basal, +1.13 apical. Burst fraction −0.384 basal, −0.058 apical.
- N6: passes.
- Cost: 1.56, 2.53 and 1.87 s per simulated second.

**One finding that doesn't depend on the bug:** amendment 2 makes K ≈ 0 for any dCaAP that synaptic input triggers.
- **The numbers:** N3's A alone, 17 contacts on the zone, gives K = 0.019, because a brief synaptic current that
  reaches θ is far larger than the threshold current for a long step.
- **What it means:** Gidon's K is defined for current steps.
  - Run 1's measure, C·dV/dt, fails for steps (the dCaAP check) and works for synapses (run 1's N3: K = 0.79 for A,
    0.08 for A+B).
  - Amendment 2's measure is the reverse.
  - Neither describes one mechanism that holds for both kinds of input.

### Amendment 6 (2026-10-30, before run 3): the spike's first step

The step on which a spike starts is solved again with the soma held at V_spike, so the dendrites see the spike's
waveform and never the upstroke's overshoot (`sim.py`). A bug fix, not a change to the model.

**The fix, checked:** the same diagnostic gives zone peaks of 48, 67 and 108 mV for w = 320, 640 and 1280 pA, rising
smoothly with w.

### Run 3 (2026-10-30, at commit 9d54abc; `runs/e0_dt0.05.json`): the result

| # | verdict | numbers |
|---|---|---|
| dCaAP check | pass, by construction | K = 1.0, 0.189, 0.036 |
| N1 | **fail** | the same-branch ratio is at most 1.17 (k = 6); the other-branch ratio stays in 0.89–1.10 |
| N2 | **fail** | pulse alone: 1 spike, *with a full-strength dCaAP* (K = 1.0). Tuft alone: 0. Together: 1–2 spikes at every Δt. K by Δt: −10: 0.71, −5: 0.57, 0: 0.26, 2: 0.09, 5–20: 1.0 |
| N3 | **fail** | spikes: A 0, B 0, A+B 1. K = 0.019 for A alone |
| N4 | **fail** | inward/outward at most 1.11 |
| N5 | **fail** | doubling basal input adds 7.93 events/s, apical 0.83 (events do follow basal input). The burst fraction changes by −0.448 with basal and −0.032 with apical |
| N6 | **pass** | conditioning or the test alone: no spike. Both: 1 spike at Δ = 50 and 100, none at 300 |
| step size (dt = 0.025 ms) | verdicts identical; 65 of 68 numbers within 5% | the three outside are N5 differences over 10 s of random input, which drift with spike timing: events with apical doubling 0.83 against 0.92; burst fraction −0.448 against −0.418 (basal) and −0.032 against −0.044 (apical). N5 fails at both step sizes. That run took 122 s, over the 2-minute limit |
| cost | prediction holds | 1.43, 2.42 and 1.70 s per simulated second (one cell; 12 as a batch; the 2 × 3 network) |

Run 3 matches run 2 to within N5's numbers. The overshoot bug was large only when a strong dendritic current met a
spike. The step-size check is not what caught it: run 2's check at dt = 0.025 gave the same verdicts (it crashed in
the comparison code, a bug in reading JSON keys, since fixed).

**What E0 established, across the three runs:**
1. **One thing works:** an NMDA plateau in a thin branch is a memory of about 100 ms, gone by 300 ms (N6).
2. **The NMDA nonlinearity shows in the branch but not at the soma** (N1).
   - **Why:** a thin segment saturates near its reversal potential. One distal segment can't fire the soma at any
     input size, while a branch's three segments together can.
   - **So:** the nonlinearity is real, but it caps, not amplifies, what reaches the soma.
3. **The apical side does not do its job** (N2, N3, N5). That is the side every candidate learning rule in DESIGN.md
   §5 reads.
   - **The bAP fires the dCaAP:** the bAP reaches the zone at about +4 mV, so every somatic spike outside the
     dCaAP's refractory period fires a full dCaAP. The apical dendrite is short (250 µm), and the bAP crosses it
     almost unattenuated.
   - **A full dCaAP doesn't make a burst:** it adds at most one spike. Its strength w was set by the zone's voltage,
     not by what reaches the soma.
   - **Bursts come from the wrong side:** the bursts that occur come from strong basal drive, and apical input barely
     changes them.
   - **No single definition of K works:** Gidon's K is defined for current steps. Run 1's measure, C·dV/dt, fits
     synaptic input and fails steps. Amendment 2's measure fits steps and gives K ≈ 0 for any synaptic input.
4. **Order sensitivity is weak** with 3 segments per branch (N4).

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

**To open it:** in the desktop app, start the launch configuration `dsnn-neuron-lab`. Or run
`../venv312/Scripts/python.exe -m dsnn.viz.server` from `experiments/` and open http://localhost:8770.
- **First start:** it calibrates the cell by its rules, about 10 s, and caches the result in `runs/`.
- **Keys:** space plays and pauses; ← and → step 0.5 ms; hovering a trace scrubs the time.
- **To add a variant:** select a run, change one setting, and press "Add run". The new row is labelled with its source
  run and the change.

## E0b: two cell types, and the apical side rebuilt (pre-registered 2026-10-30)

**The question:** E0's cell failed N1–N5. Rebuilt from two published models, does each cell type pass the experiments
that define it? And what does it cost to run?
- **Type A, human layer 2/3:** reduced by a stated rule from Gidon et al. 2020's human reconstruction, with their
  constants and their dCaAP.
- **Type B, layer 5-like:** Bahl et al. 2012's reduced layer 5 model, with its Ca²⁺ zone and channels.

**The user's decisions (2026-10-30):**
- The order: E0b, then E1 (the calcium rule against slice experiments), then E2 (credit assignment against the true
  gradient).
- The network has both cell types.
- The aim: "a scalable, biologically inspired credit assignment."

**Sources read before writing this** (the models' own code, from ModelDB's GitHub mirror):
- Gidon et al. 2020's XOR model (ModelDB 254217): the reconstruction `171122005.hoc`, `biophys.hoc`, `dCaAP.mod`,
  `xor.hoc`.
- Gidon et al. 2020's channel model of the dCaAP (ModelDB 260178, their Fig S12): HVA Ca²⁺, BK and a Ca²⁺ pump in one
  compartment.
  - **Ported to numpy as a check** (not part of E0b): the first event's peak **rises** with input, both for steps
    (18.9 → 34.5 mV over 0.425–0.65 nA) and for synaptic-shaped currents. Above threshold, the BK current then
    clamps the compartment near −39 mV.
  - **So:** the falling amplitude is not in this channel set as published. Gidon's XOR model doesn't use it; it uses
    the phenomenological dCaAP.
- Bahl et al. 2012's reduced layer 5 models (ModelDB 146026), and its authors' screenshot of example 1.
- Galloni, Laffere & Rancz 2020: in Bahl's model, BAC firing needs an apical trunk longer than about 350 µm and Na⁺
  channels along the trunk.

**No NEURON reference:** `pip index versions neuron` finds no wheel for this Python on Windows. So the ports are checked
against their authors' published outputs (check 3), and against our own passive model of the full reconstruction
(check 2).

### What E0 left, and what changes

| E0's finding | its cause | E0b's change | source |
|---|---|---|---|
| every bAP fires a full dCaAP | the apical dendrite: 250 µm, 2–2.5 µm thick, in compartments of 100–150 µm | type A's trunk is the reconstruction's own path to Gidon's dCaAP site: 220 µm, tapering from 6.1 to 1.4 µm, loaded by its side branches' membrane, in compartments ≤ 30 µm | ModelDB 254217 |
| no definition of K works for both steps and synapses | Gidon's K is defined for current steps | K from how fast the zone crosses θ: the step-free form of Gidon's own code (below) | `dCaAP.mod` |
| the dCaAP's time course | E0 followed Hashemi et al. 2025's reimplementation: current from the onset, at half amplitude | Gidon's code: both logistic terms start at 0.001, so the current is a pulse about 14–24 ms after the crossing | `dCaAP.mod` |
| a full dCaAP adds at most 1 spike | w = 320 pA, set by E0's rule | w = 2.97 nA, Gidon's value at their coupled site | `xor.hoc` |
| apical input doesn't control bursts | a human layer 2/3 cell has no BAC mechanism, as E0 predicted | bursts are type B's job: Na⁺ channels along the trunk (an active bAP) and a Ca²⁺ zone in the tuft | Bahl et al. 2012 |
| N1, N4 weak at the soma | 3 compartments of 60 µm per basal branch | type A's basal branches are real paths, 265 and 219 µm, in compartments ≤ 30 µm | ModelDB 254217 |

### Type A: human layer 2/3 (`cells/l23.py`)

**Geometry, by rule, from the reconstruction** (cell 171122005: a 20 µm soma, 9 basal trees, 101 basal and 81 apical
sections, 66,142 µm² of soma and dendrite):
- **The modelled paths.** A *main path* starts at a section and, at each branch point, follows the child whose subtree
  has the most membrane.
  - **Trunk:** the soma to apic[60](0.99), Gidon's coupled dCaAP site. 220 µm, 6 sections, 6.1 µm thick at the soma
    and 1.4 µm at the site.
  - **Two tuft branches:** the main paths of apic[60]'s two child subtrees: 489 and 504 µm long (ending 709 and 724 µm
    from the soma), mean diameters 1.26 and 1.23 µm.
  - **Two basal branches:** the main paths of the two largest basal trees: 265 and 219 µm, from 3.1 and 4.2 µm at the
    soma to 1.5 µm at the tips.
- **Every other section's membrane** (57,394 µm², 87% of the total) is added, as leak and capacitance, to the modelled
  compartment nearest the point where it attaches:
  - 19,615 µm² to the soma (the other 7 basal trees);
  - 9,484 and 7,178 µm² to the trunk at 83 and 27 µm (the obliques);
  - the rest spread along the paths.

  Total membrane is conserved.
- **Compartments:** each section is cut into ceil(L / 30 µm) pieces, as Gidon's model cuts its apical sections at
  ≤ 30 µm. Each piece's area and axial resistance come from its 3-D points: R = Σ 4·Ra·Δl / (π·d_i·d_(i+1)).
  - **The count:** 71 dendritic compartments plus the soma.
  - **Why it's fine enough:** 30 µm is below 0.1 λ at 100 Hz for every diameter in the cell (NEURON's d_lambda rule).

**Passive constants: Gidon's model's** (`biophys.hoc`), replacing E0's typical values:

| parameter | E0 | E0b, type A |
|---|---|---|
| Cm | 0.5 µF/cm² | 0.45 µF/cm² |
| Rm | 20,000 Ω·cm² | 37,000 Ω·cm² |
| Ra | 150 Ω·cm | 100 Ω·cm |
| E_L | −70 mV | −74 mV |

**The soma and synapses:** E0's AdEx and E0's AMPA/NMDA contacts, unchanged except E_L. Amendments 3 (refractory hold)
and 6 (the spike's first step) stay.

**The dCaAP,** at the compartment that holds apic[60](0.99), as in Gidon's code (`dCaAP.mod`, `xor.hoc`):
- **Onset:** the compartment crosses θ = −36 mV upward, outside a 200 ms refractory period.
- **The current:** I(t) = w·K·(A(t) − B(t)), where A and B are logistic:
  - A′ = A(1 − A)/τ_A, from A = 0.001 at the onset;
  - B′ = B(1 − B)/τ_B, from B = 0.001 at Δ after the onset;
  - τ_A = 3 ms, τ_B = 0.4 ms, Δ = 21 ms.
- **w = 2.97 nA:** Gidon's 4 mA/cm² over their site's segment (16.4 µm of apic[60], 1.44 µm thick: 74 µm²).
- **K:** Gidon's code sets K = exp(−(V⁺ − θ) / (D·(θ − V_rest))), with D = 0.3, V_rest = −75 mV, and V⁺ the voltage at
  the solver's first step past θ. V⁺ − θ depends on the solver's step, and goes to 0 as the step shrinks.
  - **Its step-free form:** V⁺ − θ = τ_K·V̇, with V̇ the compartment's dV/dt at the crossing.
  - **τ_K is fitted:** it is the one value with which current steps of 1.5 and 2 times the threshold current give
    Gidon's curve exp(−(I − I_th)/(0.3·I_th)), by least squares on ln K. I_th is as in amendment 1: the smallest
    100 ms step, in 5 pA steps, that takes the compartment to θ, with no dCaAP and spiking off.
  - **One mechanism for every kind of input:** a slow approach to θ gives K near 1, and a fast one gives a small K.
    For a single exponential charging curve, V̇ at θ is (I − I_th)/C, which is Gidon's form exactly.
  - **A bAP** that crosses θ gets K ≈ 0, but still starts the refractory period, as in Gidon's code.

### Type B: layer 5-like (`cells/l5.py`)
- **Bahl et al. 2012's reduced model 2,** ported as published. The source files (ModelDB 146026) are
  `reduced_geometry_20com.hoc`, `reduced_model.hoc` and `init_models_with_ca/init_model2.hoc`; model 2 is the one the
  authors' examples load.
  - **Geometry:** soma; one basal cylinder (257 µm); apical trunk (500 µm, 5 compartments); tuft (499 µm, 2); axon
    hillock (5), initial segment (5) and axon (1). 20 compartments, the source's own.
  - **Its fitted passive values and its mechanisms:**
    - at the soma: nat, kfast, kslow, nap and km;
    - along the trunk and tuft: nat (a linear gradient), and kfast and kslow (decaying with distance);
    - ih;
    - in the tuft: sca (HVA Ca²⁺), kca and cad.

    All at 37 °C.
- **Synapses:** E0's contacts, on its compartments.
- **It is a rat model, not a human one.**
  - I found no reduced human layer 5 model with BAC firing.
  - Human layer 5 dendrites are reported to be more isolated (Beaulieu-Laroche et al. 2018).

  Hence "layer 5-like."
- **Its basal side is one cylinder, 8.7 µm thick,** with no thin-branch subunits, so N1, N4 and N6 don't apply to it.

**E0's cell (v0) is replaced, not kept beside v1.** E0's runs stay reproducible from commit a325c72.

### The simulator (`sim.py`, rebuilt)
- **Any tree of compartments,** built from a morphology spec.
- **The step:**
  - backward Euler for the passive and synaptic parts, as in E0;
  - channel gates by exponential Euler on the previous step's voltage (NEURON's `cnexp`);
  - the AdEx soma and the dCaAP as above.
- **The solve:** Hines elimination for a tree, O(compartments) per step, compiled with numba (a new dependency in
  venv312). E0's dense solve is O(N³), too slow at 72 compartments.

### Protocols and pass criteria
E0's protocols and criteria (with amendments 2–5), with the sites restated for the new cells:

| E0's site | type A | type B |
|---|---|---|
| b1.d | basal 1's last compartment | — |
| b1.m | basal 1's compartment 60 µm nearer the soma | — |
| b2.d | basal 2's last compartment | — |
| b1's 3 segments (N4) | basal 1's 9 compartments, s ∈ {1, 2, 3} contacts each | — |
| the zone (N3) | the dCaAP compartment | — |
| the tuft segments (N2, N5) | the two tuft branches' compartments | the tuft's 2 compartments |
| the basal segments (N5) | both basal branches' compartments | the basal compartment |
| the somatic pulse (N2, N6) | the soma | the soma |

**What each type must pass:**

| | N1 | N2 | N3 | N4 | N5 | N6 |
|---|---|---|---|---|---|---|
| type A | required | reported | required | required | reported | required |
| type B | — | required | — | — | required | — |

Type A's N2 and N5 are reported to show what the network's layer 2/3 cells do with apical input. Gidon's cell has no
burst mechanism.

**Checks on the apparatus** (each must pass, or the verdicts of the type it concerns don't count):
1. **The solver:** on E0's cell with the dCaAP off, the new solver matches E0's dense solve over E0's N6 protocol:
   voltages within 10⁻⁶ mV, identical spike times.
2. **The reduction (type A):** compared against the full reconstruction, run passive with all 183 sections in
   ≤ 30 µm compartments, the same constants and the same solver. Each of these within 20%:
   - the soma's input resistance;
   - the input resistance at the dCaAP site and at basal 1's tip;
   - the steady-state transfer from the site to the soma;
   - the bAP's peak at the site, for E0's spike waveform at the soma (+30 mV for 1 ms, then V_r for 2 ms).
3. **The port (type B):** Bahl's own example 1 (0.4 nA for 500 ms at the soma, `regular_spiking.ses`), against the
   authors' screenshot:
   - 9 ± 1 spikes, the first two a doublet;
   - bAP peaks at apical(0.5) of +25 ± 10 mV;
   - peaks at tuft(0.5) of −42 ± 8 mV.
4. **The dCaAP's shape (type A):** K for steps of 1, 1.5 and 2 times I_th within 10% of Gidon's curve. There is one
   fitted constant for two points, so this checks the form.
5. **The step size:** dt = 0.025 against 0.05 ms.
   - Verdicts identical.
   - Numbers within 5%, except N5's, within 15%. N5 runs 10 s of Poisson input, which drifts with spike timing, as in
     E0's run 3.
6. **The compartment size:** type A at ≤ 15 µm, and type B's sections at ≤ 30 µm. Verdicts identical.

**The cost** (reported, not scored): wall-clock time per simulated second, and compartment updates per second, for:
- one cell of each type;
- 12 of each type as a batch;
- a 2 × 3 network of 3 cells of each type.

### Predictions (2026-10-30, before the code)

| | prediction | confidence |
|---|---|---|
| check 2 (reduction) | passes for the input resistances and the steady transfer. The bAP at the site may come out lower than in the full cell, since the obliques' membrane sits directly on the trunk | medium |
| the bAP at the dCaAP site | stays below θ, so no bAP fires a dCaAP | medium |
| check 4 (dCaAP shape) | passes | medium |
| A: N1 | uncertain. The tips are thicker than E0's (1.5 µm against 0.8), so they saturate less but need more input to unblock NMDA | low |
| A: N3 | passes. A threshold-level volley crosses θ slowly (K near 1), and w ≈ 3 nA fires the soma. A + B crosses fast (K small), and its direct EPSP reaches the soma weaker than in E0 through the loaded trunk | medium |
| A: N4 | passes, with 9 sites instead of 3 | low |
| A: N6 | passes | medium |
| A: N2, N5 | no bursts driven by apical input | high |
| check 3 (port) | passes | medium: a port can carry bugs that no other check catches |
| B: N2 | passes: the model was fitted to BAC firing | medium–high |
| B: N5 | uncertain | low |
| cost | under 1 s per simulated second for one cell; under 3 s for 12 as a batch | medium |

**What the outcomes mean:**
- **Type A passes N3:** the dCaAP's XOR survives a definition that holds for any input, and the network has a cell
  that computes it.
- **Type B passes N2 and N5:** the model has a cell whose burst fraction carries its apical input. That is the signal
  the burst-dependent rule (DESIGN.md §5) reads, and what E1's calcium rule is checked on.
- **Type B fails N2:** suspect the port first (check 3), then our synapses on a rat cell.
- **Check 2 fails:** the lumping rule is revised as an amendment (for example, each lumped subtree as an equivalent
  cylinder), before any of type A's verdicts are read.

**Not part of E0b:**
- how the two types are wired into the 2 × 2 and 2 × 3 networks (E2);
- plasticity (E1);
- the viz, which is updated after the run.

### Amendments (2026-10-30, during the build, before the run)

**What ran before these were written:** while building, I ran checks 1–3 and nothing else. No protocol of N1–N6, and
no rule that sets an input strength.
- **Check 3's protocol** (Bahl's example 1, at dt 0.05 and 0.025 ms): 9 spikes, the first two 5.0 ms apart. The bAP
  peaks at +26.4 mV at apical(0.5), and tuft(0.5) peaks at −35.7 mV. Nothing was changed after it.
- **Check 1:** first with a guessed pulse that gave no spikes. Then with E0's values, below.
- **Check 2:** failed with the pre-registered lumping, which led to amendment 2.

1. **The dCaAP's onset is level-triggered,** as in Gidon's `dCaAP.mod`, not a crossing.
   - **Onset:** the zone is above θ at the end of a step, outside the 200 ms refractory period.
   - **K = exp(−(level + τ_K·V̇)/(D·(θ − V_rest))),** capped at 1:
     - level: how far above θ the zone already was at the step's start (0 for a crossing);
     - V̇: the zone's dV/dt over the step.
   - **This matters when** the refractory period ends with the zone above θ. Gidon's code then fires with
     K = exp(−(v − θ)/…), and so does this.
   - **The current, as in the source:** w·K·(A − B) goes negative once B overtakes A, about 22 to 40 ms after onset.
     That is a hyperpolarizing tail.
2. **The lumping rule is revised (check 2 failed),** as the pre-registration says to do.
   - **The first result, with each unmodelled section's membrane lumped into the compartment nearest its
     attachment** (reduced / full):

     | soma R_in | R_in at the site | R_in at basal 1's tip | transfer, site → soma | bAP at the site |
     |---|---|---|---|---|
     | 0.969 | 0.977 | 0.991 | 0.958 | **0.70** (23.7 against 33.8 mV) |

     As predicted, the lumped membrane sits directly on the trunk as capacitance. That overloads fast signals while
     keeping the steady ones.
   - **The new rule: Rall's equivalent cylinder,** chosen on that principle before its result was seen, and kept
     whatever check 2 gave.
     - **What it replaces:** each of the 31 unmodelled subtrees (an unmodelled section whose parent is modelled,
       and everything beyond it).
     - **Where it attaches:** where the subtree attaches.
     - **Its area:** the subtree's membrane area.
     - **Its electrotonic length:** twice the subtree's area-weighted mean electrotonic distance from its start, with
       λ = sqrt(d·Rm/(4·Ra)) per 3-D piece. So the cylinder's membrane has the same mean electrotonic distance as the
       subtree's.
     - **Diameter and length** follow from those two.
     - **The cut:** NEURON's d_lambda rule at 0.1 λ at 100 Hz, which is the rule the pre-registration used to
       justify 30 µm. For check 6, it is 0.05.
   - **The cell now has 241 compartments:** 72 on the paths, and 169 in the cylinders. The full reconstruction has
     615.
   - **Check 2 with the new rule passes:** 0.990, 0.994, 0.997, 0.988, and **0.954** for the bAP (32.3 against
     33.8 mV). No type A verdict had been run.
3. **Sites are points on a path,** not compartment indices, so check 6's finer cut keeps them in place.
   - b1.d is the compartment that holds basal 1's tip.
   - b1.m holds the point 60 µm nearer the soma.
   - **N4 uses 9 points evenly spaced along basal 1,** distal first. The pre-registration's "basal 1's 9
     compartments" was wrong: basal 1 has 13 compartments at ≤ 30 µm. s ∈ {1, 2, 3} as pre-registered.
4. **Spread inputs are spread in proportion to compartment length:** N2's n tuft contacts, and N5's 60 basal and 60
   tuft contacts (E0: 10 per segment on 6 equal segments, so 60 in all).
   - The density stays uniform along the paths, and the input doesn't change with the cut.
   - Type B: the tuft's compartments and the basal compartment.
5. **Type B's N2 rule:** n = (the smallest even n whose tuft input alone fires the soma) − 2. Type B has no dCaAP.
   - **Grids:** n up to 2000 for type B, and up to 400 for type A (E0: 200).
6. **Port details (type B):**
   - **Kinetics:** the exact rate functions replace NEURON's tables. trap0 is evaluated at its true singular point
     (the source guards the wrong one). km and cad advance by exponential Euler where the source uses Euler.
   - **Integration:** the source's example runs NEURON's variable-step integrator. Ours is fixed-step.
   - **Initial state:** check 3 starts at v_init = −65 mV, as the source's session does. Every other protocol starts
     from the rest state: 20 s run silent from −65 mV (soma −70.06 mV).
   - **"A doublet":** the first interval is under 10 ms.
   - **Check 6:** each section gets nseg = max(the source's nseg, ceil(L/30 µm)). That is never coarser than the
     source.
7. **Simulator details:**
   - **Gate updates:** gates advance at the end of each step on the step's new voltage (NEURON's order). The next
     step's currents use them, which is what the pre-registration's "on the previous step's voltage" meant.
   - **Poisson input:** trains are drawn as event times, not as per-step counts; it is the same process.
   - **Check 1:** runs E0's N6 protocol with E0's run-3 values (c = 9, a 1900 pA pulse, Δ = 50), recorded in float64.
     The reference's copy changes only its recording's dtype. **Result:** voltages within 1.0 × 10⁻¹¹ mV, and the
     same spike (62.0 ms) in both. It passes.
   - **Check 2's bAP:** depolarization from rest at the site, for a soma clamped to +30 mV for 1 ms, then −58 mV
     for 2 ms.
   - **The cost's 2 × 3 network:** L2/3, L2/3 | L2/3, L5 | L5, L5. It uses the viz's stand-in wiring, with contacts
     spread over the basal side and the tuft. It is not E2's wiring.
