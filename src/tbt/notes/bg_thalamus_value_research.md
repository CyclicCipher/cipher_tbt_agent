# Basal Ganglia · Value/Dopamine · Thalamus — research + substrate decisions

Lean three-thread primary-source review (2026-07-10) for the BG/thalamus/value redesign, focused on **how each learns** and
**what substrate fits** (ANN vs modified-HTM/SDR vs deterministic) for our from-scratch SDR agent. Full reports below;
one-line verdicts and the plan are in `src/tbt/ROADMAP.md`. **No ANNs anywhere** — backprop is the wrong shape for all three.

## Verdict table
| subsystem | how it learns | substrate | our code |
|---|---|---|---|
| **basal ganglia** | local dopamine-gated **three-factor Hebbian** (pre × post × scalar δ) + ~1 s eligibility trace; D1/D2 opponent (OpAL); NO backprop | **modified-HTM / SDR** — per-bit Go/NoGo weights over the percept SDR | current `basal_ganglia.py` is the **tabular one-hot seed** of exactly this — widen the key from exact-match to SDR-overlap |
| **value critic** (`reward.py`) | online **TD δ-rule**; value = **successor-representation read-off** `V = w·(W·φ)`; NO backprop, NOT hand-coded | **SDR-linear TD (the SR)** | REUSE validated legacy `SuccessorFeatures` (`l6_sr.py`); prerequisite = wire the SR into L6a (a plain HTMLayer now) |
| **thalamus** | mostly **does NOT learn** — fixed routing/gate; only slow modulatory-gain plasticity, decisions learned elsewhere | **DETERMINISTIC** — route + default-off disinhibition gate + VSA content⊗location bind; no ANN/HTM inside | current `thalamus.py` (relay+gate) is on the right track — add the bind + a real disinhibition gate |

---

## THREAD 1 — Basal Ganglia (mechanism + substrate)

**What it computes (compressed):** a centralized selector by **release from tonic inhibition** — GPi/SNr clamp the thalamus
shut; **direct/Go (D1)** disinhibits the selected channel, **indirect/NoGo (D2)** suppresses competitors, **hyperdirect
(STN)** is the fast diffuse global brake. Off-centre/on-surround WTA (Gurney/Prescott/Redgrave GPR model). STN =
Frank's "hold-your-horses" dynamic threshold under conflict (a commitment knob, not a learning signal).

**How it learns — the teaching signal is dopamine = RPE (Schultz).** Corticostriatal synapses onto MSNs follow a
**three-factor / neo-Hebbian rule**: coincident pre (cortical) × post (MSN) sets a **silent eligibility trace** (~1 s decay;
Shindou 2019), which converts to a lasting change only if **dopamine δ** arrives in the window (Gerstner 2018). **D1/D2 are
opponent**: a dopamine BURST (δ>0) → LTP at D1 (strengthen Go); a DIP (δ<0) → LTP at D2 (strengthen NoGo).

**OpAL (Collins & Frank; Jaskir & Frank 2023) exact updates:**
```
δ_t        = R_t − V_t(a)                         # critic RPE
G_{t+1}(a) = G_t(a) + α_G·G_t(a)·(+δ_t)           # D1/Go   : δ>0 potentiates   (three-factor: weight IS the pre×post term)
N_{t+1}(a) = N_t(a) + α_N·N_t(a)·(−δ_t)           # D2/NoGo : δ<0 potentiates
Act(a)     = β_g·G(a) − β_n·N(a) ;  β_g = β·max(0,1+ρ), β_n = β·max(0,1−ρ)   # ρ = tonic DA (explore/exploit/vigor)
```
Signals the update needs: (i) pre-synaptic pattern (active percept bits), (ii) the selected action (only its weights are
eligible), (iii) a **scalar dopamine δ** from the critic, (iv) a short eligibility trace. **Local to each synapse + one
broadcast scalar — no error backprop.**

**Substrate call → (b) an SDR-native layer with a local dopamine-gated three-factor Hebbian rule.** Reject (a) ANN
(backprop is machinery the BG doesn't have/need; opponent D1/D2 ≠ one differentiable value head; fights the SDR substrate).
Treat (c) deterministic table as the **one-hot special case of (b)** — a good debuggable scaffold to get the disinhibition +
STN-brake + OpAL signs right, then **widen the key from one-hot to the full percept SDR with zero change to the rule**.
SDR-native form: per action `a`, two sparse weight vectors `W_G[a], W_N[a]` over the percept-SDR bit space;
`select`= argmax `β_g·Σ_{i∈x}W_G[a][i] − β_n·Σ_{i∈x}W_N[a][i]` (+ STN conflict brake); `learn`= tag eligibility on active
bits, on δ apply `W_G[a][i]+=α_G·e·(+δ)`, `W_N[a][i]+=α_N·e·(−δ)`. Cost O(active_bits × actions). Generalization across
states is **automatic** (shared SDR bits share Go/NoGo evidence) — exactly what a table can't do. **This is an EXTENSION of
the existing `OpponentActor`, not a rewrite.**
Sources: O'Reilly&Munakata CCN §7.2; Gurney/Prescott/Redgrave 2001; Frank 2006 (STN); Gerstner 2018 (eligibility/three-factor);
Shindou 2019 (silent trace); Jaskir&Frank 2023 (OpAL*); eLife 2024 (101747) distributed D1/D2; arXiv:1909.01575 (sparse RL).
*Uncertainty:* no canonical published "SDR basal ganglia" — this is a synthesis, expect to tune; eligibility horizon λ is a
real design choice; opponency earns its keep mainly under asymmetric reward/punishment — gate the D2 machinery on evidence.

---

## THREAD 2 — Value / Dopamine critic (`reward.py`)

**Dopamine ≈ TD error (Schultz/Montague/Dayan):** phasic DA = δ = r + γV(s′) − V(s) (fires to unpredicted reward, silent to
predicted, dips on omission). **Tonic** DA = average reward rate → vigor + explore/exploit (= our `ρ`). **Actor-critic
anatomy:** VTA/ventral striatum = **critic** (state value, emits RPE) → `reward.py`; dorsal striatum = **actor** (policy,
consumes RPE) → `basal_ganglia.py`'s OpAL. (Joel/Niv/Ruppin 2002; O'Doherty 2004.)

**How value is learned — TD δ-rule** `V(s) ← V(s) + α·δ`, online, no backprop; TD(λ) eligibility traces optional.
**The successor representation (the one that matters for us):** value factors into a reward-independent predictive map ×
reward: `V = M·R`, `M = (I−γT)⁻¹`, trained online by a **vector** occupancy error. Stachenfeld 2017: place cells = SR rows,
grid cells = SR eigenvectors — **our L6 grid frame and the SR are the same object.** **Successor features** (SDR form):
`ψ(s)=𝔼[Σγ^k φ(s_{t+k})]`, `V=w·ψ`; overlapping SDRs share operator columns → value generalizes *before* each state is
visited. **This is exactly legacy `SuccessorFeatures` (`l6_sr.py`), validated in isolation.**

**Substrate call → (b) linear value over SDR features by online TD = the SR/SF read-off.** Reject (a) ANN (needs
batch/replay gradient training — violates the no-training-on-this-machine + no-gradient-on-big-data rules; not what the
striatum does). Reject (c) hand-coded value (bitter-lesson; value must be learned from the sparse score).
`reward.py` = a thin `Critic` wrapping (reused) `SuccessorFeatures`: `value(φ)=w·(W·φ)`; `dopamine(φ,r,φ′,done)` →
`δ = r + γV(φ′) − V(φ)` (also calls `sf.observe` to train W + w online); `rho()` = tanh of a tracked average-reward. **No
change to the BG interface — it already takes a scalar δ.**
**Two DISTINCT error signals (Gardner/Gershman 2018):** the **vector** SF-Bellman error trains W (L6's map — runs even with
zero reward = the epistemic/structure signal); the **scalar** δ trains the BG. `sf.observe` computes both; keep them
conceptually separate.
**LOAD-BEARING CAVEAT (our own proof, `project_linear_value_cannot_hold_sokoban`):** a linear value over grid/SDR features
**cannot** represent conjunctive/relational V* (Sokoban) — only tabular fits, and it needs ROLLOUT. So (b) is correct for
**navigable/routine** value (reach goal, avoid cost, vigor); for **relational** value the critic shrinks to **scoring
rollout leaves + the epistemic bonus** (the hippocampus). Write `value(φ)` to serve *either* as the terminal signal *or* as
a rollout-leaf evaluator — don't hard-wire "value alone suffices."
Sources: Schultz/Dayan/Montague 1997; Schultz 1998/2016; Niv 2007 (tonic DA/vigor); Joel/Niv/Ruppin 2002; Dayan 1993 (SR);
Stachenfeld 2017; Gershman 2018; Barreto 2017 (successor features); Gardner/Schoenbaum/Gershman 2018 (generalized PE);
Sutton&Barto 2018. *Prerequisite:* L6a is a plain HTMLayer now — wiring `SuccessorFeatures` into L6a is real work, not copy-paste.

---

## THREAD 3 — Thalamus

**What it does:** a **gated context-dependent router**, not a passive relay (Sherman & Guillery). **Driver vs modulator
synapses** — drivers carry the message (few, large, proximal, all-or-none), modulators adjust gain/timing (many, small,
distal). **Core (PV → L4, driver, specific) vs Matrix (calbindin → L1/L5a, diffuse, synchronizing/binding)** — maps 1:1 onto
our two jobs: **core = percept relay to the selector; matrix = the cross-column binding/voting channel.** **TRN** = the
GABAergic inhibitory gate / Crick's attentional "searchlight" (topographic). **The cortico-BG-thalamo-cortical loop is how
"select" becomes "gate":** GPi/SNr tonically INHIBIT the thalamus (default-blocked); the BG Go pathway inhibits GPi/SNr →
**disinhibits** the selected channel. The BG doesn't push the winner — it **stops blocking** it (sign inversion).
**Higher-order/transthalamic** relay (L5 driver, cortex→thalamus→cortex) = the inter-column router (Halassa "switchboard").

**Does it learn? — Overwhelmingly NO.** Driver (content) synapses onto relay cells **do not readily plasticize** (PMC3008844);
only the **modulatory** (L6→thalamus) path has NMDAR-LTP/LTD (gain, not content); TRN plasticity is short-term/attentional,
not storage. Routing decisions (task rules, attention targets, action values) are **learned upstream (cortex/PFC/striatum)
and merely APPLIED** at the thalamus.

**Substrate call → (c) DETERMINISTIC code — routing/gating + binding, NO learner inside.** Not a shortcut — it's what the
biology says. Three deterministic operations:
- **relay/route (core)** — content-preserving pass-through/multiplex; route index set upstream.
- **BG→thalamus gate** — default-off (mirror GPi/SNr tonic block); **disinhibit only the BG-selected channel** to the motor:
  `motor = percept if channel==bg_winner else BLOCKED`. The intelligence is in the BG; the thalamus contributes default-off +
  release-the-winner.
- **content ⊗ location binding for voting (matrix)** — fixed **VSA / tensor product** (SDR permutation+overlap or
  circular-conv), **no parameters, no learning**; overlap-union unbinds/votes.
The one caveat: if modulatory-gain plasticity is needed later, add a **single learned scalar gain per channel**, updated by
the **same** RPE/salience signal (not a new HTM learner). Keep `thalamus.py` content-opaque.
Sources: Sherman&Guillery 2001, Sherman 2005/2016, Sherman&Usrey 2024 (transthalamic); Jones core/matrix 1998/2001/2009;
Crick/TRN, Pinault 2004, McAlonan 2006; BG-loop disinhibition (CSHL Perspectives; Nature 2021); PMC3008844 (driver synapses
fixed); Biane PMC4795975; Rikhye/Halassa 2018 (switchboard).
