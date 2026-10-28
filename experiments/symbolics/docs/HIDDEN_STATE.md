# Hidden state: JEPA, SIGReg, and columns whose frame is time

Written 2026-10-28 for item 4 of the open list. The user: start on the hidden state with JEPA and SIGReg; each
cortical column should be a world model of what its reference frame makes it best at modelling; how models with
similar reference frames specialise should be answered with biological insight.

## 1. What hidden state is here

A **hidden variable** changes what an action does but is not a feature of the current screen; only the history reveals
it. From the public games (`arc_agent/claude/PUBLIC_GAMES.md`):

| kind | examples |
|---|---|
| a count never shown | r11l: the 5th hazard hit ends the game; sp80: failed pours; su15: an escalating penalty |
| a clock | tn36: barriers toggle every 3 instructions |
| a timer | ka59: a fuse; bp35: a descending ceiling |
| content out of view | ls20: fog; cn04: grey masking; lf52, bp35: scrolling views |
| replayed history | tu93: followers replay your past moves |

Content out of view is already partly covered by the Spotlight under-memory (what lies beneath a column). This
document is about the first three rows: variables that live in the event history, not in space.

## 2. JEPA, and what our columns already share with it

A JEPA (LeCun 2022) has three parts:
1. an encoder that maps an observation to a representation;
2. a predictor that maps (representation, action) to the next representation;
3. a loss measured in representation space, never in pixels.

Unpredictable detail can be left out of the representation instead of being modelled.

**An arc_agent column is already a JEPA with a fixed encoder:**
- the encoder is its frame: it reads the screen as a pose, a colour map, a bar length;
- the predictor is its model of what each action does (part 26's in-context attention);
- the target is the frame's reading of the next screen.

No column predicts pixels. What the agent lacks is a column whose state is **not read off the screen at all**: a
hidden variable.

## 3. SIGReg, and what it is for

**The problem it solves: collapse.** When both the prediction and its target are learned representations, a constant
representation predicts itself perfectly. Earlier JEPAs avoided this with heuristics: stop-gradients,
teacher-student encoders with moving averages, or variance and covariance penalties (VICReg).

**LeJEPA** (Balestriero & LeCun 2025, arXiv 2511.08544) replaces them with one principle:
- **The target distribution:** the representations should be distributed as an isotropic Gaussian. The paper proves
  this is the unique minimiser of the integrated squared bias of k-NN and kernel probes, and optimal for linear probes.
- **SIGReg (Sketched Isotropic Gaussian Regularization):**
  - project the batch of representations onto random 1-D directions (by the Cramér–Wold theorem, matching every 1-D
    projection matches the whole distribution);
  - test each projection against N(0, 1) with the Epps–Pulley statistic, the weighted squared distance between the
    projection's empirical characteristic function and e^(−t²/2);
  - average over the directions.

  Linear in dimension and sample size; one trade-off weight λ against the prediction loss; no stop-gradient or teacher.

**Two consequences for us:**
1. **Part 26's in-context attention is a kernel probe:** it is a kernel-weighted average over stored episodes. So if a
   column's situation includes a learned, continuous latent, LeJEPA's theorem says that latent should be isotropic
   Gaussian for the attention to read it well. That is SIGReg's job here.
2. **The linear case is the classic one:**
   - for a linear encoder, SIGReg's second-moment part is whitening (covariance = identity);
   - a linear JEPA under whitening finds the directions whose next value is most predictable from the current one:
     slow feature analysis (Wiskott & Sejnowski 2002), equivalently a spectral embedding of the transition graph;
   - for a random walk, those are the successor representation's eigenvectors: the grid-cell code (Stachenfeld et al.
     2017; slow features give place and grid cells, Franzius et al. 2007).

   **So the original TBT column (the SR eigenframe) was a linear JEPA with SIGReg's whitening, under another name.**

**Where collapse can happen in the agent:** only where both the representation and its target are learned. An innate
frame's reading of the screen is a fixed target, so the visible columns cannot collapse. A learned hidden latent can.

## 4. How columns with similar frames specialise: what biology does

Several columns can share one kind of frame. Cortex has five answers to how they come to model different things:
1. **By input (topography).** V1 columns share a retinotopic frame and differ by receptive field: they model different
   parts of the input because they are wired to different parts.
2. **By scale.**
   - **Space:** grid cells come in modules that share the hexagonal frame and differ in scale, by a ratio of about 1.4
     (Stensola et al. 2012). Several modules together form a residue code: position modulo several periods
     identifies position over a range far larger than any one period (Fiete, Burak & Brookings 2008).
   - **Time:** lateral entorhinal cells encode time since events with a spread of decay constants (Tsao et al. 2018):
     the same "time since" frame at many scales.
3. **By decorrelation.**
   - Lateral inhibition and anti-Hebbian plasticity push similar units to carry different components (Barlow 1961,
     redundancy reduction; Földiák 1990).
   - Inhibitory plasticity balances and decorrelates excitatory populations (Vogels et al. 2011).
   - SIGReg's isotropy, applied across columns, is this constraint: equal variance, no correlation.
4. **By context.**
   - Several forward models over one motor frame specialise in different contexts through responsibility: each learns
     where it predicts best (MOSAIC: Haruno, Wolpert & Kawato 2001; COIN: Heald, Lengyel & Wolpert 2021).
   - Hippocampal maps remap between environments.
   - Part 26's similarity is this mechanism inside one column.
5. **Hidden task states.**
   - The orbitofrontal cortex represents the task states that are not observable (Wilson, Takahashi, Schoenbaum & Niv
     2014).
   - New states are inferred as latent causes when prediction errors persist (Gershman, Blei & Niv 2010).
   - Hippocampal "splitter" cells split one place by its history; clone-structured graphs model this (George et al.
     2021).

**Reading for the agent:**
- Within a game, specialisation comes from (1) what a column is wired to, (2) its scale, and (4) the situations it is
  responsible in.
- Decorrelation (3) is the slow, population-level pressure that keeps similar columns from duplicating each other: it
  matters where columns learn their frames.

## 5. The proposal (to test first)

**Hidden columns with innate temporal frames, read in context.** For each event type, two kinds of column:
- **A counter column:** the event's count since the level began, read in its frame as the count itself and as residues
  (count mod k for k = 2…6). This is the grid-cell residue code, applied to counts: similar columns differ by scale.
- **A timer column:** steps since the event last happened (the lateral-entorhinal "time since").

Plus one **clock**: steps since the level began, with the same residues. An event is an action, or a visible column's
change of operator.

**How they are used:**
- Their states are situation features for the visible columns, so part 26's learned similarity θ decides which counter,
  timer or residue matters for which action, by the same rule as for any other feature.
- Their dynamics are known (count, reset at a level start), so a plan can roll them forward exactly.

**Where JEPA and SIGReg are tested:** a learned alternative to the innate temporal frames.
- A latent h = W c over a temporal context c (phases of the counts, decaying traces of events, phases of the clock).
- Trained as LeJEPA: predict the next h from h and the action, with SIGReg on h.
- Read in context like any situation feature.

If it matched the innate frames, hidden variables of kinds not built in could be learned.

**My expectation, stated before the test:** the learned latent will lose.
- Everything in c is driven by the actions, so every direction of c is equally predictable. The JEPA objective rewards
  predictability, not relevance, so it has no reason to pick the direction the hidden rule depends on.
- If so, JEPA with SIGReg belongs where frames are learned across many games (as grid cells are learned once, not per
  room), not inside one game.

## 6. The feel-experiment (pre-registered 2026-10-28, before the code)

**Task:** the synthetic grid of `structure/claude/attention_feel.py`, with doors whose state is hidden.
- Each step the agent moves in one of 4 directions, or presses (probability 0.2), which shows nothing.
- The 8 cells around it are floor, wall or door (0.6, 0.2, 0.2). A move is blocked by a wall, or by a door while the
  hidden rule says closed.
- Every 50 steps a new level begins: counts and clocks reset, and the agent knows a level began.

**Four hidden rules:**
- **H1 toggle:** doors are open while the presses so far this level are odd.
- **H2 threshold:** open once there have been at least 3 presses this level.
- **H3 timer:** open for the 4 steps after a press.
- **H4 clock:** open while (steps since the level began) mod 6 < 3.

**Learner:** part 26's in-context memory, one per direction. The keys are the 8 cells plus a constant. Only the
situation features differ between arms:
- **N:** none.
- **T (innate temporal frames), 12 discrete features:**
  - presses this level, mod 2, 3, 4, 5 and 6;
  - presses this level, capped at 8;
  - steps since the last press, capped at 9;
  - steps this level, mod 2, 3, 4, 5 and 6.
- **J (LeJEPA latent), 4 continuous dimensions:**
  - h = W c over a 25-dimensional context: cos and sin of 2π·presses/k and of 2π·steps/k (k = 2…6), presses/8
    (capped), and exp(−steps since press / τ) for τ = 1, 2, 4, 8.
  - Predictor: ĥ' = P_a h + b_a for each action.
  - Loss: the mean squared prediction error + λ·SIGReg(h), with λ = 1 and SIGReg in mean form (16 directions,
    Epps–Pulley on 17 points of t ∈ [0, 4]).
  - Adam, 10 iterations every 25 steps, on every stored transition within a level.
  - The probe's similarity on dimension d is (h_d − h'_d)²/2.
- **O (oracle):** the true open/closed state as one discrete feature.

5 seeds, 3,000 steps. Accuracy on moves is counted over steps 100–299 ("300") and 2,000–2,999 ("3,000").

| # | prediction |
|---|---|
| P1 | O ≥ 0.97 at 300 steps on every rule (the conjunction is learnable when the hidden state is given) |
| P2 | N ≤ O − 0.04 at 3,000 steps on every rule (the hidden state matters) |
| P3 | T ≥ O − 0.05 at 300 steps and ≥ O − 0.02 at 3,000 on every rule |
| P4 | J ≥ T − 0.02 at 3,000 steps on every rule (I expect this to be refuted) |
| P5 | T's largest θ at 3,000 steps is on the defining feature (H1: presses mod 2; H2: presses capped at 8; H3: steps since the last press; H4: steps mod 6) in ≥ 4 of 5 seeds per rule |

**Diagnostics** (reported, no threshold):
- J's latent: the variance and correlations of its dimensions, to check that SIGReg did its job;
- J's prediction loss;
- how well a linear probe of h recovers the true open/closed state.

### Results (2026-10-28; `runs/hidden_state_feel.txt`; 70 s)

**Apparatus:** the memory's θ gradient with a continuous mismatch, and the JEPA + SIGReg gradient (W, P, b), match
finite differences to 6 decimals.

| rule | N, 300 / 3,000 steps | T | J (LeJEPA) | O (oracle) |
|---|---|---|---|---|
| H1 toggle | 0.887 / 0.904 | 0.941 / **1.000** | 0.893 / 0.990 | 0.956 / 1.000 |
| H2 threshold | 0.922 / 0.948 | 0.912 / **0.999** | 0.937 / 1.000 | 0.960 / 1.000 |
| H3 timer | 0.903 / 0.908 | 0.914 / **0.999** | 0.909 / 0.928 | 0.975 / 1.000 |
| H4 clock | 0.888 / 0.902 | 0.913 / **1.000** | 0.894 / 0.901 | 0.969 / 1.000 |

- **T's largest θ** was on the defining feature in all 20 runs: presses mod 2 (H1), presses capped (H2), steps since
  the press (H3), steps mod 6 (H4).
- **J's latent, the same in every rule:** variance per dimension 0.64–0.94; largest correlation between dimensions
  0.81; prediction loss 0.0015 (nearly perfect); SIGReg 0.028. A linear probe of the latent recovers the hidden state
  at 0.60 (H1), 0.92 (H2), 0.68 (H3) and 0.53 (H4), against majority-class rates of 0.53, 0.71, 0.57 and 0.52.

**Verdicts:**
- **P1 refuted:** at 300 steps even the oracle is at 0.956–0.975. Learning the door's conjunction takes more than 100
  steps of 4 separate memories. The prediction was too optimistic about the task, not a finding about hidden state.
- **P2 confirmed:** the hidden state matters (N is 0.05–0.10 below O at 3,000 steps).
- **P3:**
  - at 3,000 steps, confirmed on every rule (T within 0.001 of O);
  - at 300 steps, refuted on H3 (−0.061) and H4 (−0.056), confirmed on H1 and H2.
- **P4 refuted, as expected:** J matched T on H1 and H2 but not on H3 (0.928) or H4 (0.901, no better than nothing).
- **P5 confirmed:** 20 of 20.

**Why J failed where it did:**
- **J's training data does not depend on the rule.** Its inputs are the actions and the clock, which are identical
  across the four rules, so it learned the same latent for all four.
- **A self-supervised objective cannot know which predictable direction matters.** Its 4 dimensions kept the press
  count's phases (enough for H1 and H2) and dropped the clock and the timer.
- **A confound:** SIGReg at λ = 1 in mean form did not fully decorrelate the latent (largest correlation 0.81), so its
  effective dimension was below 4. That changes how much J keeps, not that it cannot choose. Only a latent as large as
  the context keeps everything, and then it is T's information, whitened.

**What this says:**
1. **Hidden state is a column with a temporal frame, read in context.**
   - The innate counters, timers and clock, with part 26's similarity, found every hidden rule and the exact feature
     that defines it.
   - Similar columns (counters at moduli 2–6) specialise by scale, as grid modules do. Which scale matters is chosen by
     the situations each predicts in.
2. **JEPA and SIGReg are for learning frames slowly, across many games, not within one.**
   - Within a game, relevance comes only from the effects a hidden variable has on the visible columns, and the
     in-context similarity already learns from exactly that.
   - JEPA learns what is predictable; SIGReg keeps it spread out and uncorrelated. That is the right objective for a
     general-purpose frame learned before the task (as grid and time cells are), which in-context attention then reads.

## 7. Revision: what "time" hid, and a hidden column's frame in general (2026-10-28)

**The user:** "Time? That's it? That's just one dimension."

**What section 5 actually built:**
- for each event type, a counter (a line, read as residues: circles of periods 2–6) and a timer (a line);
- one clock.

That is many one-dimensional summaries of the event history, and every one is a count or a recency.

**What they express:** hidden variables that depend on how often or how long ago something happened.

**What they cannot express:**
- order ("A then B" and "B then A" give the same counts);
- a mode with its own logic (the next state depends on the current state and the event);
- content (a sequence to replay).

**Section 6's four rules were all expressible by them,** so that experiment tested selection, not coverage.

**The revised proposal:** a hidden column's frame is the state space of a hidden variable, with a transition for each
event. Four shapes:
1. **A line:** a counter, or a timer. Innate.
2. **A circle:** cycles and toggles (the residues). Innate.
3. **A delay line:** the last n events, in order. Innate; it gives order and replay.
4. **A learned graph:** the general case.
   - A finite set of hidden states, with a transition for each (state, event): an HMM whose transitions depend on the
     event, as in clone-structured cognitive graphs (George et al. 2021).
   - It is learned by EM (Baum–Welch) over the level's stored episodes: an offline pass, like replay in the
     hippocampus.
   - What it must explain is what a visible column sees: the operator observed, given the value of that column's most
     trusted key.
   - Its current state (the most probable one, filtered before the outcome is seen) joins the visible columns'
     situations.

**The biology** is that of section 4, item 5:
- splitter cells, which split one place by the route that led to it;
- the orbitofrontal cortex's maps of states that cannot be observed;
- latent causes inferred when prediction errors persist.

Lines and circles are the shapes built in innately, as grid cells' torus is. The joint hidden state is the product of
all hidden columns: each is small, together they cover a large space.

**Collapse:** a learned graph is a learned representation, so collapse here would mean two hidden states merged into
one. EM is anchored to observed outcomes, which works against that. Specialisation between several learned graphs is
not tested here (one graph per arm).

## 8. Second feel-experiment: rules the lines and circles cannot express (pre-registered 2026-10-28, before the code)

**The world:** section 6's grid, with 7 actions: 4 moves (0.15 each) and 3 buttons (0.133 each); levels of 100 steps.
Five rules:
- **R1, an ordered combination lock:** doors open once buttons 1, 2, 3 have been pressed in that order, with no other
  button pressed in between (moves don't matter), and stay open for the rest of the level.
- **R2, a mode machine:** a hidden mode cycles A → B → C → A on button 1, returns to A on button 2, and ignores
  button 3. Doors are open in mode C.
- **R3, replay:** a follower moves each step by the move the agent made 3 steps earlier (no move if that was a button).
  The prediction target is the follower's displacement (5 classes), from one memory with only the constant key.
- **H1, toggle (regression):** doors are open while button 1's presses are odd.
- **H4, clock (regression):** doors are open while (steps this level) mod 6 < 3.

**Arms**, all as situation features of part 26's in-context memory:
- **N:** none.
- **T (lines and circles), 54 features:**
  - for each of the 7 actions, its count this level mod 2–6, its count capped at 8, and steps since it, capped at 9;
  - the clock mod 2–6.
- **TD:** T plus the delay line: the last 6 actions.
- **TDG:** TD plus the learned graph.
  - 6 states; event-conditioned transitions.
  - Emissions: the outcome, given the value of the visible memory's most trusted key (the largest summed prequential
    log probability).
  - Baum–Welch with pseudo-counts 0.01 (transitions) and 0.1 (emissions), every 250 steps, over every stored level.
    Each time, 3 starts (the last model and 2 random ones), 20 iterations each; the best likelihood is kept.
  - After each fit, every stored episode's graph state is recomputed with the new model, as J's latent was.
- **O (oracle):** the true hidden variable: the lock's stage, the mode, the move 3 steps ago, button 1's parity, or the
  clock's open/closed state.

5 seeds, 3,000 steps; accuracy over steps 2,000–2,999 (doors: moves only; R3: every step).

| # | prediction |
|---|---|
| Q1 | O ≥ 0.98 on every rule (the task is learnable given the hidden variable) |
| Q2 | T ≤ O − 0.03 on R1, R2 and R3 (the rules are beyond lines and circles) |
| Q3 | TD ≥ O − 0.02 on R3 (the delay line gives replay) |
| Q4 | TDG ≥ O − 0.02 on R1 and R2 (the learned graph gives order and modes) |
| Q5 | no harm: TDG ≥ T − 0.01 on H1 and H4, and TDG ≥ TD − 0.01 on R3 |
| Q6 | on R1 and R2, the graph's states recover the true hidden state with purity ≥ 0.95 (each graph state's steps are mostly one true state) |

**The uncertain one is Q4,** for the lock in particular. The intermediate stages (pressed 1; pressed 1 then 2) change no
outcome, so EM must find a chain of states before the chain pays. Baum–Welch is prone to local optima there.

**Build check, before the main run (2026-10-28):**
1. **The learned graph failed its apparatus check.**
   - On a known 3-state mode machine (20 levels), EM never lowered the likelihood, but its states matched the true
     modes only 57% of the time. The true model's log-likelihood was −26; EM's best was −48 to −92, even with 100
     iterations, 10 starts, or 12–48 states.
   - Cause: random initial transitions scramble the hidden state on every move, so it carries no memory, and EM gets
     almost no signal that moves should leave it alone.
   - **Amendment:** starts keep their state (transitions 0.8 identity + 0.2 random; a state persists until an event
     changes it), with 12 starts and 60 iterations per fit. With that, about a third of fits recover the machine
     exactly (log-likelihood −5.5 to −15, purity 0.99–1.0). Every fit also starts once from the last model and keeps
     the best, so a run has about 144 tries.
2. **The arms with many situation features collapsed below "no hidden state".** At steps 100–299: R2 T 0.57 / 0.48
   against N 0.91 / 0.97; R3 TD 0.93 / 0.41.
   - Cause: each θ takes normalised steps clipped at 0, so noise drifts every θ upward. With 54 features their sum
     makes every stored episode count about nothing: the quotient failure of part 26, in extreme form.
   - **Amendment: an Occam prior on θ,** θ += η (g / rms − λ) with λ = 0.3, so a feature's similarity weight grows only
     when its gradient is consistently positive. At λ = 0.3, R2 T is 0.89 / 0.94 and R3 TD 1.0 / 1.0.
   - The first experiment (section 6) ran without it (λ = 0) and stands as recorded.
   - **This affects part 26 in the agent too:** its similarity has the same drift, so the agent's in-context attention
     should get the same prior when hidden columns add many features.

### Results (2026-10-28; `runs/hidden_state_feel2.txt`; 296 s)

Accuracy over steps 2,000–2,999 (5 seeds):

| rule | N | T (lines, circles) | TD (+ delay line) | TDG (+ learned graph) | O (oracle) | graph purity |
|---|---|---|---|---|---|---|
| R1 lock | 0.903 | 0.928 | 0.927 | **0.977** | 0.999 | 1.0 in 5 of 5 |
| R2 mode | 0.976 | 0.971 | 0.970 | 0.972 | 0.976 | 1.0 in 5 of 5 |
| R3 replay | 0.413 | 0.754 | **1.000** | 1.000 | 1.000 | 0.27–0.31 |
| H1 toggle | 0.893 | 0.998 | 0.998 | 0.998 | 1.000 | 1.0 in 5 of 5 |
| H4 clock | 0.900 | 0.974 | 0.965 | 0.964 | 1.000 | 0.51–0.53 |

**What was learned:**
- **The learned graph recovered the lock, the mode machine and the toggle exactly in every seed:** each of its states
  is one true state of the hidden machine.
- **The delay line gives replay:** the largest θ is on the action 3 steps back (θ = 20).
- **Counters and timers approximate replay through "steps since action k" (0.754).**
- **In R1 the readout lagged the graph:** the graph's state has the largest θ, but θ reached only 0.7 by step 3,000,
  so episodes from the other state still counted about half and accuracy stayed at 0.977.

**Verdicts:**
- **Q1 refuted on R2:** even the oracle is at 0.976, the same as N.
  - The mode is open only about 14% of the time (two presses of button 1 with no button 2 between), so only about 3%
    of moves carry information.
  - The Occam prior (λ = 0.3) holds a feature's θ at 0 unless its gradient is positive most of the time. So it kept
    even the oracle's feature unused.
  - R2 therefore says nothing about accuracy. **That is a design error of the test and a cost of the prior:** sparse but
    real evidence is suppressed.
- **Q2 confirmed on R1 and R3; refuted on R2,** for the same reason.
- **Q3 confirmed:** TD 1.000 on R3.
- **Q4:**
  - R1 refuted by 0.002 (0.977 against 0.979), with the graph itself exactly right;
  - R2 confirmed, but uninformative.
- **Q5 confirmed:** no harm on H1 (0.998 = 0.998), R3 (1.000 = 1.000) and H4 (−0.0096, inside the −0.01 margin).
- **Q6 confirmed:** purity 1.0 on R1 and R2 in all 10 runs.

**What this says:**
1. **The four shapes cover the three new kinds of hidden variable.** The learned graph is the general one: it found
   the lock (order) and the mode machine exactly, from what the visible column saw. Once its starts keep their state,
   EM recovers a machine.
2. **The bottleneck has moved to the readout: how the situation features weight the stored episodes.**
   - Without a prior, the θs of many irrelevant features drift up together and erase the memory.
   - With the prior, a relevant feature with sparse evidence is never used.
   - Both come from one structural fact: the similarity is a product over every situation feature, so every feature's
     weight counts against every episode.
3. **The next step, a proposal: features compete instead of adding.**
   - Attention over the situation features themselves: a softmax over "which feature defines similar here", with
     each feature's weight learned by its evidence (as keys are, in part 26).
   - Irrelevant features then take a share of attention instead of multiplying every episode's weight down. A rare but
     decisive feature can win on accumulated evidence.
   - In the brain this is dimensional attention: people learn which stimulus dimension is relevant by attending to it
     (Niv et al. 2015; Leong et al. 2017).
