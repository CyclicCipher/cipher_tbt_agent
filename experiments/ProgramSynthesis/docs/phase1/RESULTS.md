# Phase 1 Results — the binding-channel sweep

> Behavior-cloning a BFS oracle on LockPath, four positional-binding arms, trunk/optimizer/data
> held identical so the encoder is the only variable (LEARNING_AGENT.md). Metric: held-out
> masked action-match accuracy. Config: 1500 steps, batch 256, compiled, 250/80 train/held-out
> layouts. **Single seed unless noted — directional, being hardened with `--seeds`.**

## The four arms

`none` (no position) · `content` (learned absolute (x,y,t) embedding) · `pope2d` (rotary x,y) ·
`pope2d1` (rotary x,y,**t**).

## Headline findings (single seed)

**1. Channel dominance (P1) holds, but is mechanic-dependent.** Final held-out accuracy:

| mechanic | none | content | pope2d | pope2d1 |
|---|---|---|---|---|
| nav (control) | 0.779 | 0.876 | **0.966** | **0.966** |
| key_door | 0.770 | 0.719 | 0.803 | **0.867** |
| block_pad | 0.638 | 0.659 | 0.695 | **0.738** |
| compose (direct) | 0.644 | **0.757** | 0.737 | 0.730 |
| parts → compose (P3) | 0.603 | 0.682 | 0.713 | **0.727** |

**2. The temporal axis earns its keep in proportion to temporal structure.** `pope2d1 − pope2d`:
`nav` 0.000 (exact tie — the clean control, no time structure), `block_pad` +0.043 (motion),
`key_door` +0.064 (causal door). The `t`-band helps exactly where the task is temporal/causal and
does nothing where it isn't.

**3. `none` is consistently weakest** (last or near-last everywhere) — no spatial handle.

**4. `content` is erratic** — best on `nav` and compose-direct, *worst* on `key_door`. The
memorizer's signature: absolute-position embeddings help when layouts share structure, hurt when
the meaningful object moves.

**5. The composition gap — the sharpest result.** `direct − (parts→compose)`:

| arm | compose direct | parts → compose | **gap** |
|---|---|---|---|
| pope2d1 | 0.730 | 0.727 | **+0.003** |
| pope2d | 0.737 | 0.713 | +0.024 |
| none | 0.644 | 0.603 | +0.041 |
| content | **0.757** | 0.682 | **+0.075** |

`content` wins compose-direct by **memorizing** the whole, but loses 0.075 when it must **build**
compose from its separately-learned parts. `pope2d1` composes **losslessly** (gap 0.003). This is
the value-centric vs program-centric distinction (Chollet's two abstractions; the A∘B test) made
quantitative: the arm that wins by memorizing is the one that cannot compose. It also matches the
discovery-paper frame — composition = transporting known structure into a new regime, and
`content`'s gap is the residual its representation cannot transport ([[reference_discovery_regime_transition]]).

## Caveats
Single seed; ceilings differ by task (block_pad/compose lower — harder, more action-ties); some
effects are within noise. Five conditions tell one coherent story, but error bars are needed
before calling it a result.

## Seed-hardening (partial — compose-direct only)

The 5-seed sweep was cut short (a `torch.compile(reduce-overhead)` CUDA-graph re-recording storm
on the varying eval shapes made it intractably slow; fixed afterward to plain fusing compile). The
transfer run never completed, so **the composition gap is NOT seed-hardened** — it remains the
single-seed result above. What we did get, `compose`-direct over seeds 0–2 (+ partial seed 3):

| arm | per-seed held-out | mean |
|---|---|---|
| none | 0.647, 0.667, 0.636, 0.642 | **0.648** |
| content | 0.763, 0.677, 0.690, 0.681 | **0.703** (high variance) |
| pope2d | 0.745, 0.739, 0.693, 0.690 | **0.717** |
| pope2d1 | 0.739, 0.739, 0.709 | **0.729** |

**This corrects the single-seed picture in a useful way:** content's apparent compose-direct *win*
(0.757) was a seed-0 outlier — across seeds content averages ~0.70 with the **highest variance**,
while the pope arms lead slightly and are more stable, and `none` is robustly worst. So P1's
ordering (`pope ≥ content > none`) holds with error bars on compose-direct; but because content's
direct number was inflated by that lucky seed, the single-seed composition gap (content +0.075)
likely **shrinks** once the transfer run is hardened. Direction (pope composes, none worst) probably
holds; magnitude is unconfirmed. **Status: Phase 1 closed here by decision — not fully hardened.**
