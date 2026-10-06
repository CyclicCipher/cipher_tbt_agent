# Ataraxos — Sokota et al., Nature 2026 — reference notes

**Source:** Samuel Sokota, Eugene Vinitsky, Hengyuan Hu, Zhiyuan Fan, J. Zico Kolter & Gabriele Farina, *Scalable
decision-making for games of imperfect information*, Nature 658:55–59, published 30 Sep 2026,
doi:10.1038/s41586-026-11036-y. Open access (CC BY 4.0). **Read 2026-10-06: main text, Methods, Extended Data
captions.** The Supplementary Information PDF (training configuration, network sizes, search hyperparameters) was NOT
read.

## The claim

Tabula-rasa self-play RL plus test-time search, made to work under large amounts of hidden information. Stratego
(over 10³³ set-ups): 15 wins, 1 loss, 4 draws against Pim Niemeijer, the most decorated player (85% effective win
rate), and 38–2 in 40 demo games at the 2025 World Championship. The same techniques give a superhuman Barrage
Stratego AI, a new state of the art in Hanabi (2–5 players; the 2-player result at ~1/100 of the previous compute) and
in dou dizhu. Training: 16 H100s for 1 week (RL) plus 4 for 4 days (belief net), "a few thousand dollars" — against
DeepNash (Science 2022), about 1/500 of the compute, 1/30 of the self-play games, 1/100 of the training examples.

## The design pattern (three parts)

1. **A policy–value network trained by self-play** (transformers: an encoder-only move network, a decoder-only set-up
   network, two coupled self-play processes).
2. **A belief network** trained on self-play games of the final policy to model the hidden information (here the
   opponent's piece types), used as a generative model at search time.
3. **Test-time search via update equivalence** (Sokota et al., ICLR 2024): sample hidden states from the belief net;
   for each candidate move run depth-limited rollouts with the move network and average the value predictions; then
   perform ONE more step of the same regularised update used in training — a tabular step of magnetic mirror descent
   (Sokota et al., ICLR 2023), with the same two reverse-KL terms — applied only to the current decision, and sample
   the move from the updated policy. "Because this step mimics those of self-play reinforcement learning, the search
   inherits improvement properties thereof." The test-time step can be MORE aggressive than training's: it is tabular
   (no interference with other positions) and its advantage estimates are better (more compute per position).

## The learning dynamics — "dynamically damped self-play"

The core innovation they name: **coordinate regularisation strength with update size.** Strong regularisation and
large updates early; weak regularisation and small updates late; both annealed by power laws.
- *Regularisation:* a max-entropy term (set-ups); a myopic reverse-KL penalty toward a fixed "magnet" policy —
  pick a movable piece uniformly, then a legal move uniformly (moves). Described as an **energy reserve**: annealing too
  cautiously leaves play underdeveloped; too aggressively gives fast early gains, then collapses entropy, "depleting
  its capacity to learn thereafter" and making the policy exploitable.
- *Update size:* four complementary mechanisms — reverse KL to the data-collection policy, importance-ratio clipping,
  gradient-norm clipping, and Adam's learning rate (power-law annealed; "crucial" both early and against late
  plateaus).
- Keeping updates commensurate with regularisation damps the cyclic / divergent / chaotic dynamics imperfect
  information otherwise causes, and lets them train ON-policy or nearly so — no trajectory importance reweighting, no
  policy averaging.

## Smaller findings worth keeping

- **Advantage filtering:** train only on moves with large estimated |advantage|. Cut wall-clock per iteration ~2.5×
  AND raised sample efficiency and asymptotic strength ("a phenomenon meriting further investigation"); removing it
  raised move entropy and cut sample efficiency (Extended Data Fig. 5).
- **Monte Carlo returns beat λ-returns for the set-up network** — "unusual in RL, although similar behaviour has also
  been observed in language model reasoning". λ-returns for moves.
- **EMA of the iterates:** similar or better mean strength, lower variance across seeds (Extended Data Fig. 6).
- A CUDA simulator merged with the rollout buffer (~10M state updates/s; recomputes rather than stores; resets to
  arbitrary states for search).

## What it means for the Neural Turing Architecture (planning doc §19)

- The hidden-information machinery (belief network) does NOT transfer: thinking is one agent with exact,
  deterministic dynamics (planning doc §2). What transfers is the pattern.
- **Update equivalence is the "one general method" principle carried through:** there is no separate search
  algorithm — test-time search is the training update, applied locally with better estimates. For thoughts: a policy
  sample refined by the same damped step training uses (`pi_grad`, §16), with on-policy targets by construction
  (cf. §16 point 8, the tree as a poor teacher).
- **Damped dynamics = a schedule for §3.6's entropy-collapse guard,** and the magnet plays the role the Solomonoff
  prior g₀ plays in the self-play paper (2609.30063: the generator is KL-regularised toward |A|^−ℓ).
- **Advantage filtering is a zone-of-proximal-development filter at the level of single decisions.**
