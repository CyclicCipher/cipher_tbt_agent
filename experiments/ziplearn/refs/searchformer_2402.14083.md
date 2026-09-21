# Searchformer — Beyond A*: better planning with transformers via search dynamics bootstrapping — reference notes

**Paper:** Lucas Lehnert, Sainbayar Sukhbaatar, DiJia Su, Qinqing Zheng, Paul McVay, Michael Rabbat, Yuandong Tian
(Meta FAIR), arXiv:2402.14083 (v1 21 Feb 2024, v2 26 Apr 2024). **License:** arXiv non-exclusive distribution —
abstract verbatim plus notes; source https://arxiv.org/abs/2402.14083. Verified 2026-09-21.

## Abstract (verbatim)

> While Transformers have enabled tremendous progress in various application settings, such architectures still
> trail behind traditional symbolic planners for solving complex decision making tasks. In this work, we demonstrate
> how to train Transformers to solve complex planning tasks. This is accomplished by training an encoder-decoder
> Transformer model to predict the search dynamics of the A* search algorithm. We fine tune this model to obtain a
> Searchformer, a Transformer model that optimally solves previously unseen Sokoban puzzles 93.7% of the time, while
> using up to 26.8% fewer search steps than the A* implementation that was used for training initially. In our
> training method, A*'s search dynamics are expressed as a token sequence outlining when task states are added and
> removed into the search tree during symbolic planning. Searchformer significantly outperforms baselines that
> predict the optimal plan directly with a 5-10× smaller model size and a 10× smaller training dataset. Lastly, we
> demonstrate how Searchformer scales to larger and more complex decision making tasks with improved percentage of
> solved tasks and shortened search dynamics.

## What it means for us

Two stages: imitate a hand-written search's trace (A*), then bootstrap on the model's OWN shorter successful traces
until it beats the teacher. The first stage is what §18 forbids (a search algorithm supplied from outside); the second
is the ZipLearner loop itself — compress the successful traces of one's own behaviour — which E29 did in a table and
§18 says must be written into the block. Their traces are tokens (search steps in language); ours would be latent.
The lesson kept: search learned from experience improves past any teacher, so the teacher is not needed if the first
successes can be had by exploration (E27) — which is how our games are played.
