# Stream of Search (SoS): learning to search in language — reference notes

**Paper:** Kanishk Gandhi, Denise Lee, Gabriel Grand, Muxin Liu, Winson Cheng, Archit Sharma, Noah D. Goodman
(Stanford), arXiv:2404.03683 (1 Apr 2024), COLM 2024. **License:** CC BY 4.0 — the PDF is beside this file
(`stream_of_search_2404.03683.pdf`); source https://arxiv.org/abs/2404.03683. Verified 2026-09-21.

## Abstract (as fetched; the first and last sentences verbatim)

> Language models are rarely shown fruitful mistakes while training. They then struggle to look beyond the next token,
> suffering from a snowballing of errors and struggling to predict the consequence of their actions several steps
> ahead. [The paper shows how to teach language models to search by representing the process of search in language
> as a flattened string — a stream of search; it proposes a unified language for several symbolic search strategies
> and tests it on the Countdown game.] SoS pretraining increases search accuracy by 25% over models trained to predict
> only the optimal search trajectory. [With policy-improvement methods (APA and STaR)] the finetuned SoS models solve
> 36% of previously unsolved problems, including problems that cannot be solved by any of the heuristic solvers.

## What it means for us

The same two-stage shape as Searchformer, with backtracking and mistakes deliberately in the training stream, and the
same second-stage result: self-improvement on its own successes solves what no hand-written strategy could. It is the
strongest evidence that the search strategy should be learned from the agent's own traces. Their search is in
language tokens; the object §18 wants is the same thing in the latent state, so that no vocabulary of search
operations has to be written either.
