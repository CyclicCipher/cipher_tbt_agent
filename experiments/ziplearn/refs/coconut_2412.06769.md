# Coconut — Chain of Continuous Thought — reference notes

**Paper:** Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian (Meta FAIR),
*Training Large Language Models to Reason in a Continuous Latent Space*, arXiv:2412.06769 (v1 9 Dec 2024, v2 11 Dec 2024).
**License:** arXiv perpetual non-exclusive licence — this file is the abstract verbatim plus notes in our words; the paper
is at https://arxiv.org/abs/2412.06769 (HTML: https://arxiv.org/html/2412.06769v2). Verified 2026-09-21.

## Abstract (verbatim)

> Large language models (LLMs) are restricted to reason in the "language space", where they typically express the
> reasoning process with a chain-of-thought (CoT) to solve a complex reasoning problem. However, we argue that language
> space may not always be optimal for reasoning. For example, most word tokens are primarily for textual coherence and
> not essential for reasoning, while some critical tokens require complex planning and pose huge challenges to LLMs. To
> explore the potential of LLM reasoning in an unrestricted latent space instead of using natural language, we
> introduce a new paradigm Coconut (Chain of Continuous Thought). We utilize the last hidden state of the LLM as a
> representation of the reasoning state (termed "continuous thought"). Rather than decoding this into a word token, we
> feed it back to the LLM as the subsequent input embedding directly in the continuous space. Experiments show that
> Coconut can effectively augment the LLM on several reasoning tasks. This novel latent reasoning paradigm leads to
> emergent advanced reasoning patterns: the continuous thought can encode multiple alternative next reasoning steps,
> allowing the model to perform a breadth-first search (BFS) to solve the problem, rather than prematurely committing
> to a single deterministic path like CoT. Coconut outperforms CoT in certain logical reasoning tasks that require
> substantial backtracking during planning, with fewer thinking tokens during inference. These findings demonstrate
> the promise of latent reasoning and offer valuable insights for future research.

## The mechanism

The recurrence is along the SEQUENCE: the last hidden state at position t is used as the input embedding at position
t+1 (a "continuous thought" position), for c such positions between the question and the answer; ordinary attention
lets each thought read the question and the earlier thoughts. Nothing is decoded between thoughts. Because a hidden
state is a vector, not a token, it can hold several candidate next steps at once — the paper's analysis on ProsQA
(graph reasoning) reads the thoughts as a frontier whose branches are weighted by their value and pruned over
successive thoughts: a breadth-first search nobody wrote.

## Difficulties the paper reports (verified quotes)

- Training needs a curriculum: "at the k-th stage, the first k reasoning steps in the CoT are replaced with k×c
  continuous thoughts" — and without it, "the models trained this way do not perform any better than no-CoT."
- Instability with more thoughts per step: "When experimenting with c=3, we observe a slight performance drop
  accompanied by increased variance... leads to a sharp spike in training loss, causing instability."
- The number of thoughts is not learned: "a) train a binary classifier on latent thoughts to enable the model to
  autonomously decide when to terminate the latent reasoning, or b) always pad the latent thoughts to a constant
  length. We found that both approaches work comparably well. Therefore, we use the second option."
- Arithmetic word problems (GSM8k): 34.1% against CoT's 42.9% (above iCoT's 30.0%); the wins are on the logical
  planning tasks (ProntoQA, ProsQA).
- The raw last hidden state is fed back with no normalisation between thoughts; the state that comes back is not
  from the distribution of token embeddings. The curriculum is partly what teaches the model to read it. (Our
  reading of the setup, not a claim of the paper.)

## What it means for the written block (DESIGN §18)

The thing to copy is not the training recipe but the object: a thought is a hidden state that stays a hidden state,
and a frontier is a superposition inside it. E28's lookup head has that knob — its temperature. At M = 30 the block
simulates one plan exactly; lowered, with every action in the anchor slot, each cell's colour subspace becomes the
mixture of what it could become, and the loops spread the reachable set. The difficulties transfer as design
constraints: normalise the state that is fed back (the boundary operator applied along the sequence axis, as Chen
et al. apply it along depth), re-inject the question/goal every thought (attention over the context does this for
free; Geiping et al. show why it is necessary), and let the number of thoughts be a convergence test, not a pad.
