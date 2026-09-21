# Value Iteration Networks — reference notes

**Paper:** Aviv Tamar, Yi Wu, Garrett Thomas, Sergey Levine, Pieter Abbeel, NeurIPS 2016 (best paper), arXiv:1602.02867
(v1 9 Feb 2016, v4 20 Mar 2017). **License:** arXiv non-exclusive distribution — abstract verbatim plus notes; source
https://arxiv.org/abs/1602.02867. Verified 2026-09-21.

## Abstract (verbatim)

> We introduce the value iteration network (VIN): a fully differentiable neural network with a 'planning module'
> embedded within. VINs can learn to plan, and are suitable for predicting outcomes that involve planning-based
> reasoning, such as policies for reinforcement learning. Key to our approach is a novel differentiable approximation
> of the value-iteration algorithm, which can be represented as a convolutional neural network, and trained
> end-to-end using standard backpropagation. We evaluate VIN based policies on discrete and continuous path-planning
> domains, and on a natural-language based search task. We show that by learning an explicit planning computation,
> VIN policies generalize better to new, unseen domains.

## The mechanism (from memory of the paper)

Value iteration V ← max_a [R + γ Σ P(s'|s,a) V(s')] on a grid is one convolution (the transition kernel, one channel
per action) followed by a max over channels; K iterations of that layer = K steps of value iteration = a looped layer
over the frame. A learned reward map feeds it; an attention over the value map at the agent's position feeds a
reactive policy; all of it trained end to end from demonstrations. Because the planning is an explicit iteration, the
policy transfers to maps it has never seen.

## What it means for us (DESIGN §18)

The nearest ancestor of what the E28 block can do: its gather heads are the convolution (one head per neighbour
offset), its lookup head is the transition, a backup is a max over the anchor slot, and the horizon is the loop
count. Planning by iteration on the frame, in latent space, with no tree and no procedure — the loop IS the planner.
The difference: VIN's kernel is learned by gradients from demonstrations; ours would be written from the rules
ZipLearner already has, so nothing new is learned for the planner to exist.
