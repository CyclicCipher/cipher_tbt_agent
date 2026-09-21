# DiffusionBlocks — block-wise neural network training via diffusion interpretation — reference notes

**Paper:** Makoto Shing, Masanori Koyama, Takuya Akiba (Sakana AI), *DiffusionBlocks: Block-wise Neural Network Training
via Diffusion Interpretation*, arXiv:2506.14202 (v1 17 Jun 2025, v4 12 Jun 2026). **License:** not checked — this file is
the abstract verbatim plus notes in our words; source https://arxiv.org/abs/2506.14202. Verified 2026-09-21. The design's
"canvas per block" idea (the network written block by block) came from this paper's earlier version; §20 records what
it settles.

## Abstract (verbatim)

> End-to-end backpropagation requires storing activations throughout all layers, creating memory bottlenecks that limit
> model scalability. Existing block-wise training methods offer means to alleviate this problem, but they rely on ad-hoc
> local objectives and remain largely unexplored beyond classification tasks. We propose DiffusionBlocks, a principled
> framework for transforming transformer-based networks into genuinely independent trainable blocks that maintain
> competitive performance with end-to-end training. Our key insight leverages the fact that residual connections
> naturally correspond to updates in a dynamical system. With minimal modifications to this system, we can convert the
> updates to those of a denoising process, where each block can be learned independently by leveraging the score
> matching objective. This independence enables training with gradients for only one block at a time, thereby reducing
> memory requirements in proportion to the number of blocks. Our experiments on a range of transformer architectures
> (vision, diffusion, autoregressive, recurrent-depth, and masked diffusion) demonstrate that DiffusionBlocks training
> matches the performance of end-to-end training while enabling scalable block-wise training on practical tasks beyond
> small-scale classification. DiffusionBlocks provides a theoretically grounded approach that successfully scales to
> modern generative tasks across diverse architectures.

## What it means for us (DESIGN §20)

Credit assignment through depth is needed only when the intermediate representations are unknown; the diffusion
interpretation defines them (block k: the data at noise level t_k → at t_{k−1}), so every block has observable
(input, target) pairs and can be fitted locally by any method — ZipLearner's counting and sleep pass included — with
no gradient through the stack. The paper trains each block by score matching with gradients; we write each block from
its pairs. Their "recurrent-depth" case is Geiping et al.'s model, i.e. our looped block: the loop is a refinement
chain, the anchor its conditioning, the noise-initialised state its start. E35 is the written form of this on frames
and on text; the continuous rework can host least-squares denoisers on the same schedule.
