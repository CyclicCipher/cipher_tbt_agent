# Universal Transformers — reference notes

**Paper:** Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Łukasz Kaiser, ICLR 2019, arXiv:1807.03819
(v1 10 Jul 2018, v3 5 Mar 2019). **License:** arXiv non-exclusive distribution — abstract verbatim plus notes; source
https://arxiv.org/abs/1807.03819. Verified 2026-09-21.

## Abstract (verbatim)

> Recurrent neural networks (RNNs) sequentially process data by updating their state with each new data point, and
> have long been the de facto choice for sequence modeling tasks. However, their inherently sequential computation
> makes them slow to train. Feed-forward and convolutional architectures have recently been shown to achieve superior
> results on some sequence modeling tasks such as machine translation, with the added advantage that they concurrently
> process all inputs in the sequence, leading to easy parallelization and faster training times. Despite these
> successes, however, popular feed-forward sequence models like the Transformer fail to generalize in many simple
> tasks that recurrent models handle with ease, e.g. copying strings or even simple logical inference when the string
> or formula lengths exceed those observed at training time. We propose the Universal Transformer (UT), a
> parallel-in-time self-attentive recurrent sequence model which can be cast as a generalization of the Transformer
> model and which addresses these issues. UTs combine the parallelizability and global receptive field of feed-forward
> sequence models like the Transformer with the recurrent inductive bias of RNNs. We also add a dynamic per-position
> halting mechanism and find that it improves accuracy on several tasks. In contrast to the standard Transformer, under
> certain assumptions, UTs can be shown to be Turing-complete. Our experiments show that UTs outperform standard
> Transformers on a wide range of algorithmic and language understanding tasks, including the challenging LAMBADA
> language modeling task where UTs achieve a new state of the art, and machine translation where UTs achieve a 0.9
> BLEU improvement over Transformers on the WMT14 En-De dataset.

## What it means for us

The original looped transformer (one block applied recurrently over depth, a timestep embedding added per step), with
Graves's Adaptive Computation Time as the per-position halting rule: how many loops a position gets is an output of the
model, trained with a small ponder cost. This is the LEARNED alternative to Geiping et al.'s convergence test, and
the two are the candidates for the block's halting rule (DESIGN §18); the timestep embedding is a precedent for a
per-pass anchor that tells the block which loop it is in.
