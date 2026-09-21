# Looped transformers as programmable computers — reference notes

**Paper:** Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D. Lee, Dimitris Papailiopoulos,
ICML 2023, arXiv:2301.13196 (30 Jan 2023). **License:** arXiv non-exclusive distribution — abstract verbatim plus
notes; source https://arxiv.org/abs/2301.13196. Verified 2026-09-21.

## Abstract (verbatim)

> We present a framework for using transformer networks as universal computers by programming them with specific
> weights and placing them in a loop. Our input sequence acts as a punchcard, consisting of instructions and memory
> for data read/writes. We demonstrate that a constant number of encoder layers can emulate basic computing blocks,
> including embedding edit operations, non-linear functions, function calls, program counters, and conditional
> branches. Using these building blocks, we emulate a small instruction-set computer. This allows us to map iterative
> algorithms to programs that can be executed by a looped, 13-layer transformer. We show how this transformer,
> instructed by its input, can emulate a basic calculator, a basic linear algebra library, and in-context learning
> algorithms that employ backpropagation. Our work highlights the versatility of the attention mechanism, and
> demonstrates that even shallow transformers can execute full-fledged, general-purpose programs.

## The mechanism (from memory of the paper; the abstract above is the verified part)

The weights are written, not trained: a 13-layer block, looped, is an interpreter for a one-instruction-set computer
(SUBLEQ: subtract and branch if ≤ 0, which is Turing-complete). The input sequence holds the program and the memory;
attention with hard-coded positional codes implements read/write by pointer, a program counter advanced each loop,
and conditional branching by editing the counter. Everything the machine does is decided by the input; the weights
never change.

## What it means for us (DESIGN §18) — "exactly the idea"

The written block is an interpreter and knowledge is a program in the context window. E28 already has this shape in
miniature: the lookup head is one instruction ("match the window, write the colour"), the memory tokens are the
program, the loop executes it once per action. Planning, valuing, exploring are then also programs — tokens
ZipLearner writes by compressing experience — executed by the same loop, and the search for the right program is a
search over descriptions priced in bits, not code around the model. What the paper does not give: how programs are
found (theirs are written by the authors); that is ZipLearner's job, and it is the open question of §18.
