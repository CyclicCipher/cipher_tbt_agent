# Adaptive Computation Time for recurrent neural networks — reference notes

**Paper:** Alex Graves (DeepMind), arXiv:1603.08983 (v1 29 Mar 2016, v6 21 Feb 2017). **License:** arXiv non-exclusive
distribution — abstract verbatim plus notes; source https://arxiv.org/abs/1603.08983. Verified 2026-09-21.

## Abstract (verbatim)

> This paper introduces Adaptive Computation Time (ACT), an algorithm that allows recurrent neural networks to learn
> how many computational steps to take between receiving an input and emitting an output. ACT requires minimal
> changes to the network architecture, is deterministic and differentiable, and does not add any noise to the
> parameter gradients. Experimental results are provided for four synthetic problems: determining the parity of
> binary vectors, applying binary logic operations, adding integers, and sorting real numbers. Overall, performance
> is dramatically improved by the use of ACT, which successfully adapts the number of computational steps to the
> requirements of the problem. We also present character-level language modelling results on the Hutter prize
> Wikipedia dataset. In this case ACT does not yield large gains in performance; however it does provide intriguing
> insight into the structure of the data, with more computation allocated to harder-to-predict transitions, such as
> spaces between words and ends of sentences. This suggests that ACT or other adaptive computation methods could
> provide a generic method for inferring segment boundaries in sequence data.

## The mechanism (from memory of the paper)

At each step the network also emits a halting probability; steps continue until the cumulative probability passes
1 − ε; the output is the halting-probability-weighted mean of the intermediate outputs; a "ponder cost" (the number
of steps plus the remainder) is added to the loss with a small weight τ so the network prefers fewer steps. The
compute allocated follows difficulty (more at word boundaries in language modelling).

## What it means for us

A halting rule that is a written or learned head rather than a threshold on the state's change: the block emits
"done" when the plan is ready. In the description-length frame, τ is the price of a thought; ACT's ponder cost is
the first place where "the value of a computation" was made a term in an objective, which is what §18 says our
budgets must become.
