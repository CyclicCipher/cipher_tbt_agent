# Scaling up test-time compute with latent reasoning: a recurrent depth approach — reference notes

**Paper:** Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya
Kailkhura, Abhinav Bhatele, Tom Goldstein, arXiv:2502.05171 (v1 7 Feb 2025, v2 17 Feb 2025). Code and weights are public
(the "Huginn" model). **License:** arXiv non-exclusive distribution — abstract verbatim plus notes; the paper is at
https://arxiv.org/abs/2502.05171 (HTML: https://arxiv.org/html/2502.05171v1). Verified 2026-09-21.

## Abstract (verbatim)

> We study a novel language model architecture that is capable of scaling test-time computation by implicitly
> reasoning in latent space. Our model works by iterating a recurrent block, thereby unrolling to arbitrary depth at
> test-time. This stands in contrast to mainstream reasoning models that scale up compute by producing more tokens.
> Unlike approaches based on chain-of-thought, our approach does not require any specialized training data, can work
> with small context windows, and can capture types of reasoning that are not easily represented in words. We scale a
> proof-of-concept model to 3.5 billion parameters and 800 billion tokens. We show that the resulting model can
> improve its performance on reasoning benchmarks, sometimes dramatically, up to a computation load equivalent to 50
> billion parameters.

## The mechanism (verified quotes)

Prelude P (embed) → recurrent core R iterated → coda C (decode): the same shape as Chen et al. 2609.19107 and as our
`LoopedModel`. The core "accepts the latent state s_{i-1} and the embedded input e and outputs a new latent state
s_i" — the input is injected at EVERY iteration, and the paper says why: "if e was provided only at the start, e.g.
via s_0 = e, then the iterative process would not be stable" because "R cannot be a monotone operator if it does not
depend on e, and so cannot represent gradient descent on strictly convex, data-dependent functions." The initial
state is noise, "s_0 ~ N(0, σ² I)", to promote "convergence to a steady state independent of initialization, i.e.
path independence." Normalisation is "sandwich" (norm before and after attention and MLP); "normalization is required
to train the recurrence at scale." Training samples the iteration count from a log-normal Poisson with mean 32 and
backpropagates through the last k = 8 iterations only.

## Halting and what the loops do (verified quotes)

Test-time compute is adaptive with no training for it: the KL divergence between successive latent states is
monitored and "If this divergence falls below 5×10⁻⁴, we stop iterating, sample the output token, and move to
generate the next token." More iterations keep helping up to 128; hard tasks (GSM8k) use 32+, easy ones saturate
at 8–12. The latent trajectories have structure nobody trained for: "many tokens simply converge to a fixed point …
the state of the token quickly falls into an orbit pattern … the trajectory noticeably drifts in a single direction."

## What it means for us (DESIGN §18)

Three rules, each with a stated reason: (1) the anchor is not optional — the input must be re-injected every pass or
the recurrence is not stable, which is what Chen et al.'s boundary operator and E28's per-pass anchor do; (2) the loop
count is a convergence test on the state, not a number (E29's "5 successes / 4,000 calls" were the wrong kind of
object); (3) a random initial state, so that the answer is a fixed point of the map and not a function of where the
iteration started. The "drift" to watch for in training a recurrence is the state leaving the input behind — the
injection is the cure, and the norms keep the magnitudes bounded.
