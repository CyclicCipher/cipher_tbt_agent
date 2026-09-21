# Latent reasoning and looped planning — what the literature says about thinking without a search procedure

Written 2026-09-21 after the E29 correction: the imagination (`imagine`, `PolicyRule`, the retina, the locus) was a search
procedure written around the model — a thought defined as "one expansion of an imagined frame", contexts defined by
hand, a strategy as an enumerated table, thinking as meta-actions. The rule from here: the only primitives are the
game's controls; the only context is the context window; thinking is the looped block's own computation over its
latent state; what the block does when it loops is written by ZipLearner from experience, never coded. These are the
sources that say how planning arises inside a looped/recurrent model with no search algorithm around it.

## Sources (abstract quotes verified 2026-09-21 unless marked "from memory"; one note file per paper beside this one:
`coconut_2412.06769.md`, `recurrent_depth_2502.05171.md`, `looped_latent_thoughts_2502.17416.md`,
`looped_transformers_programmable_computers_2301.13196.md`, `value_iteration_networks_1602.02867.md`,
`universal_transformers_1807.03819.md`, `adaptive_computation_time_1603.08983.md`, `searchformer_2402.14083.md`,
`stream_of_search_2404.03683.md`, `thinker_2307.14993.md`; PDFs for the three CC BY papers)

**Coconut — Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian (Meta FAIR), "Training Large Language Models to Reason in a
Continuous Latent Space", arXiv:2412.06769 (Dec 2024).** The last hidden state is fed back as the next input
embedding — "the last hidden state of the LLM as a representation of the reasoning state, termed 'continuous
thought.' Instead of decoding this state into words, we feed it back to the model as the next input embedding
directly in the continuous space." The result that matters here: "continuous thoughts can encode multiple
alternative next steps, allowing the model to perform a breadth-first search (BFS) rather than committing
prematurely to a single deterministic path as in CoT." Nobody wrote a BFS: the frontier lives in the superposition of
the latent state, and the model learns to prune it. This is the mechanism the E28 block already has a knob for — the
lookup head's temperature: at M = 30 the block simulates one plan exactly; lowered, with every action in the anchor
slot, each cell's colour becomes the mixture of what it could become, the reachable set spreading with the loops.

**Recurrent depth — Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein, "Scaling
up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach", arXiv:2502.05171 (Feb 2025).** "A novel
language model architecture that is capable of scaling test-time computation by implicitly reasoning in latent
space. Our model works by iterating a recurrent block, thereby unrolling to arbitrary depth at test-time." 3.5B
parameters, 800B tokens, no chain-of-thought data. Training samples the number of iterations from a log-normal
Poisson (mean 32); at test time more iterations keep helping up to 128, hard tasks use 32+, easy ones saturate at
8–12. The halting rule is not a number: "If this divergence falls below 5×10⁻⁴, we stop iterating" — the KL between
successive latent states, i.e. the state has converged. And the latent trajectories have structure nobody asked for:
"many tokens simply converge to a fixed point … the state of the token quickly falls into an orbit pattern … the
trajectory noticeably drifts in a single direction." For us: the loop count is convergence of the state, and the
"5 successes / 4,000 calls" numbers of E29 were the wrong kind of object.

**Looped transformers as latent thoughts — Saunshi, Dikkala, Li, Kumar, Reddi, "Reasoning with Latent Thoughts: On
the Power of Looped Transformers", ICLR 2025, arXiv:2502.17416 (from memory).** A k-layer block looped L times nearly
matches a kL-layer model on reasoning tasks; many reasoning problems need depth, not parameters; each loop is a latent
thought. The design consequence: the written block should stay small and be looped, which is E26/E28's shape.

**Looped transformers as computers — Giannou, Rajput, Sohn, Lee, Lee, Papailiopoulos, "Looped Transformers as
Programmable Computers", ICML 2023, arXiv:2301.13196 (from memory).** Hand-written weights of a ~13-layer looped
transformer implement a general-purpose computer: an instruction set (SUBLEQ), memory read/write by attention,
pointers, conditional branching; the PROGRAM is data in the input, the weights are the interpreter. For us: whatever
procedure a plan needs can be tokens in the context window executed by the loop — a program ZipLearner writes as
data, not code I write around the model. Not the first thing to try (it re-imports "a program"), but the proof that
the substrate loses nothing by forbidding code outside it.

**Value Iteration Networks — Tamar, Wu, Thomas, Levine, Abbeel, NeurIPS 2016 (from memory).** Planning as a
recurrence over the frame: a reward map, then K iterations of (convolution = one-step transition, max over actions =
Bellman backup) produce a value map; the policy reads the value at the agent's position. The planner is a looped
layer over the grid, differentiable, learned end to end, generalising to new maps. This is the closest ancestor of
what the E28 block can do: its gather heads are the convolution (one head per neighbour offset), its lookup head is
the transition, and a backup is a max over the anchor slot — planning by iteration, the horizon = the loop count.

**Universal Transformers with ACT — Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser, ICLR 2019; Graves, "Adaptive
Computation Time", 2016 (from memory).** A looped block with a learned per-position halting probability: how many
loops is an output of the model, trained with a small ponder cost. The learned alternative to Geiping's convergence
rule.

**Search learned from its own traces — Lehnert et al., "Beyond A*: Better Planning with Transformers via Search
Dynamics Bootstrapping" (Searchformer), arXiv:2402.14083; Gandhi et al., "Stream of Search", arXiv:2404.03683 (both
from memory).** Train a transformer on serialised search traces (A* on mazes/Sokoban; DFS/BFS on Countdown), then
bootstrap on its own shorter successful traces: the model internalises the search and then beats the teacher
(Searchformer: fewer search steps than A* on Sokoban; Stream of Search: solves problems the teacher could not). The
search is in tokens, not latent, and it starts from a hand-written teacher — but the second stage is the one that
matters: the search improves by compressing the model's OWN successful traces, which is hindsight, which is the
ZipLearner loop. Our traces are real attempts, not a teacher's.

**What I built instead and why it is out (Chung et al., "Thinker: Learning to Plan and Act", NeurIPS 2023, from
memory).** Thinker augments the MDP with imagination actions (imagine a step, reset, act) and learns a policy over
them by RL. E29's `imagine` was this shape with tables instead of a net. It is excluded by the rule above: the
primitives are the game's controls, and thinking is not an action.

## What follows for the written block

1. The context window is the context: the history of frames (cells as tokens, as in E28) and actions, plus the
   memory tokens ZipLearner writes (the rules; the goal keys; whatever compression of experience yields). No retina,
   no locus: "the cell at offset (di, dj) from the moving thing" is a gather head with a written coordinate permutation
   (E28), "is colour c anywhere to the east" is a head with a directional positional phase (E18). Which offsets and
   directions matter is the sleep pass's choice, written as which heads exist — the retina was these heads done by hand
   outside the model.
2. A thought is one loop. What a loop computes is whatever the written weights compute: with the rules written, a
   loop with one action in the anchor is a simulated step (E28); with all actions in the anchor and a low
   temperature it is Coconut's superposed frontier / VIN's backup; nothing about frames is assumed by the loop itself.
3. The loop count is convergence of the latent state (Geiping), or a written halting head (ACT); never a budget.
4. The output is a game action, read from the converged state by an unembedding ZipLearner writes — the policy, in
   the only form the rule allows: weights, from compressing the successful (context → action) pairs of the agent's
   own history (Searchformer's second stage; E29's 20-bit table is what that compression finds, and belongs in
   memory tokens, not in a Python class).
5. Open, and the actual research question: what does compressing experience write, and does looping the result
   plan? E28 answered it for the forward model. The next experiment must run the block, looped, on the rooms of E29
   with nothing around it but the game — and count whether the goal is reached.
