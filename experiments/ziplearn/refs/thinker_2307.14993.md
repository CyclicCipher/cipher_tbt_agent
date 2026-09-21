# Thinker: learning to plan and act — reference notes

**Paper:** Stephen Chung, Ivan Anokhin, David Krueger, NeurIPS 2023, arXiv:2307.14993 (v1 27 Jul 2023, v2 26 Oct 2023).
**License:** CC BY 4.0 — the PDF is beside this file (`thinker_2307.14993.pdf`); source https://arxiv.org/abs/2307.14993.
Verified 2026-09-21.

## Abstract (verbatim)

> We propose the Thinker algorithm, a novel approach that enables reinforcement learning agents to autonomously
> interact with and utilize a learned world model. The Thinker algorithm wraps the environment with a world model and
> introduces new actions designed for interacting with the world model. These model-interaction actions enable agents
> to perform planning by proposing alternative plans to the world model before selecting a final action to execute
> in the environment. This approach eliminates the need for handcrafted planning algorithms by enabling the agent to
> learn how to plan autonomously and allows for easy interpretation of the agent's plan with visualization. We
> demonstrate the algorithm's effectiveness through experimental results in the game of Sokoban and the Atari 2600
> benchmark, where the Thinker algorithm achieves state-of-the-art performance and competitive results, respectively.
> Visualizations of agents trained with the Thinker algorithm demonstrate that they have learned to plan effectively
> with the world model to select better actions. Thinker is the first work showing that an RL agent can learn to plan
> with a learned world model in complex environments.

## Why it is recorded here, and why it is out

E29's imagination was this design with tables instead of a network: thinking as extra actions (imagine a step, reset,
commit) and a policy learned over them. Thinker shows the shape works when trained by RL at scale (Sokoban). It is
excluded by the rule of 2026-09-21: the model's only primitives are the game's controls; thinking is not an action
but the block's own computation between observations. Kept as the nearest thing to what we will not build.
