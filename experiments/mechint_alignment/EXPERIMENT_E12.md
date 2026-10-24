# E12: data efficiency against the ideal

**Question (the user):** can we measure the data efficiency of our microworld experiments? Our worlds' rules are
known, so a learner can be compared with an **ideal Bayesian observer** on the same data: something impossible at
language-model scale. Context: the user's thesis that data efficiency is blocked by continual learning (in-context
learning is data-efficient, but cannot be consolidated into weights without forgetting).

## E12a, pre-registered (2026-10-23): the ideal observer for robot motion in the E8 world family

**What is measured.** E8's hidden physics carries about 36.4 bits per world. The largest block is the **action
permutation** (8! = 40,320, 15.3 bits): which action index moves the robot which way. Robot motion also depends on
which shown terrain code is the walkable fire (7 possibilities) and on the push rule (push / block / swap).

**The ideal observer** (`ideal_observer.py`):
- exact Bayes over the joint space perm × fire code × push rule (846,720 hypotheses), with a uniform prior;
- each step it sees what E8's model sees (the observation, the action index, the next observation) and keeps the
  hypotheses whose predicted robot motion matches the observed one. Robot motion is deterministic given the
  hypothesis, so this is the exact posterior;
- its prediction is the posterior predictive's most probable robot outcome;
- one known simplification: when a push would move a villager onto a terrain code whose role is still unknown (it
  could be the pit), the step is treated as uninformative. Such steps are rare. This can only make the observer
  slightly worse than ideal, so the measured gap is a lower bound.

**Scored like the model:**
- the same 64 test and 64 held-out 512-step worlds as E8c (`runs/data_family_long`);
- the same move and turn instances (`village_events` labels);
- the same 8-step chunks;
- the model's per-chunk accuracy curves are read from the E8c logs (`runs/e8/*/log.jsonl`, the end evaluation).

**Apparatus check:** in every world, the true hypothesis (regenerated from the world's seed) must stay consistent at
every step.

**Predictions:**

| # | prediction |
|---|---|
| D1 | The ideal observer is ≥ 99% accurate on move and on turn from chunk 4 on (after 32 transitions), on both splits |
| D2 | E8c's best model (L-Ac, adaptation with the calibrated prior) needs ≥ 10× as many transitions as the ideal to first reach 95% move accuracy |
| D3 | The true hypothesis stays consistent at every step in all 128 worlds |
