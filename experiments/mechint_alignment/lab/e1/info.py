"""Information measures of training (E1): what a policy holds, what the training signal carries, what a
gradient carries. All in bits.

- policy_info: entropy H(pi(.|s)), the marginal action entropy, and the policy's information rate
  I(S;A) = H(mean_s pi(.|s)) - mean_s H(pi(.|s)): how many bits per decision depend on the state
  ("policy complexity", Lai & Gershman 2021). A constant bias (always go east) is not information about the state.
- kl_bits: KL(p || q) per state, averaged: e.g. the bits per decision still separating a checkpoint from the final
  policy. "Acquired" = KL(final || init) - KL(final || now).
- reward_bits: the empirical entropy of the reward symbols x their count: the code length of the reward stream
  with no model, an upper bound on what the reward channel delivered.
- gns: the gradient noise scale B_simple = tr(Sigma)/|G|^2 (McCandlish et al. 2018, "An Empirical Model of
  Large-Batch Training", App. A) from per-minibatch gradients. The signal-to-noise of one sample's gradient is
  1/B_simple; read as a Gaussian channel, one sample carries about 0.5*log2(1 + 1/B_simple) bits about the
  gradient direction (a heuristic, one dimension).
"""
import math

import numpy as np
import torch

from nets.policy import amp

LN2 = math.log(2)


@torch.no_grad()
def log_probs(net, dev, cells, glob, chunk=1024):
    """Action log-probabilities [S, A] (fp32, on CPU) for numpy observations."""
    out = []
    for i in range(0, len(cells), chunk):
        with amp(dev):
            logits, _ = net(torch.from_numpy(cells[i:i + chunk]).to(dev), torch.from_numpy(glob[i:i + chunk]).to(dev))
        out.append(logits.float().log_softmax(-1).cpu())
    return torch.cat(out)


def policy_info(logp, ref_actions=None):
    p = logp.exp()
    h = -(p * logp).sum(-1) / LN2
    pbar = p.mean(0)
    h_marg = float(-(pbar * pbar.clamp_min(1e-12).log()).sum() / LN2)
    out = dict(H_bits=float(h.mean()), H_marg_bits=h_marg, I_SA_bits=h_marg - float(h.mean()))
    if ref_actions is not None:
        ref = torch.as_tensor(ref_actions, dtype=torch.long)
        out["ref_agree"] = float((p.argmax(-1) == ref).float().mean())
        out["ref_prob"] = float(p.gather(1, ref[:, None]).mean())
    return out


def kl_bits(logp, logq):
    """mean_s KL(p(.|s) || q(.|s)) in bits."""
    return float((logp.exp() * (logp - logq)).sum(-1).mean() / LN2)


def reward_bits(rewards):
    """(bits per step, total bits) of a reward array, by its empirical symbol distribution (rounded to 1e-3)."""
    _, counts = np.unique(np.round(np.asarray(rewards), 3), return_counts=True)
    p = counts / counts.sum()
    h = float(-(p * np.log2(p)).sum())
    return h, h * counts.sum()


def gns(sq_sum, vec_sum, k, b):
    """B_simple from k minibatch gradients of size b: sq_sum = sum_i |g_i|^2, vec_sum = sum_i g_i (a flat tensor)."""
    big = k * b
    g_small = sq_sum / k
    g_big = float((vec_sum / k).pow(2).sum())
    g2 = (big * g_big - b * g_small) / (big - b)
    tr_sigma = (g_small - g_big) / (1 / b - 1 / big)
    b_simple = tr_sigma / g2 if g2 > 0 else float("inf")
    return dict(G2=g2, trSigma=tr_sigma, B_simple=b_simple,
                bits_per_sample=0.5 * math.log2(1 + 1 / b_simple) if b_simple > 0 else float("nan"))
