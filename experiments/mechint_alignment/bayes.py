"""The Laplace (Gaussian) posterior over a network's weights: sequential Bayesian learning, Bayesian R3 (B-R3).

posterior after data D  ~  N(mu, diag(1/Lambda)),   Lambda_j = N * Fbar_j + lam0
  mu     : the current weights (the posterior mode)
  Fbar_j : the mean per-sample empirical Fisher of weight j, E_i[(d l_i / d theta_j)^2], with l_i the training loss
           of datum i (so the prior and a new datum's likelihood are on the same scale)
  N      : how many data the posterior summarises (each counted ONCE)
  lam0   : the base prior precision

Learning a new datum (or batch) = the MAP of [old posterior x its likelihood]:
    minimize  l_new(theta) + 1/2 * sum_j Lambda_j (theta_j - mu_j)^2
optimized to convergence. The datum counts once however many iterations are run -- no rationalization by
over-counting -- and the penalty protects whatever past data pinned down -- no interference. Afterwards the datum is
absorbed: Lambda += its Fisher, mu = the new mode (the "retire" of R3, made exact).
"""
import torch


def per_sample_fisher(model, loss_fn, batches, params=None):
    """Diagonal empirical Fisher: the mean over single samples of the squared gradient of loss_fn(model, sample).
    `batches` yields one-sample inputs for loss_fn. Returns {name: tensor}."""
    params = dict(model.named_parameters()) if params is None else params
    fisher = {n: torch.zeros_like(p) for n, p in params.items()}
    count = 0
    model.eval()
    for sample in batches:
        model.zero_grad(set_to_none=True)
        loss_fn(model, sample).backward()
        for n, p in params.items():
            if p.grad is not None:
                fisher[n] += p.grad.detach().float() ** 2
        count += 1
    return {n: f / max(count, 1) for n, f in fisher.items()}


class LaplacePrior:
    """The old posterior as a quadratic penalty around its mode."""

    def __init__(self, model, fisher, n_data, lam0):
        self.mu = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.prec = {n: n_data * fisher[n] + lam0 for n in self.mu}

    def penalty(self, model):
        return 0.5 * sum((self.prec[n] * (p - self.mu[n]) ** 2).sum() for n, p in model.named_parameters())

    def bayesian_surprise(self, model):
        """KL(new posterior || old) in the Laplace approximation, dominated by 1/2 (theta - mu)' Lambda (theta - mu):
        how much the update actually changed the beliefs, in nats."""
        with torch.no_grad():
            return float(self.penalty(model))

    def absorb(self, model, fisher_new, n_new=1):
        """Retire: the update becomes the new prior (mode moves, precision grows by the new data's Fisher)."""
        for n, p in model.named_parameters():
            self.mu[n] = p.detach().clone()
            self.prec[n] = self.prec[n] + n_new * fisher_new[n]
