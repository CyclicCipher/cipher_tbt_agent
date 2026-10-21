"""A count-based causal-context coder for small images (E9): the perceptual model of the aesthetic agents.

p(x[r, c] | its four causal neighbours: up-left, up, up-right, left), with neighbours beyond the image read as
OUTSIDE. The probabilities are Krichevsky-Trofimov estimates from counts, (n_x + 1/2) / (n + S/2): exact Bayesian
prediction under a Dirichlet(1/2) prior per context, so the code length of an image is a true prequential code
length once the counts were accumulated online.

One canonical component, used two ways:
- LEARNING (the appreciator's perception): code_length on what it sees, then update; the fall of the code length
  over time is its compression progress;
- FROZEN (the maximizer's beauty score): two coders trained once, on examples of beauty and on plain examples; the
  log-likelihood ratio of a canvas is its "beauty" (a learned discriminator, like a reward model).
"""
import numpy as np


class ContextCoder:
    def __init__(self, n_symbols, alpha=0.5, counts=None):
        self.S, self.alpha = n_symbols, alpha
        self.counts = np.zeros(((n_symbols + 1) ** 4, n_symbols), np.float32) if counts is None else counts

    def copy(self):
        return ContextCoder(self.S, self.alpha, self.counts.copy())

    def contexts(self, img):
        """The context index of every cell [h, w]."""
        S1 = self.S + 1
        p = np.full((img.shape[0] + 1, img.shape[1] + 2), self.S, np.int64)       # OUTSIDE = S
        p[1:, 1:-1] = img
        ul, u, ur, left = p[:-1, :-2], p[:-1, 1:-1], p[:-1, 2:], p[1:, :-2]
        return ((ul * S1 + u) * S1 + ur) * S1 + left

    def bits(self, img, ctx=None):
        """Code length of every cell [h, w] in bits under the current counts (not updated)."""
        ctx = self.contexts(img) if ctx is None else ctx
        rows = self.counts[ctx]                                                    # [h, w, S]
        num = np.take_along_axis(rows, img[..., None].astype(np.int64), -1)[..., 0] + self.alpha
        return -np.log2(num / (rows.sum(-1) + self.alpha * self.S))

    def update(self, img, ctx=None):
        ctx = self.contexts(img) if ctx is None else ctx
        np.add.at(self.counts, (ctx.ravel(), img.ravel().astype(np.int64)), 1.0)
