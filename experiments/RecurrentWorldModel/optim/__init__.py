"""Optimizers for the transformer-improvement experiments.

`SNRAdamW` implements the signal/noise gate from Litman & Guo 2026, "A Theory of
Generalization in Deep Learning" (arXiv:2605.01172): gate each parameter's update
by a per-parameter signal-to-noise test, suppressing the memorization (noise)
directions. Two ways to estimate the SNR:
  * mode="ema"       -- cheap across-step EMA of gradient variance (one extra buffer)
  * mode="faithful"  -- true within-batch per-example gradient variance (torch.func)
  * mode="none"      -- plain AdamW control
"""

from .snr import SNRAdamW, per_example_snr_gate

__all__ = ["SNRAdamW", "per_example_snr_gate"]
