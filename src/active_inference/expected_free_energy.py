"""
Expected Free Energy (EFE) Calculator for Active Inference

Based on Friston's Active Inference framework:
- Friston et al. (2015): Active Inference and Epistemic Value
- Friston et al. (2017): Active Inference, Curiosity and Insight
- Friston et al. (2020): Generalized Free Energy and Active Inference

Mathematical Foundation:

Free Energy (Variational):
    F = -E_Q(s)[ln P(o,s) - ln Q(s)]
      = D_KL[Q(s) || P(s|o)] - ln P(o)

Expected Free Energy (for policy selection):
    G(π) = E_Q(o,s|π)[ln Q(s|π) - ln P(o,s)]
         = E_Q(o,s|π)[ln Q(s|o,π) - ln Q(s|π)] + E_Q(o|π)[ln Q(o|π) - ln P(o)]
         = Ambiguity + Risk
         = (Epistemic Value) + (Pragmatic Value)

Decomposition:
    G(π) = E_Q(o|π)[H[Q(s|o,π)]] - E_Q(o,s|π)[ln P(o|s)]
           \_________________/     \______________________/
              Ambiguity                   Risk
           (Epistemic Value)         (Pragmatic Value)

For classification tasks:
    - Epistemic Value: H[p(y|x)] = Uncertainty in prediction
    - Pragmatic Value: -E[ln P(y*|x)] where y* is preferred outcome

Lower EFE → Higher priority (minimize expected free energy)
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import torch.nn.functional as F


class ExpectedFreeEnergyCalculator:
    """
    Computes Expected Free Energy for sample selection in active inference.

    EFE guides which samples to observe next by balancing:
    1. Epistemic value: Reduce uncertainty (exploration)
    2. Pragmatic value: Achieve preferred outcomes (exploitation)
    """

    def __init__(
        self,
        num_classes: int,
        epistemic_weight: float = 1.0,
        pragmatic_weight: float = 1.0,
        temperature: float = 1.0,
        use_model_uncertainty: bool = True,
    ):
        """
        Args:
            num_classes: Number of output classes
            epistemic_weight: Weight for epistemic value (exploration)
            pragmatic_weight: Weight for pragmatic value (exploitation)
            temperature: Softmax temperature for probability distributions
            use_model_uncertainty: If True, use model's predictive uncertainty
        """
        self.num_classes = num_classes
        self.epistemic_weight = epistemic_weight
        self.pragmatic_weight = pragmatic_weight
        self.temperature = temperature
        self.use_model_uncertainty = use_model_uncertainty

        # Track statistics
        self.efe_history = []

    def compute_epistemic_value(
        self,
        logits: torch.Tensor,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """
        Compute epistemic value (information gain / uncertainty reduction).

        Epistemic value = H[p(y|x)] = -∑ p(y|x) log p(y|x)

        Higher entropy → Higher epistemic value → Should sample this

        Args:
            logits: (batch_size, num_classes) or (num_classes,) model predictions
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Epistemic value (scalar or tensor)
        """
        # Convert logits to probabilities
        probs = F.softmax(logits / self.temperature, dim=-1)

        # Compute entropy: H[p] = -∑ p log p
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)

        if reduction == 'mean':
            return torch.mean(entropy)
        elif reduction == 'sum':
            return torch.sum(entropy)
        else:
            return entropy

    def compute_pragmatic_value(
        self,
        logits: torch.Tensor,
        preferred_outcome: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """
        Compute pragmatic value (expected utility / goal achievement).

        If preferred_outcome is provided:
            Pragmatic value = -KL[p(y|x) || δ(y*)]
                            = log p(y*|x)
                            = Expected log probability of preferred outcome

        If preferred_outcome is None (unsupervised):
            Use uniform prior or skip pragmatic component

        Higher pragmatic value → prediction aligns with goals → Should sample this

        Args:
            logits: (batch_size, num_classes) or (num_classes,) model predictions
            preferred_outcome: (batch_size,) or scalar - preferred class labels
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Pragmatic value (scalar or tensor)
        """
        if preferred_outcome is None:
            # Unsupervised case: no preferred outcome
            # Use uniform prior (all outcomes equally preferred)
            # Pragmatic value = log(1/K) = -log K (constant)
            return torch.zeros(logits.shape[0] if logits.dim() > 1 else 1, device=logits.device)

        # Supervised case: compute log probability of preferred outcome
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)

        if logits.dim() == 1:
            # Single sample
            pragmatic_value = log_probs[preferred_outcome]
        else:
            # Batch
            pragmatic_value = log_probs[torch.arange(logits.shape[0]), preferred_outcome]

        if reduction == 'mean':
            return torch.mean(pragmatic_value)
        elif reduction == 'sum':
            return torch.sum(pragmatic_value)
        else:
            return pragmatic_value

    def compute_expected_free_energy(
        self,
        logits: torch.Tensor,
        preferred_outcome: Optional[torch.Tensor] = None,
        reduction: str = 'none',
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Expected Free Energy (EFE) for sample selection.

        EFE = -Epistemic_Value - Pragmatic_Value
            = -H[p(y|x)] - log p(y*|x)

        Lower EFE → Higher priority for sampling

        Args:
            logits: (batch_size, num_classes) or (num_classes,) model predictions
            preferred_outcome: (batch_size,) or scalar - preferred class labels
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Dictionary with:
                - efe: Expected free energy
                - epistemic: Epistemic value component
                - pragmatic: Pragmatic value component
                - priority: Inverted EFE (higher = more priority)
        """
        # Compute components
        epistemic = self.compute_epistemic_value(logits, reduction=reduction)
        pragmatic = self.compute_pragmatic_value(logits, preferred_outcome, reduction=reduction)

        # EFE = -(weighted epistemic) - (weighted pragmatic)
        # We want to MINIMIZE EFE, which means MAXIMIZE (epistemic + pragmatic)
        efe = -(self.epistemic_weight * epistemic + self.pragmatic_weight * pragmatic)

        # For sampling, we want high-priority samples (low EFE)
        # Convert to priority score (higher = better)
        priority = -efe  # Invert: lower EFE → higher priority

        return {
            'efe': efe,
            'epistemic': epistemic,
            'pragmatic': pragmatic,
            'priority': priority,
        }

    def compute_ambiguity(
        self,
        logits: torch.Tensor,
        num_samples: int = 10,
    ) -> torch.Tensor:
        """
        Compute ambiguity term: E_Q(o|π)[H[Q(s|o,π)]]

        This is the expected entropy of the posterior over states,
        given observations under a policy.

        For deterministic networks, this is approximated by the
        entropy of the output distribution.

        For stochastic networks (e.g., with dropout), we can sample
        multiple forward passes and compute the expected entropy.

        Args:
            logits: (batch_size, num_classes) predictions
            num_samples: Number of stochastic forward passes (if applicable)

        Returns:
            Ambiguity score
        """
        # For now, use simple entropy (deterministic case)
        # TODO: Implement stochastic sampling for uncertainty estimation
        return self.compute_epistemic_value(logits, reduction='none')

    def compute_risk(
        self,
        logits: torch.Tensor,
        preferred_outcome: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute risk term: E_Q(o,s|π)[-ln P(o|s)]

        This is the expected negative log probability of observations
        under the model.

        Risk measures how well the policy achieves preferred outcomes.

        Args:
            logits: (batch_size, num_classes) predictions
            preferred_outcome: Preferred class labels

        Returns:
            Risk score
        """
        # Risk = -log P(preferred_outcome | state)
        # This is the negative pragmatic value
        pragmatic = self.compute_pragmatic_value(logits, preferred_outcome, reduction='none')
        return -pragmatic

    def select_samples_by_efe(
        self,
        logits_list: list,
        num_samples: int,
        preferred_outcomes: Optional[list] = None,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select samples to observe based on EFE.

        Args:
            logits_list: List of logits for each sample in the dataset
            num_samples: Number of samples to select
            preferred_outcomes: Optional list of preferred outcomes
            temperature: Softmax temperature for selection

        Returns:
            selected_indices: Indices of selected samples
            priorities: Priority scores for selected samples
        """
        # Compute EFE for all samples
        priorities = []

        for i, logits in enumerate(logits_list):
            if preferred_outcomes is not None:
                preferred = preferred_outcomes[i]
            else:
                preferred = None

            result = self.compute_expected_free_energy(
                logits,
                preferred_outcome=preferred,
                reduction='none'
            )

            priorities.append(result['priority'].item())

        priorities = np.array(priorities)

        # Select top-k by priority (highest priority = lowest EFE)
        # OR sample probabilistically
        if temperature == 0.0:
            # Greedy: select top-k
            selected_indices = np.argsort(priorities)[-num_samples:][::-1]
        else:
            # Softmax sampling
            exp_priorities = np.exp(priorities / temperature)
            probabilities = exp_priorities / exp_priorities.sum()
            selected_indices = np.random.choice(
                len(logits_list),
                size=num_samples,
                replace=False,
                p=probabilities,
            )

        return selected_indices, priorities[selected_indices]

    def get_statistics(self) -> Dict:
        """Get statistics about EFE computation."""
        if not self.efe_history:
            return {
                'mean_efe': 0.0,
                'std_efe': 0.0,
                'count': 0,
            }

        efe_array = np.array(self.efe_history)
        return {
            'mean_efe': float(np.mean(efe_array)),
            'std_efe': float(np.std(efe_array)),
            'min_efe': float(np.min(efe_array)),
            'max_efe': float(np.max(efe_array)),
            'count': len(self.efe_history),
        }


def compute_information_gain(
    prior_logits: torch.Tensor,
    posterior_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute information gain (KL divergence from prior to posterior).

    IG = KL[P(s|o) || P(s)] = H[P(s)] - H[P(s|o)]

    This measures how much observing o reduces uncertainty about s.

    Args:
        prior_logits: Predictions before observing sample
        posterior_logits: Predictions after observing sample

    Returns:
        Information gain
    """
    prior_probs = F.softmax(prior_logits, dim=-1)
    posterior_probs = F.softmax(posterior_logits, dim=-1)

    # KL divergence: KL[Q||P] = ∑ Q log(Q/P)
    kl = F.kl_div(
        F.log_softmax(prior_logits, dim=-1),
        posterior_probs,
        reduction='batchmean',
        log_target=False,
    )

    return kl


def compute_expected_information_gain(
    model,
    sample_candidates: list,
    device: torch.device,
) -> np.ndarray:
    """
    Compute expected information gain for each sample candidate.

    EIG(x) = E_{p(y|x)}[KL[p(θ|x,y) || p(θ|x)]]

    This is expensive to compute exactly, so we approximate with:
    EIG(x) ≈ H[p(y|x)]  (predictive entropy)

    Args:
        model: Predictive model
        sample_candidates: List of candidate samples
        device: torch device

    Returns:
        Array of EIG scores
    """
    model.eval()
    eig_scores = []

    with torch.no_grad():
        for sample in sample_candidates:
            # Get model prediction
            if isinstance(sample, torch.Tensor):
                sample = sample.to(device)
            else:
                sample = torch.tensor(sample, device=device)

            logits = model(sample.unsqueeze(0) if sample.dim() == 1 else sample)

            # Approximate EIG with entropy
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs).item()

            eig_scores.append(entropy)

    return np.array(eig_scores)
