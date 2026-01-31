"""
Active Curriculum Manager for Data-Efficient Learning

Replaces random batch sampling with intelligent sample selection based on:
1. Learning Progress (Oudeyer et al.)
2. Expected Free Energy (Friston) - to be integrated
3. Compression Progress (Schmidhuber)

The curriculum manager decides which samples to show the model next,
prioritizing those in the "Zone of Proximal Development" where learning
is most efficient.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import torch
from .learning_progress import LearningProgressTracker
from .expected_free_energy import ExpectedFreeEnergyCalculator


class ActiveCurriculumManager:
    """
    Manages training curriculum using active inference principles.

    Instead of random sampling, this manager:
    1. Tracks learning progress for each sample
    2. Prioritizes samples with high learning potential
    3. Avoids wasting time on mastered or unlearnable samples
    4. Implements exploration-exploitation balance
    """

    def __init__(
        self,
        num_samples: int,
        num_classes: int = 10,
        sampling_strategy: str = 'learning_progress',
        temperature: float = 1.0,
        exploration_rate: float = 0.1,
        batch_size: int = 1,
        epistemic_weight: float = 1.0,
        pragmatic_weight: float = 1.0,
        **tracker_kwargs,
    ):
        """
        Args:
            num_samples: Total number of samples in dataset
            num_classes: Number of output classes (for EFE)
            sampling_strategy: How to select samples:
                - 'random': Standard random sampling (baseline)
                - 'learning_progress': Prioritize high learning progress
                - 'pure_epistemic': Maximize uncertainty reduction (EFE-based)
                - 'balanced': Mix epistemic and pragmatic value (EFE-based)
            temperature: Softmax temperature (higher = more exploration)
            exploration_rate: Probability of random sampling (epsilon-greedy)
            batch_size: Default batch size for sampling
            epistemic_weight: Weight for epistemic value (exploration)
            pragmatic_weight: Weight for pragmatic value (exploitation)
            **tracker_kwargs: Additional args for LearningProgressTracker
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.sampling_strategy = sampling_strategy
        self.temperature = temperature
        self.exploration_rate = exploration_rate
        self.batch_size = batch_size

        # Initialize learning progress tracker
        self.tracker = LearningProgressTracker(num_samples, **tracker_kwargs)

        # Initialize Expected Free Energy calculator
        self.efe_calculator = ExpectedFreeEnergyCalculator(
            num_classes=num_classes,
            epistemic_weight=epistemic_weight,
            pragmatic_weight=pragmatic_weight,
            temperature=temperature,
        )

        # Cache for model predictions (for EFE-based sampling)
        self.cached_logits = {}

        # Statistics
        self.epoch = 0
        self.samples_seen = 0

    def update(
        self,
        sample_idx: int,
        error: float,
        logits: Optional[torch.Tensor] = None,
        target: Optional[int] = None,
        additional_stats: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Update statistics after processing a sample.

        Args:
            sample_idx: Index of the sample that was processed
            error: Prediction error (loss) for this sample
            logits: Optional model output logits (for EFE computation)
            target: Optional true label (for EFE computation)
            additional_stats: Optional dict with extra information
                (e.g., layer-wise errors, free energy)

        Returns:
            Statistics dict from the tracker
        """
        self.samples_seen += 1

        # Update learning progress tracker
        stats = self.tracker.update(sample_idx, error)

        # Cache logits for EFE-based sampling
        if logits is not None:
            # Detach and move to CPU to save GPU memory
            self.cached_logits[sample_idx] = {
                'logits': logits.detach().cpu(),
                'target': target,
            }

            # Compute EFE if using EFE-based strategies
            if self.sampling_strategy in ['pure_epistemic', 'balanced']:
                efe_result = self.efe_calculator.compute_expected_free_energy(
                    logits,
                    preferred_outcome=torch.tensor([target]) if target is not None else None,
                    reduction='none',
                )
                stats['efe'] = efe_result['efe'].item()
                stats['epistemic_value'] = efe_result['epistemic'].item()
                stats['pragmatic_value'] = efe_result['pragmatic'].item()
                stats['efe_priority'] = efe_result['priority'].item()

        # Store additional stats if provided
        if additional_stats is not None:
            stats.update(additional_stats)

        return stats

    def get_next_batch(
        self,
        batch_size: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        dataset: Optional[any] = None,
        device: Optional[torch.device] = None,
    ) -> List[int]:
        """
        Get the next batch of sample indices to train on.

        Args:
            batch_size: Size of batch (uses default if None)
            model: Optional model (needed for EFE-based sampling)
            dataset: Optional dataset (needed for EFE-based sampling)
            device: Optional device (needed for EFE-based sampling)

        Returns:
            List of sample indices
        """
        if batch_size is None:
            batch_size = self.batch_size

        if self.sampling_strategy == 'random':
            # Baseline: random sampling
            return list(np.random.choice(self.num_samples, size=batch_size, replace=False))

        elif self.sampling_strategy == 'learning_progress':
            # Active inference: prioritize high learning progress
            return self.tracker.get_batch_indices(
                batch_size=batch_size,
                temperature=self.temperature,
                exploration_rate=self.exploration_rate,
            )

        elif self.sampling_strategy == 'pure_epistemic':
            # EFE-based: Maximize epistemic value (pure exploration)
            return self._get_batch_by_efe(
                batch_size=batch_size,
                model=model,
                dataset=dataset,
                device=device,
                epistemic_weight=1.0,
                pragmatic_weight=0.0,
            )

        elif self.sampling_strategy == 'balanced':
            # EFE-based: Balance epistemic and pragmatic value
            return self._get_batch_by_efe(
                batch_size=batch_size,
                model=model,
                dataset=dataset,
                device=device,
                epistemic_weight=self.efe_calculator.epistemic_weight,
                pragmatic_weight=self.efe_calculator.pragmatic_weight,
            )

        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def _get_batch_by_efe(
        self,
        batch_size: int,
        model: Optional[torch.nn.Module] = None,
        dataset: Optional[any] = None,
        device: Optional[torch.device] = None,
        epistemic_weight: float = 1.0,
        pragmatic_weight: float = 1.0,
    ) -> List[int]:
        """
        Get batch indices using Expected Free Energy.

        Args:
            batch_size: Number of samples to select
            model: Predictive model
            dataset: Training dataset
            device: Torch device
            epistemic_weight: Weight for epistemic value
            pragmatic_weight: Weight for pragmatic value

        Returns:
            List of sample indices
        """
        # If we have cached logits, use them
        if len(self.cached_logits) > 0:
            # Use cached logits for efficiency
            priorities = []

            for sample_idx in range(self.num_samples):
                if sample_idx in self.cached_logits:
                    cache = self.cached_logits[sample_idx]
                    logits = cache['logits']
                    target = cache['target']

                    # Compute EFE
                    efe_result = self.efe_calculator.compute_expected_free_energy(
                        logits,
                        preferred_outcome=torch.tensor([target]) if target is not None else None,
                        reduction='none',
                    )
                    priority = efe_result['priority'].item()
                else:
                    # Not yet seen: give high exploration priority
                    priority = 10.0

                priorities.append(priority)

            priorities = np.array(priorities)

        elif model is not None and dataset is not None and device is not None:
            # Compute fresh predictions for all samples (expensive!)
            priorities = self._compute_efe_for_all_samples(
                model=model,
                dataset=dataset,
                device=device,
            )

        else:
            # Fall back to learning progress
            return self.tracker.get_batch_indices(
                batch_size=batch_size,
                temperature=self.temperature,
                exploration_rate=self.exploration_rate,
            )

        # Sample based on priorities
        if np.random.random() < self.exploration_rate:
            # Exploration: random sampling
            return list(np.random.choice(self.num_samples, size=batch_size, replace=False))

        # Softmax sampling based on priorities
        priorities = priorities / self.temperature
        priorities = priorities - priorities.max()  # Numerical stability
        exp_priorities = np.exp(priorities)
        probabilities = exp_priorities / exp_priorities.sum()

        indices = np.random.choice(
            self.num_samples,
            size=batch_size,
            replace=False,
            p=probabilities,
        )

        return list(indices)

    def _compute_efe_for_all_samples(
        self,
        model: torch.nn.Module,
        dataset: any,
        device: torch.device,
    ) -> np.ndarray:
        """
        Compute EFE for all samples in the dataset.

        Warning: This is expensive! Only use when necessary.

        Args:
            model: Predictive model
            dataset: Training dataset
            device: Torch device

        Returns:
            Array of EFE priorities
        """
        model.eval()
        priorities = []

        with torch.no_grad():
            for sample_idx in range(self.num_samples):
                data, target = dataset[sample_idx]
                data = data.unsqueeze(0).to(device)

                # Get model prediction
                logits = model(data, target=None, num_iterations=10)

                # Compute EFE
                efe_result = self.efe_calculator.compute_expected_free_energy(
                    logits.squeeze(0),
                    preferred_outcome=torch.tensor(target),
                    reduction='none',
                )

                priorities.append(efe_result['priority'].item())

        model.train()
        return np.array(priorities)

    def get_epoch_indices(
        self,
        prioritize_learnable: bool = True,
    ) -> List[int]:
        """
        Get all indices for one epoch, ordered by priority.

        Args:
            prioritize_learnable: If True, learnable samples come first

        Returns:
            List of all sample indices in priority order
        """
        if self.sampling_strategy == 'random':
            # Random shuffle (standard epoch)
            indices = list(range(self.num_samples))
            np.random.shuffle(indices)
            return indices

        # Active sampling: order by priority
        priorities = np.array([
            self.tracker.get_sample_priority(i)
            for i in range(self.num_samples)
        ])

        # Sort by priority (descending)
        sorted_indices = np.argsort(priorities)[::-1]

        if prioritize_learnable:
            # Further organize: learnable > mastered > noise
            learnable = self.tracker.get_category_samples('learnable')
            mastered = self.tracker.get_category_samples('mastered')
            noise = self.tracker.get_category_samples('noise')

            # Within each category, sort by priority
            learnable_priorities = {i: priorities[i] for i in learnable}
            mastered_priorities = {i: priorities[i] for i in mastered}
            noise_priorities = {i: priorities[i] for i in noise}

            learnable_sorted = sorted(learnable, key=lambda i: learnable_priorities.get(i, 0), reverse=True)
            mastered_sorted = sorted(mastered, key=lambda i: mastered_priorities.get(i, 0), reverse=True)
            noise_sorted = sorted(noise, key=lambda i: noise_priorities.get(i, 0), reverse=True)

            # Concatenate: learnable first, then mastered, then noise
            ordered_indices = learnable_sorted + mastered_sorted + noise_sorted

            # Add any samples not yet visited
            visited = set(ordered_indices)
            unvisited = [i for i in range(self.num_samples) if i not in visited]
            ordered_indices = unvisited + ordered_indices  # Unvisited first (exploration)

            return ordered_indices

        return list(sorted_indices)

    def start_epoch(self):
        """Mark the start of a new epoch."""
        self.epoch += 1

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the curriculum."""
        tracker_stats = self.tracker.get_statistics()

        return {
            **tracker_stats,
            'epoch': self.epoch,
            'samples_seen': self.samples_seen,
            'sampling_strategy': self.sampling_strategy,
            'temperature': self.temperature,
            'exploration_rate': self.exploration_rate,
        }

    def get_category_breakdown(self) -> Dict[str, List[int]]:
        """
        Get sample indices grouped by category.

        Returns:
            Dict with keys 'learnable', 'mastered', 'noise', 'unvisited'
        """
        learnable = self.tracker.get_category_samples('learnable')
        mastered = self.tracker.get_category_samples('mastered')
        noise = self.tracker.get_category_samples('noise')

        visited = set(learnable + mastered + noise)
        unvisited = [i for i in range(self.num_samples) if i not in visited]

        return {
            'learnable': learnable,
            'mastered': mastered,
            'noise': noise,
            'unvisited': unvisited,
        }

    def should_focus_on_learnable(self, threshold: float = 0.8) -> bool:
        """
        Determine if we should focus exclusively on learnable samples.

        This implements an adaptive curriculum: early on, explore everything,
        but once we've categorized most samples, focus on the learnable ones.

        Args:
            threshold: Fraction of samples that must be visited to switch to focus mode

        Returns:
            True if we should focus on learnable samples only
        """
        stats = self.tracker.get_statistics()
        visited_fraction = stats['visited_samples'] / stats['total_samples']

        return visited_fraction >= threshold

    def get_sample_info(self, sample_idx: int) -> Dict:
        """
        Get detailed information about a specific sample.

        Args:
            sample_idx: Index of the sample

        Returns:
            Dict with sample statistics
        """
        if sample_idx not in self.tracker.sample_stats:
            return {
                'sample_idx': sample_idx,
                'visited': False,
                'visit_count': 0,
                'priority': self.tracker.get_sample_priority(sample_idx),
                'category': 'unvisited',
            }

        stats = self.tracker.sample_stats[sample_idx]
        error_array = np.array(stats.error_history)
        avg_error = float(np.mean(error_array))
        learning_progress = self.tracker._compute_learning_progress(stats)
        category = self.tracker._classify_sample(stats, avg_error, learning_progress)

        return {
            'sample_idx': sample_idx,
            'visited': True,
            'visit_count': stats.visit_count,
            'avg_error': avg_error,
            'error_std': float(np.std(error_array)) if len(error_array) > 1 else 0.0,
            'learning_progress': learning_progress,
            'category': category,
            'priority': self.tracker.get_sample_priority(sample_idx),
            'last_visited_step': stats.last_visited_step,
            'error_history': list(stats.error_history),
        }

    def print_status(self):
        """Print a human-readable status summary."""
        stats = self.get_statistics()
        breakdown = self.get_category_breakdown()

        print(f"\n{'='*60}")
        print(f"Active Curriculum Manager Status (Epoch {stats['epoch']})")
        print(f"{'='*60}")
        print(f"Strategy: {stats['sampling_strategy']}")
        print(f"Temperature: {stats['temperature']:.2f} | Exploration: {stats['exploration_rate']:.2%}")
        print(f"\nSample Coverage:")
        print(f"  Visited: {stats['visited_samples']}/{stats['total_samples']} "
              f"({stats['visited_samples']/stats['total_samples']:.1%})")
        print(f"  Total visits: {stats['total_visits']} "
              f"(avg {stats['avg_visit_count']:.1f} per visited sample)")
        print(f"\nSample Categories:")
        print(f"  Learnable: {stats['learnable_count']} ({len(breakdown['learnable'])}) "
              f"← Zone of Proximal Development")
        print(f"  Mastered:  {stats['mastered_count']} ({len(breakdown['mastered'])}) "
              f"← Already learned")
        print(f"  Noise:     {stats['noise_count']} ({len(breakdown['noise'])}) "
              f"← Unlearnable")
        print(f"  Unvisited: {len(breakdown['unvisited'])} ← Need exploration")
        print(f"{'='*60}\n")
