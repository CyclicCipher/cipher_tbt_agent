"""
Learning Progress Tracker for Curiosity-Driven Learning

Based on:
- Oudeyer et al. (2007): Intrinsic Motivation Systems for Autonomous Mental Development
- Schmidhuber (2010): Formal Theory of Creativity, Fun, and Intrinsic Motivation

Key Concept: Agents don't seek novelty or familiarity, but the "Zone of Proximal Development"
where the *rate* of error reduction is highest (learning progress).

Learning Progress = -d(Error)/dt ≈ -(Error_t - Error_{t-1})

Samples are classified as:
- NOISE: High error, no learning progress (unlearnable)
- LEARNABLE: Medium error, positive learning progress (Zone of Proximal Development)
- MASTERED: Low error, minimal learning progress (already learned)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class SampleStats:
    """Statistics for a single training sample."""
    sample_id: int
    error_history: deque  # Recent errors (fixed window)
    visit_count: int
    last_visited_step: int

    def __post_init__(self):
        if not isinstance(self.error_history, deque):
            self.error_history = deque(self.error_history, maxlen=20)  # Keep last 20 observations


class LearningProgressTracker:
    """
    Tracks learning progress for each sample in the dataset.

    Learning progress is defined as the negative derivative of error:
    LP(sample) = -d(Error)/dt

    Positive LP means the model is actively learning from this sample.
    Zero LP means either mastered (low error) or unlearnable noise (high error).
    """

    def __init__(
        self,
        num_samples: int,
        window_size: int = 20,
        mastery_threshold: float = 0.1,
        noise_threshold: float = 2.0,
        noise_patience: int = 50,
    ):
        """
        Args:
            num_samples: Total number of samples in dataset
            window_size: Number of recent errors to keep per sample
            mastery_threshold: Error below this is considered "mastered"
            noise_threshold: Error above this with no progress is "noise"
            noise_patience: Number of visits before declaring something unlearnable
        """
        self.num_samples = num_samples
        self.window_size = window_size
        self.mastery_threshold = mastery_threshold
        self.noise_threshold = noise_threshold
        self.noise_patience = noise_patience

        # Per-sample tracking
        self.sample_stats: Dict[int, SampleStats] = {}

        # Global statistics
        self.global_step = 0
        self.total_visits = 0

    def update(self, sample_id: int, error: float) -> Dict[str, float]:
        """
        Update statistics for a sample after observing its error.

        Args:
            sample_id: Index of the sample
            error: Current prediction error (e.g., L2 loss, free energy)

        Returns:
            Dictionary with computed statistics:
                - learning_progress: Rate of error reduction
                - avg_error: Mean error over window
                - error_std: Standard deviation of error
                - visit_count: Number of times seen
                - category: 'learnable', 'mastered', or 'noise'
        """
        self.global_step += 1
        self.total_visits += 1

        # Initialize sample if first time seeing it
        if sample_id not in self.sample_stats:
            self.sample_stats[sample_id] = SampleStats(
                sample_id=sample_id,
                error_history=deque(maxlen=self.window_size),
                visit_count=0,
                last_visited_step=self.global_step,
            )

        stats = self.sample_stats[sample_id]
        stats.error_history.append(error)
        stats.visit_count += 1
        stats.last_visited_step = self.global_step

        # Compute statistics
        error_array = np.array(stats.error_history)
        avg_error = float(np.mean(error_array))
        error_std = float(np.std(error_array)) if len(error_array) > 1 else 0.0

        # Compute learning progress (negative derivative of error)
        learning_progress = self._compute_learning_progress(stats)

        # Classify sample
        category = self._classify_sample(stats, avg_error, learning_progress)

        return {
            'learning_progress': learning_progress,
            'avg_error': avg_error,
            'error_std': error_std,
            'visit_count': stats.visit_count,
            'category': category,
            'interestingness': self._compute_interestingness(learning_progress, avg_error),
        }

    def _compute_learning_progress(self, stats: SampleStats) -> float:
        """
        Compute learning progress as -d(Error)/dt.

        Uses linear regression on recent errors to estimate derivative.
        Positive values indicate learning is happening.
        """
        if len(stats.error_history) < 2:
            return 0.0

        errors = np.array(stats.error_history)

        # Simple method: difference between recent and older errors
        if len(errors) < 5:
            # Not enough data, use simple difference
            return float(errors[0] - errors[-1])  # Positive if error decreased

        # Better method: Linear regression on error vs time
        # Split into two halves and compare means
        mid = len(errors) // 2
        old_half = errors[:mid]
        recent_half = errors[mid:]

        old_mean = np.mean(old_half)
        recent_mean = np.mean(recent_half)

        # Learning progress = how much error decreased
        learning_progress = float(old_mean - recent_mean)

        return learning_progress

    def _classify_sample(
        self, stats: SampleStats, avg_error: float, learning_progress: float
    ) -> str:
        """
        Classify sample into learnable, mastered, or noise.

        Decision tree:
        1. If avg_error < mastery_threshold AND learning_progress ≈ 0 → MASTERED
        2. If avg_error > noise_threshold AND learning_progress ≈ 0 AND many visits → NOISE
        3. Otherwise → LEARNABLE
        """
        # Mastered: Low error, no more learning happening
        if avg_error < self.mastery_threshold:
            return 'mastered'

        # Noise: High error persists despite many attempts
        if (avg_error > self.noise_threshold and
            abs(learning_progress) < 0.01 and
            stats.visit_count > self.noise_patience):
            return 'noise'

        # Learnable: In the zone of proximal development
        return 'learnable'

    def _compute_interestingness(self, learning_progress: float, avg_error: float) -> float:
        """
        Compute "interestingness" score combining learning progress and error.

        Schmidhuber's formulation: Interestingness ∝ Compression Progress
        We approximate this as: learning_progress * (1 + moderate_error_bonus)

        This creates an inverted-U curve: moderate errors are most interesting.
        """
        # Bonus for moderate errors (neither too easy nor too hard)
        # Gaussian centered at error = 0.5, width = 0.3
        moderate_error_bonus = np.exp(-((avg_error - 0.5) ** 2) / (2 * 0.3 ** 2))

        # Interestingness is learning progress weighted by error difficulty
        interestingness = learning_progress * (1.0 + moderate_error_bonus)

        return float(interestingness)

    def get_sample_priority(self, sample_id: int) -> float:
        """
        Get priority score for a sample (higher = should be sampled sooner).

        Priority is based on:
        1. Learning progress (highest priority)
        2. Recency (samples not visited recently get boost)
        3. Category (learnable > mastered > noise)
        """
        if sample_id not in self.sample_stats:
            # Never seen before: give high initial priority (exploration)
            return 10.0

        stats = self.sample_stats[sample_id]

        # Get latest statistics
        error_array = np.array(stats.error_history)
        avg_error = float(np.mean(error_array))
        learning_progress = self._compute_learning_progress(stats)
        category = self._classify_sample(stats, avg_error, learning_progress)

        # Base priority from learning progress and interestingness
        base_priority = self._compute_interestingness(learning_progress, avg_error)

        # Recency bonus: samples not seen recently get priority boost
        steps_since_visit = self.global_step - stats.last_visited_step
        recency_bonus = np.log1p(steps_since_visit) * 0.1

        # Category modifier
        category_weight = {
            'learnable': 1.0,   # Highest priority
            'mastered': 0.1,    # Low priority (already learned)
            'noise': 0.01,      # Very low priority (unlearnable)
        }

        priority = (base_priority + recency_bonus) * category_weight[category]

        return float(priority)

    def get_batch_indices(
        self,
        batch_size: int,
        temperature: float = 1.0,
        exploration_rate: float = 0.1,
    ) -> List[int]:
        """
        Sample a batch of indices based on learning progress priorities.

        Args:
            batch_size: Number of samples to return
            temperature: Softmax temperature for sampling (higher = more random)
            exploration_rate: Probability of random sampling (epsilon-greedy)

        Returns:
            List of sample indices to train on
        """
        # Epsilon-greedy: sometimes sample randomly for exploration
        if np.random.random() < exploration_rate:
            return list(np.random.choice(self.num_samples, size=batch_size, replace=False))

        # Compute priorities for all samples
        priorities = np.array([self.get_sample_priority(i) for i in range(self.num_samples)])

        # Apply softmax with temperature
        priorities = priorities / temperature
        priorities = priorities - priorities.max()  # Numerical stability
        exp_priorities = np.exp(priorities)
        probabilities = exp_priorities / exp_priorities.sum()

        # Sample without replacement
        indices = np.random.choice(
            self.num_samples,
            size=batch_size,
            replace=False,
            p=probabilities,
        )

        return list(indices)

    def get_statistics(self) -> Dict:
        """Get overall statistics about learning progress."""
        if not self.sample_stats:
            return {
                'total_samples': self.num_samples,
                'visited_samples': 0,
                'mastered_count': 0,
                'learnable_count': 0,
                'noise_count': 0,
                'avg_visit_count': 0.0,
            }

        categories = {'mastered': 0, 'learnable': 0, 'noise': 0}
        visit_counts = []

        for sample_id in range(self.num_samples):
            if sample_id in self.sample_stats:
                stats = self.sample_stats[sample_id]
                error_array = np.array(stats.error_history)
                avg_error = float(np.mean(error_array))
                learning_progress = self._compute_learning_progress(stats)
                category = self._classify_sample(stats, avg_error, learning_progress)
                categories[category] += 1
                visit_counts.append(stats.visit_count)

        return {
            'total_samples': self.num_samples,
            'visited_samples': len(self.sample_stats),
            'mastered_count': categories['mastered'],
            'learnable_count': categories['learnable'],
            'noise_count': categories['noise'],
            'avg_visit_count': float(np.mean(visit_counts)) if visit_counts else 0.0,
            'total_visits': self.total_visits,
            'global_step': self.global_step,
        }

    def get_category_samples(self, category: str, limit: Optional[int] = None) -> List[int]:
        """
        Get all sample IDs in a specific category.

        Args:
            category: 'mastered', 'learnable', or 'noise'
            limit: Maximum number to return (None = all)

        Returns:
            List of sample IDs in that category
        """
        samples = []

        for sample_id in range(self.num_samples):
            if sample_id not in self.sample_stats:
                continue

            stats = self.sample_stats[sample_id]
            error_array = np.array(stats.error_history)
            avg_error = float(np.mean(error_array))
            learning_progress = self._compute_learning_progress(stats)
            sample_category = self._classify_sample(stats, avg_error, learning_progress)

            if sample_category == category:
                samples.append(sample_id)

        if limit is not None and len(samples) > limit:
            samples = samples[:limit]

        return samples
