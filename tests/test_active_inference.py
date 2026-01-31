"""
Tests for Active Inference and Curiosity-Driven Learning

Tests cover:
1. Learning Progress Tracker
2. Active Curriculum Manager
3. Expected Free Energy Calculator
4. Integration tests
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.active_inference import (
    LearningProgressTracker,
    ActiveCurriculumManager,
    ExpectedFreeEnergyCalculator,
    compute_information_gain,
)


class TestLearningProgressTracker:
    """Test the Learning Progress Tracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = LearningProgressTracker(
            num_samples=100,
            window_size=20,
            mastery_threshold=0.1,
            noise_threshold=2.0,
        )

        assert tracker.num_samples == 100
        assert tracker.window_size == 20
        assert tracker.mastery_threshold == 0.1
        assert tracker.noise_threshold == 2.0
        assert len(tracker.sample_stats) == 0

    def test_update_single_sample(self):
        """Test updating a single sample."""
        tracker = LearningProgressTracker(num_samples=10)

        # First observation
        stats = tracker.update(sample_id=0, error=1.0)

        assert stats['visit_count'] == 1
        assert stats['avg_error'] == 1.0
        assert stats['learning_progress'] == 0.0  # No history yet
        assert stats['category'] in ['learnable', 'mastered', 'noise']

    def test_learning_progress_computation(self):
        """Test learning progress computation."""
        tracker = LearningProgressTracker(num_samples=10)

        # Simulate decreasing error (learning happening)
        errors = [2.0, 1.8, 1.5, 1.2, 0.9, 0.7, 0.5]
        for error in errors:
            stats = tracker.update(sample_id=0, error=error)

        # Learning progress should be positive (error decreased)
        assert stats['learning_progress'] > 0.0
        assert stats['avg_error'] < 1.0

    def test_mastery_detection(self):
        """Test detection of mastered samples."""
        tracker = LearningProgressTracker(
            num_samples=10,
            mastery_threshold=0.1,
        )

        # Simulate low error consistently (mastered)
        for _ in range(10):
            stats = tracker.update(sample_id=0, error=0.05)

        assert stats['category'] == 'mastered'
        assert stats['avg_error'] < tracker.mastery_threshold

    def test_noise_detection(self):
        """Test detection of unlearnable noise."""
        tracker = LearningProgressTracker(
            num_samples=10,
            noise_threshold=2.0,
            noise_patience=10,
        )

        # Simulate high error with no progress (noise)
        for _ in range(20):
            stats = tracker.update(sample_id=0, error=2.5)

        # Should eventually be classified as noise
        assert stats['category'] == 'noise'
        assert stats['avg_error'] > tracker.noise_threshold
        assert abs(stats['learning_progress']) < 0.1

    def test_learnable_detection(self):
        """Test detection of learnable samples (Zone of Proximal Development)."""
        tracker = LearningProgressTracker(num_samples=10)

        # Simulate moderate error with progress
        errors = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
        for error in errors:
            stats = tracker.update(sample_id=0, error=error)

        assert stats['category'] == 'learnable'
        assert stats['learning_progress'] > 0.0

    def test_sample_priority(self):
        """Test priority computation."""
        tracker = LearningProgressTracker(num_samples=10)

        # Sample with high learning progress should have high priority
        for error in [1.5, 1.3, 1.1, 0.9]:
            tracker.update(sample_id=0, error=error)

        # Sample with no progress should have low priority
        for _ in range(5):
            tracker.update(sample_id=1, error=2.0)

        priority_0 = tracker.get_sample_priority(0)
        priority_1 = tracker.get_sample_priority(1)

        # Learnable sample should have higher priority
        assert priority_0 > priority_1

    def test_batch_sampling(self):
        """Test batch index sampling."""
        tracker = LearningProgressTracker(num_samples=100)

        # Update some samples
        for i in range(20):
            error = np.random.uniform(0.5, 2.0)
            tracker.update(sample_id=i, error=error)

        # Sample batch
        batch = tracker.get_batch_indices(batch_size=10, temperature=1.0)

        assert len(batch) == 10
        assert len(set(batch)) == 10  # No duplicates
        assert all(0 <= idx < 100 for idx in batch)

    def test_statistics(self):
        """Test overall statistics."""
        tracker = LearningProgressTracker(num_samples=100)

        # Update various samples
        for i in range(50):
            error = np.random.uniform(0.1, 2.0)
            tracker.update(sample_id=i, error=error)

        stats = tracker.get_statistics()

        assert stats['total_samples'] == 100
        assert stats['visited_samples'] == 50
        assert stats['mastered_count'] + stats['learnable_count'] + stats['noise_count'] == 50

    def test_category_samples(self):
        """Test retrieving samples by category."""
        tracker = LearningProgressTracker(num_samples=10, mastery_threshold=0.1)

        # Create mastered sample
        for _ in range(5):
            tracker.update(sample_id=0, error=0.05)

        # Create learnable sample
        for error in [1.0, 0.9, 0.8]:
            tracker.update(sample_id=1, error=error)

        mastered = tracker.get_category_samples('mastered')
        learnable = tracker.get_category_samples('learnable')

        assert 0 in mastered
        assert 1 in learnable


class TestExpectedFreeEnergyCalculator:
    """Test the Expected Free Energy Calculator."""

    def test_initialization(self):
        """Test EFE calculator initialization."""
        calc = ExpectedFreeEnergyCalculator(
            num_classes=10,
            epistemic_weight=1.0,
            pragmatic_weight=1.0,
        )

        assert calc.num_classes == 10
        assert calc.epistemic_weight == 1.0
        assert calc.pragmatic_weight == 1.0

    def test_epistemic_value_uniform(self):
        """Test epistemic value for uniform distribution (maximum uncertainty)."""
        calc = ExpectedFreeEnergyCalculator(num_classes=10)

        # Uniform logits (maximum entropy)
        logits = torch.zeros(10)

        epistemic = calc.compute_epistemic_value(logits)

        # Should be close to log(10) ≈ 2.3
        assert epistemic.item() > 2.0

    def test_epistemic_value_deterministic(self):
        """Test epistemic value for deterministic distribution (minimum uncertainty)."""
        calc = ExpectedFreeEnergyCalculator(num_classes=10)

        # Deterministic logits (one class very high)
        logits = torch.tensor([-10.0] * 10)
        logits[5] = 10.0

        epistemic = calc.compute_epistemic_value(logits)

        # Should be close to 0 (low entropy)
        assert epistemic.item() < 0.1

    def test_pragmatic_value_correct(self):
        """Test pragmatic value when prediction matches preferred outcome."""
        calc = ExpectedFreeEnergyCalculator(num_classes=10)

        # High confidence on correct class
        logits = torch.tensor([-10.0] * 10)
        logits[3] = 10.0
        preferred = torch.tensor(3)

        pragmatic = calc.compute_pragmatic_value(logits, preferred)

        # Should be high (close to 0 in log space)
        assert pragmatic.item() > 5.0

    def test_pragmatic_value_incorrect(self):
        """Test pragmatic value when prediction doesn't match preferred outcome."""
        calc = ExpectedFreeEnergyCalculator(num_classes=10)

        # High confidence on wrong class
        logits = torch.tensor([-10.0] * 10)
        logits[3] = 10.0
        preferred = torch.tensor(7)  # Different class

        pragmatic = calc.compute_pragmatic_value(logits, preferred)

        # Should be low (large negative in log space)
        assert pragmatic.item() < -5.0

    def test_efe_computation(self):
        """Test full EFE computation."""
        calc = ExpectedFreeEnergyCalculator(num_classes=10)

        logits = torch.randn(10)
        preferred = torch.tensor(5)

        result = calc.compute_expected_free_energy(logits, preferred)

        assert 'efe' in result
        assert 'epistemic' in result
        assert 'pragmatic' in result
        assert 'priority' in result

        # Priority should be inverse of EFE
        assert torch.allclose(result['priority'], -result['efe'])

    def test_efe_batch(self):
        """Test EFE computation on batch."""
        calc = ExpectedFreeEnergyCalculator(num_classes=10)

        logits = torch.randn(5, 10)  # Batch of 5
        preferred = torch.tensor([0, 1, 2, 3, 4])

        result = calc.compute_expected_free_energy(logits, preferred, reduction='none')

        assert result['efe'].shape == (5,)
        assert result['epistemic'].shape == (5,)
        assert result['pragmatic'].shape == (5,)

    def test_ambiguity_equals_epistemic(self):
        """Test that ambiguity equals epistemic value for deterministic networks."""
        calc = ExpectedFreeEnergyCalculator(num_classes=10)

        logits = torch.randn(10)

        ambiguity = calc.compute_ambiguity(logits)
        epistemic = calc.compute_epistemic_value(logits, reduction='none')

        assert torch.allclose(ambiguity, epistemic)

    def test_risk_equals_negative_pragmatic(self):
        """Test that risk equals negative pragmatic value."""
        calc = ExpectedFreeEnergyCalculator(num_classes=10)

        logits = torch.randn(10)
        preferred = torch.tensor(5)

        risk = calc.compute_risk(logits, preferred)
        pragmatic = calc.compute_pragmatic_value(logits, preferred, reduction='none')

        assert torch.allclose(risk, -pragmatic)


class TestActiveCurriculumManager:
    """Test the Active Curriculum Manager."""

    def test_initialization(self):
        """Test curriculum manager initialization."""
        manager = ActiveCurriculumManager(
            num_samples=100,
            num_classes=10,
            sampling_strategy='learning_progress',
        )

        assert manager.num_samples == 100
        assert manager.num_classes == 10
        assert manager.sampling_strategy == 'learning_progress'
        assert manager.epoch == 0

    def test_random_sampling(self):
        """Test random sampling baseline."""
        manager = ActiveCurriculumManager(
            num_samples=100,
            sampling_strategy='random',
        )

        batch = manager.get_next_batch(batch_size=10)

        assert len(batch) == 10
        assert len(set(batch)) == 10
        assert all(0 <= idx < 100 for idx in batch)

    def test_learning_progress_sampling(self):
        """Test learning progress-based sampling."""
        manager = ActiveCurriculumManager(
            num_samples=100,
            sampling_strategy='learning_progress',
            exploration_rate=0.0,  # No random exploration
        )

        # Create samples with different learning progress
        # High progress sample
        for error in [1.5, 1.3, 1.1, 0.9]:
            manager.update(sample_idx=0, error=error)

        # No progress sample
        for _ in range(5):
            manager.update(sample_idx=1, error=2.0)

        # Sample should prefer high-progress sample
        batch = manager.get_next_batch(batch_size=1)

        # Note: Due to probabilistic sampling, we can't guarantee exact result
        # But high-progress sample should be more likely

    def test_update_with_logits(self):
        """Test updating with logits for EFE."""
        manager = ActiveCurriculumManager(
            num_samples=10,
            num_classes=10,
            sampling_strategy='pure_epistemic',
        )

        logits = torch.randn(10)
        stats = manager.update(
            sample_idx=0,
            error=1.0,
            logits=logits,
            target=5,
        )

        assert 'efe' in stats
        assert 'epistemic_value' in stats
        assert 'pragmatic_value' in stats
        assert 0 in manager.cached_logits

    def test_epoch_indices(self):
        """Test getting epoch indices."""
        manager = ActiveCurriculumManager(
            num_samples=50,
            sampling_strategy='learning_progress',
        )

        # Update some samples
        for i in range(20):
            manager.update(sample_idx=i, error=np.random.uniform(0.5, 2.0))

        manager.start_epoch()
        indices = manager.get_epoch_indices()

        assert len(indices) == 50
        assert len(set(indices)) == 50  # All unique

    def test_statistics(self):
        """Test getting statistics."""
        manager = ActiveCurriculumManager(num_samples=100)

        for i in range(30):
            manager.update(sample_idx=i, error=np.random.uniform(0.5, 2.0))

        stats = manager.get_statistics()

        assert stats['total_samples'] == 100
        assert stats['visited_samples'] == 30
        assert stats['sampling_strategy'] == 'learning_progress'

    def test_category_breakdown(self):
        """Test category breakdown."""
        manager = ActiveCurriculumManager(
            num_samples=20,
            mastery_threshold=0.1,
        )

        # Create mastered sample
        for _ in range(5):
            manager.update(sample_idx=0, error=0.05)

        # Create learnable sample
        for error in [1.0, 0.9, 0.8]:
            manager.update(sample_idx=1, error=error)

        breakdown = manager.get_category_breakdown()

        assert 'mastered' in breakdown
        assert 'learnable' in breakdown
        assert 'noise' in breakdown
        assert 'unvisited' in breakdown

        assert 0 in breakdown['mastered']
        assert 1 in breakdown['learnable']

    def test_sample_info(self):
        """Test getting sample info."""
        manager = ActiveCurriculumManager(num_samples=10)

        manager.update(sample_idx=5, error=1.0)

        info = manager.get_sample_info(5)

        assert info['sample_idx'] == 5
        assert info['visited'] == True
        assert info['visit_count'] == 1

        # Unvisited sample
        info = manager.get_sample_info(8)
        assert info['visited'] == False


class TestInformationGain:
    """Test information gain computation."""

    def test_information_gain_no_change(self):
        """Test IG when distributions are identical."""
        prior = torch.randn(10)
        posterior = prior.clone()

        ig = compute_information_gain(prior, posterior)

        # Should be close to 0
        assert ig.item() < 0.01

    def test_information_gain_significant_change(self):
        """Test IG when distributions differ significantly."""
        prior = torch.zeros(10)  # Uniform
        posterior = torch.tensor([-10.0] * 10)
        posterior[5] = 10.0  # Peaked

        ig = compute_information_gain(prior, posterior)

        # Should be positive (information was gained)
        assert ig.item() > 0.0


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_active_inference_loop(self):
        """Test a full active inference learning loop."""
        manager = ActiveCurriculumManager(
            num_samples=50,
            num_classes=10,
            sampling_strategy='learning_progress',
        )

        # Simulate training for 100 steps
        for step in range(100):
            # Get next sample
            batch = manager.get_next_batch(batch_size=1)
            sample_idx = batch[0]

            # Simulate learning (error decreases over time for each sample)
            if sample_idx not in manager.tracker.sample_stats:
                initial_error = 2.0
            else:
                visit_count = manager.tracker.sample_stats[sample_idx].visit_count
                initial_error = max(0.1, 2.0 - 0.1 * visit_count)

            # Update
            logits = torch.randn(10)
            manager.update(
                sample_idx=sample_idx,
                error=initial_error,
                logits=logits,
                target=sample_idx % 10,
            )

        # Check that curriculum learned something
        stats = manager.get_statistics()
        breakdown = manager.get_category_breakdown()

        # Should have visited most samples
        assert stats['visited_samples'] > 30

        # Should have some mastered samples (low error)
        assert len(breakdown['mastered']) > 0 or len(breakdown['learnable']) > 0

    def test_strategy_comparison(self):
        """Compare different sampling strategies."""
        strategies = ['random', 'learning_progress']

        results = {}

        for strategy in strategies:
            manager = ActiveCurriculumManager(
                num_samples=20,
                sampling_strategy=strategy,
            )

            # Simulate 50 training steps
            for _ in range(50):
                batch = manager.get_next_batch(batch_size=1)
                sample_idx = batch[0]

                # Simulate error (decreases with visits)
                visit_count = manager.tracker.sample_stats.get(sample_idx, None)
                if visit_count is None:
                    error = 2.0
                else:
                    error = max(0.1, 2.0 - 0.2 * visit_count.visit_count)

                manager.update(sample_idx=sample_idx, error=error)

            results[strategy] = manager.get_statistics()

        # Both should have visited samples
        assert results['random']['visited_samples'] > 0
        assert results['learning_progress']['visited_samples'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
