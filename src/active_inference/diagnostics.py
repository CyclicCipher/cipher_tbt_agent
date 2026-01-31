"""
Diagnostics and Visualization for Active Inference Training

Provides tools to monitor and visualize:
1. Learning progress over time
2. Sample category evolution (learnable → mastered)
3. Expected Free Energy dynamics
4. Curriculum attention patterns
5. Data efficiency metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import torch
from collections import defaultdict


class ActiveInferenceDiagnostics:
    """
    Tracks and visualizes active inference training dynamics.

    Monitors:
    - Per-sample learning curves
    - Category distribution over time
    - Sample visit frequency
    - EFE evolution
    - Exploration vs exploitation balance
    """

    def __init__(self, num_samples: int, num_classes: int):
        """
        Args:
            num_samples: Total number of training samples
            num_classes: Number of output classes
        """
        self.num_samples = num_samples
        self.num_classes = num_classes

        # History tracking
        self.error_history = defaultdict(list)  # sample_idx -> [errors]
        self.visit_history = defaultdict(list)  # sample_idx -> [step numbers]
        self.category_history = []  # [(step, mastered, learnable, noise, unvisited)]
        self.efe_history = defaultdict(list)  # sample_idx -> [efe values]
        self.epistemic_history = defaultdict(list)  # sample_idx -> [epistemic values]
        self.pragmatic_history = defaultdict(list)  # sample_idx -> [pragmatic values]

        # Global statistics
        self.global_step = 0
        self.epoch_boundaries = []  # Step numbers where epochs ended

    def update(
        self,
        sample_idx: int,
        error: float,
        category: str,
        efe: Optional[float] = None,
        epistemic: Optional[float] = None,
        pragmatic: Optional[float] = None,
    ):
        """
        Update diagnostics after processing a sample.

        Args:
            sample_idx: Index of the sample
            error: Prediction error
            category: Sample category ('mastered', 'learnable', 'noise')
            efe: Expected free energy (optional)
            epistemic: Epistemic value (optional)
            pragmatic: Pragmatic value (optional)
        """
        self.global_step += 1

        # Record error
        self.error_history[sample_idx].append(error)

        # Record visit
        self.visit_history[sample_idx].append(self.global_step)

        # Record EFE components
        if efe is not None:
            self.efe_history[sample_idx].append(efe)
        if epistemic is not None:
            self.epistemic_history[sample_idx].append(epistemic)
        if pragmatic is not None:
            self.pragmatic_history[sample_idx].append(pragmatic)

    def record_epoch_boundary(self):
        """Mark the end of an epoch."""
        self.epoch_boundaries.append(self.global_step)

    def record_category_distribution(
        self,
        mastered: int,
        learnable: int,
        noise: int,
        unvisited: int,
    ):
        """
        Record category distribution at current step.

        Args:
            mastered: Number of mastered samples
            learnable: Number of learnable samples
            noise: Number of noise samples
            unvisited: Number of unvisited samples
        """
        self.category_history.append(
            (self.global_step, mastered, learnable, noise, unvisited)
        )

    def plot_learning_curves(
        self,
        sample_indices: Optional[List[int]] = None,
        max_samples: int = 10,
        figsize: Tuple[int, int] = (12, 6),
    ):
        """
        Plot learning curves for selected samples.

        Args:
            sample_indices: Specific samples to plot (None = auto-select interesting ones)
            max_samples: Maximum number of samples to plot
            figsize: Figure size
        """
        if sample_indices is None:
            # Auto-select: samples with most visits
            visit_counts = {idx: len(visits) for idx, visits in self.visit_history.items()}
            sample_indices = sorted(visit_counts.keys(), key=lambda x: visit_counts[x], reverse=True)[:max_samples]

        fig, ax = plt.subplots(figsize=figsize)

        for sample_idx in sample_indices:
            if sample_idx in self.error_history:
                errors = self.error_history[sample_idx]
                visits = self.visit_history[sample_idx]
                ax.plot(visits, errors, marker='o', alpha=0.7, label=f'Sample {sample_idx}')

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Prediction Error')
        ax.set_title('Learning Curves: Error vs Training Step')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_category_evolution(
        self,
        figsize: Tuple[int, int] = (12, 6),
    ):
        """
        Plot how sample categories evolve over training.

        Shows the number of samples in each category over time.
        """
        if not self.category_history:
            print("No category history to plot")
            return None

        steps, mastered, learnable, noise, unvisited = zip(*self.category_history)

        fig, ax = plt.subplots(figsize=figsize)

        ax.fill_between(steps, 0, unvisited, alpha=0.3, label='Unvisited', color='gray')
        ax.fill_between(steps, unvisited, [u + n for u, n in zip(unvisited, noise)],
                        alpha=0.3, label='Noise', color='red')
        ax.fill_between(steps, [u + n for u, n in zip(unvisited, noise)],
                        [u + n + l for u, n, l in zip(unvisited, noise, learnable)],
                        alpha=0.3, label='Learnable (ZPD)', color='orange')
        ax.fill_between(steps, [u + n + l for u, n, l in zip(unvisited, noise, learnable)],
                        [u + n + l + m for u, n, l, m in zip(unvisited, noise, learnable, mastered)],
                        alpha=0.3, label='Mastered', color='green')

        # Mark epoch boundaries
        for epoch_step in self.epoch_boundaries:
            ax.axvline(epoch_step, color='black', linestyle='--', alpha=0.5)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Category Evolution: Sample Distribution Over Time')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_visit_distribution(
        self,
        figsize: Tuple[int, int] = (12, 6),
    ):
        """
        Plot histogram of sample visit frequencies.

        Shows how evenly or unevenly samples are visited.
        """
        visit_counts = [len(visits) for visits in self.visit_history.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Histogram
        ax1.hist(visit_counts, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Number of Visits')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Visit Frequency Distribution')
        ax1.grid(True, alpha=0.3)

        # Cumulative
        sorted_counts = sorted(visit_counts, reverse=True)
        ax2.plot(range(len(sorted_counts)), sorted_counts, marker='o', alpha=0.7)
        ax2.set_xlabel('Sample Rank')
        ax2.set_ylabel('Number of Visits')
        ax2.set_title('Visit Frequency (Sorted)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_efe_components(
        self,
        sample_indices: Optional[List[int]] = None,
        max_samples: int = 5,
        figsize: Tuple[int, int] = (14, 8),
    ):
        """
        Plot EFE components (epistemic and pragmatic) over time.

        Args:
            sample_indices: Specific samples to plot (None = auto-select)
            max_samples: Maximum number of samples to plot
            figsize: Figure size
        """
        if sample_indices is None:
            # Auto-select: samples with EFE history
            candidates = list(self.efe_history.keys())
            sample_indices = candidates[:max_samples]

        if not sample_indices:
            print("No EFE history to plot")
            return None

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        for sample_idx in sample_indices:
            if sample_idx in self.efe_history:
                visits = self.visit_history[sample_idx]
                efe = self.efe_history[sample_idx]
                epistemic = self.epistemic_history[sample_idx]
                pragmatic = self.pragmatic_history[sample_idx]

                # EFE
                axes[0].plot(visits, efe, marker='o', alpha=0.7, label=f'Sample {sample_idx}')

                # Epistemic
                axes[1].plot(visits, epistemic, marker='s', alpha=0.7, label=f'Sample {sample_idx}')

                # Pragmatic
                axes[2].plot(visits, pragmatic, marker='^', alpha=0.7, label=f'Sample {sample_idx}')

        axes[0].set_ylabel('EFE')
        axes[0].set_title('Expected Free Energy')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel('Epistemic Value')
        axes[1].set_title('Epistemic Value (Uncertainty Reduction)')
        axes[1].grid(True, alpha=0.3)

        axes[2].set_ylabel('Pragmatic Value')
        axes[2].set_xlabel('Training Step')
        axes[2].set_title('Pragmatic Value (Goal Achievement)')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def compute_data_efficiency_metrics(self) -> Dict:
        """
        Compute metrics related to data efficiency.

        Returns:
            Dictionary with metrics:
                - samples_to_mastery: Average visits needed to master a sample
                - visit_entropy: Entropy of visit distribution (higher = more uniform)
                - exploration_exploitation_ratio: Ratio of unique samples to total visits
        """
        visit_counts = np.array([len(visits) for visits in self.visit_history.values()])

        # Average visits to mastery (approximation)
        avg_visits_to_mastery = float(np.mean(visit_counts)) if len(visit_counts) > 0 else 0.0

        # Visit entropy (measure of exploration uniformity)
        if len(visit_counts) > 0:
            visit_probs = visit_counts / visit_counts.sum()
            visit_entropy = float(-np.sum(visit_probs * np.log(visit_probs + 1e-10)))
        else:
            visit_entropy = 0.0

        # Exploration vs exploitation
        unique_samples = len(self.visit_history)
        total_visits = sum(visit_counts) if len(visit_counts) > 0 else 0
        exploration_ratio = unique_samples / self.num_samples if self.num_samples > 0 else 0.0

        return {
            'avg_visits_per_sample': avg_visits_to_mastery,
            'visit_entropy': visit_entropy,
            'exploration_ratio': exploration_ratio,
            'unique_samples_visited': unique_samples,
            'total_visits': int(total_visits),
        }

    def print_summary(self):
        """Print a summary of active inference diagnostics."""
        metrics = self.compute_data_efficiency_metrics()

        print("\n" + "=" * 60)
        print("ACTIVE INFERENCE DIAGNOSTICS SUMMARY")
        print("=" * 60)

        print(f"\nData Efficiency Metrics:")
        print(f"  Total training steps: {self.global_step}")
        print(f"  Unique samples visited: {metrics['unique_samples_visited']}/{self.num_samples} "
              f"({metrics['exploration_ratio']:.1%})")
        print(f"  Total visits: {metrics['total_visits']}")
        print(f"  Avg visits per sample: {metrics['avg_visits_per_sample']:.1f}")
        print(f"  Visit entropy: {metrics['visit_entropy']:.3f} "
              f"(max = {np.log(self.num_samples):.3f})")

        if self.category_history:
            latest = self.category_history[-1]
            _, mastered, learnable, noise, unvisited = latest
            total = mastered + learnable + noise + unvisited

            print(f"\nCurrent Category Distribution:")
            print(f"  Mastered:  {mastered:4d} ({mastered/total:.1%})")
            print(f"  Learnable: {learnable:4d} ({learnable/total:.1%}) ← Zone of Proximal Development")
            print(f"  Noise:     {noise:4d} ({noise/total:.1%})")
            print(f"  Unvisited: {unvisited:4d} ({unvisited/total:.1%})")

        print("=" * 60 + "\n")

    def save_plots(self, output_dir: str = './active_inference_plots'):
        """
        Save all diagnostic plots to disk.

        Args:
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Learning curves
        fig = self.plot_learning_curves()
        if fig:
            fig.savefig(f'{output_dir}/learning_curves.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Category evolution
        fig = self.plot_category_evolution()
        if fig:
            fig.savefig(f'{output_dir}/category_evolution.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Visit distribution
        fig = self.plot_visit_distribution()
        if fig:
            fig.savefig(f'{output_dir}/visit_distribution.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

        # EFE components
        fig = self.plot_efe_components()
        if fig:
            fig.savefig(f'{output_dir}/efe_components.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

        print(f"Plots saved to {output_dir}/")
