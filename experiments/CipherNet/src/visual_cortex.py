"""Retinotopic visual cortex — V1 columns wired to the eye.

Each V1 column reads from a FIXED retinal position. It doesn't know
where the eye is pointing — it only knows what's at ITS position on
the retina. When the eye saccades, the image shifts on the retina,
and the column sees new features.

The displacement (efference copy) from the saccade tells columns HOW
the image shifted. Columns accumulate (feature, displacement, next_feature)
triples to learn their reference frames.

This is TBT: each column builds a model of the world through
sensorimotor experience (observe → move → observe → learn).
"""
from __future__ import annotations

import numpy as np

from symbolic_column import SymbolicColumn, CorticalMessage, MAX_MEMORY
from eye import Eye, SalienceMap
from codebook import PatchCodebook


class RetinotopicV1:
    """V1 cortex: one symbolic column per retinal sample position.

    Columns are wired to the EYE, not the image. Each column
    receives from one retinal position. The retinotopic map IS
    the wiring — position is structural, not learned.

    Foveal columns (ring 0) see high-resolution details.
    Peripheral columns (ring 1+) see coarse features.
    """

    def __init__(self, eye: Eye, codebook: PatchCodebook | None = None):
        self.eye = eye
        self.codebook = codebook

        # One column per retinal sample.
        self.columns: list[SymbolicColumn] = []
        for i, (dx, dy, ring) in enumerate(eye.get_sample_positions()):
            col = SymbolicColumn(
                name=f"V1:{i}",
                receptive_field=(dx, dy, ring),
                position=(dx, dy),  # retinotopic position
                max_memory=MAX_MEMORY,
            )
            self.columns.append(col)

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def observe(self, image: np.ndarray) -> list[str]:
        """Sample the image through the eye, encode, observe at each column.

        Returns the feature code at each retinal position.
        """
        retinal_samples = self.eye.sample(image)
        features = []

        for i, col in enumerate(self.columns):
            # Quantize retinal sample to a discrete feature.
            # For single-pixel samples: simple intensity quantization.
            val = retinal_samples[i]
            feature = f"i{int(val * 15)}"  # 16 intensity levels
            col.observe(feature)
            features.append(feature)

        return features

    def get_messages(self) -> list[CorticalMessage]:
        """Collect CMP messages from all V1 columns."""
        return [col.message() for col in self.columns]

    def teach_all(self, target: str):
        """Teach all columns: current_feature → target."""
        for col in self.columns:
            if col.current_input is not None:
                col.teach(col.current_input, target)

    def displacement_teach(self, prev_features: list[str],
                           displacement: tuple[float, float],
                           curr_features: list[str]):
        """Teach displacement associations.

        For each column: (prev_feature, displacement) → curr_feature.
        This is how the column learns its reference frame:
        "when I saw X and the eye moved by D, I now see Y."

        The displacement is encoded as part of the feature key,
        making it a (feature, morphism) → feature mapping.
        """
        dx, dy = displacement
        # Quantize displacement to grid.
        qdx, qdy = int(round(dx)), int(round(dy))
        disp_key = f"d{qdx},{qdy}"

        for i, col in enumerate(self.columns):
            if prev_features[i] is not None and curr_features[i] is not None:
                # Key = "prev_feature:displacement"
                key = f"{prev_features[i]}:{disp_key}"
                col.teach(key, curr_features[i])

    def predict_after_displacement(self, features: list[str],
                                   displacement: tuple[float, float]) -> list[str | None]:
        """Predict what each column will see after a displacement.

        Uses the learned (feature, displacement) → next_feature mapping.
        """
        dx, dy = displacement
        qdx, qdy = int(round(dx)), int(round(dy))
        disp_key = f"d{qdx},{qdy}"

        predictions = []
        for i, col in enumerate(self.columns):
            if features[i] is not None:
                key = f"{features[i]}:{disp_key}"
                col.observe(key)
                predictions.append(col.predict())
            else:
                predictions.append(None)
        return predictions


class FovealExplorer:
    """Explores an image through saccadic eye movements.

    Uses salience to guide fixations, collects sensorimotor
    experience at each fixation for column learning.

    The exploration protocol:
    1. Compute salience map → suggest fixation points
    2. For each fixation: observe → saccade → observe
    3. Columns accumulate (feature, displacement, next_feature) triples
    4. After exploration: columns vote on object identity
    """

    def __init__(self, eye: Eye, v1: RetinotopicV1,
                 n_fixations: int = 5):
        self.eye = eye
        self.v1 = v1
        self.n_fixations = n_fixations

    def explore(self, image: np.ndarray, label: str | None = None,
                learn: bool = True) -> tuple[str | None, dict[str, float]]:
        """Explore an image through saccades. Optionally learn.

        Returns (predicted_label, vote_scores).
        """
        h, w = image.shape[:2]

        # Get salience-guided fixation sequence.
        fixations = SalienceMap.suggest_fixations(
            image, n=self.n_fixations,
            min_distance=max(2, self.eye.fovea_radius),
        )
        if not fixations:
            # Fallback: center
            fixations = [(w // 2, h // 2)]

        # First fixation.
        fx, fy = fixations[0]
        self.eye.fixate(float(fx), float(fy))
        prev_features = self.v1.observe(image)

        # Teach object identity at first fixation.
        if learn and label is not None:
            self.v1.teach_all(label)

        # Subsequent fixations: saccade → observe → learn displacement + identity.
        for fx, fy in fixations[1:]:
            self.eye.saccade_to(float(fx), float(fy))
            displacement = self.eye.last_displacement
            curr_features = self.v1.observe(image)

            if learn:
                # Teach displacement associations (reference frame learning).
                self.v1.displacement_teach(prev_features, displacement, curr_features)
                # Teach object identity at this fixation too.
                if label is not None:
                    self.v1.teach_all(label)

            prev_features = curr_features

        # Vote on identity across all columns and fixations.
        votes: dict[str, float] = {}
        for col in self.v1.columns:
            pred = col.prediction
            if pred is not None:
                votes[pred] = votes.get(pred, 0.0) + col.confidence

        if not votes:
            return None, votes
        winner = max(votes, key=votes.get)
        return winner, votes
