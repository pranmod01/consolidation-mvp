"""Temporal Spacing: Age-weighted and forgetting-weighted replay scheduling.

Inspired by spaced repetition in human memory consolidation. Items that:
1. Haven't been seen recently (older) get higher replay priority
2. Are being forgotten (high loss) get higher replay priority

This implements the "rehearsal at optimal intervals" principle from
cognitive psychology research on memory consolidation.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class SampleStats:
    """Statistics for a single buffer sample."""
    timestamp: float  # When sample was added
    last_replayed: float  # When sample was last replayed
    replay_count: int  # Total replay count
    loss_history: List[float]  # Recent loss values
    task_id: int


class TemporalSpacingSampler:
    """Sampler that implements spaced repetition scheduling.

    Combines multiple signals:
    1. Age: How long since sample was added
    2. Forgetting: How high is the loss on this sample
    3. Replay frequency: How often has this sample been replayed
    """

    def __init__(
        self,
        method: str = 'age_weighted',
        age_exponent: float = 1.0,
        forgetting_weight: float = 0.5,
        min_interval: float = 0.0,
        temperature: float = 1.0
    ):
        """
        Args:
            method: Weighting method
                - 'age_weighted': Prioritize older samples
                - 'forgetting_weighted': Prioritize high-loss samples
                - 'spaced_repetition': Combine age and replay frequency
                - 'combined': Use all signals
            age_exponent: Power for age weighting (higher = stronger age preference)
            forgetting_weight: Weight for loss-based component (0-1)
            min_interval: Minimum time between replays of same sample
            temperature: Stochasticity in sampling
        """
        self.method = method
        self.age_exponent = age_exponent
        self.forgetting_weight = forgetting_weight
        self.min_interval = min_interval
        self.temperature = temperature

    def compute_weights(
        self,
        timestamps: np.ndarray,
        last_replayed: np.ndarray,
        replay_counts: np.ndarray,
        losses: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute sampling weights for all buffer samples.

        Args:
            timestamps: When each sample was added
            last_replayed: When each sample was last replayed
            replay_counts: How many times each sample has been replayed
            losses: Optional recent loss for each sample

        Returns:
            Normalized sampling probabilities
        """
        current_time = time.time()
        n_samples = len(timestamps)

        if self.method == 'age_weighted':
            weights = self._age_weights(timestamps, current_time)
        elif self.method == 'forgetting_weighted':
            weights = self._forgetting_weights(losses, n_samples)
        elif self.method == 'spaced_repetition':
            weights = self._spaced_repetition_weights(
                timestamps, last_replayed, replay_counts, current_time
            )
        elif self.method == 'combined':
            weights = self._combined_weights(
                timestamps, last_replayed, replay_counts, losses, current_time
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Apply minimum interval constraint
        if self.min_interval > 0:
            time_since_replay = current_time - last_replayed
            too_recent = time_since_replay < self.min_interval
            weights[too_recent] = 0.0

        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_samples) / n_samples

        return weights

    def _age_weights(self, timestamps: np.ndarray, current_time: float) -> np.ndarray:
        """Weight by age: older samples get higher priority."""
        ages = current_time - timestamps
        # Add small constant to avoid zero weights
        weights = np.power(ages + 1.0, self.age_exponent)
        return weights

    def _forgetting_weights(
        self,
        losses: Optional[np.ndarray],
        n_samples: int
    ) -> np.ndarray:
        """Weight by forgetting: high-loss samples get higher priority."""
        if losses is None or len(losses) == 0:
            return np.ones(n_samples)

        # Higher loss = higher weight
        weights = losses + 0.1  # Add small constant to avoid zero
        return weights

    def _spaced_repetition_weights(
        self,
        timestamps: np.ndarray,
        last_replayed: np.ndarray,
        replay_counts: np.ndarray,
        current_time: float
    ) -> np.ndarray:
        """Implement Leitner-style spaced repetition.

        Samples that haven't been replayed recently and have been replayed
        fewer times get higher priority.
        """
        # Time since last replay
        time_since_replay = current_time - last_replayed

        # Interval should grow with successful replays
        # (simplified: we don't track success, just count replays)
        expected_interval = np.power(2.0, replay_counts)  # Exponential spacing

        # Overdue factor: how much past the expected interval
        overdue = time_since_replay / (expected_interval + 1.0)

        weights = np.power(overdue + 1.0, self.age_exponent)
        return weights

    def _combined_weights(
        self,
        timestamps: np.ndarray,
        last_replayed: np.ndarray,
        replay_counts: np.ndarray,
        losses: Optional[np.ndarray],
        current_time: float
    ) -> np.ndarray:
        """Combine age, forgetting, and spaced repetition signals."""
        n_samples = len(timestamps)

        # Age component
        age_weights = self._age_weights(timestamps, current_time)
        age_weights = age_weights / (age_weights.max() + 1e-8)

        # Forgetting component
        forgetting_weights = self._forgetting_weights(losses, n_samples)
        forgetting_weights = forgetting_weights / (forgetting_weights.max() + 1e-8)

        # Spaced repetition component
        sr_weights = self._spaced_repetition_weights(
            timestamps, last_replayed, replay_counts, current_time
        )
        sr_weights = sr_weights / (sr_weights.max() + 1e-8)

        # Combine (configurable weights)
        combined = (
            (1 - self.forgetting_weight) * 0.5 * age_weights +
            (1 - self.forgetting_weight) * 0.5 * sr_weights +
            self.forgetting_weight * forgetting_weights
        )

        return combined

    def sample(
        self,
        timestamps: np.ndarray,
        last_replayed: np.ndarray,
        replay_counts: np.ndarray,
        losses: Optional[np.ndarray],
        batch_size: int
    ) -> List[int]:
        """Sample indices according to temporal spacing weights.

        Returns:
            List of selected buffer indices
        """
        weights = self.compute_weights(
            timestamps, last_replayed, replay_counts, losses
        )

        n_samples = len(timestamps)
        batch_size = min(batch_size, n_samples)

        # Apply temperature for stochasticity
        if self.temperature > 0 and self.temperature != 1.0:
            weights = np.power(weights, 1.0 / self.temperature)
            weights = weights / weights.sum()

        # Sample without replacement
        indices = np.random.choice(
            n_samples,
            size=batch_size,
            replace=False,
            p=weights
        )

        return indices.tolist()


def compute_spaced_weights(
    ages: np.ndarray,
    replay_counts: np.ndarray,
    age_exponent: float = 1.0
) -> np.ndarray:
    """Convenience function for computing spaced repetition weights.

    Args:
        ages: Age of each sample in seconds
        replay_counts: Number of times each sample has been replayed
        age_exponent: Power for age weighting

    Returns:
        Normalized sampling weights
    """
    # Expected interval grows exponentially with replay count
    expected_intervals = np.power(2.0, replay_counts)

    # Overdue factor
    overdue = ages / (expected_intervals + 1.0)

    weights = np.power(overdue + 1.0, age_exponent)
    weights = weights / weights.sum()

    return weights


def forgetting_weighted_sampling(
    losses: np.ndarray,
    batch_size: int,
    temperature: float = 1.0
) -> Tuple[List[int], np.ndarray]:
    """Convenience function for forgetting-weighted sampling.

    Higher loss samples are more likely to be selected.

    Returns:
        Tuple of (selected indices, sampling weights)
    """
    n_samples = len(losses)
    batch_size = min(batch_size, n_samples)

    # Add small constant and normalize
    weights = losses + 0.1
    weights = weights / weights.sum()

    # Apply temperature
    if temperature != 1.0:
        weights = np.power(weights, 1.0 / temperature)
        weights = weights / weights.sum()

    indices = np.random.choice(
        n_samples,
        size=batch_size,
        replace=False,
        p=weights
    )

    return indices.tolist(), weights
