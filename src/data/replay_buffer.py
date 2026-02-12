"""Replay buffer with support for different sampling strategies.

This is the core data structure that consolidation mechanisms operate on.
The base buffer stores samples; consolidation strategies modify how we sample from it.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class BufferSample:
    """Single sample in the replay buffer."""
    image: torch.Tensor
    label: int
    task_id: int
    timestamp: float  # When sample was added (for temporal spacing)
    features: Optional[torch.Tensor] = None  # For pattern separation
    loss_history: List[float] = field(default_factory=list)  # For forgetting-weighted sampling
    replay_count: int = 0  # How many times this sample has been replayed


class ReplayBuffer:
    """Experience replay buffer with flexible sampling strategies.

    Supports:
    - Random sampling (baseline)
    - Class-balanced sampling
    - Feature-based diversity sampling (pattern separation)
    - Age-weighted sampling (temporal spacing)
    - Forgetting-weighted sampling (temporal spacing variant)

    The sampling strategy is configured via the `sample()` method parameters,
    allowing the same buffer to be used with different consolidation mechanisms.
    """

    def __init__(
        self,
        max_size: int = 500,
        samples_per_task: Optional[int] = None,
        feature_dim: int = 128
    ):
        """
        Args:
            max_size: Maximum total samples in buffer
            samples_per_task: Max samples to keep per task (None = no per-task limit)
            feature_dim: Dimension of feature vectors for diversity sampling
        """
        self.max_size = max_size
        self.samples_per_task = samples_per_task
        self.feature_dim = feature_dim

        # Storage
        self.samples: List[BufferSample] = []

        # Indices for efficient lookup
        self.task_indices: Dict[int, List[int]] = defaultdict(list)
        self.class_indices: Dict[int, List[int]] = defaultdict(list)

        # For diversity-based sampling
        self.feature_matrix: Optional[torch.Tensor] = None  # (N, feature_dim)

        # Statistics
        self.total_samples_seen = 0
        self.current_time = 0  # Logical time for temporal spacing

    def __len__(self) -> int:
        return len(self.samples)

    def add_samples(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        task_id: int,
        features: Optional[torch.Tensor] = None
    ):
        """Add batch of samples to buffer.

        Args:
            images: (B, C, H, W) tensor of images
            labels: (B,) tensor of labels
            task_id: Current task ID
            features: Optional (B, feature_dim) tensor of feature embeddings
        """
        batch_size = images.size(0)

        for i in range(batch_size):
            sample = BufferSample(
                image=images[i].cpu().clone(),
                label=labels[i].item(),
                task_id=task_id,
                timestamp=self.current_time,  # Use logical time
                features=features[i].cpu().clone() if features is not None else None
            )
            self._add_single_sample(sample)
            self.total_samples_seen += 1

        self.current_time += 1
        self._rebuild_indices()
        self._update_feature_matrix()

    def _add_single_sample(self, sample: BufferSample):
        """Add single sample, handling capacity constraints."""
        if len(self.samples) >= self.max_size:
            self._evict_sample()
        self.samples.append(sample)

    def _evict_sample(self):
        """Evict a sample when buffer is full. Uses FIFO by default."""
        if self.samples:
            self.samples.pop(0)

    def _rebuild_indices(self):
        """Rebuild task and class indices after modifications."""
        self.task_indices.clear()
        self.class_indices.clear()

        for idx, sample in enumerate(self.samples):
            self.task_indices[sample.task_id].append(idx)
            self.class_indices[sample.label].append(idx)

    def _update_feature_matrix(self):
        """Update cached feature matrix for diversity sampling."""
        features = [s.features for s in self.samples if s.features is not None]
        if features:
            self.feature_matrix = torch.stack(features)
        else:
            self.feature_matrix = None

    def sample(
        self,
        batch_size: int,
        strategy: str = 'random',
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample batch from buffer using specified strategy.

        Args:
            batch_size: Number of samples to return
            strategy: Sampling strategy
                - 'random': Uniform random sampling
                - 'balanced': Class-balanced sampling
                - 'diversity': Feature-space diversity sampling (pattern separation)
                - 'temporal': Age-weighted sampling (temporal spacing)
                - 'forgetting': Forgetting-weighted sampling
                - 'combined': Combines diversity + temporal

        Returns:
            images: (B, C, H, W) tensor
            labels: (B,) tensor
            indices: (B,) tensor of buffer indices (for updating stats)
        """
        if len(self.samples) == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, len(self.samples))

        if strategy == 'random':
            indices = self._sample_random(batch_size)
        elif strategy == 'balanced':
            indices = self._sample_balanced(batch_size)
        elif strategy == 'diversity':
            indices = self._sample_diversity(batch_size, **kwargs)
        elif strategy == 'temporal':
            indices = self._sample_temporal(batch_size, **kwargs)
        elif strategy == 'forgetting':
            indices = self._sample_forgetting(batch_size)
        elif strategy == 'combined':
            indices = self._sample_combined(batch_size, **kwargs)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        # Update replay counts
        for idx in indices:
            self.samples[idx].replay_count += 1

        return self._gather_samples(indices)

    def _sample_random(self, batch_size: int) -> List[int]:
        """Uniform random sampling."""
        return np.random.choice(len(self.samples), batch_size, replace=False).tolist()

    def _sample_balanced(self, batch_size: int) -> List[int]:
        """Class-balanced sampling."""
        indices = []
        classes = list(self.class_indices.keys())
        samples_per_class = max(1, batch_size // len(classes))

        for cls in classes:
            cls_indices = self.class_indices[cls]
            n_samples = min(samples_per_class, len(cls_indices))
            selected = np.random.choice(cls_indices, n_samples, replace=False)
            indices.extend(selected.tolist())

        # Fill remaining if needed
        while len(indices) < batch_size:
            idx = np.random.choice(len(self.samples))
            if idx not in indices:
                indices.append(idx)

        return indices[:batch_size]

    def _sample_diversity(self, batch_size: int, temperature: float = 1.0, **kwargs) -> List[int]:
        """Diversity-based sampling using feature space distances (Pattern Separation).

        Uses greedy farthest-point sampling to maximize coverage of feature space.
        """
        if self.feature_matrix is None:
            return self._sample_random(batch_size)

        n_samples = len(self.samples)
        selected = []

        # Start with random sample
        selected.append(np.random.randint(n_samples))

        # Greedy farthest-point sampling
        for _ in range(batch_size - 1):
            # Compute minimum distance from each point to selected set
            selected_features = self.feature_matrix[selected]  # (k, d)
            all_features = self.feature_matrix  # (n, d)

            # Distance from each point to nearest selected point
            distances = torch.cdist(all_features, selected_features)  # (n, k)
            min_distances = distances.min(dim=1)[0]  # (n,)

            # Set already selected to -inf
            min_distances[selected] = -float('inf')

            # Apply temperature for stochasticity
            if temperature > 0:
                probs = torch.softmax(min_distances / temperature, dim=0)
                next_idx = torch.multinomial(probs, 1).item()
            else:
                next_idx = min_distances.argmax().item()

            selected.append(next_idx)

        return selected

    def _sample_temporal(
        self,
        batch_size: int,
        age_weight: float = 2.0,
        prioritize_old: bool = True,
        **kwargs
    ) -> List[int]:
        """Age-weighted sampling (Temporal Spacing).

        Older samples have higher probability of being selected,
        implementing spaced repetition principle.
        """
        # Use logical time for consistent temporal spacing
        ages = np.array([self.current_time - s.timestamp for s in self.samples])

        if prioritize_old:
            # Older samples get higher weight
            weights = np.power(ages + 1, age_weight)
        else:
            # Newer samples get higher weight (for comparison)
            weights = np.power(1.0 / (ages + 1), age_weight)

        probs = weights / weights.sum()
        indices = np.random.choice(
            len(self.samples),
            size=batch_size,
            replace=False,
            p=probs
        )
        return indices.tolist()

    def _sample_forgetting(self, batch_size: int) -> List[int]:
        """Forgetting-weighted sampling.

        Samples that had high loss recently are prioritized.
        """
        weights = []
        for sample in self.samples:
            if sample.loss_history:
                # Use recent loss as weight
                recent_loss = np.mean(sample.loss_history[-5:])
                weights.append(recent_loss + 0.1)  # Add small constant to avoid zero
            else:
                weights.append(1.0)  # Default weight for new samples

        weights = np.array(weights)
        probs = weights / weights.sum()
        indices = np.random.choice(
            len(self.samples),
            size=batch_size,
            replace=False,
            p=probs
        )
        return indices.tolist()

    def _sample_combined(
        self,
        batch_size: int,
        oversample_factor: float = 4.0,
        **kwargs
    ) -> List[int]:
        """Two-stage sequential refinement: Temporal Spacing → Pattern Separation.

        Stage 1: Oversample OLD samples using temporal weighting
        Stage 2: Select DIVERSE subset from candidates using farthest-point sampling

        This guarantees both properties: samples are old AND diverse.
        """
        candidate_size = min(int(batch_size * oversample_factor), len(self.samples))
        candidate_indices = self._sample_temporal(candidate_size, **kwargs)

        if len(candidate_indices) <= batch_size or self.feature_matrix is None:
            return candidate_indices[:batch_size]

        # Diversity sampling on candidates only
        candidate_features = self.feature_matrix[candidate_indices]
        selected_relative = self._greedy_fps(candidate_features, batch_size)

        return [candidate_indices[i] for i in selected_relative]

    def _greedy_fps(self, features: torch.Tensor, n_select: int) -> List[int]:
        """Greedy farthest-point sampling on given features.

        Args:
            features: (N, D) tensor of feature vectors
            n_select: Number of points to select

        Returns:
            List of indices into features tensor
        """
        n_samples = len(features)
        n_select = min(n_select, n_samples)

        selected = [np.random.randint(n_samples)]

        for _ in range(n_select - 1):
            selected_features = features[selected]
            dists = torch.cdist(features, selected_features)
            min_dists = dists.min(dim=1)[0]
            min_dists[selected] = -float('inf')
            selected.append(min_dists.argmax().item())

        return selected

    def _gather_samples(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather samples by indices into batched tensors."""
        images = torch.stack([self.samples[i].image for i in indices])
        labels = torch.tensor([self.samples[i].label for i in indices], dtype=torch.long)
        return images, labels, torch.tensor(indices, dtype=torch.long)

    def update_losses(self, indices: torch.Tensor, losses: torch.Tensor):
        """Update loss history for samples (for forgetting-weighted sampling).

        Args:
            indices: Buffer indices of samples
            losses: Per-sample losses
        """
        for idx, loss in zip(indices.tolist(), losses.tolist()):
            if 0 <= idx < len(self.samples):
                self.samples[idx].loss_history.append(loss)
                # Keep only recent history
                if len(self.samples[idx].loss_history) > 10:
                    self.samples[idx].loss_history.pop(0)

    def update_features(self, model: torch.nn.Module, device: str = 'cpu'):
        """Update feature embeddings for all samples using current model.

        Called periodically to keep features up-to-date for diversity sampling.
        """
        if not self.samples:
            return

        model.eval()
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(self.samples), batch_size):
                batch_samples = self.samples[i:i+batch_size]
                images = torch.stack([s.image for s in batch_samples]).to(device)
                features = model.get_features(images)

                for j, sample in enumerate(batch_samples):
                    sample.features = features[j].cpu()

        self._update_feature_matrix()
        model.train()

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        task_counts = {t: len(indices) for t, indices in self.task_indices.items()}
        class_counts = {c: len(indices) for c, indices in self.class_indices.items()}

        ages = [time.time() - s.timestamp for s in self.samples]
        replay_counts = [s.replay_count for s in self.samples]

        return {
            'total_samples': len(self.samples),
            'max_size': self.max_size,
            'task_counts': task_counts,
            'class_counts': class_counts,
            'has_features': self.feature_matrix is not None,
            'avg_age': np.mean(ages) if ages else 0,
            'avg_replay_count': np.mean(replay_counts) if replay_counts else 0,
            'total_samples_seen': self.total_samples_seen,
        }
