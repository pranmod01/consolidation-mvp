"""Pattern Separation: Maximize diversity in replay buffer sampling.

Inspired by hippocampal pattern separation, which creates distinct neural
representations for similar inputs to reduce interference.

Implementation approaches:
1. Cluster-based: k-means clustering in feature space, sample from each cluster
2. Distance-based: Greedy farthest-point sampling to maximize coverage
3. Class-balanced + diversity: Ensure diversity within each class

Key insight: Random sampling may over-represent dense regions of feature space.
Diversity-based sampling ensures broader coverage of the learned representations.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.cluster import MiniBatchKMeans


class PatternSeparationSampler:
    """Sampler that maximizes feature-space diversity.

    Used with ReplayBuffer to implement pattern separation.
    """

    def __init__(
        self,
        method: str = 'farthest_point',
        n_clusters: int = 10,
        temperature: float = 1.0,
        class_balanced: bool = True
    ):
        """
        Args:
            method: Sampling method ('farthest_point', 'cluster', 'hybrid')
            n_clusters: Number of clusters for cluster-based sampling
            temperature: Stochasticity in sampling (higher = more random)
            class_balanced: Whether to ensure class balance in samples
        """
        self.method = method
        self.n_clusters = n_clusters
        self.temperature = temperature
        self.class_balanced = class_balanced

        # For cluster-based sampling
        self.kmeans = None
        self.cluster_assignments = None

    def fit(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Fit the sampler to current buffer features.

        Args:
            features: (N, D) tensor of feature embeddings
            labels: Optional (N,) tensor of class labels
        """
        if self.method in ['cluster', 'hybrid']:
            self._fit_clusters(features)

    def _fit_clusters(self, features: torch.Tensor):
        """Fit k-means clustering on features."""
        features_np = features.cpu().numpy()
        n_samples = features_np.shape[0]

        # Adjust n_clusters if we have fewer samples
        actual_clusters = min(self.n_clusters, n_samples)

        self.kmeans = MiniBatchKMeans(
            n_clusters=actual_clusters,
            random_state=42,
            batch_size=min(100, n_samples)
        )
        self.cluster_assignments = self.kmeans.fit_predict(features_np)

    def sample(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int
    ) -> List[int]:
        """Sample diverse indices from buffer.

        Args:
            features: (N, D) tensor of all buffer features
            labels: (N,) tensor of all buffer labels
            batch_size: Number of samples to select

        Returns:
            List of selected indices
        """
        if self.method == 'farthest_point':
            return self._farthest_point_sample(features, batch_size)
        elif self.method == 'cluster':
            return self._cluster_sample(features, labels, batch_size)
        elif self.method == 'hybrid':
            return self._hybrid_sample(features, labels, batch_size)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _farthest_point_sample(
        self,
        features: torch.Tensor,
        batch_size: int
    ) -> List[int]:
        """Greedy farthest-point sampling for maximum diversity."""
        n_samples = features.size(0)
        batch_size = min(batch_size, n_samples)

        selected = []
        # Start with random point
        selected.append(np.random.randint(n_samples))

        for _ in range(batch_size - 1):
            # Compute distances to selected set
            selected_features = features[selected]  # (k, D)
            distances = torch.cdist(features, selected_features)  # (N, k)
            min_distances = distances.min(dim=1)[0]  # (N,)

            # Mask already selected
            min_distances[selected] = -float('inf')

            # Sample with temperature
            if self.temperature > 0:
                probs = torch.softmax(min_distances / self.temperature, dim=0)
                next_idx = torch.multinomial(probs, 1).item()
            else:
                next_idx = min_distances.argmax().item()

            selected.append(next_idx)

        return selected

    def _cluster_sample(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int
    ) -> List[int]:
        """Sample uniformly from each cluster."""
        if self.cluster_assignments is None:
            self._fit_clusters(features)

        n_clusters = len(np.unique(self.cluster_assignments))
        samples_per_cluster = max(1, batch_size // n_clusters)

        selected = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(self.cluster_assignments == cluster_id)[0]
            n_select = min(samples_per_cluster, len(cluster_indices))
            selected.extend(
                np.random.choice(cluster_indices, n_select, replace=False).tolist()
            )

        # Fill remaining if needed
        while len(selected) < batch_size and len(selected) < features.size(0):
            idx = np.random.randint(features.size(0))
            if idx not in selected:
                selected.append(idx)

        return selected[:batch_size]

    def _hybrid_sample(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int
    ) -> List[int]:
        """Combine cluster-based and farthest-point sampling."""
        # Half from clusters, half from farthest-point
        cluster_size = batch_size // 2
        fp_size = batch_size - cluster_size

        cluster_selected = self._cluster_sample(features, labels, cluster_size)
        fp_selected = self._farthest_point_sample(features, fp_size)

        # Combine and deduplicate
        combined = list(set(cluster_selected + fp_selected))

        # Fill if needed
        while len(combined) < batch_size:
            idx = np.random.randint(features.size(0))
            if idx not in combined:
                combined.append(idx)

        return combined[:batch_size]


def compute_diversity_score(features: torch.Tensor, indices: List[int]) -> float:
    """Compute diversity score for a set of selected samples.

    Higher score = more diverse selection.

    Uses average pairwise distance in feature space.
    """
    if len(indices) < 2:
        return 0.0

    selected_features = features[indices]
    distances = torch.cdist(selected_features, selected_features)

    # Average pairwise distance (excluding diagonal)
    n = len(indices)
    total_dist = (distances.sum() - distances.trace()) / (n * (n - 1))

    return total_dist.item()


def cluster_based_sampling(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    n_clusters: int = 10
) -> Tuple[List[int], float]:
    """Convenience function for cluster-based sampling.

    Returns:
        Tuple of (selected indices, diversity score)
    """
    sampler = PatternSeparationSampler(method='cluster', n_clusters=n_clusters)
    sampler.fit(features, labels)
    indices = sampler.sample(features, labels, batch_size)
    diversity = compute_diversity_score(features, indices)
    return indices, diversity


def distance_based_sampling(
    features: torch.Tensor,
    batch_size: int,
    temperature: float = 1.0
) -> Tuple[List[int], float]:
    """Convenience function for distance-based sampling.

    Returns:
        Tuple of (selected indices, diversity score)
    """
    sampler = PatternSeparationSampler(method='farthest_point', temperature=temperature)
    indices = sampler.sample(features, torch.zeros(features.size(0)), batch_size)
    diversity = compute_diversity_score(features, indices)
    return indices, diversity
