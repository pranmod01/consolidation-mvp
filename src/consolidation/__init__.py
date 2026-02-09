"""Bio-inspired consolidation mechanisms for continual learning.

Two main mechanisms:
1. Pattern Separation: Maximize feature-space diversity in replay buffer
2. Temporal Spacing: Age-weighted or forgetting-weighted replay scheduling
"""

from .pattern_separation import (
    PatternSeparationSampler,
    compute_diversity_score,
    cluster_based_sampling,
    distance_based_sampling,
)
from .temporal_spacing import (
    TemporalSpacingSampler,
    compute_spaced_weights,
    forgetting_weighted_sampling,
)

__all__ = [
    'PatternSeparationSampler',
    'compute_diversity_score',
    'cluster_based_sampling',
    'distance_based_sampling',
    'TemporalSpacingSampler',
    'compute_spaced_weights',
    'forgetting_weighted_sampling',
]
