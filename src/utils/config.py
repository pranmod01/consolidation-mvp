"""Configuration management for experiments."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import yaml
import json
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for consolidation experiments."""

    # Experiment identification
    name: str = "consolidation_experiment"
    seed: int = 42

    # Dataset
    dataset: str = "split_mnist"  # split_mnist, split_cifar10, split_tinyimagenet
    num_tasks: int = 5
    data_root: str = "./data"

    # Model
    model_type: str = "cnn"  # cnn, resnet
    feature_dim: int = 128
    num_classes: int = 10

    # Training
    epochs_per_task: int = 5
    batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 0.0
    device: str = "auto"  # auto, cuda, cpu

    # Base method
    method: str = "vanilla_replay"  # vanilla_replay, ewc, meta_sgd

    # EWC-specific
    ewc_lambda: float = 400.0
    ewc_online: bool = False
    ewc_gamma: float = 0.95

    # Meta-SGD-specific
    meta_inner_lr: float = 0.01
    meta_inner_steps: int = 5
    meta_lr: float = 0.001

    # Replay buffer
    buffer_size: int = 500
    samples_per_task: Optional[int] = None
    replay_batch_size: int = 32
    replay_freq: int = 1

    # Consolidation mechanisms
    sampling_strategy: str = "random"  # random, diversity, temporal, combined
    update_features_freq: int = 0  # For diversity sampling: update features every N batches

    # Pattern Separation config
    ps_method: str = "farthest_point"  # farthest_point, cluster, hybrid
    ps_n_clusters: int = 10
    ps_temperature: float = 1.0

    # Temporal Spacing config
    ts_method: str = "age_weighted"  # age_weighted, forgetting_weighted, spaced_repetition, combined
    ts_age_exponent: float = 1.0
    ts_forgetting_weight: float = 0.5

    # Combined PS+TS config
    diversity_weight: float = 0.5
    temporal_weight: float = 0.5

    # Output
    results_dir: str = "./results"
    save_checkpoints: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentConfig':
        return cls(**data)

    def get_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


def load_config(path: str) -> ExperimentConfig:
    """Load config from YAML or JSON file."""
    path = Path(path)

    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    return ExperimentConfig.from_dict(data)


def save_config(config: ExperimentConfig, path: str):
    """Save config to YAML or JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        else:
            json.dump(config.to_dict(), f, indent=2)


# Preset configurations for experiments

BASELINE_RANDOM = ExperimentConfig(
    name="baseline_random",
    sampling_strategy="random",
)

PATTERN_SEPARATION_ONLY = ExperimentConfig(
    name="pattern_separation",
    sampling_strategy="diversity",
    ps_method="farthest_point",
    ps_temperature=1.0,
    update_features_freq=50,  # Update features periodically
)

TEMPORAL_SPACING_ONLY = ExperimentConfig(
    name="temporal_spacing",
    sampling_strategy="temporal",
    ts_method="age_weighted",
    ts_age_exponent=1.0,
)

COMBINED_PS_TS = ExperimentConfig(
    name="combined_ps_ts",
    sampling_strategy="combined",
    diversity_weight=0.5,
    temporal_weight=0.5,
    update_features_freq=50,
)


def get_phase1_configs(base_method: str, buffer_sizes: List[int]) -> List[ExperimentConfig]:
    """Generate configs for Phase 1 experiments.

    Phase 1: Test single mechanisms (PS or TS) separately.
    """
    configs = []

    for buffer_size in buffer_sizes:
        # Baseline: random sampling
        configs.append(ExperimentConfig(
            name=f"{base_method}_random_buf{buffer_size}",
            method=base_method,
            buffer_size=buffer_size,
            sampling_strategy="random",
        ))

        # +PS: Pattern Separation only
        configs.append(ExperimentConfig(
            name=f"{base_method}_ps_buf{buffer_size}",
            method=base_method,
            buffer_size=buffer_size,
            sampling_strategy="diversity",
            update_features_freq=50,
        ))

        # +TS: Temporal Spacing only
        configs.append(ExperimentConfig(
            name=f"{base_method}_ts_buf{buffer_size}",
            method=base_method,
            buffer_size=buffer_size,
            sampling_strategy="temporal",
        ))

    return configs


def get_phase2_configs(base_methods: List[str], buffer_size: int = 500) -> List[ExperimentConfig]:
    """Generate configs for Phase 2 experiments.

    Phase 2: Test combined mechanisms on all base methods.
    """
    configs = []
    strategies = ['random', 'diversity', 'temporal', 'combined']

    for method in base_methods:
        for strategy in strategies:
            configs.append(ExperimentConfig(
                name=f"{method}_{strategy}",
                method=method,
                buffer_size=buffer_size,
                sampling_strategy=strategy,
                update_features_freq=50 if strategy in ['diversity', 'combined'] else 0,
            ))

    return configs


def get_phase3_configs(best_method: str, best_strategy: str) -> List[ExperimentConfig]:
    """Generate configs for Phase 3 validation.

    Phase 3: Scale best combo to harder datasets.
    """
    configs = []

    # Split-CIFAR10
    configs.append(ExperimentConfig(
        name=f"{best_method}_{best_strategy}_cifar10",
        method=best_method,
        dataset="split_cifar10",
        sampling_strategy=best_strategy,
        model_type="resnet",
        feature_dim=512,
        epochs_per_task=10,
        buffer_size=1000,
        update_features_freq=50,
    ))

    # Longer task sequence on MNIST (10 tasks)
    configs.append(ExperimentConfig(
        name=f"{best_method}_{best_strategy}_mnist_10tasks",
        method=best_method,
        dataset="split_mnist",
        num_tasks=5,  # Split MNIST only has 5 tasks naturally
        sampling_strategy=best_strategy,
        buffer_size=1000,
        update_features_freq=50,
    ))

    return configs
