"""Vanilla Experience Replay - simplest baseline.

Standard experience replay with random sampling from buffer.
This is the "does consolidation help at all?" baseline.
"""

from .base_trainer import BaseTrainer, TrainingConfig


class VanillaReplayTrainer(BaseTrainer):
    """Vanilla experience replay trainer.

    This is the simplest continual learning method:
    1. Train on current task
    2. Store samples in replay buffer
    3. Mix replay samples with current task during training

    The only variable is the sampling strategy (random vs diversity vs temporal).
    """

    def __init__(self, model, config: TrainingConfig, num_classes: int = 10):
        # Vanilla replay uses default base trainer behavior
        super().__init__(model, config, num_classes)

    # All functionality inherited from BaseTrainer
    # Vanilla replay has no additional regularization or meta-learning
