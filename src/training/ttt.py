"""Test-Time Training (TTT) for continual learning.

TTT adapts the model at test time using a self-supervised auxiliary task
(rotation prediction). During training, we jointly optimize the main task
and rotation prediction. At test time, we adapt the feature extractor using
only the rotation task before making predictions.

Reference: Sun et al., "Test-Time Training with Self-Supervision for
Generalization under Distribution Shift" (ICML 2020)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm

from .base_trainer import BaseTrainer, TrainingConfig


class TTTTrainer(BaseTrainer):
    """Test-Time Training trainer.

    Adds rotation prediction as auxiliary self-supervised task.
    At test time, adapts the backbone using rotation loss before prediction.

    Args:
        model: Classifier model with backbone
        config: Training configuration
        num_classes: Number of classes for main task
        ttt_lr: Learning rate for test-time adaptation
        ttt_steps: Number of adaptation steps at test time
        rotation_weight: Weight for rotation loss during training
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        num_classes: int = 10,
        ttt_lr: float = 0.001,
        ttt_steps: int = 1,
        rotation_weight: float = 1.0,
    ):
        super().__init__(model, config, num_classes)

        self.ttt_lr = ttt_lr
        self.ttt_steps = ttt_steps
        self.rotation_weight = rotation_weight

        # Rotation prediction head (4 classes: 0, 90, 180, 270 degrees)
        self.rotation_head = nn.Linear(model.feature_dim, 4).to(self.device)

    def _rotate_batch(self, images: torch.Tensor) -> tuple:
        """Create rotated versions of images with rotation labels.

        Returns:
            rotated_images: Tensor of shape (batch_size * 4, C, H, W)
            rotation_labels: Tensor of shape (batch_size * 4,) with values 0-3
        """
        batch_size = images.size(0)
        rotated_images = []
        rotation_labels = []

        for rot_idx in range(4):  # 0, 90, 180, 270 degrees
            if rot_idx == 0:
                rotated = images
            elif rot_idx == 1:
                rotated = torch.rot90(images, k=1, dims=[2, 3])
            elif rot_idx == 2:
                rotated = torch.rot90(images, k=2, dims=[2, 3])
            else:
                rotated = torch.rot90(images, k=3, dims=[2, 3])

            rotated_images.append(rotated)
            rotation_labels.append(torch.full((batch_size,), rot_idx, device=images.device))

        return torch.cat(rotated_images, dim=0), torch.cat(rotation_labels, dim=0)

    def _compute_rotation_loss(self, images: torch.Tensor) -> torch.Tensor:
        """Compute rotation prediction loss for self-supervision."""
        rotated_images, rotation_labels = self._rotate_batch(images)

        # Get features from backbone
        features = self.model.get_features(rotated_images)
        rotation_logits = self.rotation_head(features)

        rotation_loss = nn.functional.cross_entropy(rotation_logits, rotation_labels)
        return rotation_loss

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """Train on a single task with joint main + rotation loss."""
        self.current_task = task_id
        self.model.train()
        self.rotation_head.train()

        # Optimizer includes rotation head
        optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.rotation_head.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        epoch_losses = []
        epoch_accs = []

        for epoch in range(self.config.epochs_per_task):
            batch_losses = []
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f'Task {task_id} Epoch {epoch+1} (TTT)')

            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                # Main task loss
                outputs = self.model(images)
                main_loss = criterion(outputs, labels)

                # Rotation self-supervision loss
                rotation_loss = self._compute_rotation_loss(images)

                # Add regularization
                reg_loss = self._compute_regularization_loss()

                # Combined loss
                total_loss = main_loss + self.rotation_weight * rotation_loss + reg_loss
                total_loss.backward()

                # Experience replay
                if self.config.use_replay and len(self.buffer) > 0:
                    if batch_idx % self.config.replay_freq == 0:
                        replay_loss = self._replay_step_with_rotation(criterion)
                        if replay_loss is not None:
                            replay_loss.backward()

                optimizer.step()

                # Update feature embeddings periodically
                if (self.config.update_features_freq > 0 and
                    batch_idx % self.config.update_features_freq == 0):
                    self.buffer.update_features(self.model, self.device)

                # Statistics
                batch_losses.append(main_loss.item())
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'loss': f'{np.mean(batch_losses[-10:]):.4f}',
                    'rot_loss': f'{rotation_loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

            epoch_losses.append(np.mean(batch_losses))
            epoch_accs.append(100. * correct / total)

        # Add samples to replay buffer after training
        self._populate_buffer(task_id, train_loader)

        # Post-task operations
        self._after_task(task_id, train_loader)

        return {
            'task_id': task_id,
            'final_loss': epoch_losses[-1],
            'final_acc': epoch_accs[-1],
            'epoch_losses': epoch_losses,
            'epoch_accs': epoch_accs,
        }

    def _replay_step_with_rotation(self, criterion: nn.Module) -> Optional[torch.Tensor]:
        """Replay step with rotation loss for TTT."""
        try:
            replay_images, replay_labels, replay_indices = self.buffer.sample(
                self.config.replay_batch_size,
                strategy=self.config.sampling_strategy,
                **self.config.sampling_kwargs
            )
        except ValueError:
            return None

        replay_images = replay_images.to(self.device)
        replay_labels = replay_labels.to(self.device)

        # Main task replay loss
        outputs = self.model(replay_images)
        replay_loss = criterion(outputs, replay_labels)

        # Rotation loss on replay samples
        rotation_loss = self._compute_rotation_loss(replay_images)

        # Update loss history for forgetting-weighted sampling
        with torch.no_grad():
            per_sample_loss = nn.functional.cross_entropy(
                outputs, replay_labels, reduction='none'
            )
            self.buffer.update_losses(replay_indices, per_sample_loss)

        return replay_loss + self.rotation_weight * rotation_loss

    def _adapt_at_test_time(self, images: torch.Tensor) -> None:
        """Adapt model using rotation prediction before inference."""
        # Only adapt backbone, freeze rotation head
        self.model.train()
        self.rotation_head.eval()

        # Only optimize backbone parameters
        backbone_optimizer = optim.SGD(
            self.model.backbone.parameters(),
            lr=self.ttt_lr
        )

        for _ in range(self.ttt_steps):
            backbone_optimizer.zero_grad()
            rotation_loss = self._compute_rotation_loss(images)
            rotation_loss.backward()
            backbone_optimizer.step()

        self.model.eval()

    def evaluate(self, test_loader: DataLoader) -> float:
        """Evaluate with test-time training adaptation."""
        self.model.eval()
        self.rotation_head.eval()
        correct = 0
        total = 0

        # Save original state for adaptation
        original_state = {
            name: param.clone()
            for name, param in self.model.backbone.named_parameters()
        }

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Adapt on this batch (requires grad)
                if self.ttt_steps > 0:
                    with torch.enable_grad():
                        self._adapt_at_test_time(images)

                # Predict after adaptation
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Restore original parameters for next batch
                if self.ttt_steps > 0:
                    with torch.no_grad():
                        for name, param in self.model.backbone.named_parameters():
                            param.copy_(original_state[name])

        self.model.train()
        return 100. * correct / total
