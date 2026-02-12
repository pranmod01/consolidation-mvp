"""Autoencoder-based trainer for continual learning.

Trains an autoencoder with joint classification and reconstruction loss.
The reconstruction loss acts as a regularizer that encourages learning
generalizable representations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm

from .base_trainer import BaseTrainer, TrainingConfig
from ..models.autoencoder import AutoencoderClassifier


class AutoencoderTrainer(BaseTrainer):
    """Trainer for autoencoder-based continual learning.

    Jointly optimizes:
    1. Classification loss (cross-entropy)
    2. Reconstruction loss (MSE)

    The reconstruction loss encourages the encoder to learn
    representations that capture the full input distribution,
    which may help with continual learning.

    Args:
        model: AutoencoderClassifier model
        config: Training configuration
        num_classes: Number of classes
        recon_weight: Weight for reconstruction loss (default: 1.0)
    """

    def __init__(
        self,
        model: AutoencoderClassifier,
        config: TrainingConfig,
        num_classes: int = 10,
        recon_weight: float = 1.0,
    ):
        super().__init__(model, config, num_classes)
        self.recon_weight = recon_weight

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """Train on a single task with joint classification + reconstruction."""
        self.current_task = task_id
        self.model.train()

        optimizer = self._get_optimizer()
        criterion = nn.CrossEntropyLoss()

        epoch_losses = []
        epoch_accs = []

        for epoch in range(self.config.epochs_per_task):
            batch_losses = []
            batch_recon_losses = []
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f'Task {task_id} Epoch {epoch+1} (AE)')

            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                # Forward pass with reconstruction
                logits, z, recon = self.model.forward_all(images)

                # Classification loss
                class_loss = criterion(logits, labels)

                # Reconstruction loss
                recon_loss = self.model.reconstruction_loss(images, recon)

                # Combined loss
                total_loss = class_loss + self.recon_weight * recon_loss

                # Add method-specific regularization
                reg_loss = self._compute_regularization_loss()
                total_loss = total_loss + reg_loss

                total_loss.backward()

                # Experience replay
                if self.config.use_replay and len(self.buffer) > 0:
                    if batch_idx % self.config.replay_freq == 0:
                        replay_loss = self._replay_step_with_recon(criterion)
                        if replay_loss is not None:
                            replay_loss.backward()

                optimizer.step()

                # Update feature embeddings periodically
                if (self.config.update_features_freq > 0 and
                    batch_idx % self.config.update_features_freq == 0):
                    self.buffer.update_features(self.model, self.device)

                # Statistics
                batch_losses.append(class_loss.item())
                batch_recon_losses.append(recon_loss.item())
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'cls_loss': f'{np.mean(batch_losses[-10:]):.4f}',
                    'recon': f'{np.mean(batch_recon_losses[-10:]):.4f}',
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

    def _replay_step_with_recon(self, criterion: nn.Module) -> Optional[torch.Tensor]:
        """Replay step with reconstruction loss."""
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

        # Forward with reconstruction
        logits, z, recon = self.model.forward_all(replay_images)

        # Classification loss
        class_loss = criterion(logits, replay_labels)

        # Reconstruction loss
        recon_loss = self.model.reconstruction_loss(replay_images, recon)

        # Combined replay loss
        replay_loss = class_loss + self.recon_weight * recon_loss

        # Update loss history for forgetting-weighted sampling
        with torch.no_grad():
            per_sample_loss = nn.functional.cross_entropy(
                logits, replay_labels, reduction='none'
            )
            self.buffer.update_losses(replay_indices, per_sample_loss)

        return replay_loss
