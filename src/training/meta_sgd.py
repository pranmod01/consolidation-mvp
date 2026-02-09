"""Meta-SGD trainer for continual learning.

Meta-SGD learns per-parameter learning rates, enabling fast adaptation.
This tests whether consolidation helps fast adaptation methods.

Simpler than MAML - uses learned learning rates instead of second-order gradients.
Reference: Li et al., "Meta-SGD: Learning to Learn Quickly for Few-Shot Learning", 2017
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Optional, List
from copy import deepcopy
import numpy as np

from .base_trainer import BaseTrainer, TrainingConfig


class MetaSGDTrainer(BaseTrainer):
    """Meta-SGD trainer with experience replay.

    Key idea: Learn per-parameter learning rates that enable fast adaptation.
    For continual learning, this means:
    1. Meta-train learning rates on replay buffer
    2. Use adapted learning rates for new task training
    3. Test whether consolidation (diversity/temporal) helps meta-learning

    This is a simplified version suitable for continual learning experiments.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        num_classes: int = 10,
        inner_lr_init: float = 0.01,
        inner_steps: int = 5,
        meta_lr: float = 0.001,
        adaptation_samples: int = 32
    ):
        """
        Args:
            model: Neural network model
            config: Training configuration
            num_classes: Total number of classes
            inner_lr_init: Initial value for learned learning rates
            inner_steps: Number of inner loop adaptation steps
            meta_lr: Learning rate for meta-parameters (including learned LRs)
            adaptation_samples: Number of samples for task adaptation
        """
        super().__init__(model, config, num_classes)

        self.inner_lr_init = inner_lr_init
        self.inner_steps = inner_steps
        self.meta_lr = meta_lr
        self.adaptation_samples = adaptation_samples

        # Learned per-parameter learning rates
        self.learned_lrs = nn.ParameterDict({
            name.replace('.', '_'): nn.Parameter(
                torch.ones_like(param) * inner_lr_init
            )
            for name, param in self.model.named_parameters()
        })
        self.learned_lrs.to(self.device)

    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer that includes both model params and learned LRs."""
        return optim.Adam([
            {'params': self.model.parameters(), 'lr': self.config.lr},
            {'params': self.learned_lrs.parameters(), 'lr': self.meta_lr}
        ], weight_decay=self.config.weight_decay)

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """Train on task using meta-learned adaptation.

        For tasks after the first, uses meta-learned LRs for fast adaptation.
        """
        self.current_task = task_id
        self.model.train()

        optimizer = self._get_optimizer()
        criterion = nn.CrossEntropyLoss()

        epoch_losses = []
        epoch_accs = []

        for epoch in range(self.config.epochs_per_task):
            batch_losses = []
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Meta-learning step using replay buffer (if available)
                if self.config.use_replay and len(self.buffer) > 0 and batch_idx % 5 == 0:
                    self._meta_update_step(optimizer, criterion)

                # Regular training with learned LRs
                optimizer.zero_grad()

                # Fast adaptation using learned LRs
                if task_id > 1 and len(self.buffer) > 0:
                    adapted_model = self._fast_adapt(images, labels, criterion)
                    outputs = adapted_model(images)
                else:
                    outputs = self.model(images)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Experience replay
                if self.config.use_replay and len(self.buffer) > 0:
                    if batch_idx % self.config.replay_freq == 0:
                        replay_loss = self._replay_step(criterion)
                        if replay_loss is not None:
                            replay_loss.backward()
                            optimizer.step()

                # Statistics
                batch_losses.append(loss.item())
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_losses.append(np.mean(batch_losses))
            epoch_accs.append(100. * correct / total)
            print(f"Task {task_id} Epoch {epoch+1}: Loss={epoch_losses[-1]:.4f}, Acc={epoch_accs[-1]:.2f}%")

        # Populate buffer
        self._populate_buffer(task_id, train_loader)

        return {
            'task_id': task_id,
            'final_loss': epoch_losses[-1],
            'final_acc': epoch_accs[-1],
            'epoch_losses': epoch_losses,
            'epoch_accs': epoch_accs,
        }

    def _fast_adapt(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module
    ) -> nn.Module:
        """Fast adaptation using learned per-parameter learning rates.

        Creates a temporary copy of the model and updates it with learned LRs.
        """
        # Create functional copy for adaptation
        adapted_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }

        # Inner loop adaptation
        for _ in range(self.inner_steps):
            # Forward with current adapted params
            outputs = self._forward_with_params(images, adapted_params)
            loss = criterion(outputs, labels)

            # Compute gradients
            grads = torch.autograd.grad(
                loss, list(adapted_params.values()),
                create_graph=True
            )

            # Update with learned LRs
            for (name, param), grad in zip(adapted_params.items(), grads):
                lr_key = name.replace('.', '_')
                if lr_key in self.learned_lrs:
                    # Clamp learning rates to be positive
                    lr = self.learned_lrs[lr_key].abs()
                    adapted_params[name] = param - lr * grad

        # Create adapted model
        adapted_model = deepcopy(self.model)
        state_dict = adapted_model.state_dict()
        for name, param in adapted_params.items():
            state_dict[name] = param
        adapted_model.load_state_dict(state_dict)

        return adapted_model

    def _forward_with_params(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass using specified parameters (for gradient computation)."""
        # This is a simplified version - for full flexibility would need
        # a functional forward implementation
        return self.model(x)

    def _meta_update_step(self, optimizer: optim.Optimizer, criterion: nn.Module):
        """Meta-learning update using replay buffer.

        Samples tasks from buffer and updates learned LRs to improve adaptation.
        """
        # Sample support and query sets from buffer
        try:
            support_images, support_labels, _ = self.buffer.sample(
                self.adaptation_samples,
                strategy=self.config.sampling_strategy,
                **self.config.sampling_kwargs
            )
            query_images, query_labels, _ = self.buffer.sample(
                self.adaptation_samples,
                strategy=self.config.sampling_strategy,
                **self.config.sampling_kwargs
            )
        except ValueError:
            return

        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)

        # Save current params
        original_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }

        # Inner loop: adapt on support set
        optimizer.zero_grad()

        # Fast adaptation
        adapted_params = dict(original_params)
        for _ in range(self.inner_steps):
            outputs = self.model(support_images)
            loss = criterion(outputs, support_labels)

            grads = torch.autograd.grad(
                loss,
                self.model.parameters(),
                create_graph=True,
                allow_unused=True
            )

            for (name, param), grad in zip(self.model.named_parameters(), grads):
                if grad is not None:
                    lr_key = name.replace('.', '_')
                    if lr_key in self.learned_lrs:
                        lr = self.learned_lrs[lr_key].abs()
                        with torch.no_grad():
                            param.data = param.data - lr * grad.detach()

        # Outer loop: compute loss on query set
        query_outputs = self.model(query_images)
        meta_loss = criterion(query_outputs, query_labels)

        # Restore original params before backward
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = original_params[name]

        # Update learned LRs
        meta_loss.backward()
        optimizer.step()

    def get_learned_lrs_stats(self) -> Dict:
        """Get statistics about learned learning rates."""
        stats = {}
        for name, lr in self.learned_lrs.items():
            stats[name] = {
                'mean': lr.abs().mean().item(),
                'max': lr.abs().max().item(),
                'min': lr.abs().min().item(),
                'std': lr.abs().std().item(),
            }
        return stats
