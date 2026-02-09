"""Elastic Weight Consolidation (EWC) trainer.

EWC adds a regularization term that penalizes changes to parameters
that are important for previous tasks, measured by the Fisher information.

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", 2017
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from copy import deepcopy

from .base_trainer import BaseTrainer, TrainingConfig


class EWCTrainer(BaseTrainer):
    """EWC trainer with optional replay buffer.

    Combines regularization-based approach with experience replay.
    Tests whether consolidation mechanisms help differently for
    regularization-based vs pure memory-based methods.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        num_classes: int = 10,
        ewc_lambda: float = 400.0,
        fisher_samples: int = 200,
        online: bool = False,
        gamma: float = 0.95
    ):
        """
        Args:
            model: Neural network model
            config: Training configuration
            num_classes: Total number of classes
            ewc_lambda: Regularization strength
            fisher_samples: Number of samples for Fisher estimation
            online: If True, use online EWC (running average of Fisher)
            gamma: Decay factor for online EWC
        """
        super().__init__(model, config, num_classes)

        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.online = online
        self.gamma = gamma

        # Storage for Fisher information and optimal parameters per task
        self.fisher_matrices: Dict[int, Dict[str, torch.Tensor]] = {}
        self.optimal_params: Dict[int, Dict[str, torch.Tensor]] = {}

        # For online EWC: running average of Fisher
        self.running_fisher: Optional[Dict[str, torch.Tensor]] = None
        self.running_params: Optional[Dict[str, torch.Tensor]] = None

    def _compute_regularization_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if not self.fisher_matrices:
            return torch.tensor(0.0, device=self.device)

        ewc_loss = torch.tensor(0.0, device=self.device)

        if self.online:
            # Online EWC: single consolidated Fisher
            if self.running_fisher is not None:
                for name, param in self.model.named_parameters():
                    if name in self.running_fisher:
                        fisher = self.running_fisher[name]
                        optimal = self.running_params[name]
                        ewc_loss += (fisher * (param - optimal).pow(2)).sum()
        else:
            # Standard EWC: sum over all previous tasks
            for task_id in self.fisher_matrices:
                fisher = self.fisher_matrices[task_id]
                optimal = self.optimal_params[task_id]

                for name, param in self.model.named_parameters():
                    if name in fisher:
                        ewc_loss += (fisher[name] * (param - optimal[name]).pow(2)).sum()

        return 0.5 * self.ewc_lambda * ewc_loss

    def _after_task(self, task_id: int, train_loader: DataLoader):
        """Compute Fisher information matrix after task completion."""
        print(f"Computing Fisher information for task {task_id}...")

        fisher = self._compute_fisher(train_loader)

        if self.online:
            # Online EWC: update running average
            if self.running_fisher is None:
                self.running_fisher = fisher
                self.running_params = {
                    name: param.clone().detach()
                    for name, param in self.model.named_parameters()
                }
            else:
                for name in self.running_fisher:
                    self.running_fisher[name] = (
                        self.gamma * self.running_fisher[name] +
                        (1 - self.gamma) * fisher[name]
                    )
                    # Update optimal params to current
                    self.running_params[name] = self.model.state_dict()[name].clone()
        else:
            # Standard EWC: store per-task
            self.fisher_matrices[task_id] = fisher
            self.optimal_params[task_id] = {
                name: param.clone().detach()
                for name, param in self.model.named_parameters()
            }

    def _compute_fisher(self, data_loader: DataLoader) -> Dict[str, torch.Tensor]:
        """Compute Fisher information matrix using empirical Fisher.

        Uses the gradient of the log-likelihood at the optimal parameters.
        """
        self.model.eval()
        fisher = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.model.named_parameters()
        }

        samples_used = 0
        criterion = nn.CrossEntropyLoss()

        for images, labels in data_loader:
            if samples_used >= self.fisher_samples:
                break

            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = images.size(0)

            # Forward pass
            self.model.zero_grad()
            outputs = self.model(images)

            # Use true labels for empirical Fisher
            loss = criterion(outputs, labels)
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2) * batch_size

            samples_used += batch_size

        # Normalize by number of samples
        for name in fisher:
            fisher[name] /= samples_used

        self.model.train()
        return fisher

    def get_fisher_stats(self) -> Dict:
        """Get statistics about Fisher matrices for analysis."""
        stats = {}

        for task_id, fisher in self.fisher_matrices.items():
            task_stats = {}
            for name, values in fisher.items():
                task_stats[name] = {
                    'mean': values.mean().item(),
                    'max': values.max().item(),
                    'min': values.min().item(),
                    'std': values.std().item(),
                }
            stats[task_id] = task_stats

        return stats
