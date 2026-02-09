"""Base trainer class for continual learning methods."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm

from ..data.replay_buffer import ReplayBuffer
from ..evaluation.metrics import AccuracyMatrix, compute_all_metrics


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs_per_task: int = 5
    lr: float = 0.001
    batch_size: int = 64
    weight_decay: float = 0.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Replay settings
    use_replay: bool = True
    replay_batch_size: int = 32
    replay_freq: int = 1  # Replay every N batches
    buffer_size: int = 500
    samples_per_task: Optional[int] = None

    # Sampling strategy for replay
    sampling_strategy: str = 'random'  # random, balanced, diversity, temporal, combined
    sampling_kwargs: Dict = field(default_factory=dict)

    # Feature updates for diversity sampling
    update_features_freq: int = 0  # 0 = never, N = every N batches


class BaseTrainer:
    """Base class for continual learning trainers.

    Provides common functionality:
    - Sequential task training
    - Experience replay integration
    - Evaluation across all tasks
    - Metric computation
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        num_classes: int = 10
    ):
        self.model = model
        self.config = config
        self.device = config.device
        self.num_classes = num_classes

        self.model.to(self.device)

        # Initialize replay buffer
        self.buffer = ReplayBuffer(
            max_size=config.buffer_size,
            samples_per_task=config.samples_per_task,
            feature_dim=getattr(model, 'feature_dim', 128)
        )

        # Tracking
        self.accuracy_matrix = AccuracyMatrix()
        self.training_history: List[Dict] = []
        self.current_task = 0

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """Train on a single task.

        Args:
            task_id: Current task ID
            train_loader: Training data loader
            val_loader: Optional validation loader

        Returns:
            Dictionary of training metrics
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

            pbar = tqdm(train_loader, desc=f'Task {task_id} Epoch {epoch+1}')

            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass on current task
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Add method-specific regularization (e.g., EWC)
                reg_loss = self._compute_regularization_loss()
                total_loss = loss + reg_loss

                # Backward pass
                total_loss.backward()

                # Experience replay
                if self.config.use_replay and len(self.buffer) > 0:
                    if batch_idx % self.config.replay_freq == 0:
                        replay_loss = self._replay_step(criterion)
                        if replay_loss is not None:
                            replay_loss.backward()

                optimizer.step()

                # Update feature embeddings periodically
                if (self.config.update_features_freq > 0 and
                    batch_idx % self.config.update_features_freq == 0):
                    self.buffer.update_features(self.model, self.device)

                # Statistics
                batch_losses.append(loss.item())
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'loss': f'{np.mean(batch_losses[-10:]):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

            epoch_losses.append(np.mean(batch_losses))
            epoch_accs.append(100. * correct / total)

        # Add samples to replay buffer after training on task
        self._populate_buffer(task_id, train_loader)

        # Method-specific post-task operations (e.g., compute Fisher for EWC)
        self._after_task(task_id, train_loader)

        return {
            'task_id': task_id,
            'final_loss': epoch_losses[-1],
            'final_acc': epoch_accs[-1],
            'epoch_losses': epoch_losses,
            'epoch_accs': epoch_accs,
        }

    def _replay_step(self, criterion: nn.Module) -> Optional[torch.Tensor]:
        """Perform replay step. Returns replay loss (not yet backpropagated)."""
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

        outputs = self.model(replay_images)
        replay_loss = criterion(outputs, replay_labels)

        # Update loss history for forgetting-weighted sampling
        with torch.no_grad():
            per_sample_loss = nn.functional.cross_entropy(
                outputs, replay_labels, reduction='none'
            )
            self.buffer.update_losses(replay_indices, per_sample_loss)

        return replay_loss

    def _populate_buffer(self, task_id: int, train_loader: DataLoader):
        """Add samples from current task to replay buffer."""
        self.model.eval()
        samples_added = 0
        max_samples = self.config.samples_per_task or (self.config.buffer_size // 5)

        with torch.no_grad():
            for images, labels in train_loader:
                if samples_added >= max_samples:
                    break

                images_device = images.to(self.device)
                features = self.model.get_features(images_device)

                # Add to buffer
                batch_samples = min(images.size(0), max_samples - samples_added)
                self.buffer.add_samples(
                    images[:batch_samples],
                    labels[:batch_samples],
                    task_id,
                    features[:batch_samples]
                )
                samples_added += batch_samples

        self.model.train()

    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer for current task. Override for meta-learning."""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

    def _compute_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss. Override for EWC."""
        return torch.tensor(0.0, device=self.device)

    def _after_task(self, task_id: int, train_loader: DataLoader):
        """Post-task operations. Override for EWC Fisher computation."""
        pass

    def evaluate(self, test_loader: DataLoader) -> float:
        """Evaluate model on a test set."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        self.model.train()
        return 100. * correct / total

    def evaluate_all_tasks(
        self,
        task_loaders: Dict[int, DataLoader],
        after_task: int
    ):
        """Evaluate on all tasks and record in accuracy matrix."""
        for task_id, loader in task_loaders.items():
            acc = self.evaluate(loader)
            self.accuracy_matrix.record(after_task, task_id, acc)

    def train_continual(
        self,
        dataset,
        num_tasks: int = 5
    ) -> Dict:
        """Train on all tasks sequentially.

        Args:
            dataset: Dataset object with get_task_loaders method
            num_tasks: Number of tasks to train on

        Returns:
            Dictionary with accuracy matrix and all metrics
        """
        test_loaders = {}

        for task_id in range(1, num_tasks + 1):
            train_loader, test_loader = dataset.get_task_loaders(
                task_id,
                batch_size=self.config.batch_size
            )
            test_loaders[task_id] = test_loader

            # Train on current task
            task_metrics = self.train_task(task_id, train_loader, test_loader)
            self.training_history.append(task_metrics)

            # Evaluate on all tasks seen so far
            self.evaluate_all_tasks(
                {t: test_loaders[t] for t in range(1, task_id + 1)},
                after_task=task_id
            )

            # Print progress
            current_accs = [
                self.accuracy_matrix.get(task_id, t)
                for t in range(1, task_id + 1)
            ]
            print(f"After Task {task_id}: Accuracies = {[f'{a:.1f}%' for a in current_accs]}")

        # Compute final metrics
        metrics = compute_all_metrics(self.accuracy_matrix, num_tasks)

        return {
            'accuracy_matrix': self.accuracy_matrix,
            'metrics': metrics,
            'training_history': self.training_history,
            'buffer_stats': self.buffer.get_stats(),
        }
