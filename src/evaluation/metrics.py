"""Metrics for evaluating continual learning methods."""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class AccuracyMatrix:
    """Track accuracy across tasks over time.

    accuracy[i][j] = accuracy on task j after training on task i

    After training on task i, we evaluate on all tasks 1..i
    """
    data: Dict[int, Dict[int, float]] = field(default_factory=dict)

    def record(self, after_task: int, eval_task: int, accuracy: float):
        """Record accuracy on eval_task after training on after_task."""
        if after_task not in self.data:
            self.data[after_task] = {}
        self.data[after_task][eval_task] = accuracy

    def get(self, after_task: int, eval_task: int) -> Optional[float]:
        """Get accuracy on eval_task after training on after_task."""
        if after_task in self.data and eval_task in self.data[after_task]:
            return self.data[after_task][eval_task]
        return None

    def to_numpy(self, num_tasks: int) -> np.ndarray:
        """Convert to numpy array (num_tasks x num_tasks).

        Row i = accuracies after training on task i+1
        Column j = accuracy on task j+1
        Lower triangular (only evaluated on tasks seen so far)
        """
        matrix = np.zeros((num_tasks, num_tasks))
        for after_task in range(1, num_tasks + 1):
            for eval_task in range(1, after_task + 1):
                acc = self.get(after_task, eval_task)
                if acc is not None:
                    matrix[after_task - 1, eval_task - 1] = acc
        return matrix

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return dict(self.data)


def compute_forgetting(accuracy_matrix: AccuracyMatrix, num_tasks: int) -> Dict[int, float]:
    """Compute forgetting for each task.

    Forgetting = max accuracy achieved - final accuracy
    (after all tasks have been trained)

    Higher forgetting = worse (more catastrophic forgetting)
    """
    forgetting = {}

    for task in range(1, num_tasks + 1):
        # Max accuracy achieved on this task (right after learning it or later)
        max_acc = 0.0
        for after_task in range(task, num_tasks + 1):
            acc = accuracy_matrix.get(after_task, task)
            if acc is not None:
                max_acc = max(max_acc, acc)

        # Final accuracy (after all tasks)
        final_acc = accuracy_matrix.get(num_tasks, task)

        if final_acc is not None:
            forgetting[task] = max_acc - final_acc
        else:
            forgetting[task] = 0.0

    return forgetting


def compute_forward_transfer(accuracy_matrix: AccuracyMatrix, num_tasks: int) -> Dict[int, float]:
    """Compute forward transfer for each task.

    Forward transfer = accuracy on task i right after learning it
    - random baseline accuracy

    Positive = learning previous tasks helped
    """
    forward_transfer = {}
    random_baseline = 50.0  # For 2-class tasks

    for task in range(1, num_tasks + 1):
        # Accuracy right after learning this task
        acc = accuracy_matrix.get(task, task)
        if acc is not None:
            forward_transfer[task] = acc - random_baseline
        else:
            forward_transfer[task] = 0.0

    return forward_transfer


def compute_backward_transfer(accuracy_matrix: AccuracyMatrix, num_tasks: int) -> Dict[int, float]:
    """Compute backward transfer for each task.

    Backward transfer = final accuracy - accuracy right after learning

    Negative = forgetting (learning new tasks hurt old ones)
    Positive = positive backward transfer (rare, beneficial)
    """
    backward_transfer = {}

    for task in range(1, num_tasks + 1):
        # Accuracy right after learning this task
        after_learning = accuracy_matrix.get(task, task)

        # Final accuracy
        final = accuracy_matrix.get(num_tasks, task)

        if after_learning is not None and final is not None:
            backward_transfer[task] = final - after_learning
        else:
            backward_transfer[task] = 0.0

    return backward_transfer


def compute_all_metrics(accuracy_matrix: AccuracyMatrix, num_tasks: int) -> Dict:
    """Compute all continual learning metrics.

    Returns:
        Dictionary with:
        - average_accuracy: Mean accuracy across all tasks after all training
        - learning_accuracy: Mean accuracy right after learning each task
        - forgetting: Per-task and average forgetting
        - forward_transfer: Per-task forward transfer
        - backward_transfer: Per-task backward transfer
    """
    # Average accuracy (final performance on all tasks)
    final_accuracies = []
    for task in range(1, num_tasks + 1):
        acc = accuracy_matrix.get(num_tasks, task)
        if acc is not None:
            final_accuracies.append(acc)

    average_accuracy = np.mean(final_accuracies) if final_accuracies else 0.0

    # Learning accuracy (performance right after learning each task)
    learning_accuracies = []
    for task in range(1, num_tasks + 1):
        acc = accuracy_matrix.get(task, task)
        if acc is not None:
            learning_accuracies.append(acc)

    learning_accuracy = np.mean(learning_accuracies) if learning_accuracies else 0.0

    # Forgetting
    forgetting = compute_forgetting(accuracy_matrix, num_tasks)
    average_forgetting = np.mean(list(forgetting.values())) if forgetting else 0.0

    # Forward/Backward transfer
    forward_transfer = compute_forward_transfer(accuracy_matrix, num_tasks)
    backward_transfer = compute_backward_transfer(accuracy_matrix, num_tasks)

    avg_forward = np.mean(list(forward_transfer.values())) if forward_transfer else 0.0
    avg_backward = np.mean(list(backward_transfer.values())) if backward_transfer else 0.0

    return {
        'average_accuracy': average_accuracy,
        'learning_accuracy': learning_accuracy,
        'forgetting': forgetting,
        'average_forgetting': average_forgetting,
        'forward_transfer': forward_transfer,
        'average_forward_transfer': avg_forward,
        'backward_transfer': backward_transfer,
        'average_backward_transfer': avg_backward,
    }


def compute_memory_efficiency(
    accuracy: float,
    buffer_size: int,
    baseline_accuracy: float = 50.0
) -> float:
    """Compute memory efficiency: performance gain per buffer sample.

    Higher = better use of memory
    """
    if buffer_size == 0:
        return 0.0

    improvement = accuracy - baseline_accuracy
    return improvement / buffer_size
