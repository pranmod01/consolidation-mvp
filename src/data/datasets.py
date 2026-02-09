"""Split datasets for continual learning experiments."""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Optional
import os


# Task definitions for each dataset
SPLIT_MNIST_TASKS = {
    1: [0, 1],
    2: [2, 3],
    3: [4, 5],
    4: [6, 7],
    5: [8, 9],
}

SPLIT_CIFAR10_TASKS = {
    1: [0, 1],  # airplane, automobile
    2: [2, 3],  # bird, cat
    3: [4, 5],  # deer, dog
    4: [6, 7],  # frog, horse
    5: [8, 9],  # ship, truck
}

# For TinyImageNet: 200 classes split into 10 tasks of 20 classes each
SPLIT_TINYIMAGENET_TASKS = {i: list(range((i-1)*20, i*20)) for i in range(1, 11)}


class SplitDataset(Dataset):
    """Base class for split continual learning datasets."""

    def __init__(
        self,
        base_dataset: Dataset,
        classes: List[int],
        remap_labels: bool = True
    ):
        """
        Args:
            base_dataset: Full dataset to filter
            classes: List of class indices to include
            remap_labels: If True, remap labels to 0..n-1
        """
        self.classes = classes
        self.remap_labels = remap_labels
        self.class_to_idx = {c: i for i, c in enumerate(classes)} if remap_labels else {c: c for c in classes}

        # Filter indices for specified classes
        self.indices = []
        self.labels = []
        for idx in range(len(base_dataset)):
            _, label = base_dataset[idx]
            if label in classes:
                self.indices.append(idx)
                self.labels.append(self.class_to_idx[label])

        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        base_idx = self.indices[idx]
        image, _ = self.base_dataset[base_idx]
        label = self.labels[idx]
        return image, label


class SplitMNIST:
    """Split MNIST into 5 binary classification tasks."""

    def __init__(self, data_root: str = './data', remap_labels: bool = False):
        """
        Args:
            data_root: Directory to download/store MNIST
            remap_labels: If True, remap to 0..1 per task. If False, use original 0-9 labels.
        """
        self.data_root = data_root
        self.remap_labels = remap_labels
        self.tasks = SPLIT_MNIST_TASKS
        self.num_tasks = len(self.tasks)
        self.num_classes = 10  # Total classes across all tasks

        # Standard MNIST transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load full MNIST
        self.train_dataset = datasets.MNIST(
            data_root, train=True, download=True, transform=self.transform
        )
        self.test_dataset = datasets.MNIST(
            data_root, train=False, download=True, transform=self.transform
        )

    def get_task_dataset(self, task_id: int, train: bool = True) -> SplitDataset:
        """Get dataset for a specific task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found. Available: {list(self.tasks.keys())}")

        classes = self.tasks[task_id]
        base = self.train_dataset if train else self.test_dataset
        return SplitDataset(base, classes, remap_labels=self.remap_labels)

    def get_task_loaders(
        self,
        task_id: int,
        batch_size: int = 64,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader]:
        """Get train and test loaders for a task."""
        train_ds = self.get_task_dataset(task_id, train=True)
        test_ds = self.get_task_dataset(task_id, train=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, test_loader

    def get_all_tasks_test_loader(self, batch_size: int = 64, num_workers: int = 0) -> DataLoader:
        """Get test loader for all classes seen so far."""
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


class SplitCIFAR10:
    """Split CIFAR-10 into 5 binary classification tasks."""

    def __init__(self, data_root: str = './data', remap_labels: bool = False):
        self.data_root = data_root
        self.remap_labels = remap_labels
        self.tasks = SPLIT_CIFAR10_TASKS
        self.num_tasks = len(self.tasks)
        self.num_classes = 10

        # CIFAR-10 transforms
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        self.train_dataset = datasets.CIFAR10(
            data_root, train=True, download=True, transform=self.transform_train
        )
        self.test_dataset = datasets.CIFAR10(
            data_root, train=False, download=True, transform=self.transform_test
        )

        # Also keep a non-augmented version for storing in buffer
        self.train_dataset_no_aug = datasets.CIFAR10(
            data_root, train=True, download=True, transform=self.transform_test
        )

    def get_task_dataset(self, task_id: int, train: bool = True, augment: bool = True) -> SplitDataset:
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found. Available: {list(self.tasks.keys())}")

        classes = self.tasks[task_id]
        if train:
            base = self.train_dataset if augment else self.train_dataset_no_aug
        else:
            base = self.test_dataset
        return SplitDataset(base, classes, remap_labels=self.remap_labels)

    def get_task_loaders(
        self,
        task_id: int,
        batch_size: int = 64,
        num_workers: int = 2
    ) -> Tuple[DataLoader, DataLoader]:
        train_ds = self.get_task_dataset(task_id, train=True)
        test_ds = self.get_task_dataset(task_id, train=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, test_loader

    def get_all_tasks_test_loader(self, batch_size: int = 64, num_workers: int = 2) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


class SplitTinyImageNet:
    """Split Tiny ImageNet (200 classes) into 10 tasks of 20 classes each."""

    def __init__(self, data_root: str = './data', remap_labels: bool = False):
        self.data_root = data_root
        self.remap_labels = remap_labels
        self.tasks = SPLIT_TINYIMAGENET_TASKS
        self.num_tasks = len(self.tasks)
        self.num_classes = 200

        # TinyImageNet transforms (64x64 images)
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        tiny_imagenet_path = os.path.join(data_root, 'tiny-imagenet-200')

        if not os.path.exists(tiny_imagenet_path):
            raise RuntimeError(
                f"TinyImageNet not found at {tiny_imagenet_path}. "
                "Please download from http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                "and extract to data directory."
            )

        self.train_dataset = datasets.ImageFolder(
            os.path.join(tiny_imagenet_path, 'train'),
            transform=self.transform_train
        )
        self.test_dataset = datasets.ImageFolder(
            os.path.join(tiny_imagenet_path, 'val'),
            transform=self.transform_test
        )

    def get_task_dataset(self, task_id: int, train: bool = True) -> SplitDataset:
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found. Available: {list(self.tasks.keys())}")

        classes = self.tasks[task_id]
        base = self.train_dataset if train else self.test_dataset
        return SplitDataset(base, classes, remap_labels=self.remap_labels)

    def get_task_loaders(
        self,
        task_id: int,
        batch_size: int = 64,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        train_ds = self.get_task_dataset(task_id, train=True)
        test_ds = self.get_task_dataset(task_id, train=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, test_loader


def get_dataset(name: str, data_root: str = './data', remap_labels: bool = False):
    """Factory function to get dataset by name."""
    datasets_map = {
        'split_mnist': SplitMNIST,
        'split_cifar10': SplitCIFAR10,
        'split_tinyimagenet': SplitTinyImageNet,
    }

    if name not in datasets_map:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets_map.keys())}")

    return datasets_map[name](data_root, remap_labels)
