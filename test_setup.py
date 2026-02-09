#!/usr/bin/env python3
"""Test script to verify the consolidation MVP setup works correctly."""

import sys
import torch
import numpy as np

def test_imports():
    """Test all imports work."""
    print("Testing imports...")

    from src.models import Classifier, CNNBackbone, ResNetBackbone
    from src.data import SplitMNIST, ReplayBuffer
    from src.training import VanillaReplayTrainer, EWCTrainer, MetaSGDTrainer
    from src.training.base_trainer import TrainingConfig
    from src.consolidation import PatternSeparationSampler, TemporalSpacingSampler
    from src.evaluation import AccuracyMatrix, compute_all_metrics
    from src.utils import ExperimentConfig, set_seed

    print("  All imports successful!")
    return True


def test_model():
    """Test model creation and forward pass."""
    print("\nTesting model...")

    from src.models import Classifier

    # MNIST model
    model = Classifier.create_for_mnist(num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    features = model.get_features(x)

    assert logits.shape == (4, 10), f"Expected (4, 10), got {logits.shape}"
    assert features.shape == (4, 128), f"Expected (4, 128), got {features.shape}"

    print(f"  MNIST model: logits={logits.shape}, features={features.shape}")

    # CIFAR model
    model_cifar = Classifier.create_for_cifar(num_classes=10)
    x_cifar = torch.randn(4, 3, 32, 32)
    logits_cifar = model_cifar(x_cifar)
    features_cifar = model_cifar.get_features(x_cifar)

    assert logits_cifar.shape == (4, 10)
    print(f"  CIFAR model: logits={logits_cifar.shape}, features={features_cifar.shape}")

    return True


def test_dataset():
    """Test dataset loading."""
    print("\nTesting dataset...")

    from src.data import SplitMNIST

    dataset = SplitMNIST(data_root='./data', remap_labels=False)

    # Get task 1 loaders
    train_loader, test_loader = dataset.get_task_loaders(1, batch_size=32)

    # Get one batch
    images, labels = next(iter(train_loader))

    assert images.shape[0] <= 32
    assert images.shape[1:] == (1, 28, 28)
    assert labels.max() <= 9

    print(f"  Task 1: images={images.shape}, labels unique={labels.unique().tolist()}")

    return True


def test_replay_buffer():
    """Test replay buffer with different sampling strategies."""
    print("\nTesting replay buffer...")

    from src.data import ReplayBuffer

    buffer = ReplayBuffer(max_size=100)

    # Add some samples
    images = torch.randn(50, 1, 28, 28)
    labels = torch.randint(0, 10, (50,))
    features = torch.randn(50, 128)

    buffer.add_samples(images, labels, task_id=1, features=features)

    print(f"  Buffer size: {len(buffer)}")

    # Test different sampling strategies
    for strategy in ['random', 'balanced', 'diversity', 'temporal', 'combined']:
        try:
            samples, sample_labels, indices = buffer.sample(16, strategy=strategy)
            print(f"  {strategy}: sampled {len(samples)} samples")
        except Exception as e:
            print(f"  {strategy}: ERROR - {e}")
            return False

    return True


def test_pattern_separation():
    """Test pattern separation sampler."""
    print("\nTesting pattern separation...")

    from src.consolidation import PatternSeparationSampler, compute_diversity_score

    features = torch.randn(100, 128)
    labels = torch.randint(0, 10, (100,))

    sampler = PatternSeparationSampler(method='farthest_point')
    indices = sampler.sample(features, labels, batch_size=20)

    diversity = compute_diversity_score(features, indices)

    # Compare to random sampling
    random_indices = np.random.choice(100, 20, replace=False).tolist()
    random_diversity = compute_diversity_score(features, random_indices)

    print(f"  Farthest-point diversity: {diversity:.4f}")
    print(f"  Random sampling diversity: {random_diversity:.4f}")
    print(f"  Improvement: {(diversity/random_diversity - 1)*100:.1f}%")

    return True


def test_temporal_spacing():
    """Test temporal spacing sampler."""
    print("\nTesting temporal spacing...")

    from src.consolidation import TemporalSpacingSampler
    import time

    sampler = TemporalSpacingSampler(method='age_weighted')

    # Create mock sample statistics
    n_samples = 100
    timestamps = np.array([time.time() - i*10 for i in range(n_samples)])  # Older samples first
    last_replayed = timestamps.copy()
    replay_counts = np.zeros(n_samples)
    losses = np.random.rand(n_samples)

    # Get sampling weights
    weights = sampler.compute_weights(timestamps, last_replayed, replay_counts, losses)

    # Older samples should have higher weights
    print(f"  Weight for oldest sample: {weights[0]:.4f}")
    print(f"  Weight for newest sample: {weights[-1]:.4f}")
    print(f"  Ratio (should be > 1): {weights[0]/weights[-1]:.2f}")

    return True


def test_trainer():
    """Test basic training loop."""
    print("\nTesting trainer (1 epoch, 1 task)...")

    from src.models import Classifier
    from src.data import SplitMNIST
    from src.training import VanillaReplayTrainer
    from src.training.base_trainer import TrainingConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    model = Classifier.create_for_mnist(num_classes=10)
    dataset = SplitMNIST(data_root='./data')

    config = TrainingConfig(
        epochs_per_task=1,
        batch_size=64,
        lr=0.001,
        device=device,
        buffer_size=100,
        sampling_strategy='random',
    )

    trainer = VanillaReplayTrainer(model, config, num_classes=10)

    train_loader, test_loader = dataset.get_task_loaders(1, batch_size=64)

    # Train for 1 epoch
    result = trainer.train_task(1, train_loader, test_loader)

    print(f"  Task 1 final accuracy: {result['final_acc']:.1f}%")
    print(f"  Buffer samples: {len(trainer.buffer)}")

    return True


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("CONSOLIDATION MVP - SETUP VERIFICATION")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Model", test_model),
        ("Dataset", test_dataset),
        ("Replay Buffer", test_replay_buffer),
        ("Pattern Separation", test_pattern_separation),
        ("Temporal Spacing", test_temporal_spacing),
        ("Trainer", test_trainer),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {name}: {status}")
        if not success:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed! Ready to run experiments.")
        print("\nTry: python run_experiments.py quick")
    else:
        print("Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == '__main__':
    run_all_tests()
