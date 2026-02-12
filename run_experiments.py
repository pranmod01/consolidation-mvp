#!/usr/bin/env python3
"""Main experiment runner for consolidation MVP.

Usage:
    python run_experiments.py poc           # Minimal proof-of-concept (CPU friendly)
    python run_experiments.py quick         # Quick test (1 method, 1 buffer size)
    python run_experiments.py phase1        # Run Phase 1: Single mechanism tests
    python run_experiments.py phase2        # Run Phase 2: Combined mechanisms
    python run_experiments.py phase3        # Run Phase 3: Validation at scale
    python run_experiments.py all           # Run all phases
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

from src.models import Classifier
from src.data import SplitMNIST, SplitCIFAR10
from src.training import VanillaReplayTrainer, EWCTrainer, MetaSGDTrainer, TTTTrainer, AutoencoderTrainer
from src.models.autoencoder import AutoencoderClassifier
from src.training.base_trainer import TrainingConfig
from src.evaluation import (
    plot_accuracy_matrix,
    plot_forgetting_curves,
    plot_method_comparison,
    plot_consolidation_comparison,
    create_summary_table,
)
from src.utils import ExperimentConfig, set_seed, save_config


def create_trainer(config: ExperimentConfig, model):
    """Create appropriate trainer based on config."""
    training_config = TrainingConfig(
        epochs_per_task=config.epochs_per_task,
        lr=config.lr,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        device=config.get_device(),
        use_replay=True,
        replay_batch_size=config.replay_batch_size,
        replay_freq=config.replay_freq,
        buffer_size=config.buffer_size,
        samples_per_task=config.samples_per_task,
        sampling_strategy=config.sampling_strategy,
        sampling_kwargs={
            'temperature': config.ps_temperature,
            'age_weight': config.ts_age_exponent,
            'diversity_weight': config.diversity_weight,
            'temporal_weight': config.temporal_weight,
        },
        update_features_freq=config.update_features_freq,
    )

    if config.method == 'vanilla_replay':
        return VanillaReplayTrainer(model, training_config, config.num_classes)
    elif config.method == 'ewc':
        return EWCTrainer(
            model, training_config, config.num_classes,
            ewc_lambda=config.ewc_lambda,
            online=config.ewc_online,
            gamma=config.ewc_gamma,
        )
    elif config.method == 'meta_sgd':
        return MetaSGDTrainer(
            model, training_config, config.num_classes,
            inner_lr_init=config.meta_inner_lr,
            inner_steps=config.meta_inner_steps,
            meta_lr=config.meta_lr,
        )
    elif config.method == 'ttt':
        return TTTTrainer(
            model, training_config, config.num_classes,
            ttt_lr=config.ttt_lr,
            ttt_steps=config.ttt_steps,
            rotation_weight=config.rotation_weight,
        )
    else:
        raise ValueError(f"Unknown method: {config.method}")


def create_model(config: ExperimentConfig):
    """Create model based on config."""
    if config.dataset == 'split_mnist':
        return Classifier.create_for_mnist(config.num_classes, config.feature_dim)
    else:
        return Classifier.create_for_cifar(config.num_classes, config.feature_dim)


def create_dataset(config: ExperimentConfig):
    """Create dataset based on config."""
    if config.dataset == 'split_mnist':
        return SplitMNIST(config.data_root, remap_labels=False)
    elif config.dataset == 'split_cifar10':
        return SplitCIFAR10(config.data_root, remap_labels=False)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")


def run_single_experiment(config: ExperimentConfig) -> Dict:
    """Run a single experiment with given config."""
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"Method: {config.method}, Sampling: {config.sampling_strategy}")
    print(f"Buffer size: {config.buffer_size}")
    print(f"{'='*60}\n")

    set_seed(config.seed)

    # Create components
    model = create_model(config)
    dataset = create_dataset(config)
    trainer = create_trainer(config, model)

    # Run training
    results = trainer.train_continual(dataset, config.num_tasks)

    # Add config to results
    results['config'] = config.to_dict()

    return results


def run_phase1(output_dir: Path) -> Dict[str, Dict]:
    """Phase 1: Single mechanism tests.

    Test Pattern Separation and Temporal Spacing separately
    on each base method with different buffer sizes.
    """
    print("\n" + "="*80)
    print("PHASE 1: Single Mechanism Tests")
    print("="*80)

    results = {}
    base_methods = ['vanilla_replay', 'ewc']  # Skip meta_sgd for speed
    buffer_sizes = [200, 500, 1000]
    strategies = ['random', 'diversity', 'temporal']

    for method in base_methods:
        results[method] = {}

        for buffer_size in buffer_sizes:
            for strategy in strategies:
                config = ExperimentConfig(
                    name=f"{method}_{strategy}_buf{buffer_size}",
                    method=method,
                    buffer_size=buffer_size,
                    sampling_strategy=strategy,
                    update_features_freq=50 if strategy == 'diversity' else 0,
                    epochs_per_task=3,  # Faster for phase 1
                )

                result = run_single_experiment(config)
                key = f"{strategy}_buf{buffer_size}"
                results[method][key] = result

                # Save individual result
                result_path = output_dir / f"phase1/{method}_{strategy}_buf{buffer_size}.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                with open(result_path, 'w') as f:
                    json.dump({
                        'config': config.to_dict(),
                        'metrics': result['metrics'],
                    }, f, indent=2)

    return results


def run_phase2(output_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """Phase 2: Combined mechanisms.

    Test all combinations: baseline, +PS, +TS, +PS+TS
    on each base method with standard buffer size.
    """
    print("\n" + "="*80)
    print("PHASE 2: Combined Mechanism Tests")
    print("="*80)

    results = {}
    base_methods = ['vanilla_replay', 'ewc', 'meta_sgd']
    strategies = {
        'random': 'Baseline',
        'diversity': '+PS',
        'temporal': '+TS',
        'combined': '+PS+TS',
    }

    for method in base_methods:
        results[method] = {}

        for strategy, label in strategies.items():
            config = ExperimentConfig(
                name=f"{method}_{strategy}",
                method=method,
                buffer_size=500,
                sampling_strategy=strategy,
                update_features_freq=50 if strategy in ['diversity', 'combined'] else 0,
                epochs_per_task=5,
                dataset='split_cifar10' if method != 'meta_sgd' else 'split_mnist',
            )

            result = run_single_experiment(config)
            results[method][label] = result

            # Save individual result
            result_path = output_dir / f"phase2/{method}_{strategy}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, 'w') as f:
                json.dump({
                    'config': config.to_dict(),
                    'metrics': result['metrics'],
                }, f, indent=2)

    # Generate comparison plot
    try:
        fig = plot_consolidation_comparison(
            results,
            base_methods,
            list(strategies.values()),
            metric='average_accuracy',
            title='Consolidation Effects Across Methods',
            save_path=str(output_dir / 'phase2/consolidation_comparison.png')
        )
    except Exception as e:
        print(f"Warning: Could not generate plot: {e}")

    return results


def run_phase3(output_dir: Path, best_method: str = 'vanilla_replay', best_strategy: str = 'combined') -> Dict:
    """Phase 3: Validation at scale.

    Take best-performing combination and test on harder problems.
    """
    print("\n" + "="*80)
    print("PHASE 3: Validation at Scale")
    print(f"Best method: {best_method}, Best strategy: {best_strategy}")
    print("="*80)

    results = {}

    # Test on Split-CIFAR10
    config_cifar = ExperimentConfig(
        name=f"{best_method}_{best_strategy}_cifar10",
        method=best_method,
        dataset='split_cifar10',
        sampling_strategy=best_strategy,
        buffer_size=1000,
        epochs_per_task=10,
        update_features_freq=50 if best_strategy in ['diversity', 'combined'] else 0,
    )

    results['cifar10'] = run_single_experiment(config_cifar)

    # Baseline comparison on CIFAR10
    config_cifar_baseline = ExperimentConfig(
        name=f"{best_method}_random_cifar10",
        method=best_method,
        dataset='split_cifar10',
        sampling_strategy='random',
        buffer_size=1000,
        epochs_per_task=10,
    )

    results['cifar10_baseline'] = run_single_experiment(config_cifar_baseline)

    # Save results
    for name, result in results.items():
        result_path = output_dir / f"phase3/{name}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump({
                'metrics': result['metrics'],
            }, f, indent=2)

    # Print comparison
    print("\n" + "="*60)
    print("Phase 3 Results Summary")
    print("="*60)
    print(f"CIFAR10 Baseline: {results['cifar10_baseline']['metrics']['average_accuracy']:.1f}%")
    print(f"CIFAR10 +PS+TS:   {results['cifar10']['metrics']['average_accuracy']:.1f}%")
    improvement = (
        results['cifar10']['metrics']['average_accuracy'] -
        results['cifar10_baseline']['metrics']['average_accuracy']
    )
    print(f"Improvement: {improvement:+.1f}%")

    return results


def run_quick_test(output_dir: Path) -> Dict:
    """Quick test to verify everything works."""
    print("\n" + "="*80)
    print("QUICK TEST")
    print("="*80)

    config = ExperimentConfig(
        name="quick_test",
        method='vanilla_replay',
        dataset='split_mnist',
        buffer_size=200,
        sampling_strategy='combined',
        epochs_per_task=2,
        num_tasks=3,  # Only 3 tasks for speed
        update_features_freq=20,
    )

    result = run_single_experiment(config)

    print("\n" + "="*60)
    print("Quick Test Results")
    print("="*60)
    print(f"Average Accuracy: {result['metrics']['average_accuracy']:.1f}%")
    print(f"Average Forgetting: {result['metrics']['average_forgetting']:.1f}%")

    # Save result
    result_path = output_dir / "quick_test.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({
            'config': config.to_dict(),
            'metrics': result['metrics'],
        }, f, indent=2)

    return result


def run_poc(output_dir: Path) -> Dict:
    """Minimal proof-of-concept experiment.

    Tests the core hypothesis with minimal compute:
    - 5 strategies to compare:
      1. Baseline (no replay)
      2. Vanilla + Random Replay
      3. Vanilla + PS+TS Replay
      4. TTT + Random Replay
      5. TTT + PS+TS Replay
    - Split-MNIST only
    - 5 tasks, 2 epochs each
    - Small buffer (200)

    Should run in ~10 minutes on CPU.
    Produces comparison of accuracy and forgetting scores.
    """
    print("\n" + "="*80)
    print("PROOF OF CONCEPT - Comparing Replay Strategies")
    print("="*80)
    print("Testing 5 strategies:")
    print("  1. Baseline (no replay)")
    print("  2. Vanilla + Random Replay")
    print("  3. Vanilla + PS+TS Replay")
    print("  4. AE + Random Replay")
    print("  5. AE + PS+TS Replay")
    print("="*80)

    results = {}

    # Define experiment configurations
    experiments = [
        {
            'key': 'baseline',
            'label': 'Baseline (no replay)',
            'method': 'vanilla_replay',
            'use_replay': False,
            'sampling_strategy': 'random',
        },
        {
            'key': 'vanilla_random',
            'label': 'Vanilla + Random Replay',
            'method': 'vanilla_replay',
            'use_replay': True,
            'sampling_strategy': 'random',
        },
        {
            'key': 'vanilla_psts',
            'label': 'Vanilla + PS+TS Replay',
            'method': 'vanilla_replay',
            'use_replay': True,
            'sampling_strategy': 'combined',
        },
        {
            'key': 'ae_random',
            'label': 'AE + Random Replay',
            'method': 'autoencoder',
            'use_replay': True,
            'sampling_strategy': 'random',
        },
        {
            'key': 'ae_psts',
            'label': 'AE + PS+TS Replay',
            'method': 'autoencoder',
            'use_replay': True,
            'sampling_strategy': 'combined',
        },
    ]

    for exp in experiments:
        print(f"\n>>> Running: {exp['label']}")

        config = ExperimentConfig(
            name=f"poc_{exp['key']}",
            method=exp['method'],
            dataset='split_mnist',
            buffer_size=200,
            sampling_strategy=exp['sampling_strategy'],
            epochs_per_task=2,
            num_tasks=5,
            update_features_freq=30 if exp['sampling_strategy'] == 'combined' else 0,
        )

        # Create trainer with modified replay setting
        set_seed(config.seed)
        model = create_model(config)
        dataset = create_dataset(config)

        training_config = TrainingConfig(
            epochs_per_task=config.epochs_per_task,
            lr=config.lr,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            device=config.get_device(),
            use_replay=exp['use_replay'],
            replay_batch_size=config.replay_batch_size,
            replay_freq=config.replay_freq,
            buffer_size=config.buffer_size,
            samples_per_task=config.samples_per_task,
            sampling_strategy=config.sampling_strategy,
            sampling_kwargs={
                'temperature': config.ps_temperature,
                'age_weight': config.ts_age_exponent,
                'diversity_weight': config.diversity_weight,
                'temporal_weight': config.temporal_weight,
            },
            update_features_freq=config.update_features_freq,
        )

        if exp['method'] == 'vanilla_replay':
            trainer = VanillaReplayTrainer(model, training_config, config.num_classes)
        elif exp['method'] == 'ttt':
            trainer = TTTTrainer(
                model, training_config, config.num_classes,
                ttt_lr=config.ttt_lr,
                ttt_steps=config.ttt_steps,
                rotation_weight=config.rotation_weight,
            )
        elif exp['method'] == 'autoencoder':
            # Create autoencoder model instead
            ae_model = AutoencoderClassifier.create_for_mnist(
                num_classes=config.num_classes,
                latent_dim=config.feature_dim
            )
            trainer = AutoencoderTrainer(
                ae_model, training_config, config.num_classes,
                recon_weight=1.0,
            )
        else:
            raise ValueError(f"Unknown method: {exp['method']}")

        result = trainer.train_continual(dataset, config.num_tasks)
        result['config'] = config.to_dict()
        results[exp['label']] = result

        # Save individual result
        result_path = output_dir / f"poc/{exp['key']}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump({
                'config': config.to_dict(),
                'metrics': result['metrics'],
            }, f, indent=2)

    # Print comparison table
    print("\n" + "="*80)
    print("PROOF OF CONCEPT RESULTS")
    print("="*80)
    print(f"{'Strategy':<30} {'Avg Accuracy':>15} {'Avg Forgetting':>15}")
    print("-"*80)

    baseline_acc = results['Baseline (no replay)']['metrics']['average_accuracy']
    for exp in experiments:
        label = exp['label']
        result = results[label]
        acc = result['metrics']['average_accuracy']
        fgt = result['metrics']['average_forgetting']
        delta = acc - baseline_acc
        delta_str = f"({delta:+.1f}%)" if label != 'Baseline (no replay)' else ""
        print(f"{label:<30} {acc:>14.1f}% {fgt:>14.1f}%  {delta_str}")

    print("-"*80)

    # Key findings
    vanilla_random_acc = results['Vanilla + Random Replay']['metrics']['average_accuracy']
    vanilla_psts_acc = results['Vanilla + PS+TS Replay']['metrics']['average_accuracy']
    ae_random_acc = results['AE + Random Replay']['metrics']['average_accuracy']
    ae_psts_acc = results['AE + PS+TS Replay']['metrics']['average_accuracy']

    print("\nKEY FINDINGS:")
    print(f"  - Vanilla Random vs Baseline: {vanilla_random_acc - baseline_acc:+.1f}%")
    print(f"  - Vanilla PS+TS vs Baseline:  {vanilla_psts_acc - baseline_acc:+.1f}%")
    print(f"  - AE Random vs Baseline:      {ae_random_acc - baseline_acc:+.1f}%")
    print(f"  - AE PS+TS vs Baseline:       {ae_psts_acc - baseline_acc:+.1f}%")
    print(f"  - AE PS+TS vs AE Random:      {ae_psts_acc - ae_random_acc:+.1f}%")

    # Best method
    best_label = max(results.keys(), key=lambda k: results[k]['metrics']['average_accuracy'])
    best_acc = results[best_label]['metrics']['average_accuracy']
    print(f"\nBEST: {best_label} with {best_acc:.1f}% accuracy")

    # Generate plots
    print("\nGenerating plots...")
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    # Plot 1: Accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [exp['label'] for exp in experiments]
    accuracies = [results[exp['label']]['metrics']['average_accuracy'] for exp in experiments]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    bars = ax.bar(range(len(labels)), accuracies, color=colors, edgecolor='black', linewidth=1.5)
    for bar, acc in zip(bars, accuracies):
        ax.annotate(f'{acc:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Average Accuracy (%)', fontsize=12)
    ax.set_title('POC: Average Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(accuracies) * 1.15])
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=baseline_acc, color='red', linestyle='--', alpha=0.7, label=f'Baseline: {baseline_acc:.1f}%')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'poc/accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Forgetting comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    forgettings = [results[exp['label']]['metrics']['average_forgetting'] for exp in experiments]

    bars = ax.bar(range(len(labels)), forgettings, color=colors, edgecolor='black', linewidth=1.5)
    for bar, fgt in zip(bars, forgettings):
        ax.annotate(f'{fgt:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Average Forgetting (%)', fontsize=12)
    ax.set_title('POC: Average Forgetting Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(forgettings) * 1.25])
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'poc/forgetting_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Forgetting curves for each strategy
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    num_tasks = 5

    for idx, exp in enumerate(experiments):
        ax = axes[idx]
        label = exp['label']
        acc_matrix = results[label]['accuracy_matrix']

        task_colors = plt.cm.tab10(np.linspace(0, 1, num_tasks))

        for task in range(1, num_tasks + 1):
            accs = []
            x_vals = []
            for after_task in range(task, num_tasks + 1):
                acc = acc_matrix.get(after_task, task)
                if acc is not None:
                    accs.append(acc)
                    x_vals.append(after_task)

            if accs:
                ax.plot(x_vals, accs, marker='o', label=f'Task {task}',
                        color=task_colors[task-1], linewidth=2, markersize=6)

        ax.set_xlabel('After Training on Task', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xticks(range(1, num_tasks + 1))
        ax.set_ylim([0, 105])
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide the 6th subplot (we have 5 experiments)
    axes[5].axis('off')

    plt.suptitle('POC: Forgetting Curves by Strategy', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'poc/forgetting_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Combined accuracy + forgetting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy subplot
    bars1 = ax1.bar(range(len(labels)), accuracies, color=colors, edgecolor='black')
    for bar, acc in zip(bars1, accuracies):
        ax1.annotate(f'{acc:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels([l.replace(' + ', '\n+ ').replace(' (', '\n(') for l in labels], fontsize=8)
    ax1.set_ylabel('Average Accuracy (%)', fontsize=11)
    ax1.set_title('Accuracy (Higher is Better)', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, max(accuracies) * 1.15])
    ax1.grid(True, axis='y', alpha=0.3)

    # Forgetting subplot
    bars2 = ax2.bar(range(len(labels)), forgettings, color=colors, edgecolor='black')
    for bar, fgt in zip(bars2, forgettings):
        ax2.annotate(f'{fgt:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels([l.replace(' + ', '\n+ ').replace(' (', '\n(') for l in labels], fontsize=8)
    ax2.set_ylabel('Average Forgetting (%)', fontsize=11)
    ax2.set_title('Forgetting (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, max(forgettings) * 1.25])
    ax2.grid(True, axis='y', alpha=0.3)

    plt.suptitle('POC Results: Accuracy vs Forgetting', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'poc/accuracy_vs_forgetting.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  - Saved: accuracy_comparison.png")
    print(f"  - Saved: forgetting_comparison.png")
    print(f"  - Saved: forgetting_curves.png")
    print(f"  - Saved: accuracy_vs_forgetting.png")

    # Save summary
    summary_path = output_dir / "poc/SUMMARY.md"
    with open(summary_path, 'w') as f:
        f.write("# Proof of Concept Results\n\n")
        f.write("## Hypothesis\n")
        f.write("Comparing different replay strategies and methods for continual learning:\n")
        f.write("- Does replay help? (Baseline vs Replay methods)\n")
        f.write("- Does PS+TS improve over random replay?\n")
        f.write("- Does Autoencoder (reconstruction loss) help?\n")
        f.write("- Does combining Autoencoder with PS+TS give the best results?\n\n")
        f.write("## Setup\n")
        f.write("- Dataset: Split-MNIST (5 tasks)\n")
        f.write("- Buffer: 200 samples\n")
        f.write("- Epochs: 2 per task\n\n")
        f.write("## Results\n\n")
        f.write("| Strategy | Avg Accuracy | Avg Forgetting | vs Baseline |\n")
        f.write("|----------|--------------|----------------|-------------|\n")
        for exp in experiments:
            label = exp['label']
            result = results[label]
            acc = result['metrics']['average_accuracy']
            fgt = result['metrics']['average_forgetting']
            delta = acc - baseline_acc
            delta_str = f"{delta:+.1f}%" if label != 'Baseline (no replay)' else "-"
            f.write(f"| {label} | {acc:.1f}% | {fgt:.1f}% | {delta_str} |\n")
        f.write(f"\n## Key Findings\n\n")
        f.write(f"- **Vanilla Random vs Baseline:** {vanilla_random_acc - baseline_acc:+.1f}%\n")
        f.write(f"- **Vanilla PS+TS vs Baseline:** {vanilla_psts_acc - baseline_acc:+.1f}%\n")
        f.write(f"- **AE Random vs Baseline:** {ae_random_acc - baseline_acc:+.1f}%\n")
        f.write(f"- **AE PS+TS vs Baseline:** {ae_psts_acc - baseline_acc:+.1f}%\n")
        f.write(f"- **AE PS+TS vs AE Random:** {ae_psts_acc - ae_random_acc:+.1f}%\n\n")
        f.write(f"## Conclusion\n\n")
        f.write(f"**Best Method:** {best_label} with {best_acc:.1f}% accuracy\n")

    print(f"\nResults saved to: {output_dir}/poc/")
    print(f"Summary: {summary_path}")

    return results


def generate_report(output_dir: Path, all_results: Dict):
    """Generate final report summarizing all experiments."""
    report_path = output_dir / "REPORT.md"

    with open(report_path, 'w') as f:
        f.write("# Consolidation MVP Experiment Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")
        f.write("This experiment tests whether bio-inspired consolidation mechanisms ")
        f.write("(Pattern Separation + Temporal Spacing) improve continual learning performance ")
        f.write("across different base methods.\n\n")

        if 'phase2' in all_results:
            f.write("## Phase 2: Combined Mechanisms\n\n")
            for method, strategies in all_results['phase2'].items():
                f.write(f"### {method}\n\n")
                f.write("| Strategy | Avg Accuracy | Avg Forgetting |\n")
                f.write("|----------|--------------|----------------|\n")
                for strategy, result in strategies.items():
                    acc = result['metrics']['average_accuracy']
                    fgt = result['metrics']['average_forgetting']
                    f.write(f"| {strategy} | {acc:.1f}% | {fgt:.1f}% |\n")
                f.write("\n")

        if 'phase3' in all_results:
            f.write("## Phase 3: Validation at Scale\n\n")
            f.write("| Dataset | Method | Avg Accuracy |\n")
            f.write("|---------|--------|-------------|\n")
            for name, result in all_results['phase3'].items():
                acc = result['metrics']['average_accuracy']
                f.write(f"| CIFAR10 | {name} | {acc:.1f}% |\n")

    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run consolidation experiments")
    parser.add_argument(
        'phase',
        choices=['poc', 'quick', 'phase1', 'phase2', 'phase3', 'all'],
        help='Which experiment phase to run'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./results',
        help='Output directory for results'
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.phase == 'poc':
        run_poc(output_dir)
        return

    if args.phase == 'quick':
        run_quick_test(output_dir)
        return

    if args.phase in ['phase1', 'all']:
        all_results['phase1'] = run_phase1(output_dir)

    if args.phase in ['phase2', 'all']:
        all_results['phase2'] = run_phase2(output_dir)

    if args.phase in ['phase3', 'all']:
        all_results['phase3'] = run_phase3(output_dir)

    if args.phase == 'all':
        generate_report(output_dir, all_results)

    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
