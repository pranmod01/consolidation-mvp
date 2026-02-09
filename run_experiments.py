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
from src.training import VanillaReplayTrainer, EWCTrainer, MetaSGDTrainer
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
    - 1 base method (vanilla_replay)
    - 4 strategies (random, +PS, +TS, +PS+TS)
    - Split-MNIST only
    - 3 tasks, 2 epochs each
    - Small buffer (200)

    Should run in ~5 minutes on CPU.
    Produces 4 JSON files to show consolidation helps.
    """
    print("\n" + "="*80)
    print("PROOF OF CONCEPT - Minimal CPU-Friendly Experiment")
    print("="*80)
    print("Testing: Does consolidation improve vanilla replay on Split-MNIST?")
    print("4 experiments: Baseline vs +PS vs +TS vs +PS+TS")
    print("="*80)

    results = {}
    strategies = {
        'random': 'Baseline',
        'diversity': '+PS (Pattern Separation)',
        'temporal': '+TS (Temporal Spacing)',
        'combined': '+PS+TS (Combined)',
    }

    for strategy, label in strategies.items():
        print(f"\n>>> Running: {label}")

        config = ExperimentConfig(
            name=f"poc_{strategy}",
            method='vanilla_replay',
            dataset='split_mnist',
            buffer_size=200,
            sampling_strategy=strategy,
            epochs_per_task=2,
            num_tasks=5,  # All 5 MNIST tasks
            update_features_freq=30 if strategy in ['diversity', 'combined'] else 0,
        )

        result = run_single_experiment(config)
        results[label] = result

        # Save individual result
        result_path = output_dir / f"poc/{strategy}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump({
                'config': config.to_dict(),
                'metrics': result['metrics'],
            }, f, indent=2)

    # Print comparison table
    print("\n" + "="*70)
    print("PROOF OF CONCEPT RESULTS")
    print("="*70)
    print(f"{'Strategy':<30} {'Avg Accuracy':>15} {'Avg Forgetting':>15}")
    print("-"*70)

    baseline_acc = results['Baseline']['metrics']['average_accuracy']
    for label, result in results.items():
        acc = result['metrics']['average_accuracy']
        fgt = result['metrics']['average_forgetting']
        delta = acc - baseline_acc
        delta_str = f"({delta:+.1f}%)" if label != 'Baseline' else ""
        print(f"{label:<30} {acc:>14.1f}% {fgt:>14.1f}%  {delta_str}")

    print("-"*70)

    # Key finding
    combined_acc = results['+PS+TS (Combined)']['metrics']['average_accuracy']
    improvement = combined_acc - baseline_acc
    print(f"\nKEY FINDING: Combined consolidation improves accuracy by {improvement:+.1f}%")

    if improvement > 0:
        print("✓ Hypothesis supported: Bio-inspired consolidation helps!")
    else:
        print("✗ Hypothesis not supported in this minimal test")

    # Save summary
    summary_path = output_dir / "poc/SUMMARY.md"
    with open(summary_path, 'w') as f:
        f.write("# Proof of Concept Results\n\n")
        f.write("## Hypothesis\n")
        f.write("Bio-inspired consolidation (Pattern Separation + Temporal Spacing) ")
        f.write("improves continual learning performance.\n\n")
        f.write("## Setup\n")
        f.write("- Dataset: Split-MNIST (5 tasks)\n")
        f.write("- Method: Vanilla Replay\n")
        f.write("- Buffer: 200 samples\n")
        f.write("- Epochs: 2 per task\n\n")
        f.write("## Results\n\n")
        f.write("| Strategy | Avg Accuracy | Avg Forgetting | vs Baseline |\n")
        f.write("|----------|--------------|----------------|-------------|\n")
        for label, result in results.items():
            acc = result['metrics']['average_accuracy']
            fgt = result['metrics']['average_forgetting']
            delta = acc - baseline_acc
            delta_str = f"{delta:+.1f}%" if label != 'Baseline' else "-"
            f.write(f"| {label} | {acc:.1f}% | {fgt:.1f}% | {delta_str} |\n")
        f.write(f"\n## Conclusion\n\n")
        f.write(f"Combined PS+TS improves accuracy by **{improvement:+.1f}%** over baseline.\n")

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
