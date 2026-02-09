"""Visualization utilities for experiment results."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path

from .metrics import AccuracyMatrix


def plot_accuracy_matrix(
    accuracy_matrix: AccuracyMatrix,
    num_tasks: int,
    title: str = "Accuracy Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot accuracy matrix as heatmap.

    Rows: After training on task i
    Columns: Accuracy on task j
    """
    matrix = accuracy_matrix.to_numpy(num_tasks)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Mask upper triangle (not evaluated)
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

    sns.heatmap(
        matrix,
        mask=mask,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        vmin=0,
        vmax=100,
        ax=ax,
        cbar_kws={'label': 'Accuracy (%)'}
    )

    ax.set_xlabel('Evaluated on Task')
    ax.set_ylabel('After Training on Task')
    ax.set_xticklabels([f'T{i+1}' for i in range(num_tasks)])
    ax.set_yticklabels([f'T{i+1}' for i in range(num_tasks)])
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_forgetting_curves(
    accuracy_matrix: AccuracyMatrix,
    num_tasks: int,
    title: str = "Forgetting Curves",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot accuracy on each task over training.

    Shows how accuracy on each task changes as new tasks are learned.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, num_tasks))

    for task in range(1, num_tasks + 1):
        accuracies = []
        x_values = []

        for after_task in range(task, num_tasks + 1):
            acc = accuracy_matrix.get(after_task, task)
            if acc is not None:
                accuracies.append(acc)
                x_values.append(after_task)

        if accuracies:
            ax.plot(
                x_values, accuracies,
                marker='o', label=f'Task {task}',
                color=colors[task - 1],
                linewidth=2, markersize=8
            )

    ax.set_xlabel('After Training on Task', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(1, num_tasks + 1))
    ax.set_ylim([0, 105])
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_method_comparison(
    results: Dict[str, Dict],
    metric: str = 'average_accuracy',
    title: str = "Method Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Compare different CL methods on a metric.

    Args:
        results: Dictionary mapping method name to results dict
        metric: Which metric to compare
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(results.keys())
    values = [results[m]['metrics'][metric] for m in methods]

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    bars = ax.bar(methods, values, color=colors, edgecolor='black')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{value:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim([0, max(values) * 1.15])
    ax.grid(True, axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_consolidation_comparison(
    results: Dict[str, Dict[str, Dict]],
    base_methods: List[str],
    consolidation_variants: List[str],
    metric: str = 'average_accuracy',
    title: str = "Consolidation Effects by Method",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Compare consolidation mechanisms across base methods.

    Shows grouped bar chart with base methods on x-axis and
    consolidation variants as grouped bars.

    Args:
        results: Nested dict: results[base_method][variant] = result
        base_methods: List of base method names (e.g., ['vanilla', 'ewc', 'meta_sgd'])
        consolidation_variants: List of variant names (e.g., ['random', '+PS', '+TS', '+PS+TS'])
        metric: Metric to compare
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    n_methods = len(base_methods)
    n_variants = len(consolidation_variants)
    width = 0.8 / n_variants
    x = np.arange(n_methods)

    colors = plt.cm.Set2(np.linspace(0, 1, n_variants))

    for i, variant in enumerate(consolidation_variants):
        values = []
        for method in base_methods:
            if method in results and variant in results[method]:
                values.append(results[method][variant]['metrics'][metric])
            else:
                values.append(0)

        offset = (i - n_variants / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=variant, color=colors[i])

        # Add value labels
        for bar, value in zip(bars, values):
            if value > 0:
                height = bar.get_height()
                ax.annotate(
                    f'{value:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8
                )

    ax.set_xlabel('Base Method', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in base_methods])
    ax.legend(title='Consolidation')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_buffer_ablation(
    results: Dict[int, Dict],
    title: str = "Buffer Size Ablation",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot performance vs buffer size.

    Args:
        results: Dictionary mapping buffer_size to results dict
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    buffer_sizes = sorted(results.keys())
    accuracies = [results[s]['metrics']['average_accuracy'] for s in buffer_sizes]
    forgetting = [results[s]['metrics']['average_forgetting'] for s in buffer_sizes]

    # Accuracy vs buffer size
    ax1.plot(buffer_sizes, accuracies, 'bo-', linewidth=2, markersize=10)
    ax1.set_xlabel('Buffer Size', fontsize=12)
    ax1.set_ylabel('Average Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs Buffer Size', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Forgetting vs buffer size
    ax2.plot(buffer_sizes, forgetting, 'ro-', linewidth=2, markersize=10)
    ax2.set_xlabel('Buffer Size', fontsize=12)
    ax2.set_ylabel('Average Forgetting (%)', fontsize=12)
    ax2.set_title('Forgetting vs Buffer Size', fontsize=14)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_summary_table(
    results: Dict[str, Dict],
    metrics: List[str] = None
) -> str:
    """Create markdown table summarizing results.

    Args:
        results: Dictionary mapping method name to results dict
        metrics: List of metrics to include (default: all main metrics)

    Returns:
        Markdown-formatted table string
    """
    if metrics is None:
        metrics = ['average_accuracy', 'average_forgetting', 'learning_accuracy']

    # Header
    header = "| Method | " + " | ".join([m.replace('_', ' ').title() for m in metrics]) + " |"
    separator = "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"

    rows = [header, separator]

    for method, result in results.items():
        values = []
        for metric in metrics:
            value = result['metrics'].get(metric, 0)
            if isinstance(value, float):
                values.append(f"{value:.1f}%")
            else:
                values.append(str(value))

        row = f"| {method} | " + " | ".join(values) + " |"
        rows.append(row)

    return "\n".join(rows)
