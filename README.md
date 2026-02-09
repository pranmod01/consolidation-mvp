# Bio-Inspired Consolidation for Continual Learning MVP

This project tests whether bio-inspired consolidation mechanisms (Pattern Separation + Temporal Spacing) improve continual learning performance across different base methods.

## Hypothesis

Adding biologically-inspired consolidation mechanisms to experience replay will improve performance by:
1. **Pattern Separation**: Maximizing diversity in replay samples reduces interference
2. **Temporal Spacing**: Age-weighted replay implements spaced repetition principles

## Project Structure

```
consolidation-mvp/
├── src/
│   ├── models/          # Neural network architectures (CNN, ResNet)
│   ├── data/            # Datasets (Split-MNIST, Split-CIFAR10) and replay buffer
│   ├── training/        # Trainers for each CL method
│   ├── consolidation/   # Pattern Separation and Temporal Spacing implementations
│   ├── evaluation/      # Metrics and visualization
│   └── utils/           # Configuration and utilities
├── configs/             # YAML configuration files
├── experiments/         # Phase-specific experiment scripts
├── results/             # Output directory for results
├── run_experiments.py   # Main experiment runner
├── test_setup.py        # Setup verification script
└── requirements.txt     # Python dependencies
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Verify setup
python test_setup.py

# Run quick test (3 tasks, 2 epochs)
python run_experiments.py quick

# Run full Phase 2 experiment
python run_experiments.py phase2
```

## Experimental Design

### Phase 1: Single Mechanism Tests
Test Pattern Separation (PS) and Temporal Spacing (TS) separately on each base method with different buffer sizes.

```bash
python run_experiments.py phase1
```

For each base method (Vanilla Replay, EWC):
- Baseline: random replay sampling
- +PS: diversity-based sampling (farthest-point)
- +TS: age-weighted sampling

Buffer sizes: [200, 500, 1000]

### Phase 2: Combined Mechanisms
Test all combinations on all three base methods.

```bash
python run_experiments.py phase2
```

Conditions:
- Baseline (random uniform)
- +PS only
- +TS only
- +PS+TS (combined consolidation)

Dataset: Split-CIFAR10

### Phase 3: Validation at Scale
Test best-performing combination on harder problems.

```bash
python run_experiments.py phase3
```

Validates on Split-CIFAR10 with larger buffer.

## Base Methods

### 1. Vanilla Replay
Standard experience replay with random sampling. The simplest baseline to test "does consolidation help at all?"

### 2. EWC (Elastic Weight Consolidation)
Regularization-based approach that penalizes changes to important parameters. Tests whether consolidation helps memory-based vs regularization-based methods differently.

### 3. Meta-SGD
Meta-learning approach with learned per-parameter learning rates. Tests whether consolidation helps fast adaptation methods.

## Consolidation Mechanisms

### Pattern Separation
Maximizes feature-space diversity when sampling from replay buffer.

Implementation: Farthest-point sampling
- Start with random sample
- Iteratively add sample that is farthest from current set
- Creates maximum coverage of feature space

```python
# Usage
buffer.sample(batch_size, strategy='diversity', temperature=1.0)
```

### Temporal Spacing
Prioritizes older/forgotten samples for replay.

Implementation: Age-weighted sampling
- Older samples have higher selection probability
- Implements spaced repetition principle

```python
# Usage
buffer.sample(batch_size, strategy='temporal', age_weight=1.0)
```

### Combined
Combines both mechanisms by splitting batch between strategies.

```python
# Usage
buffer.sample(batch_size, strategy='combined',
              diversity_weight=0.5, temporal_weight=0.5)
```

## Metrics

- **Average Accuracy**: Mean accuracy across all tasks after training
- **Forgetting**: Max accuracy - final accuracy per task
- **Forward Transfer**: Does learning previous tasks help new ones?
- **Backward Transfer**: Does learning new tasks hurt old ones?
- **Memory Efficiency**: Performance gain per buffer sample

## Configuration

Experiments can be configured via YAML files:

```yaml
# configs/my_experiment.yaml
name: my_experiment
method: vanilla_replay
dataset: split_mnist
buffer_size: 500
sampling_strategy: combined
update_features_freq: 50
```

Or programmatically:

```python
from src.utils import ExperimentConfig

config = ExperimentConfig(
    method='ewc',
    sampling_strategy='diversity',
    buffer_size=1000,
)
```

## Expected Results

Based on cognitive science literature, we expect:
- Combined PS+TS to outperform either mechanism alone
- Improvements to be larger for smaller buffer sizes (more efficient memory use)
- Different base methods to benefit differently from consolidation

## References

- Pattern Separation: Inspired by hippocampal pattern separation (Yassa & Stark, 2011)
- Temporal Spacing: Based on spaced repetition research (Cepeda et al., 2006)
- EWC: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", 2017
- Meta-SGD: Li et al., "Meta-SGD: Learning to Learn Quickly", 2017
