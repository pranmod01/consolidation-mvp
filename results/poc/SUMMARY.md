# Proof of Concept Results

## Hypothesis
Bio-inspired consolidation (Pattern Separation + Temporal Spacing) improves continual learning performance.

## Setup
- Dataset: Split-MNIST (5 tasks)
- Method: Vanilla Replay
- Buffer: 200 samples
- Epochs: 2 per task

## Results

| Strategy | Avg Accuracy | Avg Forgetting | vs Baseline |
|----------|--------------|----------------|-------------|
| Baseline | 83.9% | 15.9% | - |
| +PS (Pattern Separation) | 88.2% | 11.7% | +4.4% |
| +TS (Temporal Spacing) | 85.3% | 14.6% | +1.4% |
| +PS+TS (Combined) | 85.2% | 14.6% | +1.4% |

## Conclusion

Combined PS+TS improves accuracy by **+1.4%** over baseline.
