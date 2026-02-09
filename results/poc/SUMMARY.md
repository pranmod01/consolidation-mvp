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
| +TS (Temporal Spacing) | 86.4% | 13.4% | +2.5% |
| +PS+TS (Combined) | 84.8% | 14.9% | +1.0% |

## Conclusion

Combined PS+TS improves accuracy by **+1.0%** over baseline.
