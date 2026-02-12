# Proof of Concept Results

## Hypothesis
Comparing different replay strategies and methods for continual learning:
- Does replay help? (Baseline vs Replay methods)
- Does PS+TS improve over random replay?
- Does TTT (Test-Time Training) help?
- Does combining TTT with PS+TS give the best results?

## Setup
- Dataset: Split-MNIST (5 tasks)
- Buffer: 200 samples
- Epochs: 2 per task

## Results

| Strategy | Avg Accuracy | Avg Forgetting | vs Baseline |
|----------|--------------|----------------|-------------|
| Baseline (no replay) | 19.9% | 80.0% | - |
| Vanilla + Random Replay | 83.9% | 15.9% | +63.9% |
| Vanilla + PS+TS Replay | 86.0% | 13.8% | +66.0% |
| TTT + Random Replay | 88.0% | 11.8% | +68.1% |
| TTT + PS+TS Replay | 89.2% | 10.6% | +69.3% |

## Key Findings

- **Vanilla Random vs Baseline:** +63.9%
- **Vanilla PS+TS vs Baseline:** +66.0%
- **TTT Random vs Baseline:** +68.1%
- **TTT PS+TS vs Baseline:** +69.3%
- **TTT PS+TS vs TTT Random:** +1.3%

## Conclusion

**Best Method:** TTT + PS+TS Replay with 89.2% accuracy
