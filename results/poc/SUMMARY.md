# Proof of Concept Results

## Hypothesis
Comparing different replay strategies and methods for continual learning:
- Does replay help? (Baseline vs Replay methods)
- Does PS+TS improve over random replay?
- Does Autoencoder (reconstruction loss) help?
- Does combining Autoencoder with PS+TS give the best results?

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
| AE + Random Replay | 80.6% | 19.0% | +60.7% |
| AE + PS+TS Replay | 81.6% | 18.0% | +61.7% |

## Key Findings

- **Vanilla Random vs Baseline:** +63.9%
- **Vanilla PS+TS vs Baseline:** +66.0%
- **AE Random vs Baseline:** +60.7%
- **AE PS+TS vs Baseline:** +61.7%
- **AE PS+TS vs AE Random:** +1.0%

## Conclusion

**Best Method:** Vanilla + PS+TS Replay with 86.0% accuracy
