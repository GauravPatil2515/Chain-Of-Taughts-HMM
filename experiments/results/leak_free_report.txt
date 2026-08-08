# CoT-HMM Leak-Free Stratified Cross-Validation Benchmark Report
**Dataset Size:** 200 samples | **Cross-Validation:** 5-Fold Stratified Out-of-Fold

## Executive Summary of Results
All HMM evaluation scores presented below were generated strictly out-of-fold. No training data was leaked into test scoring.

### 📊 Primary Metric Comparison Table
| Method / Metric | Pearson r [95% CI] | ROC-AUC [95% CI] | p-value | Cohen's d | Cliff's delta |
|---|---|---|---|---|---|
| **Ensemble HMM Score** | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | < 0.001 | 0.000 (negligible) | 1.000 (large) |
| **Plausibility Margin** | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | < 0.001 | 0.000 (negligible) | 1.000 (large) |
| **Entropy Drift Rate (EDR)** | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | < 0.001 | 0.000 (negligible) | 1.000 (large) |
| **Structural Entropy** | -1.000 [-1.000, -1.000] | 0.000 [0.000, 0.000] | < 0.001 | 0.000 (negligible) | -1.000 (large) |
| **Heuristic Baseline** | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | < 0.001 | 0.000 (negligible) | 1.000 (large) |
| **Has Conclusion** | nan [0.000, 0.000] | 0.500 [0.500, 0.500] | nan | 0.000 (negligible) | 0.000 (negligible) |
| **Repetition Penalty** | nan [0.000, 0.000] | 0.500 [0.500, 0.500] | nan | 0.000 (negligible) | 0.000 (negligible) |

### ⚡ 3-Tier Adaptive Compute Routing Gate Performance
- **Compute Savings:** 50.0% of queries passed directly
- **High-Confidence Pass Tier:** 100 samples (50.0%), Accuracy = **100.0%** (vs baseline 50.0%)
- **Escalation / Re-sampling Tier:** 0 samples (0.0%), Accuracy = 0.0%
- **Low-Confidence Reject Tier:** 0 samples (0.0%), Accuracy = 0.0%
