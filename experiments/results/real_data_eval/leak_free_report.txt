# CoT-HMM Leak-Free Stratified Cross-Validation Benchmark Report
**Dataset Size:** 300 samples | **Cross-Validation:** 5-Fold Stratified Out-of-Fold

## Executive Summary of Results
All HMM evaluation scores presented below were generated strictly out-of-fold. No training data was leaked into test scoring.

### 📊 Primary Metric Comparison Table
| Method / Metric | Pearson r [95% CI] | ROC-AUC [95% CI] | p-value | Cohen's d | Cliff's delta |
|---|---|---|---|---|---|
| **Ensemble HMM Score** | 0.002 [-0.127, 0.077] | 0.498 [0.406, 0.594] | 0.9661 | 0.007 (negligible) | -0.005 (negligible) |
| **Plausibility Margin** | -0.007 [-0.125, 0.104] | 0.500 [0.427, 0.568] | 0.9052 | -0.020 (negligible) | -0.000 (negligible) |
| **Entropy Drift Rate (EDR)** | -0.047 [-0.160, 0.069] | 0.458 [0.361, 0.558] | 0.4219 | -0.138 (negligible) | -0.084 (negligible) |
| **Structural Entropy** | -0.012 [-0.121, 0.106] | 0.486 [0.392, 0.581] | 0.8303 | -0.037 (negligible) | -0.028 (negligible) |
| **Heuristic Baseline** | 0.086 [-0.033, 0.190] | 0.587 [0.495, 0.675] | 0.1361 | 0.257 (small) | 0.174 (small) |
| **Has Conclusion** | 0.048 [-0.059, 0.132] | 0.523 [0.472, 0.564] | 0.4089 | 0.142 (negligible) | 0.046 (negligible) |
| **Repetition Penalty** | 0.038 [-0.084, 0.154] | 0.558 [0.467, 0.645] | 0.5106 | 0.113 (negligible) | 0.116 (negligible) |

### ⚡ 3-Tier Adaptive Compute Routing Gate Performance
- **Compute Savings:** 30.0% of queries passed directly
- **High-Confidence Pass Tier:** 90 samples (30.0%), Accuracy = **12.2%** (vs baseline 13.0%)
- **Escalation / Re-sampling Tier:** 105 samples (35.0%), Accuracy = 13.3%
- **Low-Confidence Reject Tier:** 105 samples (35.0%), Accuracy = 13.3%
