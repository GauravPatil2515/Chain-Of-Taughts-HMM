# CoT-HMM Leak-Free Stratified Cross-Validation Benchmark Report
**Dataset Size:** 500 samples | **Cross-Validation:** 5-Fold Stratified Out-of-Fold

## Executive Summary of Results
All HMM evaluation scores presented below were generated strictly out-of-fold. No training data was leaked into test scoring.

### 📊 Primary Metric Comparison Table
| Method / Metric | Pearson r [95% CI] | ROC-AUC [95% CI] | p-value | Cohen's d | Cliff's delta |
|---|---|---|---|---|---|
| **Ensemble HMM Score** | 0.350 [0.193, 0.534] | 0.950 [0.928, 0.969] | < 0.001 | 0.747 (medium) | 0.899 (large) |
| **Plausibility Margin** | 0.476 [0.375, 0.716] | 0.983 [0.966, 0.997] | < 0.001 | 1.093 (large) | 0.967 (large) |
| **Entropy Drift Rate (EDR)** | -0.282 [-0.376, -0.190] | 0.273 [0.227, 0.322] | < 0.001 | -0.586 (medium) | -0.454 (medium) |
| **Structural Entropy** | -0.732 [-0.763, -0.698] | 0.057 [0.040, 0.077] | < 0.001 | -2.144 (large) | -0.885 (large) |
| **Heuristic Baseline** | 0.459 [0.372, 0.555] | 0.860 [0.826, 0.891] | < 0.001 | 1.031 (large) | 0.719 (large) |
| **Has Conclusion** | 0.029 [-0.059, 0.112] | 0.504 [0.492, 0.517] | 0.5239 | 0.057 (negligible) | 0.008 (negligible) |
| **Repetition Penalty** | 0.634 [0.580, 0.687] | 0.843 [0.809, 0.876] | < 0.001 | 1.638 (large) | 0.687 (large) |

### ⚡ 3-Tier Adaptive Compute Routing Gate Performance
- **Compute Savings:** 30.0% of queries passed directly
- **High-Confidence Pass Tier:** 150 samples (30.0%), Accuracy = **99.3%** (vs baseline 50.0%)
- **Escalation / Re-sampling Tier:** 175 samples (35.0%), Accuracy = 50.9%
- **Low-Confidence Reject Tier:** 175 samples (35.0%), Accuracy = 6.9%
