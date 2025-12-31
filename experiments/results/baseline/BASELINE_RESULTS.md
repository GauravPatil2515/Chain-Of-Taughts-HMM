# Baseline Evaluation Results

## Overview

This document summarizes the baseline evaluation comparing the HMM-based CoT scorer against simple heuristic methods.

## Baseline Methods

### 1. **Length Baseline**
- **Approach**: Scores based on CoT length (character count and step count)
- **Rationale**: Longer reasoning chains may indicate more thorough thinking
- **Formula**: `score = log(char_count) * 0.1 + log(step_count) * 0.5`

### 2. **Tag Diversity Baseline**
- **Approach**: Scores based on variety of reasoning tags used
- **Rationale**: Diverse reasoning steps suggest comprehensive problem-solving
- **Formula**: `score = (unique_tags / total_tags) * log(total_tags)`

### 3. **Structure Baseline**
- **Approach**: Checks for presence of key structural elements
- **Features**: Has calculation, conclusion, reasoning words, numbers, specific tags (PR, CA, CO)
- **Rationale**: Well-structured reasoning follows expected patterns

### 4. **Transition Baseline**
- **Approach**: Evaluates quality of tag-to-tag transitions
- **Rationale**: Good reasoning follows logical flow patterns
- **Method**: Counts transitions from predefined "good transitions" set

### 5. **Combined Baseline**
- **Approach**: Weighted combination of all above baselines
- **Weights**: Length (0.2), Diversity (0.2), Structure (0.3), Transition (0.3)

### 6. **Random Baseline**
- **Approach**: Random score (0-1)
- **Purpose**: Sanity check - all methods should beat random

### 7. **HMM Baseline (Our Method)**
- **Approach**: Normalized log-likelihood from trained HMM
- **Rationale**: Learns transition patterns from correct examples

## Results on Demo Data (7 samples)

### Correlation with Correctness

| Method | Correlation | p-value | Mean (Correct) | Mean (Incorrect) |
|--------|-------------|---------|----------------|------------------|
| **Diversity** | **0.280** | 0.544 | 1.06 | 0.82 |
| Combined | -0.277 | 0.548 | 0.59 | 0.72 |
| Structure | -0.322 | 0.482 | 4.92 | 6.00 |
| Transition | -0.358 | 0.431 | 2.38 | 3.56 |
| **HMM** | **-0.363** | 0.424 | 2.87e-17 | 8.33e-17 |
| Length | -0.433 | 0.331 | 1.24 | 1.59 |
| Random | N/A | N/A | 0.37 | 0.37 |

### Classification Metrics (using median threshold)

| Method | Accuracy | AUC | Precision | Recall | F1 |
|--------|----------|-----|-----------|--------|-----|
| Random | **0.857** | 0.500 | 0.857 | 1.000 | **0.923** |
| **Diversity** | **0.714** | **0.667** | 1.000 | 0.667 | **0.800** |
| Structure | 0.571 | 0.167 | 0.800 | 0.667 | 0.727 |
| Length | 0.428 | 0.000 | 0.750 | 0.500 | 0.600 |
| Transition | 0.428 | 0.000 | 0.750 | 0.500 | 0.600 |
| Combined | 0.428 | 0.167 | 0.750 | 0.500 | 0.600 |
| **HMM** | **0.428** | 0.000 | 0.750 | 0.500 | 0.600 |

## Key Findings

### ✅ Successful Baseline Implementation

1. **All methods functional** - Successfully computed scores for 7 baseline methods
2. **Evaluation metrics** - Correlation, classification metrics, and visualizations generated
3. **Infrastructure ready** - Pipeline can scale to larger datasets

### ⚠️ Current Limitations (Due to Small Sample Size)

1. **Small dataset** - Only 7 samples (6 correct, 1 incorrect)
2. **Statistical significance** - High p-values (>0.3) indicate need for more data
3. **Random performs well** - Due to data imbalance and small size

### 🎯 Best Performers on Demo Data

**Correlation-based:**
- **Tag Diversity** shows positive correlation (0.280)
- HMM shows negative correlation (-0.363) - likely due to small sample/training data

**Classification-based:**
- **Tag Diversity** achieves 0.714 accuracy and 0.800 F1
- **Random baseline** achieves 0.857 accuracy (red flag indicating data limitations)

## Visualizations Generated

1. **Score Distributions** - Histograms comparing correct vs incorrect CoTs for each method
2. **Correlation Comparison** - Bar chart showing correlation coefficients
3. **Classification Metrics** - Grouped bar chart for accuracy, precision, recall, F1
4. **Method Correlation Heatmap** - Spearman correlation between different scoring methods
5. **Scatter Plots** - HMM vs Combined baseline scores

## Next Steps

### Immediate Priority: Scale to Real Data

1. **Generate more CoT data** (target: 200+ samples)
   - Run `generate_local.py` on GSM8K dataset
   - Generate from multiple models if possible

2. **Retrain HMM** with larger dataset
   - More training data will improve HMM performance
   - Better statistical significance

3. **Re-run baseline evaluation**
   - More meaningful comparisons with n>50
   - Lower p-values and confidence intervals

### Expected Improvements with Larger Dataset

- **HMM should outperform** simple heuristics with sufficient training data
- **Statistical significance** will emerge with n>50 samples
- **Clearer separation** between correct/incorrect reasoning patterns

## Usage

To reproduce or update these results:

```bash
# Run baseline evaluation
python run_baseline_eval.py

# Or with custom paths
python run_baseline_eval.py --data path/to/scored.json --model path/to/model.pkl --output path/to/results
```

## Files Generated

- `experiments/results/baseline/baseline_comparison.json` - Full results
- `experiments/results/baseline/figures/*.png` - Visualization plots

## Conclusion

The baseline evaluation framework is **fully functional and ready for scale-up**. Current results are limited by small sample size (n=7), but the infrastructure successfully:

- ✅ Implements 6 baseline methods + HMM
- ✅ Computes comprehensive metrics
- ✅ Generates publication-quality visualizations
- ✅ Provides interpretable comparisons

**Next critical step**: Generate real CoT data (200+ samples) to enable meaningful baseline comparisons.
