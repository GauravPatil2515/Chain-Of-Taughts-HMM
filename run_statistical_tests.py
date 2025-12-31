"""
Phase 5: Statistical Significance Testing
Applies comprehensive statistical tests to validate baseline comparison results.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.statistical_tests import (
    cohens_d, bootstrap_ci, paired_t_test,
    mcnemar_test, correlation_with_ci
)


def load_data():
    """Load results and tagged data."""
    with open("experiments/results/phase1_baseline/baseline_comparison.json", "r") as f:
        results = json.load(f)
    
    with open("data/tagged/phase1_tagged.json", "r") as f:
        tagged_data = json.load(f)
    
    return results, tagged_data


def extract_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Convert scores to binary predictions."""
    return (scores >= threshold).astype(int)


def run_statistical_tests(results: dict, tagged_data: list) -> dict:
    """
    Run comprehensive statistical tests on baseline comparison.
    
    Tests performed:
    1. Cohen's d for effect sizes
    2. Bootstrap confidence intervals
    3. Paired t-tests for score differences
    4. McNemar's test for classification disagreements
    5. Correlation significance tests
    6. Bonferroni correction for multiple comparisons
    """
    
    all_scores = results['all_scores']
    
    # Extract scores and ground truth
    methods = ['length', 'diversity', 'structure', 'transition', 'combined', 'hmm']
    scores_dict = {}
    for method in methods:
        scores_dict[method] = np.array([item[f'{method}_score'] for item in all_scores])
    
    y_true = np.array([item['is_correct'] for item in all_scores])
    
    statistical_results = {
        'effect_sizes': {},
        'confidence_intervals': {},
        'paired_comparisons': {},
        'mcnemar_tests': {},
        'correlation_tests': {},
        'multiple_testing_correction': {}
    }
    
    # 1. Effect Sizes (Cohen's d) - Compare correct vs incorrect scores
    print("\n[1/6] Computing effect sizes (Cohen's d)...")
    for method in methods:
        scores = scores_dict[method]
        correct_scores = scores[y_true == 1]
        incorrect_scores = scores[y_true == 0]
        
        if len(correct_scores) > 0 and len(incorrect_scores) > 0:
            d = cohens_d(correct_scores, incorrect_scores)
            
            # Interpret effect size
            if abs(d) < 0.2:
                interpretation = "Negligible"
            elif abs(d) < 0.5:
                interpretation = "Small"
            elif abs(d) < 0.8:
                interpretation = "Medium"
            else:
                interpretation = "Large"
            
            statistical_results['effect_sizes'][method] = {
                'cohens_d': float(d),
                'interpretation': interpretation,
                'mean_correct': float(np.mean(correct_scores)),
                'mean_incorrect': float(np.mean(incorrect_scores)),
                'std_correct': float(np.std(correct_scores)),
                'std_incorrect': float(np.std(incorrect_scores))
            }
            print(f"   {method}: d = {d:.3f} ({interpretation})")
    
    # 2. Bootstrap Confidence Intervals
    print("\n[2/6] Computing bootstrap confidence intervals...")
    for method in methods:
        scores = scores_dict[method]
        mean_ci = bootstrap_ci(scores, statistic_fn=np.mean, n_iterations=1000, confidence_level=0.95)
        statistical_results['confidence_intervals'][method] = {
            'mean': float(np.mean(scores)),
            '95_ci_lower': float(mean_ci[0]),
            '95_ci_upper': float(mean_ci[1]),
            'ci_width': float(mean_ci[1] - mean_ci[0])
        }
        print(f"   {method}: Mean = {np.mean(scores):.4f}, 95% CI = [{mean_ci[0]:.4f}, {mean_ci[1]:.4f}]")
    
    # 3. Paired t-tests - Compare each method against HMM
    print("\n[3/6] Performing paired t-tests (vs HMM)...")
    hmm_scores = scores_dict['hmm']
    for method in ['length', 'diversity', 'structure', 'transition', 'combined']:
        method_scores = scores_dict[method]
        result = paired_t_test(method_scores, hmm_scores, f'{method}', 'hmm')
        
        statistical_results['paired_comparisons'][f'{method}_vs_hmm'] = {
            't_statistic': float(result.statistic),
            'p_value': float(result.p_value),
            'mean_difference': float(np.mean(method_scores - hmm_scores)),
            'effect_size': float(result.effect_size),
            'interpretation': result.interpretation,
            'significant': bool(result.significant)
        }
        print(f"   {method} vs HMM: t = {result.statistic:.3f}, p = {result.p_value:.4f} ({result.interpretation})")
    
    # 4. McNemar's Test - Compare classification decisions
    print("\n[4/6] Performing McNemar's tests...")
    
    # Get optimal thresholds from classification metrics
    classification_metrics = results['classification_metrics']
    
    for method in ['diversity', 'combined']:
        # Find threshold for this method
        method_metric = next((m for m in classification_metrics if m['method'] == method), None)
        if method_metric:
            method_threshold = method_metric['threshold']
            hmm_threshold = next((m for m in classification_metrics if m['method'] == 'hmm'), {}).get('threshold', 0)
            
            method_pred = extract_predictions(scores_dict[method], method_threshold)
            hmm_pred = extract_predictions(scores_dict['hmm'], hmm_threshold)
            
            result = mcnemar_test(method_pred, hmm_pred, y_true)
            
            statistical_results['mcnemar_tests'][f'{method}_vs_hmm'] = {
                'chi2_statistic': float(result.statistic),
                'p_value': float(result.p_value),
                'interpretation': result.interpretation,
                'significant': bool(result.significant)
            }
            print(f"   {method} vs HMM: χ² = {result.statistic:.3f}, p = {result.p_value:.4f} ({result.interpretation})")
    
    # 5. Correlation significance with confidence intervals
    print("\n[5/6] Computing correlation confidence intervals...")
    for method in methods:
        scores = scores_dict[method]
        result = correlation_with_ci(scores, y_true)
        
        statistical_results['correlation_tests'][method] = {
            'correlation': float(result.statistic),
            '95_ci_lower': float(result.confidence_interval[0]),
            '95_ci_upper': float(result.confidence_interval[1]),
            'p_value': float(result.p_value),
            'significant': bool(result.significant)
        }
        print(f"   {method}: r = {result.statistic:.3f}, 95% CI = [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}], p = {result.p_value:.4f}")
    
    # 6. Multiple testing correction (Bonferroni)
    print("\n[6/6] Applying Bonferroni correction for multiple comparisons...")
    
    # Collect all p-values
    p_values = []
    test_names = []
    
    # From paired t-tests
    for test_name, test_result in statistical_results['paired_comparisons'].items():
        p_values.append(test_result['p_value'])
        test_names.append(f"paired_t_{test_name}")
    
    # From McNemar tests
    for test_name, test_result in statistical_results['mcnemar_tests'].items():
        p_values.append(test_result['p_value'])
        test_names.append(f"mcnemar_{test_name}")
    
    # From correlation tests
    for test_name, test_result in statistical_results['correlation_tests'].items():
        p_values.append(test_result['p_value'])
        test_names.append(f"correlation_{test_name}")
    
    # Apply correction - simple Bonferroni
    n_tests = len(p_values)
    alpha_original = 0.05
    alpha_bonf = alpha_original / n_tests
    corrected_p = [min(p * n_tests, 1.0) for p in p_values]
    reject = [p < alpha_bonf for p in p_values]
    
    statistical_results['multiple_testing_correction'] = {
        'original_alpha': alpha_original,
        'bonferroni_alpha': float(alpha_bonf),
        'n_tests': n_tests,
        'tests': []
    }
    
    for i, (name, orig_p, corr_p, is_reject) in enumerate(zip(test_names, p_values, corrected_p, reject)):
        statistical_results['multiple_testing_correction']['tests'].append({
            'test_name': name,
            'original_p': float(orig_p),
            'corrected_p': float(corr_p),
            'reject_null': bool(is_reject),
            'significant_after_correction': bool(is_reject)
        })
        status = "✓ Significant" if is_reject else "✗ Not significant"
        print(f"   {name}: p = {orig_p:.4f} → {corr_p:.4f} ({status})")
    
    return statistical_results


def generate_statistical_report(stats: dict) -> str:
    """Generate markdown report of statistical tests."""
    
    report = ["# Phase 5: Statistical Significance Testing Report\n\n"]
    report.append("**Date:** 2025-12-27\n\n")
    report.append("---\n\n")
    
    # Effect Sizes
    report.append("## 1. Effect Sizes (Cohen's d)\n\n")
    report.append("Measures the magnitude of difference between correct and incorrect scores.\n\n")
    report.append("| Method | Cohen's d | Interpretation | Mean Correct | Mean Incorrect |\n")
    report.append("|--------|-----------|----------------|--------------|----------------|\n")
    for method, data in stats['effect_sizes'].items():
        report.append(f"| {method} | {data['cohens_d']:.3f} | {data['interpretation']} | {data['mean_correct']:.4f} | {data['mean_incorrect']:.4f} |\n")
    
    report.append("\n**Interpretation Guide:**\n")
    report.append("- Small: |d| < 0.5\n")
    report.append("- Medium: 0.5 ≤ |d| < 0.8\n")
    report.append("- Large: |d| ≥ 0.8\n\n")
    
    # Confidence Intervals
    report.append("## 2. Bootstrap Confidence Intervals (95%)\n\n")
    report.append("| Method | Mean Score | 95% CI Lower | 95% CI Upper | CI Width |\n")
    report.append("|--------|------------|--------------|--------------|----------|\n")
    for method, data in stats['confidence_intervals'].items():
        report.append(f"| {method} | {data['mean']:.4f} | {data['95_ci_lower']:.4f} | {data['95_ci_upper']:.4f} | {data['ci_width']:.4f} |\n")
    report.append("\n")
    
    # Paired Comparisons
    report.append("## 3. Paired t-Tests (vs HMM)\n\n")
    report.append("Tests whether each baseline method produces significantly different scores than HMM.\n\n")
    report.append("| Comparison | t-statistic | p-value | Mean Difference | Significant? | Interpretation |\n")
    report.append("|------------|-------------|---------|-----------------|--------------|----------------|\n")
    for test_name, data in stats['paired_comparisons'].items():
        sig = "✓ Yes" if data['significant'] else "✗ No"
        report.append(f"| {test_name} | {data['t_statistic']:.3f} | {data['p_value']:.4f} | {data['mean_difference']:.4f} | {sig} | {data['interpretation']} |\n")
    report.append("\n")
    
    # McNemar Tests
    report.append("## 4. McNemar's Tests\n\n")
    report.append("Tests whether classification decisions differ significantly between methods.\n\n")
    report.append("| Comparison | χ² | p-value | Significant? | Interpretation |\n")
    report.append("|------------|-----|---------|--------------|----------------|\n")
    for test_name, data in stats['mcnemar_tests'].items():
        sig = "✓ Yes" if data['significant'] else "✗ No"
        report.append(f"| {test_name} | {data['chi2_statistic']:.3f} | {data['p_value']:.4f} | {sig} | {data['interpretation']} |\n")
    report.append("\n")
    
    # Correlation Tests
    report.append("## 5. Correlation Tests with Confidence Intervals\n\n")
    report.append("| Method | Correlation | 95% CI Lower | 95% CI Upper | p-value | Significant? |\n")
    report.append("|--------|-------------|--------------|--------------|---------|--------------||\n")
    for method, data in stats['correlation_tests'].items():
        sig = "✓ Yes" if data['significant'] else "✗ No"
        report.append(f"| {method} | {data['correlation']:.3f} | {data['95_ci_lower']:.3f} | {data['95_ci_upper']:.3f} | {data['p_value']:.4f} | {sig} |\n")
    report.append("\n")
    
    # Multiple Testing Correction
    report.append("## 6. Multiple Testing Correction (Bonferroni)\n\n")
    report.append(f"**Total tests:** {stats['multiple_testing_correction']['n_tests']}  \n")
    report.append(f"**Original α:** {stats['multiple_testing_correction']['original_alpha']}  \n")
    report.append(f"**Bonferroni α:** {stats['multiple_testing_correction']['bonferroni_alpha']:.6f}  \n\n")
    
    report.append("| Test | Original p | Corrected p | Significant? |\n")
    report.append("|------|------------|-------------|--------------||\n")
    for test in stats['multiple_testing_correction']['tests']:
        sig = "✓ Yes" if test['significant_after_correction'] else "✗ No"
        report.append(f"| {test['test_name']} | {test['original_p']:.4f} | {test['corrected_p']:.4f} | {sig} |\n")
    
    report.append("\n---\n\n")
    report.append("## Summary\n\n")
    
    # Count significant results
    sig_effects = sum(1 for v in stats['effect_sizes'].values() if 'Large' in v['interpretation'])
    sig_paired = sum(1 for v in stats['paired_comparisons'].values() if v['significant'])
    sig_mcnemar = sum(1 for v in stats['mcnemar_tests'].values() if v['significant'])
    sig_corr = sum(1 for v in stats['correlation_tests'].values() if v['significant'])
    sig_after_bonf = sum(1 for t in stats['multiple_testing_correction']['tests'] if t['significant_after_correction'])
    
    report.append(f"- **Large effect sizes:** {sig_effects}/{len(stats['effect_sizes'])}\n")
    report.append(f"- **Significant paired t-tests:** {sig_paired}/{len(stats['paired_comparisons'])}\n")
    report.append(f"- **Significant McNemar tests:** {sig_mcnemar}/{len(stats['mcnemar_tests'])}\n")
    report.append(f"- **Significant correlations:** {sig_corr}/{len(stats['correlation_tests'])}\n")
    report.append(f"- **Significant after Bonferroni:** {sig_after_bonf}/{len(stats['multiple_testing_correction']['tests'])}\n")
    
    return "".join(report)


def main():
    print("="*70)
    print("Phase 5: Statistical Significance Testing")
    print("="*70)
    
    # Load data
    print("\n[Step 1] Loading data...")
    results, tagged_data = load_data()
    print(f"   Loaded {len(tagged_data)} samples")
    
    # Run tests
    print("\n[Step 2] Running statistical tests...")
    stats = run_statistical_tests(results, tagged_data)
    
    # Save results
    print("\n[Step 3] Saving results...")
    output_dir = Path("experiments/results/phase5_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "statistical_tests.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"   Saved to: {output_dir / 'statistical_tests.json'}")
    
    # Generate report
    print("\n[Step 4] Generating report...")
    report = generate_statistical_report(stats)
    
    with open(output_dir / "STATISTICAL_TESTS_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"   Saved to: {output_dir / 'STATISTICAL_TESTS_REPORT.md'}")
    
    print("\n" + "="*70)
    print("Statistical Testing Complete!")
    print("="*70)
    print(f"\n📁 Output: {output_dir}")
    print(f"   - statistical_tests.json")
    print(f"   - STATISTICAL_TESTS_REPORT.md")


if __name__ == "__main__":
    main()
