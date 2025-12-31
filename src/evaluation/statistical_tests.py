"""
Statistical Testing Module
Advanced statistical tests for baseline comparison.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    significant: bool
    interpretation: str
    
    def to_dict(self) -> Dict:
        return {
            "test": self.test_name,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "effect_size": float(self.effect_size),
            "ci_lower": float(self.confidence_interval[0]),
            "ci_upper": float(self.confidence_interval[1]),
            "significant": self.significant,
            "interpretation": self.interpretation
        }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Interpretation:
        0.2 = small effect
        0.5 = medium effect
        0.8 = large effect
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def bootstrap_ci(data: np.ndarray, statistic_fn=np.mean, n_iterations: int = 1000, 
                 confidence_level: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """
    Calculate confidence interval using bootstrap resampling.
    
    Args:
        data: Input data array
        statistic_fn: Function to compute statistic (default: mean)
        n_iterations: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95)
        seed: Random seed
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    np.random.seed(seed)
    
    bootstrap_stats = []
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_fn(sample))
    
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return (lower, upper)


def paired_t_test(scores1: np.ndarray, scores2: np.ndarray, 
                  method1_name: str = "Method 1", method2_name: str = "Method 2") -> StatisticalTestResult:
    """
    Perform paired t-test with effect size.
    
    Use when comparing two methods on the same samples.
    """
    # Remove NaN/Inf
    valid_mask = np.isfinite(scores1) & np.isfinite(scores2)
    scores1 = scores1[valid_mask]
    scores2 = scores2[valid_mask]
    
    if len(scores1) < 2:
        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            significant=False,
            interpretation="Insufficient data"
        )
    
    # Perform test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    # Effect size
    d = cohens_d(scores1, scores2)
    
    # Confidence interval for difference
    diff = scores1 - scores2
    ci = bootstrap_ci(diff, n_iterations=1000)
    
    # Interpretation
    if abs(d) < 0.2:
        effect_interp = "negligible"
    elif abs(d) < 0.5:
        effect_interp = "small"
    elif abs(d) < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    
    winner = method1_name if np.mean(scores1) > np.mean(scores2) else method2_name
    
    interpretation = f"{winner} performs better with {effect_interp} effect (d={d:.3f})"
    if p_value < 0.05:
        interpretation += " [SIGNIFICANT]"
    else:
        interpretation += " [NOT SIGNIFICANT]"
    
    return StatisticalTestResult(
        test_name="Paired t-test",
        statistic=t_stat,
        p_value=p_value,
        effect_size=d,
        confidence_interval=ci,
        significant=p_value < 0.05,
        interpretation=interpretation
    )


def mcnemar_test(predictions1: np.ndarray, predictions2: np.ndarray, 
                 ground_truth: np.ndarray) -> StatisticalTestResult:
    """
    McNemar's test for paired binary classification.
    
    Tests if two classifiers have significantly different error rates.
    """
    # Create contingency table
    correct1 = (predictions1 == ground_truth)
    correct2 = (predictions2 == ground_truth)
    
    # Both correct: n11, Both wrong: n00
    # Method1 correct, Method2 wrong: n10
    # Method1 wrong, Method2 correct: n01
    n10 = np.sum(correct1 & ~correct2)
    n01 = np.sum(~correct1 & correct2)
    
    if n10 + n01 == 0:
        return StatisticalTestResult(
            test_name="McNemar's test",
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            significant=False,
            interpretation="No disagreements between methods"
        )
    
    # McNemar's test statistic
    chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    # Effect size (odds ratio)
    effect = n10 / n01 if n01 > 0 else float('inf')
    
    interpretation = f"Method1 correct when Method2 wrong: {n10}, " \
                    f"Method2 correct when Method1 wrong: {n01}"
    if p_value < 0.05:
        interpretation += " [SIGNIFICANT DIFFERENCE]"
    
    return StatisticalTestResult(
        test_name="McNemar's test",
        statistic=chi2,
        p_value=p_value,
        effect_size=np.log(effect) if effect != float('inf') else 0.0,
        confidence_interval=(n10, n01),
        significant=p_value < 0.05,
        interpretation=interpretation
    )


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Significance level (default: 0.05)
        
    Returns:
        List of booleans indicating significance after correction
    """
    corrected_alpha = alpha / len(p_values)
    return [p < corrected_alpha for p in p_values]


def correlation_with_ci(x: np.ndarray, y: np.ndarray, 
                        method: str = "spearman") -> StatisticalTestResult:
    """
    Compute correlation with confidence interval.
    
    Args:
        x, y: Data arrays
        method: "pearson", "spearman", or "pointbiserial"
    """
    # Remove NaN/Inf
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    if len(x) < 3:
        return StatisticalTestResult(
            test_name=f"{method.capitalize()} correlation",
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            significant=False,
            interpretation="Insufficient data"
        )
    
    # Compute correlation
    if method == "pearson":
        corr, p_value = stats.pearsonr(x, y)
    elif method == "spearman":
        corr, p_value = stats.spearmanr(x, y)
    elif method == "pointbiserial":
        corr, p_value = stats.pointbiserialr(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Bootstrap confidence interval
    def correlation_fn(indices):
        return stats.spearmanr(x[indices], y[indices])[0] if method == "spearman" else stats.pearsonr(x[indices], y[indices])[0]
    
    indices = np.arange(len(x))
    bootstrap_corrs = []
    for _ in range(1000):
        sample_idx = np.random.choice(indices, size=len(indices), replace=True)
        try:
            if method == "spearman":
                boot_corr = stats.spearmanr(x[sample_idx], y[sample_idx])[0]
            else:
                boot_corr = stats.pearsonr(x[sample_idx], y[sample_idx])[0]
            bootstrap_corrs.append(boot_corr)
        except:
            pass
    
    ci = (np.percentile(bootstrap_corrs, 2.5), np.percentile(bootstrap_corrs, 97.5))
    
    # Interpretation
    if abs(corr) < 0.1:
        strength = "negligible"
    elif abs(corr) < 0.3:
        strength = "weak"
    elif abs(corr) < 0.5:
        strength = "moderate"
    elif abs(corr) < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if corr > 0 else "negative"
    
    interpretation = f"{strength.capitalize()} {direction} correlation (r={corr:.3f})"
    if p_value < 0.05:
        interpretation += " [SIGNIFICANT]"
    
    return StatisticalTestResult(
        test_name=f"{method.capitalize()} correlation",
        statistic=corr,
        p_value=p_value,
        effect_size=corr,  # Correlation itself is the effect size
        confidence_interval=ci,
        significant=p_value < 0.05,
        interpretation=interpretation
    )


def compare_all_methods(df: pd.DataFrame, score_columns: List[str], 
                        target_col: str = "is_correct") -> pd.DataFrame:
    """
    Comprehensive comparison of all methods with statistical tests.
    
    Returns DataFrame with test results for all pairwise comparisons.
    """
    results = []
    
    # Pairwise comparisons
    for i, col1 in enumerate(score_columns):
        for col2 in score_columns[i+1:]:
            # Filter valid data
            valid_mask = np.isfinite(df[col1]) & np.isfinite(df[col2])
            
            if valid_mask.sum() < 3:
                continue
            
            # Paired t-test
            test_result = paired_t_test(
                df[valid_mask][col1].values,
                df[valid_mask][col2].values,
                method1_name=col1,
                method2_name=col2
            )
            
            results.append({
                "comparison": f"{col1} vs {col2}",
                **test_result.to_dict()
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate data
    method1_scores = np.random.normal(0.7, 0.15, 100)
    method2_scores = np.random.normal(0.65, 0.15, 100)
    
    # Test
    result = paired_t_test(method1_scores, method2_scores, "Method 1", "Method 2")
    print("Paired t-test result:")
    print(f"  Statistic: {result.statistic:.4f}")
    print(f"  p-value: {result.p_value:.4f}")
    print(f"  Effect size (Cohen's d): {result.effect_size:.4f}")
    print(f"  95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
    print(f"  Significant: {result.significant}")
    print(f"  Interpretation: {result.interpretation}")
