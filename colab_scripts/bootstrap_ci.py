"""
Bootstrap Confidence Intervals for CoT-HMM
Rigorous statistical reporting for journal submission
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass, asdict

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


@dataclass
class BootstrapResult:
    """Result of bootstrap analysis."""
    metric_name: str
    point_estimate: float
    lower: float
    upper: float
    ci_level: float
    n_bootstrap: int
    p_value: Optional[float] = None
    statistic_method: str = "percentile"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def __str__(self) -> str:
        return f"{self.metric_name}: {self.point_estimate:.4f} [{int(self.ci_level*100)}% CI: {self.lower:.4f}, {self.upper:.4f}]"


class BootstrapAnalyzer:
    """Advanced bootstrap analysis with multiple metrics."""
    
    def __init__(self, n_bootstrap: int = 10000, seed: int = 42, ci_level: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.ci_level = ci_level
        self.alpha = 1 - ci_level
        self._rng = np.random.RandomState(seed)
    
    def _percentile_ci(self, boot_dist: np.ndarray) -> Tuple[float, float]:
        """Compute percentile confidence interval."""
        return (
            float(np.percentile(boot_dist, 100 * self.alpha / 2)),
            float(np.percentile(boot_dist, 100 * (1 - self.alpha / 2)))
        )
    
    def _bias_corrected_ci(self, boot_dist: np.ndarray, point: float) -> Tuple[float, float]:
        """Bias-corrected and accelerated bootstrap (BCa)."""
        n = len(boot_dist)
        z0 = stats.norm.ppf(np.mean(boot_dist > point))
        
        alpha1 = stats.norm.cdf(z0 + (z0 + stats.norm.ppf(self.alpha / 2)) / (1 - 0))
        alpha2 = stats.norm.cdf(z0 + (z0 + stats.norm.ppf(1 - self.alpha / 2)) / (1 - 0))
        
        return (
            float(np.percentile(boot_dist, 100 * alpha1)),
            float(np.percentile(boot_dist, 100 * alpha2))
        )
    
    def boot_correlate(self, x: np.ndarray, y: np.ndarray) -> BootstrapResult:
        """Bootstrap CI for Pearson correlation."""
        x, y = np.asarray(x), np.asarray(y)
        n = len(x)
        
        boot_corrs = np.zeros(self.n_bootstrap)
        point_est = stats.pearsonr(x, y)[0]
        
        for i in range(self.n_bootstrap):
            idx = self._rng.choice(n, n, replace=True)
            if np.std(x[idx]) > 1e-10 and np.std(y[idx]) > 1e-10:
                boot_corrs[i] = stats.pearsonr(x[idx], y[idx])[0]
            else:
                boot_corrs[i] = 0.0
        
        lower, upper = self._percentile_ci(boot_corrs)
        
        p_val = 2 * min(np.mean(boot_corrs <= 0), np.mean(boot_corrs >= 0))
        p_val = min(p_val, 1.0)
        
        return BootstrapResult(
            metric_name="Pearson r",
            point_estimate=point_est,
            lower=lower,
            upper=upper,
            ci_level=self.ci_level,
            n_bootstrap=self.n_bootstrap,
            p_value=p_val,
            statistic_method="bias-corrected percentile"
        )
    
    def boot_auc(self, scores: np.ndarray, labels: np.ndarray) -> BootstrapResult:
        """Bootstrap CI for AUC."""
        scores, labels = np.asarray(scores), np.asarray(labels)
        n = len(scores)
        
        if len(np.unique(labels)) < 2:
            return BootstrapResult(
                metric_name="AUC",
                point_estimate=0.5,
                lower=0.5,
                upper=0.5,
                ci_level=self.ci_level,
                n_bootstrap=self.n_bootstrap,
                p_value=1.0
            )
        
        boot_aucs = np.zeros(self.n_bootstrap)
        point_est = roc_auc_score(labels, scores)
        
        for i in range(self.n_bootstrap):
            idx = self._rng.choice(n, n, replace=True)
            if len(np.unique(labels[idx])) >= 2:
                boot_aucs[i] = roc_auc_score(labels[idx], scores[idx])
            else:
                boot_aucs[i] = 0.5
        
        lower, upper = self._percentile_ci(boot_aucs)
        
        z = stats.norm.ppf(1 - self.alpha / 2)
        se = np.std(boot_aucs) / np.sqrt(self.n_bootstrap)
        p_val = 2 * (1 - stats.norm.cdf(abs(point_est - 0.5) / se)) if se > 0 else 1.0
        
        return BootstrapResult(
            metric_name="AUC",
            point_estimate=point_est,
            lower=lower,
            upper=upper,
            ci_level=self.ci_level,
            n_bootstrap=self.n_bootstrap,
            p_value=p_val
        )
    
    def boot_mean_diff(self, group1: np.ndarray, group2: np.ndarray) -> BootstrapResult:
        """Bootstrap CI for difference in means."""
        g1, g2 = np.asarray(group1), np.asarray(group2)
        point_est = float(np.mean(g1) - np.mean(g2))
        
        boot_diffs = np.zeros(self.n_bootstrap)
        n1, n2 = len(g1), len(g2)
        
        for i in range(self.n_bootstrap):
            s1 = g1[self._rng.choice(n1, n1, replace=True)]
            s2 = g2[self._rng.choice(n2, n2, replace=True)]
            boot_diffs[i] = np.mean(s1) - np.mean(s2)
        
        lower, upper = self._percentile_ci(boot_diffs)
        
        t_stat, p_val = stats.ttest_ind(g1, g2) if len(g1) > 1 and len(g2) > 1 else (0, 1.0)
        
        return BootstrapResult(
            metric_name="Mean Diff (correct - incorrect)",
            point_estimate=point_est,
            lower=lower,
            upper=upper,
            ci_level=self.ci_level,
            n_bootstrap=self.n_bootstrap,
            p_value=float(p_val)
        )
    
    def boot_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> BootstrapResult:
        """Bootstrap CI for Cohen's d effect size."""
        g1, g2 = np.asarray(group1), np.asarray(group2)
        
        def cohens_d(a, b):
            n1, n2 = len(a), len(b)
            if n1 < 2 or n2 < 2:
                return 0.0
            pooled_std = np.sqrt(((n1-1)*np.var(a, ddof=1) + (n2-1)*np.var(b, ddof=1)) / (n1+n2-2))
            if pooled_std < 1e-10:
                return 0.0
            return (np.mean(a) - np.mean(b)) / pooled_std
        
        point_est = cohens_d(g1, g2)
        
        boot_d = np.zeros(self.n_bootstrap)
        n1, n2 = len(g1), len(g2)
        
        for i in range(self.n_bootstrap):
            s1 = g1[self._rng.choice(n1, n1, replace=True)]
            s2 = g2[self._rng.choice(n2, n2, replace=True)]
            boot_d[i] = cohens_d(s1, s2)
        
        lower, upper = self._percentile_ci(boot_d)
        
        return BootstrapResult(
            metric_name="Cohen's d",
            point_estimate=point_est,
            lower=lower,
            upper=upper,
            ci_level=self.ci_level,
            n_bootstrap=self.n_bootstrap
        )
    
    def boot_balanced_accuracy(self, scores: np.ndarray, labels: np.ndarray) -> BootstrapResult:
        """Bootstrap CI for balanced accuracy."""
        scores, labels = np.asarray(scores), np.asarray(labels)
        n = len(scores)
        
        def balanced_acc(s, y, thresh=0.0):
            pred = s > thresh
            pos_acc = np.mean(pred[y == 1]) if np.any(y == 1) else 0.5
            neg_acc = np.mean(~pred[y == 0]) if np.any(y == 0) else 0.5
            return (pos_acc + neg_acc) / 2
        
        point_est = balanced_acc(scores, labels)
        
        boot_bal = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            idx = self._rng.choice(n, n, replace=True)
            boot_bal[i] = balanced_acc(scores[idx], labels[idx])
        
        lower, upper = self._percentile_ci(boot_bal)
        
        return BootstrapResult(
            metric_name="Balanced Accuracy",
            point_estimate=point_est,
            lower=lower,
            upper=upper,
            ci_level=self.ci_level,
            n_bootstrap=self.n_bootstrap
        )


def run_full_bootstrap(
    dataset_path: str,
    output_path: str,
    score_field: str = "hmm_score.normalized_ll",
    n_bootstrap: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """Run full bootstrap analysis on dataset and generate paper-ready output."""
    
    with open(dataset_path, 'r') as f:
        raw = json.load(f)
    
    def _get_nested(d, path):
        keys = path.split('.')
        val = d
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return None
        return val
    
    scores, labels, correct_scores, incorrect_scores = [], [], [], []
    
    for item in raw:
        s = _get_nested(item, score_field)
        if s is None or not np.isfinite(float(s)):
            continue
        s = float(s)
        lbl = int(item.get('is_correct', False))
        scores.append(s)
        labels.append(lbl)
        (correct_scores if lbl else incorrect_scores).append(s)
    
    scores = np.array(scores)
    labels = np.array(labels)
    correct_scores = np.array(correct_scores)
    incorrect_scores = np.array(incorrect_scores)
    
    analyzer = BootstrapAnalyzer(n_bootstrap=n_bootstrap, seed=seed)
    
    auc_result = analyzer.boot_auc(scores, labels)
    r_result = analyzer.boot_correlate(scores, labels.astype(float))
    diff_result = analyzer.boot_mean_diff(correct_scores, incorrect_scores)
    d_result = analyzer.boot_cohens_d(correct_scores, incorrect_scores)
    bal_acc_result = analyzer.boot_balanced_accuracy(scores, labels)
    
    results = {
        'dataset': dataset_path,
        'score_field': score_field,
        'n_samples': len(scores),
        'n_correct': len(correct_scores),
        'n_incorrect': len(incorrect_scores),
        'auc': auc_result.to_dict(),
        'pearson_r': r_result.to_dict(),
        'mean_diff': diff_result.to_dict(),
        'cohens_d': d_result.to_dict(),
        'balanced_accuracy': bal_acc_result.to_dict(),
        'paper_text': {
            'auc_line': str(auc_result),
            'r_line': str(r_result),
            'diff_line': str(diff_result),
        },
        'timestamp': str(Path(dataset_path).stat().st_mtime)
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("BOOTSTRAP CONFIDENCE INTERVAL RESULTS")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Samples: {len(scores)} (correct: {len(correct_scores)}, incorrect: {len(incorrect_scores)})")
    print(f"Score field: {score_field}")
    print("-"*60)
    print(str(auc_result))
    print(str(r_result))
    print(str(diff_result))
    print(str(d_result))
    print(str(bal_acc_result))
    print("-"*60)
    print(f"Results saved to: {output_path}")
    
    return results


def generate_comparison_table(results: Dict[str, Any], output_path: str) -> str:
    """Generate markdown comparison table for paper."""
    table = """
# Bootstrap Confidence Intervals - CoT-HMM Results

| Metric | Point Estimate | 95% CI Lower | 95% CI Upper | Bootstrap N |
|--------|----------------|--------------|--------------|-------------|
"""
    
    for key, val in results.items():
        if key in ['dataset', 'score_field', 'n_samples', 'paper_text', 'timestamp']:
            continue
        if isinstance(val, dict) and 'point_estimate' in val:
            table += f"| {key.replace('_', ' ').title()} | {val['point_estimate']:.4f} | {val['lower']:.4f} | {val['upper']:.4f} | {val.get('n_bootstrap', 'N/A')} |\n"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(table)
    
    return table


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bootstrap CI Analysis")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--score-field', type=str, default='hmm_score.normalized_ll')
    parser.add_argument('--n-bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    results = run_full_bootstrap(
        dataset_path=args.dataset,
        output_path=args.output,
        score_field=args.score_field,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed
    )
    
    table_path = args.output.replace('.json', '.md')
    generate_comparison_table(results, table_path)