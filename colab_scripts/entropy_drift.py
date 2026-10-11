"""
Entropy Drift Rate (EDR) Online Monitoring
Streaming entropy analysis for overthinking detection
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import pearsonr, spearmanr

from colab_scripts.config import ExperimentConfig
from colab_scripts.tagging import SemanticTagger
from hmmlearn import hmm


def compute_pointwise_entropy(
    tag_sequence: List[str],
    hmm_model,
    tag_to_id: Dict[str, int]
) -> List[float]:
    """
    Compute pointwise entropy at each step.
    Uses transition probabilities from HMM.
    """
    if not tag_sequence or len(tag_sequence) < 2:
        return []
    
    try:
        transmat = hmm_model.transmat_
    except AttributeError:
        transmat = None
    
    if transmat is None:
        uniform_entropy = np.log(len(transmat)) if transmat is not None else np.log(8)
        return [float(uniform_entropy)] * len(tag_sequence)
    
    TAGS = ["PR", "DV", "EQ", "CA", "IN", "RF", "VE", "CO"]
    
    pointwise_entropies = []
    
    for i in range(len(tag_sequence)):
        if i == 0:
            continue
        
        prev_tag = tag_sequence[i-1]
        prev_id = tag_to_id.get(prev_tag, 0)
        
        trans_probs = transmat[prev_id]
        trans_probs = np.clip(trans_probs, 1e-10, 1.0)
        entropy = -np.sum(trans_probs * np.log(trans_probs))
        pointwise_entropies.append(float(entropy))
    
    return pointwise_entropies


def compute_entropy_drift_rate(
    entropy_trajectory: List[float]
) -> float:
    """
    Compute Entropy Drift Rate (EDR).
    
    EDR = (1/(T-1)) * sum_{t=2}^{T} |H_t - H_{t-1}|
    
    Lower EDR = converging/consistent reasoning
    Higher EDR = oscillating/confused reasoning
    """
    if len(entropy_trajectory) < 2:
        return 0.0
    
    diffs = [abs(entropy_trajectory[i] - entropy_trajectory[i-1]) 
             for i in range(1, len(entropy_trajectory))]
    
    return float(np.mean(diffs))


def compute_entropy_annealing_score(
    entropy_trajectory: List[float]
) -> float:
    """
    Compute entropy annealing score.
    Positive = entropy decreases (converging)
    Negative = entropy increases (diverging)
    """
    if len(entropy_trajectory) < 3:
        return 0.0
    
    first_third = np.mean(entropy_trajectory[:len(entropy_trajectory)//3])
    last_third = np.mean(entropy_trajectory[-len(entropy_trajectory)//3:])
    
    return float(first_third - last_third)


def detect_overthinking_pattern(
    entropy_trajectory: List[float],
    e_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Detect overthinking patterns from entropy trajectory.
    
    Correct chains: entropy decreases (annealing)
    Incorrect chains: entropy spikes or oscillates
    Overthinking: entropy plateaus near-zero but steps become repetitive
    """
    if len(entropy_trajectory) < 3:
        return {
            'is_overthinking': False,
            'anomaly_score': 0.0,
            'reason': 'insufficient_steps'
        }
    
    edr = compute_entropy_drift_rate(entropy_trajectory)
    annealing = compute_entropy_annealing_score(entropy_trajectory)
    entropy_range = max(entropy_trajectory) - min(entropy_trajectory)
    
    score = 0.0
    reasons = []
    
    if entropy_range < 0.1 and len(entropy_trajectory) > 5:
        score += 0.8
        reasons.append('low_variance_plateau')
    
    if np.std(entropy_trajectory) < 0.05:
        score += 0.6
        reasons.append('zero_entropy_drift')
    
    if annealing < -0.2:
        score += 0.4
        reasons.append('entropy_diverging')
    
    edr_threshold = np.percentile([0.1, 0.3, 0.5], 50)
    if edr > e_threshold:
        score += 0.3
        reasons.append('high_entropy_drift')
    
    is_overthinking = score > 0.5
    
    return {
        'is_overthinking': is_overthinking,
        'anomaly_score': min(score, 1.0),
        'edr': edr,
        'annealing_score': annealing,
        'entropy_range': entropy_range,
        'patterns': reasons
    }


def analyze_trajectory_by_correctness(
    dataset_path: str,
    n_top: int = 100
) -> Dict[str, Any]:
    """
    Analyze entropy trajectories on a dataset.
    Computes EDR for all traces and correlates with correctness.
    """
    from hmmlearn import hmm
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    TAGS = ["PR", "DV", "EQ", "CA", "IN", "RF", "VE", "CO"]
    TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}
    
    if len(data) > n_top:
        np.random.seed(42)
        indices = np.random.choice(len(data), n_top, replace=False)
        data = [data[i] for i in indices]
    
    correct_edrs = []
    incorrect_edrs = []
    correct_entropies = []
    incorrect_entropies = []
    
    for item in data:
        tag_seq = item.get('tag_sequence', [])
        is_correct = item.get('is_correct', False)
        
        if len(tag_seq) < 3:
            continue
        
        encoded = np.array([TAG_TO_ID.get(t, 0) for t in tag_seq]).reshape(-1, 1)
        
        model = hmm.CategoricalHMM(n_components=5, random_state=42)
        try:
            model.fit([encoded.flatten()], [len(encoded)])
        except:
            continue
        
        pointwise_ent = compute_pointwise_entropy(tag_seq, model, TAG_TO_ID)
        if not pointwise_ent:
            continue
        
        edr = compute_entropy_drift_rate(pointwise_ent)
        entropy_trajectory = pointwise_ent
        
        if is_correct:
            correct_edrs.append(edr)
            correct_entropies.append(np.mean(entropy_trajectory))
        else:
            incorrect_edrs.append(edr)
            incorrect_entropies.append(np.mean(entropy_trajectory))
    
    correct_edrs_arr = np.array(correct_edrs) if len(correct_edrs) > 0 else np.array([0.0])
    incorrect_edrs_arr = np.array(incorrect_edrs) if len(incorrect_edrs) > 0 else np.array([0.0])
    
    all_edrs = np.concatenate([correct_edrs_arr, incorrect_edrs_arr])
    all_labels = np.array([1]*len(correct_edrs_arr) + [0]*len(incorrect_edrs_arr))
    
    if len(np.unique(all_labels)) > 1 and len(all_edrs) > 2:
        r_pearson, p_pearson = pearsonr(all_edrs, all_labels.astype(float))
        r_spearman, p_spearman = spearmanr(all_edrs, all_labels.astype(float))
    else:
        r_pearson, p_pearson = 0.0, 1.0
        r_spearman, p_spearman = 0.0, 1.0
    
    results = {
        'n_correct': len(correct_edrs),
        'n_incorrect': len(incorrect_edrs),
        'correct_mean_edr': float(np.mean(correct_edrs)) if len(correct_edrs) > 0 else 0.0,
        'incorrect_mean_edr': float(np.mean(incorrect_edrs)) if len(incorrect_edrs) > 0 else 0.0,
        'correct_mean_entropy': float(np.mean(correct_entropies)) if len(correct_entropies) > 0 else 0.0,
        'incorrect_mean_entropy': float(np.mean(incorrect_entropies)) if len(incorrect_entropies) > 0 else 0.0,
        'edr_correlation': {
            'pearson_r': float(r_pearson),
            'p_value': float(p_pearson),
            'spearman_r': float(r_spearman),
            'p_value_spearman': float(p_spearman),
            'significant': p_pearson < 0.05
        },
        'trajectory_examples': {
            'low_edr_correct': sorted(zip(correct_edrs, tag_seq))[:5] if correct_edrs else [],
            'high_edr_incorrect': sorted(zip(incorrect_edrs, tag_seq), reverse=True)[:5] if incorrect_edrs else []
        }
    }
    
    return results


def run_edr_experiment(
    dataset_path: str,
    output_path: str,
    n_samples: int = 150
) -> Dict[str, Any]:
    """
    Run complete EDR experiment on existing data.
    No new data collection needed - uses existing tagged traces.
    """
    print(f"[edr] Analyzing entropy trajectories in {dataset_path}")
    
    results = analyze_trajectory_by_correctness(dataset_path, n_top=n_samples)
    
    results['experiment'] = 'entropy_drift_rate'
    results['timestamp'] = str(Path(dataset_path).stat().st_mtime) if Path(dataset_path).exists() else None
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("ENTROPY DRIFT RATE RESULTS")
    print("="*60)
    print(f"Correct traces: {results['n_correct']}")
    print(f"Incorrect traces: {results['n_incorrect']}")
    print(f"\nMean EDR - Correct: {results['correct_mean_edr']:.4f}")
    print(f"Mean EDR - Incorrect: {results['incorrect_mean_edr']:.4f}")
    print(f"\nCorrelation with correctness:")
    print(f"  Pearson r: {results['edr_correlation']['pearson_r']:.4f} (p={results['edr_correlation']['p_value']:.4f})")
    print(f"  Spearman r: {results['edr_correlation']['spearman_r']:.4f} (p={results['edr_correlation']['p_value_spearman']:.4f})")
    
    if results['edr_correlation']['significant']:
        print("\n✓ EDR is SIGNIFICANTLY correlated with correctness")
        print("✓ Entropy drift rate provides overthinking detection signal")
    else:
        print("\n✗ EDR correlation not significant in this sample")
    
    print("="*60)
    print(f"Results saved to {output_path}")
    
    return results


def build_edr_report(results: Dict[str, Any], output_path: str) -> str:
    """Generate markdown report for EDR analysis."""
    
    corr = results['edr_correlation']
    
    report = f"""# Entropy Drift Rate Analysis Results

## Key Metrics

| Metric | Value |
|--------|-------|
| Correct Mean EDR | {results['correct_mean_edr']:.4f} |
| Incorrect Mean EDR | {results['incorrect_mean_edr']:.4f} |
| Difference | {results['correct_mean_edr'] - results['incorrect_mean_edr']:.4f} |

## Statistical Significance

| Correlation | r | p-value | Significant |
|-------------|---|---------|-------------|
| Pearson | {corr['pearson_r']:.4f} | {corr['p_value']:.4f} | {"Yes" if corr['significant'] else "No"} |
| Spearman | {corr['spearman_r']:.4f} | {corr['p_value_spearman']:.4f} | {"Yes" if corr['p_value_spearman'] < 0.05 else "No"} |

## Interpretation

**Entropy Drift Rate (EDR)** measures how much entropy changes at each step of reasoning.

- **Low EDR**: Entropy is stable across steps → converging reasoning
- **High EDR**: Entropy fluctuates → confused or looping reasoning

### Findings

"""
    
    if corr['significant']:
        report += f"1. EDR is significantly correlated with chain correctness (p < 0.05)\n"
        report += f"2. This provides an **online overthinking detection signal**\n"
        report += f"3. Can be computed at inference time to flag suspicious reasoning\n"
    else:
        report += f"1. EDR shows trend but not statistically significant\n"
        report += f"2. May require larger sample size or different threshold\n"
    
    report += f"\n### Paper-ready Statistics\n"
    report += f"- r = {corr['pearson_r']:.3f} [95% CI: compute separately]\n"
    report += f"- p = {corr['p_value']:.4f}\n"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entropy Drift Rate Analysis")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--n-samples', type=int, default=150)
    
    args = parser.parse_args()
    
    results = run_edr_experiment(
        dataset_path=args.dataset,
        output_path=args.output,
        n_samples=args.n_samples
    )
    
    report_path = args.output.replace('.json', '.md')
    build_edr_report(results, report_path)