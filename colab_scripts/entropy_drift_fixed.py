"""
Entropy Drift Rate (EDR) Online Monitoring - FIXED VERSION
Streaming entropy analysis for overthinking detection
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
from hmmlearn import hmm

from colab_scripts.config import ExperimentConfig
from colab_scripts.tagging import SemanticTagger


TAGS = ["PR", "DV", "EQ", "CA", "IN", "RF", "VE", "CO"]
TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}


def compute_pointwise_entropy(
    tag_sequence: List[str],
    hmm_model,
    tag_to_id: Dict[str, int]
) -> List[float]:
    """Compute pointwise entropy at each step using pre-trained HMM transition matrix."""
    if not tag_sequence or len(tag_sequence) < 2:
        return []
    
    try:
        transmat = hmm_model.transmat_
    except AttributeError:
        return []
    
    if transmat is None:
        return []
    
    pointwise_entropies = []
    
    for i in range(1, len(tag_sequence)):
        prev_tag = tag_sequence[i-1]
        prev_id = tag_to_id.get(prev_tag, 0)
        
        trans_probs = transmat[prev_id]
        trans_probs = np.clip(trans_probs, 1e-10, 1.0)
        entropy = -np.sum(trans_probs * np.log(trans_probs))
        pointwise_entropies.append(float(entropy))
    
    return pointwise_entropies


def compute_entropy_drift_rate(entropy_trajectory: List[float]) -> float:
    """Compute Entropy Drift Rate (EDR). Lower = converging, Higher = oscillating."""
    if len(entropy_trajectory) < 2:
        return 0.0
    diffs = [abs(entropy_trajectory[i] - entropy_trajectory[i-1]) 
             for i in range(1, len(entropy_trajectory))]
    return float(np.mean(diffs))


def compute_entropy_annealing_score(entropy_trajectory: List[float]) -> float:
    """Positive = entropy decreases (converging), Negative = increases (diverging)."""
    if len(entropy_trajectory) < 3:
        return 0.0
    first_third = np.mean(entropy_trajectory[:len(entropy_trajectory)//3])
    last_third = np.mean(entropy_trajectory[-len(entropy_trajectory)//3:])
    return float(first_third - last_third)


def detect_overthinking_pattern(
    entropy_trajectory: List[float],
    e_threshold: float = 0.5
) -> Dict[str, Any]:
    """Detect overthinking patterns from entropy trajectory."""
    if len(entropy_trajectory) < 3:
        return {'is_overthinking': False, 'anomaly_score': 0.0, 'reason': 'insufficient_steps'}
    
    edr = compute_entropy_drift_rate(entropy_trajectory)
    annealing = compute_entropy_annealing_score(entropy_trajectory)
    entropy_range = max(entropy_trajectory) - min(entropy_trajectory)
    
    score = 0.0
    reasons = []
    
    if entropy_range < 0.1 and len(entropy_trajectory) > 5:
        score += 0.8; reasons.append('low_variance_plateau')
    if np.std(entropy_trajectory) < 0.05:
        score += 0.6; reasons.append('zero_entropy_drift')
    if annealing < -0.2:
        score += 0.4; reasons.append('entropy_diverging')
    if edr > e_threshold:
        score += 0.3; reasons.append('high_entropy_drift')
    
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
    """Analyze entropy trajectories using a single pre-trained HMM on correct traces."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if len(data) > n_top:
        np.random.seed(42)
        indices = np.random.choice(len(data), n_top, replace=False)
        data = [data[i] for i in indices]
    
    # Collect correct sequences for HMM training
    correct_seqs = [d['tag_sequence'] for d in data if d['is_correct'] and len(d.get('tag_sequence', [])) >= 2]
    if not correct_seqs:
        return {'error': 'No correct sequences found'}
    
    # Train single HMM on ALL correct sequences
    enc_all, lengths_all = [], []
    for seq in correct_seqs:
        e = [TAG_TO_ID.get(t, 0) for t in seq]
        enc_all.extend(e)
        lengths_all.append(len(e))
    
    X_all = np.array(enc_all).reshape(-1, 1)
    hmm_model = hmm.CategoricalHMM(n_components=5, random_state=42, verbose=False)
    try:
        hmm_model.fit(X_all, lengths_all)
    except:
        return {'error': 'HMM training failed'}
    
    correct_edrs, incorrect_edrs = [], []
    correct_entropies, incorrect_entropies = [], []
    
    for item in data:
        tag_seq = item.get('tag_sequence', [])
        is_correct = item.get('is_correct', False)
        
        if len(tag_seq) < 3:
            continue
        
        pointwise_ent = compute_pointwise_entropy(tag_seq, hmm_model, TAG_TO_ID)
        if not pointwise_ent or len(pointwise_ent) < 2:
            continue
        
        edr = compute_entropy_drift_rate(pointwise_ent)
        entropy_trajectory = pointwise_ent
        
        if is_correct:
            correct_edrs.append(edr)
            correct_entropies.append(np.mean(entropy_trajectory))
        else:
            incorrect_edrs.append(edr)
            incorrect_entropies.append(np.mean(entropy_trajectory))
    
    correct_edrs_arr = np.array(correct_edrs) if correct_edrs else np.array([0.0])
    incorrect_edrs_arr = np.array(incorrect_edrs) if incorrect_edrs else np.array([0.0])
    
    all_edrs = np.concatenate([correct_edrs_arr, incorrect_edrs_arr])
    all_labels = np.array([1]*len(correct_edrs_arr) + [0]*len(incorrect_edrs_arr))
    
    if len(np.unique(all_labels)) > 1 and len(all_edrs) > 2:
        r_pearson, p_pearson = pearsonr(all_edrs, all_labels.astype(float))
        r_spearman, p_spearman = spearmanr(all_edrs, all_labels.astype(float))
    else:
        r_pearson, p_pearson = 0.0, 1.0
        r_spearman, p_spearman = 0.0, 1.0
    
    return {
        'n_correct': len(correct_edrs),
        'n_incorrect': len(incorrect_edrs),
        'correct_mean_edr': float(np.mean(correct_edrs)) if correct_edrs else 0.0,
        'incorrect_mean_edr': float(np.mean(incorrect_edrs)) if incorrect_edrs else 0.0,
        'correct_mean_entropy': float(np.mean(correct_entropies)) if correct_entropies else 0.0,
        'incorrect_mean_entropy': float(np.mean(incorrect_entropies)) if incorrect_entropies else 0.0,
        'edr_correlation': {
            'pearson_r': float(r_pearson),
            'p_value': float(p_pearson),
            'spearman_r': float(r_spearman),
            'p_value_spearman': float(p_spearman),
            'significant': p_pearson < 0.05
        }
    }


def run_edr_experiment(
    dataset_path: str,
    output_path: str,
    n_samples: int = 150
) -> Dict[str, Any]:
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entropy Drift Rate Analysis")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--n-samples', type=int, default=150)
    args = parser.parse_args()
    run_edr_experiment(args.dataset, args.output, args.n_samples)
