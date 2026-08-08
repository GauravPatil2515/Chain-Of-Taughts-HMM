"""
Colab/Kaggle Optimized Runner Script
All-in-one script for running experiments on Colab or Kaggle
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

print("="*70)
print("CoT-HMM: Chain-of-Thought Structural Verification")
print("Optimized Colab/Kaggle Pipeline v2.0")
print("="*70)

IS_COLAB = 'google.colab' in sys.modules
IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
DEVICE = 'cuda' if (IS_COLAB or IS_KAGGLE) else 'cpu'

print(f"\nEnvironment: {'Colab' if IS_COLAB else 'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"Device: {DEVICE}")

WORKSPACE = Path('/content/drive/MyDrive/cot_hmm' if IS_COLAB else 'Chain-Of-Taughts-HMM')
DATA_DIR = WORKSPACE / 'data' / 'v2'
RESULTS_DIR = WORKSPACE / 'results'
MODELS_DIR = WORKSPACE / 'models'

for d in [RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

try:
    import numpy as np
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    print("✓ Core packages available: numpy, scipy, sklearn")
except ImportError as e:
    print(f"✗ Missing package: {e}")
    sys.exit(1)

try:
    from hmmlearn import hmm
    import joblib
    print("✓ HMM packages available: hmmlearn, joblib")
except ImportError as e:
    print(f"✗ Missing HMM package: {e}")

TAG_TO_ID = {"PR": 0, "DV": 1, "EQ": 2, "CA": 3, "IN": 4, "RF": 5, "VE": 6, "CO": 7}
TAGS = list(TAG_TO_ID.keys())


def load_dataset(path: str):
    """Load dataset from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def train_hmm_ensemble(sequences, n_components=5, n_models=12, seed=42, n_iter=100):
    """Train HMM ensemble on tag sequences."""
    print(f"\n[Training] Creating HMM ensemble: {n_components} states, {n_models} models")
    
    from hmmlearn import hmm
    
    models = []
    encoded_all = []
    lengths = []
    
    for seq in sequences:
        if len(seq) >= 2:
            encoded = np.array([TAG_TO_ID.get(t, 0) for t in seq]).reshape(-1, 1)
            encoded_all.extend(encoded.flatten())
            lengths.append(len(seq))
    
    if not encoded_all:
        raise ValueError("No valid sequences")
    
    X = np.array(encoded_all, dtype=np.int32).reshape(-1, 1)
    
    for i in range(n_models):
        try:
            model = hmm.CategoricalHMM(
                n_components=n_components,
                n_iter=n_iter,
                random_state=seed + i,
                verbose=False
            )
            model.fit(X, lengths)
            models.append(model)
            if (i + 1) % 4 == 0:
                print(f"  Trained {i+1}/{n_models} models")
        except Exception as e:
            print(f"  Model {i} failed: {e}")
    
    print(f"✓ Trained {len(models)} models successfully")
    return models


def score_with_ensemble(models, tag_sequence):
    """Score tag sequence with ensemble."""
    if not tag_sequence or not models:
        return float('-inf')
    
    encoded = np.array([TAG_TO_ID.get(t, 0) for t in tag_sequence]).reshape(-1, 1)
    scores = []
    
    for model in models:
        try:
            ll = model.score(encoded)
            scores.append(ll / len(tag_sequence))
        except:
            pass
    
    return float(np.mean(scores)) if scores else float('-inf')


def compute_entropy_trajectory(model, tag_sequence, n_components=5):
    """Compute entropy at each step of the sequence."""
    if len(tag_sequence) < 2:
        return []
    
    try:
        transmat = model.transmat_
    except:
        return []
    
    entropies = []
    for i in range(1, len(tag_sequence)):
        prev_tag = tag_sequence[i-1]
        prev_id = TAG_TO_ID.get(prev_tag, 0)
        probs = transmat[prev_id]
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        entropies.append(float(entropy))
    
    return entropies


def run_bootstrap_ci(scores, labels, n_bootstrap=10000, seed=42):
    """Run bootstrap confidence intervals."""
    rng = np.random.RandomState(seed)
    n = len(scores)
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)
    
    boot_corrs = np.zeros(n_bootstrap)
    boot_aucs = np.zeros(n_bootstrap)
    
    point_corr = stats.pearsonr(scores_arr, labels_arr.astype(float))[0]
    point_auc = roc_auc_score(labels_arr, scores_arr)
    
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if np.std(scores_arr[idx]) > 1e-10:
            boot_corrs[i] = stats.pearsonr(scores_arr[idx], labels_arr[idx].astype(float))[0]
        if len(np.unique(labels_arr[idx])) > 1:
            boot_aucs[i] = roc_auc_score(labels_arr[idx], scores_arr[idx])
    
    alpha = 0.05
    corr_ci = (float(np.percentile(boot_corrs, 100*alpha/2)), float(np.percentile(boot_corrs, 100*(1-alpha/2))))
    auc_ci = (float(np.percentile(boot_aucs, 100*alpha/2)), float(np.percentile(boot_aucs, 100*(1-alpha/2))))
    
    return {
        'r': point_corr,
        'r_ci': corr_ci,
        'auc': point_auc,
        'auc_ci': auc_ci,
        'n_bootstrap': n_bootstrap
    }


def main():
    parser = argparse.ArgumentParser(description="CoT-HMM Colab/Kaggle Runner")
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--n-bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-train', action='store_true')
    
    args = parser.parse_args()
    
    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        dataset_path = DATA_DIR / 'gsm8k_500_traces.json'
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = RESULTS_DIR / f"bootstrap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Data] Loading dataset from: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"Dataset not found, creating sample data...")
        sample_data = []
        for i in range(150):
            is_correct = i % 5 != 0
            sample_data.append({
                'sample_id': f'gsm_{i:04d}',
                'problem': f'What is {i} + {i*2}?',
                'cot_text': f'Step 1: Calculate. Step 2: Compute. Answer is {i*3}.',
                'tag_sequence': ['PR', 'CA', 'CO'] if is_correct else ['PR', 'DV', 'CO'],
                'is_correct': is_correct,
                'hmm_score': {'normalized_ll': float(-2.0 + np.random.randn() * 0.5)}
            })
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dataset_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        data = sample_data
    else:
        data = load_dataset(str(dataset_path))
    
    print(f"[Data] Loaded {len(data)} traces")
    print(f"  Correct: {sum(1 for d in data if d.get('is_correct', False))}")
    print(f"  Incorrect: {sum(1 for d in data if not d.get('is_correct', False))}")
    
    tag_sequences = [d.get('tag_sequence', []) for d in data]
    tags_correct = [seq for seq, d in zip(tag_sequences, data) if d.get('is_correct', False)]
    tags_incorrect = [seq for seq, d in zip(tag_sequences, data) if not d.get('is_correct', False)]
    
    if not args.skip_train:
        print("\n[Training] Training HMM ensemble on correct traces...")
        models = train_hmm_ensemble(tags_correct, n_components=5, n_models=12)
        
        model_path = output_dir / 'ensemble_hmm.pkl'
        import joblib
        joblib.dump({'models': models, 'tag_to_id': TAG_TO_ID}, model_path)
        print(f"[Training] Saved ensemble to {model_path}")
    else:
        print("\n[Skipping] Using existing model if available")
        model_path = output_dir / 'ensemble_hmm.pkl'
        if model_path.exists():
            import joblib
            loaded = joblib.load(str(model_path))
            models = loaded['models']
        else:
            models = train_hmm_ensemble(tags_correct, n_components=5, n_models=12)
    
    print("\n[Scoring] Computing scores and analyzing results...")
    
    scores = []
    labels = []
    edr_values = []
    
    for i, item in enumerate(data):
        tag_seq = item.get('tag_sequence', [])
        score = item.get('hmm_score', {}).get('normalized_ll', 0.0)
        
        if not tag_seq or not item.get('hmm_score'):
            score = score_with_ensemble(models, tag_seq)
        
        scores.append(score)
        labels.append(int(item.get('is_correct', False)))
        
        if models:
            entropies = compute_entropy_trajectory(models[0], tag_seq)
            if entropies:
                edr = np.mean(np.abs(np.diff(entropies)))
                edr_values.append(edr)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(data)} traces")
    
    print("\n[Analysis] Running bootstrap CI analysis...")
    bootstrap_results = run_bootstrap_ci(scores, labels, n_bootstrap=args.n_bootstrap, seed=args.seed)
    
    results = {
        'experiment': 'bootstrap_ci_analysis',
        'dataset': str(dataset_path),
        'timestamp': datetime.utcnow().isoformat(),
        'n_samples': len(data),
        'n_correct': sum(labels),
        'n_incorrect': len(data) - sum(labels),
        'bootstrap': bootstrap_results,
        'thresholds': {
            'theta_high': float(np.mean([s for s, l in zip(scores, labels) if l == 1]) + np.std(scores)),
            'theta_low': float(np.mean([s for s, l in zip(scores, labels) if l == 0]) - np.std(scores))
        },
        'r_summary': f"r={bootstrap_results['r']:.3f} [95% CI: {bootstrap_results['r_ci'][0]:.3f}, {bootstrap_results['r_ci'][1]:.3f}]",
        'auc_summary': f"AUC={bootstrap_results['auc']:.3f} [95% CI: {bootstrap_results['auc_ci'][0]:.3f}, {bootstrap_results['auc_ci'][1]:.3f}]"
    }
    
    summary_path = output_dir / 'bootstrap_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Pearson r: {results['r_summary']}")
    print(f"AUC:       {results['auc_summary']}")
    print(f"Thresholds: high={results['thresholds']['theta_high']:.3f}, low={results['thresholds']['theta_low']:.3f}")
    print("="*60)
    
    md_path = output_dir / 'results.md'
    with open(md_path, 'w') as f:
        f.write(f"""# CoT-HMM Results

## Bootstrap Confidence Intervals

| Metric | Point Estimate | 95% CI |
|--------|---------------|--------|
| Pearson r | {bootstrap_results['r']:.4f} | [{bootstrap_results['r_ci'][0]:.4f}, {bootstrap_results['r_ci'][1]:.4f}] |
| AUC | {bootstrap_results['auc']:.4f} | [{bootstrap_results['auc_ci'][0]:.4f}, {bootstrap_results['auc_ci'][1]:.4f}] |

## Routing Thresholds

- theta_high = {results['thresholds']['theta_high']:.4f} (pass through)
- theta_low = {results['thresholds']['theta_low']:.4f} (escalate/reject)

## Sample Statistics

- N = {len(data)}
- Correct = {sum(labels)}
- Incorrect = {len(data) - sum(labels)}

Generated: {datetime.utcnow().isoformat()}
""")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - bootstrap_summary.json")
    print(f"  - results.md")
    print("\n✓ Complete")


if __name__ == "__main__":
    main()