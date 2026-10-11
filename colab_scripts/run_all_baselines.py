"""
Run All Baselines for Paper Comparison
ROSCOE (TF-IDF), PiCSAR (Rule-based), Length, Random, HMM
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TAGS = ["PR", "DV", "EQ", "CA", "IN", "RF", "VE", "CO"]
TAG_TO_ID = {t: i for i, t in enumerate(TAGS)}


def roscoe_score(problem: str, cot_text: str) -> float:
    """Compute ROSCOE composite score using TF-IDF."""
    steps = [s.strip() for s in cot_text.split('\n') if s.strip()]
    if not steps:
        steps = [cot_text]
    
    all_texts = [problem] + steps
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    try:
        embeddings = vectorizer.fit_transform(all_texts).toarray()
    except:
        return 0.0
    
    problem_emb = embeddings[0]
    step_embs = embeddings[1:]
    
    # Semantic alignment
    if len(step_embs) >= 2:
        sa = np.mean([cosine_similarity(step_embs[i:i+1], step_embs[i+1:i+2])[0, 0] 
                      for i in range(len(step_embs) - 1)])
    else:
        sa = 0.5
    
    # Step consistency
    if len(step_embs) > 0:
        sc = np.mean([cosine_similarity(problem_emb.reshape(1, -1), step_embs[i:i+1])[0, 0] 
                      for i in range(len(step_embs))])
    else:
        sc = 0.5
    
    return float(0.5 * sa + 0.5 * sc)


def picsar_score(cot_text: str) -> float:
    """Compute PiCSAR-style score using simple perplexity proxy."""
    words = cot_text.split()
    if not words:
        return 0.0
    # Simple perplexity proxy: based on word repetition
    unique_ratio = len(set(words)) / len(words)
    # Higher unique ratio = more diverse = better (lower perplexity proxy)
    return float(unique_ratio)


def length_score(cot_text: str) -> float:
    """Baseline: length of CoT."""
    return float(len(cot_text.split()))


def random_score(seed: int = 42) -> float:
    """Random baseline."""
    np.random.seed(seed)
    return float(np.random.random())


def run_all_baselines(dataset_path: str, output_path: str) -> Dict[str, Any]:
    print(f"[Baselines] Running comparison on {dataset_path}")
    
    with open(dataset_path) as f:
        data = json.load(f)
    
    methods = {
        'hmm_score': [],
        'roscoe': [],
        'picsar': [],
        'length': [],
        'random': [],
    }
    labels = []
    
    for i, item in enumerate(data):
        problem = item.get('problem', '')
        cot_text = item.get('cot_text', item.get('cot', ''))
        is_correct = int(item.get('is_correct', False))
        
        methods['hmm_score'].append(item.get('hmm_score', {}).get('normalized_ll', 0.0))
        methods['roscoe'].append(roscoe_score(problem, cot_text))
        methods['picsar'].append(picsar_score(cot_text))
        methods['length'].append(length_score(cot_text))
        methods['random'].append(random_score(i))
        labels.append(is_correct)
    
    labels_arr = np.array(labels)
    
    results = {}
    for method_name, scores in methods.items():
        scores_arr = np.array(scores)
        if len(np.unique(labels_arr)) > 1 and np.std(scores_arr) > 0:
            auc = roc_auc_score(labels_arr, scores_arr)
            r, p = pearsonr(scores_arr, labels_arr.astype(float))
        else:
            auc, r, p = 0.5, 0.0, 1.0
        
        results[method_name] = {
            'auc': float(auc),
            'pearson_r': float(r),
            'p_value': float(p),
            'n_samples': len(scores)
        }
        print(f"  {method_name:12s}: AUC={auc:.4f}, r={r:.4f}, p={p:.4f}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBaseline comparison saved to {output_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run All Baselines")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    run_all_baselines(args.dataset, args.output)
