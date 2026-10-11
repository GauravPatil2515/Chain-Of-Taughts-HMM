"""
ROSCOE Baseline - Lightweight Version (TF-IDF based)
No sentence-transformers needed - uses scikit-learn TfidfVectorizer
"""

import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from typing import Dict, Any, List


def roscoe_score(problem: str, cot_steps: List[str]) -> Dict[str, float]:
    """Compute ROSCOE-style scores using TF-IDF embeddings."""
    if not cot_steps:
        return {'semantic_alignment': 0.0, 'step_consistency': 0.0, 
                'informativeness': 0.0, 'roscoe_composite': 0.0}
    
    all_texts = [problem] + cot_steps
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    try:
        embeddings = vectorizer.fit_transform(all_texts)
        embeddings = embeddings.toarray()
    except:
        return {'semantic_alignment': 0.0, 'step_consistency': 0.0, 
                'informativeness': 0.0, 'roscoe_composite': 0.0}
    
    problem_emb = embeddings[0]
    step_embs = embeddings[1:]
    
    # Semantic Alignment: consecutive step similarity
    if len(step_embs) >= 2:
        sa_sims = [cosine_similarity(step_embs[i:i+1], step_embs[i+1:i+2])[0, 0] 
                   for i in range(len(step_embs) - 1)]
        semantic_alignment = float(np.mean(sa_sims))
    else:
        semantic_alignment = 0.0
    
    # Step Consistency: step-to-problem similarity
    if len(step_embs) > 0:
        sc_sims = [cosine_similarity(problem_emb.reshape(1, -1), step_embs[i:i+1])[0, 0] 
                   for i in range(len(step_embs))]
        step_consistency = float(np.mean(sc_sims))
    else:
        step_consistency = 0.0
    
    # Informativeness: novelty vs prior steps
    if len(step_embs) >= 2:
        novelties = []
        for i in range(1, len(step_embs)):
            prior_sims = [cosine_similarity(step_embs[i:i+1], step_embs[j:j+1])[0, 0] 
                          for j in range(i)]
            novelties.append(1.0 - max(prior_sims))
        informativeness = float(np.mean(novelties))
    else:
        informativeness = 1.0
    
    composite = 0.35 * semantic_alignment + 0.35 * step_consistency + 0.30 * informativeness
    
    return {
        'semantic_alignment': semantic_alignment,
        'step_consistency': step_consistency,
        'informativeness': informativeness,
        'roscoe_composite': composite
    }


def run_roscoe_on_dataset(dataset_path: str, output_path: str) -> Dict[str, Any]:
    print(f"[ROSCOE] Scoring dataset from {dataset_path}")
    
    with open(dataset_path) as f:
        data = json.load(f)
    
    roscoe_scores = []
    labels = []
    
    for i, item in enumerate(data):
        problem = item.get('problem', '')
        cot_text = item.get('cot_text', '')
        is_correct = int(item.get('is_correct', False))
        
        # Split CoT into steps
        steps = [s.strip() for s in cot_text.split('\n') if s.strip()]
        if not steps:
            steps = [cot_text]
        
        sc = roscoe_score(problem, steps)
        item['roscoe_score'] = sc
        roscoe_scores.append(sc['roscoe_composite'])
        labels.append(is_correct)
        
        if (i + 1) % 50 == 0:
            print(f"  [ROSCOE] Scored {i+1}/{len(data)}")
    
    # Save scored data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Compute summary stats
    roscoe_arr = np.array(roscoe_scores)
    labels_arr = np.array(labels)
    
    if len(np.unique(labels_arr)) > 1:
        auc = roc_auc_score(labels_arr, roscoe_arr)
        r, p = pearsonr(roscoe_arr, labels_arr.astype(float))
    else:
        auc, r, p = 0.5, 0.0, 1.0
    
    summary = {
        'method': 'ROSCOE (TF-IDF)',
        'n_samples': len(data),
        'auc': round(float(auc), 4),
        'pearson_r': round(float(r), 4),
        'p_value': round(float(p), 5),
        'mean_correct': float(roscoe_arr[labels_arr == 1].mean()) if (labels_arr == 1).any() else 0.0,
        'mean_incorrect': float(roscoe_arr[labels_arr == 0].mean()) if (labels_arr == 0).any() else 0.0,
    }
    
    print("\n" + "="*55)
    print("ROSCOE RESULTS")
    print("="*55)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ROSCOE Baseline (Light)")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    run_roscoe_on_dataset(args.dataset, args.output)
