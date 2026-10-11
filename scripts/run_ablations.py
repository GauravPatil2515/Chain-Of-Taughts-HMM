#!/usr/bin/env python3
import json
import random
import sys
import math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tagging.rule_based_tagger import tag_cot
from src.hmm.ensemble import EnsembleHMM
from src.hmm.trainer import train_hmm

def compute_entropy(tag_seq):
    if not tag_seq: return 0.0
    counts = {}
    for t in tag_seq:
        counts[t] = counts.get(t, 0) + 1
    n = len(tag_seq)
    return -sum((c/n)*math.log2(c/n) for c in counts.values())

def main():
    with open("gsm8k_qwen_300_real_traces.json", "r") as f:
        data = json.load(f)

    for item in data:
        if "tag_sequence" not in item:
            item["tag_sequence"] = [s["tag"] for s in tag_cot(item.get("cot_text", item.get("cot_raw", "")))]
    
    train_data = data[:200]
    test_data = data[200:]
    
    train_correct_seqs = [item["tag_sequence"] for item in train_data if item.get("is_correct", False)]
    test_seqs = [item["tag_sequence"] for item in test_data]
    test_labels = [item.get("is_correct", False) for item in test_data]
    
    from sklearn.metrics import roc_auc_score
    
    # Priority 1: HMM state-count ablation
    print("\n--- Priority 1: HMM State-Count Ablation ---")
    for n_comp in [3, 5, 8, 12]:
        try:
            model = train_hmm(train_correct_seqs, n_components=n_comp, n_iter=100, random_state=42)
            
            from src.hmm.trainer import TAG_TO_ID
            scores = []
            for seq in test_seqs:
                if len(seq) == 0:
                    scores.append(float('-inf'))
                    continue
                encoded = [TAG_TO_ID.get(tag, TAG_TO_ID["IN"]) for tag in seq]
                ll = model.score(np.array(encoded).reshape(-1, 1))
                scores.append(ll / len(seq))
            
            valid = [(s, l) for s, l in zip(scores, test_labels) if s != float('-inf') and not np.isnan(s)]
            if valid:
                s, l = zip(*valid)
                auc = roc_auc_score(l, s)
                print(f"N={n_comp}: AUC = {auc:.3f}")
        except Exception as e:
            print(f"N={n_comp} failed: {e}")

    # Priority 1: HMM vs HMM+Entropy Ablation
    print("\n--- Priority 1: HMM vs HMM+Entropy ---")
    ensemble = EnsembleHMM(n_iter=100)
    ensemble.fit(train_correct_seqs, verbose=False)
    
    hmm_scores = [ensemble.score(seq)["ensemble_ll"] for seq in test_seqs]
    entropies = [compute_entropy(seq) for seq in test_seqs]
    
    # Entropy alone is usually negatively correlated with correctness (higher entropy = more confusion)
    # We negate it so higher score = better
    entropy_scores = [-e for e in entropies]
    
    valid_data = [(hmm_scores[i], entropy_scores[i], test_labels[i]) 
                  for i in range(len(test_seqs)) 
                  if hmm_scores[i] != float('-inf') and not np.isnan(hmm_scores[i])]
    
    if valid_data:
        h_s, e_s, labels = zip(*valid_data)
        
        # Standardize for combination
        h_s = np.array(h_s)
        e_s = np.array(e_s)
        h_s_norm = (h_s - np.mean(h_s)) / np.std(h_s)
        e_s_norm = (e_s - np.mean(e_s)) / np.std(e_s)
        
        combined_scores = h_s_norm + e_s_norm
        
        auc_hmm = roc_auc_score(labels, h_s)
        auc_ent = roc_auc_score(labels, e_s)
        auc_combo = roc_auc_score(labels, combined_scores)
        
        print(f"HMM Alone AUC: {auc_hmm:.3f}")
        print(f"Entropy Alone AUC: {auc_ent:.3f}")
        print(f"HMM + Entropy (Combined) AUC: {auc_combo:.3f}")

if __name__ == '__main__':
    main()
