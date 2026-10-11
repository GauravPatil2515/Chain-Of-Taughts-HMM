#!/usr/bin/env python3
import json
import random
import copy
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tagging.rule_based_tagger import tag_cot
from src.hmm.ensemble import EnsembleHMM
from src.evaluation.baselines import score_with_all_baselines

def main():
    print("Loading 300 real traces...")
    with open("gsm8k_qwen_300_real_traces.json", "r") as f:
        data = json.load(f)

    print("Tagging traces...")
    for item in data:
        if "tag_sequence" not in item:
            tagged_steps = tag_cot(item.get("cot_text", item.get("cot_raw", "")))
            item["tag_sequence"] = [s["tag"] for s in tagged_steps]
    
    # Split
    train_data = data[:200]
    test_data = data[200:]
    
    train_correct_seqs = [item["tag_sequence"] for item in train_data if item.get("is_correct", False)]
    print(f"Training on {len(train_correct_seqs)} correct traces from train split.")
    
    ensemble = EnsembleHMM(n_iter=100)
    ensemble.fit(train_correct_seqs, verbose=False)
    
    # Evaluate Natural Errors
    test_seqs = [item["tag_sequence"] for item in test_data]
    test_labels = [item.get("is_correct", False) for item in test_data]
    
    # NOTE: EnsembleHMM.score returns normalized log-likelihood in "ensemble_ll" 
    scores = [ensemble.score(seq)["ensemble_ll"] for seq in test_seqs]
    
    from sklearn.metrics import roc_auc_score
    import scipy.stats as stats
    
    valid_scores = [s for s in scores if s != float('-inf') and not np.isnan(s)]
    valid_labels = [test_labels[i] for i in range(len(scores)) if scores[i] != float('-inf') and not np.isnan(scores[i])]
    
    auc = roc_auc_score(valid_labels, valid_scores)
    r, p = stats.pointbiserialr(valid_labels, valid_scores)
    
    print("\n--- Priority 0+: Natural-Error ROC-AUC on Held-Out Traces ---")
    print(f"Test size: {len(valid_labels)} ({sum(valid_labels)} correct, {len(valid_labels)-sum(valid_labels)} incorrect)")
    print(f"AUC: {auc:.3f}")
    print(f"Point-Biserial r: {r:.3f} (p={p:.4f})")
    
    # Let's also do Priority 1: Tag-noise robustness (flip 10% tags)
    print("\n--- Priority 1: Tag-Noise Robustness ---")
    tag_options = ["CO", "VE", "PR", "EQ", "CA", "DV", "RF", "IN"]
    
    for noise_pct in [0.05, 0.10, 0.15, 0.20]:
        noisy_scores = []
        for seq in test_seqs:
            n_flips = int(len(seq) * noise_pct)
            noisy_seq = list(seq)
            if n_flips > 0 and len(seq) > 0:
                flip_indices = random.sample(range(len(seq)), min(n_flips, len(seq)))
                for idx in flip_indices:
                    noisy_seq[idx] = random.choice(tag_options)
            noisy_scores.append(ensemble.score(noisy_seq)["ensemble_ll"])
            
        valid_noisy_scores = [s for s in noisy_scores if s != float('-inf') and not np.isnan(s)]
        valid_noisy_labels = [test_labels[i] for i in range(len(noisy_scores)) if noisy_scores[i] != float('-inf') and not np.isnan(noisy_scores[i])]
        
        noisy_auc = roc_auc_score(valid_noisy_labels, valid_noisy_scores)
        print(f"Noise {int(noise_pct*100)}%: AUC = {noisy_auc:.3f}")

if __name__ == '__main__':
    main()
