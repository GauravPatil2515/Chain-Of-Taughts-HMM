"""
Dataset Manual Audit & Diagnostics Script for CoT-HMM
Inspects raw CoT text, tag sequences, structural entropy, and baseline score distributions
for randomly sampled correct and incorrect reasoning chains.
"""

import json
import random
import numpy as np
from pathlib import Path

def run_audit(dataset_path: str = "data/v2/gsm8k_500_proper_cv_scored.json", sample_size: int = 15, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"======================================================================")
    print(f"CoT-HMM DATASET AUDIT & DIAGNOSTIC REPORT")
    print(f"Dataset File: {dataset_path}")
    print(f"======================================================================\n")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    correct_samples = [d for d in data if d["is_correct"]]
    incorrect_samples = [d for d in data if not d["is_correct"]]
    
    print(f"Total Traces: {len(data)} | Correct: {len(correct_samples)} | Incorrect: {len(incorrect_samples)}\n")
    
    sampled_correct = random.sample(correct_samples, min(sample_size, len(correct_samples)))
    sampled_incorrect = random.sample(incorrect_samples, min(sample_size, len(incorrect_samples)))
    
    print("----------------------------------------------------------------------")
    print("1. AUDIT OF CORRECT REASONING TRACES (Sample of 15)")
    print("----------------------------------------------------------------------")
    for idx, item in enumerate(sampled_correct, 1):
        print(f"\n[Correct Sample #{idx}] ID: {item['sample_id']}")
        print(f"  Raw CoT Text: {repr(item.get('cot_raw', item.get('cot_text', ''))[:120])}...")
        print(f"  Tag Sequence: {item.get('tag_sequence')}")
        print(f"  OOF Ensemble Score : {item.get('oof_hmm_score', 0.0):.4f}")
        print(f"  OOF Margin Score   : {item.get('oof_plausibility_margin', 0.0):.4f}")
        print(f"  Structural Entropy : {item.get('structural_entropy', 0.0):.4f}")
        print(f"  Repetition Penalty : {item.get('repetition_penalty', 0.0):.4f}")
        
    print("\n----------------------------------------------------------------------")
    print("2. AUDIT OF INCORRECT REASONING TRACES (Sample of 15)")
    print("----------------------------------------------------------------------")
    for idx, item in enumerate(sampled_incorrect, 1):
        print(f"\n[Incorrect Sample #{idx}] ID: {item['sample_id']}")
        print(f"  Raw CoT Text: {repr(item.get('cot_raw', item.get('cot_text', ''))[:120])}...")
        print(f"  Tag Sequence: {item.get('tag_sequence')}")
        print(f"  OOF Ensemble Score : {item.get('oof_hmm_score', 0.0):.4f}")
        print(f"  OOF Margin Score   : {item.get('oof_plausibility_margin', 0.0):.4f}")
        print(f"  Structural Entropy : {item.get('structural_entropy', 0.0):.4f}")
        print(f"  Repetition Penalty : {item.get('repetition_penalty', 0.0):.4f}")
        
    print("\n----------------------------------------------------------------------")
    print("3. DIAGNOSTIC STATISTICAL SUMMARY")
    print("----------------------------------------------------------------------")
    
    correct_entropies = [d.get('structural_entropy', 0.0) for d in correct_samples]
    incorrect_entropies = [d.get('structural_entropy', 0.0) for d in incorrect_samples]
    
    correct_reps = [d.get('repetition_penalty', 0.0) for d in correct_samples]
    incorrect_reps = [d.get('repetition_penalty', 0.0) for d in incorrect_samples]
    
    correct_margins = [d.get('oof_plausibility_margin', 0.0) for d in correct_samples]
    incorrect_margins = [d.get('oof_plausibility_margin', 0.0) for d in incorrect_samples]
    
    print(f"Mean Structural Entropy - Correct  : {np.mean(correct_entropies):.4f} ± {np.std(correct_entropies):.4f}")
    print(f"Mean Structural Entropy - Incorrect: {np.mean(incorrect_entropies):.4f} ± {np.std(incorrect_entropies):.4f}")
    print(f"Mean Repetition Penalty - Correct  : {np.mean(correct_reps):.4f} ± {np.std(correct_reps):.4f}")
    print(f"Mean Repetition Penalty - Incorrect: {np.mean(incorrect_reps):.4f} ± {np.std(incorrect_reps):.4f}")
    print(f"Mean Plausibility Margin - Correct : {np.mean(correct_margins):.4f} ± {np.std(correct_margins):.4f}")
    print(f"Mean Plausibility Margin - Incorrect: {np.mean(incorrect_margins):.4f} ± {np.std(incorrect_margins):.4f}")
    print("======================================================================")

if __name__ == "__main__":
    run_audit()
