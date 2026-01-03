"""
Evaluate a pre-trained HMM Ensemble on a NEW domain (Cross-Domain Test).
Does NOT retrain the model.
"""

import os
import argparse
import sys
import json
import numpy as np
from pathlib import Path
from scipy.stats import pointbiserialr

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tagging.rule_based_tagger import tag_cot_file
from src.hmm.ensemble import EnsembleHMM
# Import correctness logic. Since it's a script in root, we might need to load it dynamically or just copy logic.
# Easier to call it via subprocess if import is tricky, but let's try import first via sys.path
import post_process_correctness as ppc

def evaluate_cross_domain(input_file, model_dir, output_dir, dataset_name):
    print(f"🚀 Starting Cross-Domain Evaluation")
    print(f"Target Data: {input_file}")
    print(f"Using Model: {model_dir}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Tagging
    tagged_file = str(Path(output_dir) / f"{Path(input_file).stem}_tagged.json")
    print(f"\n[1/4] Tagging data...")
    if not os.path.exists(tagged_file):
        tag_cot_file(input_file, tagged_file)
    else:
        print("Tagged file exists, using it.")

    # 2. Correctness
    corrected_file = str(Path(output_dir) / f"{Path(input_file).stem}_corrected.json")
    print(f"\n[2/4] Verifying correctness for domain '{dataset_name}'...")
    
    # We can use the script via subprocess to avoid import complexity with main blocks
    # OR better, since I verified ppc imports above (if path is right), let's implement the loop here
    
    with open(tagged_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    correct_count = 0
    for item in data:
        # Determine dataset type for extraction
        ds_type = dataset_name if dataset_name else item.get('dataset', 'gsm8k')
        
        # Use simple fallback if ground truth missing
        gt = item.get('ground_truth', '')
        if not gt:
            gt = item.get('answer', '') # Some datasets use 'answer'
            
        cot = item.get('cot_raw', '')
        
        # Trust existing is_correct if valid
        if 'is_correct' in item:
            is_correct = item['is_correct']
            pred = item.get('predicted_answer', '')
        else:
            # Re-compute if missing
            pred = ppc.extract_answer(cot, dataset=ds_type)
            is_correct = ppc.check_correctness(pred, gt)
        
        item['predicted_answer'] = pred
        item['is_correct'] = is_correct
        if is_correct:
            correct_count += 1
            
    print(f"Accuracy on {dataset_name}: {correct_count}/{len(data)} ({correct_count/len(data)*100:.1f}%)")
    
    with open(corrected_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # 3. Load Model
    print(f"\n[3/4] Loading Pre-trained Ensemble...")
    try:
        ensemble = EnsembleHMM.load(model_dir)
        print(f"Loaded ensemble with {len(ensemble.models)} models.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 4. Score
    print(f"\n[4/4] Scoring & Evaluating...")
    scores = []
    labels = []
    
    for item in data:
        tag_seq = item.get("tag_sequence", [])
        res = ensemble.score(tag_seq)
        
        # Store score
        item['hmm_score'] = res
        
        scores.append(res['ensemble_ll'])
        labels.append(1 if item['is_correct'] else 0)
        
    # Save final results
    final_file = str(Path(output_dir) / f"{Path(input_file).stem}_final.json")
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    # Stats
    valid_indices = [i for i, s in enumerate(scores) if s != float('-inf')]
    valid_scores = [scores[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    
    if len(set(valid_labels)) < 2:
        print("⚠️ Cannot calculate correlation: All samples have same label (all correct or all incorrect).")
        return

    r, p = pointbiserialr(valid_labels, valid_scores)
    
    print("\n" + "="*50)
    print("CROSS-DOMAIN RESULTS")
    print("="*50)
    print(f"Domain: {dataset_name}")
    print(f"Model ID: {Path(model_dir).parent.name}")
    print(f"Correlation (r): {r:.4f}")
    print(f"P-value (p):     {p:.4f}")
    
    if p < 0.05:
        print("✅ Statistically Significant!")
    else:
        print("❌ Not Significant (Expected for cross-domain?)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="experiments/results/cross_domain")
    parser.add_argument("--dataset-name", default="arc_challenge")
    args = parser.parse_args()
    
    evaluate_cross_domain(args.input, args.model_dir, args.output_dir, args.dataset_name)
