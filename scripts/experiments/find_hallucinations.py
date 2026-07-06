"""
Find 'Structural Hallucinations'.
Cases where the HMM plausibility score is HIGH, but the answer is INCORRECT.
"""

import json
import argparse
import numpy as np
from pathlib import Path

def find_hallucinations(input_file, top_k=2):
    print(f"Analyzing {input_file}...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Filter for valid entries (have score and are incorrect)
    candidates = []
    
    # Collect all scores to calculate percentiles
    all_scores = []
    for item in data:
        if 'hmm_score' in item and 'ensemble_ll' in item['hmm_score']:
            score = item['hmm_score']['ensemble_ll']
            if score != float('-inf'):
                all_scores.append(score)
                
    if not all_scores:
        print("No valid scores found.")
        return

    score_threshold = np.percentile(all_scores, 75) # Top 25% plausibility
    print(f"Score Threshold (75th percentile): {score_threshold:.4f}")
    
    for item in data:
        if not item.get('is_correct', False) and 'hmm_score' in item:
             score = item['hmm_score'].get('ensemble_ll', float('-inf'))
             if score > score_threshold:
                 candidates.append((score, item))
                 
    # Sort by score (highest first)
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\nFound {len(candidates)} Structural Hallucinations (High Score + Incorrect)")
    print("-" * 60)
    
    for i, (score, item) in enumerate(candidates[:top_k]):
        print(f"Example #{i+1} | Score: {score:.4f}")
        print(f"Question: {item.get('question', '')[:100]}...")
        print(f"Predicted: {item.get('predicted_answer', '')} | Ground Truth: {item.get('ground_truth', '')}")
        print(f"Tag Sequence: {item.get('tag_sequence', [])}")
        print(f"CoT Snippet: {item.get('cot_raw', '')[:150]}...")
        print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to _final.json file from pipeline")
    args = parser.parse_args()
    
    find_hallucinations(args.input)
