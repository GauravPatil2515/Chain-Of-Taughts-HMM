"""
Novel Analysis: "Difficulty Estimator".

Hypothesis: HMM Score negatively correlates with Problem Difficulty.
Proxy for Difficulty: Reasoning Length (Number of Steps).

Logic:
1. Extract 'Reasoning Length' (Number of tags) for each sample.
2. Extract 'HMM Plausibility Score'.
3. Correlate them.
4. Visualize with a Regression Plot.

If correlation is strong, the HMM can be used as an 'Unsupervised Difficulty Scorer'.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os
from pathlib import Path
from scipy.stats import pearsonr

def analyze_difficulty(input_file, output_dir):
    print(f"Starting Difficulty Analysis: Length vs Score...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    lengths = []
    scores = []
    labels = []
    
    for item in data:
        tag_seq = item.get("tag_sequence", [])
        if len(tag_seq) < 2: continue
        
        # Check if score exists
        if 'hmm_score' not in item: continue
        score = item['hmm_score'].get('ensemble_ll', float('-inf'))
        
        if score == float('-inf'): continue
        
        lengths.append(len(tag_seq))
        scores.append(score)
        labels.append("Correct" if item.get('is_correct') else "Incorrect")
        
    # Analysis
    lengths = np.array(lengths)
    scores = np.array(scores)
    
    corr, p_val = pearsonr(lengths, scores)
    
    print(f"\nResults:")
    print(f"Correlation (Length vs Score): {corr:.4f}")
    print(f"P-value:                     {p_val:.4e}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=lengths, y=scores, hue=labels, palette={"Correct": "#2ecc71", "Incorrect": "#e74c3c"}, alpha=0.6)
    sns.regplot(x=lengths, y=scores, scatter=False, color="#34495e", line_kws={"linestyle": "--"})
    
    plt.title(f"Difficulty Estimator: Structure vs Length (r={corr:.2f})", fontsize=14, fontweight='bold')
    plt.xlabel("Reasoning Length (Proxy for Difficulty)")
    plt.ylabel("Structural Plausibility Score")
    plt.grid(alpha=0.3)
    
    out_path = Path(output_dir) / "difficulty_estimator.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")
    
    if p_val < 0.05 and abs(corr) > 0.2:
        print("Verified: HMM Score is a valid Difficulty Proxy!")
    else:
        print("Result: Weak correlation. Structure is independent of length (Robustness?).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="experiments/results/ensemble_150_final/gsm8k_qwen_150_final.json")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()
    
    analyze_difficulty(args.input, args.output_dir)
