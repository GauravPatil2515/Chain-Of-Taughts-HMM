"""
Novel Analysis Script: "Cognitive Fingerprint" & "Structural Entropy".

1. Cognitive Fingerprint:
   - Visualizes the Transition Matrix of the HMM (The "Brain Map").
   - Compares it (if possible) or just showcases the learned structure.
   
2. Structural Entropy:
   - Computes the Entropy of the Hidden State distribution for each step.
   - Correlates "Mean Entropy" with Correctness.
   - Hypothesis: Confused models (Incorrect) have Higher Entropy (Uncertainty in structure).

 """

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os
from pathlib import Path
from scipy.stats import entropy

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hmm.ensemble import EnsembleHMM, TAG_TO_ID
from src.hmm.trainer import ID_TO_TAG

def analyze_novelty(input_file, model_dir, output_dir):
    print(f"Starting Novelty Analysis: Fingerprints & Entropy")
    
    # Load Data & Model
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    try:
        ensemble = EnsembleHMM.load(model_dir)
        # Pick a representative 5-state model for visualization
        model = next((m['model'] for m in ensemble.models if m['model'].n_components == 5), ensemble.models[0]['model'])
        print("Loaded representative HMM for analysis.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- PART 1: COGNITIVE FINGERPRINT (Transition Matrix) ---
    print("\n[1/2] Generating Cognitive Fingerprint...")
    
    trans_mat = model.transmat_
    n_states = model.n_components
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(trans_mat, annot=True, cmap="Blues", fmt=".2f", cbar=True)
    plt.title(f"Cognitive Fingerprint (Transition Dynamics, N={n_states})", fontsize=14, fontweight='bold')
    plt.xlabel("To State (Next Step)", fontsize=11)
    plt.ylabel("From State (Current Step)", fontsize=11)
    
    # Try to map states to dominant tags if possible (Heuristic)
    # We check Emission Matrix to label axes
    emission_mat = model.emissionprob_
    state_labels = []
    for i in range(n_states):
        top_tag_idx = np.argmax(emission_mat[i])
        prob = emission_mat[i, top_tag_idx]
        tag = ID_TO_TAG.get(top_tag_idx, "UNK")
        state_labels.append(f"S{i} ({tag})")
        
    plt.xticks(np.arange(n_states) + 0.5, state_labels, rotation=45)
    plt.yticks(np.arange(n_states) + 0.5, state_labels, rotation=0)
    
    plt.tight_layout()
    fingerprint_path = Path(output_dir) / "cognitive_fingerprint.png"
    plt.savefig(fingerprint_path, dpi=300)
    print(f"Saved Fingerprint to {fingerprint_path}")
    
    # --- PART 2: STRUCTURAL ENTROPY ---
    print("\n[2/2] Calculating Structural Entropy...")
    
    correct_entropies = []
    incorrect_entropies = []
    
    for item in data:
        tag_seq = item.get("tag_sequence", [])
        if len(tag_seq) < 2: continue
        
        # Encode
        X = np.array([TAG_TO_ID.get(t, 0) for t in tag_seq]).reshape(-1, 1)
        
        # Get Posterior Probabilities of states for each step
        # posterior shape: (n_samples, n_components)
        try:
            posterior = model.predict_proba(X)
            
            # Calculate Entropy per step: H(t) = - sum(p * log(p))
            # We want scalar entropy for the whole chain. Mean Entropy?
            step_entropies = entropy(posterior, axis=1) # Entropy of distribution across states
            mean_entropy = np.mean(step_entropies)
            
            item['structural_entropy'] = mean_entropy
            
            if item.get('is_correct'):
                correct_entropies.append(mean_entropy)
            else:
                incorrect_entropies.append(mean_entropy)
                
        except Exception:
            continue
            
    # Visualize Entropy Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(correct_entropies, fill=True, color="#2ecc71", label=f"Correct (Avg: {np.mean(correct_entropies):.2f})", warn_singular=False)
    sns.kdeplot(incorrect_entropies, fill=True, color="#e74c3c", label=f"Incorrect (Avg: {np.mean(incorrect_entropies):.2f})", warn_singular=False)
    
    plt.title("Structural Entropy Distribution (Uncertainty)", fontsize=14, fontweight='bold')
    plt.xlabel("Mean Entropy (Lower = More Confident Structure)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    
    entropy_path = Path(output_dir) / "structural_entropy.png"
    plt.savefig(entropy_path, dpi=300)
    print(f"Saved Entropy Plot to {entropy_path}")
    
    # Statistical Test for Entropy
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(correct_entropies, incorrect_entropies, equal_var=False)
    print(f"\nEntropy Difference Stats:")
    print(f"Correct Mean:   {np.mean(correct_entropies):.4f}")
    print(f"Incorrect Mean: {np.mean(incorrect_entropies):.4f}")
    print(f"P-value:        {p_val:.4e}")
    if p_val < 0.05:
        print("X-Factor Verified: Statistically Significant!")
    else:
        print("Not Significant.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="experiments/results/ensemble_150_final/gsm8k_qwen_150_final.json")
    parser.add_argument("--model-dir", default="experiments/results/ensemble_150_final/models")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()
    
    analyze_novelty(args.input, args.model_dir, args.output_dir)
