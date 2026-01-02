"""
Visualize the 'Reasoning Manifold' using PCA/t-SNE on HMM State Distributions.
This script:
1. Loads the trained HMM (or Ensemble).
2. Computes the 'State Occupancy' vector for each sample (Correct vs. Incorrect).
   - This vector represents "How much time did this reasoning chain spend in State 0, State 1, etc."
3. Projects these high-dim vectors into 2D using PCA/t-SNE.
4. Generates a 'killer plot' showing the separation (or lack thereof).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib
import pandas as pd
from pathlib import Path
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hmm.ensemble import EnsembleHMM, TAG_TO_ID
from src.hmm.trainer import TAGS

def visualize_manifold(input_file, model_dir, output_dir):
    print(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load one representative model for state analysis (using the 5-state model is usually best for viz)
    # Ideally we'd use the ensemble, but visualizing 9 models * N states is hard.
    # We will pick the first 5-state model from the ensemble if available, or just the first model.
    
    try:
        ensemble = EnsembleHMM.load(model_dir)
        print("Ensemble loaded.")
        
        # Find a 5-component model
        target_model = None
        for m in ensemble.models:
            if m['model'].n_components == 5:
                target_model = m['model']
                print("Selected 5-state model for visualization.")
                break
        
        if target_model is None:
            target_model = ensemble.models[0]['model']
            print(f"Fallback: Selected {target_model.n_components}-state model.")
            
    except Exception as e:
        print(f"Could not load ensemble: {e}")
        return

    # Extract features: State Occupancy Counts
    # For each sample, we get the Viterbi path (sequence of states)
    # and count normalized frequency of each state.
    
    vectors = []
    labels = []
    
    print("Computing state representations...")
    for item in data:
        tag_seq = item.get("tag_sequence", [])
        if len(tag_seq) < 2:
            continue
            
        # Encode
        encoded = [TAG_TO_ID.get(t, 0) for t in tag_seq]
        X = np.array(encoded).reshape(-1, 1)
        
        # Decode to hidden states
        try:
            _, state_path = target_model.decode(X)
            
            # Count states
            counts = np.zeros(target_model.n_components)
            for state in state_path:
                counts[state] += 1
            
            # Normalize (to account for length differences)
            freqs = counts / len(state_path)
            
            vectors.append(freqs)
            labels.append("Correct" if item.get("is_correct") else "Incorrect")
            
        except Exception as e:
            continue

    X_data = np.array(vectors)
    y_labels = np.array(labels)
    
    print(f"Processed {len(X_data)} samples.")
    
    # 1. PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_data)
    
    # 2. t-SNE (better for clusters)
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(X_data)-1), random_state=42)
    X_tsne = tsne.fit_transform(X_data)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # PCA Plot
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], 
        hue=y_labels, 
        style=y_labels,
        palette={"Correct": "#2ecc71", "Incorrect": "#e74c3c"},
        alpha=0.7,
        s=80,
        ax=axes[0]
    )
    axes[0].set_title(f"PCA of Reasoning Structure (N={len(data)})", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("PC1 (Structural Variance)")
    axes[0].set_ylabel("PC2")
    axes[0].legend(title="Ground Truth")
    
    # t-SNE Plot
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1], 
        hue=y_labels, 
        style=y_labels,
        palette={"Correct": "#2ecc71", "Incorrect": "#e74c3c"},
        alpha=0.7,
        s=80,
        ax=axes[1]
    )
    axes[1].set_title(f"t-SNE Manifold of Reasoning Chains", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Dimension 1")
    axes[1].set_ylabel("Dimension 2")
    axes[1].legend(title="Ground Truth")
    
    plt.tight_layout()
    
    out_path = Path(output_dir) / "reasoning_manifold.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"✅ Manifold plot saved to {out_path}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="experiments/results/ensemble_150_final/gsm8k_qwen_150_corrected.json")
    parser.add_argument("--model-dir", default="experiments/results/ensemble_150_final/models")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()
    
    visualize_manifold(args.input, args.model_dir, args.output_dir)
