"""
Qualitative Analysis Module
Generates interpretable visualizations and example extracts for the paper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

TAG_ORDER = ["PR", "IN", "EQ", "CA", "VE", "RF", "CO", "DV"]

def compute_transition_matrix(tag_sequences: List[List[str]], tags: List[str] = TAG_ORDER) -> pd.DataFrame:
    """Compute empirical transition matrix P(t_next | t_curr)."""
    tag_to_idx = {t: i for i, t in enumerate(tags)}
    n_tags = len(tags)
    matrix = np.zeros((n_tags, n_tags))
    
    for seq in tag_sequences:
        if not seq or len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            curr_t, next_t = seq[i], seq[i+1]
            if curr_t in tag_to_idx and next_t in tag_to_idx:
                matrix[tag_to_idx[curr_t], tag_to_idx[next_t]] += 1
                
    # Normalize
    row_sums = matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    norm_matrix = np.divide(matrix, row_sums, where=row_sums!=0)
    
    return pd.DataFrame(norm_matrix, index=tags, columns=tags)

def plot_transition_matrices(data: List[Dict], output_path: str):
    """Plot correct vs incorrect transition matrices."""
    # Split data
    correct_seqs = [d['tag_sequence'] for d in data if d.get('is_correct', False)]
    incorrect_seqs = [d['tag_sequence'] for d in data if not d.get('is_correct', False)]
    
    if not correct_seqs or not incorrect_seqs:
        print("Warning: Not enough data to plot transition matrices split.")
        return

    trans_correct = compute_transition_matrix(correct_seqs)
    trans_incorrect = compute_transition_matrix(incorrect_seqs)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(trans_correct, annot=True, fmt=".2f", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Correct Solutions: Tag Transitions", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Current Tag")
    axes[0].set_xlabel("Next Tag")
    
    sns.heatmap(trans_incorrect, annot=True, fmt=".2f", cmap="Reds", ax=axes[1], cbar=False)
    axes[1].set_title("Incorrect Solutions: Tag Transitions", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Next Tag")
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved transition comparison to {output_path}")

def extract_qualitative_examples(df: pd.DataFrame, output_path: str, n=3):
    """
    Extract interesting examples:
    1. True Positives (High Score, Correct)
    2. False Positives (High Score, Incorrect) - Structure mimics verification but logic fails
    3. True Negatives (Low Score, Incorrect)
    """
    results = []
    
    # Sort by HMM Score
    df_sorted = df.sort_values("hmm_score", ascending=False)
    
    # 1. High Score + Correct (Model Confident & Right)
    top_correct = df_sorted[df_sorted["is_correct"] == True].head(n)
    for _, row in top_correct.iterrows():
        results.append({
            "type": "High Confidence Correct",
            "id": row["id"],
            "score": row["hmm_score"],
            "tags": str(row.get("tag_sequence", [])),
            "text": row.get("cot_raw", "")[:200] + "..."
        })
        
    # 2. High Score + Incorrect (The "Dangerous" Ones - Hallucination with good structure)
    top_incorrect = df_sorted[df_sorted["is_correct"] == False].head(n)
    for _, row in top_incorrect.iterrows():
        results.append({
            "type": "High Confidence Incorrect (Structural Hallucination)",
            "id": row["id"],
            "score": row["hmm_score"],
            "tags": str(row.get("tag_sequence", [])),
            "text": row.get("cot_raw", "")[:200] + "..."
        })
        
    # 3. Low Score + Incorrect (Easy Rejects)
    bottom_incorrect = df_sorted[df_sorted["is_correct"] == False].tail(n)
    for _, row in bottom_incorrect.iterrows():
        results.append({
            "type": "Low Confidence Incorrect (Obvious Fail)",
            "id": row["id"],
            "score": row["hmm_score"],
            "tags": str(row.get("tag_sequence", [])),
            "text": row.get("cot_raw", "")[:200] + "..."
        })
        
    # Save as Markdown
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Qualitative Examples Analysis\n\n")
        
        for item in results:
            f.write(f"## {item['type']}\n")
            f.write(f"- **ID:** {item['id']}\n")
            f.write(f"- **HMM Score:** {item['score']:.3f}\n")
            f.write(f"- **Tags:** `{item['tags']}`\n")
            f.write(f"- **Snippet:** {item['text']}\n\n")
            
    print(f"Saved qualitative examples to {output_path}")
