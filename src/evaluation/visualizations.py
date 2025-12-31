"""
Visualization Functions
Generate plots for HMM analysis and results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import joblib


TAGS = ["PR", "DV", "EQ", "CA", "IN", "RF", "VE", "CO"]


def plot_transition_matrix(
    model_path: str,
    output_path: str,
    figsize: tuple = (10, 8)
):
    """Plot HMM transition matrix as heatmap."""
    model = joblib.load(model_path)
    trans = model.transmat_
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        trans,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[f"S{i}" for i in range(model.n_components)],
        yticklabels=[f"S{i}" for i in range(model.n_components)],
        ax=ax,
        vmin=0,
        vmax=1,
    )
    
    ax.set_xlabel("To State", fontsize=12)
    ax.set_ylabel("From State", fontsize=12)
    ax.set_title("HMM Transition Matrix", fontsize=14)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved transition matrix to {output_path}")


def plot_emission_matrix(
    model_path: str,
    output_path: str,
    figsize: tuple = (12, 6)
):
    """Plot HMM emission matrix as heatmap."""
    model = joblib.load(model_path)
    emit = model.emissionprob_
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        emit,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=TAGS,
        yticklabels=[f"State {i}" for i in range(model.n_components)],
        ax=ax,
        vmin=0,
        vmax=1,
    )
    
    ax.set_xlabel("Observed Tag", fontsize=12)
    ax.set_ylabel("Hidden State", fontsize=12)
    ax.set_title("HMM Emission Probabilities", fontsize=14)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved emission matrix to {output_path}")


def plot_score_distribution(
    scored_data_path: str,
    output_path: str,
    figsize: tuple = (10, 6)
):
    """Plot distribution of HMM scores for correct vs incorrect CoTs."""
    with open(scored_data_path, "r") as f:
        data = json.load(f)
    
    scores_correct = []
    scores_incorrect = []
    
    for item in data:
        ll = item.get("hmm_scores", {}).get("log_likelihood", None)
        if ll is not None and ll > float('-inf'):
            if item.get("is_correct", False):
                scores_correct.append(ll)
            else:
                scores_incorrect.append(ll)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if scores_correct:
        ax.hist(scores_correct, bins=20, alpha=0.6, label=f"Correct (n={len(scores_correct)})", color="green")
    if scores_incorrect:
        ax.hist(scores_incorrect, bins=20, alpha=0.6, label=f"Incorrect (n={len(scores_incorrect)})", color="red")
    
    ax.axvline(np.mean(scores_correct) if scores_correct else 0, 
               color="green", linestyle="--", linewidth=2, label="Correct Mean")
    ax.axvline(np.mean(scores_incorrect) if scores_incorrect else 0,
               color="red", linestyle="--", linewidth=2, label="Incorrect Mean")
    
    ax.set_xlabel("Log-Likelihood Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of HMM Scores by Answer Correctness", fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved score distribution to {output_path}")


def plot_model_comparison(
    model_scores: Dict[str, Dict],
    output_path: str,
    figsize: tuple = (12, 6)
):
    """
    Plot comparison of metrics across models.
    
    Args:
        model_scores: Dict mapping model name to {accuracy, avg_score, avg_entropy}
    """
    models = list(model_scores.keys())
    accuracies = [model_scores[m].get("accuracy", 0) * 100 for m in models]
    scores = [model_scores[m].get("avg_score", 0) for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy
    axes[0].bar(models, accuracies, color="steelblue", alpha=0.8)
    axes[0].set_ylabel("Accuracy (%)", fontsize=12)
    axes[0].set_title("Answer Accuracy by Model", fontsize=14)
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=10)
    
    # HMM Score
    axes[1].bar(models, scores, color="coral", alpha=0.8)
    axes[1].set_ylabel("Avg Log-Likelihood", fontsize=12)
    axes[1].set_title("Average HMM Score by Model", fontsize=14)
    for i, v in enumerate(scores):
        axes[1].text(i, v + 0.5, f"{v:.1f}", ha='center', fontsize=10)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved model comparison to {output_path}")


def plot_tag_distribution(
    tagged_data_path: str,
    output_path: str,
    figsize: tuple = (10, 6)
):
    """Plot distribution of tags in the dataset."""
    with open(tagged_data_path, "r") as f:
        data = json.load(f)
    
    tag_counts = {tag: 0 for tag in TAGS}
    
    for item in data:
        for tag in item.get("tag_sequence", []):
            if tag in tag_counts:
                tag_counts[tag] += 1
    
    fig, ax = plt.subplots(figsize=figsize)
    
    tags = list(tag_counts.keys())
    counts = list(tag_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(tags)))
    
    bars = ax.bar(tags, counts, color=colors, edgecolor="black", linewidth=0.5)
    
    ax.set_xlabel("Tag", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Reasoning Step Tags", fontsize=14)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', fontsize=10)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved tag distribution to {output_path}")


if __name__ == "__main__":
    print("Visualization functions loaded. Import and call as needed.")
