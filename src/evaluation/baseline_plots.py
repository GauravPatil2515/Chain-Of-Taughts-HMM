"""
Baseline Comparison Visualizations
Generate plots comparing HMM against baseline methods.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


def plot_score_distributions_comparison(
    report_path: str,
    output_path: str,
    figsize: tuple = (14, 10)
):
    """Plot score distributions for all methods, separated by correctness."""
    with open(report_path, "r") as f:
        report = json.load(f)
    
    df = pd.DataFrame(report["all_scores"])
    
    # Get score columns
    score_cols = [col for col in df.columns if col.endswith("_score") or col == "hmm_score"]
    
    # Remove infinite values
    for col in score_cols:
        df = df[np.isfinite(df[col])]
    
    # Create subplots
    n_methods = len(score_cols)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_methods > 1 else [axes]
    
    for idx, col in enumerate(score_cols):
        ax = axes[idx]
        
        correct_scores = df[df["is_correct"] == True][col]
        incorrect_scores = df[df["is_correct"] == False][col]
        
        ax.hist(correct_scores, alpha=0.6, label="Correct", bins=10, color="green", edgecolor="black")
        ax.hist(incorrect_scores, alpha=0.6, label="Incorrect", bins=10, color="red", edgecolor="black")
        
        method_name = col.replace("_score", "").replace("_", " ").title()
        ax.set_title(method_name, fontsize=10, fontweight='bold')
        ax.set_xlabel("Score", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(score_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Score Distributions: Correct vs Incorrect CoTs", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved score distributions to {output_path}")


def plot_correlation_comparison(
    report_path: str,
    output_path: str,
    figsize: tuple = (10, 6)
):
    """Bar plot comparing correlation coefficients."""
    with open(report_path, "r") as f:
        report = json.load(f)
    
    corr_df = pd.DataFrame(report["correlation_analysis"])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by correlation
    corr_df = corr_df.sort_values("correlation", ascending=True)
    
    colors = ['green' if x > 0 else 'red' for x in corr_df['correlation']]
    
    ax.barh(corr_df["method"], corr_df["correlation"], color=colors, alpha=0.7, edgecolor="black")
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    
    ax.set_xlabel("Correlation with Correctness", fontsize=12, fontweight='bold')
    ax.set_ylabel("Method", fontsize=12, fontweight='bold')
    ax.set_title("Baseline Comparison: Correlation with Ground Truth", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (method, corr) in enumerate(zip(corr_df["method"], corr_df["correlation"])):
        ax.text(corr + 0.02, i, f"{corr:.3f}", va='center', fontsize=9)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation comparison to {output_path}")


def plot_metrics_comparison(
    report_path: str,
    output_path: str,
    figsize: tuple = (12, 8)
):
    """Grouped bar plot for classification metrics."""
    with open(report_path, "r") as f:
        report = json.load(f)
    
    clf_df = pd.DataFrame(report["classification_metrics"])
    
    # Select metrics to plot
    metrics = ["accuracy", "precision", "recall", "f1"]
    clf_df = clf_df.sort_values("f1", ascending=False)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(clf_df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2 + 0.5) * width
        ax.bar(x + offset, clf_df[metric], width, label=metric.title(), alpha=0.8)
    
    ax.set_xlabel("Method", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title("Classification Metrics Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(clf_df["method"], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved metrics comparison to {output_path}")


def plot_method_correlation_heatmap(
    report_path: str,
    output_path: str,
    figsize: tuple = (10, 8)
):
    """Heatmap showing correlation between different scoring methods."""
    with open(report_path, "r") as f:
        report = json.load(f)
    
    rank_corr = pd.DataFrame(report["rank_correlation"])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Clean up column names
    rank_corr.columns = [col.replace("_score", "").replace("_", " ").title() for col in rank_corr.columns]
    rank_corr.index = [idx.replace("_score", "").replace("_", " ").title() for idx in rank_corr.index]
    
    sns.heatmap(
        rank_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"label": "Spearman Correlation"}
    )
    
    ax.set_title("Inter-Method Rank Correlation", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation heatmap to {output_path}")


def plot_score_scatter(
    report_path: str,
    output_path: str,
    method1: str = "hmm_score",
    method2: str = "combined_score",
    figsize: tuple = (8, 8)
):
    """Scatter plot comparing two scoring methods."""
    with open(report_path, "r") as f:
        report = json.load(f)
    
    df = pd.DataFrame(report["all_scores"])
    
    # Filter infinite values
    valid_mask = np.isfinite(df[method1]) & np.isfinite(df[method2])
    df = df[valid_mask]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    correct = df[df["is_correct"] == True]
    incorrect = df[df["is_correct"] == False]
    
    ax.scatter(correct[method1], correct[method2], 
               alpha=0.6, s=100, c='green', label='Correct', edgecolors='black', linewidth=0.5)
    ax.scatter(incorrect[method1], incorrect[method2], 
               alpha=0.6, s=100, c='red', label='Incorrect', edgecolors='black', linewidth=0.5)
    
    # Add correlation line
    if len(df) > 1:
        z = np.polyfit(df[method1], df[method2], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[method1].min(), df[method1].max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=2, label='Trend')
    
    method1_name = method1.replace("_score", "").replace("_", " ").title()
    method2_name = method2.replace("_score", "").replace("_", " ").title()
    
    ax.set_xlabel(f"{method1_name} Score", fontsize=12, fontweight='bold')
    ax.set_ylabel(f"{method2_name} Score", fontsize=12, fontweight='bold')
    ax.set_title(f"{method1_name} vs {method2_name}", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved scatter plot to {output_path}")


def generate_all_comparison_plots(report_path: str, output_dir: str):
    """Generate all comparison visualizations."""
    print("="*60)
    print("Generating Baseline Comparison Plots")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Score distributions...")
    plot_score_distributions_comparison(
        report_path,
        output_dir / "baseline_score_distributions.png"
    )
    
    print("\n2. Correlation comparison...")
    plot_correlation_comparison(
        report_path,
        output_dir / "baseline_correlation_comparison.png"
    )
    
    print("\n3. Classification metrics...")
    plot_metrics_comparison(
        report_path,
        output_dir / "baseline_metrics_comparison.png"
    )
    
    print("\n4. Method correlation heatmap...")
    plot_method_correlation_heatmap(
        report_path,
        output_dir / "baseline_method_correlation.png"
    )
    
    print("\n5. Score scatter plots...")
    plot_score_scatter(
        report_path,
        output_dir / "baseline_hmm_vs_combined.png",
        method1="hmm_score",
        method2="combined_score"
    )
    
    print("\n" + "="*60)
    print(f"All plots saved to {output_dir}")
    print("="*60)


if __name__ == "__main__":
    report_path = "experiments/results/baseline_comparison.json"
    output_dir = "experiments/results/figures/baseline"
    
    generate_all_comparison_plots(report_path, output_dir)
