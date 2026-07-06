
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_ablations(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Hidden States
    states_df = df[df["type"] == "states"]
    if not states_df.empty:
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=states_df, x="value", y="auc", marker="o", linewidth=2.5)
        plt.title("HMM Performance vs. Number of Hidden States", fontsize=14, fontweight='bold')
        plt.xlabel("Number of Hidden States (N)", fontsize=12)
        plt.ylabel("AUC Score", fontsize=12)
        plt.ylim(0.40, 0.65)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "fig_ablation_states.png", dpi=300)
        plt.close()
        print("Saved fig_ablation_states.png")

    # 2. Training Size
    size_df = df[df["type"] == "size"]
    if not size_df.empty:
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=size_df, x="value", y="auc", marker="o", linewidth=2.5, color='orange')
        plt.title("HMM Performance vs. Training Data Size", fontsize=14, fontweight='bold')
        plt.xlabel("Number of Correct Training Samples", fontsize=12)
        plt.ylabel("AUC Score", fontsize=12)
        plt.axhline(0.5, color='gray', linestyle='--', label='Random Chance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "fig_ablation_size.png", dpi=300)
        plt.close()
        print("Saved fig_ablation_size.png")

if __name__ == "__main__":
    plot_ablations("experiments/results/ablations/ablation_results.csv", "paper/figures")
