"""
Organize paper folder for Overleaf upload
"""
import shutil
from pathlib import Path

# Source and destination
base = Path(".")
paper_dir = base / "paper"
figures_dir = paper_dir / "figures"
figures_dir.mkdir(exist_ok=True)

# Copy figures with proper names
figures_to_copy = [
    ("experiments/results/phase5_scaled/figures/baseline_correlation_comparison.png", "fig_correlation.png"),
    ("experiments/results/phase5_scaled/figures/baseline_score_distributions.png", "fig_distributions.png"),
    ("experiments/results/qwen_1.5b_test_15_qualitative/figures/transition_matrices.png", "fig_transition.png"),
    ("experiments/results/phase5_scaled/figures/baseline_metrics_comparison.png", "fig_metrics.png"),
]

for src, dst in figures_to_copy:
    src_path = base / src
    dst_path = figures_dir / dst
    if src_path.exists():
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {dst}")
    else:
        print(f"Missing: {src}")

# Clean up markdown files (keep only .tex and .bib)
for md_file in paper_dir.glob("*.md"):
    print(f"Markdown file (can be removed for Overleaf): {md_file.name}")

print("\nPaper folder ready for Overleaf!")
print("Files to zip:")
print("- main.tex")
print("- references.bib")
print("- figures/ (directory with all images)")
