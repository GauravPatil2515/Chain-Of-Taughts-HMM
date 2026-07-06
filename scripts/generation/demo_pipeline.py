"""
Demo Pipeline Script
Runs the complete CoT-HMM workflow on synthetic demo data.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.tagging.rule_based_tagger import tag_cot_file
from src.hmm.trainer import train_and_save
from src.hmm.scorer import score_cot_file
from src.evaluation.visualizations import (
    plot_transition_matrix,
    plot_emission_matrix,
    plot_score_distribution
)

def setup_dirs():
    """Ensure all directories exist."""
    dirs = [
        "data/cot_outputs",
        "data/tagged",
        "models",
        "experiments/results/figures"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def main():
    print("="*60)
    print("CoT-HMM Demo Pipeline")
    print("="*60)
    
    setup_dirs()
    
    # Paths
    raw_cot_path = "data/cot_outputs/demo_cot.json"
    tagged_cot_path = "data/tagged/demo_tagged.json"
    model_path = "models/demo_hmm.pkl"
    scored_cot_path = "experiments/results/demo_scored.json"
    
    # 1. Tagging
    print(f"\n[1/4] Tagging CoTs from {raw_cot_path}...")
    if not os.path.exists(raw_cot_path):
        print(f"Error: {raw_cot_path} not found. Please verify setup.")
        return
        
    stats = tag_cot_file(raw_cot_path, tagged_cot_path)
    
    # 2. Training
    print(f"\n[2/4] Training HMM on correct CoTs...")
    # Using fewer components for small demo data
    train_and_save(
        input_path=tagged_cot_path,
        output_path=model_path,
        n_components=4, 
        correct_only=True
    )
    
    # 3. Scoring
    print(f"\n[3/4] Scoring CoTs...")
    score_cot_file(model_path, tagged_cot_path, scored_cot_path)
    
    # 4. Visualization
    print(f"\n[4/4] Generating Visualizations...")
    
    plot_transition_matrix(
        model_path, 
        "experiments/results/figures/demo_transition_matrix.png"
    )
    
    plot_emission_matrix(
        model_path,
        "experiments/results/figures/demo_emission_matrix.png"
    )
    
    plot_score_distribution(
        scored_cot_path,
        "experiments/results/figures/demo_score_dist.png"
    )
    
    print("\n" + "="*60)
    print("Demo Complete! Check experiments/results/ for outputs.")
    print("="*60)

if __name__ == "__main__":
    main()
