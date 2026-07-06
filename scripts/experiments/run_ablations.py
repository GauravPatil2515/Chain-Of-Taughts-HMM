"""
Run Ablation Studies
Vary HMM components and training data size.
"""

import json
import logging
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.hmm.trainer import train_and_save
from src.hmm.scorer import HMMScorer
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)

def evaluate_model(model_path, data_path, output_dir):
    scorer = HMMScorer(str(model_path))
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    scores = []
    labels = []
    
    for item in data:
        if "tag_sequence" not in item: continue
        score = scorer.score(item["tag_sequence"])
        scores.append(score.normalized_ll) # Use normalized LL
        labels.append(1 if item["is_correct"] else 0)
    
    if len(set(labels)) < 2:
        return 0.5 # Default AUC if only one class
        
    return roc_auc_score(labels, scores)

def run_ablations(tagged_data_path, output_dir, train=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # 1. Vary Hidden States
    print("Running Hidden States Ablation...")
    for n in [3, 5, 7, 9, 12]:
        model_path = output_dir / f"hmm_states_{n}.pkl"
        try:
            if train:
                train_and_save(
                    input_path=tagged_data_path,
                    output_path=str(model_path),
                    n_components=n,
                    correct_only=True
                )
            
            if model_path.exists():
                auc = evaluate_model(model_path, tagged_data_path, output_dir)
                results.append({"type": "states", "value": n, "auc": auc})
                print(f"States={n} -> AUC={auc:.4f}")
        except Exception as e:
            print(f"Failed states={n}: {e}")

    # 2. Vary Training Size
    print("\nRunning Training Size Ablation...")
    with open(tagged_data_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)
    correct_data = [d for d in full_data if d["is_correct"]]
    
    subset_sizes = [10, 25, 50, 75, 100, len(correct_data)]
    subset_sizes = sorted(list(set([s for s in subset_sizes if s <= len(correct_data)])))
    
    for size in subset_sizes:
        subset_path = output_dir / f"temp_subset_{size}.json"
        
        if train:
            subset = correct_data[:size]
            with open(subset_path, "w", encoding="utf-8") as f:
                json.dump(subset, f)
        
        model_path = output_dir / f"hmm_size_{size}.pkl"
        try:
            if train:
                train_and_save(
                    input_path=str(subset_path),
                    output_path=str(model_path),
                    n_components=5, 
                    correct_only=True 
                )
            
            if model_path.exists():
                auc = evaluate_model(model_path, tagged_data_path, output_dir)
                results.append({"type": "size", "value": size, "auc": auc})
                print(f"Size={size} -> AUC={auc:.4f}")
        except Exception as e:
             print(f"Failed size={size}: {e}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "ablation_results.csv", index=False)
    print("\nAblations Complete. Saved to ablation_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training, just evaluate")
    args = parser.parse_args()
    
    run_ablations("data/tagged/phase5_corrected.json", "experiments/results/ablations", train=not args.evaluate_only)
