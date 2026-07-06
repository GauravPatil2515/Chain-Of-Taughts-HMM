"""
Pipeline to run Ensemble HMM experiment.
1. Tags raw CoT outputs.
2. Checks correctness (optional if already done).
3. Trains Ensemble HMM.
4. Evaluates and reports metrics.
"""

import os
import json
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.tagging.rule_based_tagger import tag_cot_file as tag_file
from src.hmm.ensemble import EnsembleHMM
from src.hmm.trainer import load_tagged_data

def run_pipeline(input_file, output_dir, model_name="ensemble"):
    print(f"🚀 Starting Ensemble Pipeline for {input_file}")
    
    # 1. Tagging
    tagged_file = str(Path(output_dir) / f"{Path(input_file).stem}_tagged.json")
    print(f"\n[1/4] Tagging data to {tagged_file}...")
    try:
        tag_file(input_file, tagged_file)
    except Exception as e:
        print(f"Tagging failed: {e}")
        return

    # 2. Correctness Check
    corrected_file = str(Path(output_dir) / f"{Path(input_file).stem}_corrected.json")
    print(f"\n[2/4] Verifying correctness to {corrected_file}...")
    try:
        # Assuming verify_correctness logic is available or we use the existing script logic
        # For now, we reuse the existing script's main logic or call it via subprocess if import is hard
        # But let's try to import the function if possible. 
        # Looking at file list, 'post_process_correctness.py' is in scripts/analysis.
        # I imported process_file from it above (assuming I can modify it or it exists)
        # If not, I'll use os.system as a fallback
        ret = os.system(f"python scripts/analysis/post_process_correctness.py --input {tagged_file} --output {corrected_file}")
        if ret != 0:
            raise Exception("Correctness check failed")
    except Exception as e:
        print(f"Correctness check failed: {e}")
        return

    # 3. Train Ensemble
    print(f"\n[3/4] Training Ensemble HMM...")
    sequences, correctness = load_tagged_data(corrected_file)
    
    # Filter for correct only for training
    train_sequences = [s for s, c in zip(sequences, correctness) if c]
    print(f"Training on {len(train_sequences)} correct samples out of {len(sequences)} total.")
    
    if len(train_sequences) < 5:
        print("⚠️  Not enough correct samples to train! (Need at least 5)")
        print("Skipping training.")
        return

    ensemble = EnsembleHMM(component_configs=[3, 5, 8], n_seeds_per_config=3)
    ensemble.fit(train_sequences)
    
    model_dir = str(Path(output_dir) / "models")
    ensemble.save(model_dir)

    # 4. Evaluation
    print(f"\n[4/4] Evaluating...")
    scores = []
    labels = []
    
    corrected_data = [] # We need to reload or keep valid objects mapped to sequences.
    # Actually, load_tagged_data only returns lists. We need the original dicts to save results.
    with open(corrected_file, "r", encoding="utf-8") as f:
         corrected_data = json.load(f)

    for i, item in enumerate(corrected_data):
        # We assume order is preserved, which load_tagged_data does (it iterates sequentially).
        # We need to filter exactly as load_tagged_data does to match indices?
        # load_tagged_data filters len < 2.
        tag_seq = item.get("tag_sequence", [])
        if len(tag_seq) < 2:
             item['hmm_score'] = {'ensemble_ll': float('-inf')}
             scores.append(float('-inf'))
             labels.append(1 if item.get('is_correct') else 0)
             continue

        res = ensemble.score(tag_seq)
        item['hmm_score'] = res
        scores.append(res['ensemble_ll'])
        labels.append(1 if item.get('is_correct') else 0)

    # Save final results
    final_file = str(Path(output_dir) / f"{Path(input_file).stem}_final.json")
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(corrected_data, f, indent=2)
    print(f"Saved scored results to {final_file}")
        
    # Calculate Correlation using provided baseline logic or simple scipy
    from scipy.stats import pointbiserialr
    
    # Filter out -inf
    valid_indices = [i for i, s in enumerate(scores) if s != float('-inf')]
    valid_scores = [scores[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    
    if len(set(valid_labels)) < 2:
        print("Cannot calculate correlation: All samples are same class (all correct or all incorrect).")
    else:
        r, p = pointbiserialr(valid_labels, valid_scores)
        print(f"\nRESULTS:")
        print(f"Correlation (r): {r:.4f}")
        print(f"P-value (p):     {p:.4f}")
        
    print("\n✅ Pipeline Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CoT outputs")
    parser.add_argument("--output-dir", default="experiments/ensemble_run", help="Output directory")
    args = parser.parse_args()
    
    run_pipeline(args.input, args.output_dir)
