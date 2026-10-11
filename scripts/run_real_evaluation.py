#!/usr/bin/env python3
"""
Run Leak-Free Evaluation on REAL GSM8K Data
This replaces the synthetic evaluation with genuine benchmark results.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_leakfree_evaluation import (
    run_leakfree_cross_validation,
    evaluate_statistical_performance,
    simulate_3tier_routing,
    generate_leak_free_report
)


def main():
    # Input: real generated traces
    dataset_path = "data/real/gsm8k_qwen_300_real_traces.json"
    output_dir = Path("experiments/results/real_data_eval")
    
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please run the Colab generation script first and download the results.")
        sys.exit(1)
    
    print(f"Loading real dataset: {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total traces: {len(data)}")
    correct = sum(1 for d in data if d['is_correct'])
    print(f"Correct: {correct}, Incorrect: {len(data)-correct}")
    print(f"Accuracy: {correct/len(data)*100:.1f}%")
    
    # Tag the data first (need to apply rule-based tagger)
    print("\nTagging CoT traces...")
    from src.tagging.rule_based_tagger import tag_cot_file
    
    tagged_path = output_dir / "tagged.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    tag_cot_file(dataset_path, str(tagged_path))
    
    # Now run leak-free CV
    print("\nRunning leak-free 5-fold cross-validation...")
    scored_data, cv_info = run_leakfree_cross_validation(
        str(tagged_path), 
        n_splits=5, 
        random_state=42
    )
    
    # Save scored dataset
    cv_dataset_path = output_dir / "cv_scored.json"
    with open(cv_dataset_path, 'w') as f:
        json.dump(scored_data, f, indent=2)
    
    # Evaluate statistical performance
    print("\nEvaluating statistical performance...")
    eval_results = evaluate_statistical_performance(scored_data)
    
    # Simulate 3-tier routing
    print("\nSimulating 3-tier compute routing...")
    routing_results = simulate_3tier_routing(scored_data)
    
    # Save evaluation JSON
    eval_json_path = output_dir / "evaluation_results_real.json"
    with open(eval_json_path, 'w') as f:
        json.dump({
            "cross_validation": cv_info,
            "metrics": eval_results,
            "compute_routing": routing_results
        }, f, indent=2)
    
    # Generate human-readable report
    print("\nGenerating report...")
    report_text = generate_leak_free_report(eval_results, routing_results, cv_info, output_dir)
    
    print("\n" + "="*70)
    print(report_text)
    print("="*70)
    
    # Create CANONICAL_RESULTS.json (single source of truth)
    canonical = {
        "dataset": "gsm8k_qwen_300_real_traces",
        "generation_model": "Qwen2.5-0.5B-Instruct",
        "n_samples": len(data),
        "accuracy": correct / len(data),
        "cross_validation": cv_info,
        "metrics": eval_results,
        "compute_routing": routing_results,
        "timestamp": str(datetime.now())
    }
    
    canonical_path = Path("experiments/results/CANONICAL_RESULTS.json")
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    with open(canonical_path, 'w') as f:
        json.dump(canonical, f, indent=2)
    
    print(f"\n✅ CANONICAL_RESULTS.json saved to {canonical_path}")
    print("This is the single source of truth for all paper numbers.")


if __name__ == "__main__":
    from datetime import datetime
    main()