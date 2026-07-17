#!/usr/bin/env python3
"""
run_experiment_suite.py - Run all experiments with validation
Orchestrates data → tagging → training → evaluation with checks.

Usage:
    python scripts/run_experiment_suite.py --phase all --validate --gpu
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.experiments_v2.data_schema import ScoredTrace, validate_dataset


class ExperimentSuite:
    """Complete experiment runner with validation at each stage."""
    
    def __init__(self, data_dir: str = "data/v2", models_dir: str = "models", 
                 results_dir: str = "experiments/results/v2", device: str = "cpu"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.device = device
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_data(self, dataset_path: str) -> bool:
        """Check dataset is valid ScoredTrace format."""
        print(f"\n[validation] Checking {dataset_path}...")
        
        path = Path(dataset_path)
        if not path.exists():
            print(f"  ✗ File not found")
            return False
        
        with open(path) as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            print(f"  ✗ Invalid format")
            return False
        
        required_keys = ["sample_id", "problem", "cot_text", "is_correct", "tag_sequence"]
        sample = data[0]
        missing = [k for k in required_keys if k not in sample]
        
        if missing:
            print(f"  ✗ Missing keys: {missing}")
            return False
        
        print(f"  ✓ Valid dataset ({len(data)} samples)")
        return True
    
    def validate_models(self) -> bool:
        """Check HMM ensemble exists."""
        ensemble_path = self.models_dir / "ensemble.pkl"
        if ensemble_path.exists():
            print(f"  ✓ Model exists: {ensemble_path}")
            return True
        else:
            print(f"  ⚠ Model not found: {ensemble_path}")
            return False
    
    def phase1_data_scaling(self, model: str, limit: int) -> Dict:
        """Scale data phase."""
        print("\n[phase 1] Data Scaling")
        print("-" * 40)
        
        # Simulate - actual training would call model API
        result = {
            "phase": 1,
            "status": "planned",
            "model": model,
            "target_samples": limit,
            "output_file": str(self.data_dir / f"gsm8k_{limit}_traces.json")
        }
        
        print(f"  Model: {model}")
        print(f"  Target samples: {limit}")
        print(f"  Output: {result['output_file']}")
        
        return result
    
    def phase2_baselines(self) -> Dict:
        """Run baseline evaluations."""
        print("\n[phase 2] Baseline Evaluation")
        print("-" * 40)
        
        result = {
            "phase": 2,
            "status": "ready",
            "methods": ["LengthBaseline", "TagDiversityBaseline", "TransitionBaseline", 
                       "CombinedBaseline", "ROSCOE", "PiCSAR"]
        }
        
        for m in result["methods"]:
            print(f"  Will run: {m}")
        
        return result
    
    def phase3_stats_rigor(self, dataset_path: str) -> Dict:
        """Statistical rigor phase."""
        print("\n[phase 3] Statistical Rigor")
        print("-" * 40)
        
        if not self.validate_data(dataset_path):
            return {"phase": 3, "status": "skipped", "reason": "invalid data"}
        
        result = {
            "phase": 3,
            "status": "ready",
            "bootstrap_samples": 10000,
            "output": str(self.results_dir / "bootstrap_cis.json")
        }
        
        print(f"  Bootstrap samples: 10000")
        print(f"  Output: {result['output']}")
        
        return result
    
    def phase4_neural_hmm(self) -> Dict:
        """Neural-HMM training phase."""
        print("\n[phase 4] Neural-HMM Hybrid")
        print("-" * 40)
        
        if self.device == "cpu":
            print("  ⚠ Using CPU (slower)")
        
        result = {
            "phase": 4,
            "status": "ready",
            "backbone": "distilbert-base-uncased",
            "epochs": 5,
            "output": str(self.models_dir / "neural_tagger_v1.pt")
        }
        
        print(f"  Training script: scripts/train_neural_tagger_gpu.py")
        print(f"  Output: {result['output']}")
        
        return result
    
    def run_all(self, **kwargs) -> Dict[str, Any]:
        """Run all phases in sequence."""
        results = {}
        
        for phase_num in range(1, 5):
            method_name = f"phase{phase_num}_data_scaling" if phase_num == 1 else \
                         f"phase{phase_num}_baselines" if phase_num == 2 else \
                         f"phase{phase_num}_stats_rigor" if phase_num == 3 else \
                         f"phase{phase_num}_neural_hmm"
            method = getattr(self, method_name)
            results[f"phase{phase_num}"] = method(**kwargs)
        
        # Save summary
        summary_path = self.results_dir / "suite_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Summary saved to {summary_path}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Run complete experiment suite")
    parser.add_argument("--phase", choices=["all", "1", "2", "3", "4"], default="all")
    parser.add_argument("--data", type=str, default="data/v2/gsm8k_500_traces.json")
    parser.add_argument("--model", type=str, default="qwen-0.5b")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--validate", action="store_true", help="Run validation checks")
    
    args = parser.parse_args()
    
    device = "cuda" if args.gpu else "cpu"
    
    # Check validation prerequisites
    if args.validate:
        print("=" * 60)
        print("VALIDATION CHECKS")
        print("=" * 60)
        
        suite = ExperimentSuite(device=device)
        suite.validate_models()
        suite.validate_data(args.data)
    
    # Run selected phases
    print("\n" + "=" * 60)
    print("EXPERIMENT SUITE")
    print("=" * 60)
    
    suite = ExperimentSuite(device=device)
    
    if args.phase == "all":
        results = suite.run_all(
            dataset_path=args.data,
            model=args.model,
            limit=args.limit
        )
    else:
        phase_num = int(args.phase)
        method_name = f"phase{phase_num}_data_scaling" if phase_num == 1 else \
                     f"phase{phase_num}_baselines" if phase_num == 2 else \
                     f"phase{phase_num}_stats_rigor" if phase_num == 3 else \
                     f"phase{phase_num}_neural_hmm"
        method = getattr(suite, method_name)
        results = {f"phase{phase_num}": method(
            dataset_path=args.data,
            model=args.model,
            limit=args.limit
        )}
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()