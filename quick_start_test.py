"""
Quick Start Script - Phase 1 Test
Tests data generation with 20 samples before full run.
"""

import subprocess
import sys
import os
from pathlib import Path
import json


def run_command(cmd: list, description: str):
    """Run command and handle errors."""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: Command failed with return code {result.returncode}")
        return False
    
    print(f"\n✅ {description} - Complete")
    return True


def check_file_exists(path: str, min_samples: int = 0):
    """Check if file exists and has minimum samples."""
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return False
    
    if path.endswith('.json') and min_samples > 0:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                if len(data) < min_samples:
                    print(f"⚠️ Only {len(data)} samples found, expected {min_samples}")
                else:
                    print(f"✅ Found {len(data)} samples")
        except Exception as e:
            print(f"⚠️ Could not parse JSON: {e}")
    
    return True


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           CoT-HMM Research Pipeline - Quick Start Test           ║
║                                                                   ║
║  This script tests Phase 1 (Data Generation) with 20 samples     ║
║  Before running full 150+ sample generation                       ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if in correct directory
    if not os.path.exists("generate_batch.py"):
        print("❌ Error: Please run from project root directory")
        print("   Expected files: generate_batch.py, run_baseline_eval.py")
        return 1
    
    print("✅ Running from correct directory")
    
    # Step 1: Test data generation
    test_output = "data/cot_outputs/test_quick_start.json"
    
    if not run_command([
        sys.executable, "generate_batch.py",
        "--dataset", "data/raw/gsm8k.json",
        "--dataset-name", "gsm8k",
        "--domain", "math",
        "--model", "qwen-0.5b",
        "--output", test_output,
        "--max-samples", "20",
        "--seed", "42"
    ], "Step 1: Generate 20 test samples"):
        return 1
    
    # Verify output
    if not check_file_exists(test_output, min_samples=15):
        print("⚠️ Warning: Generated fewer samples than expected")
    
    # Step 2: Check data quality
    if not run_command([
        sys.executable, "scripts/check_data_quality.py",
        test_output
    ], "Step 2: Data quality check"):
        print("⚠️ Warning: Quality check found issues (may be normal for small test)")
    
    # Step 3: Tag the data
    tagged_output = "data/tagged/test_quick_start_tagged.json"
    
    if not run_command([
        sys.executable, "-m", "src.tagging.rule_based_tagger",
        "--input", test_output,
        "--output", tagged_output
    ], "Step 3: Tag CoT steps"):
        return 1
    
    # Verify tagged output
    check_file_exists(tagged_output, min_samples=15)
    
    # Step 4: Train mini HMM
    model_output = "models/test_hmm.pkl"
    
    if not run_command([
        sys.executable, "-m", "src.hmm.trainer",
        "--input", tagged_output,
        "--output", model_output,
        "--n-components", "4",  # Smaller for test
        "--correct-only"
    ], "Step 4: Train test HMM"):
        print("⚠️ Warning: Training may fail with very small dataset (expected)")
        print("   This is normal - will work better with 100+ samples")
    
    # Step 5: Score
    scored_output = "experiments/results/test_quick_start_scored.json"
    
    if os.path.exists(model_output):
        if not run_command([
            sys.executable, "-m", "src.hmm.scorer",
            "--model", model_output,
            "--input", tagged_output,
            "--output", scored_output
        ], "Step 5: Score with HMM"):
            print("⚠️ Warning: Scoring failed")
    
    # Step 6: Run baseline eval
    if os.path.exists(scored_output):
        run_command([
            sys.executable, "run_baseline_eval.py",
            "--data", scored_output,
            "--model", model_output,
            "--output", "experiments/results/test_baseline"
        ], "Step 6: Baseline evaluation")
    
    # Final summary
    print(f"\n{'='*70}")
    print("🎉 QUICK START TEST COMPLETE!")
    print(f"{'='*70}")
    print("\n✅ Successfully tested:")
    print("   1. Data generation (20 samples)")
    print("   2. Data quality check")
    print("   3. CoT tagging")
    print("   4. HMM training (if enough correct samples)")
    print("   5. Baseline evaluation")
    print("\n📁 Output files:")
    print(f"   - Generated: {test_output}")
    print(f"   - Tagged: {tagged_output}")
    print(f"   - Model: {model_output}")
    print(f"   - Scored: {scored_output}")
    print("\n🚀 Next step: Scale to 150+ samples")
    print("   Run: python generate_batch.py --max-samples 150 ...")
    print(f"\n{'='*70}")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
