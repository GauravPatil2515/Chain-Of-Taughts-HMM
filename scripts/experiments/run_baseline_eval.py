"""
Baseline Runner Script
Complete pipeline to evaluate HMM against all baseline methods.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.baseline_evaluation import generate_comparison_report, print_summary
from src.evaluation.baseline_plots import generate_all_comparison_plots
from src.analysis.qualitative import plot_transition_matrices, extract_qualitative_examples


def run_baseline_evaluation(
    data_path: str,
    model_path: str,
    output_dir: str = "experiments/results/baseline"
):
    """
    Run complete baseline evaluation pipeline.
    
    Args:
        data_path: Path to scored CoT data (JSON)
        model_path: Path to trained HMM model
        output_dir: Directory for output files
    """
    print("="*70)
    print(" "*20 + "BASELINE EVALUATION PIPELINE")
    print("="*70)
    print(f"\nData: {data_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate comparison report
    report_path = output_dir / "baseline_comparison.json"
    print("\n" + "="*70)
    print("STEP 1: Computing Baseline Scores and Metrics")
    print("="*70)
    
    report = generate_comparison_report(data_path, model_path, str(report_path))
    
    # Step 2: Print summary
    print_summary(report)
    
    # Step 3: Generate visualizations
    print("\n" + "="*70)
    print("STEP 2: Generating Comparison Visualizations")
    print("="*70)
    
    figures_dir = output_dir / "figures"
    figures_dir = output_dir / "figures"
    generate_all_comparison_plots(str(report_path), str(figures_dir))
    
    # Step 4: Qualitative Analysis
    print("\n" + "="*70)
    print("STEP 3: Qualitative Analysis")
    print("="*70)
    
    # Transition Matrices
    print("Generating transition matrix heatmaps...")
    plot_transition_matrices(
        report["all_scores"], 
        str(figures_dir / "transition_matrices.png")
    )
    
    # Example Extraction
    print("Extracting qualitative examples...")
    import pandas as pd
    df_scores = pd.DataFrame(report["all_scores"])
    extract_qualitative_examples(
        df_scores,
        str(output_dir / "qualitative_examples.md")
    )
    
    # Final summary
    print("\n" + "="*70)
    print("BASELINE EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n📊 Results saved to: {output_dir}")
    print(f"   - Report: {report_path}")
    print(f"   - Figures: {figures_dir}")
    print("\n📈 Key Outputs:")
    print(f"   - Correlation analysis")
    print(f"   - Classification metrics (accuracy, precision, recall, F1)")
    print(f"   - Score distributions")
    print(f"   - Method comparison plots")
    print("\n✅ Baseline methods evaluated:")
    print(f"   1. Length-based scoring")
    print(f"   2. Tag diversity scoring")
    print(f"   3. Structure-based scoring")
    print(f"   4. Transition quality scoring")
    print(f"   5. Combined baseline")
    print(f"   6. Random baseline (sanity check)")
    print(f"   7. HMM-based scoring")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation comparing HMM against simple heuristics"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/tagged/phase1_tagged.json",
        help="Path to scored CoT data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/phase1_hmm.pkl",
        help="Path to trained HMM model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/phase1_baseline",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        print("\nTip: Run demo_pipeline.py first to generate scored data.")
        return 1
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("\nTip: Run demo_pipeline.py first to train the HMM model.")
        return 1
    
    # Run evaluation
    run_baseline_evaluation(args.data, args.model, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
