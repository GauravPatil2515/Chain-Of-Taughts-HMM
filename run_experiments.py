#!/usr/bin/env python3
"""
Single Entrypoint for CoT-HMM Experiments
Executes the full pipeline: semantic tagging -> HMM training -> evaluation
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tagging.rule_based_tagger import tag_cot_file
from src.data_collection.gsm8k_loader import load_gsm8k
from src.hmm.trainer import EnsembleHMMTrainer
from src.hmm.scorer import EnsembleScorer
from src.evaluation.baseline_evaluation import evaluate_baselines, compare_methods
from src.evaluation.baselines import get_all_baselines


def run_tagging(input_file: str, output_file: str) -> None:
    """Run semantic tagging on CoT data."""
    print(f"[1/4] Tagging CoT data: {input_file} -> {output_file}")
    tag_cot_file(input_file, output_file)
    print(f"    Done. Tagged data saved to {output_file}")


def run_training(tagged_file: str, model_dir: str, n_ensemble: int = 12) -> None:
    """Train HMM ensemble on tagged correct solutions."""
    print(f"[2/4] Training HMM ensemble ({n_ensemble} models)...")
    
    # Load tagged data
    with open(tagged_file, 'r') as f:
        data = json.load(f)
    
    # Filter for correct solutions only
    correct_sequences = [
        item['tags'] for item in data 
        if item.get('is_correct', False) and item.get('tags')
    ]
    
    print(f"    Training on {len(correct_sequences)} correct sequences")
    
    # Train ensemble
    trainer = EnsembleHMMTrainer(
        n_components=5,  # Optimal from ablations
        n_ensemble=n_ensemble,
        n_iter=100,
        random_state=42
    )
    trainer.fit(correct_sequences)
    
    # Save models
    model_path = Path(model_dir) / "ensemble_hmm.pkl"
    trainer.save(str(model_path))
    print(f"    Ensemble saved to {model_path}")


def run_evaluation(tagged_file: str, model_dir: str, output_dir: str) -> dict:
    """Evaluate trained models against baselines."""
    print(f"[3/4] Evaluating models...")
    
    # Load tagged data
    with open(tagged_file, 'r') as f:
        data = json.load(f)
    
    # Load ensemble
    model_path = Path(model_dir) / "ensemble_hmm.pkl"
    scorer = EnsembleScorer.load(str(model_path))
    
    # Get sequences and labels
    sequences = [item['tags'] for item in data if item.get('tags')]
    labels = [item.get('is_correct', False) for item in data if item.get('tags')]
    
    # Score with HMM
    hmm_scores = scorer.score_batch(sequences)
    
    # Score with baselines
    baselines = get_all_baselines()
    baseline_scores = {}
    for name, baseline_fn in baselines.items():
        baseline_scores[name] = [baseline_fn(seq) for seq in sequences]
    
    # Evaluate
    results = evaluate_baselines(hmm_scores, baseline_scores, labels)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"    Results saved to {output_path / 'evaluation_results.json'}")
    return results


def generate_report(results: dict, output_dir: str) -> None:
    """Generate human-readable report."""
    print(f"[4/4] Generating report...")
    
    report = []
    report.append("=" * 60)
    report.append("CoT-HMM Structural Verification - Results Report")
    report.append("=" * 60)
    report.append("")
    
    # Main results
    report.append("MAIN RESULTS (GSM8K)")
    report.append("-" * 30)
    
    if 'hmm' in results:
        hmm = results['hmm']
        report.append(f"Ensemble HMM:  r = {hmm['correlation']:.3f}, "
                      f"p = {hmm['p_value']:.4f}, AUC = {hmm['auc']:.3f}, F1 = {hmm['f1']:.3f}")
    
    # Baselines
    for name, metrics in results.items():
        if name != 'hmm':
            report.append(f"{name:15s}: r = {metrics['correlation']:.3f}, "
                          f"p = {metrics['p_value']:.4f}, AUC = {metrics['auc']:.3f}")
    
    report.append("")
    report.append("INTERPRETATION")
    report.append("-" * 30)
    
    if 'hmm' in results and results['hmm']['p_value'] < 0.05:
        report.append("✓ HMM correlation is STATISTICALLY SIGNIFICANT (p < 0.05)")
        report.append("✓ Structural patterns predict reasoning correctness")
    else:
        report.append("✗ HMM correlation NOT statistically significant")
    
    # Check if HMM beats baselines
    hmm_corr = results.get('hmm', {}).get('correlation', 0)
    baseline_corrs = [m['correlation'] for n, m in results.items() if n != 'hmm']
    if baseline_corrs and hmm_corr > max(baseline_corrs):
        report.append("✓ HMM outperforms all heuristic baselines")
    
    report.append("")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report
    output_path = Path(output_dir) / "results_report.txt"
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"    Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CoT-HMM: Structural Verification of Chain-of-Thought Reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_experiments.py --input data/cot_outputs.json --output experiments/results
  
  # Run only tagging
  python run_experiments.py --input data/cot_outputs.json --output experiments/results --stage tagging
  
  # Run only training (requires tagged data)
  python run_experiments.py --input data/tagged.json --output experiments/results --stage training
        """
    )
    
    parser.add_argument('--input', required=True, help='Input CoT data file (JSON)')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--stage', choices=['all', 'tagging', 'training', 'evaluation'], 
                        default='all', help='Pipeline stage to run')
    parser.add_argument('--n-ensemble', type=int, default=12, help='Number of ensemble models')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    model_dir = output_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"CoT-HMM Experiment Pipeline")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Stage:  {args.stage}")
    print()
    
    tagged_file = output_path / "tagged_data.json"
    
    if args.stage in ['all', 'tagging']:
        run_tagging(str(input_path), str(tagged_file))
    
    if args.stage in ['all', 'training']:
        if not tagged_file.exists():
            print(f"Error: Tagged file not found: {tagged_file}")
            print("Run tagging stage first or provide pre-tagged data.")
            sys.exit(1)
        run_training(str(tagged_file), str(model_dir), args.n_ensemble)
    
    if args.stage in ['all', 'evaluation']:
        if not tagged_file.exists():
            print(f"Error: Tagged file not found: {tagged_file}")
            sys.exit(1)
        if not (model_dir / "ensemble_hmm.pkl").exists():
            print(f"Error: Trained model not found: {model_dir / 'ensemble_hmm.pkl'}")
            sys.exit(1)
        
        results = run_evaluation(str(tagged_file), str(model_dir), str(output_path))
        generate_report(results, str(output_path))
    
    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()