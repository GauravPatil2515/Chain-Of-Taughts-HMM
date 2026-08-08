#!/usr/bin/env python3
"""
Leak-Free Stratified 5-Fold Cross-Validation & Benchmark Evaluation Script for CoT-HMM.
Ensures zero data leakage: HMM model ensembles are fitted strictly on training fold correct traces,
and out-of-fold predictions are collected for every sample.

Computes:
 1. 5-Fold Cross-Validated Ensemble HMM Scores (normalized log-likelihood)
 2. Anomaly HMM & Plausibility Margin
 3. Online Structural Entropy & Entropy Drift Rate (EDR)
 4. Statistical Metrics: ROC-AUC, Pearson r, Point-Biserial r, p-values
 5. 95% Bootstrap Confidence Intervals (10,000 resamples)
 6. Effect Sizes: Cohen's d, Cliff's delta
 7. Baseline Heuristic Comparisons
 8. 3-Tier Compute Routing Simulation
"""

import json
import math
import sys
import numpy as np
import scipy.stats as stats
from pathlib import Path
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Ensure src is in import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hmm.ensemble import EnsembleHMM
from src.hmm.trainer import TAG_TO_ID
from src.experiments_v2.bootstrap_ci import bootstrap_correlation, bootstrap_auc, bootstrap_mean_diff
from src.experiments_v2.effect_size import cohens_d, cliffs_delta


def compute_sequence_entropy(tag_seq: List[str]) -> float:
    """Compute final structural Shannon entropy (base 2) of a tag sequence."""
    if not tag_seq:
        return 0.0
    counts = {}
    for tag in tag_seq:
        counts[tag] = counts.get(tag, 0) + 1
    n = len(tag_seq)
    entropy = 0.0
    for tag, count in counts.items():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def compute_entropy_drift_rate(tag_seq: List[str]) -> Tuple[float, List[float]]:
    """
    Compute online step-by-step rolling structural entropy trajectory H_t
    and Entropy Drift Rate (EDR) = avg absolute change in entropy per step.
    """
    if len(tag_seq) < 2:
        return 0.0, [0.0] * len(tag_seq)
    
    entropies = []
    for k in range(1, len(tag_seq) + 1):
        prefix = tag_seq[:k]
        h_k = compute_sequence_entropy(prefix)
        entropies.append(h_k)
    
    drift_sum = sum(abs(entropies[i] - entropies[i-1]) for i in range(1, len(entropies)))
    edr = drift_sum / (len(tag_seq) - 1)
    return edr, entropies


def compute_heuristic_baselines(tag_seq: List[str]) -> Dict[str, float]:
    """Compute baseline heuristic scores for a tag sequence."""
    if not tag_seq:
        return {
            "has_conclusion": 0.0,
            "repetition_penalty": 0.0,
            "length_penalty": 0.0,
            "heuristic_composite": 0.0
        }
    
    has_co = 1.0 if "CO" in tag_seq else 0.0
    
    # Count consecutive repeated tags
    repetitions = sum(1 for i in range(1, len(tag_seq)) if tag_seq[i] == tag_seq[i-1])
    rep_penalty = repetitions / max(1, len(tag_seq))
    
    # Length penalty: optimal reasoning length ~4-8 steps
    length = len(tag_seq)
    len_penalty = abs(length - 5) / 5.0
    
    # Composite heuristic: prefers conclusion, discourages repetition & extreme length
    heuristic_score = (has_co * 2.0) - (rep_penalty * 1.5) - (len_penalty * 0.5)
    
    return {
        "has_conclusion": has_co,
        "repetition_penalty": float(rep_penalty),
        "length_penalty": float(len_penalty),
        "heuristic_composite": float(heuristic_score)
    }


def run_leakfree_cross_validation(
    dataset_path: str,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run stratified 5-fold CV with out-of-fold scoring to ensure zero leakage."""
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    n_samples = len(data)
    print(f"Total samples: {n_samples}")
    labels = np.array([1 if item["is_correct"] else 0 for item in data])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    scored_dataset = [dict(item) for item in data]
    
    print(f"\n--- Starting Leak-Free Stratified {n_splits}-Fold Cross-Validation ---")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels), 1):
        print(f"\nFold {fold}/{n_splits}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        # Split data
        train_items = [data[i] for i in train_idx]
        val_items = [data[i] for i in val_idx]
        
        # Correct traces for standard HMM ensemble
        train_correct_seqs = [item["tag_sequence"] for item in train_items if item["is_correct"]]
        # Incorrect traces for Anomaly HMM ensemble
        train_incorrect_seqs = [item["tag_sequence"] for item in train_items if not item["is_correct"]]
        
        print(f"  Training Ensemble HMM on {len(train_correct_seqs)} correct traces...")
        correct_ensemble = EnsembleHMM(
            component_configs=[3, 5, 8, 12],
            n_seeds_per_config=3,
            n_iter=100
        )
        correct_ensemble.fit(train_correct_seqs, verbose=False)
        
        print(f"  Training Anomaly HMM on {len(train_incorrect_seqs)} incorrect traces...")
        anomaly_ensemble = EnsembleHMM(
            component_configs=[3, 5, 8, 12],
            n_seeds_per_config=3,
            n_iter=100
        )
        anomaly_ensemble.fit(train_incorrect_seqs, verbose=False)
        
        # Out-of-fold scoring for validation items
        print(f"  Scoring {len(val_items)} out-of-fold validation traces...")
        for orig_idx in val_idx:
            item = scored_dataset[orig_idx]
            tag_seq = item["tag_sequence"]
            
            # Standard HMM score
            correct_res = correct_ensemble.score(tag_seq)
            norm_ll = correct_res.get("ensemble_norm_ll", correct_res.get("ensemble_ll", 0.0) / max(1, len(tag_seq)))
            raw_ll = correct_res.get("ensemble_ll", float("-inf"))
            
            # Anomaly HMM score
            anomaly_res = anomaly_ensemble.score(tag_seq)
            anomaly_norm_ll = anomaly_res.get("ensemble_norm_ll", anomaly_res.get("ensemble_ll", 0.0) / max(1, len(tag_seq)))
            anomaly_raw_ll = anomaly_res.get("ensemble_ll", float("-inf"))
            
            # Plausibility Margin = score(correct_hmm) - score(anomaly_hmm)
            plausibility_margin = norm_ll - anomaly_norm_ll
            
            # Entropy and EDR metrics
            edr, entropy_trajectory = compute_entropy_drift_rate(tag_seq)
            final_entropy = compute_sequence_entropy(tag_seq)
            
            # Heuristic baselines
            heuristics = compute_heuristic_baselines(tag_seq)
            
            # Update item with CV metrics
            item["cv_fold"] = fold
            item["oof_hmm_score"] = norm_ll
            item["oof_raw_ll"] = raw_ll
            item["oof_anomaly_score"] = anomaly_norm_ll
            item["oof_plausibility_margin"] = plausibility_margin
            item["entropy_drift_rate"] = edr
            item["structural_entropy"] = final_entropy
            item["entropy_trajectory"] = entropy_trajectory
            item.update(heuristics)
    
    print("\n--- Out-of-Fold Cross-Validation Scoring Complete ---")
    return scored_dataset, {"n_splits": n_splits, "n_samples": n_samples}


def evaluate_statistical_performance(scored_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate thorough statistical metrics and bootstrap CIs across out-of-fold predictions."""
    labels = np.array([1 if d["is_correct"] else 0 for d in scored_data])
    
    metrics_to_eval = {
        "Ensemble HMM Score": np.array([d["oof_hmm_score"] for d in scored_data]),
        "Plausibility Margin": np.array([d["oof_plausibility_margin"] for d in scored_data]),
        "Entropy Drift Rate (EDR)": np.array([-d["entropy_drift_rate"] for d in scored_data]), # Invert so higher = more plausibility
        "Structural Entropy": np.array([-d["structural_entropy"] for d in scored_data]),       # Invert so higher = more plausible
        "Heuristic Baseline": np.array([d["heuristic_composite"] for d in scored_data]),
        "Has Conclusion": np.array([d["has_conclusion"] for d in scored_data]),
        "Repetition Penalty": np.array([-d["repetition_penalty"] for d in scored_data]),
    }
    
    results = {}
    
    for metric_name, scores in metrics_to_eval.items():
        # Handle nan / inf
        valid_mask = np.isfinite(scores)
        v_scores = scores[valid_mask]
        v_labels = labels[valid_mask]
        
        # Pearson point-biserial correlation & p-value
        r, p_val = stats.pointbiserialr(v_labels, v_scores)
        
        # ROC-AUC
        auc = float(roc_auc_score(v_labels, v_scores))
        
        # Bootstrap CIs (10,000 iterations)
        r_ci = bootstrap_correlation(v_scores, v_labels, n_bootstrap=10000, seed=42)
        auc_ci = bootstrap_auc(v_scores, v_labels, n_bootstrap=10000, seed=42)
        
        # Group stats
        correct_vals = v_scores[v_labels == 1]
        incorrect_vals = v_scores[v_labels == 0]
        
        d_res = cohens_d(correct_vals, incorrect_vals)
        delta_res = cliffs_delta(correct_vals, incorrect_vals)
        
        results[metric_name] = {
            "pearson_r": float(r),
            "p_value": float(p_val),
            "r_ci_lower": float(r_ci.lower),
            "r_ci_upper": float(r_ci.upper),
            "auc": float(auc),
            "auc_ci_lower": float(auc_ci.lower),
            "auc_ci_upper": float(auc_ci.upper),
            "cohens_d": float(d_res.d),
            "cohens_d_interpretation": d_res.interpretation,
            "cliffs_delta": float(delta_res.delta),
            "cliffs_delta_interpretation": delta_res.interpretation,
            "mean_correct": float(np.mean(correct_vals)),
            "std_correct": float(np.std(correct_vals)),
            "mean_incorrect": float(np.mean(incorrect_vals)),
            "std_incorrect": float(np.std(incorrect_vals)),
        }
    
    return results


def simulate_3tier_routing(scored_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simulate 3-tier adaptive compute routing gate."""
    scores = np.array([d["oof_hmm_score"] for d in scored_data])
    labels = np.array([1 if d["is_correct"] else 0 for d in scored_data])
    
    # Determine thresholds (e.g. 35th percentile and 70th percentile)
    theta_low = float(np.percentile(scores, 35))
    theta_high = float(np.percentile(scores, 70))
    
    passed_mask = scores >= theta_high
    rejected_mask = scores < theta_low
    escalated_mask = (scores >= theta_low) & (scores < theta_high)
    
    n_total = len(scores)
    n_passed = int(np.sum(passed_mask))
    n_rejected = int(np.sum(rejected_mask))
    n_escalated = int(np.sum(escalated_mask))
    
    passed_acc = float(np.mean(labels[passed_mask])) if n_passed > 0 else 0.0
    rejected_acc = float(np.mean(labels[rejected_mask])) if n_rejected > 0 else 0.0
    escalated_acc = float(np.mean(labels[escalated_mask])) if n_escalated > 0 else 0.0
    overall_baseline_acc = float(np.mean(labels))
    
    # Compute saved: Queries passed directly without needing re-sampling or LLM-judge
    compute_saved_pct = (n_passed / n_total) * 100.0
    
    return {
        "theta_low": theta_low,
        "theta_high": theta_high,
        "n_total": n_total,
        "tier_passed": {"count": n_passed, "pct": (n_passed/n_total)*100, "accuracy": passed_acc},
        "tier_escalated": {"count": n_escalated, "pct": (n_escalated/n_total)*100, "accuracy": escalated_acc},
        "tier_rejected": {"count": n_rejected, "pct": (n_rejected/n_total)*100, "accuracy": rejected_acc},
        "baseline_accuracy": overall_baseline_acc,
        "compute_saved_pct": compute_saved_pct,
        "passed_vs_baseline_acc_gain": (passed_acc - overall_baseline_acc) * 100.0
    }


def generate_leak_free_report(
    eval_results: Dict[str, Any],
    routing_results: Dict[str, Any],
    cv_info: Dict[str, Any],
    output_dir: Path
) -> str:
    """Generate markdown and text reports."""
    lines = []
    lines.append("# CoT-HMM Leak-Free Stratified Cross-Validation Benchmark Report")
    lines.append(f"**Dataset Size:** {cv_info['n_samples']} samples | **Cross-Validation:** {cv_info['n_splits']}-Fold Stratified Out-of-Fold")
    lines.append("")
    lines.append("## Executive Summary of Results")
    lines.append("All HMM evaluation scores presented below were generated strictly out-of-fold. No training data was leaked into test scoring.")
    lines.append("")
    
    lines.append("### 📊 Primary Metric Comparison Table")
    lines.append("| Method / Metric | Pearson r [95% CI] | ROC-AUC [95% CI] | p-value | Cohen's d | Cliff's delta |")
    lines.append("|---|---|---|---|---|---|")
    
    for metric_name, m in eval_results.items():
        r_str = f"{m['pearson_r']:.3f} [{m['r_ci_lower']:.3f}, {m['r_ci_upper']:.3f}]"
        auc_str = f"{m['auc']:.3f} [{m['auc_ci_lower']:.3f}, {m['auc_ci_upper']:.3f}]"
        p_str = "< 0.001" if m['p_value'] < 0.001 else f"{m['p_value']:.4f}"
        d_str = f"{m['cohens_d']:.3f} ({m['cohens_d_interpretation']})"
        delta_str = f"{m['cliffs_delta']:.3f} ({m['cliffs_delta_interpretation']})"
        lines.append(f"| **{metric_name}** | {r_str} | {auc_str} | {p_str} | {d_str} | {delta_str} |")
    
    lines.append("")
    lines.append("### ⚡ 3-Tier Adaptive Compute Routing Gate Performance")
    lines.append(f"- **Compute Savings:** {routing_results['compute_saved_pct']:.1f}% of queries passed directly")
    lines.append(f"- **High-Confidence Pass Tier:** {routing_results['tier_passed']['count']} samples ({routing_results['tier_passed']['pct']:.1f}%), Accuracy = **{routing_results['tier_passed']['accuracy']*100:.1f}%** (vs baseline {routing_results['baseline_accuracy']*100:.1f}%)")
    lines.append(f"- **Escalation / Re-sampling Tier:** {routing_results['tier_escalated']['count']} samples ({routing_results['tier_escalated']['pct']:.1f}%), Accuracy = {routing_results['tier_escalated']['accuracy']*100:.1f}%")
    lines.append(f"- **Low-Confidence Reject Tier:** {routing_results['tier_rejected']['count']} samples ({routing_results['tier_rejected']['pct']:.1f}%), Accuracy = {routing_results['tier_rejected']['accuracy']*100:.1f}%")
    lines.append("")
    
    report_text = "\n".join(lines)
    
    # Write to output files
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "leak_free_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(output_dir / "leak_free_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
        
    return report_text


def main():
    dataset_path = "data/v2/gsm8k_500_proper.json"
    output_dir = Path("experiments/results")
    
    # 1. Run leak-free cross-validation
    scored_data, cv_info = run_leakfree_cross_validation(dataset_path, n_splits=5, random_state=42)
    
    # 2. Save out-of-fold scored dataset
    cv_dataset_path = Path("data/v2/gsm8k_500_proper_cv_scored.json")
    cv_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cv_dataset_path, "w", encoding="utf-8") as f:
        json.dump(scored_data, f, indent=2)
    print(f"\nSaved cross-validated dataset to: {cv_dataset_path}")
    
    # 3. Evaluate statistical performance
    eval_results = evaluate_statistical_performance(scored_data)
    
    # 4. Simulate 3-tier compute routing
    routing_results = simulate_3tier_routing(scored_data)
    
    # 5. Save evaluation JSON
    eval_json_path = output_dir / "evaluation_results_leak_free.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "cross_validation": cv_info,
            "metrics": eval_results,
            "compute_routing": routing_results
        }, f, indent=2)
    print(f"Saved JSON evaluation results to: {eval_json_path}")
    
    # 6. Generate and print human-readable report
    report_text = generate_leak_free_report(eval_results, routing_results, cv_info, output_dir)
    print("\n" + "="*70)
    print(report_text)
    print("="*70)


if __name__ == "__main__":
    main()
