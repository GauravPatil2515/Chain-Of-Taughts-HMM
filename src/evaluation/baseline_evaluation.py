"""
Baseline Evaluation Framework
Compare HMM-based scorer against simple baseline methods.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import joblib

from .baselines import score_with_all_baselines
from ..hmm.scorer import HMMScorer


def load_scored_data(file_path: str) -> List[Dict]:
    """Load scored CoT data."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_baseline_scores(data: List[Dict]) -> pd.DataFrame:
    """Compute all baseline scores for dataset."""
    results = []
    
    for item in data:
        cot_text = item.get("cot_raw", "")
        tag_sequence = item.get("tag_sequence", [])
        
        baseline_scores = score_with_all_baselines(cot_text, tag_sequence)
        
        row = {
            "id": item["id"],
            "is_correct": item.get("is_correct", False),
            "model": item.get("model", "unknown"),
            "cot_raw": cot_text,
            "tag_sequence": tag_sequence
        }
        
        # Add all baseline scores
        for method_name, score_obj in baseline_scores.items():
            row[f"{method_name}_score"] = score_obj.score
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_hmm_scores(data: List[Dict], model_path: str) -> pd.DataFrame:
    """Compute HMM scores for dataset."""
    scorer = HMMScorer(model_path)
    
    results = []
    for item in data:
        tag_sequence = item.get("tag_sequence", [])
        
        if len(tag_sequence) >= 1:
            score_obj = scorer.score(tag_sequence)
            hmm_score = score_obj.normalized_ll  # Use normalized log-likelihood
        else:
            hmm_score = float('-inf')
        
        results.append({
            "id": item["id"],
            "hmm_score": hmm_score,
            "hmm_ll": score_obj.log_likelihood if len(tag_sequence) >= 1 else float('-inf'),
            "hmm_anomalies": score_obj.anomaly_count if len(tag_sequence) >= 1 else 0
        })
    
    return pd.DataFrame(results)


def correlation_analysis(df: pd.DataFrame, score_cols: List[str], target_col: str = "is_correct") -> pd.DataFrame:
    """Compute correlation between scores and correctness."""
    results = []
    
    for col in score_cols:
        # Filter out infinite values
        valid_mask = np.isfinite(df[col])
        valid_df = df[valid_mask]
        
        if len(valid_df) < 2:
            continue
        
        # Point-biserial correlation (binary target vs continuous score)
        corr, p_value = stats.pointbiserialr(valid_df[target_col], valid_df[col])
        
        # Mean scores for correct vs incorrect
        mean_correct = valid_df[valid_df[target_col] == True][col].mean()
        mean_incorrect = valid_df[valid_df[target_col] == False][col].mean()
        
        results.append({
            "method": col.replace("_score", ""),
            "correlation": corr,
            "p_value": p_value,
            "mean_correct": mean_correct,
            "mean_incorrect": mean_incorrect,
            "difference": mean_correct - mean_incorrect
        })
    
    return pd.DataFrame(results).sort_values("correlation", ascending=False)


def classification_metrics(df: pd.DataFrame, score_cols: List[str], target_col: str = "is_correct") -> pd.DataFrame:
    """Compute classification metrics using median threshold."""
    results = []
    
    for col in score_cols:
        # Filter out infinite values
        valid_mask = np.isfinite(df[col])
        valid_df = df[valid_mask]
        
        if len(valid_df) < 2:
            continue
        
        # Use median as threshold
        threshold = valid_df[col].median()
        predictions = (valid_df[col] >= threshold).astype(int)
        true_labels = valid_df[target_col].astype(int)
        
        # Compute metrics
        acc = accuracy_score(true_labels, predictions)
        
        # Try to compute AUC if possible
        try:
            auc = roc_auc_score(true_labels, valid_df[col])
        except:
            auc = np.nan
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        
        results.append({
            "method": col.replace("_score", ""),
            "accuracy": acc,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": threshold
        })
    
    return pd.DataFrame(results).sort_values("f1", ascending=False)


def rank_correlation(df: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
    """Compute Spearman rank correlation between different scoring methods."""
    # Filter numeric columns only
    score_data = df[score_cols].copy()
    
    # Replace inf with NaN
    score_data = score_data.replace([np.inf, -np.inf], np.nan)
    
    # Compute correlation matrix
    corr_matrix = score_data.corr(method='spearman')
    
    return corr_matrix


def generate_comparison_report(
    data_path: str,
    model_path: str,
    output_path: str
) -> Dict:
    """Generate comprehensive comparison report."""
    print("="*60)
    print("Baseline Evaluation Report")
    print("="*60)
    
    # Load data
    print(f"\n1. Loading data from {data_path}")
    data = load_scored_data(data_path)
    print(f"   Loaded {len(data)} CoT traces")
    
    # Compute baseline scores
    print("\n2. Computing baseline scores...")
    baseline_df = compute_baseline_scores(data)
    
    # Compute HMM scores
    print("\n3. Computing HMM scores...")
    hmm_df = compute_hmm_scores(data, model_path)
    
    # Merge dataframes
    df = baseline_df.merge(hmm_df, on="id")
    
    # Get all score columns
    score_cols = [col for col in df.columns if col.endswith("_score") or col == "hmm_score"]
    
    # Correlation analysis
    print("\n4. Computing correlations with correctness...")
    corr_results = correlation_analysis(df, score_cols)
    print("\nCorrelation Results:")
    print(corr_results.to_string(index=False))
    
    # Classification metrics
    print("\n5. Computing classification metrics...")
    clf_results = classification_metrics(df, score_cols)
    print("\nClassification Metrics:")
    print(clf_results.to_string(index=False))
    
    # Rank correlation
    print("\n6. Computing inter-method correlations...")
    rank_corr = rank_correlation(df, score_cols)
    
    # Save results
    print(f"\n7. Saving results to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types
    def convert_to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    report = {
        "summary": {
            "total_samples": int(len(df)),
            "correct_samples": int(df["is_correct"].sum()),
            "incorrect_samples": int((~df["is_correct"]).sum())
        },
        "correlation_analysis": convert_to_native(corr_results.to_dict(orient="records")),
        "classification_metrics": convert_to_native(clf_results.to_dict(orient="records")),
        "rank_correlation": convert_to_native(rank_corr.to_dict()),
        "all_scores": convert_to_native(df.to_dict(orient="records"))
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("Baseline evaluation complete!")
    print("="*60)
    
    return report


def print_summary(report: Dict):
    """Print summary of baseline comparison."""
    print("\n" + "="*60)
    print("SUMMARY: Best Performing Methods")
    print("="*60)
    
    # Best by correlation
    corr_df = pd.DataFrame(report["correlation_analysis"])
    if len(corr_df) > 0:
        best_corr = corr_df.iloc[0]
        print(f"\n📊 Highest Correlation with Correctness:")
        print(f"   Method: {best_corr['method']}")
        print(f"   Correlation: {best_corr['correlation']:.3f} (p={best_corr['p_value']:.4f})")
    
    # Best by F1
    clf_df = pd.DataFrame(report["classification_metrics"])
    if len(clf_df) > 0:
        best_f1 = clf_df.iloc[0]
        print(f"\n🎯 Best Classification Performance:")
        print(f"   Method: {best_f1['method']}")
        print(f"   F1 Score: {best_f1['f1']:.3f}")
        print(f"   Accuracy: {best_f1['accuracy']:.3f}")
        if not np.isnan(best_f1['auc']):
            print(f"   AUC: {best_f1['auc']:.3f}")


if __name__ == "__main__":
    # Example usage
    data_path = "experiments/results/demo_scored.json"
    model_path = "models/demo_hmm.pkl"
    output_path = "experiments/results/baseline_comparison.json"
    
    report = generate_comparison_report(data_path, model_path, output_path)
    print_summary(report)
