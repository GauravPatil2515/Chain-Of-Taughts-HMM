"""
HMM Scorer
Score CoT sequences using trained HMM model.
"""

import json
import argparse
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Union
from dataclasses import dataclass

# Tag vocabulary (must match trainer.py)
TAGS = ["PR", "DV", "EQ", "CA", "IN", "RF", "VE", "CO"]
TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}


@dataclass
class CoTScore:
    """Container for CoT scoring results."""
    log_likelihood: float
    normalized_ll: float  # Per-step log-likelihood
    transition_entropy: float
    anomaly_count: int
    anomalies: List[Dict]
    path_length: int
    viterbi_path: List[int]
    
    def to_dict(self) -> Dict:
        return {
            "log_likelihood": self.log_likelihood,
            "normalized_ll": self.normalized_ll,
            "transition_entropy": self.transition_entropy,
            "anomaly_count": self.anomaly_count,
            "anomalies": self.anomalies,
            "path_length": self.path_length,
            "viterbi_path": self.viterbi_path,
        }


class HMMScorer:
    """Score CoT sequences using trained HMM."""
    
    def __init__(self, model_path: str, anomaly_threshold: float = 0.05):
        """
        Initialize scorer.
        
        Args:
            model_path: Path to trained HMM model
            anomaly_threshold: Transition probability below which to flag anomaly
        """
        self.model = joblib.load(model_path)
        self.anomaly_threshold = anomaly_threshold
    
    def encode_sequence(self, tag_sequence: List[str]) -> np.ndarray:
        """Convert tag sequence to integer array."""
        encoded = [TAG_TO_ID.get(tag, TAG_TO_ID["IN"]) for tag in tag_sequence]
        return np.array(encoded).reshape(-1, 1)
    
    def compute_log_likelihood(self, tag_sequence: List[str]) -> float:
        """Compute log-likelihood of sequence."""
        if len(tag_sequence) < 1:
            return float('-inf')
        
        X = self.encode_sequence(tag_sequence)
        return self.model.score(X)
    
    def compute_transition_entropy(self) -> float:
        """Compute entropy of transition matrix."""
        A = self.model.transmat_
        # Add small epsilon to avoid log(0)
        A_safe = np.clip(A, 1e-10, 1.0)
        entropy = -np.sum(A * np.log(A_safe))
        return float(entropy)
    
    def detect_anomalies(self, tag_sequence: List[str]) -> List[Dict]:
        """Detect low-probability transitions."""
        anomalies = []
        A = self.model.transmat_
        
        # Get Viterbi path (most likely hidden state sequence)
        X = self.encode_sequence(tag_sequence)
        _, state_path = self.model.decode(X)
        
        for i in range(len(state_path) - 1):
            from_state = state_path[i]
            to_state = state_path[i + 1]
            prob = A[from_state, to_state]
            
            if prob < self.anomaly_threshold:
                anomalies.append({
                    "position": i,
                    "from_tag": tag_sequence[i],
                    "to_tag": tag_sequence[i + 1],
                    "from_state": int(from_state),
                    "to_state": int(to_state),
                    "transition_prob": float(prob),
                })
        
        return anomalies
    
    def get_viterbi_path(self, tag_sequence: List[str]) -> List[int]:
        """Get most likely hidden state sequence."""
        if len(tag_sequence) < 1:
            return []
        
        X = self.encode_sequence(tag_sequence)
        _, path = self.model.decode(X)
        return path.tolist()
    
    def score(self, tag_sequence: List[str]) -> CoTScore:
        """
        Compute all scores for a tag sequence.
        
        Args:
            tag_sequence: List of tags (e.g., ["PR", "CA", "CO"])
        
        Returns:
            CoTScore object with all metrics
        """
        if len(tag_sequence) < 1:
            return CoTScore(
                log_likelihood=float('-inf'),
                normalized_ll=float('-inf'),
                transition_entropy=0.0,
                anomaly_count=0,
                anomalies=[],
                path_length=0,
                viterbi_path=[],
            )
        
        ll = self.compute_log_likelihood(tag_sequence)
        anomalies = self.detect_anomalies(tag_sequence)
        viterbi_path = self.get_viterbi_path(tag_sequence)
        
        return CoTScore(
            log_likelihood=ll,
            normalized_ll=ll / len(tag_sequence),
            transition_entropy=self.compute_transition_entropy(),
            anomaly_count=len(anomalies),
            anomalies=anomalies,
            path_length=len(tag_sequence),
            viterbi_path=viterbi_path,
        )


def score_cot_file(
    model_path: str,
    input_path: str,
    output_path: str
) -> Dict:
    """Score all CoTs in a file."""
    scorer = HMMScorer(model_path)
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = []
    scores_correct = []
    scores_incorrect = []
    
    for item in data:
        tag_seq = item.get("tag_sequence", [])
        score = scorer.score(tag_seq)
        
        # Track by correctness
        if item.get("is_correct", False):
            scores_correct.append(score.log_likelihood)
        else:
            scores_incorrect.append(score.log_likelihood)
        
        results.append({
            **item,
            "hmm_scores": score.to_dict(),
        })
    
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Compute summary statistics
    def safe_stats(values):
        if not values:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    
    stats = {
        "total_scored": len(results),
        "correct_count": len(scores_correct),
        "incorrect_count": len(scores_incorrect),
        "correct_scores": safe_stats(scores_correct),
        "incorrect_scores": safe_stats(scores_incorrect),
    }
    
    # Print summary
    print("\n" + "="*50)
    print("SCORING SUMMARY")
    print("="*50)
    print(f"Total CoTs scored: {stats['total_scored']}")
    print(f"Correct: {stats['correct_count']}, Incorrect: {stats['incorrect_count']}")
    
    if scores_correct:
        print(f"\nCorrect CoTs - Avg LL: {stats['correct_scores']['mean']:.2f}")
    if scores_incorrect:
        print(f"Incorrect CoTs - Avg LL: {stats['incorrect_scores']['mean']:.2f}")
    
    if scores_correct and scores_incorrect:
        diff = stats['correct_scores']['mean'] - stats['incorrect_scores']['mean']
        print(f"\nScore difference (correct - incorrect): {diff:.2f}")
        if diff > 0:
            print("✓ Higher scores for correct CoTs (as expected)")
        else:
            print("✗ Warning: Incorrect CoTs have higher scores")
    
    print(f"\nResults saved to {output_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Score CoTs using HMM")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained HMM model")
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSON file with tagged CoTs")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file with scores")
    
    args = parser.parse_args()
    
    score_cot_file(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
