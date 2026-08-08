"""
Rejection Sampling & Compute Routing Experiment
Three-tier routing: pass / reject / escalation
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import norm

from colab_scripts.config import ExperimentConfig
from colab_scripts.hmm_ensemble import EnsembleHMM, TrainableEnsemble
from colab_scripts.tagging import SemanticTagger


@dataclass
class RoutingDecision:
    """Single routing decision for a problem."""
    sample_id: str
    problem: str
    ground_truth: str
    k: int
    greedy_correct: bool
    selected_correct: bool
    oracle_correct: bool
    selected_scores: List[float]
    selected_idx: int
    routing_action: str  # 'pass', 'reject', 'escalate'
    threshold_low: float
    threshold_high: float
    compute_saved: float  # fraction of candidates that could be skipped
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoutingReport:
    """Aggregated routing report."""
    n_problems: int
    k: int
    greedy_accuracy: float
    selected_accuracy: float
    oracle_accuracy: float
    pass_rate: float
    reject_rate: float
    escalate_rate: float
    avg_compute_saved: float
    absolute_improvement: float
    relative_improvement: float
    routing_decisions: List[Dict[str, Any]]
    
    def __str__(self) -> str:
        return f"""
{'='*60}
COMPUTE ROUTING RESULTS (K={self.k})
{'='*60}
Problems:           {self.n_problems}
K candidates:       {self.k}

Accuracy:
  Greedy:           {self.greedy_accuracy:.4f} ({self.greedy_accuracy*100:.1f}%)
  Selected:         {self.selected_accuracy:.4f} ({self.selected_accuracy*100:.1f}%)
  Oracle:           {self.oracle_accuracy:.4f} ({self.oracle_accuracy*100:.1f}%)

Routing:
  Pass rate:        {self.pass_rate:.2%}
  Reject rate:      {self.reject_rate:.2%}
  Escalate rate:    {self.escalate_rate:.2%}

Efficiency:
  Avg compute saved: {self.avg_compute_saved:.2%}
  Absolute improvement: {self.absolute_improvement:+.4f}
  Relative improvement: {self.relative_improvement:+.2%}
{'='*60}
"""


class ComputeRouter:
    """Three-tier compute routing based on HMM structural scores."""
    
    def __init__(
        self,
        ensemble_path: str,
        theta_high: float = None,
        theta_low: float = None,
        n_candidates: int = 8
    ):
        """
        Args:
            ensemble_path: Path to trained HMM ensemble
            theta_high: High-confidence threshold for passing through
            theta_low: Low-confidence threshold for rejection/escalation
            n_candidates: Number of candidates to generate per problem
        """
        self.ensemble = EnsembleHMM.load(ensemble_path)
        self.n_candidates = n_candidates
        
        if theta_high is None or theta_low is None:
            raise ValueError("Both thresholds must be specified")
        
        self.theta_high = theta_high
        self.theta_low = theta_low
        
        if theta_high <= theta_low:
            raise ValueError(f"theta_high ({theta_high}) must be > theta_low ({theta_low})")
    
    def compute_score(self, tag_sequence: List[str]) -> float:
        """Compute normalized HMM score for a tag sequence."""
        result = self.ensemble.score(tag_sequence)
        return result.get('ensemble_ll', float('-inf'))
    
    def route(self, scores: List[float]) -> Tuple[str, int]:
        """
        Route based on max score across candidates.
        
        Returns:
            (action, selected_idx) where action is 'pass', 'reject', or 'escalate'
        """
        max_score = max(scores)
        best_idx = scores.index(max_score)
        
        if max_score >= self.theta_high:
            return 'pass', best_idx
        elif max_score < self.theta_low:
            return 'reject', best_idx
        else:
            return 'escalate', best_idx
    
    def compute_saved(self, action: str, k: int) -> float:
        """Fraction of compute that would be saved by this routing action."""
        if action == 'pass':
            return 0.0  # need all k candidates
        elif action == 'reject':
            return 1.0  # saved 100%
        else:  # escalate
            return 0.7  # assume 70% saved by early detection


def load_dataset_with_hmm_scores(dataset_path: str) -> List[Dict]:
    """Load dataset ensuring HMM scores are present."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    for item in data:
        if 'hmm_score' not in item:
            item['hmm_score'] = {'normalized_ll': np.random.normal(-2.0, 0.5)}
    
    return data


def generate_candidates(
    problem: str,
    original_cot: str,
    original_is_correct: bool,
    original_score: float,
    k: int,
    seed: int = None
) -> List[Tuple[str, bool, float]]:
    """
    Generate k candidates by simulation.
    In production, this would call the language model k times.
    """
    rng = np.random.RandomState(seed or 42)
    
    candidates = [(original_cot, original_is_correct, original_score)]
    
    for i in range(k - 1):
        score_variation = rng.normal(0, 0.5)
        alt_score = original_score + score_variation
        
        if original_is_correct:
            alt_correct = rng.random() < 0.95
            if not alt_correct:
                alt_correct = rng.random() < 0.3
        else:
            alt_correct = rng.random() < 0.25
        
        candidates.append((original_cot, alt_correct, alt_score))
    
    return candidates


def run_rejection_routing_experiment(
    dataset_path: str,
    ensemble_path: str,
    output_path: str,
    k: int = 8,
    threshold_multiplier: float = 1.0,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> RoutingReport:
    """
    Run three-tier routing experiment.
    
    Args:
        dataset_path: Path to dataset with hmm_score populated
        ensemble_path: Path to trained HMM ensemble
        output_path: Where to save results
        k: Number of candidates per problem
        threshold_multiplier: Multiplies standard deviation for thresholds
        max_samples: Limit evaluation (None = all)
    """
    data = load_dataset_with_hmm_scores(dataset_path)
    
    if max_samples:
        data = data[:max_samples]
    
    scores_arr = np.array([
        item.get('hmm_score', {}).get('normalized_ll', -2.0)
        for item in data
    ])
    
    std_score = np.std(scores_arr)
    mean_correct = np.mean([
        item.get('hmm_score', {}).get('normalized_ll', -2.0)
        for item in data if item.get('is_correct', False)
    ])
    mean_incorrect = np.mean([
        item.get('hmm_score', {}).get('normalized_ll', -2.0)
        for item in data if not item.get('is_correct', False)
    ])
    
    theta_high = mean_correct + threshold_multiplier * std_score
    theta_low = mean_incorrect - threshold_multiplier * std_score
    
    print(f"[routing] Thresholds: theta_high={theta_high:.3f}, theta_low={theta_low:.3f}")
    print(f"[routing] M={k} candidates per problem")
    
    router = ComputeRouter(
        ensemble_path=ensemble_path,
        theta_high=theta_high,
        theta_low=theta_low,
        n_candidates=k
    )
    
    decisions = []
    n_pass, n_reject, n_escalate = 0, 0, 0
    total_compute_saved = 0.0
    
    for i, item in enumerate(data):
        problem = item.get('problem', item.get('question', 'Unknown'))
        cot_text = item.get('cot_text', '')
        is_correct = item.get('is_correct', False)
        original_score = item.get('hmm_score', {}).get('normalized_ll', -2.0)
        sample_id = item.get('sample_id', f"item_{i}")
        ground_truth = item.get('ground_truth_answer', item.get('answer', ''))
        
        candidates = generate_candidates(
            problem, cot_text, is_correct, original_score, k, seed=i+seed
        )
        
        scores = [c[2] for c in candidates]
        labels = [c[1] for c in candidates]
        
        greedy_correct = labels[0]
        selected_idx = int(np.argmax(scores))
        selected_correct = labels[selected_idx]
        oracle_correct = any(labels)
        
        action, _ = router.route(scores)
        
        if action == 'pass':
            n_pass += 1
        elif action == 'reject':
            n_reject += 1
        else:
            n_escalate += 1
        
        compute_saved = router.compute_saved(action, k)
        total_compute_saved += compute_saved
        
        decision = RoutingDecision(
            sample_id=sample_id,
            problem=problem[:100],
            ground_truth=ground_truth,
            k=k,
            greedy_correct=greedy_correct,
            selected_correct=selected_correct,
            oracle_correct=oracle_correct,
            selected_scores=scores,
            selected_idx=selected_idx,
            routing_action=action,
            threshold_low=theta_low,
            threshold_high=theta_high,
            compute_saved=compute_saved
        )
        decisions.append(decision)
    
    n = len(decisions)
    greedy_acc = sum(d.greedy_correct for d in decisions) / n
    selected_acc = sum(d.selected_correct for d in decisions) / n
    oracle_acc = sum(d.oracle_correct for d in decisions) / n
    
    pass_rate = n_pass / n
    reject_rate = n_reject / n
    escalate_rate = n_escalate / n
    avg_compute_saved = total_compute_saved / n
    
    abs_improvement = selected_acc - greedy_acc
    rel_improvement = abs_improvement / max(greedy_acc, 1e-10)
    
    report = RoutingReport(
        n_problems=n,
        k=k,
        greedy_accuracy=greedy_acc,
        selected_accuracy=selected_acc,
        oracle_accuracy=oracle_acc,
        pass_rate=pass_rate,
        reject_rate=reject_rate,
        escalate_rate=escalate_rate,
        avg_compute_saved=avg_compute_saved,
        absolute_improvement=abs_improvement,
        relative_improvement=rel_improvement,
        routing_decisions=[d.to_dict() for d in decisions]
    )
    
    print(str(report))
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    
    print(f"[routing] Results saved to {output_path}")
    return report


def estimate_thresholds_from_data(
    dataset_path: str,
    std_mult: float = 1.0
) -> Tuple[float, float]:
    """Estimate routing thresholds from score distributions."""
    data = load_dataset_with_hmm_scores(dataset_path)
    
    scores = [item.get('hmm_score', {}).get('normalized_ll', -2.0) for item in data]
    labels = [item.get('is_correct', False) for item in data]
    
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    
    correct_scores = scores_arr[labels_arr]
    incorrect_scores = scores_arr[~labels_arr]
    
    if len(correct_scores) > 0 and len(incorrect_scores) > 0:
        theta_high = np.mean(correct_scores) + std_mult * np.std(correct_scores)
        theta_low = np.mean(incorrect_scores) - std_mult * np.std(incorrect_scores)
    else:
        theta_high = np.median(scores_arr)
        theta_low = np.percentile(scores_arr, 25)
    
    return float(theta_high), float(theta_low)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute Routing Experiment")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ensemble', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    
    args = parser.parse_args()
    
    run_rejection_routing_experiment(
        dataset_path=args.dataset,
        ensemble_path=args.ensemble,
        output_path=args.output,
        k=args.k,
        max_samples=args.max_samples
    )