"""
roscoe_baseline.py  —  Week 3 Deliverable
Real ROSCOE scorer using sentence-transformer embeddings.

Implements 3 of the 4 ROSCOE sub-scores that are feasible without a separate
NLI model server:
  1. Semantic Alignment (SA)     — cosine similarity between consecutive steps
  2. Step Consistency (SC)       — avg sim of each step to the problem statement
  3. Informativeness (INFO)      — avg novelty of each step vs. prior steps

Reference:
    Golovneva et al. (2022) "ROSCOE: A Suite of Metrics for Scoring Step-by-Step Reasoning"
    https://arxiv.org/abs/2212.07919

Install:
    pip install sentence-transformers scikit-learn

Usage:
    scorer = ROSCOEScorer()
    score = scorer.score(problem="...", cot_steps=["Step 1", "Step 2", ...])
    print(score.roscoe_composite)
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("[roscoe] WARNING: sentence-transformers not installed. Run: pip install sentence-transformers")


# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------

@dataclass
class ROSCOEScore:
    semantic_alignment: float           # consecutive-step cosine sim
    step_consistency: float             # step-to-problem cosine sim
    informativeness: float              # step novelty vs prior steps
    roscoe_composite: float             # weighted mean
    n_steps: int = 0
    weights: Dict[str, float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class ROSCOEScorer:
    """
    ROSCOE scorer using sentence embeddings.

    Args:
        model_name: SentenceTransformer model. 'all-MiniLM-L6-v2' is fast + free.
        weights: Weights for composite score. Must sum to 1.0.
        device: 'cpu' | 'cuda' | 'mps'
    """

    DEFAULT_WEIGHTS = {
        "semantic_alignment": 0.35,
        "step_consistency": 0.35,
        "informativeness": 0.30,
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        weights: Optional[Dict[str, float]] = None,
        device: str = "cpu",
    ):
        if not SBERT_AVAILABLE:
            raise ImportError("Run: pip install sentence-transformers")

        self.model = SentenceTransformer(model_name, device=device)
        self.weights = weights or self.DEFAULT_WEIGHTS
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    # ------------------------------------------------------------------ helpers

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized embeddings, shape (N, D)."""
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embs

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of two L2-normalized vectors."""
        return float(np.dot(a, b))

    # ------------------------------------------------------------------ sub-scores

    def _semantic_alignment(self, step_embs: np.ndarray) -> float:
        """
        Avg cosine similarity between consecutive steps.
        High SA = reasoning flows coherently step-to-step.
        """
        if len(step_embs) < 2:
            return 0.0
        sims = [
            self._cosine(step_embs[i], step_embs[i + 1])
            for i in range(len(step_embs) - 1)
        ]
        return float(np.mean(sims))

    def _step_consistency(self, problem_emb: np.ndarray, step_embs: np.ndarray) -> float:
        """
        Avg cosine similarity of each step to the problem statement.
        High SC = steps stay on-topic with the original problem.
        """
        if len(step_embs) == 0:
            return 0.0
        sims = [self._cosine(problem_emb, s) for s in step_embs]
        return float(np.mean(sims))

    def _informativeness(self, step_embs: np.ndarray) -> float:
        """
        Avg novelty of each step relative to all prior steps.
        Measured as 1 - max_cosine_similarity(step_i, {step_1..step_{i-1}}).
        Low INFO = highly redundant / repetitive reasoning.
        """
        if len(step_embs) < 2:
            return 1.0  # single step is maximally novel by default
        novelties = []
        for i in range(1, len(step_embs)):
            prior_sims = [self._cosine(step_embs[i], step_embs[j]) for j in range(i)]
            novelties.append(1.0 - max(prior_sims))
        return float(np.mean(novelties))

    # ------------------------------------------------------------------ main

    def score(
        self,
        problem: str,
        cot_steps: List[str],
    ) -> ROSCOEScore:
        """
        Score a CoT trace.

        Args:
            problem: The original problem statement.
            cot_steps: List of reasoning steps (split on '\\n' or '.' is fine).

        Returns:
            ROSCOEScore with sub-scores and composite.
        """
        if not cot_steps:
            return ROSCOEScore(
                semantic_alignment=0.0,
                step_consistency=0.0,
                informativeness=0.0,
                roscoe_composite=0.0,
                n_steps=0,
            )

        # Embed everything in one batch for efficiency
        all_texts = [problem] + cot_steps
        all_embs = self._embed(all_texts)
        problem_emb = all_embs[0]
        step_embs = all_embs[1:]

        sa = self._semantic_alignment(step_embs)
        sc = self._step_consistency(problem_emb, step_embs)
        info = self._informativeness(step_embs)

        composite = (
            self.weights["semantic_alignment"] * sa
            + self.weights["step_consistency"] * sc
            + self.weights["informativeness"] * info
        )

        return ROSCOEScore(
            semantic_alignment=sa,
            step_consistency=sc,
            informativeness=info,
            roscoe_composite=composite,
            n_steps=len(cot_steps),
            weights=self.weights,
        )

    def score_text(self, problem: str, cot_text: str, sep: str = "\n") -> ROSCOEScore:
        """
        Convenience wrapper: split cot_text on `sep` then score.
        Filters out empty strings after splitting.
        """
        steps = [s.strip() for s in cot_text.split(sep) if s.strip()]
        return self.score(problem, steps)


# ---------------------------------------------------------------------------
# Batch scoring against the v2 dataset
# ---------------------------------------------------------------------------

def run_roscoe_on_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    model_name: str = "all-MiniLM-L6-v2",
    step_sep: str = "\n",
) -> Dict[str, Any]:
    """
    Load a v2 ScoredTrace dataset, run ROSCOE on every trace, and save results.

    Returns a summary dict with AUC and correlation vs. is_correct.
    """
    from sklearn.metrics import roc_auc_score
    from scipy.stats import pearsonr

    scorer = ROSCOEScorer(model_name=model_name)

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    results = []
    roscoe_scores: List[float] = []
    labels: List[int] = []

    for i, item in enumerate(raw):
        problem = item.get("problem", "")
        cot_text = item.get("cot_text", "")
        is_correct = int(item.get("is_correct", False))

        sc = scorer.score_text(problem, cot_text, sep=step_sep)

        item["roscoe_score"] = sc.to_dict()
        results.append(item)
        roscoe_scores.append(sc.roscoe_composite)
        labels.append(is_correct)

        if (i + 1) % 50 == 0:
            print(f"  [ROSCOE] Scored {i+1}/{len(raw)} traces...")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Summary stats
    roscoe_arr = np.array(roscoe_scores)
    labels_arr = np.array(labels)

    auc = roc_auc_score(labels_arr, roscoe_arr) if len(set(labels_arr)) > 1 else float("nan")
    r, p = pearsonr(roscoe_arr, labels_arr.astype(float))

    summary = {
        "method": "ROSCOE",
        "n_samples": len(results),
        "auc": round(float(auc), 4),
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 5),
        "mean_correct": float(roscoe_arr[labels_arr == 1].mean()),
        "mean_incorrect": float(roscoe_arr[labels_arr == 0].mean()),
    }

    print("\n" + "="*55)
    print("ROSCOE RESULTS")
    print("="*55)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not SBERT_AVAILABLE:
        print("Install sentence-transformers first: pip install sentence-transformers")
        exit(1)

    scorer = ROSCOEScorer()

    # Good CoT (should score higher)
    good_steps = [
        "The problem asks how many apples John has after giving some away.",
        "John starts with 5 apples.",
        "He gives away 2 apples. So we compute 5 - 2 = 3.",
        "Therefore, John has 3 apples.",
    ]

    # Bad CoT (incoherent, off-topic)
    bad_steps = [
        "Apples are a fruit.",
        "The sky is blue sometimes when weather permits.",
        "Multiply 7 by 99 to get something.",
        "The answer is probably 42.",
    ]

    problem = "If John has 5 apples and gives 2 away, how many does he have?"

    good_score = scorer.score(problem, good_steps)
    bad_score = scorer.score(problem, bad_steps)

    print(f"Good CoT — composite: {good_score.roscoe_composite:.4f}")
    print(f"  SA={good_score.semantic_alignment:.3f}, SC={good_score.step_consistency:.3f}, INFO={good_score.informativeness:.3f}")
    print(f"Bad CoT  — composite: {bad_score.roscoe_composite:.4f}")
    print(f"  SA={bad_score.semantic_alignment:.3f}, SC={bad_score.step_consistency:.3f}, INFO={bad_score.informativeness:.3f}")

    assert good_score.roscoe_composite > bad_score.roscoe_composite, \
        "FAIL: Good CoT should score higher than bad CoT"
    print("\n✅ roscoe_baseline.py self-test passed.")
