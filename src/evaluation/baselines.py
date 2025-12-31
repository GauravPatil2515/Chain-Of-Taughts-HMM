"""
Baseline Methods for CoT Scoring
Simple heuristic-based scorers to compare against HMM approach.
"""

import re
import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class BaselineScore:
    """Container for baseline scoring results."""
    method: str
    score: float
    components: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "score": self.score,
            "components": self.components
        }


class LengthBaseline:
    """Score based on CoT length (longer = better reasoning)."""
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Scaling factor for length score
        """
        self.alpha = alpha
        self.name = "length_baseline"
    
    def score(self, cot_text: str, tag_sequence: List[str] = None) -> BaselineScore:
        """Score based on character count and step count."""
        char_count = len(cot_text)
        step_count = len(tag_sequence) if tag_sequence else len(cot_text.split('.'))
        
        # Normalize: longer CoTs score higher, but with diminishing returns
        char_score = np.log1p(char_count) * 0.1
        step_score = np.log1p(step_count) * 0.5
        
        total_score = (char_score + step_score) * self.alpha
        
        return BaselineScore(
            method=self.name,
            score=total_score,
            components={
                "char_count": char_count,
                "step_count": step_count,
                "char_score": char_score,
                "step_score": step_score
            }
        )


class TagDiversityBaseline:
    """Score based on diversity of reasoning tags."""
    
    def __init__(self):
        self.name = "tag_diversity_baseline"
    
    def score(self, cot_text: str, tag_sequence: List[str] = None) -> BaselineScore:
        """Score based on variety of reasoning steps."""
        if not tag_sequence or len(tag_sequence) == 0:
            return BaselineScore(
                method=self.name,
                score=0.0,
                components={"unique_tags": 0, "total_tags": 0, "diversity_ratio": 0.0}
            )
        
        unique_tags = len(set(tag_sequence))
        total_tags = len(tag_sequence)
        diversity_ratio = unique_tags / total_tags if total_tags > 0 else 0
        
        # Score: reward diversity but penalize too few steps
        base_score = diversity_ratio * np.log1p(total_tags)
        
        return BaselineScore(
            method=self.name,
            score=base_score,
            components={
                "unique_tags": unique_tags,
                "total_tags": total_tags,
                "diversity_ratio": diversity_ratio
            }
        )


class StructureBaseline:
    """Score based on structural features (calculations, conclusions, etc.)."""
    
    def __init__(self):
        self.name = "structure_baseline"
    
    def score(self, cot_text: str, tag_sequence: List[str] = None) -> BaselineScore:
        """Score based on presence of key structural elements."""
        text_lower = cot_text.lower()
        
        # Count structural indicators
        has_calculation = bool(re.search(r'\d+\s*[+\-×*/=]\s*\d+', cot_text))
        has_conclusion = any(p in text_lower for p in ["therefore", "answer is", "final answer", "thus"])
        has_reasoning = any(p in text_lower for p in ["because", "since", "so", "hence", "which means"])
        has_numbers = bool(re.search(r'\d+', cot_text))
        
        # Count tag-based features
        has_PR = "PR" in tag_sequence if tag_sequence else False
        has_CA = "CA" in tag_sequence if tag_sequence else False
        has_CO = "CO" in tag_sequence if tag_sequence else False
        
        # Scoring
        structure_score = sum([
            1.0 if has_calculation else 0,
            1.5 if has_conclusion else 0,
            0.5 if has_reasoning else 0,
            0.5 if has_numbers else 0,
            1.0 if has_PR else 0,
            1.0 if has_CA else 0,
            1.5 if has_CO else 0,
        ])
        
        return BaselineScore(
            method=self.name,
            score=structure_score,
            components={
                "has_calculation": has_calculation,
                "has_conclusion": has_conclusion,
                "has_reasoning": has_reasoning,
                "has_numbers": has_numbers,
                "has_PR": has_PR,
                "has_CA": has_CA,
                "has_CO": has_CO
            }
        )


class TransitionBaseline:
    """Score based on expected tag transitions (simple rule-based)."""
    
    def __init__(self):
        self.name = "transition_baseline"
        # Define expected good transitions
        self.good_transitions = {
            ("PR", "DV"), ("PR", "RF"), ("PR", "CA"), ("PR", "EQ"),  # Problem -> Setup
            ("DV", "EQ"), ("DV", "CA"),  # Variable -> Equation/Calc
            ("EQ", "CA"), ("CA", "CA"),  # Equation -> Calc
            ("RF", "CA"), ("RF", "EQ"),  # Recall -> Apply
            ("CA", "IN"), ("CA", "VE"), ("CA", "CO"),  # Calc -> Result
            ("IN", "CA"), ("IN", "VE"), ("IN", "CO"),  # Inference -> Next step
            ("VE", "CO"),  # Verify -> Conclude
        }
    
    def score(self, cot_text: str, tag_sequence: List[str] = None) -> BaselineScore:
        """Score based on transition quality."""
        if not tag_sequence or len(tag_sequence) < 2:
            return BaselineScore(
                method=self.name,
                score=0.0,
                components={"good_transitions": 0, "total_transitions": 0, "ratio": 0.0}
            )
        
        good_count = 0
        total_transitions = len(tag_sequence) - 1
        
        for i in range(len(tag_sequence) - 1):
            transition = (tag_sequence[i], tag_sequence[i + 1])
            if transition in self.good_transitions:
                good_count += 1
        
        ratio = good_count / total_transitions if total_transitions > 0 else 0
        score = ratio * np.log1p(total_transitions) * 2
        
        return BaselineScore(
            method=self.name,
            score=score,
            components={
                "good_transitions": good_count,
                "total_transitions": total_transitions,
                "ratio": ratio
            }
        )


class CombinedBaseline:
    """Combined baseline using weighted average of all heuristics."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.name = "combined_baseline"
        self.length = LengthBaseline()
        self.diversity = TagDiversityBaseline()
        self.structure = StructureBaseline()
        self.transition = TransitionBaseline()
        
        # Default weights
        self.weights = weights or {
            "length": 0.2,
            "diversity": 0.2,
            "structure": 0.3,
            "transition": 0.3
        }
    
    def score(self, cot_text: str, tag_sequence: List[str] = None) -> BaselineScore:
        """Compute weighted combination of all baseline scores."""
        scores = {
            "length": self.length.score(cot_text, tag_sequence),
            "diversity": self.diversity.score(cot_text, tag_sequence),
            "structure": self.structure.score(cot_text, tag_sequence),
            "transition": self.transition.score(cot_text, tag_sequence)
        }
        
        # Normalize scores to [0, 1] range approximately
        normalized = {
            "length": min(scores["length"].score / 5.0, 1.0),
            "diversity": min(scores["diversity"].score / 3.0, 1.0),
            "structure": min(scores["structure"].score / 6.0, 1.0),
            "transition": min(scores["transition"].score / 3.0, 1.0)
        }
        
        # Weighted sum
        combined_score = sum(
            normalized[key] * self.weights[key] 
            for key in self.weights
        )
        
        return BaselineScore(
            method=self.name,
            score=combined_score,
            components={
                "length_score": scores["length"].score,
                "diversity_score": scores["diversity"].score,
                "structure_score": scores["structure"].score,
                "transition_score": scores["transition"].score,
                "combined_score": combined_score
            }
        )


class RandomBaseline:
    """Random baseline for sanity check."""
    
    def __init__(self, seed: int = 42):
        self.name = "random_baseline"
        self.rng = np.random.RandomState(seed)
    
    def score(self, cot_text: str, tag_sequence: List[str] = None) -> BaselineScore:
        """Return random score."""
        score = self.rng.uniform(0, 1)
        return BaselineScore(
            method=self.name,
            score=score,
            components={"random_value": score}
        )


def score_with_all_baselines(cot_text: str, tag_sequence: List[str] = None) -> Dict[str, BaselineScore]:
    """Score a CoT with all baseline methods."""
    baselines = {
        "length": LengthBaseline(),
        "diversity": TagDiversityBaseline(),
        "structure": StructureBaseline(),
        "transition": TransitionBaseline(),
        "combined": CombinedBaseline(),
        "random": RandomBaseline()
    }
    
    results = {}
    for name, baseline in baselines.items():
        results[name] = baseline.score(cot_text, tag_sequence)
    
    return results
