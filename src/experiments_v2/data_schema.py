"""
data_schema.py  —  Week 1 Deliverable
Standardized JSON schema + dataclasses for scaled v2 dataset.

Every trace stored in data/v2/ should be serialized via ScoredTrace.to_dict()
so that all downstream scripts (roscoe, picsar, neural_hmm, rejection_sampling)
share a single, validated schema.

Usage:
    trace = ScoredTrace(
        sample_id="gsm8k_0001",
        dataset="gsm8k",
        model_family="qwen",
        model_name="Qwen/Qwen-0.5B",
        problem="If John has 5 apples...",
        ground_truth_answer="8",
        cot_text="First I note... Then compute...",
        model_answer="8",
        is_correct=True,
        tag_sequence=["PR", "DV", "CA", "CO"],
    )
    trace.hmm_score = HMMScoreRecord(log_likelihood=-3.2, normalized_ll=-0.8, ...)
    trace.to_json("data/v2/gsm8k_0001.json")
"""

from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime


# ---------------------------------------------------------------------------
# Sub-records
# ---------------------------------------------------------------------------

@dataclass
class HMMScoreRecord:
    """HMM ensemble scorer output."""
    log_likelihood: float
    normalized_ll: float            # ll / sequence_length
    transition_entropy: float
    anomaly_count: int
    ensemble_size: int = 12
    hidden_states_used: List[int] = field(default_factory=lambda: [3, 5, 8, 12])
    viterbi_path: List[int] = field(default_factory=list)


@dataclass
class ROSCOEScoreRecord:
    """ROSCOE scorer output (Week 3)."""
    semantic_alignment: float       # cosine sim between consecutive steps
    step_consistency: float         # entailment score premise→step
    informativeness: float          # avg step novelty vs. problem statement
    roscoe_composite: float         # weighted average of the three


@dataclass
class PiCSARScoreRecord:
    """PiCSAR scorer output (Week 4)."""
    joint_log_likelihood: float     # LM logprob of full CoT given problem
    normalized_jll: float           # per-token normalized
    perplexity: float
    model_used: str = "distilgpt2"


@dataclass
class NeuralHMMScoreRecord:
    """Neural-HMM hybrid scorer output (Week 7–8)."""
    neural_hmm_ll: float
    normalized_neural_ll: float
    neural_tag_confidences: List[float] = field(default_factory=list)
    backbone_model: str = "distilbert-base-uncased"


@dataclass
class RejectionSamplingRecord:
    """Rejection-sampling experiment record (Week 9)."""
    candidate_scores: List[float]    # HMM scores for K candidates
    selected_candidate_idx: int
    selected_is_correct: bool
    greedy_is_correct: bool          # baseline comparison
    k_candidates: int = 8


# ---------------------------------------------------------------------------
# Core ScoredTrace dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScoredTrace:
    """
    Single Chain-of-Thought trace with all scoring metadata.
    This is the canonical schema for data/v2/*.json.
    """
    # ---- Identity ----
    sample_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    dataset: str = ""                   # "gsm8k" | "arc_challenge" | "math"
    split: str = "test"                 # "train" | "test" | "validation"
    domain: str = ""                    # "math" | "science" | "mixed"

    # ---- Model info ----
    model_family: str = ""              # "qwen" | "llama" | "mistral" | "phi"
    model_name: str = ""                # HuggingFace model identifier
    model_params_b: float = 0.0         # parameter count in billions

    # ---- Problem ----
    problem: str = ""
    ground_truth_answer: str = ""

    # ---- CoT ----
    cot_text: str = ""
    model_answer: str = ""
    is_correct: bool = False
    tag_sequence: List[str] = field(default_factory=list)
    tagging_method: str = "rule_based"  # "rule_based" | "neural"

    # ---- Scores (populated incrementally) ----
    hmm_score: Optional[HMMScoreRecord] = None
    roscoe_score: Optional[ROSCOEScoreRecord] = None
    picsar_score: Optional[PiCSARScoreRecord] = None
    neural_hmm_score: Optional[NeuralHMMScoreRecord] = None
    rejection_sampling: Optional[RejectionSamplingRecord] = None

    # ---- Metadata ----
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    generation_seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ I/O

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dict (JSON-safe)."""
        return asdict(self)

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScoredTrace":
        """Deserialize from plain dict."""
        trace = cls()
        for k, v in d.items():
            if k == "hmm_score" and v:
                setattr(trace, k, HMMScoreRecord(**v))
            elif k == "roscoe_score" and v:
                setattr(trace, k, ROSCOEScoreRecord(**v))
            elif k == "picsar_score" and v:
                setattr(trace, k, PiCSARScoreRecord(**v))
            elif k == "neural_hmm_score" and v:
                setattr(trace, k, NeuralHMMScoreRecord(**v))
            elif k == "rejection_sampling" and v:
                setattr(trace, k, RejectionSamplingRecord(**v))
            else:
                if hasattr(trace, k):
                    setattr(trace, k, v)
        return trace

    @classmethod
    def from_json(cls, path: str | Path) -> "ScoredTrace":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Dataset-level I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path) -> List[ScoredTrace]:
    """Load a JSON list of traces from a dataset file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [ScoredTrace.from_dict(d) for d in raw]


def save_dataset(traces: List[ScoredTrace], path: str | Path, indent: int = 2) -> None:
    """Save a list of traces to a dataset JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in traces], f, indent=indent)
    print(f"[data_schema] Saved {len(traces)} traces → {path}")


def validate_dataset(traces: List[ScoredTrace]) -> Dict[str, Any]:
    """
    Run schema validation checks and return a summary report.
    Call this after loading any dataset to catch missing fields early.
    """
    issues = []
    stats: Dict[str, Any] = {
        "total": len(traces),
        "by_dataset": {},
        "by_model": {},
        "by_domain": {},
        "correct": 0,
        "has_hmm_score": 0,
        "has_roscoe_score": 0,
        "has_picsar_score": 0,
        "has_neural_hmm": 0,
        "empty_tag_sequences": 0,
    }

    for i, t in enumerate(traces):
        # Required fields
        if not t.cot_text:
            issues.append(f"[{i}] {t.sample_id}: empty cot_text")
        if not t.tag_sequence:
            stats["empty_tag_sequences"] += 1

        # Counters
        if t.is_correct:
            stats["correct"] += 1
        if t.hmm_score:
            stats["has_hmm_score"] += 1
        if t.roscoe_score:
            stats["has_roscoe_score"] += 1
        if t.picsar_score:
            stats["has_picsar_score"] += 1
        if t.neural_hmm_score:
            stats["has_neural_hmm"] += 1

        stats["by_dataset"][t.dataset] = stats["by_dataset"].get(t.dataset, 0) + 1
        stats["by_model"][t.model_name] = stats["by_model"].get(t.model_name, 0) + 1
        stats["by_domain"][t.domain] = stats["by_domain"].get(t.domain, 0) + 1

    stats["accuracy"] = round(stats["correct"] / max(len(traces), 1), 4)
    stats["issues"] = issues
    return stats


# ---------------------------------------------------------------------------
# Migration helper: convert old v1 format → ScoredTrace
# ---------------------------------------------------------------------------

def migrate_v1_trace(old: Dict[str, Any]) -> ScoredTrace:
    """
    Convert an old-format trace dict (from run_experiments.py output)
    to the new ScoredTrace schema.

    Old keys expected:
        question, answer, cot, correct, tags, hmm_scores
    """
    trace = ScoredTrace(
        dataset=old.get("dataset", "gsm8k"),
        domain=old.get("domain", "math"),
        model_family=old.get("model_family", "qwen"),
        model_name=old.get("model", "Qwen/Qwen-0.5B"),
        problem=old.get("question", old.get("problem", "")),
        ground_truth_answer=str(old.get("answer", "")),
        cot_text=old.get("cot", old.get("cot_text", "")),
        model_answer=str(old.get("predicted_answer", "")),
        is_correct=bool(old.get("correct", old.get("is_correct", False))),
        tag_sequence=old.get("tags", old.get("tag_sequence", [])),
        tagging_method=old.get("tagging_method", "rule_based"),
    )

    # Migrate old HMM scores
    old_hmm = old.get("hmm_scores", {})
    if old_hmm:
        trace.hmm_score = HMMScoreRecord(
            log_likelihood=old_hmm.get("log_likelihood", 0.0),
            normalized_ll=old_hmm.get("normalized_ll", 0.0),
            transition_entropy=old_hmm.get("transition_entropy", 0.0),
            anomaly_count=old_hmm.get("anomaly_count", 0),
        )

    return trace


def migrate_v1_dataset(v1_path: str | Path, v2_path: str | Path) -> None:
    """Migrate a full v1 dataset file to v2 ScoredTrace format."""
    with open(v1_path, "r", encoding="utf-8") as f:
        v1_data = json.load(f)

    traces = [migrate_v1_trace(d) for d in v1_data]
    save_dataset(traces, v2_path)
    print(f"[migrate] {len(traces)} traces: {v1_path} → {v2_path}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Round-trip test
    trace = ScoredTrace(
        sample_id="test_001",
        dataset="gsm8k",
        domain="math",
        model_family="qwen",
        model_name="Qwen/Qwen-0.5B",
        problem="If John has 5 apples and gives 2 away, how many does he have?",
        ground_truth_answer="3",
        cot_text="John starts with 5. He gives away 2. 5 - 2 = 3. The answer is 3.",
        model_answer="3",
        is_correct=True,
        tag_sequence=["PR", "DV", "CA", "CO"],
        tagging_method="rule_based",
    )
    trace.hmm_score = HMMScoreRecord(
        log_likelihood=-3.21,
        normalized_ll=-0.80,
        transition_entropy=0.14,
        anomaly_count=0,
    )

    d = trace.to_dict()
    restored = ScoredTrace.from_dict(d)
    assert restored.sample_id == trace.sample_id
    assert restored.hmm_score.log_likelihood == trace.hmm_score.log_likelihood

    # Validate dataset
    report = validate_dataset([trace])
    print("Schema validation report:")
    print(json.dumps(report, indent=2))
    print("\n✅ data_schema.py self-test passed.")
