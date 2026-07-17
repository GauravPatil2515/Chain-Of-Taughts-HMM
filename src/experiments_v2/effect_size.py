"""
effect_size.py  —  Week 6 Deliverable
Effect size reporting: Cohen's d, Cliff's delta, and FDR tracking.

Addresses reviewer flags:
  1. "Add Cohen's d alongside p-values for entropy comparison (0.14 vs 0.23)"
  2. "Report the total number of comparisons in your BH-FDR correction"

Usage:
    d = cohens_d(correct_entropies, incorrect_entropies)
    delta = cliffs_delta(hmm_scores_correct, hmm_scores_incorrect)
    tracker = FDRTracker()
    tracker.add("entropy_comparison", p_value=0.01)
    corrected = tracker.correct()
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------

@dataclass
class CohensDResult:
    d: float
    interpretation: str
    n1: int
    n2: int
    mean1: float
    mean2: float
    pooled_std: float

    def __str__(self) -> str:
        return f"Cohen's d = {self.d:.4f} ({self.interpretation})"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def cohens_d(group1, group2) -> CohensDResult:
    """Cohen's d. Positive = group1 > group2."""
    a = np.array(group1, dtype=float)
    b = np.array(group2, dtype=float)
    n1, n2 = len(a), len(b)
    mean1, mean2 = np.mean(a), np.mean(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (mean1 - mean2) / pooled_std if pooled_std > 1e-12 else 0.0
    abs_d = abs(d)
    interp = "negligible" if abs_d < 0.2 else ("small" if abs_d < 0.5 else ("medium" if abs_d < 0.8 else "large"))
    return CohensDResult(d=float(d), interpretation=interp, n1=n1, n2=n2,
                         mean1=float(mean1), mean2=float(mean2), pooled_std=float(pooled_std))


# ---------------------------------------------------------------------------
# Cliff's delta
# ---------------------------------------------------------------------------

@dataclass
class CliffsDeltaResult:
    delta: float
    interpretation: str
    n1: int
    n2: int

    def __str__(self) -> str:
        return f"Cliff's delta = {self.delta:.4f} ({self.interpretation})"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def cliffs_delta(group1, group2) -> CliffsDeltaResult:
    """Cliff's delta: P(X>Y) - P(X<Y). Range [-1,1]."""
    a = np.array(group1, dtype=float)
    b = np.array(group2, dtype=float)
    dominance = np.sign(a[:, np.newaxis] - b[np.newaxis, :])
    delta = float(np.mean(dominance))
    abs_d = abs(delta)
    interp = "negligible" if abs_d < 0.147 else ("small" if abs_d < 0.330 else ("medium" if abs_d < 0.474 else "large"))
    return CliffsDeltaResult(delta=delta, interpretation=interp, n1=len(a), n2=len(b))


# ---------------------------------------------------------------------------
# FDR Tracker (Benjamini-Hochberg)
# ---------------------------------------------------------------------------

@dataclass
class ComparisonRecord:
    name: str
    p_value: float
    description: str = ""
    section: str = ""


@dataclass
class FDRResult:
    name: str
    original_p: float
    adjusted_p: float
    rejected: bool
    rank: int
    threshold: float


class FDRTracker:
    """
    Track all statistical comparisons and apply Benjamini-Hochberg FDR.

    USAGE:
        tracker = FDRTracker(alpha=0.05)
        tracker.add("entropy_comparison", p_value=0.01, section="IV-A")
        tracker.add("auc_vs_random", p_value=0.003, section="IV-B")
        results = tracker.correct()
        tracker.save("results/v2/fdr_corrections.json")
        print(tracker.paper_text())
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.comparisons: List[ComparisonRecord] = []
        self._corrected: Optional[List[FDRResult]] = None

    def add(self, name: str, p_value: float, description: str = "", section: str = "") -> None:
        self.comparisons.append(ComparisonRecord(
            name=name, p_value=p_value,
            description=description, section=section))
        self._corrected = None

    def correct(self) -> List[FDRResult]:
        if not self.comparisons:
            return []
        m = len(self.comparisons)
        sorted_comps = sorted(enumerate(self.comparisons), key=lambda x: x[1].p_value)
        results = []
        for rank, (_, comp) in enumerate(sorted_comps, start=1):
            bh_threshold = (rank / m) * self.alpha
            adjusted_p = min(comp.p_value * m / rank, 1.0)
            results.append(FDRResult(
                name=comp.name, original_p=comp.p_value,
                adjusted_p=adjusted_p, rejected=(comp.p_value <= bh_threshold),
                rank=rank, threshold=bh_threshold))
        self._corrected = results
        return results

    def save(self, path) -> None:
        results = self.correct()
        output = {
            "alpha": self.alpha,
            "total_comparisons": len(self.comparisons),
            "n_significant_before": sum(c.p_value < self.alpha for c in self.comparisons),
            "n_significant_after": sum(r.rejected for r in results),
            "method": "Benjamini-Hochberg FDR",
            "comparisons": [asdict(c) for c in self.comparisons],
            "corrected_results": [asdict(r) for r in results],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[FDR] Saved {len(results)} corrections → {path}")

    def paper_text(self) -> str:
        results = self.correct()
        m = len(self.comparisons)
        n_sig = sum(r.rejected for r in results)
        lines = [
            f"All p-values are corrected for multiple comparisons using Benjamini-Hochberg",
            f"FDR (alpha=0.05). A total of {m} comparisons were performed;",
            f"{n_sig} of {m} remain significant after correction.",
            "", "Individual comparisons (original p | adjusted p | significant):",
        ]
        for r in results:
            mark = "✓" if r.rejected else "✗"
            lines.append(f"  [{mark}] {r.name}: p={r.original_p:.4f} → p_adj={r.adjusted_p:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def generate_effect_size_report(dataset_path, output_path) -> Dict[str, Any]:
    """Load v2 dataset, compute Cohen's d and Cliff's delta for all key comparisons."""
    with open(dataset_path, "r") as f:
        raw = json.load(f)

    correct_ents, incorrect_ents = [], []
    correct_lls, incorrect_lls = [], []

    for item in raw:
        hs = item.get("hmm_score", {})
        lbl = int(item.get("is_correct", False))
        if hs:
            ent = hs.get("transition_entropy")
            ll = hs.get("normalized_ll")
            if ent is not None:
                (correct_ents if lbl else incorrect_ents).append(float(ent))
            if ll is not None and np.isfinite(float(ll)):
                (correct_lls if lbl else incorrect_lls).append(float(ll))

    results = {}
    if correct_ents and incorrect_ents:
        d = cohens_d(correct_ents, incorrect_ents)
        delta = cliffs_delta(correct_ents, incorrect_ents)
        _, p = scipy_stats.mannwhitneyu(correct_ents, incorrect_ents, alternative="two-sided")
        results["structural_entropy"] = {
            "cohens_d": d.to_dict(), "cliffs_delta": delta.to_dict(),
            "mann_whitney_p": float(p),
            "mean_correct": float(np.mean(correct_ents)),
            "mean_incorrect": float(np.mean(incorrect_ents)),
        }
        print(f"Entropy: {d}  |  {delta}")

    if correct_lls and incorrect_lls:
        d = cohens_d(correct_lls, incorrect_lls)
        delta = cliffs_delta(correct_lls, incorrect_lls)
        results["hmm_ll"] = {
            "cohens_d": d.to_dict(), "cliffs_delta": delta.to_dict(),
            "mean_correct": float(np.mean(correct_lls)),
            "mean_incorrect": float(np.mean(incorrect_lls)),
        }
        print(f"HMM LL: {d}  |  {delta}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[effect_size] Saved → {output_path}")
    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(42)
    correct_ents = rng.normal(0.14, 0.05, 100).tolist()
    incorrect_ents = rng.normal(0.23, 0.07, 35).tolist()

    d = cohens_d(correct_ents, incorrect_ents)
    delta = cliffs_delta(correct_ents, incorrect_ents)

    print("=== Effect Size Self-Test ===")
    print(f"Entropy (0.14 vs 0.23): {d}")
    print(f"                        {delta}")

    tracker = FDRTracker(alpha=0.05)
    tracker.add("entropy_comparison",  p_value=0.01,  section="IV-A")
    tracker.add("auc_vs_random",       p_value=0.003, section="IV-B")
    tracker.add("correlation_main",    p_value=0.008, section="IV-B")
    tracker.add("cross_model_qwen15b", p_value=0.032, section="IV-C")
    tracker.add("arc_transfer",        p_value=0.41,  section="IV-D")

    print("\n--- FDR Correction ---")
    print(tracker.paper_text())

    assert d.d < 0, "Correct should have lower entropy → negative d"
    assert d.interpretation in ("small", "medium", "large")
    print("\n✅ effect_size.py self-test passed.")
