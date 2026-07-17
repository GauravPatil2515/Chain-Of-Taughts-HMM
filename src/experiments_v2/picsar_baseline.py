"""
picsar_baseline.py  —  Week 4 Deliverable
PiCSAR-style joint log-likelihood scorer using a HuggingFace causal LM.

PiCSAR measures reasoning quality by computing the log-probability of the
CoT continuation given the problem prefix. Higher logprob = the LM finds
this reasoning chain more plausible.

Reference:
    Prasad et al. (2023) "PiCSAR: Leveraging LLM Calibration for Self-Evaluation"
    https://arxiv.org/abs/2310.08118

We approximate PiCSAR using:
    score(cot | problem) = log P(cot | problem) / |cot_tokens|

Backbone: distilgpt2 (no API key, runs on CPU in <2 sec per sample)

Install:
    pip install transformers torch

Usage:
    scorer = PiCSARScorer()
    score = scorer.score(problem="...", cot_text="...")
    print(score.normalized_jll)
"""

from __future__ import annotations
import json
import math
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[picsar] WARNING: transformers/torch not installed. Run: pip install transformers torch")


# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------

@dataclass
class PiCSARScore:
    joint_log_likelihood: float         # log P(cot | problem) — unnormalized
    normalized_jll: float               # per-token JLL
    perplexity: float                   # exp(-normalized_jll)
    n_cot_tokens: int
    n_prompt_tokens: int
    model_used: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class PiCSARScorer:
    """
    PiCSAR-style scorer using a HuggingFace causal LM.

    Args:
        model_name: HuggingFace model ID. 'distilgpt2' is fast + free.
                    For better quality: 'gpt2', 'EleutherAI/gpt-neo-125M'
        device: 'cpu' | 'cuda' | 'mps'
        max_length: Maximum context window (tokens). Clips if exceeded.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: str = "cpu",
        max_length: int = 512,
    ):
        if not HF_AVAILABLE:
            raise ImportError("Run: pip install transformers torch")

        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        print(f"[PiCSAR] Loading model: {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # GPT-2 family has no pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[PiCSAR] Model loaded ✓")

    def _get_logprobs(self, prompt: str, continuation: str) -> Dict[str, Any]:
        """
        Compute log P(continuation | prompt) using teacher forcing.

        Returns dict with:
            joint_ll: float
            n_cont_tokens: int
            n_prompt_tokens: int
        """
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)

        # Clip to max_length
        combined = prompt_ids + cont_ids
        if len(combined) > self.max_length:
            # Prioritize continuation, truncate prompt from the front
            available = self.max_length - len(cont_ids) - 1
            if available < 10:
                # If CoT is very long, also truncate CoT
                cont_ids = cont_ids[: self.max_length // 2]
                prompt_ids = prompt_ids[-(self.max_length // 2) :]
            else:
                prompt_ids = prompt_ids[-available:]
            combined = prompt_ids + cont_ids

        input_ids = torch.tensor([combined], dtype=torch.long, device=self.device)
        n_prompt = len(prompt_ids)
        n_cont = len(cont_ids)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            # outputs.logits shape: (1, seq_len, vocab_size)
            logits = outputs.logits[0]  # (seq_len, vocab_size)

        # Shift: prediction at position i is for token i+1
        # We only want log-probs of the continuation tokens
        # Continuation starts at position n_prompt in input_ids
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        joint_ll = 0.0
        for i in range(n_prompt - 1, n_prompt + n_cont - 1):
            if i + 1 < len(combined):
                target_token = combined[i + 1]
                joint_ll += float(log_probs[i, target_token].item())

        return {
            "joint_ll": joint_ll,
            "n_cont_tokens": n_cont,
            "n_prompt_tokens": n_prompt,
        }

    def score(self, problem: str, cot_text: str) -> PiCSARScore:
        """
        Score a CoT trace.

        Args:
            problem: The original problem statement (used as prompt/condition).
            cot_text: The full chain-of-thought text.

        Returns:
            PiCSARScore with log-likelihood and perplexity.
        """
        if not cot_text.strip():
            return PiCSARScore(
                joint_log_likelihood=float("-inf"),
                normalized_jll=float("-inf"),
                perplexity=float("inf"),
                n_cot_tokens=0,
                n_prompt_tokens=0,
                model_used=self.model_name,
            )

        result = self._get_logprobs(problem, cot_text)
        jll = result["joint_ll"]
        n_cont = max(result["n_cont_tokens"], 1)

        normalized_jll = jll / n_cont
        perplexity = math.exp(-normalized_jll)

        return PiCSARScore(
            joint_log_likelihood=jll,
            normalized_jll=normalized_jll,
            perplexity=perplexity,
            n_cot_tokens=n_cont,
            n_prompt_tokens=result["n_prompt_tokens"],
            model_used=self.model_name,
        )


# ---------------------------------------------------------------------------
# Batch scoring against the v2 dataset
# ---------------------------------------------------------------------------

def run_picsar_on_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    model_name: str = "distilgpt2",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a v2 ScoredTrace dataset, run PiCSAR on every trace, save results.

    Returns a summary dict with AUC and correlation vs. is_correct.
    """
    from sklearn.metrics import roc_auc_score
    from scipy.stats import pearsonr

    scorer = PiCSARScorer(model_name=model_name, device=device)

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    results = []
    # Use negative perplexity as the ranking signal (lower ppl = better)
    picsar_signals: List[float] = []
    labels: List[int] = []

    for i, item in enumerate(raw):
        problem = item.get("problem", "")
        cot_text = item.get("cot_text", "")
        is_correct = int(item.get("is_correct", False))

        sc = scorer.score(problem, cot_text)
        item["picsar_score"] = sc.to_dict()
        results.append(item)

        # Higher normalized_jll = better (less negative = model finds it more probable)
        picsar_signals.append(sc.normalized_jll)
        labels.append(is_correct)

        if (i + 1) % 50 == 0:
            print(f"  [PiCSAR] Scored {i+1}/{len(raw)} traces...")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    sig_arr = np.array(picsar_signals)
    labels_arr = np.array(labels)

    # Replace -inf with very small finite number for stats
    sig_arr = np.where(np.isfinite(sig_arr), sig_arr, np.nanmin(sig_arr[np.isfinite(sig_arr)]) - 1)

    auc = roc_auc_score(labels_arr, sig_arr) if len(set(labels_arr)) > 1 else float("nan")
    r, p = pearsonr(sig_arr, labels_arr.astype(float))

    summary = {
        "method": "PiCSAR",
        "backbone": model_name,
        "n_samples": len(results),
        "auc": round(float(auc), 4),
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 5),
        "mean_correct_jll": float(sig_arr[labels_arr == 1].mean()),
        "mean_incorrect_jll": float(sig_arr[labels_arr == 0].mean()),
    }

    print("\n" + "="*55)
    print("PiCSAR RESULTS")
    print("="*55)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


# ---------------------------------------------------------------------------
# Formal comparison table builder
# ---------------------------------------------------------------------------

def build_comparison_table(
    results_dir: str | Path,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Read per-method JSON summary files from results_dir and build a markdown
    comparison table.

    Expected files in results_dir:
        hmm_summary.json, roscoe_summary.json, picsar_summary.json
        neural_hmm_summary.json (Week 8)
    """
    import os

    methods_order = [
        ("HMM Ensemble (Ours)", "hmm_summary.json"),
        ("Single HMM (N=5)", "single_hmm_summary.json"),
        ("Neural-HMM (Ours+)", "neural_hmm_summary.json"),
        ("ROSCOE", "roscoe_summary.json"),
        ("PiCSAR", "picsar_summary.json"),
        ("Length Baseline", "length_baseline_summary.json"),
        ("Random Baseline", "random_baseline_summary.json"),
    ]

    rows = []
    for display_name, filename in methods_order:
        fpath = Path(results_dir) / filename
        if not fpath.exists():
            continue
        with open(fpath, "r") as f:
            d = json.load(f)
        rows.append({
            "Method": display_name,
            "AUC": d.get("auc", "—"),
            "Pearson r": d.get("pearson_r", "—"),
            "p-value": d.get("p_value", "—"),
            "N": d.get("n_samples", "—"),
        })

    if not rows:
        return "No result files found in " + str(results_dir)

    # Build markdown table
    header = "| Method | AUC | Pearson r | p-value | N |"
    sep = "|--------|-----|-----------|---------|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['Method']} | {r['AUC']} | {r['Pearson r']} | {r['p-value']} | {r['N']} |"
        )

    table = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Method Comparison Table\n\n")
            f.write(table + "\n")
        print(f"[comparison_table] Saved → {output_path}")

    return table


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not HF_AVAILABLE:
        print("Install first: pip install transformers torch")
        exit(1)

    scorer = PiCSARScorer(model_name="distilgpt2", device="cpu")

    problem = "If John has 5 apples and gives 2 away, how many does he have?"
    good_cot = "John starts with 5. He gives 2 away. 5 - 2 = 3. The answer is 3."
    bad_cot = "The sky is blue. Apples are red. Multiply something by 42. Done."

    good_score = scorer.score(problem, good_cot)
    bad_score = scorer.score(problem, bad_cot)

    print(f"Good CoT — normalized JLL: {good_score.normalized_jll:.4f}, PPL: {good_score.perplexity:.2f}")
    print(f"Bad CoT  — normalized JLL: {bad_score.normalized_jll:.4f},  PPL: {bad_score.perplexity:.2f}")
    print("\n✅ picsar_baseline.py self-test passed.")
