# Data Construction & Audit Documentation: GSM8K Synthetic vs. Organic Traces

## 1. Overview & Dataset Audit Purpose

This document provides a transparent, auditable breakdown of how `data/v2/gsm8k_500_proper.json` was constructed, the structural characteristics of its correct and incorrect reasoning traces, and the exact mechanism behind heuristic baseline performance.

---

## 2. Construction Methodology of `gsm8k_500_proper.json`

### 2.1 Target Balance & Sourcing

To establish a balanced 500-sample benchmark dataset ($N=500$, 250 correct, 250 incorrect) for 5-fold cross-validation without sample imbalance bias:
- **Correct Traces ($N=250$)**: Sourced from structural templates modeling canonical reasoning flows:
  - `PR → CA → IN → CO` (Problem Restatement $\rightarrow$ Calculation $\rightarrow$ Inference $\rightarrow$ Conclusion)
  - `PR → EQ → CA → IN → CO` (Problem Restatement $\rightarrow$ Equation $\rightarrow$ Calculation $\rightarrow$ Inference $\rightarrow$ Conclusion)
  - `PR → IN → CA → CO`
  - `PR → CA → RF → CA → CO`
- **Incorrect Traces ($N=250$)**: Sourced from structural templates modeling common LLM failure modes:
  - Simple Diversion (`PR → DV → CO`)
  - Calculation Loops (`PR → CA → CA → CA → CO`)
  - Inference Loops (`PR → IN → IN → IN → CO`)
  - Reflection Loops (`PR → RF → RF → RF → CO`)

### 2.2 Text Template Mapping & Noise Injection

Each step tag in the sequence was mapped to recognizable text snippets (e.g., `PR` $\rightarrow$ "First, I need to understand the problem...", `CA` $\rightarrow$ "Calculate the result: 5 + 3 = 8...", `CO` $\rightarrow$ "The final answer is..."). Random step insertion and deletion noise ($\text{prob} = 0.25$) were applied to create sequence variation.

---

## 3. Audit Findings: Baseline Inflation & Structural Mechanics

### 3.1 Explanation of Heuristic Baseline AUC (0.843 - 0.860)

The manual eyeball audit (`scripts/audit_dataset.py`) revealed why basic heuristic baselines scored exceptionally high:

- **Repetition Penalty Baseline (AUC = 0.843)**:
  - **Correct Traces Mean Repetition**: **0.0486 ± 0.0779** (low step repetition).
  - **Incorrect Traces Mean Repetition**: **0.2646 ± 0.1691** (high step repetition due to `CA-CA-CA`, `IN-IN-IN`, `RF-RF-RF` templates).
  - **Mechanism**: The explicit inclusion of repetition loop templates in the synthetic incorrect set allowed the simple Repetition Penalty heuristic to detect incorrectness with high accuracy (AUC = 0.843).

- **Structural Entropy Metric (r = -0.732, AUC = 0.057 / 0.943 flipped)**:
  - **Correct Traces Mean Entropy**: **2.2589 ± 0.2729** (higher tag diversity across 5–6 distinct tags).
  - **Incorrect Traces Mean Entropy**: **1.6557 ± 0.2886** (lower entropy due to tag repetition or short 3-step sequences).

---

## 4. Organic Model-Generated Datasets (`data/v2/qwen-0.5b_gsm8k.json`, etc.)

For real organic LLM reasoning evaluations without synthetic template bias, the repository includes raw model generations:

- `qwen-0.5b_gsm8k.json` (100 organic traces from Qwen-0.5B)
- `llama-3.2-1b_gsm8k.json` (100 organic traces from Llama-3.2-1B)
- `mistral-7b_gsm8k.json` (100 organic traces from Mistral-7B)
- `phi-3-mini_gsm8k.json` (100 organic traces from Phi-3-mini)

---

## 5. Summary & Audit Guarantee

The 5-fold cross-validation pipeline (`scripts/run_leakfree_evaluation.py`) is fully reproducible, deterministic, and free of data leakage across folds. All scoring scripts, datasets, and diagnostic tools are committed to version control.
