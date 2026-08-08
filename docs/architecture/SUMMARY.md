# Architecture & Evaluation Summary: CoT-HMM

## What is CoT-HMM?

**Unsupervised Plausibility Scoring of Chain-of-Thought Reasoning via Structural Hidden Markov Models**

A lightweight, text-only framework that models the structural transition dynamics of valid reasoning chains, evaluating new CoT traces without human step labels, GPU retraining, or domain-specific fine-tuning.

---

## Core Pipeline Architecture

1. **CoT Collection** → `scripts/run_real_generation.py` / `colab_scripts/colab_experiments.py`
2. **Step Tagging** → `src/tagging/rule_based_tagger.py` → Abstract reasoning steps into 8 semantic tags ($\kappa = 0.953$).
3. **Ensemble & Anomaly Training** → `src/hmm/ensemble.py` → Train Ensemble HMM ($M=12$, $N \in \{3,5,8,12\}$) on correct traces; train Anomaly HMM on incorrect traces.
4. **Out-of-Fold Scoring** → `src/hmm/scorer.py` → Compute log-likelihood, Plausibility Margin, Structural Entropy, and Entropy Drift Rate (EDR).
5. **Compute Routing** → `scripts/run_leakfree_evaluation.py` → 3-Tier adaptive gate (Pass / Escalate / Reject).

---

## Key Metrics & Formulations

### Ensemble HMM Score
$$\text{Score}_{\text{Ensemble}}(\mathbf{o}) = \frac{1}{M} \sum_{m=1}^{M} \frac{1}{T} \log P(\mathbf{o} \mid \lambda_m)$$

### Plausibility Margin
$$\text{Margin}(\mathbf{o}) = \text{Score}_{\text{Ensemble}}(\mathbf{o}) - \text{Score}_{\text{Anomaly}}(\mathbf{o})$$

### Entropy Drift Rate (EDR)
$$\text{EDR}(\mathbf{o}) = \frac{H(\mathbf{o}_{>T/2}) - H(\mathbf{o}_{\le T/2})}{T/2}$$

---

## Final Out-of-Fold Benchmark Results (5-Fold CV, N=500, B=10,000)

| Method / Metric | Out-of-Fold Pearson r [95% CI] | Out-of-Fold ROC-AUC [95% CI] | p-value | Effect Size |
|---|---|---|---|---|
| **Plausibility Margin** 🏆 | **0.476** [0.375, 0.716] | **0.983** [0.966, 0.997] | $< 0.001$ | Cohen's $d = 1.093$ (*large*) |
| **Ensemble HMM Score** | **0.350** [0.193, 0.534] | **0.950** [0.928, 0.969] | $< 0.001$ | Cohen's $d = 0.747$ (*medium*) |
| **Entropy Drift Rate (EDR)** | -0.282 [-0.376, -0.190] | 0.273 [0.227, 0.322] | $< 0.001$ | Cohen's $d = -0.586$ (*medium*) |
| **Structural Entropy** | -0.732 [-0.763, -0.698] | 0.057 [0.040, 0.077] | $< 0.001$ | Cohen's $d = -2.144$ (*large*) |
| **Heuristic Baseline** | 0.459 [0.372, 0.555] | 0.860 [0.826, 0.891] | $< 0.001$ | Cohen's $d = 1.031$ (*large*) |
| **ROSCOE** | — | 0.380 | — | — |
| **PiCSAR** | — | 0.460 | — | — |

---

## ⚡ 3-Tier Adaptive Compute Routing Performance

- **Direct Pass Tier (Top 30% Margin)**: **99.3% Accuracy** (Pass 150 queries directly).
- **Escalation Tier (Middle 35% Margin)**: **50.9% Accuracy** (Escalate 175 queries for re-sampling).
- **Early Reject Tier (Bottom 35% Margin)**: **6.9% Accuracy** (Reject 175 flawed queries immediately).
- **Compute Savings**: **30.0% reduction** in LLM inference cost.