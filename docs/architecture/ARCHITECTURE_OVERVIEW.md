# Chain-of-Thought HMM Architecture Documentation

## Executive Summary

CoT-HMM is a lightweight, unsupervised framework for scoring Chain-of-Thought reasoning plausibility using Hidden Markov Models. The system learns structural transition dynamics from valid reasoning chains and detects anomalous reasoning without requiring human step labels, GPU retraining, or domain-specific fine-tuning.

---

## 1. System Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               CoT-HMM ENSEMBLE PIPELINE                                │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌────────────────────────────────────┐
│  CoT Collection │ →  │ Step Tagging    │ →  │ Ensemble HMM Training              │
│                 │    │                 │    │                                    │
│ - LLM inference │    │ - Step          │    │ ┌────────────────────────────────┐ │
│ - Multi-model   │    │   segmentation  │    │ │ Correct-HMM Ensemble (M=12)    │ │
│ - Multi-domain  │    │ - 8-tag         │    │ │ N ∈ {3, 5, 8, 12} hidden states│ │
│                 │    │   rule classifier│    │ └────────────────────────────────┘ │
│                 │    │   (κ = 0.953)   │    │ ┌────────────────────────────────┐ │
└─────────────────┘    └─────────────────┘    │ │ Anomaly-HMM Ensemble (M=12)    │ │
                                              │ └────────────────────────────────┘ │
                                              └──────────────────┬─────────────────┘
                                                                 │
                                                                 ▼
                                              ┌────────────────────────────────────┐
                                              │ Scoring & 3-Tier Compute Routing   │
                                              │                                    │
                                              │ - Plausibility Margin (AUC = 0.983)│
                                              │ - Entropy Drift Rate (EDR)         │
                                              │ - Pass / Escalate / Reject Gate    │
                                              └────────────────────────────────────┘
```

---

## 2. Core ML Model: Ensemble & Anomaly HMMs

### 2.1 Mathematical Formulation

**Hidden States (Latent Reasoning Phases):**
- $N$ hidden states representing latent reasoning states (e.g., Problem Understanding $\rightarrow$ Equation Setup $\rightarrow$ Calculation $\rightarrow$ Verification $\rightarrow$ Conclusion).
- State space size diversity: $N \in \{3, 5, 8, 12\}$.

**Observations (Emissions):**
- Discrete tag sequence $\mathbf{o} = (o_1, \dots, o_T)$ over 8 semantic tags: `PR`, `DV`, `EQ`, `CA`, `IN`, `RF`, `VE`, `CO`.

**Learned Parameters:**
- Initial Distribution $\pi_i = P(z_1 = i)$
- Transition Matrix $A_{ij} = P(z_{t+1} = j \mid z_t = i)$
- Emission Matrix $B_{ik} = P(o_t = k \mid z_t = i)$

### 2.2 Ensemble HMM Formulation ($M=12$)

To avoid initialization bias and local minima in Baum-Welch EM fitting:
$$\text{Score}_{\text{Ensemble}}(\mathbf{o}) = \frac{1}{M} \sum_{m=1}^{M} \frac{1}{T} \log P(\mathbf{o} \mid \lambda_m)$$

### 2.3 Contrastive Plausibility Margin

$$\text{Margin}(\mathbf{o}) = \text{Score}_{\text{Ensemble}}(\mathbf{o}) - \text{Score}_{\text{Anomaly}}(\mathbf{o})$$

---

## 3. Semantic Tagging Architecture

### 3.1 Tag Vocabulary (8 Tags)

| Tag | Name | Key Regex Patterns |
|---|---|---|
| `PR` | Problem Restatement | "need to find", "question asks", "given that" |
| `DV` | Define Variable | "let x =", "define", "suppose" |
| `EQ` | Equation Setup | "= ", "formula", "equation is" |
| `CA` | Calculation | Arithmetic operators: $+$, $-$, $\times$, $\div$, $=$ |
| `IN` | Inference | "therefore", "thus", "hence", "implies" |
| `RF` | Reflection | "wait", "recheck", "mistake", "actually" |
| `VE` | Verification | "check", "verify", "confirm" |
| `CO` | Conclusion | "answer is", "final answer", "\boxed" |

---

## 4. 3-Tier Adaptive Compute Routing Gate Architecture

```
                            Plausibility Margin Score
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
          Margin ≥ +0.5          -0.5 < Margin < +0.5       Margin ≤ -0.5
        ┌───────────────┐      ┌────────────────────┐    ┌───────────────┐
        │ Pass Tier     │      │ Escalation Tier    │    │ Reject Tier   │
        │ (Direct Pass) │      │ (Re-sample / PRM)  │    │ (Early Stop)  │
        └───────────────┘      └────────────────────┘    └───────────────┘
          Accuracy: 99.3%        Accuracy: 50.9%           Accuracy: 6.9%
          Volume: 30%            Volume: 35%               Volume: 35%
```

---

## 5. Out-of-Fold Benchmark Performance (5-Fold CV)

| Metric / Method | Out-of-Fold Pearson r [95% CI] | Out-of-Fold ROC-AUC [95% CI] | p-value | Effect Size |
|---|---|---|---|---|
| **Plausibility Margin** | **0.476** [0.375, 0.716] | **0.983** [0.966, 0.997] | $< 0.001$ | Cohen's $d = 1.093$ (*large*) |
| **Ensemble HMM Score** | **0.350** [0.193, 0.534] | **0.950** [0.928, 0.969] | $< 0.001$ | Cohen's $d = 0.747$ (*medium*) |
| **Entropy Drift Rate (EDR)** | -0.282 [-0.376, -0.190] | 0.273 [0.227, 0.322] | $< 0.001$ | Cohen's $d = -0.586$ (*medium*) |
| **Structural Entropy** | -0.732 [-0.763, -0.698] | 0.057 [0.040, 0.077] | $< 0.001$ | Cohen's $d = -2.144$ (*large*) |
| **Heuristic Baseline** | 0.459 [0.372, 0.555] | 0.860 [0.826, 0.891] | $< 0.001$ | Cohen's $d = 1.031$ (*large*) |
| **ROSCOE** | — | 0.380 | — | — |
| **PiCSAR** | — | 0.460 | — | — |