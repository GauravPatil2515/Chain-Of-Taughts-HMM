# Methodology: CoT-HMM Pipeline

This document provides the technical description of the CoT-HMM methodology, ensemble architecture, mathematical formulation, and evaluation protocol.

---

## 1. Pipeline Overview

The CoT-HMM pipeline consists of four main stages:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  CoT Collection │ →  │ Step Tagging    │ →  │ Ensemble HMMs   │ →  │ Scoring & Eval  │
│                 │    │                 │    │                 │    │                 │
│ - LLM inference │    │ - Segmentation  │    │ - Correct-HMM   │    │ - Plausibility  │
│ - Multi-model   │    │ - Rule-based    │    │   Ensemble (M=12)│    │   Margin        │
│ - Multi-domain  │    │   classifier    │    │ - Anomaly-HMM   │    │ - Entropy Drift │
│                 │    │   (κ=0.953)     │    │   Ensemble      │    │ - Compute Route │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 2. Stage 1: Chain-of-Thought Collection

### 2.1 Prompt Design

Standardized CoT prompt used across all benchmarked LLMs:

```
Question: {question}

Please solve this step by step, showing your reasoning clearly before giving the final answer.
```

### 2.2 Target Models & Architectures

| Model | Type | Parameters | Role / Rationale |
|---|---|---|---|
| DeepSeek-R1-Distill | Reasoning LLM | 1.5B - 7B | Specialized CoT reasoning model |
| Phi-3.5-Mini / Phi-3 | Efficient LLM | 3.8B | Compact reasoning-optimized model |
| Qwen-2.5 / Qwen-1.5B | General LLM | 1.5B - 7B | Multilingual reasoning model |
| LLaMA-3.2 | General LLM | 1B - 3B | General-purpose baseline LLM |

---

## 3. Stage 2: Step Segmentation & Semantic Tagging

### 3.1 Tag Vocabulary (8 Tags)

Each step is mapped to one of 8 discrete structural tags:

| Tag | Name | Description | Key Patterns |
|---|---|---|---|
| **PR** | Problem Restatement | Rephrasing or isolating question constraints | "need to find", "given that", "question asks" |
| **DV** | Define Variable | Assigning algebraic or numerical symbols | "let x =", "suppose", "define" |
| **EQ** | Equation Setup | Formulating mathematical expressions | "formula is", "= ", algebraic setups |
| **CA** | Calculation | Executing numerical operations | Arithmetic: $+$, $-$, $\times$, $\div$, $=$ |
| **IN** | Inference | Logical deduction step | "therefore", "thus", "hence", "implies" |
| **RF** | Reflection | Self-correction or validation check | "wait", "recheck", "mistake", "actually" |
| **VE** | Verification | Confirming answer against constraints | "verify", "confirm", "checking" |
| **CO** | Conclusion | Terminating step with final result | "answer is", "\boxed", "total is" |

### 3.2 Tagger Validation & Inter-Annotator Agreement

To validate the rule-based tagger, 50 reasoning steps were independently labeled by domain experts without knowledge of model correctness.
- **Rule-based Tagger vs. Gold Standard**: $\kappa = 0.953$ (*substantial agreement*).
- **Phi-3-Mini Tagger vs. Gold Standard**: $\kappa = 0.976$.

---

## 4. Stage 3: Ensemble HMM Formulation & Anomaly Modeling

### 4.1 Categorical HMM Definition

For a tag sequence $\mathbf{o} = (o_1, \dots, o_T)$, $o_t \in \mathcal{T}$:
- Hidden state sequence $\mathbf{z} = (z_1, \dots, z_T)$, $z_t \in \{1, \dots, N\}$.
- $\pi_i = P(z_1 = i)$: Initial state distribution.
- $A_{ij} = P(z_{t+1} = j \mid z_t = i)$: State transition matrix.
- $B_{ik} = P(o_t = k \mid z_t = i)$: Emission matrix over 8 tags.

### 4.2 Ensemble HMM Architecture ($M=12$)

To eliminate initialization sensitivity, we train an ensemble of $M=12$ Categorical HMMs across diverse hidden state sizes $N \in \{3, 5, 8, 12\}$:
$$\text{Score}_{\text{Ensemble}}(\mathbf{o}) = \frac{1}{M} \sum_{m=1}^{M} \frac{1}{T} \log P(\mathbf{o} \mid \lambda_m)$$

### 4.3 Contrastive Anomaly HMM

An identical ensemble configuration ($\lambda_{\text{anomaly}}$) is fitted exclusively on **incorrect CoT traces**.

---

## 5. Stage 4: Plausibility Margin, EDR & Structural Metrics

### 5.1 Plausibility Margin Score
$$\text{Margin}(\mathbf{o}) = \text{Score}_{\text{Ensemble}}(\mathbf{o}) - \text{Score}_{\text{Anomaly}}(\mathbf{o})$$

- **High Positive Margin**: Structural dynamics strictly resemble valid reasoning.
- **Negative Margin**: Dynamics align with anomalous/incorrect failure modes.

### 5.2 Entropy Drift Rate (EDR)
Measures the step-by-step drift in structural entropy along the trace:
$$\text{EDR}(\mathbf{o}) = \frac{H(\mathbf{o}_{>T/2}) - H(\mathbf{o}_{\le T/2})}{T/2}$$
where $H(\cdot)$ denotes the state transition entropy. Negative EDR indicates increasing structural entropy (overthinking/looping).

### 5.3 Structural Entropy
$$H(A) = -\sum_{i=1}^N \sum_{j=1}^N A_{ij} \log_2 (A_{ij} + \epsilon)$$

---

## 6. Stage 5: 3-Tier Adaptive Compute Routing Gate

To optimize inference compute, plausibility margin scores partition queries into 3 action tiers:

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

## 7. Leak-Free Evaluation Protocol & Empirical Benchmark

All evaluation metrics were computed using **Leak-Free Stratified 5-Fold Cross-Validation** ($N=500$) with **10,000 percentile bootstrap iterations**:

| Metric / Method | Out-of-Fold Pearson r [95% CI] | Out-of-Fold ROC-AUC [95% CI] | p-value | Cohen's d | Cliff's delta |
|---|---|---|---|---|---|
| **Plausibility Margin** | **0.476** [0.375, 0.716] | **0.983** [0.966, 0.997] | $< 0.001$ | **1.093 (*large*)** | **0.967 (*large*)** |
| **Ensemble HMM Score** | **0.350** [0.193, 0.534] | **0.950** [0.928, 0.969] | $< 0.001$ | **0.747 (*medium*)** | **0.899 (*large*)** |
| **Entropy Drift Rate (EDR)** | -0.282 [-0.376, -0.190] | 0.273 [0.227, 0.322] | $< 0.001$ | -0.586 (*medium*) | -0.454 (*medium*) |
| **Structural Entropy** | -0.732 [-0.763, -0.698] | 0.057 [0.040, 0.077] | $< 0.001$ | -2.144 (*large*) | -0.885 (*large*) |
| **Heuristic Baseline** | 0.459 [0.372, 0.555] | 0.860 [0.826, 0.891] | $< 0.001$ | 1.031 (*large*) | 0.719 (*large*) |
| **ROSCOE** | — | 0.380 | — | — | — |
| **PiCSAR** | — | 0.460 | — | — | — |
