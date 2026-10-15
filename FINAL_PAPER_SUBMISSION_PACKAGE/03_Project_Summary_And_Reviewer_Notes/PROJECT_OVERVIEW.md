# CoT-HMM: Technical & Research Overview

This document provides a concise academic summary of the paper for co-authors, academic mentors, and reviewers.

---

## 1. Core Problem & Motivation
* **Failure Modes in LLM CoT:** Large Language Models (LLMs) frequently produce convincing hallucinations, circular loops, and premature conclusions in complex reasoning chains.
* **Limitations of Existing Verifiers:**
  * Neural Process Reward Models (PRMs) require heavy GPU compute and thousands of step-level annotations.
  * Self-consistency requires 20–40 samples per query (massive token cost).
  * LLM-as-Judge compounds latency and suffers from self-bias.
* **CoT-HMM Hypothesis:** Correct reasoning exhibits characteristic structural transition dynamics across semantic operations that can be captured probabilistically by Hidden Markov Models.

---

## 2. Methodology Architecture
1. **Semantic Abstraction:** CoT reasoning text is parsed into 8 discrete semantic operation tags:
   * `PR`: Problem Restatement
   * `IN`: Logical Inference
   * `CA`: Arithmetic Calculation
   * `EQ`: Equation Setup
   * `CO`: Conclusion / Answer
   * `RF`: Reflection
   * `VE`: Verification
   * `DV`: Diversion
2. **One-Class Multi-Resolution Ensemble HMM ($M=12$):**
   * Trained exclusively on positive (verified correct) reasoning traces.
   * Spans latent state counts $N \in \{3, 5, 8, 12\}$ across multiple random seeds.
   * Evaluates log-likelihood of tag sequences to measure structural plausibility.
3. **Adaptive Compute Routing Gate:**
   * Samples are triaged into:
     - *Pass Tier (Top 35%):* High plausibility, passes directly (99.3% accuracy).
     - *Escalation Tier (Middle 35%):* Undergoes self-consistency / neural re-ranking.
     - *Early Reject Tier (Bottom 30%):* Low structural plausibility, pruned early (93.1% actual error rate).

---

## 3. Key Empirical Results (Validated via 10,000 Bootstrap Resamples)
* **Premature Conclusions:** ROC-AUC = 0.990 [95% CI: 0.966, 1.000]
* **Circular Loops:** ROC-AUC = 0.761 [95% CI: 0.645, 0.866]
* **Omitted Steps:** ROC-AUC = 0.658 [95% CI: 0.531, 0.778]
* **Semantic Control:** Semantic-only corruption yields AUC = 0.501 (proves clean isolation of structural logic from calculation noise).
* **Cross-Architecture Transfer:** Zero-shot transfer from Qwen2.5 to Llama-3.2 ($r=0.946$), Phi-3 ($r=0.941$), and Mistral-7B ($r=0.970$).
* **Efficiency:** ~150 ms per trace on standard CPU; reduces downstream LLM token costs by 42%–47%.
