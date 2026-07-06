# Abstract

Chain-of-Thought (CoT) prompting significantly improves Large Language Model (LLM) reasoning, but verifying the correctness of generated reasoning chains remains distinct from evaluating their final answers. Existing approaches often rely on expensive "Process Reward Models" (PRMs) trained on human feedback or extensive sampling. We propose a lightweight, unsupervised method to estimate the plausibility of a reasoning chain by modeling its *semantic structure*. We convert reasoning traces into sequences of abstract tags (e.g., "Inference", "Calculation") using a rule-based tagger and train a Hidden Markov Model (HMM) on correct solutions. We demonstrate that the structure alone—ignoring specific content—contains significant signal for correctness. On the GSM8K benchmark, our Ensemble HMM-based scorer achieves a statistically significant correlation with ground truth correctness ($r=0.21, p<0.01$) and outperforms structure-blind baselines (AUC 0.62 vs 0.50). This suggests that valid reasoning follows latent structural patterns that can be captured by simple probabilistic models.

# 1. Introduction

Large Language Models (LLMs) have demonstrated impressive capabilities in complex reasoning tasks, largely driven by Chain-of-Thought (CoT) prompting (Wei et al., 2022). However, LLMs are prone to "hallucinations" and logical fallacies, where the reasoning steps may appear coherent surface-level text but fail to logically derive the answer.

Validating these reasoning chains typically requires either:

1. **Ground Truth access:** Comparison against a known gold-standard answer.
2. **Heavy Supervision:** Training "Verifier" or "Reward" models on large datasets of annotated reasoning steps (Lightman et al., 2023).
3. **Self-Consistency:** Sampling dozens of paths to find the consensus (Wang et al., 2022).

These methods are computationally expensive or require data that is unavailable in open-ended deployment scenarios.

In this work, we ask: *Can we distinguish valid reasoning from invalid reasoning based solely on its structural form?*

We hypothesize that correct reasoning follows specific latent structural transitions—for example, a "Problem Setup" is typically followed by a "Plan" or "Equation", whereas incorrect reasoning may exhibit anomalous transitions (e.g., circular logic, skipping calculations).

To test this, we introduce a pipeline that:

1. **Abstracts** CoT text into semantic tags.
2. **Models** valid transition dynamics using a Hidden Markov Model (HMM).
3. **Scores** new samples based on their likelihood under the learned structural prior.

Our experiments on the GSM8K math benchmark using the Qwen model family show that this unsupervised structural score acts as a proxy for correctness, offering a "cheap" reject-option mechanism for LLM agents without requiring access to the final answer key.
