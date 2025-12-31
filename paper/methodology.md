# Methodology

We propose a lightweight, unsupervised method for evaluating the plausibility of Chain-of-Thought (CoT) reasoning without requiring access to ground truth answers or large reward models. Our approach models the *structure* of reasoning as a sequence of semantic states using a Hidden Markov Model (HMM).

## 1. Pipeline Overview

Our pipeline consists of three stages:

1. **Tagging:** Converting raw text CoT steps into a sequence of abstract semantic tags (e.g., "Calculation", "Conclusion").
2. **Structuring:** Training a Categorical HMM on the tag sequences of known-correct solutions to learn the latent structure of valid reasoning.
3. **Scoring:** Calculating the log-likelihood of new, unseen CoT traces under the trained HMM to quantify their "structural plausibility."

## 2. Semantic Tagging

To abstract away from specific domain content (e.g., numbers, specific entities) and focus on reasoning structure, we employ a rule-based tagger that maps each reasoning step to one of 8 semantic categories.

The tag set $T$ consists of:

- **PR (Problem Restatement):** Initial setup or restating given information.
- **IN (Inference):** Logical deduction or reasoning step (default tag).
- **CA (Calculation):** Explicit arithmetic or computation.
- **EQ (Equation):** Setting up mathematical formulas or LaTeX expressions.
- **CO (Conclusion):** Final answer derivation ("Therefore...", "\boxed{}").
- **RF (Reflection):** Self-correction or verification steps.
- **VE (Verification):** Explicit checking of the answer.
- **DV (Diversion):** Irrelevant or structurally anomalous text.

We use a prioritized regex-based extraction method. Steps are first segmented by newline or "Step X:" delimiters, then classified based on keyword matching (e.g., "calculate", "implies", "=").

## 3. Hidden Markov Model (HMM)

We model the sequence of tags $O = (o_1, o_2, ..., o_T)$ where $o_t \in T$ using a **Categorical Hidden Markov Model**.

The HMM is defined by:

- $N$ hidden states $S = \{s_1, ..., s_N\}$ representing latent stages of reasoning (e.g., "Planning", "Execution").
- Transition matrix $A = \{a_{ij}\}$, where $a_{ij} = P(s_{t+1}=j | s_t=i)$.
- Emission matrix $B = \{b_{ik}\}$, where $b_{ik} = P(o_t=k | s_t=i)$.
- Initial state distribution $\pi = \{\pi_i\}$.

We chose $N=5$ hidden states based on empirical validation, which allows the model to capture stages typically found in GSM8K solutions (Setup -> Plan -> Calc -> Deduce -> Answer).

**Training:** The model parameters $\lambda = (A, B, \pi)$ are estimated using the Baum-Welch algorithm (EM) on a dataset of *correct* CoT sequences. This ensures the model learns the transition dynamics of valid reasoning patterns.

## 4. Plausibility Scoring

For a given inference CoT sequence $O_{test}$, we compute the **Length-Normalized Log-Likelihood (LNLL)**:

$$ Score(O_{test}) = \frac{1}{T} \log P(O_{test} | \lambda) $$

where $T$ is the number of steps in the sequence. Normalization is crucial because standard log-likelihood decreases linearly with sequence length, unfairly penalizing longer (potentially more detailed) reasoning chains.

A higher score indicates the sequence follows a transition pattern similar to the correct solutions in the training set.
