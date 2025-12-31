# Experiments

We assess the effectiveness of our structural HMM scorer on three axes:

1. **Scale:** How does performance change with more training data?
2. **Model Validity:** Does the method work across different LLMs (Qwen-0.5B, Qwen-1.5B)?
3. **Task Generalization:** Does it transfer to non-math reasoning (ARC-Challenge)?

## 1. Experimental Setup

### Models

We generate Chain-of-Thought solutions using the **Qwen-2.5-Instruct** family:

- **Qwen-0.5B:** A lightweight model prone to hallucinations, serving as a challenging testbed for detecting plausible-but-incorrect reasoning.
- **Qwen-1.5B:** A stronger reasoning model to verify if the method scales to higher-quality traces.

### Datasets

- **GSM8K:** High-quality grade school math problems (500 samples). This is our primary domain for analyzing reasoning structure.
- **ARC-Challenge:** Science question answering requiring multi-hop reasoning (200 samples). Used to test cross-domain applicability.

### Baselines

We compare our HMM-based `Structure Score` against four unsupervised baselines:

1. **Random:** Assigns a random score $[0, 1]$.
2. **Length:** Number of steps in the chain (assuming longer $\approx$ better).
3. **Diversity:** Ratio of unique tags to total steps (capturing complexity).
4. **Transition Probability:** A simpler Bigram model $P(t_{i+1} | t_i)$ without hidden states.

## 2. Evaluation Metrics

We treat the problem as a binary classification task (Correct vs. Incorrect) based on the structural score.

- **AUC-ROC:** Area Under the Receiver Operating Characteristic curve.
- **Point-Biserial Correlation:** Correlation between the continuous score and binary correctness label.
- **p-value:** Statistical significance of the correlation (null hypothesis: no relationship).

We also report the **Detection Error Trade-off (DET)** where applicable to show the false positive vs. false negative rates at optimal thresholds.
