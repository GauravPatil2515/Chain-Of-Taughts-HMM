# Methodology: CoT-HMM Pipeline

This document provides a detailed technical description of the CoT-HMM methodology.

---

## 1. Overview

The CoT-HMM pipeline consists of four main stages:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  CoT Collection │ →  │ Step Tagging    │ →  │ HMM Training    │ →  │ Scoring & Eval  │
│                 │    │                 │    │                 │    │                 │
│ - LLM inference │    │ - Segmentation  │    │ - Fit on good   │    │ - Log-likelihood│
│ - Multi-model   │    │ - Rule-based    │    │   CoT traces    │    │ - Entropy       │
│ - Multi-domain  │    │   classification│    │ - Learn A, B, π │    │ - Anomalies     │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 2. Stage 1: Chain-of-Thought Collection

### 2.1 Prompt Design

We use a standardized CoT prompt across all models:

```
Question: {question}

Please solve this step by step, showing your reasoning clearly before giving the final answer.
```

### 2.2 Target Models

| Model | Type | Parameters | Rationale |
|-------|------|------------|-----------|
| DeepSeek-R1-Distill | Reasoning LLM | 1.5B-7B | Explicitly trained for reasoning |
| Phi-3.5-Mini | Efficient LLM | 3.8B | Microsoft's reasoning-optimized model |
| Qwen2.5 | General LLM | 1.5B-7B | Strong multilingual reasoning |
| LLaMA-3.2 | General LLM | 1B-3B | Baseline general-purpose model |

### 2.3 Datasets

| Dataset | Domain | Task Type | Samples |
|---------|--------|-----------|---------|
| GSM8K | Mathematics | Word problems | 200 |
| StrategyQA | Commonsense | Yes/No reasoning | 100 |
| ARC-Challenge | Science/Logic | Multiple choice | 100 |

### 2.4 Data Collection Schema

```json
{
  "id": "gsm8k_001",
  "question": "A store sells pens at $3 each...",
  "ground_truth": "42",
  "model": "deepseek-r1-distill",
  "cot_raw": "First, I need to understand the problem...",
  "final_answer": "42",
  "is_correct": true,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

---

## 3. Stage 2: Step Segmentation & Tagging

### 3.1 Step Segmentation

Each CoT trace is split into discrete reasoning steps using:

1. **Sentence Boundaries**: Split on `.`, `!`, `?`
2. **Numbered Steps**: Detect patterns like `1.`, `Step 1:`, `First,`
3. **Newline Delimiters**: Split on `\n` when present

**Example:**

```
Input CoT:
"First, I need to find the cost of pens. Each pen costs $3 and he buys 4.
So 4 × 3 = 12 dollars. Adding the notebook at $5, total is $17."

Segmented Output:
[
  "First, I need to find the cost of pens",
  "Each pen costs $3 and he buys 4",
  "So 4 × 3 = 12 dollars",
  "Adding the notebook at $5, total is $17"
]
```

### 3.2 Semantic Role Tagging

Each step is assigned a tag based on rule-based pattern matching:

#### Tag Definitions

| Tag | Description | Example Patterns |
|-----|-------------|------------------|
| `PR` | Problem Restatement | "need to find", "the question asks", "we are given" |
| `DV` | Define Variable | "let x =", "define", "suppose" |
| `EQ` | Equation Setup | "= ", "equation", "formula" |
| `CA` | Calculation | Numbers with operators: `×`, `+`, `-`, `÷`, `=` |
| `IN` | Inference | "therefore", "thus", "so", "hence", "implies" |
| `RF` | Recall Fact | "we know that", "recall", "the formula for" |
| `VE` | Verification | "check", "verify", "let's confirm" |
| `CO` | Conclusion | "answer is", "final answer", "result" |

#### Tagging Algorithm

```python
def tag_step(step: str) -> str:
    step_lower = step.lower()
    
    # Priority-based matching
    if any(p in step_lower for p in ["answer is", "final answer", "result"]):
        return "CO"
    if any(p in step_lower for p in ["verify", "check", "confirm"]):
        return "VE"
    if any(p in step_lower for p in ["therefore", "thus", "hence", "so"]):
        return "IN"
    if any(p in step_lower for p in ["let", "define", "suppose"]):
        return "DV"
    if any(p in step_lower for p in ["need to find", "question asks", "given"]):
        return "PR"
    if any(p in step_lower for p in ["we know", "recall", "formula for"]):
        return "RF"
    if re.search(r'\d+\s*[×+\-÷*/=]\s*\d+', step):
        return "CA"
    if "=" in step and any(c.isdigit() for c in step):
        return "EQ"
    
    return "IN"  # Default to inference
```

### 3.3 Tagged Output Format

```json
{
  "id": "gsm8k_001_deepseek",
  "steps": [
    {"text": "First, I need to find the cost of pens", "tag": "PR"},
    {"text": "Each pen costs $3 and he buys 4", "tag": "RF"},
    {"text": "So 4 × 3 = 12 dollars", "tag": "CA"},
    {"text": "Adding the notebook at $5, total is $17", "tag": "CO"}
  ],
  "tag_sequence": ["PR", "RF", "CA", "CO"],
  "is_correct": true
}
```

---

## 4. Stage 3: HMM Training

### 4.1 HMM Formulation

We model the tagged CoT as a discrete-time Hidden Markov Model:

- **Hidden States** $Z = \{z_1, z_2, ..., z_K\}$: Latent reasoning phases (discovered automatically)
- **Observations** $X = \{x_1, x_2, ..., x_T\}$: The tag sequence (PR, EQ, CA, etc.)
- **Parameters**:
  - $\pi$: Initial state distribution
  - $A$: Transition matrix $P(z_t | z_{t-1})$
  - $B$: Emission matrix $P(x_t | z_t)$

### 4.2 Training Data Selection

**Critical Design Choice**: We train the HMM **only on correct CoT traces**.

Rationale:

- The HMM learns the "canonical" structure of successful reasoning
- Incorrect traces will have lower likelihood under this model
- This enables anomaly detection

### 4.3 Training Procedure

```python
from hmmlearn import hmm

# Prepare sequences
sequences = []  # List of tag sequences (as integers)
lengths = []    # Length of each sequence

for cot in correct_cots:
    seq = [tag_to_id[tag] for tag in cot["tag_sequence"]]
    sequences.append(seq)
    lengths.append(len(seq))

# Flatten for hmmlearn format
X = np.concatenate(sequences).reshape(-1, 1)

# Train HMM
model = hmm.MultinomialHMM(
    n_components=6,      # Number of hidden states
    n_iter=100,          # Max iterations
    random_state=42
)
model.fit(X, lengths)

# Save model
joblib.dump(model, "models/hmm_model.pkl")
```

### 4.4 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_components` | 6 | Matches approximate number of reasoning phases |
| `n_iter` | 100 | Sufficient for convergence |
| `covariance_type` | N/A (Multinomial) | Discrete observations |

---

## 5. Stage 4: Scoring & Evaluation

### 5.1 Scoring Metrics

#### 5.1.1 Log-Likelihood Score

The primary metric: probability of the observed tag sequence under the trained HMM.

```python
def compute_log_likelihood(model, tag_sequence):
    seq = np.array([tag_to_id[t] for t in tag_sequence]).reshape(-1, 1)
    return model.score(seq)
```

**Interpretation:**

- **High score** (e.g., -5 to -10): Typical, coherent reasoning
- **Low score** (e.g., < -20): Unusual, potentially flawed reasoning

#### 5.1.2 Transition Entropy

Measures the uncertainty/randomness of state transitions:

$$H(A) = -\sum_{i,j} A_{ij} \log A_{ij}$$

```python
def compute_transition_entropy(model):
    A = model.transmat_
    entropy = -np.sum(A * np.log(A + 1e-10))
    return entropy
```

**Interpretation:**

- **Low entropy**: Deterministic, predictable transitions
- **High entropy**: Chaotic, erratic reasoning

#### 5.1.3 Anomaly Detection

Flags improbable state jumps:

```python
def detect_anomalies(model, tag_sequence, threshold=0.01):
    anomalies = []
    A = model.transmat_
    
    for i in range(len(tag_sequence) - 1):
        from_state = tag_to_id[tag_sequence[i]]
        to_state = tag_to_id[tag_sequence[i + 1]]
        
        if A[from_state, to_state] < threshold:
            anomalies.append({
                "position": i,
                "transition": f"{tag_sequence[i]} → {tag_sequence[i+1]}",
                "probability": A[from_state, to_state]
            })
    
    return anomalies
```

### 5.2 Evaluation Protocol

1. **Correlation with Correctness**
   - Compute Pearson/Spearman correlation between HMM score and answer correctness
   - Hypothesis: Higher scores correlate with correct answers

2. **Cross-Model Comparison**
   - Compare score distributions across models
   - Hypothesis: Reasoning-trained models (DeepSeek) have higher average scores

3. **Ablation Studies**
   - Vary number of hidden states (4, 6, 8)
   - Vary tag granularity (3-tag vs 8-tag schemes)

4. **Case Studies**
   - Manually analyze high-score correct CoTs
   - Manually analyze low-score incorrect CoTs
   - Identify patterns

---

## 6. Baselines for Comparison

| Baseline | Description |
|----------|-------------|
| Random | Random plausibility score |
| Step Count | Number of reasoning steps |
| Token Entropy | Entropy of token distribution |
| Bigram Model | Simple bigram over tags |

---

## 7. Expected Outputs

### 7.1 Quantitative Results

| Model | Accuracy | Avg HMM Score | Avg Entropy | Anomaly Rate |
|-------|----------|---------------|-------------|--------------|
| DeepSeek-R1 | 82% | -12.3 | 0.88 | 5% |
| Phi-3.5 | 75% | -14.1 | 1.01 | 12% |
| Qwen2.5 | 70% | -15.5 | 1.15 | 18% |
| LLaMA-3.2 | 65% | -18.2 | 1.42 | 25% |

### 7.2 Visualizations

1. **Transition Matrix Heatmap**: Shows learned reasoning flow
2. **Score Distribution**: Correct vs. incorrect CoTs
3. **Per-Model Comparison**: Box plots of scores across models
4. **Viterbi Path Examples**: Most likely hidden state sequences

---

## 8. Limitations

1. **Tagging Noise**: Rule-based tagging may misclassify ambiguous steps
2. **Markov Assumption**: First-order HMM ignores long-range dependencies
3. **Domain Specificity**: Model trained on math may not generalize to other domains
4. **Semantic Blindness**: HMM checks structure, not factual correctness

---

## 9. References

- Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in LLMs. NeurIPS.
- EIRL (ICLR 2026 submission). Hidden Markov Reasoning Lens.
- Wu et al. (EMNLP 2025). State-Aware Reasoning Transitions.
- Rabiner (1989). A Tutorial on Hidden Markov Models. IEEE.
