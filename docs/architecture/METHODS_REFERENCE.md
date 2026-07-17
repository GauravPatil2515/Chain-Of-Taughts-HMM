# Methods Reference - CoT-HMM

## OVERVIEW

This document catalogs all methods, algorithms, and computational approaches used in the CoT-HMM project.

---

## 1. HMM TRAINING METHODS

### 1.1 Categorical HMM Training
- **File:** `src/hmm/trainer.py`
- **Algorithm:** Baum-Welch Expectation-Maximization
- **Library:** `hmmlearn.hmm.CategoricalHMM`
- **Training Data:** Only correct CoT traces (correct_only=True)
- **Hyperparameters:**
  - n_components: 6 hidden states (typical: 5-8)
  - n_iter: 100 EM iterations
  - random_state: 42

### 1.2 Ensemble HMM Training
- **File:** `src/hmm/ensemble.py`
- **Approach:** Train multiple HMMs with different random seeds
- **Ensemble size:** 10-12 models
- **Purpose:** Robust scoring, uncertainty estimation

### 1.3 Model Analysis
- **transition_matrix:** P(state_t+1 | state_t) - state transition probabilities
- **emission_matrix:** P(observation | state) - tag emission probabilities
- **initial_distribution:** P(start state) - π parameters

---

## 2. SCORING METHODS

### 2.1 Log-Likelihood Scoring
```python
def compute_log_likelihood(model, tag_sequence):
    seq = np.array([TAG_TO_ID[t] for t in tag_sequence]).reshape(-1, 1)
    return model.score(seq)
```
- **Output:** Raw log-likelihood (negative, typically -10 to -50)
- **Normalization:** Score/T to get per-step likelihood

### 2.2 Transition Entropy
```
H(A) = -Σᵢⱼ Aᵢⱼ log Aᵢⱼ
```
- Measures randomness in state transitions
- Lower entropy = more deterministic, structured reasoning

### 2.3 Anomaly Detection
- Flag transitions where P(transition) < threshold
- Default threshold: 0.05
- Reports position, tags, and probability

### 2.4 Viterbi Decoding
- Find most likely hidden state sequence
- Used for detailed analysis of reasoning paths

---

## 3. TAGGING METHODS

### 3.1 Rule-Based Tagging (Current)
- **File:** `src/tagging/rule_based_tagger.py`
- **Method:** Regex pattern matching with priority ordering
- **Patterns per tag:** See TAG_PATTERNS dict in file

### 3.2 Step Segmentation
- **File:** `src/tagging/step_segmenter.py`
- **Splits on:**
  - Newlines
  - "Step X:" headers
  - Sentence boundaries (., !, ?)

### 3.3 NeuralTagger (Future/Weeks 7-8)
- **File:** `src/neural_hmm/neural_hmm_hybrid.py`
- **Class:** `NeuralTagger` - fine-tuned DistilBERT-base-uncased
- **Fine-tuning:** Cross-entropy on tagged sequences (5 epochs, lr=2e-5)
- **Output:** Softmax probabilities over 8 tags
- **Neural-HMM Integration:** Confidence-weighted emissions into HMM

---

## 4. BASELINE METHODS (Heuristic)

| Method | Formula | File |
|--------|---------|------|
| LengthBaseline | log(char) * 0.1 + log(steps) * 0.5 | `src/evaluation/baselines.py` |
| TagDiversityBaseline | (unique/total) * log(total) | `src/evaluation/baselines.py` |
| StructureBaseline | Sum of binary features (max 6.0) | `src/evaluation/baselines.py` |
| TransitionBaseline | (good_trans/total_trans) * log*2 | `src/evaluation/baselines.py` |
| CombinedBaseline | 0.2L + 0.2D + 0.3S + 0.3T | `src/evaluation/baselines.py` |
| RandomBaseline | uniform(0, 1) | `src/evaluation/baselines.py` |

---

## 5. NEURAL BASELINE METHODS (Planned)

### 5.1 ROSCOE Baseline
- **File:** `src/experiments_v2/roscoe_baseline.py`
- **Metrics:**
  - Semantic alignment (cosine sim between steps)
  - Step consistency (entailment premise→step)
  - Informativeness (novelty vs problem)
- **Backbone:** sentence-transformers/all-MiniLM-L6-v2

### 5.2 PiCSAR Baseline
- **File:** `src/experiments_v2/picsar_baseline.py`
- **Metric:** Joint log-likelihood of CoT given problem
- **Model:** distilgpt2 (no API key needed)

---

## 6. STATISTICAL METHODS

### 6.1 Bootstrap Confidence Intervals
- **File:** `src/experiments_v2/bootstrap_ci.py`
- **Procedure:** n=10,000 bootstrap resamples
- **Output:** 95% CI around correlation and AUC

### 6.2 Effect Sizes
- **File:** `src/experiments_v2/effect_size.py`
- **Metrics:**
  - Cohen's d: Standardised mean difference
  - Cliff's delta: Non-parametric effect size
- **FDR Correction:** Benjamini-Hochberg for multiple comparisons

### 6.3 Correlation Analysis
- Pearson correlation between HMM scores and correctness
- Spearman rank correlation

---

## 7. REJECTION SAMPLING METHODS

### 7.1 StructureReward
- **File:** `src/rejection_sampling/rejection_sampling.py`
- **Score range:** [-10, 0] (clipped)
- **Normalized:** sigmoid transformation relative to baseline

### 7.2 PPO Integration
- **Combined reward:** (1-α) * task_reward + α * structure_reward
- **α (weight):** 0.3 (default)
- **Purpose:** Train LLMs with structural guidance

---

## 8. EVALUATION METRICS

### 8.1 Correlation Metrics
- Pearson r
- Spearman ρ
- p-values (two-tailed)

### 8.2 Classification Metrics
- Accuracy (using median threshold)
- AUC (Area Under ROC Curve)
- Precision, Recall, F1

### 8.3 Domain Transfer Metrics
- Arc-cosine similarity between tag distributions
- KL divergence: Math vs Science patterns

---

## 9. EXPERIMENT SCRIPTS

| Script | Purpose | Entry Point |
|--------|---------|-------------|
| `run_real_generation.py` | Generate CoT from LLMs | scripts/generation/ |
| `run_baseline_eval.py` | Run full evaluation | scripts/experiments/ |
| `run_ablations.py` | Hidden state ablation | scripts/experiments/ |
| `post_process_correctness.py` | Verify answers | scripts/analysis/ |

---

## 10. DATA PIPELINE

```
Input: Raw problem + ground truth
  ↓
generate_cot.py → LLM inference → cot_raw
  ↓
rule_based_tagger.py → tag_sequence, steps
  ↓
post_process_correctness.py → is_correct
  ↓
trainer.py (correct_only) → HMM model
  ↓
scorer.py → hmm_scores
  ↓
baselines.py → baseline_scores
  ↓
evaluate() → correlation, AUC, stats
```

---

## 11. MODEL VERSIONS

| Model | Size | Provider | Notes |
|-------|------|----------|-------|
| Qwen-0.5B | 0.5B | HuggingFace | Current baseline |
| Qwen-1.5B | 1.5B | HuggingFace | Current in paper |
| Llama-3.2-1B | 1B | Meta | Week 2 target |
| Mistral-7B | 7B | Mistral | Week 2 target |
| Phi-3.5-Mini | 3.8B | Microsoft | Alternative |

---

## 12. KEY HYPERPARAMETERS

| Parameter | Current | Target | File |
|-----------|---------|--------|------|
| n_hidden_states | 5-6 | 3,5,8,12 (ablation) | trainer.py |
| n_ensemble | 10 | 12 | trainer.py |
| anomaly_threshold | 0.05 | 0.01-0.1 | scorer.py |
| tagging_method | rule_based | neural | tagger.py |
| K candidates | 8 | 8, 16 | rejection_sampling.py |
| bootstrap_samples | - | 10000 | bootstrap_ci.py |