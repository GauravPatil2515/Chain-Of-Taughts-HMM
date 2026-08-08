# Methods Reference - CoT-HMM

## OVERVIEW

Catalog of all computational methods, algorithms, mathematical formulations, and evaluation metrics in the CoT-HMM framework.

---

## 1. HMM TRAINING & ENSEMBLE METHODS

### 1.1 Categorical HMM Training
- **File:** `src/hmm/trainer.py`
- **Algorithm:** Baum-Welch Expectation-Maximization (EM)
- **Library:** `hmmlearn.hmm.CategoricalHMM`
- **Training Data:** Fitted on correct tag sequences only
- **Hyperparameters:** `n_components` $\in \{3, 5, 8, 12\}$, `n_iter` = 100, `random_state` = seed

### 1.2 Ensemble HMM Training ($M=12$)
- **File:** `src/hmm/ensemble.py`
- **Model Class:** `EnsembleHMM`
- **Architecture:** Ensemble of $M=12$ Categorical HMMs with seed diversity across hidden state configs $N \in \{3, 5, 8, 12\}$.
- **Scoring:** Normalized mean step log-likelihood across ensemble:
  $$\text{Score}_{\text{Ensemble}}(\mathbf{o}) = \frac{1}{M} \sum_{m=1}^{M} \frac{1}{T} \log P(\mathbf{o} \mid \lambda_m)$$

### 1.3 Contrastive Anomaly HMM
- **File:** `src/hmm/ensemble.py`
- **Model Class:** `AnomalyHMM`
- **Training Data:** Fitted exclusively on incorrect reasoning traces.

---

## 2. SCORING METRICS & FORMULATIONS

### 2.1 Plausibility Margin Score
- **Formula:** $\text{Margin}(\mathbf{o}) = \text{Score}_{\text{Ensemble}}(\mathbf{o}) - \text{Score}_{\text{Anomaly}}(\mathbf{o})$
- **Performance:** Out-of-fold ROC-AUC = **0.983** [95% CI: 0.966, 0.997], Pearson $r = 0.476$, Cohen's $d = 1.093$.

### 2.2 Entropy Drift Rate (EDR)
- **Formula:** $\text{EDR}(\mathbf{o}) = \frac{H(\mathbf{o}_{>T/2}) - H(\mathbf{o}_{\le T/2})}{T/2}$
- **Performance:** Pearson $r = -0.282$ [95% CI: -0.376, -0.190], $p < 0.001$. Indicates structural chaos accumulation over long reasoning chains.

### 2.3 Structural Entropy
- **Formula:** $H(A) = -\sum_{i,j} A_{ij} \log_2 (A_{ij} + \epsilon)$
- **Performance:** Pearson $r = -0.732$ [95% CI: -0.763, -0.698], $p < 0.001$, Cohen's $d = -2.144$.

---

## 3. COMPUTE ROUTING GATE METHOD

### 3.1 3-Tier Adaptive Routing Gate
- **File:** `scripts/run_leakfree_evaluation.py`
- **Logic:**
  - **Pass Tier ($\text{Margin} \ge +0.5$)**: 30% volume, **99.3% accuracy** (direct output, no re-sampling required).
  - **Escalation Tier ($-0.5 < \text{Margin} < +0.5$)**: 35% volume, **50.9% accuracy** (escalate for self-consistency re-sampling).
  - **Reject Tier ($\text{Margin} \le -0.5$)**: 35% volume, **6.9% accuracy** (early stopping, reject flawed reasoning).
- **Compute Savings:** **30.0% reduction** in LLM inference overhead.

---

## 4. BASELINE METHODS

| Method | Description / Formula | ROC-AUC | Pearson r |
|---|---|---|---|
| **Heuristic Baseline** | Weighted ensemble of step count, tag diversity, and transition rules | 0.860 | 0.459 |
| **Repetition Penalty** | Penalty for repeated tag loops | 0.843 | 0.634 |
| **ROSCOE** | Neural embedding verifier (`sentence-transformers/all-MiniLM-L6-v2`) | 0.380 | — |
| **PiCSAR** | Content-based LLM log-likelihood (`distilgpt2`) | 0.460 | — |

---

## 5. STATISTICAL EVALUATION PROTOCOL

### 5.1 Percentile Bootstrap Confidence Intervals
- **File:** `src/experiments_v2/bootstrap_ci.py`
- **Iterations:** $B = 10,000$ bootstrap resamples.
- **Intervals:** 95% Percentile Confidence Intervals.

### 5.2 Effect Size Calculations
- **File:** `src/experiments_v2/effect_size.py`
- **Metrics:** Cohen's $d$ (parametric standardized mean difference) and Cliff's $\delta$ (non-parametric ordinal dominance effect size).