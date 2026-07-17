# Chain-of-Thought HMM Architecture Documentation

Generated: 2026-07-17

## EXECUTIVE SUMMARY

CoT-HMM is a lightweight, unsupervised method for scoring Chain-of-Thought reasoning plausibility using Hidden Markov Models. The system learns structural patterns from correct reasoning chains and flags anomalous reasoning.

---

## 1. SYSTEM ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CoT-HMM PIPELINE                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  CoT Collection │ →  │ Step Tagging    │ →  │ HMM Training    │ →  │ Scoring & Eval  │
│                 │    │                 │    │                 │    │                 │
│ - LLM inference │    │ - Segmentation  │    │ - Fit on good   │    │ - Log-likelihood│
│ - Multi-model   │    │ - Rule-based    │    │   CoT traces    │    │ - Entropy       │
│ - Multi-domain  │    │   classification│    │ - Learn A, B, π │    │ - Anomalies     │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                   │
         ▼                       ▼                       ▼                   ▼
    data/cot_outputs/      data/tagged/            models/*.pkl      experiments/results/
```

---

## 2. CORE ML MODEL: HIDDEN MARKOV MODEL

### 2.1 Mathematical Formulation

**Hidden States (Latent):**
- N hidden states representing latent reasoning phases
- Discovered automatically during training
- Typical patterns: Setup → Plan → Calculate → Deduce → Answer

**Observations (Emissions):**
- Discrete tag sequence from vocabulary of 8 semantic tags
- Tags: PR, DV, EQ, CA, IN, RF, VE, CO

**Parameters Learned:**

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Initial State Distribution | π | P(z₁) - probability of starting in each hidden state |
| Transition Matrix | A | P(zₜ₊₁ | zₜ) - state-to-state transitions |
| Emission Matrix | B | P(xₜ | zₜ) - probability of observing each tag in each state |

### 2.2 HMM Implementation Details

```
File: src/hmm/trainer.py
- Model class: hmmlearn.hmm.CategoricalHMM
- n_components: Number of hidden states (default: 6)
- n_iter: Baum-Welch EM iterations (default: 100)
- Training: Only on CORRECT CoT traces (correct_only=True)
```

### 2.3 Scoring Algorithm

```
Score(O_test) = (1/T) * log P(O_test | λ)

Where:
- T = sequence length (number of steps)
- λ = trained HMM parameters (A, B, π)
- Higher score = more plausible reasoning structure

Additional metrics:
- Transition Entropy: H(A) = -Σᵢⱼ Aᵢⱼ log Aᵢⱼ
- Anomaly Detection: Flag transitions with P < threshold
```

---

## 3. SEMANTIC TAGGING METHODS

### 3.1 Tag Vocabulary (8 Tags)

| Tag | Meaning | Patterns Detected |
|-----|---------|-------------------|
| PR | Problem Restatement | "need to find", "question asks", "given" |
| DV | Define Variable | "let x =", "define", "suppose" |
| EQ | Equation Setup | "= ", "formula", "equation" |
| CA | Calculation | Numerical operations: ×, +, -, ÷, = |
| IN | Inference | "therefore", "thus", "hence", "implies" |
| RF | Reflection | "wait", "recheck", "mistake", "actually" |
| VE | Verification | "check", "verify", "confirm" |
| CO | Conclusion | "answer is", "final answer", "boxed" |

### 3.2 Rule-Based Tagger (Current)

```
File: src/tagging/rule_based_tagger.py
- Regex pattern matching with priority ordering
- Priority order: CO > VE > PR > EQ > CA > DV > RF > IN
- Returns: (tag, confidence) tuple
- Confidence: 0.5 + (matches * 0.1), capped at 1.0
```

### 3.3 Neural Tagger (Planned/Future)

```
File: src/neural_hmm/neural_hmm_hybrid.py
- Model: DistilBERT-base-uncased → 8-class classifier
- Fine-tuned on existing tagged sequences
- Output: Softmax distribution over 8 tags
- Advantage: Handles ambiguous cases better than rules
```

---

## 4. BASELINE METHODS (6 Heuristics)

### 4.1 Length Baseline
```
Score = log(char_count) * 0.1 + log(step_count) * 0.5
Hypothesis: Longer reasoning chains indicate more thorough thinking
```

### 4.2 Tag Diversity Baseline
```
Score = (unique_tags / total_tags) * log(total_tags)
Hypothesis: Diverse reasoning steps suggest comprehensive problem-solving
```

### 4.3 Structure Baseline
```
Score = Σ binary features:
  - Has calculation (+1.0)
  - Has conclusion (+1.5)
  - Has reasoning (+0.5)
  - Has numbers (+0.5)
  - Has PR tag (+1.0)
  - Has CA tag (+1.0)
  - Has CO tag (+1.5)
```

### 4.4 Transition Baseline
```
Good transitions defined:
  - PR → {DV, RF, CA, EQ}
  - DV → {EQ, CA}
  - EQ → {CA}
  - CA → {IN, VE, CO}
  - IN → {CA, VE, CO}
  - VE → {CO}

Score = (good_transitions / total_transitions) * log(total_transitions) * 2
```

### 4.5 Combined Baseline
```
Score = 0.2 * length + 0.2 * diversity + 0.3 * structure + 0.3 * transition
```

### 4.6 Random Baseline
```
Score = uniform(0, 1)
Purpose: Sanity check - all methods should beat random
```

---

## 5. NEURAL-HMM HYBRID (Weeks 7-8)

```
Architecture:

┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│   Step Text     │ →   │ NeuralTagger        │ →   │ NeuralHMMScorer │
│                 │     │                     │     │                 │
│ "5 + 3 = 8"     │     │ DistilBERT          │     │ HMM Ensemble    │
└─────────────────┘     │ (fine-tuned)        │     │                 │
                        │ Softmax(8 tags)     │     │ Soft emissions    │
                        └─────────────────────┘     └─────────────────┘
                                 │                           │
                                 ▼                           ▼
                          [0.1, 0.0, 0.0,  │          Score = E[log P(obs|state)]
                           0.8, 0.0, 0.0, │
                           0.0, 0.1]      │
```

---

## 6. REJECTION SAMPLING (Weeks 9-10)

```
Pipeline:

Problem
  │
  ▼
┌─────────────────────────────────────┐
│ Generate K=8 candidates             │
│   - Greedy decoding (candidate 0)   │
│   - Alternatives 1..K-1             │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ Score each with HMM Ensemble          │
│   - StructureReward.compute()        │
│   - Returns: [-5, 0] (higher = better)│
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ Select highest-scoring candidate      │
│ Report:                             │
│   - greedy_accuracy                 │
│   - selected_accuracy               │
│   - oracle_accuracy (upper bound)   │
│   - absolute_improvement            │
└─────────────────────────────────────┘

PPO Integration (Week 10):
reward = (1-α) * task_reward + α * structural_reward
```

---

## 7. DATA SCHEMA (v2 Format)

```
File: src/experiments_v2/data_schema.py

ScoredTrace {
  sample_id: str
  dataset: "gsm8k" | "arc_challenge" | "math"
  domain: "math" | "science" | "mixed"
  model_family: "qwen" | "llama" | "mistral" | "phi"
  model_name: str (HuggingFace ID)
  model_params_b: float
  
  problem: str
  ground_truth_answer: str
  
  cot_text: str
  model_answer: str
  is_correct: bool
  tag_sequence: List[str]
  tagging_method: "rule_based" | "neural"
  
  hmm_score: HMMScoreRecord
  roscoe_score: ROSCOEScoreRecord
  picsar_score: PiCSARScoreRecord
  neural_hmm_score: NeuralHMMScoreRecord
  rejection_sampling: RejectionSamplingRecord
}
```

---

## 8. EXPERIMENT PIPELINE FLOW

```
Week 1-2 (Data Scaling):
generate_cot.py → tag_cot → verify_correctness → data/v2/*.json

Week 3 (ROSCOE Baseline):
roscoe_baseline.py -- data/v2/*.json → roscoe_score

Week 4 (PiCSAR Baseline):
picsar_baseline.py -- data/v2/*.json → picsar_score

Week 5 (Bootstrap CIs):
bootstrap_ci.py -- scored data → confidence_intervals

Week 6 (Effect Sizes):
effect_size.py -- scored data → effect_sizes

Week 7-8 (Neural-HMM):
neural_hmm_hybrid.py -- data/v2/*.json → neural_hmm_score

Week 9-10 (PPO/RL Validation):
rejection_sampling.py -- data/v2/*.json → rejection_testing

Week 11 (ARC Investigation):
arc_domain_analysis.py -- cross_domain scores → analysis

Week 12 (Submission Polish):
consolidate → main.pdf
```

---

## 9. FILES STRUCTURE REFERENCE

| Component | Primary Files |
|-----------|--------------|
| HMM Training | src/hmm/trainer.py, src/hmm/ensemble.py |
| HMM Scoring | src/hmm/scorer.py |
| Rule Tagging | src/tagging/rule_based_tagger.py |
| Neural Tagging | src/neural_hmm/neural_hmm_hybrid.py |
| Baselines | src/evaluation/baselines.py |
| ROSCOE | src/experiments_v2/roscoe_baseline.py |
| PiCSAR | src/experiments_v2/picsar_baseline.py |
| Rejection Sampling | src/rejection_sampling/rejection_sampling.py |
| Bootstrap CIs | src/experiments_v2/bootstrap_ci.py |
| Effect Sizes | src/experiments_v2/effect_size.py |
| Data Schema | src/experiments_v2/data_schema.py |
| Statistical Tests | src/evaluation/statistical_tests.py |
| Visualizations | src/evaluation/baseline_plots.py |

---

## 10. KEY RESULTS (Current State)

| Dataset | Model | Correlation | AUC | p-value |
|---------|-------|-------------|-----|---------|
| GSM8K | Qwen-0.5B | 0.21 | 0.62 | 0.008 |
| GSM8K | Qwen-1.5B | 0.25 | 0.65 | 0.032 |
| ARC (Transfer) | Qwen-0.5B | 0.04 | 0.46 | 0.62 |

**Note:** ARC AUC=0.46 is below chance - requires root cause analysis (Week 11)

---

## 11. TARGETS FOR IEEE SUBMISSION

| Metric | Current | Target |
|--------|---------|--------|
| Test set size | ~150 (GSM8K) | 500–1000+ |
| Model families | 2 (Qwen) | 3+ (Qwen + Llama + Mistral) |
| Neural baselines | 0 | ROSCOE + PiCSAR |
| AUC (main method) | 0.62 | 0.68+ or reframed |
| Downstream utility demo | ❌ | ✅ Rejection-sampling |
| Bootstrap CIs reported | ❌ | ✅ All headline numbers |
| Neural-HMM hybrid | Future work | ✅ Implemented |