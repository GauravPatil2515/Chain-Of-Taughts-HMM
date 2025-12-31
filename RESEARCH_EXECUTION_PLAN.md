# Research Execution Plan: CoT-HMM

**Project:** Chain-of-Thought Plausibility Scorer using Hidden Markov Models  
**Target:** Research-grade publication (ACL SRW / EMNLP Findings / IEEE)  
**Timeline:** 8 weeks  
**Status:** Phase 1 - Data Collection

---

## 📅 Timeline Overview

| Phase | Duration | Status | Deliverable |
|-------|----------|--------|-------------|
| Phase 1: Data Collection | Week 1-2 | 🟡 IN PROGRESS | 300+ CoT samples |
| Phase 2: Tagging & Validation | Week 2-3 | ⚪ PENDING | Tagged dataset + validation report |
| Phase 3: HMM Training | Week 3 | ⚪ PENDING | Trained models + validation |
| Phase 4: Statistical Evaluation | Week 4-5 | ⚪ PENDING | Full results + significance tests |
| Phase 5: Analysis | Week 5-6 | ⚪ PENDING | Case studies + ablations |
| Phase 6: Paper Writing | Week 6-8 | ⚪ PENDING | Submission-ready manuscript |

---

## 🎯 Phase 1: Data Collection (CURRENT)

### Target Specifications

#### Dataset Distribution
```
GSM8K (Math):
  - Questions: 150
  - Models: 4 (DeepSeek, Qwen, Phi, LLaMA)
  - Per Model: 150 samples
  - Subtotal: 600 samples

StrategyQA (Commonsense):
  - Questions: 100
  - Models: 3 (DeepSeek, Qwen, Phi)
  - Per Model: 100 samples
  - Subtotal: 300 samples

ARC-Easy (Science/Logic):
  - Questions: 50
  - Models: 2 (Qwen, Phi)
  - Per Model: 50 samples
  - Subtotal: 100 samples

TOTAL: 1000 samples
```

#### Correctness Distribution Target
- ✅ 60% correct answers
- ❌ 40% incorrect answers

### Models to Use

1. **Qwen/Qwen2.5-0.5B-Instruct** (Primary - CPU friendly)
2. **Qwen/Qwen2.5-1.5B-Instruct** (If GPU available)
3. **microsoft/Phi-3.5-mini-instruct** (Alternative)
4. **DeepSeek-R1-Distill-Qwen-1.5B** (If accessible)

### Generation Parameters (MUST BE CONSISTENT)

```python
generation_config = {
    "max_new_tokens": 512,
    "temperature": 0.7,      # NOT 0 - need diversity
    "top_p": 0.9,
    "do_sample": True,
    "random_seed": 42        # For reproducibility
}
```

### Data Schema (REQUIRED FIELDS)

```json
{
  "id": "gsm8k_qwen_0001",
  "dataset": "gsm8k",
  "domain": "math",
  "question": "...",
  "ground_truth": "42",
  "model": "qwen2.5-0.5b",
  "model_full_name": "Qwen/Qwen2.5-0.5B-Instruct",
  "cot_raw": "First, I need to...",
  "predicted_answer": "42",
  "is_correct": true,
  "generation_params": {
    "temperature": 0.7,
    "max_tokens": 512,
    "seed": 42
  },
  "timestamp": "2025-12-27T10:30:00Z",
  "prompt_template": "standard_cot"
}
```

### Progress Tracking

**Week 1:**
- [ ] Day 1-2: GSM8K dataset download (150 questions)
- [ ] Day 3-4: Generate Qwen model (150 samples)
- [ ] Day 5-6: Generate Phi model (150 samples)
- [ ] Day 7: Quality check + correct/incorrect balance

**Week 2:**
- [ ] Day 1-3: StrategyQA + ARC generation
- [ ] Day 4-5: Additional model runs if needed
- [ ] Day 6-7: Data validation + statistics

### Success Criteria ✅
- [ ] n ≥ 300 total samples
- [ ] 60/40 correct/incorrect split (±10%)
- [ ] All fields populated correctly
- [ ] No duplicate questions
- [ ] Consistent generation parameters

---

## 🏷️ Phase 2: Tagging & Validation

### Tagging Process

1. **Auto-tag** all samples using rule_based_tagger.py
2. **Manual validation** on 50-100 samples (10%)
3. **Inter-annotator agreement** (if multiple taggers)
4. **Edge case documentation**

### Tag Quality Metrics

```python
target_metrics = {
    "avg_steps_per_cot": (4, 8),        # 4-8 steps
    "tag_diversity": 3,                  # ≥3 unique tags
    "tagging_confidence": 0.75,          # ≥75% high confidence
    "inter_annotator_kappa": 0.70        # Cohen's κ ≥ 0.7
}
```

### Validation Checklist
- [ ] Manual review of 50 random samples
- [ ] Document 10+ edge cases
- [ ] Compute tag distribution
- [ ] Check for systematic errors
- [ ] Update tagging rules if needed

---

## 🧮 Phase 3: HMM Training

### Training Strategy

```python
training_config = {
    "correct_only": True,
    "train_test_split": 0.8,
    "n_components": [4, 5, 6],           # Tune
    "n_iterations": 100,
    "cv_folds": 5,
    "random_state": 42
}
```

### Domain-Specific Models

- **Math HMM**: Train on GSM8K correct samples
- **Commonsense HMM**: Train on StrategyQA correct samples
- **General HMM**: Train on all domains combined

### Validation
- [ ] 5-fold cross-validation
- [ ] Held-out test set (20%)
- [ ] Convergence monitoring
- [ ] Hyperparameter tuning results

---

## 📊 Phase 4: Statistical Evaluation

### Required Statistical Tests

1. **Correlation Analysis**
   - Point-biserial correlation (score vs correctness)
   - Spearman rank correlation (method comparison)
   - 95% Confidence Intervals
   - p-values with Bonferroni correction

2. **Classification Metrics**
   - Accuracy, Precision, Recall, F1
   - AUC-ROC with DeLong's test
   - McNemar's test for paired comparisons

3. **Effect Sizes**
   - Cohen's d for mean differences
   - Interpretation: small (0.2), medium (0.5), large (0.8)

4. **Robustness**
   - Bootstrap resampling (1000 iterations)
   - Cross-domain validation
   - Model-agnostic evaluation

### Significance Threshold
- α = 0.05 (with multiple testing correction)
- Minimum effect size: d ≥ 0.3 (small-to-medium)

---

## 🔬 Phase 5: Deep Analysis

### Case Study Selection (6-8 examples)

**Category 1: HMM Success Cases** (2 examples)
- High HMM score, low baseline scores
- Correct answer
- Shows subtle reasoning pattern detection

**Category 2: HMM Failure Cases** (2 examples)
- Low HMM score, high baseline scores
- Analyze why HMM failed
- Identify limitations

**Category 3: Agreement Cases** (2 examples)
- All methods agree (1 correct, 1 incorrect)
- Validate on obvious cases

**Category 4: Interesting Edge Cases** (2 examples)
- Coherent reasoning, wrong answer
- Correct answer, poor reasoning

### Ablation Studies

- [ ] Remove each tag type (8 ablations)
- [ ] Vary n_components (3-8 states)
- [ ] Domain transfer (train Math, test Logic)
- [ ] Chain length sensitivity

---

## 📝 Phase 6: Paper Writing

### Target Venue Priority

**Primary Targets:**
1. **ACL Student Research Workshop** (June deadline)
2. **EMNLP Findings** (May deadline)
3. **COLING** (Rolling)

**Backup Targets:**
4. **IEEE ICSC** (Semantic Computing)
5. **AAAI Student Abstract**

### Paper Structure (8 pages ACL format)

```
1. Abstract (200 words)
2. Introduction (1 page)
3. Related Work (0.75 page)
4. Method (2 pages)
   - Tagging Scheme
   - HMM Formulation
   - Baseline Methods
5. Experiments (1 page)
6. Results (2 pages)
7. Analysis (1.5 pages)
8. Conclusion (0.5 page)
9. References
10. Appendices (unlimited)
```

### Writing Schedule

**Week 6:**
- [ ] Draft Abstract, Introduction, Method
- [ ] Create all result tables
- [ ] Generate all figures

**Week 7:**
- [ ] Draft Results, Analysis, Conclusion
- [ ] Related Work section
- [ ] First complete draft

**Week 8:**
- [ ] Revisions based on advisor feedback
- [ ] Proofread and polish
- [ ] Format for submission
- [ ] Submit!

---

## ✅ Success Criteria (Paper-Ready)

### Minimum Requirements (Publishable)
- ✅ n ≥ 300 samples
- ✅ 3+ models evaluated
- ✅ 2+ domains
- ✅ Correlation r > 0.3 OR F1 > 0.70
- ✅ HMM outperforms ≥2 baselines (p < 0.05)
- ✅ 4+ case studies

### Strong Paper (High Acceptance Rate)
- ✅ n ≥ 500 samples
- ✅ 4 models × 3 domains
- ✅ Domain transfer experiments
- ✅ HMM outperforms ≥4 baselines
- ✅ Human evaluation (100 samples)
- ✅ Released code + dataset

---

## 🚨 Critical Dos and Don'ts

### DATA COLLECTION ⚠️

**DO:**
- ✅ Log ALL generation parameters
- ✅ Save raw model outputs
- ✅ Use consistent prompting
- ✅ Balance correct/incorrect
- ✅ Include diverse question types

**DON'T:**
- ❌ Mix different prompts
- ❌ Cherry-pick examples
- ❌ Use only easy questions
- ❌ Forget random seeds

### EVALUATION ⚠️

**DO:**
- ✅ Report ALL metrics
- ✅ Include confidence intervals
- ✅ Test significance
- ✅ Check assumptions

**DON'T:**
- ❌ P-hack metrics
- ❌ Hide negative results
- ❌ Claim causality
- ❌ Ignore baselines

### WRITING ⚠️

**DO:**
- ✅ Be honest about limitations
- ✅ Release code + data
- ✅ Clear methodology
- ✅ Proofread thoroughly

**DON'T:**
- ❌ Oversell results
- ❌ Ignore related work
- ❌ Use vague language
- ❌ Submit without review

---

## 📊 Progress Dashboard

**Overall Progress: 15% Complete**

| Phase | Tasks | Complete | Progress |
|-------|-------|----------|----------|
| Phase 1 | 10 | 2 | ████░░░░░░ 20% |
| Phase 2 | 8 | 0 | ░░░░░░░░░░ 0% |
| Phase 3 | 6 | 0 | ░░░░░░░░░░ 0% |
| Phase 4 | 10 | 0 | ░░░░░░░░░░ 0% |
| Phase 5 | 12 | 0 | ░░░░░░░░░░ 0% |
| Phase 6 | 15 | 0 | ░░░░░░░░░░ 0% |

**Next Immediate Actions:**
1. ✅ Download GSM8K (200 questions)
2. ✅ Modify generate_local.py for batch generation
3. ⏳ Run first batch (50 samples) - TEST
4. ⏳ Scale to 150 samples per model

---

## 📞 Getting Help

**Stuck on something?**
- Data generation issues → Check generate_local.py logs
- Tagging problems → Review TAGGING_GUIDE.md
- Statistical tests → See src/evaluation/statistical_tests.py
- Paper writing → Use paper_template.tex

**Regular Check-ins:**
- End of Week 1: Data collection progress
- End of Week 3: Preliminary results
- End of Week 5: Full analysis complete
- End of Week 7: Paper draft ready

---

**Last Updated:** December 27, 2025  
**Current Phase:** Phase 1 - Data Collection  
**Days Elapsed:** 0  
**Days Remaining:** 56
