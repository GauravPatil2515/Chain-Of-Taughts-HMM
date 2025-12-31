# Research Plan: CoT-HMM Project

## 📋 Project Overview

**Title**: CoT-HMM: A Lightweight, Text-Only Plausibility Scorer for Chain-of-Thought Reasoning

**Objective**: Develop and evaluate a Hidden Markov Model-based framework for scoring the coherence and plausibility of Chain-of-Thought reasoning in LLMs.

**Duration**: 8 weeks

**Team Size**: 1-3 students

---

## 🎯 Research Questions

1. **RQ1**: Can an HMM trained on tagged CoT steps effectively distinguish between correct and incorrect reasoning chains?

2. **RQ2**: How do HMM-based coherence scores compare across different LLM architectures (reasoning-specific vs. general-purpose)?

3. **RQ3**: What reasoning patterns (transition sequences) are characteristic of successful vs. failing reasoning?

4. **RQ4**: Is the proposed text-only approach competitive with embedding-based methods while being significantly more lightweight?

---

## 📅 Week-by-Week Plan

### Week 1: Literature Review & Setup

**Objectives**:

- [ ] Complete literature review of related work
- [ ] Set up development environment
- [ ] Download datasets

**Tasks**:

| Task | Time | Output |
|------|------|--------|
| Read EIRL, MCoT, MarCos, EMNLP state-aware papers | 4 hours | Annotated bibliography |
| Read HMM tutorial (Rabiner 1989) | 2 hours | Understanding of HMM math |
| Set up Python environment | 1 hour | Working `venv` with dependencies |
| Download GSM8K, StrategyQA datasets | 1 hour | `data/raw/` populated |
| Initialize Git repository | 0.5 hours | GitHub repo created |

**Deliverables**:

- `docs/LITERATURE_REVIEW.md`
- Working development environment
- Raw datasets downloaded

---

### Week 2: Data Collection (CoT Generation)

**Objectives**:

- [ ] Generate CoT outputs from 3-4 models
- [ ] Build data collection pipeline

**Tasks**:

| Task | Time | Output |
|------|------|--------|
| Write CoT generation script | 3 hours | `src/data_collection/generate_cot.py` |
| Run inference on DeepSeek-R1-Distill | 3 hours | 200 CoT outputs |
| Run inference on Phi-3.5-Mini | 3 hours | 200 CoT outputs |
| Run inference on Qwen2.5 | 3 hours | 200 CoT outputs |
| Run inference on LLaMA-3.2 | 3 hours | 200 CoT outputs |
| Validate and store outputs | 2 hours | `data/cot_outputs/` |

**Deliverables**:

- ~800 CoT traces across 4 models
- JSON files with question, CoT, answer, correctness

**Notes**:

- Use HuggingFace Transformers for local inference
- If compute-limited, use smaller model variants (1B-3B)
- Consider using Google Colab for GPU access

---

### Week 3: Tagging System Development

**Objectives**:

- [ ] Implement step segmentation
- [ ] Implement rule-based tagger
- [ ] Manually validate tagging quality

**Tasks**:

| Task | Time | Output |
|------|------|--------|
| Implement step segmenter | 2 hours | `src/tagging/step_segmenter.py` |
| Implement rule-based tagger | 4 hours | `src/tagging/rule_based_tagger.py` |
| Manually tag 50 CoTs for validation | 3 hours | Gold standard annotations |
| Compute tagging accuracy | 1 hour | Accuracy report |
| Iterate and refine patterns | 2 hours | Improved tagger |

**Deliverables**:

- Working tagging pipeline
- Validation report (accuracy ≥ 80%)
- `data/tagged/` with all tagged CoTs

**Metrics**:

- Pattern precision: ≥ 85%
- Default tag (`IN`) rate: ≤ 20%
- Step segmentation accuracy: ≥ 95%

---

### Week 4: HMM Training & Initial Experiments

**Objectives**:

- [ ] Train HMM on correct CoTs
- [ ] Implement scoring functions
- [ ] Run initial experiments

**Tasks**:

| Task | Time | Output |
|------|------|--------|
| Implement HMM trainer | 3 hours | `src/hmm/trainer.py` |
| Train HMM on correct CoTs only | 1 hour | `models/hmm_model.pkl` |
| Implement scoring functions | 2 hours | `src/hmm/scorer.py` |
| Score all CoTs (correct + incorrect) | 1 hour | Score dataframe |
| Visualize transition matrix | 1 hour | Heatmap figure |
| Initial correlation analysis | 2 hours | Preliminary results |

**Deliverables**:

- Trained HMM model
- Scoring pipeline
- Initial results notebook

**Key Metrics to Compute**:

- Log-likelihood scores for all CoTs
- Correlation between score and correctness
- Transition entropy

---

### Week 5: Evaluation & Baseline Comparisons

**Objectives**:

- [ ] Full evaluation on all models/datasets
- [ ] Compare with baseline methods
- [ ] Statistical significance testing

**Tasks**:

| Task | Time | Output |
|------|------|--------|
| Implement baseline methods | 3 hours | Random, step count, bigram |
| Run full evaluation | 2 hours | Complete results |
| Cross-model comparison | 2 hours | Model comparison table |
| Statistical significance tests | 2 hours | p-values |
| Per-dataset analysis | 2 hours | Domain-specific insights |

**Deliverables**:

- Complete evaluation results
- Comparison with baselines
- Statistical analysis

**Expected Results Table**:

| Model | Accuracy | HMM Score (μ) | Correlation (ρ) |
|-------|----------|---------------|-----------------|
| DeepSeek-R1 | 82% | -12.3 | 0.45 |
| Phi-3.5 | 75% | -14.1 | 0.38 |
| Qwen2.5 | 70% | -15.5 | 0.32 |
| LLaMA-3.2 | 65% | -18.2 | 0.28 |

---

### Week 6: Ablation Studies & Deep Analysis

**Objectives**:

- [ ] Ablation on number of hidden states
- [ ] Ablation on tag granularity
- [ ] Case study analysis

**Tasks**:

| Task | Time | Output |
|------|------|--------|
| Ablation: vary n_components (4, 6, 8, 10) | 2 hours | Ablation table |
| Ablation: vary tag granularity (3, 5, 8 tags) | 3 hours | Granularity analysis |
| Prompt robustness test | 2 hours | Robustness results |
| Case study: 10 high-score correct CoTs | 2 hours | Qualitative analysis |
| Case study: 10 low-score incorrect CoTs | 2 hours | Error patterns |

**Deliverables**:

- Ablation study results
- Case study analysis
- Insights on failure patterns

---

### Week 7: Visualizations & Paper Drafting

**Objectives**:

- [ ] Create all paper figures
- [ ] Draft paper sections
- [ ] Prepare supplementary materials

**Tasks**:

| Task | Time | Output |
|------|------|--------|
| Create transition matrix heatmaps | 2 hours | Figure 1 |
| Create score distribution plots | 2 hours | Figure 2 |
| Create model comparison bar charts | 2 hours | Figure 3 |
| Create Viterbi path visualizations | 2 hours | Figure 4 |
| Draft Abstract & Introduction | 3 hours | 1.5 pages |
| Draft Methodology section | 4 hours | 2 pages |
| Draft Related Work section | 3 hours | 1 page |

**Deliverables**:

- All paper figures (PNG/PDF)
- Draft of Abstract, Intro, Methodology, Related Work

---

### Week 8: Paper Completion & Submission

**Objectives**:

- [ ] Complete paper draft
- [ ] Internal review and revisions
- [ ] Prepare code for release

**Tasks**:

| Task | Time | Output |
|------|------|--------|
| Draft Experiments & Results | 4 hours | 2 pages |
| Draft Discussion & Conclusion | 2 hours | 1 page |
| Internal review | 2 hours | Feedback |
| Revisions | 4 hours | Final draft |
| Clean and document code | 3 hours | GitHub release |
| Prepare supplementary materials | 2 hours | Appendix |
| Submit to target venue | 1 hour | Submission confirmation |

**Deliverables**:

- Complete paper (6-8 pages)
- Cleaned GitHub repository
- Supplementary materials

---

## 🎯 Target Venues

| Venue | Deadline | Type | Fit |
|-------|----------|------|-----|
| ACL SRW (Student Research Workshop) | Feb 2025 | Workshop | ⭐⭐⭐ Excellent |
| EMNLP Findings | May 2025 | Main track | ⭐⭐ Good |
| NeurIPS TinyPapers | Sep 2025 | Short paper | ⭐⭐⭐ Excellent |
| IEEE ICMLA | Jul 2025 | Conference | ⭐⭐ Good |
| arXiv | Anytime | Preprint | ⭐⭐⭐ Immediate |

---

## 📊 Success Metrics

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Correlation (score vs correctness) | ρ ≥ 0.30 | ρ ≥ 0.50 |
| AUC-ROC (classifying correct/incorrect) | ≥ 0.65 | ≥ 0.75 |
| Models evaluated | 3 | 5 |
| Datasets tested | 2 | 4 |
| Paper pages | 6 | 8 |
| GitHub stars (post-release) | 10 | 50 |

---

## 🛠️ Resource Requirements

### Compute

- **Minimum**: CPU-only (HMM training, scoring)
- **Recommended**: 1× GPU (NVIDIA T4 or better) for CoT generation
- **Alternative**: Google Colab Pro ($10/month)

### Storage

- Raw datasets: ~100 MB
- CoT outputs: ~50 MB
- Models: ~10 MB
- Total: ~200 MB

### Time Commitment

- Per week: 15-20 hours
- Total: 120-160 hours

---

## ⚠️ Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Weak correlation results | Medium | High | Refine tagging, add more data |
| Compute limitations | Low | Medium | Use smaller models, Colab |
| Tagging quality issues | Medium | Medium | Manual validation, iteration |
| Paper rejection | Medium | Medium | Target multiple venues |

---

## 📝 Checkpoints & Reviews

- **Week 2 Review**: Data collection complete?
- **Week 4 Review**: Initial results promising?
- **Week 6 Review**: Strong enough for publication?
- **Week 8 Review**: Paper submission ready?

---

## 📚 Key References

1. Wei et al. (2022). Chain-of-Thought Prompting. NeurIPS.
2. EIRL (ICLR 2026). Hidden Markov Reasoning Lens.
3. Wu et al. (EMNLP 2025). State-Aware Reasoning Transitions.
4. Rabiner (1989). A Tutorial on Hidden Markov Models. IEEE.
5. hmmlearn documentation: <https://hmmlearn.readthedocs.io/>

---

## ✅ Final Checklist

Before submission:

- [ ] All experiments reproducible
- [ ] Code cleaned and documented
- [ ] README updated
- [ ] Paper proofread
- [ ] Figures high quality (300 DPI)
- [ ] References complete
- [ ] Supplementary materials ready
- [ ] GitHub repository public
