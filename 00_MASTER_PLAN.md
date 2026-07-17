# 🗺️ COT-HMM: 12-Week Journal Submission Action Plan
> **Target Venues (Priority Order):** IEEE TNNLS → IEEE TAI → KDD 2027 Workshop
> **Start Date:** Week of 2026-07-14  |  **Submission Deadline:** ~2026-10-14
> **Core Goal:** Move from "acceptable at KI" → "competitive at Q1 IEEE journal"

---

## 📊 Current State Snapshot

| Metric | Current | Target (Q1 Journal) |
|--------|---------|---------------------|
| Test set size | 150 (GSM8K) | 500–1000+ |
| Model families | 2 (Qwen-0.5B, 1.5B) | 3+ (Qwen + Llama + Mistral) |
| Neural baselines compared | 0 | ROSCOE + PiCSAR |
| AUC (main method) | 0.62 | 0.68+ or reframed |
| Downstream utility demo | ❌ | ✅ Rejection-sampling experiment |
| Bootstrap CIs reported | ❌ | ✅ All headline numbers |
| Below-chance ARC explained | ❌ | ✅ Root-cause analysis |
| Neural-HMM hybrid | Future work only | ✅ Small-scale implemented |

---

## 📅 Phase-by-Phase Roadmap

### PHASE 1 — Data Scaling (Weeks 1–2)
**Reviewer Blocker:** N=150 is thin; reviewers expect 500–1000+ for journal claims.

#### Week 1: Schema + Pipeline + First 500 Samples
**File:** `plan/week01_data_scaling/WEEK01_PLAN.md`

- [ ] Define standardized `ScoredTrace` JSON schema (`src/experiments_v2/data_schema.py`)
- [ ] Extend data collection pipeline to GSM8K full test set (1,319 problems)
- [ ] Add Llama-3.2-1B as third model family (via `ollama` or HuggingFace)
- [ ] Target: 500 labeled traces across 3 model families
- [ ] Commit intermediate dataset checkpoint to `data/v2/gsm8k_500_traces.json`

**Deliverable:** `data/v2/gsm8k_500_traces.json` (500 samples, 3 models)

#### Week 2: Scale to 1000+ and Add Science Domain
**File:** `plan/week02_data_scaling_cont/WEEK02_PLAN.md`

- [ ] Run Mistral-7B (or Phi-3-mini) on GSM8K subset for 4th model family
- [ ] Add ARC-Challenge traces (300 samples) for domain-transfer study
- [ ] Re-run HMM training on new v2 dataset, report updated metrics
- [ ] Target: 1000+ total traces, 3 domains (Math / Science / Mixed)

**Deliverable:** `data/v2/full_1000_traces.json`, updated `results_v2.txt`

---

### PHASE 2 — Neural Baseline Implementation (Weeks 3–4)
**Reviewer Blocker:** "Related work inflation without empirical grounding" — cite ROSCOE/PiCSAR but never benchmark against them.

#### Week 3: ROSCOE Baseline
**File:** `plan/week03_roscoe_baseline/WEEK03_PLAN.md`
**Code:** `src/experiments_v2/roscoe_baseline.py`

- [ ] Implement ROSCOE scorer (semantic alignment + step consistency + informativeness)
- [ ] Use `sentence-transformers/all-MiniLM-L6-v2` for embedding backbone
- [ ] Run ROSCOE on the 500-sample v2 dataset
- [ ] Add ROSCOE column to comparison table

**Deliverable:** `results/v2/roscoe_scores.json`, comparison table draft

#### Week 4: PiCSAR Baseline
**File:** `plan/week04_picsar_baseline/WEEK04_PLAN.md`
**Code:** `src/experiments_v2/picsar_baseline.py`

- [ ] Implement PiCSAR joint log-likelihood scorer using HuggingFace logprobs
- [ ] Use `gpt2` or `distilgpt2` as the reference LM (no API key needed)
- [ ] Run PiCSAR on same 500-sample subset
- [ ] Build `Table_Comparison.md` — formal 5-method comparison table

**Deliverable:** `results/v2/picsar_scores.json`, `Table_Comparison.md`

---

### PHASE 3 — Statistical Rigor (Weeks 5–6)
**Reviewer Blocker:** Point estimates without CIs; FDR correction count not reported; no effect sizes.

#### Week 5: Bootstrap Confidence Intervals
**File:** `plan/week05_bootstrap_ci/WEEK05_PLAN.md`
**Code:** `src/experiments_v2/bootstrap_ci.py`

- [ ] Bootstrap CIs (n=10,000 resamples) around r=0.21 correlation and AUC=0.62
- [ ] Report as: `r=0.21 [95% CI: 0.09, 0.32]`, `AUC=0.62 [95% CI: 0.54, 0.70]`
- [ ] Run M-ablation (M=4, 8, 12, 16 ensemble sizes) with CIs on each

**Deliverable:** `results/v2/bootstrap_cis.json`, updated Table I with CI columns

#### Week 6: Effect Sizes + FDR Accounting
**File:** `plan/week06_effect_sizes/WEEK06_PLAN.md`
**Code:** `src/experiments_v2/effect_size.py`

- [ ] Add Cohen's d for structural entropy comparison (0.14 vs 0.23)
- [ ] Explicitly track and report total number of comparisons in BH-FDR correction
- [ ] Add Cliff's delta as non-parametric effect size for AUC differences
- [ ] Update paper Section IV statistics accordingly

**Deliverable:** `results/v2/effect_sizes.json`, updated paper stats section

---

### PHASE 4 — Neural-HMM Hybrid (Weeks 7–8)
**Reviewer Opportunity:** Converting "future work" into a real evaluated contribution.

#### Week 7: Neural Tagger Implementation
**File:** `plan/week07_neural_hmm/WEEK07_PLAN.md`
**Code:** `src/neural_hmm/neural_hmm_hybrid.py`

- [ ] Fine-tune `distilbert-base-uncased` (or `bert-tiny`) on existing tagged sequences as classification head
- [ ] Replace rule-based `TAG_TO_ID` emission with confidence-weighted neural emissions
- [ ] Implement `NeuralHMMScorer` inheriting from `HMMScorer` interface

**Deliverable:** `models/neural_hmm_v1.pkl`, `results/v2/neural_hmm_scores.json`

#### Week 8: Neural-HMM Evaluation + Comparison
**File:** `plan/week08_neural_hmm_eval/WEEK08_PLAN.md`

- [ ] Run NeuralHMM on full v2 dataset
- [ ] Compare: Rule-HMM vs Neural-HMM vs ROSCOE vs PiCSAR
- [ ] Report AUC improvement (expect: 0.62 → 0.66–0.72)
- [ ] Write "Method Extension" subsection for paper

**Deliverable:** Complete 5-method comparison table, paper section draft

---

### PHASE 5 — Downstream Utility Demo (Weeks 9–10)
**Reviewer Blocker:** No proof the score *does anything useful*. Papers accepted at 2026 venues show downstream utility.

#### Week 9: Rejection Sampling with StructureReward
**File:** `plan/week09_rejection_sampling/WEEK09_PLAN.md`
**Code:** `src/rejection_sampling/rejection_sampling.py`

- [ ] Wire existing `StructureReward` class into a proper rejection-sampling loop
- [ ] For each problem: generate K=8 candidates, keep highest-scoring trace, evaluate final accuracy
- [ ] Baseline: greedy decoding; Ours: structure-filtered top-K
- [ ] Report: pass@1 accuracy improvement (target: +2–5% absolute)

**Deliverable:** `results/v2/rejection_sampling_results.json`, new Table III

#### Week 10: PPO Reward Shaping Hook (Small-Scale)
**File:** `plan/week10_rl_validation/WEEK10_PLAN.md`

- [ ] Implement minimal PPO reward shaping using `StructureReward` as auxiliary signal
- [ ] Run on 100-problem GSM8K subset with Qwen-0.5B (CPU-feasible)
- [ ] Even null/negative results are publishable as honest negative finding
- [ ] Report: accuracy change + training stability with vs without structural reward

**Deliverable:** `results/v2/ppo_reward_results.json`, paper section on RL validation

---

### PHASE 6 — ARC Failure Investigation (Weeks 11)
**Reviewer Flag:** Math→Science AUC=0.46 (below chance) is unexplained — reviewers will demand a root-cause analysis.

#### Week 11: Below-Chance Transfer Root Cause
**File:** `plan/week11_arc_investigation/WEEK11_PLAN.md`

- [ ] Analyze tag-sequence distributions in Math vs Science CoTs (KL divergence)
- [ ] Visualize HMM state occupation in Math vs ARC domains (heatmaps)
- [ ] Hypothesize: Is it tag vocabulary mismatch? Or domain-specific transition structure?
- [ ] Run domain-adapted HMM (trained on small Science set) and compare to Math-trained
- [ ] Write "Domain Specificity Analysis" subsection with causal explanation

**Deliverable:** `results/v2/arc_domain_analysis.json`, visualization figures, paper subsection

---

### PHASE 7 — Final Polish + Submission (Week 12)
**File:** `plan/week12_submission_polish/WEEK12_PLAN.md`

- [ ] Reframe core claim: "lightweight unsupervised pre-filter" not "verifier"
- [ ] Strip author metadata from PDF (use `exiftool -all= main.pdf`)
- [ ] Final blind submission check (no self-citations that reveal identity)
- [ ] Proofread abstract + conclusion against final numbers
- [ ] Submit to IEEE TNNLS (primary) or IEEE TAI (secondary)

**Deliverable:** Submission-ready `main.pdf` + anonymized supplementary code zip

---

## 🔥 Priority Stack (If Time-Constrained)

If you only have 4 weeks, do these in order:
1. **Week 3–4:** ROSCOE + PiCSAR baselines (single biggest credibility gap)
2. **Week 9:** Rejection-sampling downstream utility (converts analysis→method)
3. **Week 5:** Bootstrap CIs on existing results (1 day of work, big reviewer signal)
4. **Week 11:** ARC below-chance explanation (turns a weakness into a finding)

---

## 📁 Deliverable File Index

| File | Phase | Week |
|------|-------|------|
| `src/experiments_v2/data_schema.py` | 1 | 1 |
| `src/experiments_v2/roscoe_baseline.py` | 2 | 3 |
| `src/experiments_v2/picsar_baseline.py` | 2 | 4 |
| `src/experiments_v2/bootstrap_ci.py` | 3 | 5 |
| `src/experiments_v2/effect_size.py` | 3 | 6 |
| `src/neural_hmm/neural_hmm_hybrid.py` | 4 | 7–8 |
| `src/rejection_sampling/rejection_sampling.py` | 5 | 9–10 |
| `scripts/analysis/arc_domain_analysis.py` | 6 | 11 |
| `PROMPTS.md` | All | All |

---

*Generated: 2026-07-16 | Reviewer feedback: Claude Sonnet 5 Thinking*
