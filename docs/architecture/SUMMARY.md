# Project Summary: CoT-HMM

## What is CoT-HMM?

**Unsupervised Plausibility Scoring of Chain-of-Thought Reasoning via Structural Hidden Markov Models**

A lightweight method that learns the structural patterns of correct reasoning chains and scores new CoTs based on how well they match those patterns.

---

## Core Architecture

### Pipeline Stages
1. **CoT Collection** → scripts/generation/*.py → LLM generates reasoning traces
2. **Step Tagging** → src/tagging/rule_based_tagger.py → Convert text to 8 semantic tags
3. **HMM Training** → src/hmm/trainer.py → Learn transition patterns from correct traces
4. **Scoring & Evaluation** → src/hmm/scorer.py → Compute likelihood, entropy, anomalies

---

## Key Methods & Formulas

### HMM Scoring
```
Score = (1/T) × log P(observation_sequence | λ)

Where:
- T = sequence length (steps)
- λ = (A, B, π) trained HMM parameters
- A = transition matrix (state-to-state)
- B = emission matrix (tag-to-state)
- π = initial state distribution
```

### Semantic Tags (8 Categories)
| Tag | Meaning | Examples |
|-----|---------|----------|
| PR | Problem Restatement | "need to find", "question asks" |
| DV | Define Variable | "let x =", "suppose" |
| EQ | Equation Setup | "equation is", mathematical formulas |
| CA | Calculation | "5 × 3 = 15", arithmetic |
| IN | Inference | "therefore", "thus", "implies" |
| RF | Reflection | "wait", "actually", "correction" |
| VE | Verification | "check", "verify", "confirm" |
| CO | Conclusion | "answer is", "final answer", "boxed" |

---

## ML Models Used

### Primary: Categorical HMM
- **Library:** `hmmlearn.hmm.CategoricalHMM`
- **Type:** Discrete Hidden Markov Model
- **Parameters:** N states (5-6), M observations (8 tags)
- **Training:** Baum-Welch EM on correct-only data

### Neural Extensions (Planned)
- **NeuralTagger:** DistilBERT-base-uncased → 8-way classifier
- **Backbone:** sentence-transformers/all-MiniLM-L6-v2 (ROSCOE)
- **Baseline LM:** distilgpt2 (PiCSAR)

---

## Baseline Methods (6 Heuristics)

| Method | Score Formula | Purpose |
|--------|--------------|---------|
| Length | log(char) × 0.1 + log(steps) × 0.5 | Longer = more thorough |
| Tag Diversity | (unique/total) × log(total) | Diverse steps = good reasoning |
| Structure | Binary features summed | Expected patterns present |
| Transition | Good transitions / total | Logical flow |
| Combined | 0.2L + 0.2D + 0.3S + 0.3T | Weighted ensemble |
| Random | uniform(0, 1) | Sanity check |

---

## Current Results

| Dataset | Model | Correlation | AUC | p-value |
|---------|-------|-------------|-----|---------|
| GSM8K | Qwen-0.5B | 0.21 | 0.62 | 0.008 |
| GSM8K | Qwen-1.5B | 0.25 | 0.65 | 0.032 |
| ARC (Science) | Qwen-0.5B | 0.04 | 0.46 | 0.62 |

**Note:** ARC AUC=0.46 is below chance → requires domain analysis

---

## Files Created

| File | Description |
|------|-------------|
| `docs/architecture/ARCHITECTURE_OVERVIEW.md` | Technical architecture details |
| `docs/architecture/METHODS_REFERENCE.md` | Complete methods catalog |
| `docs/architecture/VISUAL_ARCHITECTURE.html` | Interactive diagrams |
| `docs/architecture/ARCHITECTURE_DIAGRAMS.html` | SVG pipeline visualizations |

---

## 12-Week Research Plan

| Phase | Weeks | Focus | Deliverable |
|-------|-------|-------|-------------|
| 1 | 1-2 | Data Scaling | 500+ samples, 3+ models |
| 2 | 3-4 | Neural Baselines | ROSCOE, PiCSAR comparison |
| 3 | 5-6 | Statistical Rigor | Bootstrap CIs, effect sizes |
| 4 | 7-8 | Neural-HMM Hybrid | DistilBERT tagger, evaluate |
| 5 | 9-10 | Downstream Utility | PPO/rejection sampling |
| 6 | 11 | ARC Investigation | Root cause analysis |
| 7 | 12 | Submission Polish | IEEE TNNLS ready |