# CoT-HMM: Structural Verification of Chain-of-Thought Reasoning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE Paper](https://img.shields.io/badge/Paper-IEEE--LaTeX-blueviolet.svg)](paper/main.tex)
[![Status: Accepted](https://img.shields.io/badge/Status-Accepted%20at%20IEEE%20IDCIoT%202027-success.svg)](camera_ready/IDCIoT_154_CameraReady.pdf)
[![Tests](https://img.shields.io/badge/Unit%20Tests-30%20Passing-brightgreen.svg)](tests/)

> 🎉 **Accepted at IEEE IDCIoT 2027**: *5th International Conference on Intelligent Data Communication Technologies and Internet of Things (IDCIoT 2027), IEEE Conference Record #71706*.  
> **Camera-Ready Submission Package**: [`camera_ready/`](camera_ready/) | [Submission Instructions](camera_ready/SUBMISSION_INSTRUCTIONS.md)

An unsupervised, lightweight framework for scoring the plausibility of Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) using **Ensemble Hidden Markov Models**.

---

## 🎯 Overview

CoT-HMM abstracts reasoning chains into sequences of 8 semantic tags (`PR`, `DV`, `EQ`, `CA`, `IN`, `RF`, `VE`, `CO`) and models their latent transition dynamics. It requires **no human step annotation**, **no GPU retraining**, and operates in **~150ms on CPU**.

```
Input CoT Text ──► Semantic Tagger ──► Tag Sequence ──► Ensemble HMM ──► Plausibility Margin (AUC 0.983)
```

---

## 📊 Headline Benchmark Results (Out-of-Fold 5-Fold Cross-Validation)

All metrics were computed via **Leak-Free Stratified 5-Fold Cross-Validation** ($N=500$) with **10,000 percentile bootstrap resamples**. Zero training data was leaked into test scoring.

| Method / Metric | Pearson r [95% CI] | ROC-AUC [95% CI] | p-value | Effect Size |
|---|---|---|---|---|
| **Plausibility Margin** 🏆 | **0.476** [0.375, 0.716] | **0.983** [0.966, 0.997] | $< 0.001$ | Cohen's $d = 1.093$ (*large*) |
| **Ensemble HMM Score** ($M=12$) | **0.350** [0.193, 0.534] | **0.950** [0.928, 0.969] | $< 0.001$ | Cohen's $d = 0.747$ (*medium*) |
| **Entropy Drift Rate (EDR)** | -0.282 [-0.376, -0.190] | 0.273 [0.227, 0.322] | $< 0.001$ | Cohen's $d = -0.586$ (*medium*) |
| **Structural Entropy** | -0.732 [-0.763, -0.698] | 0.057 [0.040, 0.077] | $< 0.001$ | Cohen's $d = -2.144$ (*large*) |
| **Heuristic Baseline** | 0.459 [0.372, 0.555] | 0.860 [0.826, 0.891] | $< 0.001$ | Cohen's $d = 1.031$ (*large*) |
| **ROSCOE (Neural Verifier)** | — | 0.380 | — | — |
| **PiCSAR (Content Confidence)** | — | 0.460 | — | — |

---

## ⚡ 3-Tier Adaptive Compute Routing Gate

- **High-Confidence Pass Tier (Top 30%)**: 150 queries passed directly with **99.3% accuracy**.
- **Re-sampling Tier (Middle 35%)**: 175 queries escalated for self-consistency. **Accuracy = 50.9%**.
- **Low-Confidence Reject Tier (Bottom 35%)**: 175 queries rejected immediately. **Accuracy = 6.9%** (93.1% actual errors).
- **Compute Savings**: **30.0% reduction** in LLM inference token overhead.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/GauravPatil2515/Chain-Of-Taughts-HMM.git
cd Chain-Of-Taughts-HMM
pip install -r requirements.txt
```

### 2. Run Unit Tests (30 Tests)

```bash
pytest tests/
```

### 3. Run Leak-Free 5-Fold Cross-Validation Benchmark

```bash
PYTHONPATH=. python3 scripts/run_leakfree_evaluation.py
```

Generated report files:
- Scored Dataset: `data/v2/gsm8k_500_proper_cv_scored.json`
- Metrics JSON: `experiments/results/evaluation_results_leak_free.json`
- Summary Markdown: `experiments/results/leak_free_report.md`

---

## 📄 Camera-Ready Deliverables (IDCIoT-154)

- **Camera-Ready Manuscript (PDF):** [`camera_ready/IDCIoT_154_CameraReady.pdf`](camera_ready/IDCIoT_154_CameraReady.pdf)
- **Response to Reviewers (PDF):** [`camera_ready/IDCIoT_154_Response_to_Reviewers.pdf`](camera_ready/IDCIoT_154_Response_to_Reviewers.pdf)
- **Response to Reviewers (Markdown):** [`camera_ready/RESPONSE_TO_REVIEWERS.md`](camera_ready/RESPONSE_TO_REVIEWERS.md)
- **LaTeX Source Package (ZIP):** [`camera_ready/IDCIoT_154_Source.zip`](camera_ready/IDCIoT_154_Source.zip)
- **Master Submission Package (ZIP):** [`camera_ready/IDCIoT_154_CameraReady_Package.zip`](camera_ready/IDCIoT_154_CameraReady_Package.zip)
- **Submission Instructions & Checklist:** [`camera_ready/SUBMISSION_INSTRUCTIONS.md`](camera_ready/SUBMISSION_INSTRUCTIONS.md)
