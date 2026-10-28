# CoT-HMM Paper Submission Package — Readme & Instructions

Welcome! This folder contains everything required to review, verify, and submit the **CoT-HMM** research paper.

---

## 📁 Package Directory Structure

```
FINAL_PAPER_SUBMISSION_PACKAGE/
│
├── README_START_HERE.md                 <-- YOU ARE HERE
├── SUBMISSION_METADATA.txt              <-- Pure copy-paste text for conference portals
│
├── 01_Springer_ANTIC_Submission/        <-- Springer CCIS / LNCS Format (9 Pages)
│   ├── CoT_HMM_Springer_ANTIC_9Pages.pdf <-- Compiled PDF ready for upload
│   ├── main.tex                         <-- Primary LaTeX manuscript source
│   ├── llncs.cls                        <-- Official Springer class file
│   ├── references.bib                   <-- BibTeX citations
│   ├── figures/                         <-- High-resolution vector & PNG plots
│   └── springer_antic_source.zip        <-- Standalone source zip for upload
│
├── 02_IEEE_Conference_Submission/       <-- IEEE 2-Column Conference Format (5 Pages)
│   ├── CoT_HMM_IEEE_Conference_5Pages.pdf<-- Compiled PDF ready for upload
│   ├── main.tex                         <-- Primary IEEE LaTeX manuscript source
│   ├── references.bib                   <-- BibTeX citations
│   ├── figures/                         <-- High-resolution vector & PNG plots
│   └── ieee_conference_source.zip       <-- Standalone source zip for upload
│
└── 03_Project_Summary_And_Reviewer_Notes/
    └── PROJECT_OVERVIEW.md              <-- Full methodology & empirical results overview
```

---

## 🎯 Which Paper Should Be Submitted Where?

### Option 1: Springer Conferences (e.g., ANTIC-2026, CCIS Series)
* **Folder to use:** `01_Springer_ANTIC_Submission/`
* **File to upload:** `CoT_HMM_Springer_ANTIC_9Pages.pdf`
* **Source zip (if required by portal):** `springer_antic_source.zip`
* **Portal Link:** [Springer Meteor ANTIC Portal](https://meteor.springer.com/ANTIC2026) | [Official Site](https://antic.co.in)
* **Format:** Single-column Springer LNCS, 9 pages, page numbers centered at bottom.

### Option 2: IEEE Conferences (e.g., ICACECT, TENCON, INDICON)
* **Folder to use:** `02_IEEE_Conference_Submission/`
* **File to upload:** `CoT_HMM_IEEE_Conference_5Pages.pdf`
* **Source zip (if required by portal):** `ieee_conference_source.zip`
* **Portal Link:** Microsoft CMT (`https://cmt3.research.microsoft.com/`) or EasyChair
* **Format:** Standard IEEE Transactions 2-column format, 5 pages.

---

## ⚡ Quick Step-by-Step Submission Guide

1. **Open the Conference Portal** (Meteor, CMT, or EasyChair).
2. **Copy Details from `SUBMISSION_METADATA.txt`**:
   - Title: `CoT-HMM: Structural Plausibility Scoring of Chain-of-Thought Reasoning via Hidden Markov Models`
   - Abstract (copy from `SUBMISSION_METADATA.txt`)
   - Keywords (copy from `SUBMISSION_METADATA.txt`)
   - Add all 5 Authors in order (Gaurav Patil, Suhail Shaikh, Prof. Bhagyalakshmi V, Parth Parmar, Chirayu Marathe).
3. **Attach PDF**:
   - For Springer: Attach `01_Springer_ANTIC_Submission/CoT_HMM_Springer_ANTIC_9Pages.pdf`
   - For IEEE: Attach `02_IEEE_Conference_Submission/CoT_HMM_IEEE_Conference_5Pages.pdf`
4. **Submit & Download Confirmation Receipt**.

---

## 🔬 Brief Paper Summary

* **What it does:** Verifies step-by-step reasoning (Chain-of-Thought) of LLMs using an ensemble of Hidden Markov Models ($M=12$) on discrete semantic tags (`PR`, `IN`, `CA`, `EQ`, `CO`, `RF`, `VE`, `DV`).
* **Why it matters:** Unlike heavy neural process reward models (PRMs) that require GPUs and thousands of labeled steps, CoT-HMM runs in **~150ms on CPU** with **zero human step-level negative annotations**.
* **Key Numbers:**
  - Premature conclusion detection: **AUC = 0.990**
  - Circular reasoning loops: **AUC = 0.761**
  - Benchmark vs baselines: **ROC-AUC = 0.982**
  - Inference compute savings: **42%–47%** reduction via 3-Tier Compute Routing Gate.
