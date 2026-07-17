# MASTER PLAN v2: Repositioning Strategy for Top-Tier-Adjacent Publication
### From "Correlational Pilot Study" to "Credible, Honestly-Positioned Contribution"

**Owner:** Gaurav Patil
**Context:** Competitive landscape scan revealed two directly competing 2026 papers
(ICLR 2026 Hierarchical HMM causal intervention paper, and "Unsupervised Process
Reward Models" with RL validation) that are significantly ahead of the current
draft in scope and rigor. This plan does NOT chase head-to-head top-tier
competition. It builds the strongest achievable version of the HONEST,
resource-constrained framing, while leaving the door open to a more ambitious
follow-up later.

---

## Two-Track Strategy (Choose Track Based on Time Available)

| Track | Time | Target | Goal |
|---|---|---|---|
| **Track A: Pragmatic** | 4-6 weeks | KI, Big Data Mining and Analytics, workshop tracks | Get a real, honest, citable publication |
| **Track B: Ambitious** | 10-14 weeks | NeurIPS/ICLR/ICML workshop, stronger journal | Add causal + RL validation to be genuinely competitive |

This document covers BOTH tracks. Track A is a subset of Track B's early phases.
You can start Track A and upgrade to Track B if time and results permit.

---

## Phase 0 (Week 0): Positioning and Literature Honesty — DO THIS FIRST

This phase is non-negotiable regardless of track. Skipping this is what gets
papers rejected with "insufficient novelty" or "unaware of related work."

### Tasks
- [ ] Read and annotate the ICLR 2026 Hierarchical HMM paper in full (OpenReview id: fr9t7r43am)
- [ ] Read and annotate "Unsupervised Process Reward Models" (arXiv:2605.10158)
- [ ] Write a one-page internal comparison memo: what they do, what you do differently,
      why your resource-constrained angle is still valuable
- [ ] Rewrite the paper's core positioning statement (see PROMPTS.md Week 0 prompt)
- [ ] Draft the new "Comparison to Concurrent Work" subsection for Related Work

### Deliverable
`docs/positioning_memo.md` + rewritten Abstract/Intro framing

---

## TRACK A: Pragmatic Path (Weeks 1-6)

### Week 1: Honest Reframing of Existing Results
**Goal:** Rewrite the paper's claims to match what the data actually supports, with the new "low-resource alternative" framing.

Tasks:
- Rewrite Abstract: frame as "lightweight, CPU-only, unsupervised alternative for settings where LLM-judge fine-tuning or RL training is infeasible"
- Add explicit resource comparison table: your method (CPU, seconds, no labels) vs PRM/uPRM (GPU, hours/days, larger data)
- Soften every "verifier" claim to "pre-filter" or "complementary signal" throughout
- Remove any language implying you are competing on raw performance with neural PRMs

Deliverable: Rewritten Abstract + Introduction (see PROMPTS.md Week 1)

---

### Week 2: Related Work Rewrite with Concurrent-Work Section
**Goal:** Add explicit, honest positioning against the two 2026 competing papers.

Tasks:
- Add new subsection: "Relation to Concurrent Work"
- Discuss HHMM paper: acknowledge it does causal intervention + hierarchical structure; explain your simpler ensemble approach targets a different resource regime
- Discuss uPRM paper: acknowledge its RL validation and benchmark scale; explain your contribution is orthogonal (classical probabilistic model vs. distillation-based neural approach)
- Cite both papers properly with full bibliographic details

Deliverable: New Related Work subsection (~300-400 words)

---

### Week 3: Rejection Sampling Experiment (Minimum Viable Downstream Utility)
**Goal:** Get ONE real downstream-utility number — this is the single highest-leverage addition for Track A.

Tasks:
- Use `rejection_sampling.py` (already built) to run Best-of-5 or Best-of-8 selection
  on 200+ held-out GSM8K problems using Qwen-0.5B and Qwen-1.5B
- Compare: HMM-selected accuracy vs random-selected vs oracle vs single-sample baseline
- Report honestly, even if the gain is small (2-3%) — small honest gains beat overclaiming
- Write new "Downstream Utility: Rejection Sampling" Results subsection

Deliverable: New results table + subsection with real accuracy deltas

---

### Week 4: Scale Data Modestly + Add Confidence Intervals
**Goal:** Strengthen statistical credibility without requiring a huge compute budget.

Tasks:
- Scale from 150/74 samples to at least 300 samples per model (achievable on CPU/free-tier GPU)
- Add bootstrap 95% CIs to all headline numbers using `bootstrap_ci.py` (already built)
- Add Cohen's d effect sizes using `effect_size.py` (already built)
- Investigate the ARC below-chance result with the tag-distribution analysis from the 12-week plan

Deliverable: Updated Tables II & III with CIs, effect sizes, and ARC investigation paragraph

---

### Week 5: Full Paper Rewrite Integrating All Changes
**Goal:** Produce a coherent, fully consistent v4 draft.

Tasks:
- Merge: new positioning, concurrent work section, rejection sampling results, CIs, effect sizes
- Full read-through for internal consistency (numbers matching across abstract/tables/text)
- LaTeX compile check: no broken refs, consistent terminology (Ensemble HMM everywhere)
- Get 1-2 peer reviews from AUTOMACHINA/Aelura colleagues if possible

Deliverable: `main_v4.tex` — full clean draft

---

### Week 6: Venue Selection + Submission
**Goal:** Submit to realistic venue.

Tasks:
- Finalize venue choice: KI – Künstliche Intelligenz (fast, ~56% acceptance) OR
  Big Data Mining and Analytics (no APC) OR a NeurIPS/ICLR 2026 workshop
  (check current workshop CFPs for "efficient reasoning," "interpretability," or "verification")
- Prepare cover letter explicitly stating the resource-constrained contribution angle
- Submit; also post to arXiv as preprint in parallel (with endorsement now hopefully resolved)

Deliverable: Submitted paper

---

## TRACK B: Ambitious Path (Weeks 7-14, continues after Track A)

Only pursue this if Track A results are promising and you have more time/compute.

### Week 7-8: Minimal RL Validation Loop
**Goal:** Add the causal/interventional validation that both competing papers have and you don't.

Tasks:
- Set up TRL-based GRPO/PPO loop on Qwen-0.5B using `StructureReward` as auxiliary reward
- Train on a subset of GSM8K (500-1000 problems) for a small number of steps
  (compute-budget permitting — use free-tier Colab/Kaggle GPU hours)
- Compare: baseline RL (task reward only) vs RL + structural reward
- Report training curves and final accuracy delta, honestly, even if negative/null

Deliverable: New "Reinforcement Learning Validation" section with real experiment

---

### Week 9: Neural-HMM Hybrid Implementation
**Goal:** Close the gap toward the supervised upper bound (r=0.69) using neural emissions.

Tasks:
- Fine-tune DistilBERT on expanded 150-200 step gold standard using `neural_hmm_hybrid.py`
- Integrate as soft emissions into HMM via confidence-weighted forward algorithm
- Benchmark against rule-based Ensemble HMM and Logistic Regression upper bound

Deliverable: New results row + subsection: "Neural-HMM Hybrid: Closing the Gap"

---

### Week 10-11: Real Baseline Benchmarking (ROSCOE + PiCSAR)
**Goal:** Replace citations with actual empirical comparisons.

Tasks:
- Run `roscoe_baseline.py` and `picsar_baseline.py` (already built) on your full test set
- Add 3-way comparison table: Ensemble HMM vs ROSCOE vs PiCSAR vs Neural-HMM
- Discuss where your method wins/loses and why (interpretability vs raw accuracy trade-off)

Deliverable: Full head-to-head comparison table and discussion

---

### Week 12: Scale to 3 Model Families + 2 Benchmarks
**Goal:** Match the scale expectations of top-tier reviewers.

Tasks:
- Add a third model family (Llama-3.2-1B or Mistral-7B)
- Add MATH-500 or a subset of ProcessBench as a second benchmark alongside GSM8K
- Re-run all core experiments (main results, cross-model, ablations) at this new scale

Deliverable: Fully scaled dataset + updated all tables

---

### Week 13: Full Draft Rewrite for Ambitious Venue
**Goal:** Produce a genuinely stronger, more complete paper.

Tasks:
- Rewrite framing: now you have RL validation, neural hybrid, real baselines, and scale
- Reposition Abstract to claim both efficiency AND validated downstream utility
- Full statistical audit pass (CIs, effect sizes, FDR correction across ALL new tests)

Deliverable: `main_v5.tex` — ambitious-track draft

---

### Week 14: Target Stronger Venue
**Goal:** Submit to a more ambitious venue than Track A.

Tasks:
- Target: NeurIPS/ICLR/ICML 2026/2027 workshop on efficient reasoning, interpretability,
  or LLM verification (check current CFPs closer to submission time)
- Alternative: a stronger journal (Neural Computing and Applications, Artificial
  Intelligence Review) given the added rigor
- Prepare cover letter explicitly citing your RL validation and scale as differentiators

Deliverable: Submitted paper to ambitious-track venue

---

## Risk Notes (Read Before Starting)

- **Do not skip Phase 0.** Submitting without acknowledging the two concurrent 2026
  papers is the single fastest way to get rejected for "unaware of related work."
- **Track B's RL experiment is the highest-risk, highest-reward item.** If compute
  is genuinely unavailable, Track A alone with honest framing is still publishable
  and valuable — don't let Track B block Track A's submission.
- **Never claim causal validation you haven't run.** If the RL experiment shows
  null/negative results, report it honestly — this is still more credible than
  overclaiming.
- **Re-check target venue/workshop CFP deadlines** before committing weeks to Track B;
  workshop deadlines are often tight and seasonal.

---

## Files in This Package

- `00_MASTER_PLAN_V2.md` — this file
- `PROMPTS_V2.md` — AI prompts for every phase/week above
- `code/positioning_diff_analysis.py` — helper to structure the concurrent-work comparison
- `code/rejection_sampling.py` — (reused from previous package) Best-of-N experiment
- `code/bootstrap_ci.py` — (reused) confidence intervals
- `code/effect_size.py` — (reused) Cohen's d + FDR tracking
- `code/neural_hmm_hybrid.py` — (reused) Neural-HMM hybrid skeleton
- `code/roscoe_baseline.py` / `code/picsar_baseline.py` — (reused) real baseline implementations
- `code/rl_validation_scaffold.py` — new: minimal GRPO/PPO scaffold using TRL
