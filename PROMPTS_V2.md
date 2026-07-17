# PROMPTS_V2.md
### AI-Assistant Prompts for the Repositioning + Two-Track Plan

Use these with Claude/GPT/Copilot. Attach your current paper draft and relevant
code files when using each prompt.

---

## Phase 0 (Week 0): Positioning and Literature Honesty

```
I am repositioning my Chain-of-Thought verification paper (Ensemble HMM +
Structural Entropy) after discovering two closely related 2026 papers:

1. "Hidden Markov Modeling of Reasoning Dynamics in LLMs" (ICLR 2026,
   OpenReview id fr9t7r43am) - uses Hierarchical HMM + causal steering
   vectors to CONVERT failing reasoning into correct reasoning.

2. "Unsupervised Process Reward Models" (arXiv:2605.10158) - unsupervised
   PRM validated with RL training, test-time scaling, and ProcessBench
   benchmarks, showing real accuracy gains.

My current paper only shows CORRELATION (r=0.21, AUC=0.62) between my
Ensemble HMM score and correctness, with no causal or RL validation.

Help me:
1. Write a one-page comparison memo distinguishing my approach:
   classical ensemble HMM on RULE-BASED tags (not hidden states),
   CPU-only inference (~150ms), works with only 50-100 correct examples,
   fully interpretable transition matrices - versus their more complex,
   compute-intensive, larger-scale approaches
2. Draft a new "Comparison to Concurrent Work" subsection (300-400 words)
   for my Related Work section that honestly acknowledges both papers
   are more advanced in scope, while explaining my resource-constrained
   niche contribution
3. Rewrite my abstract's core claim to avoid overlapping too closely
   with their stronger claims, while staying honest about my actual results
```

---

## TRACK A

### Week 1: Honest Reframing

```
I need to rewrite my paper's Abstract and Introduction to reposition it as:
"A lightweight, CPU-only, unsupervised alternative for chain-of-thought
verification in settings where fine-tuning a neural verifier or running
RL training is infeasible" - rather than claiming to compete on raw
performance with neural Process Reward Models.

My current abstract is: [paste current abstract]

Help me:
1. Rewrite the abstract with this framing, keeping all true numbers
   (r=0.21, p<0.01, AUC=0.62) but changing the narrative around them
2. Create a resource-comparison table: my method (CPU, seconds, no
   labels, 50-100 examples) vs typical neural PRM/uPRM (GPU, hours,
   larger labeled/preference data)
3. Go through my full paper and flag every sentence that uses the word
   "verifier" or implies I'm replacing semantic verification, and
   suggest softer, more defensible phrasing ("pre-filter",
   "complementary signal", "efficient first-pass screen")
```

### Week 2: Related Work Rewrite

```
Using the positioning memo from Phase 0, help me write a polished
"Relation to Concurrent Work" subsection for my Related Work section
that:
1. Properly cites both the ICLR 2026 HHMM paper and the uPRM paper
   with full bibliographic details (I'll provide exact citation info)
2. Explains what they do that I don't (causal intervention, RL
   validation, larger scale)
3. Explains what makes my contribution still valuable (simplicity,
   interpretability, near-zero compute cost, minimal data requirements)
4. Keeps the tone respectful and precise, not defensive
```

### Week 3: Rejection Sampling Experiment

```
I have this rejection sampling scaffold: [paste rejection_sampling.py]

Help me:
1. Adapt it to run Best-of-5 and Best-of-8 selection experiments on
   200+ held-out GSM8K problems using my Qwen-0.5B and Qwen-1.5B
   generation pipeline
2. Compute and compare: HMM-selected accuracy, random-selected accuracy,
   oracle (best-possible) accuracy, and single-sample baseline accuracy
3. Write a new Results subsection titled "Downstream Utility: Rejection
   Sampling" reporting these numbers honestly, whether the gain is
   large or small
4. Suggest an appropriate visualization (bar chart) comparing the four
   accuracy numbers
```

### Week 4: Scale Data + Statistical Rigor

```
I need to scale my dataset from 150/74 samples to 300+ samples per model
and add statistical rigor.

Here are my utilities: [paste bootstrap_ci.py and effect_size.py]

Help me:
1. Write a data collection script to generate 300 GSM8K samples each
   for Qwen-0.5B and Qwen-1.5B (reusing my existing generation pipeline)
2. Compute bootstrap 95% CIs for all headline correlation/AUC numbers
3. Compute Cohen's d for the structural entropy correct-vs-incorrect
   comparison
4. Investigate the ARC below-chance AUC=0.46 result: compare tag
   frequency distributions between GSM8K and ARC traces, and check
   whether the below-chance result is due to small sample noise
   (95% CI width) or a genuine signal inversion
```

### Week 5: Full Rewrite

```
I need to merge all my recent changes into one coherent draft:
1. New positioning (Track A Week 1)
2. New Related Work concurrent-work section (Track A Week 2)
3. New rejection sampling results (Track A Week 3)
4. New confidence intervals, effect sizes, ARC investigation (Track A Week 4)

Here is my current full paper: [paste current main.tex or PDF text]
Here are the new sections/results to integrate: [paste from previous weeks]

Help me:
1. Produce a fully merged, internally consistent draft
2. Check every number mentioned in the Abstract matches the tables exactly
3. Verify terminology consistency (Ensemble HMM used everywhere, not
   "Ensembled HMM")
4. Flag any remaining overclaims given the new honest positioning
```

### Week 6: Venue Selection + Submission

```
I'm ready to submit my repositioned CoT verification paper.

Help me:
1. Compare KI - Kunstliche Intelligenz vs Big Data Mining and Analytics
   vs a relevant NeurIPS/ICLR 2026 workshop for fit, given my paper's
   "efficient, resource-constrained alternative" framing
2. Draft a cover letter for my chosen venue explaining the contribution
   and explicitly noting awareness of concurrent stronger work
3. Prepare a submission checklist: formatting requirements, page limits,
   supplementary material needs, double-blind requirements
```

---

## TRACK B (Only If Time/Compute Permits)

### Week 7-8: RL Validation Loop

```
I want to implement a minimal RL validation experiment using my
StructureReward class as an auxiliary reward signal, to add causal
validation my paper currently lacks.

Here is my StructureReward implementation: [paste rejection_sampling.py
StructureReward class]

Help me:
1. Set up a minimal GRPO or PPO training loop using the TRL library
   on Qwen-0.5B, training on 500-1000 GSM8K problems
2. Combine task reward (final answer correctness) with structural
   reward from StructureReward using a weighted combination
   (see minimal_ppo_reward_hook function)
3. Design the experiment to run within free-tier Colab/Kaggle GPU
   hour limits (T4/P100, ~12hr sessions)
4. Write code to log training curves and compute final accuracy
   delta: baseline RL (task reward only) vs RL + structural reward
5. Help me report results honestly regardless of outcome (positive,
   negative, or null)
```

### Week 9: Neural-HMM Hybrid

```
I want to close the gap toward my supervised Logistic Regression upper
bound (r=0.69) by replacing rule-based tagging with a neural emission
model.

Here is my skeleton: [paste neural_hmm_hybrid.py]

Help me:
1. Fine-tune DistilBERT on my expanded 150-200 step gold-standard labels
2. Validate neural tagger accuracy against gold standard (target:
   exceed rule-based kappa of 0.953)
3. Integrate soft tag probabilities as HMM emissions using the
   confidence-weighted forward algorithm
4. Benchmark: rule-based Ensemble HMM vs Neural-HMM hybrid vs
   Logistic Regression upper bound
5. Write the new Methodology and Results subsections
```

### Week 10-11: Real Baseline Benchmarking

```
I need to replace citation-only baselines with real empirical runs.

Here are my implementations: [paste roscoe_baseline.py and picsar_baseline.py]

Help me:
1. Run both on my full scaled test set (300+ samples per model)
2. Build a comprehensive comparison table: Ensemble HMM vs Neural-HMM
   vs ROSCOE vs PiCSAR vs Logistic Regression upper bound
3. Write a discussion paragraph explaining trade-offs: which method
   wins on raw performance vs interpretability vs compute cost
```

### Week 12: Scale to 3 Models + 2 Benchmarks

```
I need to scale my evaluation to match top-tier reviewer expectations.

Help me:
1. Add a third model family (Llama-3.2-1B or Mistral-7B) to my
   generation pipeline
2. Add MATH-500 or a ProcessBench subset as a second benchmark
   alongside GSM8K
3. Re-run all core experiments (main results, cross-model, cross-domain,
   ablations) at this new scale
4. Update all tables and figures with the expanded results
```

### Week 13-14: Ambitious Rewrite + Submission

```
I now have: RL validation results, Neural-HMM hybrid, real baseline
comparisons, and 3-model/2-benchmark scale.

Help me:
1. Rewrite my paper's Abstract and Introduction to claim BOTH efficiency
   AND validated downstream utility (citing the RL results)
2. Run a full statistical audit: bootstrap CIs and Cohen's d for every
   new result, with Benjamini-Hochberg FDR correction across the
   complete set of statistical tests in the paper
3. Identify the best-fit NeurIPS/ICLR/ICML 2026/2027 workshop or a
   stronger journal (Neural Computing and Applications, Artificial
   Intelligence Review) given this expanded scope
4. Draft a cover letter emphasizing the causal validation and scale
   as differentiators from the earlier draft
```
