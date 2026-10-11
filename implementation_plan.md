# CoT-HMM: Validated Audit + 5-Week Action Plan for Conference Submission

> **Audit Date:** Aug 11, 2026 | **Target:** ICACECT 2026/2027 (deadline ~Sept 15)
> **Status:** CRITICAL data integrity issues confirmed — paper is NOT submission-ready as-is.

---

## Part 1: Validation of the Review (What's True, What's Exaggerated)

### CONFIRMED CRITICAL: The "Organic" Data Is Also Synthetic

The review said `gsm8k_500_proper.json` is synthetic. **This is true, and worse — the supposedly "organic" datasets are also fabricated.**

Direct audit findings from the actual `data/v2/` files:

| Dataset File | N | Unique cot_text Values | Verdict |
|---|---|---|---|
| `gsm8k_500_proper.json` | 500 | Documented as synthetic in DATA_CONSTRUCTION.md | SYNTHETIC |
| `gsm8k_500_traces.json` | 500 | **1 unique** ("Calculate. Therefore answer.") | SYNTHETIC |
| `gsm8k_organic_balanced.json` | 200 | **1 unique** ("Calculate. Therefore answer.") | SYNTHETIC |
| `qwen-0.5b_gsm8k.json` | 100 | **5 unique** — all 20 incorrect = same trace | SEMI-SYNTHETIC |
| `llama-3.2-1b_gsm8k.json` | 100 | **5 unique** | SEMI-SYNTHETIC |
| `mistral-7b_gsm8k.json` | 100 | **5 unique** | SEMI-SYNTHETIC |
| `phi-3-mini_gsm8k.json` | 100 | **5 unique** | SEMI-SYNTHETIC |

**What this means for your results:**
- `leak_free_report.md` got AUC = 1.000, r = 1.000 because every incorrect sample is the same 3-token sequence ['PR','DV','CO'] with text "Calculate. Therefore answer.". The HMM trivially memorizes one pattern.
- Per-model files have 20 identical incorrect traces and 80 identical correct traces — only 5 unique texts total.
- **None of your headline numbers come from real LLM outputs.**

> [!CAUTION]
> Every benchmark number in the paper — AUC 0.983, r=0.476, the 3-tier routing, cross-architecture transfer — is from fabricated data. A reviewer will catch this immediately.

---

### CONFIRMED: Numbers Are Inconsistent Across Documents

| Location | Claims |
|---|---|
| `README.md` headline table | Plausibility Margin AUC = **0.983**, r = **0.476** |
| `paper/main.tex` abstract | r = **0.76**, AUC = **0.98**, cross-arch r >= **0.94**, EDR r = **-1.00** |
| `leak_free_report.md` actual run | AUC = **1.000**, r = **1.000** (synthetic data artifact) |
| `ACTION_PLAN.md` | r = 0.25 for Qwen-1.5B, AUC 0.950 for Ensemble HMM |

Three different headline AUC numbers (0.950, 0.983, 1.000) for the same metric. No single source of truth exists.

---

### PARTIALLY CONFIRMED: EDR r = -1.00

On synthetic data (1 unique incorrect pattern) this is mathematically guaranteed — not a real finding. Whether it's meaningful on real LLM outputs is unknown. Must be treated as unverified.

---

### PARTIALLY CONFIRMED: Scope vs. Time Tension

`00_MASTER_PLAN_V2.md` describes: RL validation loop (GRPO/PPO), Neural-HMM hybrid (DistilBERT), SCoT-Bench benchmark, 5–6 research questions. This is 3–6 months of work. For a Sept 15 deadline, only 1–2 are achievable. The review's scope-cutting recommendation is correct.

---

### CONFIRMED VALID: The Core Method Is Genuinely Underexplored

Rule-based discrete tagging of CoT steps into 8 semantic roles + Ensemble Categorical HMM scored on correct-only traces is **not done by ProcessBench, PRMBench, or CLoT**. The method idea is novel. The implementation and evaluation are the problems, not the concept.

---

### CONFIRMED VALID: Must Cite Lanham et al. 2023

The faithfulness critique (scoring written trace != scoring actual computation) is legitimate and needs an explicit limitations statement with citation.

---

## Part 2: What Actually Exists vs. What's Needed

### What You Have (Real Assets)

| Asset | Quality | Usable? |
|---|---|---|
| `src/tagging/rule_based_tagger.py` | Real, working code | Yes |
| `src/hmm/ensemble.py` + `trainer.py` | Real, working code | Yes |
| `scripts/run_leakfree_evaluation.py` | Real, working pipeline | Yes — just needs real data |
| `paper/main.tex` | Well-structured IEEE LaTeX | Yes — needs rewriting |
| `data/raw/gsm8k.json` | Real GSM8K benchmark questions | Yes — need to generate traces |
| `colab_scripts/` | GPU generation infrastructure | Yes — this is your way out |
| Unit test suite (30 tests) | Passing | Yes |

### What You Don't Have (Critical Gaps)

| Gap | Why Critical |
|---|---|
| **Real LLM-generated CoT traces** | Every evaluation metric is meaningless without this |
| **Problem-level train/test split** | Current split is trace-level which can leak |
| **Verified correctness labels** | Need final-answer match against GSM8K gold |
| **Single authoritative results table** | Multiple conflicting numbers across all docs |
| **Tagger validation beyond 50 steps** | k=0.953 on N=50 has wide CIs |

---

## Part 3: 5-Week Action Plan (Aug 11 – Sept 15, 2026)

> [!IMPORTANT]
> **Do not write a single new word of the paper until Week 1 is complete.** The data must be fixed before anything else matters.

---

### Week 1 (Aug 11–18): Generate Real Data — Priority Zero

**Goal:** Replace all synthetic data with real LLM-generated CoT traces.

#### Step 1.1 — Generate Real GSM8K CoT Traces (Free GPU)

Use Kaggle/Colab (free T4/P100) with the existing `colab_scripts/` infrastructure.

**Target:** 200–300 real problems from GSM8K with real CoT outputs from at least 2 models:
- **Model A (primary):** Qwen-0.5B or Qwen-1.5B (already in `colab_scripts/`)
- **Model B (transfer test):** Llama-3.2-1B or Phi-3-mini

**Script:** `colab_scripts/colab_experiments.py` with `--generate --model qwen-0.5b`

**Correctness labels:** Extract final numeric answer from model output and compare to `data/raw/gsm8k.json` gold answers using regex. Do NOT trust template labels.

**Minimum viable:** 150 correct + 150 incorrect = 300 traces, balanced, from real model.

#### Step 1.2 — Enforce Problem-Level Split

When creating train/test folds, split by **GSM8K problem ID**, not by trace. The same problem must not appear in both train and test, even across models.

#### Step 1.3 — Audit EDR on Real Data

Run `scripts/run_leakfree_evaluation.py` on the new real dataset.
- If EDR r = -1.00 again: it's a bug — investigate and fix.
- If EDR returns to ~-0.28 as in the docs: real finding, keep it.

#### Step 1.4 — Create Single Source-of-Truth Table

Name it `experiments/results/CANONICAL_RESULTS.json`. Every paper number must point back to exactly this file. Archive all other results files.

**Deliverables:**
- `data/real/gsm8k_qwen_300.json` — 300 real traces, correctness verified
- `experiments/results/CANONICAL_RESULTS.json` — one authoritative metrics table
- `data/real/GENERATION_LOG.md` — model, prompt, hardware, timestamp

---

### Week 2 (Aug 18–25): The One Experiment That Makes This Defensible

**Goal:** Controlled error injection on organic traces.

#### Step 2.1 — Error Injection Experiment

Take 50–100 verified correct organic traces. Apply 4 controlled corruptions:

| Corruption Type | Method | Expected Detection Rate |
|---|---|---|
| Structural: Missing computation | Remove all CA tags from sequence | Should be HIGH |
| Structural: Premature conclusion | Truncate sequence after PR -> CO | Should be HIGH |
| Structural: Circular reasoning | Inject IN->IN->IN repetition loop | Should be HIGH |
| Semantic-only: Wrong arithmetic | Change numbers in CA step, keep structure identical | Should be LOW — this is the key finding |

**The honest citable claim:** "CoT-HMM is a structural verifier, complementary to semantic verifiers. It catches structural errors efficiently, with low false-positive rate on structurally-valid-but-arithmetically-wrong traces." A low Y for semantic-only is not a failure — it defines your scope.

#### Step 2.2 — Write Injection Section (~500 words)

New section in `paper/main.tex`: Section IV.C "Controlled Structural Error Injection Experiment."

**Deliverables:**
- `scripts/experiments/controlled_injection.py`
- Results table: detection rate per corruption type
- Written section in `paper/main.tex`

---

### Week 3 (Aug 25 – Sept 1): Tighten What's Salvageable

**Goal:** Keep the good parts, cut the indefensible.

#### Step 3.1 — Real Cross-Architecture Transfer

Using 2+ models of real data:
- Train EnsembleHMM on Model A correct traces only
- Score Model B traces (which it has never seen)
- Report: cross-architecture Pearson r and AUC

If r >= 0.7: report as "structural transfer."
If r < 0.5: still report honestly — structural signal may be model-specific at small scale.

#### Step 3.2 — Expand Tagger Validation

- Currently: k=0.953 on N=50 steps (sample too small for a robust claim)
- New target: 150 steps (~3 hours of annotation work)
- Report: k at N=150 with exact confidence interval

#### Step 3.3 — Cut These Sections (They Hurt More Than They Help)

| Item to Cut/Shrink | Reason |
|---|---|
| t-SNE manifold plot | Visualization of synthetic data = meaningless |
| "Cognitive fingerprint" language | Overclaims interpretability without evidence |
| StructureReward / RL hook | Unvalidated; invites "where's the RL result?" question |
| EDR r = -1.00 claim in abstract | Must be re-verified or removed |
| Cross-arch r >= 0.94 | Was on synthetic data; must come from Week 3's real test |

**Deliverables:**
- Real cross-architecture transfer table
- Updated k at N=150
- `paper/main.tex` with cuts applied

---

### Week 4 (Sept 1–8): Rewrite the Paper

**Goal:** Make every claim match one source of truth.

#### Step 4.1 — Rewrite Abstract (One Version)

No r=0.76 AND r=0.476 in the same submission. Sample framing:

> "We propose CoT-HMM, an unsupervised structural plausibility scorer for LLM Chain-of-Thought reasoning, operating at ~150ms/sample on CPU without any labeled reasoning steps. On [N] real GSM8K traces from [models], our Ensemble HMM achieves AUC=[X] ([CI]), outperforming structural heuristic baselines. Controlled error injection demonstrates reliable detection of structural failures (missing computation, premature conclusion) while semantically-wrong-but-structurally-valid traces remain a known limitation."

#### Step 4.2 — Add Mandatory Limitations Paragraph

> "A fundamental limitation of our approach is that we score the written reasoning trace, not the model's internal computation. As Lanham et al. [2023] demonstrated, LLM CoT faithfulness can be as low as 25–39% — the written steps may not reflect the actual factors driving the model's answer. Our HMM scores structural plausibility of the observable trace and should not be interpreted as measuring internal reasoning quality."

#### Step 4.3 — Rewrite Related Work

Add explicit paragraphs distinguishing from:
1. ProcessBench and PRMBench (step-correctness at semantic level vs. structural tag level)
2. CLoT (hidden-representation Markov chain for generation, not scoring discrete tags)
3. ThinkPRM / uPRM (neural verifiers, different resource regime)

#### Step 4.4 — Remove All Stale Numbers

Run grep sweep and verify each number against CANONICAL_RESULTS.json:

```bash
grep -n "AUC\|r=\|r =\|p<\|p <\|0\.9\|0\.8\|0\.7\|0\.6\|0\.5" paper/main.tex
```

**Deliverables:**
- `paper/main_v4.tex` — single clean draft
- All numbers verified against `CANONICAL_RESULTS.json`
- Limitations section with Lanham et al. 2023 citation

---

### Week 5 (Sept 8–15): Buffer + Submit

#### Step 5.1 — Full Number Audit
Final pass: every number in abstract, introduction, results, and conclusion must be in `CANONICAL_RESULTS.json`.

#### Step 5.2 — IEEE Formatting Check
- IEEE column width, figure sizes, reference format verified
- Author list and affiliations correct
- No broken LaTeX refs
- Clean compile: `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

#### Step 5.3 — Verify ICACECT Deadline
The review could not confirm the Sept 15 deadline. **Check the actual CMT portal or PDF call-for-papers before building your timeline around it.**

**Deliverables:**
- Final `paper/main.pdf` — submission-ready
- Submitted to conference

---

## Part 4: What to Abandon for Sept 15

These are good ideas for a journal follow-up. They are not achievable before Sept 15 without sacrificing the foundational data fix:

| Item | Defer To |
|---|---|
| RL/GRPO validation loop | Journal v2 (Spring 2027) |
| Neural-HMM hybrid (DistilBERT) | Journal v2 |
| SCoT-Bench / full benchmark | Journal v2 |
| 5–6 research questions | Collapse to 2–3 for this submission |
| ProcessBench evaluation at full scale | Journal v2 |

---

## Part 5: Risk Matrix

| Risk | Likelihood | Mitigation |
|---|---|---|
| Can't generate real GPU traces in time | MEDIUM | Kaggle free GPU (30hr/wk T4). Start Day 1. 300 traces = ~2–4 hrs. |
| Real data gives lower AUC than synthetic | HIGH | Expected and acceptable. An honest AUC=0.65 on real data beats a fabricated AUC=0.98. |
| EDR dies on real data | MEDIUM | Remove EDR from headline claims. Keep as supplementary diagnostic. |
| Cross-arch transfer doesn't generalize | MEDIUM | Report truthfully; frame as "within-domain transfer" with caveats. |
| ICACECT deadline is wrong | LOW-MEDIUM | Verify this week on the actual CMT page. |

---

## Immediate Next Steps (Today, Aug 11)

1. **[30 min]** Verify the ICACECT Sept 15 deadline on the official CMT portal.
2. **[2–4 hrs]** Open Kaggle/Colab, run `colab_scripts/colab_experiments.py --generate --model qwen-0.5b` on `data/raw/gsm8k.json` to generate ~200–300 real traces.
3. **[1 hr]** Write `scripts/verify_correctness.py` that extracts the final numeric answer from raw CoT text and matches it against GSM8K gold answers.
4. **[30 min]** Run `scripts/run_leakfree_evaluation.py` on the new real data and save to `experiments/results/CANONICAL_RESULTS.json`.
5. **[30 min]** Read the new AUC. Accept it. The paper's credibility depends on it being real.
