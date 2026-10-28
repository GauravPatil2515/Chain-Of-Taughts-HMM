# Point-by-Point Author Response to Reviewers

**Manuscript ID:** IDCIoT-154  
**Title:** CoT-HMM: Structural Plausibility Scoring of Chain-of-Thought Reasoning via Hidden Markov Models  
**Authors:** Gaurav Patil, Suhail Shaikh, Bhagyalakshmi V, Parth Parmar, Chirayu Marathe  
**Target Venue:** 5th International Conference on Intelligent Data Communication Technologies and Internet of Things (IDCIoT 2027) — IEEE Conference Record #71706  

---

We sincerely thank the Technical Program Committee and both anonymous reviewers for their thoughtful, rigorous, and constructive evaluations of our manuscript. In response to your invaluable recommendations, we have conducted an extensive revision of our paper. Below, we provide our detailed point-by-point responses mapping every remark to specific modifications in the camera-ready manuscript.

---

## 📋 Response to Reviewer 1

### **Remark R1.1: Sample size of 39 correct traces from Qwen2.5-0.5B; generalizability & statistical power.**
> *"Evaluates only 39 correct traces from Qwen2.5-0.5B; sensitive sample size."*

**Author Response:**  
We thank the reviewer for highlighting the importance of statistical power and sample size.
1. **Clarification on Pilot Design:** In Section IV-A (*Setup: Baselines and Metrics*), we have explicitly clarified that the $N=39$ answer-verified correct traces from Qwen2.5-0.5B (and their 4 derived corruption variants, $N=195$ total variants) serve as a focused pilot perturbation benchmark. We explicitly acknowledge that variants derived from the same source problem share underlying problem complexity.
2. **Mitigating Sample Variance via Bootstrap:** To evaluate sampling uncertainty, we employed non-parametric percentile bootstrap resampling with $B=10{,}000$ iterations to compute all 95% confidence intervals reported in Table II (e.g., premature conclusion AUC = $0.990$ [95% CI: $0.966, 1.000$]).
3. **Cross-Architecture Generalization:** Furthermore, we direct the reviewer to Section IV-C and Table IV (*Zero-Shot Cross-Architecture Structural Transfer Benchmark*), which demonstrates that the transition grammar learned from this pilot corpus transfers zero-shot to substantially larger model families without retraining: **Llama-3.2-1B** ($r=0.946, p<0.001$), **Phi-3-mini-4k** ($r=0.941, p<0.001$), and **Mistral-7B** ($r=0.970, p<0.001$), confirming structural regularities across diverse model families.

---

### **Remark R1.2: Synthetic corruption vs. natural reasoning failures.**
> *"Synthetic corruption vs. natural reasoning failures."*

**Author Response:**  
We appreciate this insightful point regarding the realism of evaluated failure modes.
1. **Methodological Justification:** In Section IV-A and IV-B, we added explicit justification for synthetic error injection: controlled perturbation (premature truncation, loop injection, step deletion) allows causal isolation of structural failure modes without confounding noise from arithmetic miscalculations or vocabulary shifts—an established paradigm in reasoning evaluation benchmarks (e.g., ROSCOE \cite{golovneva2023roscoe}).
2. **Empirical Link to Natural Reasoning Failures:** Crucially, our synthetic detection directly mirrors empirical failure modes observed in natural LLM generations. In Section IV-D, we highlight that among uncorrupted, real model generations on GSM8K, **77.8% of looping chains ($N=150$) were mathematically incorrect vs. 23.8% of non-looping ones ($z=6.18, p=6.3\times 10^{-10}$)**. This proves that the synthetic loop failure mode directly corresponds to a dominant real-world failure mode in LLM reasoning.

---

### **Remark R1.3: Regex-based semantic tagger ambiguity & error propagation.**
> *"Regex-based semantic tagger ambiguity & error propagation."*

**Author Response:**  
We completely agree with the reviewer that rule-based pattern matching has inherent linguistic boundaries.
1. **Inter-Annotator Agreement:** As reported in Section III-B, our prioritized rule-based tagger was evaluated against a 50-step expert-annotated gold standard, achieving Cohen's $\kappa=0.953$, closely matching an LLM-based tagger (Phi-3-mini, $\kappa=0.976$).
2. **Error Analysis in Camera-Ready:** We have added a dedicated paragraph titled *"Tagger Error Analysis"* in Section III-B discussing specific failure modes:
   - *Multi-clause steps:* Sentences containing multiple discourse markers (e.g., *"Hence we have $x=2$"* contains both deductive inference and equation definition).
   - *Colloquial discourse markers:* Conversational uses of keywords (e.g., *"Hence"* used colloquially).
3. **Soft Absorption by Ensemble HMM:** We note that because the Ensemble HMM models continuous emission distributions across $M=12$ models, isolated local tagging errors are softly absorbed across the full trace rather than causing brittle parser failures.
4. **Future Neural Extension:** We explicitly propose fine-tuning a lightweight sentence transformer (e.g., DeBERTa-v3-small) as a natural future extension in Section VI.

---

### **Remark R1.4: Asymmetric AUC across error types (0.658 for missing computation vs. 0.990 for premature conclusions).**
> *"Table II AUC: 0.658 for missing computation & 0.761 for circular reasoning vs. high for premature conclusions."*

**Author Response:**  
This is a keen observation. In Section IV-B, we added a dedicated paragraph titled *\textit{Asymmetric Detection Dynamics}* candidly analyzing these boundaries:
1. **Premature Conclusions (AUC = 0.990 [95% CI: 0.966, 1.000]):** Skipping from problem restatement directly to a final answer ($\text{PR} \rightarrow \text{CO}$) represents a catastrophic terminal transition violation ($P(\text{CO} \mid \text{PR}) < 10^{-4}$ in the learned matrix), triggering an immediate, extreme penalty in normalized log-likelihood ($d=4.166$).
2. **Circular Reasoning (AUC = 0.761 [95% CI: 0.645, 0.866]):** Repeated loops trigger cumulative transition penalties ($d=0.896$), though legitimate multi-step deductions occasionally overlap.
3. **Missing Computation (AUC = 0.658 [95% CI: 0.531, 0.778]):** Omitting arithmetic calculation tags ($\text{CA}$) leaves residual chains of logical deductions ($\text{PR} \rightarrow \text{IN} \rightarrow \text{IN} \rightarrow \text{CO}$) that locally resemble qualitative explanations, preserving partial transition likelihood. We candidly note that the lower bound approaches chance ($0.50$), and outline token-length weighting and entropy thresholds as future mitigations.

---

### **Remark R1.5: Distinction between Table II (0.990) and Table III (0.982) AUC.**
> *"Distinction between Table II (0.990) and Table III (0.982) AUC."*

**Author Response:**  
We thank the reviewer for identifying this potential point of confusion. We have clarified the distinction in Section IV-B and IV-C:
* **Table II** reports *per-category controlled error injection AUCs* on the focused pilot corpus to test specific structural failure modes (premature conclusions: $0.990$, circular loops: $0.761$, missing computation: $0.658$).
* **Table III** reports the *composite plausibility discrimination AUC* ($0.982$) on out-of-fold cross-validated benchmark traces, comparing CoT-HMM directly against established baseline verifiers (ROSCOE, PiCSAR, Length, Random).
Clear explanatory phrasing has been added to Section IV-B to cross-reference these evaluations explicitly.

---

### **Remark R1.6: Conclusion should be a separate section after Results.**
> *"Conclusion should be a separate section after Results."*

**Author Response:**  
We have restructured the manuscript accordingly. Section V is now dedicated exclusively to **Discussion**, and Section VI is now a standalone section titled **Conclusion and Future Work** summarizing our findings and outlining three concrete future directions (neural DeBERTa tagging, hybrid arithmetic verification, and multi-domain transfer).

---

### **Remark R1.7: Proofread for grammar/spelling/typos.**
> *"Proofread for grammar/spelling/typos."*

**Author Response:**  
A thorough editorial review of the entire manuscript was performed:
1. **Algorithmic Correction:** Corrected Section III-D from the misnomer "Viterbi training" to "Expectation-Maximization (Baum-Welch) training", clarifying that Viterbi decoding is used exclusively for inferring optimal state sequences.
2. **Terminology Precision:** Explicitly classified training chains as "answer-verified" (matching benchmark ground truth).
3. **Typography & Layout:** Refined float spacing, equation formatting, and verified exact 5-page IEEEtran conference compliance.

---

## 📋 Response to Reviewer 2

### **Remark R2.1: How does CoT-HMM handle arithmetic errors?**
> *"How does CoT-HMM handle arithmetic errors?"*

**Author Response:**  
We thank the reviewer for raising this essential architectural question. We have added a dedicated subsection in Section V titled **Structural vs. Arithmetic Correctness**:
1. **Explicit Scope Boundary:** CoT-HMM is strictly a *structural verification filter*, not an arithmetic calculator. A derivation containing an arithmetic error such as $12 \times 4 = 52$ follows valid reasoning syntax ($\text{EQ} \rightarrow \text{CA} \rightarrow \text{IN}$); therefore, CoT-HMM appropriately assigns it high structural plausibility.
2. **Empirical Validation via Semantic Control:** In Table II, our semantic-only corruption condition (where reasoning structure is pristine but arithmetic is corrupted) yielded an AUC of **0.501** (random guessing). This confirms empirically that CoT-HMM isolates reasoning flow without confounding calculation errors.
3. **Role in Pipeline:** Catching numerical calculation errors is the role of deterministic symbolic tools (e.g., Python REPL / calculator APIs). CoT-HMM is engineered as a fast ($\approx$150ms), zero-GPU first-pass filter that eliminates structurally broken chains (loops, missing steps, premature stops) before calling expensive neural verifiers or symbolic tools.

---

### **Remark R2.2: Missing-computation detection AUC remains insufficient (0.658).**
> *"Missing-computation detection AUC remains insufficient (0.658)."*

**Author Response:**  
We concur with the reviewer's candid assessment. As detailed in our response to R1.4 and Section IV-B:
* Removing calculation tags leaves inference sub-chains ($\text{PR} \rightarrow \text{IN} \rightarrow \text{CO}$) that locally resemble qualitative explanations, preserving partial transition likelihood.
* In the revised manuscript, we explicitly acknowledge this limitation and propose two concrete algorithmic improvements for future iterations: (1) incorporating token-length weighted transition probabilities, and (2) combining log-likelihood with step-level Structural Entropy $H(\mathbf{o})$ to flag anomalous omission of quantitative derivation.

---

### **Remark R2.3: How was the 99.3% routing accuracy calculated?**
> *"How was the 99.3% routing accuracy calculated?"*

**Author Response:**  
We thank the reviewer for requesting methodological clarity. In Section IV-D, we have added the explicit mathematical formulation for the High-Confidence Pass Tier:
$$\text{Acc}_{\text{pass}} = \frac{|\mathcal{T}_{\text{pass}} \cap \mathcal{C}_{\text{correct}}|}{|\mathcal{T}_{\text{pass}}|} = \frac{TP}{TP + FP} = 99.3\%$$
where $\mathcal{T}_{\text{pass}}$ is the subset of reasoning traces whose plausibility margin exceeds the calibrated decision threshold $\tau_{\text{high}}$ (determined via Youden's $J$ statistic on the validation set), and $\mathcal{C}_{\text{correct}}$ represents ground-truth correct chains. Out of the top 35% most confident traces routed through this gate, 99.3% were mathematically correct, requiring zero expensive secondary verification.

---

### **Remark R2.4: Regex-based semantic tagging needs error analysis.**
> *"Regex-based semantic tagging needs error analysis."*

**Author Response:**  
In response to this comment (and R1.3), we have added an error analysis in Section III-B detailing the specific failure modes of keyword regex matching:
* Ambiguities arise predominantly in multi-clause sentences with competing discourse markers (e.g., steps combining equations and deductions) and informal discourse markers.
* We demonstrate that despite these isolated edge cases, the tagger maintains strong gold-standard agreement ($\kappa=0.953$), and the Ensemble HMM's soft probability distributions prevent individual misclassifications from degrading sequence-level scores.

---

### **Remark R2.5: Semantic error detection limitations require discussion.**
> *"Semantic error detection limitations require discussion."*

**Author Response:**  
In Section V (*Discussion*), under **Structural vs. Arithmetic Correctness** and **Limitations**, we have expanded the discussion on why structural scoring and semantic verification are fundamentally orthogonal and complementary:
* Structural verifiers (like CoT-HMM) excel at detecting macroeconomic reasoning flaws: recursive loops, premature conclusions, and disorganized progression.
* Semantic verifiers (like ROSCOE or neural PRMs) evaluate token-level factual truth and arithmetic accuracy.
* Deploying CoT-HMM as a front-line structural filter prunes defective candidates with zero GPU compute, saving 42%–47% of downstream token costs while letting specialized downstream tools verify semantic accuracy.

---

## 📌 Summary of Changes Made to Manuscript (`paper/main.tex`)

| Section in Revised Paper | Key Changes & Additions | Addressed Remarks |
| :--- | :--- | :--- |
| **Section III-B** (*Semantic Tagging*) | Added error analysis of regex matching, multi-clause ambiguity, and neural tagger roadmap. | R1.3, R2.4 |
| **Section IV-A** (*Setup*) | Clarified $N=39$ pilot seed traces, $B=10{,}000$ bootstrap CIs, and cross-architecture generalization. | R1.1 |
| **Section IV-B** (*Error Injection*) | Justified synthetic perturbation methodology; detailed asymmetric AUC mechanism (terminal vs sub-chain); clarified Table II vs Table III. | R1.2, R1.4, R1.5, R2.2 |
| **Section IV-D** (*Routing Gate*) | Added explicit mathematical formula for 99.3% pass-tier routing accuracy (Eq. 3). | R2.3 |
| **Section V** (*Discussion*) | Dedicated section with new subsection *Structural vs. Arithmetic Correctness* delineating scope. | R2.1, R2.5 |
| **Section VI** (*Conclusion & Future Work*) | Created standalone Conclusion and Future Work section with 3 concrete research directions. | R1.6 |
| **Throughout** | Editorial proofreading, formatting polish, and strict 5-page IEEE layout maintenance. | R1.7 |

---
We once again express our gratitude to the reviewers for their constructive insights, which have significantly strengthened the academic rigor and clarity of our manuscript.
