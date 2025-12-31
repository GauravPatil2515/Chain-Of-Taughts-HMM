# Phase 4 Status: Scaling Up (200 Samples)

**Status:**

- 🔄 **Generation Started:** `run_real_generation.py` is generating 200 samples.
- 🕒 **Estimated Time:** ~1.5 - 2 Hours.
- 📂 **Output File:** `data/cot_outputs/real_incremental_200.json` (Watch this file grow).

---

## 🛑 Action Item: Wait for Completion

Do not run the next commands until the generation script finishes or you see ~200 items in the output file.

---

## 🚀 After Generation Completes

Run these 4 commands in order to analyze the new large dataset.

### 1. Tag the Data

```bash
python -m src.tagging.rule_based_tagger \
    --input data/cot_outputs/real_test_200.json \
    --output data/tagged/phase4_tagged.json
```

### 2. Verify Correctness & Extract Answers

```bash
python post_process_correctness.py \
    --input data/tagged/phase4_tagged.json \
    --output data/tagged/phase4_tagged_corrected.json
```

### 3. Train HMM (Scale Up)

We increase hidden states to 5 since we have more data.

```bash
python src/hmm/trainer.py \
    --input data/tagged/phase4_tagged_corrected.json \
    --output models/phase4_hmm.pkl \
    --n-components 5
```

### 4. Evaluate

```bash
python run_baseline_eval.py \
    --data data/tagged/phase4_tagged_corrected.json \
    --model models/phase4_hmm.pkl \
    --output experiments/results/phase4_baseline
```

---

## 🏆 Success Criteria for Phase 4

- **Correlation > 0.3**: Stronger signal than Phase 3 (0.16).
- **Smooth Transition Matrix**: Visualizing the HMM should show clear "reasoning paths".
- **Statistical Significance**: With ~200 samples, the p-values in `baseline_comparison.json` should be much lower (significant).
