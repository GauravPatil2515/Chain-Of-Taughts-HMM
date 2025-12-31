# Step-by-Step Execution Guide

## 🚀 PHASE 1: Data Collection - START HERE

### Step 1.1: Test Data Generation (5 minutes)

**Test with small batch first:**

```bash
# Generate 20 samples to verify everything works
python generate_batch.py \
    --dataset data/raw/gsm8k.json \
    --dataset-name gsm8k \
    --domain math \
    --model qwen-0.5b \
    --output data/cot_outputs/test_gsm8k_qwen.json \
    --max-samples 20 \
    --seed 42
```

**Expected output:**
- File: `data/cot_outputs/test_gsm8k_qwen.json`
- ~20 CoT samples
- Accuracy: 20-40% (expected for 0.5B model)

**Verify:**
```bash
# Check file was created
ls data/cot_outputs/test_gsm8k_qwen.json

# Check sample count
python -c "import json; data=json.load(open('data/cot_outputs/test_gsm8k_qwen.json')); print(f'Samples: {len(data)}')"
```

---

### Step 1.2: Full GSM8K Generation (30-60 minutes)

**Generate 150 samples from Qwen model:**

```bash
python generate_batch.py \
    --dataset data/raw/gsm8k.json \
    --dataset-name gsm8k \
    --domain math \
    --model qwen-0.5b \
    --output data/cot_outputs/gsm8k_qwen_150.json \
    --max-samples 150 \
    --seed 42
```

**Progress tracking:**
- Watch progress bar
- Intermediate saves every 10 samples
- Final stats printed at end

---

### Step 1.3: Check Data Quality

**Run quality check:**

```bash
python scripts/check_data_quality.py data/cot_outputs/gsm8k_qwen_150.json
```

**Look for:**
- ✅ Correct/Incorrect ratio: 40-70%
- ✅ All fields populated
- ✅ No duplicate IDs
- ✅ Average CoT length: 100-500 characters

---

### Step 1.4: Generate from Additional Models (Optional)

**If you have more compute, generate from Phi model:**

```bash
python generate_batch.py \
    --dataset data/raw/gsm8k.json \
    --dataset-name gsm8k \
    --domain math \
    --model phi-3.5 \
    --output data/cot_outputs/gsm8k_phi_150.json \
    --max-samples 150 \
    --seed 42
```

---

### Step 1.5: Download Additional Datasets

**StrategyQA (Commonsense Reasoning):**

```bash
python src/data_collection/download_datasets.py \
    --dataset strategyqa \
    --samples 100 \
    --output data/raw/strategyqa.json
```

**ARC-Easy (Science/Logic):**

```bash
python src/data_collection/download_datasets.py \
    --dataset arc_easy \
    --samples 50 \
    --output data/raw/arc_easy.json
```

---

## 🏷️ PHASE 2: Tagging & Validation

### Step 2.1: Tag Generated CoTs

**Tag the GSM8K data:**

```bash
python -m src.tagging.rule_based_tagger \
    --input data/cot_outputs/gsm8k_qwen_150.json \
    --output data/tagged/gsm8k_qwen_150_tagged.json
```

**Check output:**
```bash
# View first tagged sample
python -c "import json; data=json.load(open('data/tagged/gsm8k_qwen_150_tagged.json')); print(json.dumps(data[0], indent=2))"
```

---

### Step 2.2: Validate Tagging Quality

**Manual validation (CRITICAL for paper quality):**

1. Open `data/tagged/gsm8k_qwen_150_tagged.json`
2. Randomly select 20 samples
3. Check each tag assignment
4. Document any errors in `docs/tagging_validation.txt`

**Quality metrics to compute:**

```python
# Run this in Python or create validation script
import json

with open('data/tagged/gsm8k_qwen_150_tagged.json') as f:
    data = json.load(f)

# Compute statistics
total_steps = sum(len(d['tag_sequence']) for d in data)
unique_tags = set()
for d in data:
    unique_tags.update(d['tag_sequence'])

print(f"Total CoTs: {len(data)}")
print(f"Total steps: {total_steps}")
print(f"Avg steps/CoT: {total_steps/len(data):.1f}")
print(f"Unique tags: {unique_tags}")
```

---

## 🧮 PHASE 3: HMM Training

### Step 3.1: Train on GSM8K Data

**Train HMM on correct samples only:**

```bash
python -m src.hmm.trainer \
    --input data/tagged/gsm8k_qwen_150_tagged.json \
    --output models/hmm_gsm8k.pkl \
    --n-components 5 \
    --correct-only
```

**Check model:**
```bash
# Verify model file exists
ls models/hmm_gsm8k.pkl

# Check analysis
cat models/hmm_gsm8k.analysis.json
```

---

### Step 3.2: Score All CoTs

**Score with trained HMM:**

```bash
python -m src.hmm.scorer \
    --model models/hmm_gsm8k.pkl \
    --input data/tagged/gsm8k_qwen_150_tagged.json \
    --output experiments/results/gsm8k_qwen_scored.json
```

---

## 📊 PHASE 4: Baseline Evaluation

### Step 4.1: Run Full Baseline Comparison

**Compare HMM against all baselines:**

```bash
python run_baseline_eval.py \
    --data experiments/results/gsm8k_qwen_scored.json \
    --model models/hmm_gsm8k.pkl \
    --output experiments/results/baseline_gsm8k
```

**Output files:**
- `experiments/results/baseline_gsm8k/baseline_comparison.json`
- `experiments/results/baseline_gsm8k/figures/*.png` (5 plots)

---

### Step 4.2: Review Results

**Key metrics to check:**

1. **Correlation table** - Which method correlates best with correctness?
2. **F1 scores** - Which method classifies best?
3. **Statistical significance** - Are differences real? (p < 0.05)
4. **Effect sizes** - How large are the differences?

**Open in browser/viewer:**
```bash
# View correlation plot
start experiments/results/baseline_gsm8k/figures/baseline_correlation_comparison.png

# View metrics
start experiments/results/baseline_gsm8k/figures/baseline_metrics_comparison.png
```

---

## 🔬 PHASE 5: Analysis & Case Studies

### Step 5.1: Identify Interesting Cases

**Find examples where HMM wins:**

```python
import json
import pandas as pd

# Load results
with open('experiments/results/baseline_gsm8k/baseline_comparison.json') as f:
    report = json.load(f)

df = pd.DataFrame(report['all_scores'])

# HMM high, baselines low
hmm_wins = df[
    (df['hmm_score'] > df['hmm_score'].median()) &
    (df['combined_score'] < df['combined_score'].median())
].sort_values('hmm_score', ascending=False)

print("Top 5 cases where HMM outperforms baselines:")
print(hmm_wins[['id', 'hmm_score', 'combined_score', 'is_correct']].head())
```

---

### Step 5.2: Write Case Studies

**For each selected case (6-8 total):**

1. Print the CoT text
2. Show tag sequence
3. Show HMM score vs baseline scores
4. Explain WHY HMM caught something baselines missed
5. Save to `experiments/analysis/case_studies.md`

---

## 📝 PHASE 6: Paper Writing

### Step 6.1: Generate All Figures

**Create publication-quality figures:**

```bash
# All baseline plots already created
# Now create additional analysis plots

python scripts/create_paper_figures.py \
    --results experiments/results/baseline_gsm8k/ \
    --output paper/figures/
```

---

### Step 6.2: Create Results Tables

**Main results table (LaTeX format):**

```bash
python scripts/generate_results_table.py \
    --results experiments/results/baseline_gsm8k/baseline_comparison.json \
    --format latex \
    --output paper/tables/main_results.tex
```

---

### Step 6.3: Draft Paper Sections

**Order of writing:**

1. **Method section first** (easiest)
   - Describe tagging scheme
   - Explain HMM training
   - List baseline methods

2. **Results section** (use your data)
   - Insert tables and figures
   - Describe key findings
   - Report statistical tests

3. **Introduction** (context)
   - Motivation
   - Problem statement
   - Your contribution

4. **Related work** (position your work)
   - Compare to EIRL, ECCoT, etc.
   - Highlight your novelty

5. **Analysis** (deep dive)
   - Case studies
   - Error analysis
   - Limitations

6. **Conclusion** (summary)

---

## ✅ Daily Checklist

### Week 1 (Data Collection)

**Monday:**
- [ ] Test generation script (20 samples)
- [ ] Verify output format
- [ ] Start full generation (150 samples)

**Tuesday:**
- [ ] Complete GSM8K generation
- [ ] Quality check data
- [ ] Start second model (if applicable)

**Wednesday-Thursday:**
- [ ] Download StrategyQA dataset
- [ ] Generate CoTs for StrategyQA
- [ ] Quality check

**Friday:**
- [ ] Review all generated data
- [ ] Compute statistics (correct/incorrect ratio)
- [ ] Document any issues

---

### Week 2 (Tagging & Training)

**Monday:**
- [ ] Tag all GSM8K data
- [ ] Manual validation (20 samples)
- [ ] Fix any tagging issues

**Tuesday:**
- [ ] Tag StrategyQA data
- [ ] Compute tagging statistics
- [ ] Document edge cases

**Wednesday:**
- [ ] Train HMM on GSM8K
- [ ] Validate convergence
- [ ] Train on StrategyQA

**Thursday:**
- [ ] Score all data with HMM
- [ ] Verify scores look reasonable
- [ ] Save analysis files

**Friday:**
- [ ] Run baseline evaluation
- [ ] Review initial results
- [ ] Plan case studies

---

### Week 3-4 (Analysis)

**Focus:**
- Deep dive into results
- Case study selection
- Statistical testing
- Cross-validation

---

### Week 5-8 (Writing)

**Focus:**
- Draft all sections
- Create figures/tables
- Get feedback
- Revise and polish

---

## 🚨 Troubleshooting

### Problem: Generation is too slow

**Solution:**
- Reduce max_samples to 50-100
- Use smaller model (qwen-0.5b instead of qwen-1.5b)
- Run overnight

### Problem: Low accuracy (< 20%)

**Check:**
- Is model loaded correctly?
- Is answer extraction working?
- Try different temperature

### Problem: Tagging inconsistent

**Fix:**
- Add more patterns to rule_based_tagger.py
- Manual review and document edge cases
- Consider semi-automated tagging

### Problem: HMM doesn't converge

**Try:**
- Reduce n_components (try 4 instead of 6)
- Increase n_iterations (try 200)
- Check if you have enough training data (need > 50 correct samples)

---

## 📊 Progress Tracking

**Use this template to track daily progress:**

```
Date: ___________
Phase: ___________
Tasks completed:
  - [ ] Task 1
  - [ ] Task 2
  - [ ] Task 3

Issues encountered:
  - Issue 1: ___________
  - Solution: ___________

Next steps:
  - Step 1
  - Step 2

Metrics:
  - Samples generated: ___/300
  - Samples tagged: ___/300
  - Models trained: ___/3
  - Paper progress: ___%
```

---

## 🎯 Quick Commands Reference

```bash
# Test generation
python generate_batch.py --dataset data/raw/gsm8k.json --dataset-name gsm8k --domain math --model qwen-0.5b --output data/cot_outputs/test.json --max-samples 20

# Full generation
python generate_batch.py --dataset data/raw/gsm8k.json --dataset-name gsm8k --domain math --model qwen-0.5b --output data/cot_outputs/gsm8k_qwen.json --max-samples 150

# Tagging
python -m src.tagging.rule_based_tagger --input data/cot_outputs/gsm8k_qwen.json --output data/tagged/gsm8k_qwen_tagged.json

# Training
python -m src.hmm.trainer --input data/tagged/gsm8k_qwen_tagged.json --output models/hmm_gsm8k.pkl --correct-only

# Scoring
python -m src.hmm.scorer --model models/hmm_gsm8k.pkl --input data/tagged/gsm8k_qwen_tagged.json --output experiments/results/scored.json

# Baseline eval
python run_baseline_eval.py --data experiments/results/scored.json --model models/hmm_gsm8k.pkl --output experiments/results/baseline
```

---

**Ready to start? Begin with Step 1.1!**
