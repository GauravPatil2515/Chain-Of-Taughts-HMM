# Phase 1: Data Collection - In-Depth Detailed Plan

**Duration:** Week 1-2 (10-14 days)  
**Goal:** Generate 300-1000 high-quality CoT samples across multiple models and domains  
**Status:** READY TO EXECUTE

---

## 📊 Overview & Targets

### Final Deliverables

| Metric | Minimum (Publishable) | Target (Strong Paper) |
|--------|----------------------|----------------------|
| **Total Samples** | 300 | 600-1000 |
| **Models** | 2 | 3-4 |
| **Domains** | 2 | 3 |
| **Correct/Incorrect** | 50/50 ± 15% | 60/40 ± 10% |
| **Avg CoT Length** | 100-300 chars | 150-400 chars |
| **Empty/Error Rate** | <5% | <2% |

### Dataset Breakdown (Target Configuration)

```
Configuration A (Minimum - 300 samples):
├── GSM8K (Math) - 150 samples
│   ├── Qwen-0.5B: 150 samples
│   └── Expected: ~30-40% correct
├── StrategyQA (Commonsense) - 150 samples
│   └── Qwen-0.5B: 150 samples
└── Total: 300 samples, 1 model, 2 domains

Configuration B (Target - 600 samples):
├── GSM8K (Math) - 300 samples
│   ├── Qwen-0.5B: 150 samples
│   └── Phi-3.5-mini: 150 samples
├── StrategyQA (Commonsense) - 200 samples
│   ├── Qwen-0.5B: 100 samples
│   └── Phi-3.5-mini: 100 samples
└── ARC-Easy (Science) - 100 samples
    └── Qwen-0.5B: 100 samples
    Total: 600 samples, 2 models, 3 domains

Configuration C (Ambitious - 900 samples):
├── GSM8K: 450 (3 models × 150)
├── StrategyQA: 300 (3 models × 100)
└── ARC-Easy: 150 (3 models × 50)
    Total: 900 samples, 3 models, 3 domains
```

**Recommendation:** Start with Configuration A (300 samples), then expand if time/compute allows.

---

## 🎯 Day-by-Day Execution Plan

### **Day 1: Setup & Testing (2-3 hours)**

#### Morning: Environment Verification

**Step 1.1: Verify Project Structure** (5 min)

```bash
# Check you're in correct directory
pwd
# Should show: .../chain of tough -hmm

# Verify key files exist
ls generate_batch.py
ls data/raw/gsm8k.json
ls scripts/check_data_quality.py
```

**Expected output:** All files should exist. If gsm8k.json missing, run:
```bash
python src/data_collection/download_datasets.py --dataset gsm8k --samples 200 --output data/raw/gsm8k.json
```

---

**Step 1.2: Test Model Loading** (10-15 min)

Create test script `test_model_load.py`:

```python
"""Quick model loading test"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Testing model: {MODEL_ID}")
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("✅ Tokenizer loaded")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)
print("✅ Model loaded")
print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
print("\n✅ Model loading successful!")
```

Run:
```bash
python test_model_load.py
```

**Troubleshooting:**
- If download fails: Check internet connection, try again
- If out of memory: Close other programs
- If "trust_remote_code" error: Update transformers: `pip install --upgrade transformers`

---

**Step 1.3: Generate Test Batch (20-30 min)**

Generate 10 samples to verify pipeline:

```bash
python generate_batch.py \
    --dataset data/raw/gsm8k.json \
    --dataset-name gsm8k \
    --domain math \
    --model qwen-0.5b \
    --output data/cot_outputs/test_batch_10.json \
    --max-samples 10 \
    --temperature 0.7 \
    --seed 42
```

**What to watch:**
- Model downloads (first time only, ~1-2GB)
- Progress bar advancing
- No errors in output
- Final statistics printed

**Expected output:**
```
Loading model: Qwen/Qwen2.5-0.5B-Instruct
Model loaded! Generating 10 samples...
Generating (qwen-0.5b): 100%|████████| 10/10 [02:30<00:00, 15.0s/it]

============================================================
Generation Complete: qwen-0.5b on gsm8k
============================================================
Total samples: 10
Correct: 3 (30.0%)
Incorrect: 7 (70.0%)
Saved to: data/cot_outputs/test_batch_10.json
```

**Quality check:**
```bash
python scripts/check_data_quality.py data/cot_outputs/test_batch_10.json
```

**If test passes:** ✅ Continue to afternoon session  
**If test fails:** 🔧 Debug issues before proceeding

---

#### Afternoon: Small-Scale Generation

**Step 1.4: Generate 20-Sample Batch** (20-30 min)

```bash
python generate_batch.py \
    --dataset data/raw/gsm8k.json \
    --dataset-name gsm8k \
    --domain math \
    --model qwen-0.5b \
    --output data/cot_outputs/gsm8k_qwen_20.json \
    --max-samples 20 \
    --temperature 0.7 \
    --seed 42
```

**Step 1.5: Manual Inspection** (15 min)

Open `data/cot_outputs/gsm8k_qwen_20.json` and review:

**Check these samples:**
- Sample 0 (first)
- Sample 10 (middle)
- Sample 19 (last)

**What to look for:**

✅ **Good CoT Example:**
```json
{
  "cot_raw": "First, I need to find the total cost. Each pen costs $3 and the student buys 4 pens. So 4 × 3 = 12 dollars for pens. The notebook costs $5. Adding these: 12 + 5 = 17. Therefore, the total is $17.",
  "is_correct": true,
  "predicted_answer": "17"
}
```

❌ **Bad CoT Example (fix needed):**
```json
{
  "cot_raw": "",  // Empty - generation failed
  "is_correct": false
}
```

**Document findings in:** `logs/phase1_day1_inspection.txt`

Example:
```
Date: 2025-12-27
Batch: gsm8k_qwen_20.json
Issues found:
- Sample 5: Empty CoT (generation error)
- Sample 12: Very short CoT (<20 chars)
- Sample 15: Gibberish output

Actions:
- Acceptable error rate: 2/20 = 10% (within tolerance)
- Continue to full generation
```

---

### **Day 2: Full-Scale GSM8K Generation (4-6 hours)**

#### Morning: First Full Batch

**Step 2.1: Generate 150 GSM8K Samples** (60-90 min)

```bash
# Create logs directory
mkdir -p logs

# Run generation with output redirection
python generate_batch.py \
    --dataset data/raw/gsm8k.json \
    --dataset-name gsm8k \
    --domain math \
    --model qwen-0.5b \
    --output data/cot_outputs/gsm8k_qwen_150.json \
    --max-samples 150 \
    --temperature 0.7 \
    --seed 42 \
    2>&1 | tee logs/generation_gsm8k_qwen_150.log
```

**Monitor progress:**
- Open another terminal
- Watch file size: `watch -n 10 "ls -lh data/cot_outputs/gsm8k_qwen_150.json"`
- Check sample count: `watch -n 30 "python -c \"import json; print(len(json.load(open('data/cot_outputs/gsm8k_qwen_150.json'))))\""`

**Time estimates by system:**
- Modern laptop (8GB RAM, no GPU): 60-90 minutes
- Older laptop (4GB RAM): 90-120 minutes
- If taking >2 hours: Consider overnight run

**Intermediate checkpoint** (optional, if generation is slow):

After 50 samples, you can:
1. Stop (Ctrl+C)
2. Review intermediate results
3. Resume with `--max-samples 150` (will continue from where stopped)

---

**Step 2.2: Comprehensive Quality Check** (15 min)

```bash
python scripts/check_data_quality.py data/cot_outputs/gsm8k_qwen_150.json > logs/quality_gsm8k_qwen_150.log
```

**Detailed analysis:**

```python
# Create analysis script: analyze_batch.py
import json
import numpy as np
from collections import Counter

with open('data/cot_outputs/gsm8k_qwen_150.json') as f:
    data = json.load(f)

print("="*70)
print("DETAILED BATCH ANALYSIS")
print("="*70)

# Basic stats
total = len(data)
correct = sum(1 for d in data if d['is_correct'])
incorrect = total - correct

print(f"\nTotal samples: {total}")
print(f"Correct: {correct} ({correct/total*100:.1f}%)")
print(f"Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")

# CoT length distribution
lengths = [len(d['cot_raw']) for d in data]
print(f"\nCoT Length Statistics:")
print(f"  Min: {min(lengths)}")
print(f"  Max: {max(lengths)}")
print(f"  Mean: {np.mean(lengths):.1f}")
print(f"  Median: {np.median(lengths):.1f}")
print(f"  Std: {np.std(lengths):.1f}")

# Empty/error detection
empty = sum(1 for l in lengths if l < 20)
print(f"\nEmpty/Very Short CoTs: {empty} ({empty/total*100:.1f}%)")

# Predicted answer distribution
answers = [d['predicted_answer'] for d in data]
print(f"\nUnique predicted answers: {len(set(answers))}")

# Model info
print(f"\nModel: {data[0]['model']}")
print(f"Temperature: {data[0]['generation_params']['temperature']}")
print(f"Max tokens: {data[0]['generation_params']['max_new_tokens']}")

print("\n" + "="*70)
```

Run:
```bash
python analyze_batch.py > logs/analysis_gsm8k_quen_150.txt
cat logs/analysis_gsm8k_qwen_150.txt
```

---

#### Afternoon: Data Cleaning & Validation

**Step 2.3: Filter & Clean Data** (30 min)

Create cleaning script `clean_batch.py`:

```python
"""Clean generated data by removing errors and very poor samples"""
import json
import sys

def clean_data(input_path, output_path, min_cot_length=20):
    """Remove samples with errors or very short CoTs"""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    original_count = len(data)
    
    # Filter criteria
    cleaned = []
    removed = []
    
    for item in data:
        cot = item.get('cot_raw', '')
        
        # Remove if:
        # 1. Empty or very short CoT
        # 2. Contains error message
        # 3. Predicted answer is empty
        
        if len(cot) < min_cot_length:
            removed.append((item['id'], 'too_short'))
            continue
        
        if '[Generation Error' in cot or '[Error:' in cot:
            removed.append((item['id'], 'error'))
            continue
        
        if not item.get('predicted_answer', '').strip():
            removed.append((item['id'], 'no_answer'))
            continue
        
        cleaned.append(item)
    
    # Save cleaned data
    with open(output_path, 'w') as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    
    # Report
    print(f"Original: {original_count}")
    print(f"Cleaned: {len(cleaned)}")
    print(f"Removed: {len(removed)} ({len(removed)/original_count*100:.1f}%)")
    
    if removed:
        print("\nRemoval reasons:")
        from collections import Counter
        reasons = Counter(r[1] for r in removed)
        for reason, count in reasons.items():
            print(f"  {reason}: {count}")
    
    return len(cleaned)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_batch.py <input.json> <output.json>")
        sys.exit(1)
    
    clean_data(sys.argv[1], sys.argv[2])
```

Run:
```bash
python clean_batch.py \
    data/cot_outputs/gsm8k_qwen_150.json \
    data/cot_outputs/gsm8k_qwen_150_cleaned.json
```

**Decision point:**
- If removed <10%: ✅ Acceptable, continue
- If removed 10-20%: ⚠️ Review settings, but usable
- If removed >20%: 🔧 Regenerate with adjusted parameters

---

**Step 2.4: Balance Correct/Incorrect** (Optional, 30 min)

If your ratio is very imbalanced (e.g., 80% correct or 80% incorrect):

```python
# balance_dataset.py
import json
import random

def balance_data(input_path, output_path, target_correct_ratio=0.6, seed=42):
    """Balance correct/incorrect ratio"""
    random.seed(seed)
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    correct = [d for d in data if d['is_correct']]
    incorrect = [d for d in data if not d['is_correct']]
    
    print(f"Original: {len(correct)} correct, {len(incorrect)} incorrect")
    
    # Calculate target counts
    if len(correct) / len(data) > target_correct_ratio + 0.15:
        # Too many correct, downsample
        target_correct = int(len(data) * target_correct_ratio)
        correct = random.sample(correct, min(target_correct, len(correct)))
        print(f"Downsampled correct to {len(correct)}")
    
    elif len(incorrect) / len(data) > (1 - target_correct_ratio) + 0.15:
        # Too many incorrect, downsample
        target_incorrect = int(len(data) * (1 - target_correct_ratio))
        incorrect = random.sample(incorrect, min(target_incorrect, len(incorrect)))
        print(f"Downsampled incorrect to {len(incorrect)}")
    
    balanced = correct + incorrect
    random.shuffle(balanced)
    
    with open(output_path, 'w') as f:
        json.dump(balanced, f, indent=2, ensure_ascii=False)
    
    print(f"Balanced: {len(correct)} correct ({len(correct)/len(balanced)*100:.1f}%), "
          f"{len(incorrect)} incorrect ({len(incorrect)/len(balanced)*100:.1f}%)")
    print(f"Total: {len(balanced)}")

if __name__ == "__main__":
    balance_data(
        'data/cot_outputs/gsm8k_qwen_150_cleaned.json',
        'data/cot_outputs/gsm8k_qwen_150_balanced.json'
    )
```

**Use cleaned OR balanced version as your final GSM8K dataset.**

---

### **Day 3: StrategyQA Generation (3-4 hours)**

**Step 3.1: Download StrategyQA** (5 min)

```bash
python src/data_collection/download_datasets.py \
    --dataset strategyqa \
    --samples 150 \
    --output data/raw/strategyqa.json
```

**Verify:**
```bash
python -c "import json; data=json.load(open('data/raw/strategyqa.json')); print(f'Loaded {len(data)} questions'); print('Sample:', data[0]['question'])"
```

---

**Step 3.2: Generate StrategyQA CoTs** (60-90 min)

```bash
python generate_batch.py \
    --dataset data/raw/strategyqa.json \
    --dataset-name strategyqa \
    --domain commonsense \
    --model qwen-0.5b \
    --output data/cot_outputs/strategyqa_qwen_150.json \
    --max-samples 150 \
    --temperature 0.7 \
    --seed 43 \
    2>&1 | tee logs/generation_strategyqa_qwen_150.log
```

**Note:** Different seed (43) for diversity

---

**Step 3.3: Quality Check & Clean** (20 min)

```bash
# Quality check
python scripts/check_data_quality.py data/cot_outputs/strategyqa_qwen_150.json

# Clean
python clean_batch.py \
    data/cot_outputs/strategyqa_qwen_150.json \
    data/cot_outputs/strategyqa_qwen_150_cleaned.json

# Balance if needed
python balance_dataset.py
```

---

### **Day 4: Consolidation & Backup (2 hours)**

**Step 4.1: Merge All Datasets**

```python
# merge_datasets.py
import json

datasets = [
    'data/cot_outputs/gsm8k_qwen_150_cleaned.json',
    'data/cot_outputs/strategyqa_qwen_150_cleaned.json'
]

merged = []
for path in datasets:
    with open(path) as f:
        data = json.load(f)
        merged.extend(data)
        print(f"Added {len(data)} from {path}")

print(f"\nTotal merged: {len(merged)}")

# Save
with open('data/cot_outputs/merged_all_datasets.json', 'w') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print("Saved to: data/cot_outputs/merged_all_datasets.json")
```

---

**Step 4.2: Create Backup**

```bash
# Create backup directory with timestamp
BACKUP_DIR="backups/phase1_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Copy all generated data
cp -r data/cot_outputs/*.json $BACKUP_DIR/
cp -r logs/*.log $BACKUP_DIR/

# Create metadata
cat > $BACKUP_DIR/README.txt << EOF
Phase 1 Data Collection Backup
Generated: $(date)
Total samples: $(python -c "import json; print(len(json.load(open('data/cot_outputs/merged_all_datasets.json'))))")
Datasets: GSM8K, StrategyQA
Models: Qwen-0.5B
EOF

echo "Backup created: $BACKUP_DIR"
```

---

**Step 4.3: Generate Phase 1 Report**

```bash
python scripts/generate_phase1_report.py > reports/phase1_completion_report.md
```

Create `scripts/generate_phase1_report.py`:

```python
"""Generate comprehensive Phase 1 completion report"""
import json
import os
from pathlib import Path
from datetime import datetime

print("# Phase 1: Data Collection - Completion Report")
print(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n---\n")

# Load all datasets
datasets = {}
total_samples = 0

for file in Path('data/cot_outputs').glob('*_cleaned.json'):
    with open(file) as f:
        data = json.load(f)
        datasets[file.stem] = data
        total_samples += len(data)

print("## 📊 Dataset Summary\n")
print(f"**Total Samples:** {total_samples}\n")

for name, data in datasets.items():
    correct = sum(1 for d in data if d['is_correct'])
    print(f"### {name}")
    print(f"- Samples: {len(data)}")
    print(f"- Correct: {correct} ({correct/len(data)*100:.1f}%)")
    print(f"- Incorrect: {len(data)-correct} ({(len(data)-correct)/len(data)*100:.1f}%)")
    print()

print("## ✅ Quality Metrics\n")
# Add more analysis here

print("## 🎯 Next Steps\n")
print("1. Tag all generated CoTs (Phase 2)")
print("2. Train HMM on correct samples (Phase 3)")
print("3. Run baseline evaluation (Phase 4)")
```

---

### **Day 5-7: Optional Expansion**

If you have time/compute, expand dataset:

**Option A: Add Second Model (Phi-3.5)**

```bash
# Repeat Day 2-3 with phi-3.5 model
python generate_batch.py --model phi-3.5 ...
```

**Option B: Add Third Domain (ARC-Easy)**

```bash
# Download ARC
python src/data_collection/download_datasets.py \
    --dataset arc_easy \
    --samples 100 \
    --output data/raw/arc_easy.json

# Generate
python generate_batch.py \
    --dataset data/raw/arc_easy.json \
    --dataset-name arc_easy \
    --domain science \
    --model qwen-0.5b \
    --output data/cot_outputs/arc_qwen_100.json \
    --max-samples 100 \
    --seed 44
```

---

## 🔧 Troubleshooting Guide

### Issue: Generation is Too Slow

**Symptoms:** <0.5 samples/minute

**Solutions:**
1. **Reduce max_tokens:**
   ```bash
   --max-tokens 256  # Instead of 512
   ```

2. **Run overnight:**
   ```bash
   nohup python generate_batch.py ... > generation.log 2>&1 &
   ```

3. **Batch processing:**
   Generate 50 at a time, merge later

---

### Issue: Low Accuracy (<20%)

**This is actually OK!** You want incorrect examples.

But if concerned:
1. Check answer extraction logic
2. Try temperature=0.5 (more deterministic)
3. Verify ground truth labels are correct

---

### Issue: High Error Rate (>10% empty/failed)

**Diagnosis:**
```bash
grep "Error" logs/generation_*.log | wc -l
```

**Solutions:**
1. Update transformers: `pip install --upgrade transformers`
2. Check model compatibility
3. Reduce max_tokens
4. Add error recovery to generate_batch.py

---

### Issue: Out of Memory

**Solutions:**
1. Close all other programs
2. Use smaller model (qwen-0.5b not qwen-1.5b)
3. Add to generate_batch.py:
   ```python
   import gc
   torch.cuda.empty_cache()  # If using GPU
   gc.collect()  # After each batch
   ```

---

### Issue: Inconsistent Results Between Runs

**This is normal** due to random sampling.

To verify consistency:
```bash
# Generate same samples with same seed twice
python generate_batch.py --seed 42 ... --output run1.json
python generate_batch.py --seed 42 ... --output run2.json

# Compare
python -c "import json; r1=json.load(open('run1.json')); r2=json.load(open('run2.json')); print('Same:', r1[0]['cot_raw'] == r2[0]['cot_raw'])"
```

---

## ✅ Phase 1 Completion Checklist

### Minimum Requirements (Publishable)
- [ ] 300+ total samples generated
- [ ] 2+ datasets (GSM8K + StrategyQA)
- [ ] 1+ model (Qwen-0.5B)
- [ ] Correct/Incorrect ratio: 40-70%
- [ ] Error rate: <10%
- [ ] All samples have required fields
- [ ] Quality check passed
- [ ] Data backed up

### Strong Dataset (Recommended)
- [ ] 500+ total samples
- [ ] 3 datasets (GSM8K + StrategyQA + ARC)
- [ ] 2 models (Qwen + Phi)
- [ ] Balanced 60/40 correct/incorrect
- [ ] Error rate: <5%
- [ ] Diverse CoT lengths (100-500 chars)
- [ ] Manual validation on 50 samples
- [ ] Comprehensive quality report

---

## 📈 Success Metrics

**By end of Phase 1, you should have:**

✅ **Quantitative:**
- n ≥ 300 CoT samples
- 2+ domains covered
- 40-70% accuracy distribution
- <10% error/empty rate

✅ **Qualitative:**
- CoTs show clear reasoning steps
- Both simple and complex problems included
- Diverse reasoning patterns
- Minimal gibberish/errors

✅ **Documentation:**
- Generation logs saved
- Quality reports generated
- Issues documented
- Backup created

---

## 🚀 Moving to Phase 2

Once Phase 1 complete:

**Immediate next steps:**
1. Review `data/cot_outputs/merged_all_datasets.json`
2. Read Phase 2 plan (tagging & validation)
3. Run: `python -m src.tagging.rule_based_tagger ...`

**Success criteria to proceed:**
- ✅ At least 200 usable samples
- ✅ Quality check passes
- ✅ Data backed up
- ✅ No critical errors remaining

**Congratulations! You'll have a research-grade dataset!** 🎉

---

**Estimated Total Time:**
- Minimum path (300 samples): 8-10 hours
- Target path (600 samples): 15-20 hours
- Ambitious path (900+ samples): 25-30 hours

**Spread over 5-7 days = ~2-4 hours/day**

This is your most important phase - take time to do it right!
