# CoT-HMM Dataset (v2)

## Overview

This directory contains the scaled dataset for the CoT-HMM research pipeline. All traces follow the `ScoredTrace` schema defined in `src/experiments_v2/data_schema.py`.

## File Structure

```
data/v2/
├── gsm8k_500_traces.json       # Phase 1: 500 GSM8K samples
├── gsm8k_llama_200.json        # Phase 2: Llama-3.2-1B samples
├── full_1000_traces.json       # Phase 2: Merged dataset
├── arc_300_traces.json         # ARC Challenge samples
├── scored/                     # Traces with computed scores
│   ├── gsm8k_rosocre.json      # ROSCOE scored
│   ├── gsm8k_picsar.json       # PiCSAR scored
│   └── gsm8k_neural_hmm.json    # Neural-HMM scored
└── README.md
```

## ScoredTrace Schema

Each trace is a JSON object with these fields:

```json
{
  "sample_id": "gsm8k_0001",
  "dataset": "gsm8k",
  "domain": "math",
  "model_family": "qwen",
  "model_name": "Qwen/Qwen-0.5B",
  "model_params_b": 0.5,
  
  "problem": "If John has 5 apples...",
  "ground_truth_answer": "3",
  
  "cot_text": "John starts with 5. He gives 2 away...",
  "model_answer": "3",
  "is_correct": true,
  "tag_sequence": ["PR", "DV", "CA", "CO"],
  "tagging_method": "rule_based",
  
  "hmm_score": { "log_likelihood": -3.21, ... },
  "roscoe_score": { "semantic_alignment": 0.45, ... },
  "picsar_score": { "joint_log_likelihood": -2.15, ... },
  "neural_hmm_score": { "neural_hmm_ll": -2.8, ... },
  
  "created_at": "2026-07-14T12:00:00"
}
```

## Model Families

| Family | Models | Parameters |
|--------|--------|------------|
| qwen | Qwen-0.5B, Qwen-1.5B | 0.5B, 1.5B |
| llama | Llama-3.2-1B | 1B |
| mistral | Mistral-7B | 7B |
| phi | Phi-3.5-Mini | 3.8B |

## Datasets

| Dataset | Domain | Samples | File |
|---------|--------|---------|------|
| GSM8K | Mathematics | 500-1319 | gsm8k_*.json |
| ARC-Challenge | Science/Logic | 300 | arc_*.json |
| StrategyQA | Commonsense | 100 | strategyqa_*.json |

## Loading Data

```python
from src.experiments_v2.data_schema import load_dataset, validate_dataset

traces = load_dataset("data/v2/gsm8k_500_traces.json")
report = validate_dataset(traces)
print(f"Accuracy: {report['accuracy']}")
```

## Scoring Pipeline

1. Load dataset → `list[ScoredTrace]`
2. Compute HMM scores → `hmm_score` field populated
3. Compute baselines → `roscoe_score`, `picsar_score` fields
4. Compute Neural-HMM → `neural_hmm_score` field
5. Validate → `validate_dataset()` for quality checks

## Notes

- All traces tagged with rule-based tagger (neural option available)
- HMM trained on correct traces only (correct_only=True)
- Dataset ready for IEEE submission (v2 format)