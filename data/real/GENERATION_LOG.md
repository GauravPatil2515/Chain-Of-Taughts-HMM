# GENERATION LOG: Real GSM8K CoT Traces

## Generation Session
- **Date**: 2026-08-11
- **Model**: Qwen2.5-0.5B-Instruct
- **Platform**: Google Colab (T4 GPU) / Kaggle (P100 GPU)
- **Prompt Template**: Step-by-step with boxed answer format
- **Temperature**: 0.7
- **Max New Tokens**: 512
- **Target Samples**: 300

## Dataset Statistics
- **Total Traces Generated**: 300
- **Correct Traces**: [TO BE FILLED]
- **Incorrect Traces**: [TO BE FILLED]
- **Accuracy**: [TO BE FILLED]%

## Verification Method
- Final answer extracted via regex patterns (\boxed{}, "answer is", "final answer", etc.)
- Compared against GSM8K gold answers (extracted from #### format)
- Numeric comparison with 0.01 tolerance

## File Outputs
- `data/real/gsm8k_qwen_300_real_traces.json` - Raw traces with correctness labels
- `experiments/results/real_data_eval/tagged.json` - Tagged traces
- `experiments/results/real_data_eval/cv_scored.json` - Cross-validated scores
- `experiments/results/CANONICAL_RESULTS.json` - Single source of truth

## Next Steps
1. Run tagging: `python -m src.tagging.rule_based_tagger --input data/real/gsm8k_qwen_300_real_traces.json --output experiments/results/real_data_eval/tagged.json`
2. Run leak-free CV: `python scripts/run_real_evaluation.py`
3. Verify all paper numbers match CANONICAL_RESULTS.json
4. Rewrite paper with real numbers

## Hardware & Environment
- GPU: NVIDIA T4 (Colab) / P100 (Kaggle)
- VRAM: ~16GB / ~16GB
- Transformers: 4.x.x
- PyTorch: 2.x.x with CUDA
- Generation Time: ~[TO BE FILLED] minutes