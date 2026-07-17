# Scripts Directory

## Organization

Scripts are organized by purpose for clarity.

```
scripts/
├── run_all_phases.py          # Master runner for 12-week pipeline
├── validate_setup.py          # Environment validation check
│
├── collection/                # Data generation scripts
│   └── collect_traces.py      # Collect CoT traces from LLMs
│
├── evaluation/                # Model evaluation
│   ├── run_baseline_eval.py   # Run baseline comparison
│   └── evaluate_cross_domain.py
│
├── analysis/                  # Analysis scripts
│   ├── arc_domain_analysis.py   # ARC below-chance investigation
│   ├── run_significance_tests.py # Statistical tests
│   ├── post_process_correctness.py # Verify answers
│   └── analyze_case_studies.py
│
└── utils/                     # Utilities
    ├── quick_start_test.py     # Quick validation
    └── test_model_load.py      # Model loading test
```

## Main Entry Points

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_all_phases.py` | Full pipeline orchestration | `python scripts/run_all_phases.py --phase 3` |
| `validate_setup.py` | Check environment | `python scripts/validate_setup.py` |
| `run_baseline_eval.py` | Evaluate models | `python scripts/evaluation/run_baseline_eval.py` |

## GPU Configuration

For GPU acceleration, set environment variables:

```bash
# For CUDA GPU
export CUDA_VISIBLE_DEVICES=0
python scripts/run_all_phases.py --gpu --phase 4

# For Apple Silicon (M1/M2)
export DEVICE=mps
python scripts/run_all_phases.py --phase 4
```

## Phase Scripts

Each phase has dedicated runners in `scripts/evaluation/`:

- Phase 1-2: Data collection → `scripts/collection/`
- Phase 3-4: Baseline implementation → `scripts/evaluation/`
- Phase 5-6: Analysis → `scripts/analysis/`
- Phase 7-8: Neural-HMM → `src/neural_hmm/`

## Test Scripts

Run tests before experiments:

```bash
# Unit tests
pytest tests/

# Quick smoke tests
python scripts/utils/quick_start_test.py

# Model loading test
python scripts/utils/test_model_load.py
```