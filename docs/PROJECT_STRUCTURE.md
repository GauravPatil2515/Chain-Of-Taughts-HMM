# Project Structure Guide

Recommended organization for CoT-HMM project (IEEE submission ready).

## IDEAL FOLDER STRUCTURE

```
chain-of-thought-hmm/
├── README.md                           # Main project guide
├── pyproject.toml                      # Python project config
├── requirements.txt                    # Dependencies
│
├── src/
│   ├── __init__.py
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── gsm8k_loader.py             # GSM8K dataset utilities
│   │   ├── generate_cot.py             # LLM CoT generation
│   │   └── generate_multidataset.py      # Multi-dataset support
│   │
│   ├── tagging/
│   │   ├── __init__.py
│   │   ├── rule_based_tagger.py        # Current rule-based tagger
│   │   └── step_segmenter.py           # Step segmentation logic
│   │
│   ├── hmm/
│   │   ├── __init__.py
│   │   ├── trainer.py                  # HMM training
│   │   ├── scorer.py                   # HMM scoring
│   │   └── ensemble.py                   # HMM ensemble utilities
│   │
│   ├── neural_hmm/
│   │   ├── __init__.py
│   │   └── neural_hmm_hybrid.py        # Neural tagger + hybrid scorer
│   │
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── baselines.py                # Heuristic scorers
│   │   └── baseline_evaluation.py        # Comparison framework
│   │
│   ├── experiments_v2/
│   │   ├── __init__.py
│   │   ├── roscoe_baseline.py          # ROSCOE implementation
│   │   ├── picsar_baseline.py          # PiCSAR implementation
│   │   ├── bootstrap_ci.py             # Bootstrap confidence intervals
│   │   ├── effect_size.py              # Effect size calculations
│   │   └── data_schema.py              # ScoredTrace dataclass
│   │
│   ├── rejection_sampling/
│   │   ├── __init__.py
│   │   └── rejection_sampling.py       # PPO/RL validation
│   │
│   └── analysis/
│       ├── __init__.py
│       ├── qualitative.py              # Case studies, examples
│       └── visualize.py                  # Plotting utilities
│
├── data/
│   ├── v2/                            # Current scaled dataset
│   │   ├── README.md                   # Dataset documentation
│   │   ├── gsm8k_500_traces.json       # Phase 1 output
│   │   ├── full_1000_traces.json       # Phase 2 output
│   │   └── scored/                     # Scored traces
│   │
│   ├── raw/                           # Raw dataset files
│   │   ├── gsm8k.json
│   │   └── arc_challenge.json
│   │
│   ├── cot_outputs/
│   │   └── real_*.json                # Generated CoT outputs
│   │
│   └── tagged/
│       └── phase*_tagged*.json          # Tagged traces
│
├── models/
│   ├── README.md                      # Model documentation
│   ├── hmm_model.pkl                  # Single HMM model
│   ├── ensemble.pkl                   # Ensemble of 12 HMMs
│   ├── neural_tagger_v1.pt            # Fine-tuned DistilBERT
│   └── .gitkeep
│
├── experiments/
│   └── results/
│       ├── v2/                       # Results by phase
│       │   ├── baseline_comparison.json
│       │   ├── bootstrap_cis.json
│       │   ├── effect_sizes.json
│       │   ├── roscoe_scores.json
│       │   ├── picsar_scores.json
│       │   ├── neural_hmm_scores.json
│       │   └── rejection_sampling.json
│       │
│       ├── baseline/                   # Demo results
│       │   ├── BASELINE_RESULTS.md
│       │   └── figures/
│       │
│       └── phase5_analysis/
│           ├── CASE_STUDIES_REPORT.md
│           └── SIGNIFICANCE_REPORT.md
│
├── paper/
│   ├── main.tex                       # Main LaTeX manuscript
│   ├── references.bib                 # Bibliography
│   ├── figures/
│   │   ├── cognitive_fingerprint.png
│   │   ├── transition_matrix.png
│   │   └── comparison_table.png
│   └── supplementary/                 # Code zip for review
│
├── scripts/
│   ├── run_all_phases.py              # Master runner script
│   │
│   ├── collection/
│   │   └── collect_traces.py          # Data collection pipeline
│   │
│   ├── evaluation/
│   │   └── run_baseline_eval.py       # Baseline evaluation
│   │
│   ├── analysis/
│   │   ├── arc_domain_analysis.py     # Domain investigation
│   │   └── run_statistical_tests.py   # Statistical tests
│   │
│   └── utils/
│       └── quick_start_test.py        # Validation script
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_tagger.py
│   ├── test_hmm_scorer.py
│   ├── test_baselines.py
│   └── test_integration.py
│
├── docs/
│   ├── architecture/
│   │   ├── ARCHITECTURE_OVERVIEW.md
│   │   ├── METHODS_REFERENCE.md
│   │   ├── VISUAL_ARCHITECTURE.html
│   │   └── ARCHITECTURE_DIAGRAMS.html
│   ├── API.md
│   └── METHODOLOGY.md
│
└── logs/
    └── generation_*.log
```

## FOLDER REORGANIZATION TASKS

- [ ] Move `src/hmm/ensemble.py` to `src/hmm/ensemble.py` (already there)
- [ ] Consolidate root-level scripts into `scripts/` subdirectories
- [ ] Rename `data/cot_outputs` → `data/v2/` for consistency
- [ ] Create `models/README.md` with model documentation
- [ ] Create `data/v2/README.md` with dataset schema
- [ ] Move `src/experiments_v2/` to `src/experiments/` or keep as is
- [ ] Clean up `.gitkeep` files or replace with placeholder data files