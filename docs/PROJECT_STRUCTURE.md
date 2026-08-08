# Project Structure Guide

CoT-HMM repository layout and module organization (IEEE submission ready).

```
Chain-Of-Taughts-HMM/
├── README.md                           # Main project documentation & quickstart
├── ACTION_PLAN.md                      # Milestone action plan & history
├── pyproject.toml                      # Python build configuration
├── requirements.txt                    # Project dependencies
│
├── src/                                # Core Python package
│   ├── __init__.py
│   ├── tagging/
│   │   ├── __init__.py
│   │   ├── rule_based_tagger.py        # Rule-based 8-tag semantic tagger (κ=0.953)
│   │   └── step_segmenter.py           # CoT trace sentence/step segmentation
│   │
│   ├── hmm/
│   │   ├── __init__.py
│   │   ├── trainer.py                  # CategoricalHMM training logic
│   │   ├── scorer.py                   # Log-likelihood & entropy scoring
│   │   └── ensemble.py                 # EnsembleHMM (M=12) & AnomalyHMM models
│   │
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── baselines.py                # Heuristic scorers (Length, Diversity, Structure, Repetition)
│   │   └── baseline_evaluation.py      # Baseline benchmark comparison engine
│   │
│   ├── experiments_v2/
│   │   ├── __init__.py
│   │   ├── roscoe_baseline.py          # ROSCOE neural embedding baseline
│   │   ├── picsar_baseline.py          # PiCSAR content-based confidence baseline
│   │   ├── bootstrap_ci.py             # 10,000 percentile bootstrap CIs
│   │   ├── effect_size.py              # Cohen's d & Cliff's delta effect sizes
│   │   └── data_schema.py              # ScoredTrace dataclass & JSON schema
│   │
│   └── analysis/
│       ├── __init__.py
│       ├── qualitative.py              # Case studies & cognitive fingerprinting
│       └── visualize.py                # Transition heatmaps & ROC curve plots
│
├── scripts/                            # Execution & benchmark scripts
│   ├── run_leakfree_evaluation.py      # Master 5-fold cross-validation & evaluation script
│   ├── run_all_phases.py               # Master end-to-end pipeline runner
│   ├── run_real_generation.py          # Local LLM reasoning trace generator
│   └── colab_scripts/                  # GPU-accelerated Colab/Kaggle experiment runners
│       ├── colab_experiments.py
│       └── setup_env.sh
│
├── data/
│   └── v2/                            # Verified benchmark datasets
│       ├── gsm8k_500_proper.json       # Clean 500-sample balanced dataset (250 correct, 250 incorrect)
│       └── gsm8k_500_proper_cv_scored.json # Out-of-fold cross-validated scored dataset
│
├── experiments/
│   └── results/                       # Empirical benchmark results
│       ├── evaluation_results_leak_free.json # Full benchmark metrics & CIs
│       ├── leak_free_report.md         # Markdown benchmark summary report
│       └── leak_free_report.txt        # Text benchmark summary report
│
├── paper/                             # IEEE conference manuscript
│   ├── main.tex                        # Updated LaTeX manuscript
│   └── figures/                        # Generated vector & PNG paper figures
│
├── tests/                             # Unit test suite (30 tests)
│   ├── test_tagger.py                  # Semantic tagger unit tests
│   ├── test_hmm_scorer.py              # HMM & Ensemble unit tests
│   ├── test_baselines.py               # Heuristic baseline unit tests
│   └── test_integration.py             # End-to-end integration tests
│
└── docs/                              # Project documentation
    ├── NEXT_STEPS.md                   # Current status & roadmap
    ├── RESEARCH_PLAN.md                # Research questions & empirical answers
    ├── METHODOLOGY.md                  # Complete mathematical & pipeline methodology
    ├── API.md                          # API reference guide
    └── architecture/                   # Architectural blueprints & diagrams
        ├── ARCHITECTURE_OVERVIEW.md
        ├── METHODS_REFERENCE.md
        └── SUMMARY.md
```