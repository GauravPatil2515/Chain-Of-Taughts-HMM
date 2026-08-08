# CoT-HMM Colab/Kaggle Pipeline

This directory contains optimized scripts for running CoT-HMM experiments on Colab and Kaggle.

## Quick Start

```python
# Run in Colab/Kaggle notebook
!pip install -r colab_scripts/requirements.txt
!python colab_scripts/run_pipeline.py --dataset data/v2/gsm8k_500_traces.json --all
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `config.py` | Environment detection and configuration |
| `data_generation.py` | Scale datasets to N=500+ samples |
| `bootstrap_ci.py` | Bootstrap confidence intervals |
| `cross_architecture_transfer.py` | Transfer tests across model families |
| `compute_routing.py` | Three-tier routing experiment |
| `anomaly_hmm.py` | Anomaly HMM + plausibility margin |
| `entropy_drift.py` | Entropy Drift Rate analysis |
| `run_pipeline.py` | Unified experiment runner |

## Key Experiments (from Research Strategy)

1. **Bootstrap CIs** (W1) - Rigorous statistical reporting
2. **Cross-architecture transfer** (W1) - Structural universality claim  
3. **Compute routing** (W3) - Rejection sampling + cost savings
4. **Anomaly HMM + Margin** (W4) - Structural hallucination reduction
5. **EDR overthinking detection** (W7) - Online monitoring signal
6. **ARC 10-shot adaptation** (W6) - Domain transfer

## Threshold Strategy (from paper)

```python
theta_high = mean_correct_score + std_score  # Pass through
theta_low = mean_incorrect_score - std_score  # Escalate/reject
```

## Expected Results

- AUC improvement: 2-5 points with margin
- Compute savings: 30-60% via routing
- Transfer: r > 0.15 for 2/3 architectures
- EDR correlation: p < 0.05 with correctness