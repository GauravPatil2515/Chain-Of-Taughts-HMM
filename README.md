# CoT-HMM: Structural Verification of Chain-of-Thought Reasoning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, interpretable, and unsupervised method for scoring the plausibility of Chain-of-Thought (CoT) reasoning from Large Language Models using Hidden Markov Models.

## 🎯 Overview

This project demonstrates that **correct reasoning follows characteristic structural patterns** that can be learned by simple probabilistic models. We abstract CoT text into semantic tags and use HMMs to score how "plausible" a reasoning chain appears based on its structure alone.

### Key Features

- **Unsupervised**: No human annotation required; learns from correctness labels only
- **Lightweight**: Trains in seconds on CPU (no GPU needed)
- **Interpretable**: Transition matrices reveal reasoning patterns
- **Model-Agnostic**: Works on any LLM's text output

### Results Summary

| Dataset | Model | Correlation | AUC | p-value |
|---------|-------|-------------|-----|---------|
| GSM8K | Qwen-0.5B | 0.11 | 0.56 | 0.015 |
| GSM8K | Qwen-1.5B | 0.30 | 0.70 | <0.01 |
| ARC (Transfer) | Qwen-0.5B | 0.04 | 0.46 | 0.62 |

## 📁 Project Structure

```
chain-of-thought-hmm/
├── src/
│   ├── hmm/                    # HMM training and scoring
│   │   ├── trainer.py          # Train CategoricalHMM on tag sequences
│   │   └── scorer.py           # Score new CoT outputs
│   ├── tagging/                # Semantic tagging
│   │   └── rule_based_tagger.py # Regex-based step classifier
│   ├── data_collection/        # Dataset loaders
│   │   └── gsm8k_loader.py     # GSM8K dataset utilities
│   ├── evaluation/             # Evaluation and baselines
│   │   ├── baselines.py        # Heuristic baseline scorers
│   │   ├── baseline_evaluation.py # Generate comparison reports
│   │   └── baseline_plots.py   # Visualization functions
│   └── analysis/               # Qualitative analysis tools
│       └── qualitative.py      # Transition heatmaps, examples
├── data/
│   ├── raw/                    # Raw dataset files
│   ├── cot_outputs/            # Generated CoT responses
│   └── tagged/                 # Tagged and processed data
├── models/                     # Trained HMM models (.pkl)
├── experiments/
│   └── results/                # Evaluation results and figures
├── paper/                      # LaTeX manuscript
│   ├── main.tex                # Main paper file
│   ├── figures/                # Paper figures
│   └── references.bib          # Bibliography
├── run_real_generation.py      # Generate CoT from LLMs
├── run_baseline_eval.py        # Run full evaluation pipeline
├── run_ablations.py            # Ablation studies
├── post_process_correctness.py # Verify answer correctness
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/GauravPatil2515/Chain-Of-Taughts-HMM.git
cd Chain-Of-Taughts-HMM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate CoT Samples

```bash
# Generate 100 CoT samples using Qwen-0.5B
python run_real_generation.py --model qwen-0.5b --limit 100
```

### 3. Tag Reasoning Steps

```bash
# Convert raw text to semantic tags
python -m src.tagging.rule_based_tagger \
    --input data/cot_outputs/your_file.json \
    --output data/tagged/your_file_tagged.json
```

### 4. Check Answer Correctness

```bash
# Verify against ground truth
python post_process_correctness.py \
    --input data/tagged/your_file_tagged.json \
    --output data/tagged/your_file_corrected.json
```

### 5. Train HMM & Evaluate

```bash
# Run full evaluation pipeline
python run_baseline_eval.py \
    --data data/tagged/your_file_corrected.json \
    --output experiments/results/my_experiment
```

## 📊 Semantic Tags

We define 8 semantic tags to classify reasoning steps:

| Tag | Name | Description |
|-----|------|-------------|
| PR | Problem Restatement | Restating given information |
| IN | Inference | Logical deduction or reasoning |
| CA | Calculation | Arithmetic operations |
| EQ | Equation | Mathematical formulas |
| CO | Conclusion | Final answer derivation |
| RF | Reflection | Self-correction or uncertainty |
| VE | Verification | Checking results |
| DV | Diversion | Off-topic or anomalous content |

## 🔧 Configuration

### Model Selection

Available models in `run_real_generation.py`:

- `qwen-0.5b`: Qwen2.5-0.5B-Instruct
- `qwen-1.5b`: Qwen2.5-1.5B-Instruct

### HMM Hyperparameters

In `src/hmm/trainer.py`:

- `n_components`: Number of hidden states (default: 5)
- `n_iter`: EM iterations (default: 100)
- `correct_only`: Train only on correct solutions (default: True)

## 📈 Reproducing Paper Results

```bash
# Phase 1: Scale up GSM8K (500 samples)
python run_real_generation.py --model qwen-0.5b --limit 500

# Phase 2: Multi-model (Qwen-1.5B)
python run_real_generation.py --model qwen-1.5b --limit 100

# Phase 3: Cross-domain (ARC)
python generate_multidataset.py --dataset arc --limit 120

# Phase 4: Ablations
python run_ablations.py

# Generate all figures
python run_baseline_eval.py --data data/tagged/phase5_corrected.json
```

## 📝 Citation

If you use this work, please cite:

```bibtex
@inproceedings{patil2025cothmm,
  title={Unsupervised Plausibility Scoring of Chain-of-Thought Reasoning via Structural Hidden Markov Models},
  author={Patil, Gaurav},
  booktitle={Proceedings of [Conference]},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📧 Contact

- **Author**: Gaurav Patil
- **Email**: <gauravppaiml123@gst.sies.edu.in>
- **Institution**: SIES Graduate School of Technology, Mumbai
