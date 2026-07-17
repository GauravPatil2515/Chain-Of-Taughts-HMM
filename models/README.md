# Trained Models Directory

## Overview

This directory contains trained HMM models and NeuralTagger checkpoints for the CoT-HMM project.

## Model Files

| File | Description | Training Data |
|------|-------------|-------------|
| `hmm_model.pkl` | Single Categorical HMM trained on correct traces | ~150 tagged GSM8K |
| `ensemble.pkl` | Ensemble of 12 HMMs (n_components: 3,5,8,12) | GSM8K correct traces |
| `neural_tagger_v1.pt` | Fine-tuned DistilBERT step classifier | Rule-tagged sequences |

## Model Loading

```python
import joblib
from src.hmm.trainer import TAG_TO_ID
from src.neural_hmm.neural_hmm_hybrid import NeuralTagger, NeuralHMMScorer

# Load HMM ensemble
models = joblib.load("models/ensemble.pkl")

# Load NeuralTagger
tagger = NeuralTagger.load("models/neural_tagger_v1.pt")
```

## Model Training

Train HMM on tagged data:
```bash
python -m src.hmm.trainer \
    --input data/tagged/phase4_tagged_corrected.json \
    --output models/phase4_hmm.pkl \
    --n-components 5
```

Train NeuralTagger:
```bash
python -m src.neural_hmm.neural_hmm_hybrid
# Then fine-tune with train_on_data()
```

## Model Format

### HMM Model (.pkl)
- Uses `hmmlearn.hmm.CategoricalHMM`
- Contains: transition_matrix (A), emissionprob_ (B), startprob_ (π)

### NeuralTagger (.pt)
- PyTorch state_dict
- Architecture: DistilBERT + linear classifier (8 classes)

## Notes
- Models trained on correct CoT traces only
- GPU recommended for NeuralTagger (CPU works but slower)
- See `docs/architecture/METHODS_REFERENCE.md` for hyperparameters