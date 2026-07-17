bash#!/bin/bash
# train_neural_tagger.sh - Train NeuralTagger with GPU acceleration
# Usage: ./scripts/train_neural_tagger.sh [GPU_ID]

set -e

GPU_ID=${1:-0}
DEVICE="cuda:${GPU_ID}"

echo "=========================================="
echo "NeuralTagger Training (GPU)"
echo "=========================================="
echo "Device: ${DEVICE}"
echo "Model: distilbert-base-uncased"
echo "Output: models/neural_tagger_v1.pt"
echo ""

# Set CUDA device
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Train on v2 dataset
python -c "
import torch
from pathlib import Path
from src.neural_hmm.neural_hmm_hybrid import NeuralTagger, build_training_data_from_dataset
import sys

print('Loading training data...')
steps, tags = build_training_data_from_dataset('data/v2/gsm8k_500_traces.json')
print(f'Training samples: {len(steps)}')

print('Initializing NeuralTagger...')
tagger = NeuralTagger(device='${DEVICE}')

print('Training (3 epochs)...')
losses = tagger.train_on_data(steps, tags, n_epochs=3, batch_size=16, lr=2e-5)

print('Saving model...')
Path('models').mkdir(exist_ok=True)
tagger.save('models/neural_tagger_v1.pt')

print('Done! Losses per epoch:', [f'{l:.3f}' for l in losses])
"

echo ""
echo "✓ Model saved to models/neural_tagger_v1.pt"