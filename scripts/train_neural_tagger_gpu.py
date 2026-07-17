#!/usr/bin/env python3
"""
train_neural_tagger_gpu.py - GPU-optimized NeuralTagger training script
Optimized for multi-GPU, mixed precision, and memory efficiency.

Usage:
    python scripts/train_neural_tagger_gpu.py --device cuda --batch-size 32 --epochs 5
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer, AutoModel
from src.neural_hmm.neural_hmm_hybrid import NeuralTagger, TAG_TO_ID, N_TAGS, build_training_data_from_dataset


class CoTDataset(Dataset):
    """PyTorch Dataset for CoT step -> tag pairs."""
    
    def __init__(self, steps: List[str], tags: List[str], tokenizer_name: str, max_length: int = 128):
        self.steps = steps
        self.tags = [TAG_TO_ID.get(t, 0) for t in tags]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    
    def __len__(self):
        return len(self.steps)
    
    def __getitem__(self, idx):
        text = self.steps[idx]
        tag_id = self.tags[idx]
        
        # Tokenize with padding/truncation
        enc = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(tag_id, dtype=torch.long)
        }


def train_neural_tagger_gpu(
    steps: List[str],
    tags: List[str],
    output_path: str,
    device: str = "cuda",
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 2e-5,
    use_amp: bool = True,
    accumulation_steps: int = 2
):
    """
    Train NeuralTagger with GPU optimizations.
    
    Args:
        steps: List of CoT step texts
        tags: List of tag labels (PR, DV, EQ, etc.)
        output_path: Where to save the model
        device: cuda or cpu
        batch_size: Training batch size
        epochs: Number of epochs
        lr: Learning rate
        use_amp: Use Automatic Mixed Precision
        accumulation_steps: Gradient accumulation steps
    """
    
    # Check CUDA availability
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        use_amp = False
    
    device_obj = torch.device(device)
    
    print(f"Training on {device_obj}")
    print(f"Dataset size: {len(steps)} samples")
    print(f"Batch size: {batch_size}")
    print(f"Effective batch (accumulated): {batch_size * accumulation_steps}")
    
    # Create model
    model = NeuralTagger(model_name="distilbert-base-uncased", device=device)
    
    # Create dataset
    dataset = CoTDataset(steps, tags, "distilbert-base-uncased")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp and device.startswith("cuda"))
    
    epoch_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device_obj)
            attention_mask = batch['attention_mask'].to(device_obj)
            labels = batch['labels'].to(device_obj)
            
            with autocast(enabled=use_amp and device.startswith("cuda")):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}, batch {i+1}/{len(loader)}, loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} complete - avg loss: {avg_loss:.4f}")
    
    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"\nModel saved to {output_path}")
    
    return epoch_losses


def main():
    parser = argparse.ArgumentParser(description="Train NeuralTagger with GPU")
    parser.add_argument("--data", type=str, default="data/v2/gsm8k_500_traces.json",
                        help="Dataset path")
    parser.add_argument("--output", type=str, default="models/neural_tagger_v1.pt",
                        help="Output model path")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable Automatic Mixed Precision")
    parser.add_argument("--accumulation-steps", type=int, default=2,
                        help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NeuralTagger GPU Training")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    steps, tags = build_training_data_from_dataset(args.data)
    print(f"Loaded {len(steps)} (step, tag) pairs")
    
    # Train
    losses = train_neural_tagger_gpu(
        steps=steps,
        tags=tags,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        use_amp=not args.no_amp,
        accumulation_steps=args.accumulation_steps
    )
    
    print(f"\nFinal losses: {[f'{l:.3f}' for l in losses]}")
    print("✓ Training complete")


if __name__ == "__main__":
    main()